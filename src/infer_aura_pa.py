from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torchaudio

from src.aura_pa import AuraPA
from src.train_aura_pa import (
    _cleanup_distributed,
    _build_dataset,
    _build_eval_loader,
    _ensure_presence_metadata,
    _init_distributed_if_needed,
    _is_main_process,
    _prepare_batch_for_model,
    _resolve_temporal_presence_config,
)
from src.utils.aura_pa_absent import build_absent_manager_from_source
from src.utils.aura_pa_metrics import (
    build_causal_gate_trace,
    build_gate_wave_weight_from_trace,
    causal_smooth_chunk_probs,
    collect_aura_pa_metric_records,
    select_detection_threshold,
    summarize_aura_pa_metric_records,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='AuraPA inference and evaluation entrypoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--config', type=str, default=None, help='Optional config path; defaults to resolved config near checkpoint')
    parser.add_argument('--mode', type=str, choices=['manifest_eval', 'single_sample'], default='manifest_eval')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--save-audio-limit', type=int, default=0, help='How many samples to save as wavs during manifest_eval')
    parser.add_argument('--threshold-mode', type=str, choices=['auto', 'fixed', 'from_checkpoint', 'sweep'], default='auto')
    parser.add_argument('--fixed-threshold', type=float, default=None)
    parser.add_argument('--mix-wav', type=str, default=None, help='single_sample mode only')
    parser.add_argument('--speaker-id', type=str, default=None, help='single_sample mode: speaker id to pull enrollment from registry')
    parser.add_argument('--speaker-registry', type=str, default=None, help='Optional registry path override')
    parser.add_argument('--enroll-emb-paths', type=str, default=None, help='Comma-separated embedding paths for single_sample mode')
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _resolve_config_path(checkpoint_path: Path, config_path: str | None) -> Path:
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f'config not found: {path}')
        return path
    candidate = checkpoint_path.parent.parent / 'config' / 'resolved_config.json'
    if candidate.exists():
        return candidate
    raise FileNotFoundError('could not infer resolved config path; pass --config explicitly')


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == 'auto':
        if torch.cuda.is_available():
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            return torch.device('cuda', local_rank)
        return torch.device('cpu')
    if device_cfg == 'cuda' and torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        return torch.device('cuda', local_rank)
    return torch.device(device_cfg)


def _is_distributed_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_world_size() -> int:
    return dist.get_world_size() if _is_distributed_available_and_initialized() else 1


def _all_gather_objects(local_object: Any) -> list[Any]:
    if not _is_distributed_available_and_initialized():
        return [local_object]
    gathered = [None for _ in range(_get_world_size())]
    dist.all_gather_object(gathered, local_object)
    return gathered


def _load_checkpoint(path: str | Path, model: torch.nn.Module) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def _safe_audio_load(path: str | Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.dim() != 2:
        raise ValueError(f'invalid audio shape at {path}: {tuple(wav.shape)}')
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0).float()


def _load_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    registry = {}
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            registry[str(payload['speaker_id'])] = payload
    return registry


def _load_embedding(path: str | Path) -> torch.Tensor:
    arr = torch.from_numpy(__import__('numpy').load(str(path)).astype('float32').reshape(-1))
    return arr


def _load_enrollment_from_registry(
    registry_path: str | Path,
    speaker_id: str,
    *,
    enroll_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    registry = _load_registry(registry_path)
    entry = registry.get(str(speaker_id))
    if entry is None:
        raise KeyError(f'speaker not found in registry: {speaker_id}')
    paths = [str(path) for path in entry.get('enroll_emb_paths', [])]
    if not paths:
        raise ValueError(f'speaker has no enrollment embeddings: {speaker_id}')
    fixed = [int(idx) for idx in entry.get('eval_fixed_indices', []) if 0 <= int(idx) < len(paths)]
    chosen = fixed[: min(enroll_k, len(fixed))]
    if len(chosen) < min(enroll_k, len(paths)):
        for idx in range(len(paths)):
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= min(enroll_k, len(paths)):
                break
    embeddings = [_load_embedding(paths[idx]) for idx in chosen]
    emb_dim = int(embeddings[0].numel())
    enroll_seq = torch.zeros(enroll_k, emb_dim, dtype=torch.float32)
    enroll_mask = torch.zeros(enroll_k, dtype=torch.bool)
    for pos, emb in enumerate(embeddings[:enroll_k]):
        enroll_seq[pos] = emb
        enroll_mask[pos] = True
    return enroll_seq, enroll_mask


def _load_enrollment_from_paths(paths_text: str, *, enroll_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    paths = [segment.strip() for segment in str(paths_text).split(',') if segment.strip()]
    if not paths:
        raise ValueError('--enroll-emb-paths did not provide any valid paths')
    embeddings = [_load_embedding(path) for path in paths[:enroll_k]]
    emb_dim = int(embeddings[0].numel())
    enroll_seq = torch.zeros(enroll_k, emb_dim, dtype=torch.float32)
    enroll_mask = torch.zeros(enroll_k, dtype=torch.bool)
    for pos, emb in enumerate(embeddings):
        enroll_seq[pos] = emb
        enroll_mask[pos] = True
    return enroll_seq, enroll_mask


def _resolve_gated_runtime(eval_cfg: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    gated_cfg = dict(eval_cfg.get('gated', {}))
    on_threshold = float(gated_cfg.get('on_threshold', threshold))
    off_threshold = gated_cfg.get('off_threshold')
    if off_threshold is None:
        off_threshold = max(0.0, min(on_threshold, on_threshold - float(gated_cfg.get('off_delta', 0.2))))
    return {
        'smoothing_mode': str(gated_cfg.get('smoothing_mode', 'moving_average')),
        'smoothing_hops': int(gated_cfg.get('smoothing_hops', 2)),
        'ema_alpha': float(gated_cfg.get('ema_alpha', 0.6)),
        'on_threshold': on_threshold,
        'off_threshold': float(off_threshold),
        'min_on_chunks': int(gated_cfg.get('min_on_chunks', 2)),
        'min_off_chunks': int(gated_cfg.get('min_off_chunks', 2)),
    }


def _compute_gate_outputs(
    chunk_probs: Sequence[float] | np.ndarray,
    *,
    eval_cfg: Mapping[str, Any],
    threshold: float,
    temporal_cfg: Mapping[str, Any],
    num_samples: int,
    sample_rate: int,
) -> dict[str, Any]:
    runtime = _resolve_gated_runtime(eval_cfg, threshold)
    chunk_probs = np.asarray(chunk_probs, dtype=np.float64)
    smoothed = causal_smooth_chunk_probs(
        chunk_probs,
        mode=runtime['smoothing_mode'],
        window_hops=runtime['smoothing_hops'],
        ema_alpha=runtime['ema_alpha'],
    )
    gate_trace = build_causal_gate_trace(
        smoothed,
        on_threshold=runtime['on_threshold'],
        off_threshold=runtime['off_threshold'],
        min_on_chunks=runtime['min_on_chunks'],
        min_off_chunks=runtime['min_off_chunks'],
    )
    gate_wave = build_gate_wave_weight_from_trace(
        gate_trace,
        num_samples=int(num_samples),
        sample_rate=int(sample_rate),
        chunk_length_ms=int(temporal_cfg.get('chunk_length_ms', 100)),
        chunk_hop_ms=int(temporal_cfg.get('chunk_hop_ms', 50)),
    )
    return {
        'chunk_probs': chunk_probs,
        'smoothed_chunk_probs': smoothed,
        'gate_trace': gate_trace,
        'gate_wave': gate_wave,
    }


def _select_threshold(
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    eval_cfg: Mapping[str, Any],
) -> tuple[float, dict[str, float] | None]:
    det_cfg = eval_cfg.get('detection', {})
    threshold_mode = args.threshold_mode
    if threshold_mode == 'auto':
        if len(records) <= 1:
            threshold_mode = 'from_checkpoint'
        else:
            threshold_mode = str(det_cfg.get('threshold_mode', 'sweep')).lower()
    if threshold_mode == 'fixed':
        if args.fixed_threshold is None:
            raise ValueError('--fixed-threshold is required when --threshold-mode fixed')
        return float(args.fixed_threshold), None
    if threshold_mode == 'from_checkpoint':
        return float(checkpoint.get('detection_threshold', det_cfg.get('fixed_threshold', 0.5))), None
    if threshold_mode == 'sweep':
        threshold, det_metrics = select_detection_threshold(
            records,
            metric_name=str(det_cfg.get('threshold_selection_metric', 'det/f0_5')),
            beta=float(det_cfg.get('beta', 0.5)),
            thresholds=[
                round(value, 6)
                for value in torch.arange(
                    float(det_cfg.get('threshold_min', 0.05)),
                    float(det_cfg.get('threshold_max', 0.95)) + 1e-8,
                    float(det_cfg.get('threshold_step', 0.01)),
                ).tolist()
            ],
            gated_cfg=eval_cfg.get('gated', {}),
        )
        return float(threshold), det_metrics
    raise ValueError(f'unsupported threshold mode: {threshold_mode}')


def _save_wave(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.unsqueeze(0).cpu(), sample_rate)


def run_manifest_eval(args: argparse.Namespace, config: Mapping[str, Any], checkpoint: Mapping[str, Any], model: AuraPA, device: torch.device) -> None:
    output_dir = Path(args.output_dir)
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _build_dataset(config, split=args.split)
    loader = _build_eval_loader(config, dataset)
    eval_cfg = config.get('evaluation', {})
    temporal_cfg = _resolve_temporal_presence_config(config)
    absent_manager = None
    absent_eval_cfg = eval_cfg.get('absent_eval', {})
    external_absent_cfg = config.get('data', {}).get('external_absent', {})
    if bool(absent_eval_cfg.get('enabled', False)) and float(absent_eval_cfg.get('absent_ratio', 1.0)) > 0.0:
        absent_manager = build_absent_manager_from_source(
            source=str(absent_eval_cfg.get('source', 'internal')),
            split=args.split,
            internal_registry=dataset.registry,
            external_registry=external_absent_cfg.get('speaker_registry'),
            enroll_k=int(config.get('data', {}).get('enroll_k', 8)),
            seed=int(config.get('data', {}).get('seed', 20260318)),
            cache_embeddings=bool(external_absent_cfg.get('cache_embeddings', True)),
        )

    records = []
    saved_candidates: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            present_batch = _ensure_presence_metadata(batch)
            dev_batch = _prepare_batch_for_model(present_batch, device=device, temporal_cfg=temporal_cfg)
            est_target, est_residual, aux = model(
                dev_batch['mix'],
                dev_batch['enroll_seq'],
                enroll_mask=dev_batch['enroll_mask'],
            )
            records.extend(
                collect_aura_pa_metric_records(
                    est_target=est_target,
                    est_residual=est_residual,
                    aux=aux,
                    target=dev_batch['target'],
                    mix=dev_batch['mix'],
                    target_presence=dev_batch['target_presence'],
                    speaker_match_target=dev_batch['speaker_match_target'],
                    clip_activity_target=dev_batch['clip_activity_target'],
                    chunk_eval_target_strict=dev_batch['chunk_eval_target_strict'],
                    chunk_eval_mask_strict=dev_batch['chunk_eval_mask_strict'],
                    chunk_eval_target_tolerant=dev_batch['chunk_eval_target_tolerant'],
                    chunk_eval_mask_tolerant=dev_batch['chunk_eval_mask_tolerant'],
                    presence_source_types=present_batch['presence_source_type'],
                    sample_rate=int(dataset.sample_rate),
                    task_buckets=present_batch['task_bucket'],
                    scenarios=present_batch['scenario'],
                    sample_ids=present_batch['sample_id'],
                    use_stoi=bool(eval_cfg.get('raw_metrics', {}).get('stoi', False)),
                    use_pesq=bool(eval_cfg.get('raw_metrics', {}).get('pesq', False)),
                )
            )
            if len(saved_candidates) < int(args.save_audio_limit):
                clip_probs = torch.sigmoid(aux['presence_logit_clip']).detach().cpu()
                chunk_probs = torch.sigmoid(aux['presence_logit_chunk']).detach().cpu()
                for idx, sample_id in enumerate(present_batch['sample_id']):
                    if len(saved_candidates) >= int(args.save_audio_limit):
                        break
                    saved_candidates.append({
                        'sample_id': sample_id,
                        'mix': dev_batch['mix'][idx].detach().cpu(),
                        'target': dev_batch['target'][idx].detach().cpu(),
                        'est_target': est_target[idx].detach().cpu(),
                        'est_residual': est_residual[idx].detach().cpu(),
                        'clip_prob': float(clip_probs[idx].item()),
                        'chunk_probs': chunk_probs[idx].numpy(),
                    })

            if absent_manager is not None:
                absent_batch = absent_manager.make_eval_absent_batch(
                    present_batch,
                    absent_ratio=float(absent_eval_cfg.get('absent_ratio', 1.0)),
                    source_type=str(absent_eval_cfg.get('source_type', 'deterministic_swap')),
                )
                dev_absent_batch = _prepare_batch_for_model(absent_batch, device=device, temporal_cfg=temporal_cfg)
                est_target_a, est_residual_a, aux_a = model(
                    dev_absent_batch['mix'],
                    dev_absent_batch['enroll_seq'],
                    enroll_mask=dev_absent_batch['enroll_mask'],
                )
                records.extend(
                    collect_aura_pa_metric_records(
                        est_target=est_target_a,
                        est_residual=est_residual_a,
                        aux=aux_a,
                        target=dev_absent_batch['target'],
                        mix=dev_absent_batch['mix'],
                        target_presence=dev_absent_batch['target_presence'],
                        speaker_match_target=dev_absent_batch['speaker_match_target'],
                        clip_activity_target=dev_absent_batch['clip_activity_target'],
                        chunk_eval_target_strict=dev_absent_batch['chunk_eval_target_strict'],
                        chunk_eval_mask_strict=dev_absent_batch['chunk_eval_mask_strict'],
                        chunk_eval_target_tolerant=dev_absent_batch['chunk_eval_target_tolerant'],
                        chunk_eval_mask_tolerant=dev_absent_batch['chunk_eval_mask_tolerant'],
                        presence_source_types=absent_batch['presence_source_type'],
                        sample_rate=int(dataset.sample_rate),
                        task_buckets=absent_batch['task_bucket'],
                        scenarios=absent_batch['scenario'],
                        sample_ids=absent_batch['sample_id'],
                        use_stoi=False,
                        use_pesq=False,
                    )
                )

    gathered_records = _all_gather_objects(records)
    gathered_candidates = _all_gather_objects(saved_candidates)
    if not _is_main_process():
        return

    merged_records = [record for shard in gathered_records for record in shard]
    merged_candidates = [candidate for shard in gathered_candidates for candidate in shard]
    save_limit = max(0, int(args.save_audio_limit))
    if save_limit > 0:
        merged_candidates = merged_candidates[:save_limit]
    else:
        merged_candidates = []

    threshold, det_override = _select_threshold(args=args, checkpoint=checkpoint, records=merged_records, eval_cfg=eval_cfg)
    summary = summarize_aura_pa_metric_records(
        merged_records,
        detection_threshold=threshold,
        detection_beta=float(eval_cfg.get('detection', {}).get('beta', 0.5)),
        gated_present_floor_db=float(eval_cfg.get('gated', {}).get('present_floor_db', -80.0)),
        gated_absent_suppression_db=float(eval_cfg.get('gated', {}).get('absent_suppression_db', 80.0)),
        gated_cfg=eval_cfg.get('gated', {}),
    )
    if det_override is not None:
        summary.update(det_override)

    for candidate in merged_candidates:
        gate_outputs = _compute_gate_outputs(
            candidate['chunk_probs'],
            eval_cfg=eval_cfg,
            threshold=threshold,
            temporal_cfg=temporal_cfg,
            num_samples=int(candidate['est_target'].numel()),
            sample_rate=int(dataset.sample_rate),
        )
        gated_target = candidate['est_target'] * gate_outputs['gate_wave']
        sample_dir = output_dir / 'samples' / str(candidate['sample_id'])
        _save_wave(sample_dir / 'est_target.wav', candidate['est_target'], int(dataset.sample_rate))
        _save_wave(sample_dir / 'est_residual.wav', candidate['est_residual'], int(dataset.sample_rate))
        _save_wave(sample_dir / 'gated_target.wav', gated_target, int(dataset.sample_rate))
        _save_wave(sample_dir / 'mix.wav', candidate['mix'], int(dataset.sample_rate))
        _save_wave(sample_dir / 'target.wav', candidate['target'], int(dataset.sample_rate))
        with (sample_dir / 'meta.json').open('w', encoding='utf-8') as handle:
            json.dump({
                'sample_id': candidate['sample_id'],
                'presence_prob_clip': candidate['clip_prob'],
                'threshold': threshold,
                'chunk_probs': gate_outputs['chunk_probs'].tolist(),
                'smoothed_chunk_probs': gate_outputs['smoothed_chunk_probs'].tolist(),
                'gate_trace': gate_outputs['gate_trace'].tolist(),
            }, handle, ensure_ascii=False, indent=2)

    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as handle:
        json.dump({
            'split': args.split,
            'threshold': threshold,
            'num_records': len(merged_records),
            'metrics': summary,
        }, handle, ensure_ascii=False, indent=2)
    with (output_dir / 'records.jsonl').open('w', encoding='utf-8') as handle:
        for record in merged_records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def run_single_sample(args: argparse.Namespace, config: Mapping[str, Any], checkpoint: Mapping[str, Any], model: AuraPA, device: torch.device) -> None:
    if args.mix_wav is None:
        raise ValueError('--mix-wav is required in single_sample mode')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = int(config.get('data', {}).get('sample_rate', 16000))
    enroll_k = int(config.get('data', {}).get('enroll_k', 8))
    temporal_cfg = _resolve_temporal_presence_config(config)
    mix = _safe_audio_load(args.mix_wav, sample_rate).unsqueeze(0).to(device)

    registry_path = args.speaker_registry or config.get('data', {}).get('speaker_registry')
    if args.enroll_emb_paths is not None:
        enroll_seq, enroll_mask = _load_enrollment_from_paths(args.enroll_emb_paths, enroll_k=enroll_k)
    else:
        if registry_path is None or args.speaker_id is None:
            raise ValueError('single_sample mode requires either --enroll-emb-paths or both --speaker-id and --speaker-registry/config data.speaker_registry')
        enroll_seq, enroll_mask = _load_enrollment_from_registry(registry_path, args.speaker_id, enroll_k=enroll_k)

    enroll_seq = enroll_seq.unsqueeze(0).to(device)
    enroll_mask = enroll_mask.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        est_target, est_residual, aux = model(mix, enroll_seq, enroll_mask=enroll_mask)
    clip_prob = float(torch.sigmoid(aux['presence_logit_clip'])[0].item())
    chunk_probs = torch.sigmoid(aux['presence_logit_chunk'])[0].detach().cpu().numpy()

    threshold, _ = _select_threshold(
        args=args,
        checkpoint=checkpoint,
        records=[{'presence_prob_clip': clip_prob, 'clip_activity_target': 1, 'presence_prob': clip_prob}],
        eval_cfg=config.get('evaluation', {}),
    )
    gate_outputs = _compute_gate_outputs(
        chunk_probs,
        eval_cfg=config.get('evaluation', {}),
        threshold=threshold,
        temporal_cfg=temporal_cfg,
        num_samples=int(est_target.size(-1)),
        sample_rate=sample_rate,
    )
    gated_target = est_target[0].cpu() * gate_outputs['gate_wave'].cpu()

    _save_wave(output_dir / 'est_target.wav', est_target[0].cpu(), sample_rate)
    _save_wave(output_dir / 'est_residual.wav', est_residual[0].cpu(), sample_rate)
    _save_wave(output_dir / 'gated_target.wav', gated_target, sample_rate)
    _save_wave(output_dir / 'mix.wav', mix[0].cpu(), sample_rate)
    with (output_dir / 'meta.json').open('w', encoding='utf-8') as handle:
        json.dump(
            {
                'mix_wav': str(args.mix_wav),
                'speaker_id': args.speaker_id,
                'presence_prob_clip': clip_prob,
                'threshold': threshold,
                'chunk_probs': gate_outputs['chunk_probs'].tolist(),
                'smoothed_chunk_probs': gate_outputs['smoothed_chunk_probs'].tolist(),
                'gate_trace': gate_outputs['gate_trace'].tolist(),
                'gate_open': bool(np.any(gate_outputs['gate_trace'] >= 0.5)),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint)
    config_path = _resolve_config_path(checkpoint_path, args.config)
    config = _load_json(config_path)
    distributed = False
    if args.mode == 'manifest_eval':
        distributed, _, _, _ = _init_distributed_if_needed(config.get('training', {}))
    try:
        device = _resolve_device(str(config.get('training', {}).get('device', 'auto')))

        model = AuraPA(**config['model']).to(device)
        checkpoint = _load_checkpoint(checkpoint_path, model)

        if args.mode == 'manifest_eval':
            run_manifest_eval(args, config, checkpoint, model, device)
        else:
            run_single_sample(args, config, checkpoint, model, device)
    finally:
        if distributed:
            _cleanup_distributed()


if __name__ == '__main__':
    main()
