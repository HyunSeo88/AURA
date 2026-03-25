from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from datetime import timedelta
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from data_pipeline.dataset import AuraManifestDataset, aura_collate_fn
from data_pipeline.sampler import BucketAwareSpeakerSampler, SamplerRecord
from src.aura_pa import AuraPA
from src.utils.aura_pa_absent import AuraPAOnlineAbsentManager, build_absent_manager_from_source
from src.utils.aura_pa_losses import build_mrstft_loss, compute_aura_pa_temporal_loss
from src.utils.aura_pa_metrics import AverageMeter, collect_aura_pa_metric_records, select_detection_threshold, summarize_aura_pa_metric_records
from src.utils.aura_pa_temporal import AuraPATemporalTargets, build_chunk_activity_targets
from src.utils.logger import append_epoch_metrics, create_experiment_dirs, save_experiment_config, setup_experiment_logger


def _is_distributed_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed_available_and_initialized() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_distributed_available_and_initialized() else 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _barrier() -> None:
    if _is_distributed_available_and_initialized():
        dist.barrier()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f'{hours:d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def _maybe_log_loop_progress(
    *,
    logger,
    prefix: str,
    epoch: int,
    stage_name: str,
    current: int,
    total: int,
    start_time: float,
    interval: int,
    extra_message: str = '',
) -> None:
    if logger is None or total <= 0:
        return
    interval = max(1, int(interval))
    if current not in {1, total} and (current % interval) != 0:
        return

    elapsed = max(1e-6, time.monotonic() - start_time)
    avg_time_per_unit = elapsed / max(1, current)
    eta_seconds = avg_time_per_unit * max(0, total - current)
    percent = 100.0 * current / max(1, total)
    suffix = f' | {extra_message}' if extra_message else ''
    logger.info(
        'Epoch %d | Stage %s | %s %5.1f%% (%d/%d) | elapsed=%s | eta=%s%s',
        epoch,
        stage_name,
        prefix,
        percent,
        current,
        total,
        _format_duration(elapsed),
        _format_duration(eta_seconds),
        suffix,
    )



def _all_gather_objects(local_object: Any) -> list[Any]:
    if not _is_distributed_available_and_initialized():
        return [local_object]
    gathered = [None for _ in range(_get_world_size())]
    dist.all_gather_object(gathered, local_object)
    return gathered


def _merge_average_meter_states(states: list[Mapping[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for state in states:
        for key, value in dict(state.get('sum', {})).items():
            sums[key] = sums.get(key, 0.0) + float(value)
        for key, value in dict(state.get('count', {})).items():
            counts[key] = counts.get(key, 0) + int(value)
    return {
        key: sums[key] / counts[key]
        for key in sums.keys()
        if counts.get(key, 0) > 0
    }


def _init_distributed_if_needed(train_cfg: Mapping[str, Any]) -> tuple[bool, int, int, int]:
    dist_cfg = train_cfg.get('distributed', {})
    enabled = bool(dist_cfg.get('enabled', False))
    if not enabled:
        return False, 0, 0, 1

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        return False, 0, 0, 1

    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    backend = str(dist_cfg.get('backend', 'nccl'))
    timeout_minutes = float(dist_cfg.get('timeout_minutes', 180.0))
    dist.init_process_group(backend=backend, init_method='env://', timeout=timedelta(minutes=timeout_minutes))
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def _cleanup_distributed() -> None:
    if _is_distributed_available_and_initialized():
        dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='AuraPA training entrypoint')
    parser.add_argument('--config', type=str, required=True, help='Path to AuraPA experiment JSON config')
    parser.add_argument('--resume', type=str, default=None, help='Optional checkpoint path to resume from')
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'config not found: {path}')
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _resolve_device(device_cfg: str, local_rank: int) -> torch.device:
    if device_cfg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda', local_rank)
        return torch.device('cpu')
    if device_cfg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda', local_rank)
    return torch.device(device_cfg)


def _autocast_context(device: torch.device, precision: str):
    if device.type != 'cuda':
        return nullcontext()
    precision = str(precision).lower()
    if precision == 'bf16':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    if precision == 'fp16':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def _use_grad_scaler(device: torch.device, precision: str) -> bool:
    return device.type == 'cuda' and str(precision).lower() == 'fp16'


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _resolve_stage_config(base_config: Mapping[str, Any], stage_cfg: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base_config))
    for key in ('model', 'data', 'training', 'loss', 'evaluation'):
        if key in stage_cfg:
            merged[key] = _deep_update(dict(merged.get(key, {})), stage_cfg[key])
    merged['stage'] = {
        'name': str(stage_cfg.get('name', 'unnamed_stage')),
        'epochs': int(stage_cfg.get('epochs', 1)),
    }
    return merged


def _build_dataset(config: Mapping[str, Any], *, split: str) -> AuraManifestDataset:
    data_cfg = config['data']
    manifest_key = {'train': 'train_manifest', 'val': 'val_manifest', 'test': 'test_manifest'}[split]
    return AuraManifestDataset(
        manifest_path=data_cfg[manifest_key],
        speaker_registry_path=data_cfg['speaker_registry'],
        split=split,
        enroll_k=int(data_cfg.get('enroll_k', 8)),
        sample_rate=int(data_cfg.get('sample_rate', 16000)),
        duration_sec=data_cfg.get('duration_sec', 6.0),
        random_crop=(split == 'train'),
        deterministic_eval_enroll=bool(data_cfg.get('deterministic_eval_enroll', True)),
        strict_registry=bool(data_cfg.get('strict_registry', True)),
        registry_embedding_model=data_cfg.get('registry_embedding_model'),
        seed=int(data_cfg.get('seed', 20260318)),
    )


def _build_train_loader(config: Mapping[str, Any], dataset: AuraManifestDataset) -> DataLoader:
    data_cfg = config['data']
    train_cfg = config['training']
    records = [
        SamplerRecord(index=idx, speaker_id=row.target_speaker_id, task_bucket=row.task_bucket)
        for idx, row in enumerate(dataset.rows)
    ]
    sampler = BucketAwareSpeakerSampler(
        records,
        bucket_weights=dict(data_cfg.get('bucket_ratios', {})),
        epoch_size=data_cfg.get('epoch_size'),
        seed=int(data_cfg.get('seed', 20260318)),
        rank=_get_rank(),
        world_size=_get_world_size(),
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get('batch_size_per_gpu', 1)),
        sampler=sampler,
        num_workers=int(data_cfg.get('num_workers', 0)),
        pin_memory=bool(data_cfg.get('pin_memory', True)),
        collate_fn=aura_collate_fn,
        drop_last=bool(data_cfg.get('drop_last', False)),
    )


class _ShardedEvalSampler(Sampler[int]):
    def __init__(self, dataset_size: int, *, rank: int, world_size: int) -> None:
        self.indices = list(range(int(rank), int(dataset_size), int(world_size)))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _build_eval_loader(config: Mapping[str, Any], dataset: AuraManifestDataset) -> DataLoader:
    data_cfg = config['data']
    train_cfg = config['training']
    sampler = None
    if _get_world_size() > 1:
        sampler = _ShardedEvalSampler(len(dataset.rows), rank=_get_rank(), world_size=_get_world_size())
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get('eval_batch_size_per_gpu', train_cfg.get('batch_size_per_gpu', 1))),
        shuffle=False,
        sampler=sampler,
        num_workers=int(data_cfg.get('num_workers', 0)),
        pin_memory=bool(data_cfg.get('pin_memory', True)),
        collate_fn=aura_collate_fn,
        drop_last=False,
    )


def _split_param_groups(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    model_ref = model.module if isinstance(model, DDP) else model
    backbone_ids = set()
    for attr in ('encoder', 'decoder', 'masknet'):
        module = getattr(model_ref, attr, None)
        if isinstance(module, torch.nn.Module):
            for param in module.parameters():
                backbone_ids.add(id(param))

    backbone_params = []
    body_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in backbone_ids:
            backbone_params.append(param)
        else:
            body_params.append(param)
    return backbone_params, body_params


def _build_optimizer(model: torch.nn.Module, train_cfg: Mapping[str, Any]) -> optim.Optimizer:
    opt_cfg = train_cfg.get('optimizer', {})
    opt_type = str(opt_cfg.get('type', 'adamw')).lower()
    lr = float(opt_cfg.get('lr', 1e-4))
    weight_decay = float(opt_cfg.get('weight_decay', 1e-4))
    betas = tuple(opt_cfg.get('betas', [0.9, 0.98]))

    backbone_lr_scale = float(opt_cfg.get('param_groups', {}).get('backbone_lr_scale', 0.1))
    body_lr_scale = float(opt_cfg.get('param_groups', {}).get('body_lr_scale', 1.0))
    backbone_params, body_params = _split_param_groups(model)
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr * backbone_lr_scale})
    if body_params:
        param_groups.append({'params': body_params, 'lr': lr * body_lr_scale})

    if opt_type == 'adamw':
        return optim.AdamW(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)
    if opt_type == 'adam':
        return optim.Adam(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)
    raise ValueError(f'unsupported optimizer type: {opt_type}')


def _build_scheduler(optimizer: optim.Optimizer, scheduler_cfg: Mapping[str, Any], total_epochs: int):
    sched_type = str(scheduler_cfg.get('type', 'none')).lower()
    if sched_type in {'none', 'off', 'disabled'}:
        return None, None
    if sched_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(scheduler_cfg.get('t_max', total_epochs))),
            eta_min=float(scheduler_cfg.get('eta_min', 1e-6)),
        ), 'epoch'
    if sched_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(scheduler_cfg.get('step_size', 10))),
            gamma=float(scheduler_cfg.get('gamma', 0.5)),
        ), 'epoch'
    if sched_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(scheduler_cfg.get('factor', 0.5)),
            patience=max(1, int(scheduler_cfg.get('patience', 3))),
            min_lr=float(scheduler_cfg.get('min_lr', 1e-6)),
        ), 'plateau'
    raise ValueError(f'unsupported scheduler type: {sched_type}')


def _warmup_settings(scheduler_cfg: Mapping[str, Any]) -> tuple[bool, int, float]:
    warmup_cfg = scheduler_cfg.get('warmup', {})
    enabled = bool(warmup_cfg.get('enabled', False))
    epochs = int(warmup_cfg.get('epochs', 0))
    start_factor = float(warmup_cfg.get('start_factor', 0.1))
    return enabled and epochs > 0, epochs, start_factor


def _apply_warmup(optimizer: optim.Optimizer, base_lrs: list[float], epoch: int, warmup_epochs: int, start_factor: float) -> bool:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return False
    progress = epoch / max(1, warmup_epochs)
    factor = start_factor + (1.0 - start_factor) * progress
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group['lr'] = float(base_lr) * factor
    return True


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=(device.type == 'cuda'))
    return out


def _ensure_presence_metadata(batch: Mapping[str, Any]) -> dict[str, Any]:
    output = _clone_batch(batch)
    batch_size = len(output['speaker_id'])
    if 'enroll_speaker_id' not in output:
        output['enroll_speaker_id'] = list(output['speaker_id'])
    if 'presence_source_type' not in output:
        output['presence_source_type'] = ['present' for _ in range(batch_size)]
    return output


def _resolve_temporal_presence_config(stage_config: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = stage_config.get('model', {})
    data_cfg = stage_config.get('data', {})
    temporal_cfg = dict(stage_config.get('loss', {}).get('temporal_presence', {}))
    temporal_cfg.setdefault('sample_rate', int(data_cfg.get('sample_rate', 16000)))
    temporal_cfg['chunk_length_ms'] = int(model_cfg.get('presence_chunk_length_ms', temporal_cfg.get('chunk_length_ms', 100)))
    temporal_cfg['chunk_hop_ms'] = int(model_cfg.get('presence_chunk_hop_ms', temporal_cfg.get('chunk_hop_ms', 50)))
    temporal_cfg.setdefault('ref_percentile', 95.0)
    temporal_cfg.setdefault('positive_db', -20.0)
    temporal_cfg.setdefault('negative_db', -35.0)
    temporal_cfg.setdefault('temperature_db', 3.0)
    temporal_cfg.setdefault('ambiguous_weight', 0.25)
    temporal_cfg.setdefault('dilation_chunks', 1)
    temporal_cfg.setdefault('min_active_chunks', 2)
    temporal_cfg.setdefault('energy_floor_db', -80.0)
    temporal_cfg.setdefault('eps', 1e-8)
    return temporal_cfg


def _resolve_temporal_shift_config(stage_config: Mapping[str, Any]) -> dict[str, Any]:
    shift_cfg = dict(stage_config.get('training', {}).get('temporal_shift', {}))
    shift_cfg.setdefault('enabled', False)
    shift_cfg.setdefault('probability', 1.0)
    shift_cfg.setdefault('max_shift_ms', 1500.0)
    shift_cfg.setdefault('fill_value', 0.0)
    return shift_cfg


def _zero_padded_shift_1d(wave: torch.Tensor, shift: int, *, fill_value: float = 0.0) -> torch.Tensor:
    if wave.dim() != 1:
        raise ValueError(f'wave must have shape [T], got {tuple(wave.shape)}')
    out = torch.full_like(wave, float(fill_value))
    if shift == 0:
        out.copy_(wave)
        return out
    num_samples = int(wave.numel())
    if shift > 0:
        if shift < num_samples:
            out[shift:] = wave[: num_samples - shift]
        return out
    shift_abs = abs(int(shift))
    if shift_abs < num_samples:
        out[: num_samples - shift_abs] = wave[shift_abs:]
    return out


def _apply_synchronized_temporal_shift(
    batch: Mapping[str, Any],
    *,
    shift_cfg: Mapping[str, Any],
    sample_rate: int,
    epoch: int,
    step: int,
    rank: int,
    seed: int,
) -> dict[str, Any]:
    if not bool(shift_cfg.get('enabled', False)):
        return _clone_batch(batch)
    max_shift_samples = int(round(float(shift_cfg.get('max_shift_ms', 0.0)) * float(sample_rate) / 1000.0))
    if max_shift_samples <= 0:
        return _clone_batch(batch)

    output = _clone_batch(batch)
    mix = output['mix']
    target = output['target']
    if mix.dim() != 2 or target.dim() != 2 or mix.shape != target.shape:
        raise ValueError('temporal shift expects mix/target tensors with shape [B, T]')
    probability = float(shift_cfg.get('probability', 1.0))
    fill_value = float(shift_cfg.get('fill_value', 0.0))

    batch_size, num_samples = mix.shape
    max_shift_samples = min(max_shift_samples, max(0, int(num_samples) - 1))
    for idx in range(batch_size):
        rng = random.Random(seed + (epoch * 1_000_003) + (step * 9_973) + (rank * 31) + (idx * 101))
        if rng.random() > probability:
            continue
        shift = rng.randint(-max_shift_samples, max_shift_samples)
        output['mix'][idx] = _zero_padded_shift_1d(mix[idx], shift, fill_value=fill_value)
        output['target'][idx] = _zero_padded_shift_1d(target[idx], shift, fill_value=fill_value)
    return output


def _attach_temporal_targets(batch: Mapping[str, Any], temporal_cfg: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(batch)
    speaker_match = torch.tensor(
        [1.0 if str(spk) == str(enroll) else 0.0 for spk, enroll in zip(output['speaker_id'], output['enroll_speaker_id'])],
        device=output['target'].device,
        dtype=torch.float32,
    )
    temporal_targets: AuraPATemporalTargets = build_chunk_activity_targets(
        target_wave=output['target'],
        speaker_match_target=speaker_match,
        sample_rate=int(temporal_cfg.get('sample_rate', 16000)),
        chunk_length_ms=int(temporal_cfg.get('chunk_length_ms', 100)),
        chunk_hop_ms=int(temporal_cfg.get('chunk_hop_ms', 50)),
        ref_percentile=float(temporal_cfg.get('ref_percentile', 95.0)),
        positive_db=float(temporal_cfg.get('positive_db', -20.0)),
        negative_db=float(temporal_cfg.get('negative_db', -35.0)),
        temperature_db=float(temporal_cfg.get('temperature_db', 3.0)),
        ambiguous_weight=float(temporal_cfg.get('ambiguous_weight', 0.25)),
        dilation_chunks=int(temporal_cfg.get('dilation_chunks', 1)),
        min_active_chunks=int(temporal_cfg.get('min_active_chunks', 2)),
        energy_floor_db=float(temporal_cfg.get('energy_floor_db', -80.0)),
        eps=float(temporal_cfg.get('eps', 1e-8)),
    )

    output['speaker_match_target'] = temporal_targets.speaker_match_target
    output['clip_activity_target'] = temporal_targets.clip_activity_target
    output['target_presence'] = temporal_targets.clip_activity_target
    output['chunk_activity_target_soft'] = temporal_targets.chunk_activity_target_soft
    output['chunk_activity_weight'] = temporal_targets.chunk_activity_weight
    output['chunk_eval_target_strict'] = temporal_targets.chunk_eval_target_strict
    output['chunk_eval_mask_strict'] = temporal_targets.chunk_eval_mask_strict
    output['chunk_eval_target_tolerant'] = temporal_targets.chunk_eval_target_tolerant
    output['chunk_eval_mask_tolerant'] = temporal_targets.chunk_eval_mask_tolerant
    output['chunk_active_seed_mask'] = temporal_targets.chunk_active_seed_mask
    output['chunk_active_mask'] = temporal_targets.chunk_active_mask
    output['chunk_inactive_mask'] = temporal_targets.chunk_inactive_mask
    output['inactive_wave_weight'] = temporal_targets.inactive_wave_weight
    output['chunk_center_sec'] = temporal_targets.chunk_center_sec
    output['silent_present_mask'] = temporal_targets.silent_present_mask

    presence_source_type = list(output['presence_source_type'])
    for idx, is_silent in enumerate(temporal_targets.silent_present_mask.tolist()):
        if bool(is_silent) and float(temporal_targets.speaker_match_target[idx].item()) >= 0.5:
            presence_source_type[idx] = 'silent_present'
    output['presence_source_type'] = presence_source_type
    return output


def _prepare_batch_for_model(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    temporal_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    return _attach_temporal_targets(_move_batch_to_device(batch, device), temporal_cfg)


def _resolve_loss_runtime_config(loss_cfg: Mapping[str, Any]) -> dict[str, Any]:
    presence_cfg = dict(loss_cfg.get('presence', {}))
    chunk_cfg = dict(loss_cfg.get('chunk_activity', {}))
    clip_cfg = dict(loss_cfg.get('clip_activity', {}))
    inactive_cfg = dict(loss_cfg.get('inactive_clip_target_zero', loss_cfg.get('absent_target_zero', {})))
    return {
        'w_si': float(loss_cfg.get('w_si', 1.0)),
        'w_stft': float(loss_cfg.get('w_stft', 1.0)),
        'w_res_si': float(loss_cfg.get('w_res_si', 0.0)),
        'w_res_stft': float(loss_cfg.get('w_res_stft', 0.0)),
        'w_residual_target_leak': float(loss_cfg.get('w_residual_target_leak', 0.0)),
        'w_target_residual_leak': float(loss_cfg.get('w_target_residual_leak', 0.0)),
        'w_cons': float(loss_cfg.get('w_cons', 0.02)),
        'w_chunk': float(loss_cfg.get('w_chunk', loss_cfg.get('w_presence', 0.5))),
        'w_clip': float(loss_cfg.get('w_clip', 0.2)),
        'w_inactive': float(loss_cfg.get('w_inactive', 0.2)),
        'w_abs_zero': float(loss_cfg.get('w_abs_zero', 0.2)),
        'w_abs_res': float(loss_cfg.get('w_abs_res', 0.05)),
        'chunk_loss_type': str(chunk_cfg.get('type', loss_cfg.get('chunk_loss_type', 'soft_focal_bce'))),
        'chunk_focal_gamma': float(chunk_cfg.get('focal_gamma', loss_cfg.get('chunk_focal_gamma', presence_cfg.get('focal_gamma', 1.5)) or 1.5)),
        'chunk_pos_weight': chunk_cfg.get('pos_weight', loss_cfg.get('chunk_pos_weight', 2.0)),
        'clip_pos_weight': clip_cfg.get('pos_weight', loss_cfg.get('clip_pos_weight', presence_cfg.get('pos_weight'))),
        'clip_focal_gamma': clip_cfg.get('focal_gamma', loss_cfg.get('clip_focal_gamma')),
        'consistency_scope': str(loss_cfg.get('consistency_scope', 'active_only')),
        'inactive_target_zero_type': str(inactive_cfg.get('type', 'energy_ratio')),
    }


def _loss_metrics_to_float(losses: Mapping[str, torch.Tensor]) -> dict[str, float]:
    metrics = {}
    for key, value in losses.items():
        if torch.is_tensor(value) and value.ndim == 0:
            metrics[f'loss/{key}'] = float(value.item())
    return metrics


def _prefix_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f'{prefix}/{key}': float(value) for key, value in metrics.items()}


def _get_monitor_value(metrics: Mapping[str, float], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, float],
    threshold: float,
    config: Mapping[str, Any],
    best_monitors: Mapping[str, float],
) -> None:
    model_ref = model.module if isinstance(model, DDP) else model
    payload = {
        'epoch': int(epoch),
        'global_step': int(global_step),
        'model_state_dict': model_ref.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': dict(metrics),
        'detection_threshold': float(threshold),
        'config': dict(config),
        'best_monitors': dict(best_monitors),
    }
    torch.save(payload, path)


def _load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: optim.Optimizer | None = None, scheduler=None) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    model_ref = model.module if isinstance(model, DDP) else model
    model_ref.load_state_dict(state_dict, strict=True)
    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint


def _evaluate_model(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    temporal_cfg: Mapping[str, Any],
    mrstft_loss,
    absent_manager: AuraPAOnlineAbsentManager | None,
    logger=None,
    stage_name: str = 'default',
    epoch: int = 0,
    progress_log_interval: int = 100,
) -> tuple[dict[str, float], float]:
    model.eval()
    loss_meter = AverageMeter()
    records = []
    loss_runtime_cfg = _resolve_loss_runtime_config(loss_cfg)

    eval_absent_cfg = eval_cfg.get('absent_eval', {})
    eval_absent_enabled = bool(eval_absent_cfg.get('enabled', False)) and absent_manager is not None
    eval_absent_ratio = float(eval_absent_cfg.get('absent_ratio', 1.0))
    total_eval_units = len(loader) * (2 if eval_absent_enabled else 1)
    completed_eval_units = 0
    eval_start_time = time.monotonic()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            present_batch = _ensure_presence_metadata(batch)
            dev_batch = _prepare_batch_for_model(present_batch, device=device, temporal_cfg=temporal_cfg)
            est_target, est_residual, aux = model(
                dev_batch['mix'],
                dev_batch['enroll_seq'],
                enroll_mask=dev_batch['enroll_mask'],
            )
            _, loss_dict = compute_aura_pa_temporal_loss(
                est_target=est_target,
                est_residual=est_residual,
                aux=aux,
                target=dev_batch['target'],
                mix=dev_batch['mix'],
                speaker_match_target=dev_batch['speaker_match_target'],
                clip_activity_target=dev_batch['clip_activity_target'],
                chunk_activity_target_soft=dev_batch['chunk_activity_target_soft'],
                chunk_activity_weight=dev_batch['chunk_activity_weight'],
                inactive_wave_weight=dev_batch['inactive_wave_weight'],
                mrstft_loss=mrstft_loss,
                **loss_runtime_cfg,
            )
            loss_meter.update(_loss_metrics_to_float(loss_dict))
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
                    sample_rate=int(loader.dataset.sample_rate),
                    task_buckets=present_batch['task_bucket'],
                    scenarios=present_batch['scenario'],
                    sample_ids=present_batch['sample_id'],
                    use_stoi=bool(eval_cfg.get('raw_metrics', {}).get('stoi', False)),
                    use_pesq=bool(eval_cfg.get('raw_metrics', {}).get('pesq', False)),
                )
            )
            completed_eval_units += 1
            _maybe_log_loop_progress(
                logger=logger,
                prefix='Eval',
                epoch=epoch,
                stage_name=stage_name,
                current=completed_eval_units,
                total=total_eval_units,
                start_time=eval_start_time,
                interval=progress_log_interval,
                extra_message=f'phase=present batch={batch_idx}/{len(loader)}',
            )

            if eval_absent_enabled:
                absent_batch = absent_manager.make_eval_absent_batch(
                    present_batch,
                    absent_ratio=eval_absent_ratio,
                    source_type=str(eval_absent_cfg.get('source_type', 'deterministic_swap')),
                )
                dev_absent_batch = _prepare_batch_for_model(absent_batch, device=device, temporal_cfg=temporal_cfg)
                est_target_a, est_residual_a, aux_a = model(
                    dev_absent_batch['mix'],
                    dev_absent_batch['enroll_seq'],
                    enroll_mask=dev_absent_batch['enroll_mask'],
                )
                _, loss_dict_a = compute_aura_pa_temporal_loss(
                    est_target=est_target_a,
                    est_residual=est_residual_a,
                    aux=aux_a,
                    target=dev_absent_batch['target'],
                    mix=dev_absent_batch['mix'],
                    speaker_match_target=dev_absent_batch['speaker_match_target'],
                    clip_activity_target=dev_absent_batch['clip_activity_target'],
                    chunk_activity_target_soft=dev_absent_batch['chunk_activity_target_soft'],
                    chunk_activity_weight=dev_absent_batch['chunk_activity_weight'],
                    inactive_wave_weight=dev_absent_batch['inactive_wave_weight'],
                    mrstft_loss=mrstft_loss,
                    **loss_runtime_cfg,
                )
                loss_meter.update(_loss_metrics_to_float(loss_dict_a))
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
                        sample_rate=int(loader.dataset.sample_rate),
                        task_buckets=absent_batch['task_bucket'],
                        scenarios=absent_batch['scenario'],
                        sample_ids=absent_batch['sample_id'],
                        use_stoi=False,
                        use_pesq=False,
                    )
                )
                completed_eval_units += 1
                _maybe_log_loop_progress(
                    logger=logger,
                    prefix='Eval',
                    epoch=epoch,
                    stage_name=stage_name,
                    current=completed_eval_units,
                    total=total_eval_units,
                    start_time=eval_start_time,
                    interval=progress_log_interval,
                    extra_message=f'phase=absent batch={batch_idx}/{len(loader)}',
                )

    gathered_records = _all_gather_objects(records)
    gathered_loss_states = _all_gather_objects({
        'sum': dict(loss_meter._sum),
        'count': dict(loss_meter._count),
    })

    det_cfg = eval_cfg.get('detection', {})
    payload = None
    if _is_main_process():
        merged_records = [record for shard in gathered_records for record in shard]
        merged_loss_metrics = _merge_average_meter_states(gathered_loss_states)

        threshold_mode = str(det_cfg.get('threshold_mode', 'sweep')).lower()
        if threshold_mode == 'fixed':
            threshold = float(det_cfg.get('fixed_threshold', 0.5))
        else:
            metric_name = str(det_cfg.get('threshold_selection_metric', 'det/f0_5'))
            threshold_grid = [
                round(value, 6)
                for value in torch.arange(
                    float(det_cfg.get('threshold_min', 0.05)),
                    float(det_cfg.get('threshold_max', 0.95)) + 1e-8,
                    float(det_cfg.get('threshold_step', 0.01)),
                ).tolist()
            ]
            threshold, _ = select_detection_threshold(
                merged_records,
                metric_name=metric_name,
                beta=float(det_cfg.get('beta', 0.5)),
                thresholds=threshold_grid,
                gated_cfg=eval_cfg.get('gated', {}),
            )

        summary = summarize_aura_pa_metric_records(
            merged_records,
            detection_threshold=threshold,
            detection_beta=float(det_cfg.get('beta', 0.5)),
            gated_present_floor_db=float(eval_cfg.get('gated', {}).get('present_floor_db', -80.0)),
            gated_absent_suppression_db=float(eval_cfg.get('gated', {}).get('absent_suppression_db', 80.0)),
            gated_cfg=eval_cfg.get('gated', {}),
        )
        metrics = {}
        metrics.update(_prefix_metrics('val', merged_loss_metrics))
        metrics.update(_prefix_metrics('val', summary))
        metrics['val/det/selected_threshold'] = float(threshold)
        payload = {'metrics': metrics, 'threshold': float(threshold)}

    if _is_distributed_available_and_initialized():
        payload_list = [payload]
        dist.broadcast_object_list(payload_list, src=0)
        payload = payload_list[0]

    if payload is None:
        return {}, float(threshold)
    return dict(payload['metrics']), float(payload['threshold'])


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    precision: str,
    loss_cfg: Mapping[str, Any],
    temporal_cfg: Mapping[str, Any],
    temporal_shift_cfg: Mapping[str, Any],
    absent_cfg: Mapping[str, Any],
    absent_manager: AuraPAOnlineAbsentManager | None,
    seed: int,
    epoch: int,
    global_step: int,
    grad_accum_steps: int,
    clip_grad_norm: float | None,
    logger=None,
    stage_name: str = 'default',
    progress_log_interval: int = 50,
) -> tuple[dict[str, float], int]:
    model.train()
    dataset = loader.dataset
    if hasattr(dataset, 'set_epoch'):
        dataset.set_epoch(epoch)
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    if absent_manager is not None:
        absent_manager.set_epoch(epoch)

    meter = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    mrstft_loss = build_mrstft_loss(loss_cfg.get('mrstft', {})) if float(loss_cfg.get('w_stft', 1.0)) > 0.0 else None
    loss_runtime_cfg = _resolve_loss_runtime_config(loss_cfg)
    total_steps = len(loader)
    epoch_start_time = time.monotonic()

    for step, batch in enumerate(loader, start=1):
        work_batch = _ensure_presence_metadata(batch)
        work_batch = _apply_synchronized_temporal_shift(
            work_batch,
            shift_cfg=temporal_shift_cfg,
            sample_rate=int(temporal_cfg.get('sample_rate', 16000)),
            epoch=epoch,
            step=global_step + step,
            rank=_get_rank(),
            seed=int(seed),
        )

        absent_mode = str(absent_cfg.get('mode', 'off')).lower()
        if absent_mode == 'online_swap' and absent_manager is not None:
            work_batch = absent_manager.make_train_batch(
                work_batch,
                absent_ratio=float(absent_cfg.get('absent_ratio', 0.0)),
                step=global_step + step,
                rank=_get_rank(),
                ensure_min_present=bool(absent_cfg.get('ensure_min_present', False)),
                present_ratio_min=float(absent_cfg.get('present_ratio_min', 0.0)),
                absent_ratio_max=float(absent_cfg.get('absent_ratio_max', 1.0)),
                source_type=str(absent_cfg.get('source_type', 'online_swap')),
            )

        dev_batch = _prepare_batch_for_model(work_batch, device=device, temporal_cfg=temporal_cfg)
        sync_context = nullcontext()
        if isinstance(model, DDP) and grad_accum_steps > 1 and (step % grad_accum_steps) != 0:
            sync_context = model.no_sync()

        with sync_context:
            with _autocast_context(device, precision):
                est_target, est_residual, aux = model(
                    dev_batch['mix'],
                    dev_batch['enroll_seq'],
                    enroll_mask=dev_batch['enroll_mask'],
                )
                total_loss, loss_dict = compute_aura_pa_temporal_loss(
                    est_target=est_target,
                    est_residual=est_residual,
                    aux=aux,
                    target=dev_batch['target'],
                    mix=dev_batch['mix'],
                    speaker_match_target=dev_batch['speaker_match_target'],
                    clip_activity_target=dev_batch['clip_activity_target'],
                    chunk_activity_target_soft=dev_batch['chunk_activity_target_soft'],
                    chunk_activity_weight=dev_batch['chunk_activity_weight'],
                    inactive_wave_weight=dev_batch['inactive_wave_weight'],
                    mrstft_loss=mrstft_loss,
                    **loss_runtime_cfg,
                )
                total_loss = total_loss / max(1, grad_accum_steps)

            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

        if step % max(1, grad_accum_steps) == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if clip_grad_norm is not None and clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_values = _loss_metrics_to_float(loss_dict)
        meter.update(loss_values)
        avg_values = meter.average()
        primary_lr = float(optimizer.param_groups[0]['lr']) if optimizer.param_groups else 0.0
        last_total = float(loss_values.get('loss/total', loss_values.get('total', 0.0)))
        avg_total = float(avg_values.get('loss/total', avg_values.get('total', 0.0)))
        _maybe_log_loop_progress(
            logger=logger,
            prefix='Train',
            epoch=epoch,
            stage_name=stage_name,
            current=step,
            total=total_steps,
            start_time=epoch_start_time,
            interval=progress_log_interval,
            extra_message=f'loss={last_total:.4f} avg_loss={avg_total:.4f} lr={primary_lr:.2e}',
        )

    return _prefix_metrics('train', meter.average()), global_step + len(loader)


def main() -> None:
    args = _parse_args()
    base_config = _load_json(args.config)
    distributed, rank, local_rank, world_size = _init_distributed_if_needed(base_config.get('training', {}))
    device = _resolve_device(str(base_config.get('training', {}).get('device', 'auto')), local_rank)

    dirs = create_experiment_dirs(exp_name=base_config.get('experiment_name')) if _is_main_process() else None
    logger = setup_experiment_logger(dirs['logs_dir'] / 'train.log') if _is_main_process() else None
    if _is_main_process() and dirs is not None:
        save_experiment_config(dirs['config_dir'] / 'resolved_config.json', base_config)

    train_dataset = _build_dataset(base_config, split='train')
    train_loader = _build_train_loader(base_config, train_dataset)
    val_loader = _build_eval_loader(base_config, _build_dataset(base_config, split='val'))

    model = AuraPA(**base_config['model']).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None, output_device=local_rank if device.type == 'cuda' else None)

    precision = str(base_config.get('training', {}).get('precision', 'bf16')).lower()
    optimizer = _build_optimizer(model, base_config.get('training', {}))
    stages = list(base_config.get('training', {}).get('stages', []))
    total_epochs = sum(int(stage.get('epochs', 1)) for stage in stages) if stages else int(base_config.get('training', {}).get('epochs', 1))
    scheduler, scheduler_step_mode = _build_scheduler(optimizer, base_config.get('training', {}).get('scheduler', {}), total_epochs)
    warmup_enabled, warmup_epochs, warmup_start = _warmup_settings(base_config.get('training', {}).get('scheduler', {}))
    base_lrs = [float(group['lr']) for group in optimizer.param_groups]
    scaler = torch.cuda.amp.GradScaler(enabled=_use_grad_scaler(device, precision))

    start_epoch = 1
    global_step = 0
    current_threshold = float(base_config.get('evaluation', {}).get('detection', {}).get('fixed_threshold', 0.5))
    best_monitors: dict[str, float] = {}
    if args.resume is not None:
        checkpoint = _load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler)
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        global_step = int(checkpoint.get('global_step', 0))
        current_threshold = float(checkpoint.get('detection_threshold', current_threshold))
        best_monitors = {str(k): float(v) for k, v in checkpoint.get('best_monitors', {}).items()}

    checkpoint_cfg = base_config.get('training', {}).get('checkpointing', {})
    monitor_specs = list(checkpoint_cfg.get('monitors', []))
    keep_last_n = int(checkpoint_cfg.get('keep_last_n', 2))
    last_checkpoints: list[Path] = []

    stage_list = stages or [{'name': 'default', 'epochs': int(base_config.get('training', {}).get('epochs', 1))}]
    epoch_cursor = start_epoch
    for stage_cfg in stage_list:
        stage_config = _resolve_stage_config(base_config, stage_cfg)
        stage_name = stage_config['stage']['name']
        stage_epochs = int(stage_config['stage']['epochs'])
        if logger is not None:
            logger.info('Starting stage=%s epochs=%d', stage_name, stage_epochs)

        temporal_cfg = _resolve_temporal_presence_config(stage_config)
        temporal_shift_cfg = _resolve_temporal_shift_config(stage_config)

        stage_data_cfg = stage_config.get('data', {})
        external_absent_cfg = stage_data_cfg.get('external_absent', {})
        external_absent_registry = external_absent_cfg.get('speaker_registry')
        absent_cache_embeddings = bool(external_absent_cfg.get('cache_embeddings', True))

        train_absent_cfg = stage_config.get('training', {}).get('absent', {})
        train_absent_enabled = (
            str(train_absent_cfg.get('mode', 'off')).lower() == 'online_swap'
            and float(train_absent_cfg.get('absent_ratio', 0.0)) > 0.0
        )
        eval_absent_cfg = stage_config.get('evaluation', {}).get('absent_eval', {})
        eval_absent_enabled = bool(eval_absent_cfg.get('enabled', False)) and float(eval_absent_cfg.get('absent_ratio', 1.0)) > 0.0

        absent_manager_train = None
        absent_manager_val = None
        if train_absent_enabled:
            absent_manager_train = build_absent_manager_from_source(
                source=str(train_absent_cfg.get('source', 'internal')),
                split='train',
                internal_registry=train_dataset.registry,
                external_registry=external_absent_registry,
                enroll_k=int(stage_data_cfg.get('enroll_k', 8)),
                seed=int(stage_data_cfg.get('seed', 20260318)),
                cache_embeddings=absent_cache_embeddings,
            )
        if val_loader is not None and eval_absent_enabled:
            absent_manager_val = build_absent_manager_from_source(
                source=str(eval_absent_cfg.get('source', train_absent_cfg.get('source', 'internal'))),
                split='val',
                internal_registry=val_loader.dataset.registry,
                external_registry=external_absent_registry,
                enroll_k=int(stage_data_cfg.get('enroll_k', 8)),
                seed=int(stage_data_cfg.get('seed', 20260318)),
                cache_embeddings=absent_cache_embeddings,
            )

        for _ in range(stage_epochs):
            epoch = epoch_cursor
            if warmup_enabled:
                _apply_warmup(optimizer, base_lrs, epoch, warmup_epochs, warmup_start)

            train_metrics, global_step = _train_one_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler if _use_grad_scaler(device, precision) else None,
                precision=precision,
                loss_cfg=stage_config['loss'],
                temporal_cfg=temporal_cfg,
                temporal_shift_cfg=temporal_shift_cfg,
                absent_cfg=stage_config.get('training', {}).get('absent', {}),
                absent_manager=absent_manager_train,
                seed=int(stage_data_cfg.get('seed', 20260318)),
                epoch=epoch,
                global_step=global_step,
                grad_accum_steps=int(stage_config.get('training', {}).get('grad_accum_steps', 1)),
                clip_grad_norm=stage_config.get('training', {}).get('clip_grad_norm'),
                logger=logger if _is_main_process() else None,
                stage_name=stage_name,
                progress_log_interval=int(stage_config.get('training', {}).get('progress_log_interval_steps', 50)),
            )

            if scheduler is not None and scheduler_step_mode == 'epoch' and (not warmup_enabled or epoch > warmup_epochs):
                scheduler.step()

            val_metrics = {}
            if val_loader is not None:
                val_metrics, current_threshold = _evaluate_model(
                    model=model.module if isinstance(model, DDP) else model,
                    loader=val_loader,
                    device=device,
                    loss_cfg=stage_config['loss'],
                    eval_cfg=stage_config.get('evaluation', {}),
                    temporal_cfg=temporal_cfg,
                    mrstft_loss=build_mrstft_loss(stage_config['loss'].get('mrstft', {})) if float(stage_config['loss'].get('w_stft', 1.0)) > 0.0 else None,
                    absent_manager=absent_manager_val,
                    logger=logger if _is_main_process() else None,
                    stage_name=stage_name,
                    epoch=epoch,
                    progress_log_interval=int(stage_config.get('evaluation', {}).get('progress_log_interval_steps', 100)),
                )
                if scheduler is not None and scheduler_step_mode == 'plateau':
                    monitor_metric = float(val_metrics.get('val/loss/total', 0.0))
                    scheduler.step(monitor_metric)

            _barrier()

            if _is_main_process() and dirs is not None:
                epoch_row = {'epoch': epoch, 'stage': stage_name, **train_metrics, **val_metrics}
                append_epoch_metrics(
                    dirs['metrics_dir'] / 'epoch_metrics.jsonl',
                    dirs['metrics_dir'] / 'epoch_metrics.csv',
                    epoch_row,
                )
                if logger is not None:
                    logger.info('Epoch %d | %s', epoch, json.dumps(epoch_row, ensure_ascii=False))

                last_path = dirs['checkpoints_dir'] / f'last_epoch_{epoch:04d}.pth'
                _save_checkpoint(
                    last_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=epoch_row,
                    threshold=current_threshold,
                    config=stage_config,
                    best_monitors=best_monitors,
                )
                last_checkpoints.append(last_path)
                while len(last_checkpoints) > keep_last_n:
                    stale = last_checkpoints.pop(0)
                    if stale.exists():
                        stale.unlink()

                for monitor in monitor_specs:
                    name = str(monitor['name'])
                    metric_key = str(monitor['metric'])
                    mode = str(monitor.get('mode', 'max')).lower()
                    value = _get_monitor_value(epoch_row, metric_key)
                    if value is None:
                        continue
                    best_value = best_monitors.get(name)
                    improved = best_value is None or (value < best_value if mode == 'min' else value > best_value)
                    if improved:
                        best_monitors[name] = value
                        _save_checkpoint(
                            dirs['checkpoints_dir'] / f'{name}.pth',
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=global_step,
                            metrics=epoch_row,
                            threshold=current_threshold,
                            config=stage_config,
                            best_monitors=best_monitors,
                        )

            epoch_cursor += 1
            _barrier()

    _cleanup_distributed()


if __name__ == '__main__':
    main()
