from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from src.utils.aura_pa_losses import si_sdr_per_sample
from src.utils.aura_pa_temporal import build_chunk_grid, overlap_normalized_wave_weight

try:
    pesq_mod = importlib.import_module('pesq')
    pesq_fn = getattr(pesq_mod, 'pesq', None)
except Exception:
    pesq_fn = None

try:
    pystoi_mod = importlib.import_module('pystoi')
    stoi_fn = getattr(pystoi_mod, 'stoi', None)
except Exception:
    stoi_fn = None


@dataclass
class AuraPAMetricAccumulator:
    records: list[dict[str, Any]]

    def __init__(self) -> None:
        self.records = []

    def update(self, records: Iterable[dict[str, Any]]) -> None:
        self.records.extend(records)

    def summarize(
        self,
        *,
        detection_threshold: float = 0.5,
        detection_beta: float = 0.5,
        gated_present_floor_db: float = -80.0,
        gated_absent_suppression_db: float = 80.0,
        gated_cfg: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        return summarize_aura_pa_metric_records(
            self.records,
            detection_threshold=detection_threshold,
            detection_beta=detection_beta,
            gated_present_floor_db=gated_present_floor_db,
            gated_absent_suppression_db=gated_absent_suppression_db,
            gated_cfg=gated_cfg,
        )


class AverageMeter:
    def __init__(self) -> None:
        self._sum: dict[str, float] = {}
        self._count: dict[str, int] = {}

    def update(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            self._sum[key] = self._sum.get(key, 0.0) + numeric
            self._count[key] = self._count.get(key, 0) + 1

    def average(self) -> dict[str, float]:
        return {
            key: total / self._count[key]
            for key, total in self._sum.items()
            if self._count.get(key, 0) > 0
        }


def _ensure_batch_wave(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f'{name} must have shape [B, T], got {tuple(x.shape)}')
    return x


def _ensure_vector_target(value: torch.Tensor | None, batch_size: int, device: torch.device, name: str, default: float) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), float(default), device=device, dtype=torch.float32)
    if value.dim() == 2 and value.size(-1) == 1:
        value = value.squeeze(-1)
    if value.dim() != 1 or value.numel() != batch_size:
        raise ValueError(f'{name} must have shape [B], got {tuple(value.shape)} for batch_size={batch_size}')
    return value.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def _ensure_chunk_matrix(value: torch.Tensor | None, batch_size: int, *, name: str, dtype: torch.dtype, default: float | bool) -> torch.Tensor | None:
    if value is None:
        return None
    if value.dim() != 2 or value.size(0) != batch_size:
        raise ValueError(f'{name} must have shape [B, S], got {tuple(value.shape)} for batch_size={batch_size}')
    if dtype == torch.bool:
        return value.to(dtype=torch.bool)
    return value.to(dtype=dtype)


def _to_list(values: Sequence[Any] | None, batch_size: int, default: str) -> list[str]:
    if values is None:
        return [default for _ in range(batch_size)]
    out = list(values)
    if len(out) != batch_size:
        raise ValueError(f'metadata length mismatch: expected {batch_size}, got {len(out)}')
    return [str(v) for v in out]


def _safe_common_length(*tensors: torch.Tensor) -> list[torch.Tensor]:
    min_len = min(int(t.size(-1)) for t in tensors)
    return [t[..., :min_len] for t in tensors]


def _target_suppression_db_per_sample(est_target: torch.Tensor, mix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est_target, mix = _safe_common_length(est_target, mix)
    target_energy = est_target.pow(2).mean(dim=-1)
    mix_energy = mix.pow(2).mean(dim=-1)
    return -10.0 * torch.log10((target_energy + eps) / (mix_energy + eps))


def _projection_suppression_db_per_sample(pred: torch.Tensor, reference_wave: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred, reference_wave = _safe_common_length(pred, reference_wave)
    pred = pred - pred.mean(dim=-1, keepdim=True)
    reference_wave = reference_wave - reference_wave.mean(dim=-1, keepdim=True)
    reference_energy = torch.sum(reference_wave.pow(2), dim=-1, keepdim=True).clamp_min(eps)
    projected = (torch.sum(pred * reference_wave, dim=-1, keepdim=True) * reference_wave) / reference_energy
    projected_energy = torch.sum(projected.pow(2), dim=-1)
    pred_energy = torch.sum(pred.pow(2), dim=-1).clamp_min(eps)
    return -10.0 * torch.log10((projected_energy + eps) / pred_energy)


def _presence_probs_from_aux(aux: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    clip_logits = aux.get('presence_logit_clip')
    chunk_logits = aux.get('presence_logit_chunk')
    if clip_logits is None:
        raise KeyError("AuraPA metrics expect aux['presence_logit_clip']")
    if chunk_logits is None:
        raise KeyError("AuraPA metrics expect aux['presence_logit_chunk']")
    if clip_logits.dim() == 2 and clip_logits.size(-1) == 1:
        clip_logits = clip_logits.squeeze(-1)
    if clip_logits.dim() != 1:
        raise ValueError(f'presence_logit_clip must have shape [B], got {tuple(clip_logits.shape)}')
    if chunk_logits.dim() != 2 or chunk_logits.size(0) != clip_logits.size(0):
        raise ValueError(
            f'presence_logit_chunk must have shape [B, S] aligned with clip logits, got {tuple(chunk_logits.shape)}'
        )
    return torch.sigmoid(clip_logits), torch.sigmoid(chunk_logits)


def _optional_stoi(pred: np.ndarray, target: np.ndarray, sample_rate: int) -> float:
    if stoi_fn is None:
        return float('nan')
    try:
        return float(stoi_fn(target.astype(np.float64), pred.astype(np.float64), sample_rate, extended=False))
    except Exception:
        return float('nan')


def _optional_pesq(pred: np.ndarray, target: np.ndarray, sample_rate: int) -> float:
    if pesq_fn is None or sample_rate not in (8000, 16000):
        return float('nan')
    mode = 'wb' if sample_rate == 16000 else 'nb'
    try:
        return float(pesq_fn(sample_rate, target.astype(np.float64), pred.astype(np.float64), mode))
    except Exception:
        return float('nan')


def _record_slice_label(speaker_match: float, clip_activity: float) -> str:
    if clip_activity >= 0.5:
        return 'active_present'
    if speaker_match >= 0.5:
        return 'inactive_present'
    return 'donor_absent'


def build_gate_wave_weight_from_trace(
    gate_trace: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    num_samples: int,
    sample_rate: int,
    chunk_length_ms: int,
    chunk_hop_ms: int,
) -> torch.Tensor:
    gate_tensor = torch.as_tensor(gate_trace, dtype=torch.float32)
    if gate_tensor.dim() != 1:
        raise ValueError(f'gate_trace must have shape [S], got {tuple(gate_tensor.shape)}')
    grid = build_chunk_grid(
        int(num_samples),
        sample_rate=int(sample_rate),
        chunk_length_ms=int(chunk_length_ms),
        chunk_hop_ms=int(chunk_hop_ms),
        device=gate_tensor.device,
    )
    if gate_tensor.numel() != grid.num_chunks:
        raise ValueError(f'gate trace length mismatch: {gate_tensor.numel()} vs grid {grid.num_chunks}')
    wave_weight = overlap_normalized_wave_weight(gate_tensor.unsqueeze(0), grid=grid, num_samples=int(num_samples))
    return wave_weight.squeeze(0)


def causal_smooth_chunk_probs(
    probs: Sequence[float] | np.ndarray,
    *,
    mode: str = 'moving_average',
    window_hops: int = 2,
    ema_alpha: float = 0.6,
) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f'chunk probs must be 1D, got {values.shape}')
    mode_key = str(mode).lower()
    if mode_key in {'none', 'off', 'disabled'} or len(values) == 0:
        return values.copy()
    if mode_key == 'ema':
        alpha = float(ema_alpha)
        out = np.zeros_like(values)
        running = 0.0
        for idx, value in enumerate(values):
            if idx == 0:
                running = value
            else:
                running = (alpha * value) + ((1.0 - alpha) * running)
            out[idx] = running
        return out
    window = max(1, int(window_hops))
    out = np.zeros_like(values)
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        count = min(window, idx + 1)
        out[idx] = running_sum / max(1, count)
    return out


def build_causal_gate_trace(
    probs: Sequence[float] | np.ndarray,
    *,
    on_threshold: float,
    off_threshold: float,
    min_on_chunks: int = 2,
    min_off_chunks: int = 2,
) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f'chunk probs must be 1D, got {values.shape}')
    gate = np.zeros_like(values, dtype=np.float64)
    state = 0
    above_count = 0
    below_count = 0
    on_required = max(1, int(min_on_chunks))
    off_required = max(1, int(min_off_chunks))
    for idx, value in enumerate(values):
        if state == 0:
            above_count = above_count + 1 if value >= float(on_threshold) else 0
            below_count = 0
            if above_count >= on_required:
                state = 1
                gate[idx] = 1.0
            else:
                gate[idx] = 0.0
        else:
            below_count = below_count + 1 if value <= float(off_threshold) else 0
            above_count = 0
            if below_count >= off_required:
                state = 0
                gate[idx] = 0.0
            else:
                gate[idx] = 1.0
    return gate


def _compute_binary_confusion(probs: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, float]:
    decisions = probs >= float(threshold)
    positives = targets >= 0.5
    negatives = ~positives

    tp = float(np.sum(decisions & positives))
    fp = float(np.sum(decisions & negatives))
    tn = float(np.sum((~decisions) & negatives))
    fn = float(np.sum((~decisions) & positives))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'far': fp / max(1.0, float(np.sum(negatives))),
        'frr': fn / max(1.0, float(np.sum(positives))),
    }


def _fbeta(precision: float, recall: float, beta: float) -> float:
    beta_sq = float(beta) ** 2
    denom = (beta_sq * precision) + recall
    if denom <= 0.0:
        return 0.0
    return ((1.0 + beta_sq) * precision * recall) / denom


def _roc_auc(probs: np.ndarray, targets: np.ndarray) -> float:
    positives = targets >= 0.5
    negatives = ~positives
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1, dtype=np.float64)
    rank_sum_pos = float(np.sum(ranks[positives]))
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _average_precision(probs: np.ndarray, targets: np.ndarray) -> float:
    positives = targets >= 0.5
    n_pos = int(np.sum(positives))
    if n_pos == 0:
        return float('nan')
    order = np.argsort(-probs)
    sorted_targets = targets[order]
    tp = 0.0
    fp = 0.0
    precisions = []
    recalls = []
    for value in sorted_targets:
        if value >= 0.5:
            tp += 1.0
        else:
            fp += 1.0
        precisions.append(tp / max(1.0, tp + fp))
        recalls.append(tp / n_pos)
    ap = 0.0
    prev_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return float(ap)


def _compute_binary_metrics(prefix: str, probs: np.ndarray, targets: np.ndarray, *, threshold: float, beta: float) -> dict[str, float]:
    if probs.size == 0:
        return {
            f'{prefix}/threshold': float(threshold),
            f'{prefix}/precision': float('nan'),
            f'{prefix}/recall': float('nan'),
            f"{prefix}/f{str(beta).replace('.', '_')}": float('nan'),
            f'{prefix}/far': float('nan'),
            f'{prefix}/frr': float('nan'),
            f'{prefix}/auroc': float('nan'),
            f'{prefix}/auprc': float('nan'),
        }
    confusion = _compute_binary_confusion(probs, targets, threshold=threshold)
    return {
        f'{prefix}/threshold': float(threshold),
        f'{prefix}/precision': float(confusion['precision']),
        f'{prefix}/recall': float(confusion['recall']),
        f"{prefix}/f{str(beta).replace('.', '_')}": float(_fbeta(confusion['precision'], confusion['recall'], beta)),
        f'{prefix}/far': float(confusion['far']),
        f'{prefix}/frr': float(confusion['frr']),
        f'{prefix}/auroc': _roc_auc(probs, targets),
        f'{prefix}/auprc': _average_precision(probs, targets),
    }


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float('nan')
    return float(np.mean(finite))


def _slice_records(records: Sequence[dict[str, Any]], *, clip_activity: int | None = None, slice_label: str | None = None, task_bucket: str | None = None, scenario: str | None = None) -> list[dict[str, Any]]:
    out = []
    for record in records:
        if clip_activity is not None and int(record['clip_activity_target']) != int(clip_activity):
            continue
        if slice_label is not None and str(record['slice_label']) != str(slice_label):
            continue
        if task_bucket is not None and str(record['task_bucket']) != str(task_bucket):
            continue
        if scenario is not None and str(record['scenario']) != str(scenario):
            continue
        out.append(record)
    return out


def _resolve_gated_config(gated_cfg: Mapping[str, Any] | None, detection_threshold: float) -> dict[str, Any]:
    cfg = dict(gated_cfg or {})
    on_threshold = float(cfg.get('on_threshold', detection_threshold))
    off_threshold = cfg.get('off_threshold')
    if off_threshold is None:
        off_threshold = max(0.0, min(on_threshold, on_threshold - float(cfg.get('off_delta', 0.2))))
    return {
        'smoothing_mode': str(cfg.get('smoothing_mode', 'moving_average')),
        'smoothing_hops': int(cfg.get('smoothing_hops', 2)),
        'ema_alpha': float(cfg.get('ema_alpha', 0.6)),
        'on_threshold': on_threshold,
        'off_threshold': float(off_threshold),
        'min_on_chunks': int(cfg.get('min_on_chunks', 2)),
        'min_off_chunks': int(cfg.get('min_off_chunks', 2)),
    }


def _flatten_chunk_detection(records: Sequence[dict[str, Any]], variant: str) -> tuple[np.ndarray, np.ndarray]:
    probs_all = []
    targets_all = []
    target_key = f'chunk_eval_target_{variant}'
    mask_key = f'chunk_eval_mask_{variant}'
    for record in records:
        probs = np.asarray(record.get('presence_prob_chunk', []), dtype=np.float64)
        targets = np.asarray(record.get(target_key, []), dtype=np.float64)
        mask = np.asarray(record.get(mask_key, []), dtype=np.bool_)
        if probs.size == 0 or targets.size == 0 or mask.size == 0:
            continue
        valid = mask.astype(bool)
        if probs.size != targets.size or probs.size != mask.size:
            continue
        probs_all.append(probs[valid])
        targets_all.append(targets[valid])
    if not probs_all:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return np.concatenate(probs_all, axis=0), np.concatenate(targets_all, axis=0)


def _collect_gated_records(records: Sequence[dict[str, Any]], *, detection_threshold: float, gated_cfg: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    cfg = _resolve_gated_config(gated_cfg, detection_threshold)
    per_record = []
    gated_decisions = []
    gated_targets = []
    for record in records:
        probs = np.asarray(record.get('presence_prob_chunk', []), dtype=np.float64)
        tolerant_target = np.asarray(record.get('chunk_eval_target_tolerant', []), dtype=np.float64)
        tolerant_mask = np.asarray(record.get('chunk_eval_mask_tolerant', []), dtype=np.bool_)
        if probs.ndim != 1:
            continue
        smoothed = causal_smooth_chunk_probs(
            probs,
            mode=cfg['smoothing_mode'],
            window_hops=cfg['smoothing_hops'],
            ema_alpha=cfg['ema_alpha'],
        )
        gate = build_causal_gate_trace(
            smoothed,
            on_threshold=cfg['on_threshold'],
            off_threshold=cfg['off_threshold'],
            min_on_chunks=cfg['min_on_chunks'],
            min_off_chunks=cfg['min_off_chunks'],
        )
        valid = tolerant_mask.astype(bool) if tolerant_mask.size == gate.size else np.ones_like(gate, dtype=bool)
        if tolerant_target.size == gate.size:
            gated_decisions.append(gate[valid])
            gated_targets.append(tolerant_target[valid])
        per_record.append({
            'sample_id': record['sample_id'],
            'slice_label': record['slice_label'],
            'gate_open_rate': float(np.mean(gate)) if gate.size > 0 else float('nan'),
            'gate_switch_rate': float(np.mean(gate[1:] != gate[:-1])) if gate.size > 1 else 0.0,
            'gate_any_open': bool(np.any(gate >= 0.5)),
        })
    if gated_decisions:
        return per_record, np.concatenate(gated_decisions, axis=0), np.concatenate(gated_targets, axis=0)
    return per_record, np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)


def collect_aura_pa_metric_records(
    *,
    est_target: torch.Tensor,
    est_residual: torch.Tensor,
    aux: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    mix: torch.Tensor,
    target_presence: torch.Tensor | None = None,
    speaker_match_target: torch.Tensor | None = None,
    clip_activity_target: torch.Tensor | None = None,
    chunk_eval_target_strict: torch.Tensor | None = None,
    chunk_eval_mask_strict: torch.Tensor | None = None,
    chunk_eval_target_tolerant: torch.Tensor | None = None,
    chunk_eval_mask_tolerant: torch.Tensor | None = None,
    presence_source_types: Sequence[str] | None = None,
    sample_rate: int = 16000,
    task_buckets: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    use_stoi: bool = False,
    use_pesq: bool = False,
) -> list[dict[str, Any]]:
    est_target, est_residual, target, mix = _safe_common_length(est_target, est_residual, target, mix)
    _ensure_batch_wave(est_target, 'est_target')
    _ensure_batch_wave(est_residual, 'est_residual')
    _ensure_batch_wave(target, 'target')
    _ensure_batch_wave(mix, 'mix')

    batch_size = est_target.size(0)
    clip_activity = _ensure_vector_target(clip_activity_target if clip_activity_target is not None else target_presence, batch_size, est_target.device, 'clip_activity_target', 1.0)
    speaker_match = _ensure_vector_target(speaker_match_target, batch_size, est_target.device, 'speaker_match_target', 1.0)
    strict_target = _ensure_chunk_matrix(chunk_eval_target_strict, batch_size, name='chunk_eval_target_strict', dtype=torch.bool, default=False)
    strict_mask = _ensure_chunk_matrix(chunk_eval_mask_strict, batch_size, name='chunk_eval_mask_strict', dtype=torch.bool, default=False)
    tolerant_target = _ensure_chunk_matrix(chunk_eval_target_tolerant, batch_size, name='chunk_eval_target_tolerant', dtype=torch.bool, default=False)
    tolerant_mask = _ensure_chunk_matrix(chunk_eval_mask_tolerant, batch_size, name='chunk_eval_mask_tolerant', dtype=torch.bool, default=False)
    clip_probs, chunk_probs = _presence_probs_from_aux(aux)
    clip_probs = clip_probs.detach().to(torch.float32).cpu()
    chunk_probs = chunk_probs.detach().to(torch.float32).cpu()
    task_buckets = _to_list(task_buckets, batch_size, default='unknown')
    scenarios = _to_list(scenarios, batch_size, default='unknown')
    sample_ids = _to_list(sample_ids, batch_size, default='unknown')
    presence_source_types = _to_list(presence_source_types, batch_size, default='present')

    residual_ref = mix - target
    raw_si_sdr = si_sdr_per_sample(est_target, target).detach().to(torch.float32).cpu()
    mix_si_sdr = si_sdr_per_sample(mix, target).detach().to(torch.float32).cpu()
    raw_si_sdri = raw_si_sdr - mix_si_sdr
    raw_target_suppression = _target_suppression_db_per_sample(est_target, mix).detach().to(torch.float32).cpu()
    raw_residual_mix = si_sdr_per_sample(est_residual, mix).detach().to(torch.float32).cpu()
    raw_residual_ref_si_sdr = si_sdr_per_sample(est_residual, residual_ref).detach().to(torch.float32).cpu()
    raw_residual_target_suppression = _projection_suppression_db_per_sample(est_residual, target).detach().to(torch.float32).cpu()
    raw_target_residual_suppression = _projection_suppression_db_per_sample(est_target, residual_ref).detach().to(torch.float32).cpu()

    est_target_np = est_target.detach().to(torch.float32).cpu().numpy()
    target_np = target.detach().to(torch.float32).cpu().numpy()

    records: list[dict[str, Any]] = []
    for idx in range(batch_size):
        clip_value = float(clip_activity[idx].item())
        speaker_value = float(speaker_match[idx].item())
        slice_label = _record_slice_label(speaker_value, clip_value)
        record = {
            'sample_id': sample_ids[idx],
            'task_bucket': task_buckets[idx],
            'scenario': scenarios[idx],
            'presence_source_type': presence_source_types[idx],
            'speaker_match_target': 1 if speaker_value >= 0.5 else 0,
            'clip_activity_target': 1 if clip_value >= 0.5 else 0,
            'slice_label': slice_label,
            'presence_prob_clip': float(clip_probs[idx].item()),
            'presence_prob': float(clip_probs[idx].item()),
            'presence_prob_chunk': chunk_probs[idx].tolist(),
            'raw_active_si_sdr': float(raw_si_sdr[idx].item()) if clip_value >= 0.5 else float('nan'),
            'raw_active_si_sdri': float(raw_si_sdri[idx].item()) if clip_value >= 0.5 else float('nan'),
            'raw_active_residual_ref_si_sdr': float(raw_residual_ref_si_sdr[idx].item()) if clip_value >= 0.5 else float('nan'),
            'raw_active_residual_target_suppression_db': float(raw_residual_target_suppression[idx].item()) if clip_value >= 0.5 else float('nan'),
            'raw_active_target_residual_suppression_db': float(raw_target_residual_suppression[idx].item()) if clip_value >= 0.5 else float('nan'),
            'raw_inactive_target_suppression_db': float(raw_target_suppression[idx].item()) if clip_value < 0.5 else float('nan'),
            'raw_inactive_residual_mix_si_sdr': float(raw_residual_mix[idx].item()) if clip_value < 0.5 else float('nan'),
        }
        record['raw_present_si_sdr'] = record['raw_active_si_sdr']
        record['raw_present_si_sdri'] = record['raw_active_si_sdri']
        record['raw_present_residual_ref_si_sdr'] = record['raw_active_residual_ref_si_sdr']
        record['raw_present_residual_target_suppression_db'] = record['raw_active_residual_target_suppression_db']
        record['raw_present_target_residual_suppression_db'] = record['raw_active_target_residual_suppression_db']
        record['raw_absent_target_suppression_db'] = record['raw_inactive_target_suppression_db']
        record['raw_absent_residual_mix_si_sdr'] = record['raw_inactive_residual_mix_si_sdr']

        if strict_target is not None and strict_mask is not None:
            record['chunk_eval_target_strict'] = strict_target[idx].to(torch.int32).cpu().tolist()
            record['chunk_eval_mask_strict'] = strict_mask[idx].to(torch.int32).cpu().tolist()
        else:
            record['chunk_eval_target_strict'] = []
            record['chunk_eval_mask_strict'] = []
        if tolerant_target is not None and tolerant_mask is not None:
            record['chunk_eval_target_tolerant'] = tolerant_target[idx].to(torch.int32).cpu().tolist()
            record['chunk_eval_mask_tolerant'] = tolerant_mask[idx].to(torch.int32).cpu().tolist()
        else:
            record['chunk_eval_target_tolerant'] = []
            record['chunk_eval_mask_tolerant'] = []

        if use_stoi and clip_value >= 0.5:
            record['raw_active_stoi'] = _optional_stoi(est_target_np[idx], target_np[idx], sample_rate)
        else:
            record['raw_active_stoi'] = float('nan')
        record['raw_present_stoi'] = record['raw_active_stoi']
        if use_pesq and clip_value >= 0.5:
            record['raw_active_pesq'] = _optional_pesq(est_target_np[idx], target_np[idx], sample_rate)
        else:
            record['raw_active_pesq'] = float('nan')
        record['raw_present_pesq'] = record['raw_active_pesq']
        records.append(record)
    return records


def compute_detection_metrics(
    probs: Sequence[float],
    targets: Sequence[float],
    *,
    threshold: float = 0.5,
    beta: float = 0.5,
    prefix: str = 'det',
) -> dict[str, float]:
    probs_np = np.asarray(list(probs), dtype=np.float64)
    targets_np = np.asarray(list(targets), dtype=np.float64)
    return _compute_binary_metrics(prefix, probs_np, targets_np, threshold=threshold, beta=beta)


def build_threshold_grid(min_value: float = 0.05, max_value: float = 0.95, step: float = 0.01) -> list[float]:
    if step <= 0.0:
        raise ValueError('step must be > 0')
    values = []
    current = float(min_value)
    while current <= float(max_value) + 1e-8:
        values.append(round(current, 6))
        current += float(step)
    return values


def _metrics_for_threshold(
    records: Sequence[dict[str, Any]],
    *,
    metric_name: str,
    threshold: float,
    beta: float,
    gated_cfg: Mapping[str, Any] | None,
) -> dict[str, float]:
    metric_key = str(metric_name)
    if metric_key.startswith('chunk/strict/'):
        probs, targets = _flatten_chunk_detection(records, 'strict')
        return _compute_binary_metrics('chunk/strict', probs, targets, threshold=threshold, beta=beta)
    if metric_key.startswith('chunk/tolerant/'):
        probs, targets = _flatten_chunk_detection(records, 'tolerant')
        return _compute_binary_metrics('chunk/tolerant', probs, targets, threshold=threshold, beta=beta)
    if metric_key.startswith('gated/demo/'):
        _, gate_probs, gate_targets = _collect_gated_records(records, detection_threshold=threshold, gated_cfg=gated_cfg)
        return _compute_binary_metrics('gated/demo', gate_probs, gate_targets, threshold=0.5, beta=beta)
    clip_probs = np.asarray([float(r['presence_prob_clip']) for r in records], dtype=np.float64)
    clip_targets = np.asarray([float(r['clip_activity_target']) for r in records], dtype=np.float64)
    return _compute_binary_metrics('det', clip_probs, clip_targets, threshold=threshold, beta=beta)


def select_detection_threshold(
    records: Sequence[dict[str, Any]],
    *,
    metric_name: str = 'det/f0_5',
    beta: float = 0.5,
    thresholds: Sequence[float] | None = None,
    gated_cfg: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    if thresholds is None:
        thresholds = build_threshold_grid()
    best_threshold = float(thresholds[0]) if thresholds else 0.5
    best_metrics = _metrics_for_threshold(records, metric_name=metric_name, threshold=best_threshold, beta=beta, gated_cfg=gated_cfg)
    best_value = float(best_metrics.get(metric_name, float('-inf')))
    for threshold in thresholds[1:]:
        metrics = _metrics_for_threshold(records, metric_name=metric_name, threshold=float(threshold), beta=beta, gated_cfg=gated_cfg)
        value = float(metrics.get(metric_name, float('-inf')))
        if value > best_value:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_value = value
    return best_threshold, best_metrics


def summarize_aura_pa_metric_records(
    records: Sequence[dict[str, Any]],
    *,
    detection_threshold: float = 0.5,
    detection_beta: float = 0.5,
    gated_present_floor_db: float = -80.0,
    gated_absent_suppression_db: float = 80.0,
    gated_cfg: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    if not records:
        return {}

    metrics: dict[str, float] = {}
    beta_key = f"f{str(detection_beta).replace('.', '_')}"

    clip_probs = np.asarray([float(r['presence_prob_clip']) for r in records], dtype=np.float64)
    clip_targets = np.asarray([float(r['clip_activity_target']) for r in records], dtype=np.float64)
    clip_metrics = _compute_binary_metrics('det', clip_probs, clip_targets, threshold=detection_threshold, beta=detection_beta)
    metrics.update(clip_metrics)

    strict_probs, strict_targets = _flatten_chunk_detection(records, 'strict')
    tolerant_probs, tolerant_targets = _flatten_chunk_detection(records, 'tolerant')
    metrics.update(_compute_binary_metrics('chunk/strict', strict_probs, strict_targets, threshold=detection_threshold, beta=detection_beta))
    metrics.update(_compute_binary_metrics('chunk/tolerant', tolerant_probs, tolerant_targets, threshold=detection_threshold, beta=detection_beta))

    active_records = _slice_records(records, slice_label='active_present')
    inactive_present_records = _slice_records(records, slice_label='inactive_present')
    donor_absent_records = _slice_records(records, slice_label='donor_absent')
    inactive_records = [*inactive_present_records, *donor_absent_records]

    def add_mean(key: str, values: Iterable[float]) -> None:
        metrics[key] = _finite_mean(values)

    add_mean('raw/active_present/si_sdr', (r['raw_active_si_sdr'] for r in active_records))
    add_mean('raw/active_present/si_sdri', (r['raw_active_si_sdri'] for r in active_records))
    add_mean('raw/active_present/residual_ref_si_sdr', (r['raw_active_residual_ref_si_sdr'] for r in active_records))
    add_mean('raw/active_present/residual_target_suppression_db', (r['raw_active_residual_target_suppression_db'] for r in active_records))
    add_mean('raw/active_present/target_residual_suppression_db', (r['raw_active_target_residual_suppression_db'] for r in active_records))
    add_mean('raw/active_present/stoi', (r['raw_active_stoi'] for r in active_records))
    add_mean('raw/active_present/pesq', (r['raw_active_pesq'] for r in active_records))
    add_mean('raw/inactive_present/target_suppression_db', (r['raw_inactive_target_suppression_db'] for r in inactive_present_records))
    add_mean('raw/inactive_present/residual_mix_si_sdr', (r['raw_inactive_residual_mix_si_sdr'] for r in inactive_present_records))
    add_mean('raw/donor_absent/target_suppression_db', (r['raw_inactive_target_suppression_db'] for r in donor_absent_records))
    add_mean('raw/donor_absent/residual_mix_si_sdr', (r['raw_inactive_residual_mix_si_sdr'] for r in donor_absent_records))
    add_mean('raw/inactive/target_suppression_db', (r['raw_inactive_target_suppression_db'] for r in inactive_records))
    add_mean('raw/inactive/residual_mix_si_sdr', (r['raw_inactive_residual_mix_si_sdr'] for r in inactive_records))

    metrics['raw/present/si_sdr'] = metrics['raw/active_present/si_sdr']
    metrics['raw/present/si_sdri'] = metrics['raw/active_present/si_sdri']
    metrics['raw/present/residual_ref_si_sdr'] = metrics['raw/active_present/residual_ref_si_sdr']
    metrics['raw/present/residual_target_suppression_db'] = metrics['raw/active_present/residual_target_suppression_db']
    metrics['raw/present/target_residual_suppression_db'] = metrics['raw/active_present/target_residual_suppression_db']
    metrics['raw/present/stoi'] = metrics['raw/active_present/stoi']
    metrics['raw/present/pesq'] = metrics['raw/active_present/pesq']
    metrics['raw/absent/target_suppression_db'] = metrics['raw/inactive/target_suppression_db']
    metrics['raw/absent/residual_mix_si_sdr'] = metrics['raw/inactive/residual_mix_si_sdr']

    for bucket in ('1spk', '2spk', '3spk'):
        bucket_present = _slice_records(records, slice_label='active_present', task_bucket=bucket)
        add_mean(f'raw/present/{bucket}/si_sdr', (r['raw_active_si_sdr'] for r in bucket_present))
        add_mean(f'raw/present/{bucket}/si_sdri', (r['raw_active_si_sdri'] for r in bucket_present))

    hardcase_present = [r for r in active_records if str(r['task_bucket']) in {'2spk', '3spk'}]
    add_mean('raw/present/hardcase/si_sdr', (r['raw_active_si_sdr'] for r in hardcase_present))
    add_mean('raw/present/hardcase/si_sdri', (r['raw_active_si_sdri'] for r in hardcase_present))

    scenarios = sorted({str(r['scenario']) for r in records})
    for scenario in scenarios:
        scenario_present = _slice_records(records, slice_label='active_present', scenario=scenario)
        add_mean(f'raw/present/scenario/{scenario}/si_sdr', (r['raw_active_si_sdr'] for r in scenario_present))
        add_mean(f'raw/present/scenario/{scenario}/si_sdri', (r['raw_active_si_sdri'] for r in scenario_present))

    gate_records, gate_flat, gate_targets = _collect_gated_records(records, detection_threshold=detection_threshold, gated_cfg=gated_cfg)
    metrics.update(_compute_binary_metrics('gated/demo', gate_flat, gate_targets, threshold=0.5, beta=detection_beta))
    add_mean('gated/demo/open_rate', (r['gate_open_rate'] for r in gate_records))
    add_mean('gated/demo/switch_rate', (r['gate_switch_rate'] for r in gate_records))
    add_mean('gated/active_present/open_rate', (r['gate_open_rate'] for r in gate_records if r['slice_label'] == 'active_present'))
    add_mean('gated/inactive_present/open_rate', (r['gate_open_rate'] for r in gate_records if r['slice_label'] == 'inactive_present'))
    add_mean('gated/donor_absent/open_rate', (r['gate_open_rate'] for r in gate_records if r['slice_label'] == 'donor_absent'))

    gate_lookup = {str(r['sample_id']): r for r in gate_records}
    gated_present_values = []
    gated_inactive_values = []
    for record in active_records:
        gate_record = gate_lookup.get(str(record['sample_id']))
        gate_open = bool(gate_record['gate_any_open']) if gate_record is not None else False
        gated_present_values.append(float(record['raw_active_si_sdr']) if gate_open else float(gated_present_floor_db))
    for record in inactive_records:
        gate_record = gate_lookup.get(str(record['sample_id']))
        gate_open = bool(gate_record['gate_any_open']) if gate_record is not None else False
        gated_inactive_values.append(float(record['raw_inactive_target_suppression_db']) if gate_open else float(gated_absent_suppression_db))
    add_mean('gated/present/si_sdr', gated_present_values)
    add_mean('gated/inactive/target_suppression_db', gated_inactive_values)
    metrics['gated/absent/target_suppression_db'] = metrics['gated/inactive/target_suppression_db']

    return metrics
