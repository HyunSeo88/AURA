from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AuraPATemporalLossBreakdown:
    total: torch.Tensor
    si_sdr: torch.Tensor
    mrstft: torch.Tensor
    residual_si_sdr: torch.Tensor
    residual_mrstft: torch.Tensor
    residual_target_leakage: torch.Tensor
    target_residual_leakage: torch.Tensor
    consistency: torch.Tensor
    chunk_activity: torch.Tensor
    clip_activity: torch.Tensor
    present_inactive_suppression: torch.Tensor
    inactive_clip_target_zero: torch.Tensor
    inactive_clip_residual_mix: torch.Tensor
    num_speaker_match: torch.Tensor
    num_active_clips: torch.Tensor
    num_inactive_clips: torch.Tensor
    num_silent_present: torch.Tensor
    weighted_chunk_count: torch.Tensor
    weighted_chunk_positive: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            'total': self.total,
            'si_sdr': self.si_sdr,
            'mrstft': self.mrstft,
            'residual_si_sdr': self.residual_si_sdr,
            'residual_mrstft': self.residual_mrstft,
            'residual_target_leakage': self.residual_target_leakage,
            'target_residual_leakage': self.target_residual_leakage,
            'consistency': self.consistency,
            'chunk_activity': self.chunk_activity,
            'clip_activity': self.clip_activity,
            'present_inactive_suppression': self.present_inactive_suppression,
            'inactive_clip_target_zero': self.inactive_clip_target_zero,
            'inactive_clip_residual_mix': self.inactive_clip_residual_mix,
            'num_speaker_match': self.num_speaker_match,
            'num_active_clips': self.num_active_clips,
            'num_inactive_clips': self.num_inactive_clips,
            'num_silent_present': self.num_silent_present,
            'weighted_chunk_count': self.weighted_chunk_count,
            'weighted_chunk_positive': self.weighted_chunk_positive,
        }


def _validate_wave_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.dim() != 2 or target.dim() != 2:
        raise ValueError(
            f'expected waveform tensors [B, T], got {tuple(pred.shape)} and {tuple(target.shape)}'
        )
    if pred.shape != target.shape:
        raise ValueError(f'waveform shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}')


def _validate_vector_target(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
    default: float,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), float(default), device=device, dtype=torch.float32)
    if value.dim() == 2 and value.size(-1) == 1:
        value = value.squeeze(-1)
    if value.dim() != 1 or value.numel() != batch_size:
        raise ValueError(f'{name} must have shape [B], got {tuple(value.shape)} for batch_size={batch_size}')
    return value.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def _masked_weighted_mean(per_sample: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    if per_sample.dim() != 1:
        raise ValueError(f'per_sample must have shape [B], got {tuple(per_sample.shape)}')
    if sample_weight is None:
        return per_sample.mean()
    if sample_weight.shape != per_sample.shape:
        raise ValueError(
            f'sample_weight shape mismatch: expected {tuple(per_sample.shape)}, got {tuple(sample_weight.shape)}'
        )
    sample_weight = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype).clamp_min(0.0)
    denom = sample_weight.sum()
    if float(denom.item()) <= 0.0:
        return per_sample.new_zeros(())
    return torch.sum(per_sample * sample_weight) / denom.clamp_min(1.0)


def _safe_common_length(*tensors: torch.Tensor) -> list[torch.Tensor]:
    if not tensors:
        raise ValueError('at least one tensor is required')
    min_len = min(int(t.size(-1)) for t in tensors)
    return [t[..., :min_len] for t in tensors]


def si_sdr_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    _validate_wave_pair(pred, target)
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    target_energy = torch.sum(target.pow(2), dim=-1, keepdim=True).clamp_min(eps)
    projected = (torch.sum(pred * target, dim=-1, keepdim=True) * target) / target_energy
    noise = pred - projected
    ratio = torch.sum(projected.pow(2), dim=-1) / torch.sum(noise.pow(2), dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def negative_si_sdr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    per_sample = -si_sdr_per_sample(pred, target, eps=eps)
    return _masked_weighted_mean(per_sample, sample_weight=sample_weight)


def l1_wave_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_wave_pair(pred, target)
    per_sample = torch.mean(torch.abs(pred - target), dim=-1)
    return _masked_weighted_mean(per_sample, sample_weight=sample_weight)


def inactive_target_zero_energy_loss(
    est_target: torch.Tensor,
    mix: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    est_target, mix = _safe_common_length(est_target, mix)
    _validate_wave_pair(est_target, mix)
    target_energy = est_target.pow(2).mean(dim=-1)
    mix_energy = mix.pow(2).mean(dim=-1).clamp_min(eps)
    per_sample = torch.log1p(target_energy / mix_energy)
    return _masked_weighted_mean(per_sample, sample_weight=sample_weight)


def masked_wave_energy_ratio_loss(
    pred: torch.Tensor,
    reference_wave: torch.Tensor,
    *,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred, reference_wave, mask = _safe_common_length(pred, reference_wave, mask)
    _validate_wave_pair(pred, reference_wave)
    _validate_wave_pair(mask, reference_wave)
    mask = mask.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    mask_mass = mask.sum(dim=-1)
    pred_energy = (mask * pred.pow(2)).sum(dim=-1)
    ref_energy = (mask * reference_wave.pow(2)).sum(dim=-1).clamp_min(eps)
    per_sample = torch.log1p(pred_energy / ref_energy)
    valid = (mask_mass > 0.0).to(dtype=pred.dtype)
    if sample_weight is None:
        sample_weight = valid
    else:
        sample_weight = sample_weight.to(device=pred.device, dtype=pred.dtype) * valid
    return _masked_weighted_mean(per_sample, sample_weight=sample_weight)


def _projection_energy_ratio_per_sample(
    pred: torch.Tensor,
    reference_wave: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred, reference_wave = _safe_common_length(pred, reference_wave)
    _validate_wave_pair(pred, reference_wave)
    pred = pred - pred.mean(dim=-1, keepdim=True)
    reference_wave = reference_wave - reference_wave.mean(dim=-1, keepdim=True)
    reference_energy = torch.sum(reference_wave.pow(2), dim=-1, keepdim=True).clamp_min(eps)
    projected = (torch.sum(pred * reference_wave, dim=-1, keepdim=True) * reference_wave) / reference_energy
    projected_energy = torch.sum(projected.pow(2), dim=-1)
    pred_energy = torch.sum(pred.pow(2), dim=-1).clamp_min(eps)
    return projected_energy / pred_energy


def projection_energy_ratio_loss(
    pred: torch.Tensor,
    reference_wave: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    per_sample = torch.log1p(_projection_energy_ratio_per_sample(pred, reference_wave, eps=eps))
    return _masked_weighted_mean(per_sample, sample_weight=sample_weight)


def clip_activity_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    pos_weight: float | None = None,
    focal_gamma: float | None = None,
) -> torch.Tensor:
    if logits.dim() == 2 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    if logits.dim() != 1:
        raise ValueError(f'clip activity logits must have shape [B], got {tuple(logits.shape)}')
    if logits.shape != target.shape:
        raise ValueError(f'clip activity target shape mismatch: logits {tuple(logits.shape)} vs targets {tuple(target.shape)}')

    pos_weight_tensor = None
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)

    per_sample = F.binary_cross_entropy_with_logits(
        logits,
        target.to(dtype=logits.dtype),
        reduction='none',
        pos_weight=pos_weight_tensor,
    )
    if focal_gamma is not None and float(focal_gamma) > 0.0:
        probs = torch.sigmoid(logits)
        pt = (target * probs) + ((1.0 - target) * (1.0 - probs))
        per_sample = per_sample * torch.pow((1.0 - pt).clamp_min(0.0), float(focal_gamma))
    return _masked_weighted_mean(per_sample, sample_weight=sample_weight)


def masked_soft_focal_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor,
    focal_gamma: float = 1.5,
    pos_weight: float | None = None,
) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != weight.shape:
        raise ValueError(
            f'chunk activity shape mismatch: logits={tuple(logits.shape)} target={tuple(target.shape)} weight={tuple(weight.shape)}'
        )
    pos_weight_tensor = None
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    target = target.to(device=logits.device, dtype=logits.dtype)
    weight = weight.to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    per_entry = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction='none',
        pos_weight=pos_weight_tensor,
    )
    if float(focal_gamma) > 0.0:
        probs = torch.sigmoid(logits)
        pt = (target * probs) + ((1.0 - target) * (1.0 - probs))
        per_entry = per_entry * torch.pow((1.0 - pt).clamp_min(0.0), float(focal_gamma))
    denom = weight.sum()
    if float(denom.item()) <= 0.0:
        return logits.new_zeros(())
    return torch.sum(per_entry * weight) / denom.clamp_min(1.0)


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (256, 512, 1024),
        hop_sizes: Sequence[int] = (64, 128, 256),
        win_lengths: Sequence[int] = (256, 512, 1024),
        *,
        mag_weight: float = 1.0,
        logmag_weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError('fft_sizes, hop_sizes, and win_lengths must have the same length')
        self.fft_sizes = [int(v) for v in fft_sizes]
        self.hop_sizes = [int(v) for v in hop_sizes]
        self.win_lengths = [int(v) for v in win_lengths]
        self.mag_weight = float(mag_weight)
        self.logmag_weight = float(logmag_weight)
        self.eps = float(eps)

        for idx, win_length in enumerate(self.win_lengths):
            window = torch.hann_window(win_length)
            self.register_buffer(f'window_{idx}', window, persistent=False)

    def _single_resolution_loss(self, pred: torch.Tensor, target: torch.Tensor, idx: int) -> torch.Tensor:
        # cuFFT does not support bf16 inputs for torch.stft on our runtime, so run STFT in fp32
        # while keeping the surrounding training path in autocast/bf16.
        pred_stft = pred.to(dtype=torch.float32)
        target_stft = target.to(dtype=torch.float32)
        window = getattr(self, f'window_{idx}').to(device=pred.device, dtype=torch.float32)
        pred_spec = torch.stft(
            pred_stft,
            n_fft=self.fft_sizes[idx],
            hop_length=self.hop_sizes[idx],
            win_length=self.win_lengths[idx],
            window=window,
            center=True,
            return_complex=True,
        )
        target_spec = torch.stft(
            target_stft,
            n_fft=self.fft_sizes[idx],
            hop_length=self.hop_sizes[idx],
            win_length=self.win_lengths[idx],
            window=window,
            center=True,
            return_complex=True,
        )
        pred_mag = pred_spec.abs()
        target_mag = target_spec.abs()
        mag_loss = torch.mean(torch.abs(pred_mag - target_mag), dim=(-2, -1))
        logmag_loss = torch.mean(
            torch.abs(torch.log(pred_mag.clamp_min(self.eps)) - torch.log(target_mag.clamp_min(self.eps))),
            dim=(-2, -1),
        )
        return (self.mag_weight * mag_loss) + (self.logmag_weight * logmag_loss)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_wave_pair(pred, target)
        losses = [self._single_resolution_loss(pred, target, idx) for idx in range(len(self.fft_sizes))]
        per_sample = torch.stack(losses, dim=0).mean(dim=0)
        return _masked_weighted_mean(per_sample, sample_weight=sample_weight)


def build_mrstft_loss(config: Mapping[str, Any] | None = None) -> MultiResolutionSTFTLoss:
    config = dict(config or {})
    return MultiResolutionSTFTLoss(
        fft_sizes=config.get('fft_sizes', (256, 512, 1024)),
        hop_sizes=config.get('hop_sizes', (64, 128, 256)),
        win_lengths=config.get('win_lengths', (256, 512, 1024)),
        mag_weight=config.get('mag_weight', 1.0),
        logmag_weight=config.get('logmag_weight', 1.0),
        eps=config.get('eps', 1e-8),
    )


def compute_aura_pa_temporal_loss(
    *,
    est_target: torch.Tensor,
    est_residual: torch.Tensor,
    aux: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    mix: torch.Tensor,
    speaker_match_target: torch.Tensor | None,
    clip_activity_target: torch.Tensor | None,
    chunk_activity_target_soft: torch.Tensor,
    chunk_activity_weight: torch.Tensor,
    inactive_wave_weight: torch.Tensor,
    mrstft_loss: MultiResolutionSTFTLoss | None = None,
    w_si: float = 1.0,
    w_stft: float = 1.0,
    w_res_si: float = 0.0,
    w_res_stft: float = 0.0,
    w_residual_target_leak: float = 0.0,
    w_target_residual_leak: float = 0.0,
    w_cons: float = 0.02,
    w_chunk: float = 0.5,
    w_clip: float = 0.2,
    w_inactive: float = 0.2,
    w_abs_zero: float = 0.2,
    w_abs_res: float = 0.05,
    chunk_loss_type: str = 'soft_focal_bce',
    chunk_focal_gamma: float = 1.5,
    chunk_pos_weight: float | None = 2.0,
    clip_pos_weight: float | None = None,
    clip_focal_gamma: float | None = None,
    consistency_scope: str = 'active_only',
    inactive_target_zero_type: str = 'energy_ratio',
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    est_target, est_residual, target, mix, inactive_wave_weight = _safe_common_length(
        est_target, est_residual, target, mix, inactive_wave_weight
    )
    _validate_wave_pair(est_target, target)
    _validate_wave_pair(est_residual, target)
    _validate_wave_pair(mix, target)
    _validate_wave_pair(inactive_wave_weight, target)

    batch_size = est_target.size(0)
    device = est_target.device
    speaker_match = _validate_vector_target(
        speaker_match_target,
        batch_size=batch_size,
        device=device,
        name='speaker_match_target',
        default=1.0,
    )
    clip_activity = _validate_vector_target(
        clip_activity_target,
        batch_size=batch_size,
        device=device,
        name='clip_activity_target',
        default=1.0,
    )
    inactive_clip = 1.0 - clip_activity
    silent_present = speaker_match * inactive_clip

    if chunk_activity_target_soft.shape != chunk_activity_weight.shape:
        raise ValueError(
            'chunk activity target/weight shape mismatch: '
            f'{tuple(chunk_activity_target_soft.shape)} vs {tuple(chunk_activity_weight.shape)}'
        )
    chunk_logits = aux.get('presence_logit_chunk')
    if chunk_logits is None:
        raise KeyError("AuraPA temporal loss expects aux['presence_logit_chunk']")
    if chunk_logits.shape != chunk_activity_target_soft.shape:
        raise ValueError(
            f'chunk activity logits shape mismatch: logits={tuple(chunk_logits.shape)} target={tuple(chunk_activity_target_soft.shape)}'
        )
    clip_logits = aux.get('presence_logit_clip')
    if clip_logits is None:
        raise KeyError("AuraPA temporal loss expects aux['presence_logit_clip']")

    active_sample_weight = clip_activity
    residual_ref = mix - target

    if float(w_si) > 0.0:
        si_sdr_loss = negative_si_sdr_loss(est_target, target, sample_weight=active_sample_weight)
    else:
        si_sdr_loss = est_target.new_zeros(())

    if mrstft_loss is not None and float(w_stft) > 0.0:
        mrstft_value = mrstft_loss(est_target, target, sample_weight=active_sample_weight)
    else:
        mrstft_value = est_target.new_zeros(())

    if float(w_res_si) > 0.0:
        residual_si_sdr_loss = negative_si_sdr_loss(est_residual, residual_ref, sample_weight=active_sample_weight)
    else:
        residual_si_sdr_loss = est_target.new_zeros(())

    if mrstft_loss is not None and float(w_res_stft) > 0.0:
        residual_mrstft_value = mrstft_loss(est_residual, residual_ref, sample_weight=active_sample_weight)
    else:
        residual_mrstft_value = est_target.new_zeros(())

    if float(w_residual_target_leak) > 0.0:
        residual_target_leakage = projection_energy_ratio_loss(
            est_residual,
            target,
            sample_weight=active_sample_weight,
        )
    else:
        residual_target_leakage = est_target.new_zeros(())

    if float(w_target_residual_leak) > 0.0:
        target_residual_leakage = projection_energy_ratio_loss(
            est_target,
            residual_ref,
            sample_weight=active_sample_weight,
        )
    else:
        target_residual_leakage = est_target.new_zeros(())

    if float(w_cons) > 0.0:
        if str(consistency_scope).lower() == 'all':
            consistency_weight = None
        else:
            consistency_weight = active_sample_weight
        consistency_loss = l1_wave_loss(est_target + est_residual, mix, sample_weight=consistency_weight)
    else:
        consistency_loss = est_target.new_zeros(())

    chunk_loss_key = str(chunk_loss_type).lower()
    if float(w_chunk) > 0.0:
        if chunk_loss_key != 'soft_focal_bce':
            raise ValueError(f'unsupported chunk_loss_type: {chunk_loss_type}')
        chunk_activity_loss = masked_soft_focal_bce_loss(
            chunk_logits,
            chunk_activity_target_soft,
            weight=chunk_activity_weight,
            focal_gamma=float(chunk_focal_gamma),
            pos_weight=chunk_pos_weight,
        )
    else:
        chunk_activity_loss = est_target.new_zeros(())

    clip_activity_loss = (
        clip_activity_bce_loss(
            clip_logits,
            clip_activity,
            pos_weight=clip_pos_weight,
            focal_gamma=clip_focal_gamma,
        )
        if float(w_clip) > 0.0
        else est_target.new_zeros(())
    )

    if inactive_target_zero_type != 'energy_ratio':
        raise ValueError(f'unsupported inactive_target_zero_type: {inactive_target_zero_type}')

    present_inactive_mask = (clip_activity > 0.5).to(dtype=est_target.dtype)
    present_inactive_suppression = (
        masked_wave_energy_ratio_loss(
            est_target,
            mix,
            mask=inactive_wave_weight,
            sample_weight=present_inactive_mask,
        )
        if float(w_inactive) > 0.0
        else est_target.new_zeros(())
    )

    inactive_clip_target_zero = (
        inactive_target_zero_energy_loss(est_target, mix, sample_weight=inactive_clip)
        if float(w_abs_zero) > 0.0
        else est_target.new_zeros(())
    )
    inactive_clip_residual_mix = (
        l1_wave_loss(est_residual, mix, sample_weight=inactive_clip)
        if float(w_abs_res) > 0.0
        else est_target.new_zeros(())
    )

    total = (
        (float(w_si) * si_sdr_loss)
        + (float(w_stft) * mrstft_value)
        + (float(w_res_si) * residual_si_sdr_loss)
        + (float(w_res_stft) * residual_mrstft_value)
        + (float(w_residual_target_leak) * residual_target_leakage)
        + (float(w_target_residual_leak) * target_residual_leakage)
        + (float(w_cons) * consistency_loss)
        + (float(w_chunk) * chunk_activity_loss)
        + (float(w_clip) * clip_activity_loss)
        + (float(w_inactive) * present_inactive_suppression)
        + (float(w_abs_zero) * inactive_clip_target_zero)
        + (float(w_abs_res) * inactive_clip_residual_mix)
    )

    breakdown = AuraPATemporalLossBreakdown(
        total=total,
        si_sdr=si_sdr_loss,
        mrstft=mrstft_value,
        residual_si_sdr=residual_si_sdr_loss,
        residual_mrstft=residual_mrstft_value,
        residual_target_leakage=residual_target_leakage,
        target_residual_leakage=target_residual_leakage,
        consistency=consistency_loss,
        chunk_activity=chunk_activity_loss,
        clip_activity=clip_activity_loss,
        present_inactive_suppression=present_inactive_suppression,
        inactive_clip_target_zero=inactive_clip_target_zero,
        inactive_clip_residual_mix=inactive_clip_residual_mix,
        num_speaker_match=speaker_match.sum(),
        num_active_clips=clip_activity.sum(),
        num_inactive_clips=inactive_clip.sum(),
        num_silent_present=silent_present.sum(),
        weighted_chunk_count=chunk_activity_weight.sum(),
        weighted_chunk_positive=torch.sum(chunk_activity_weight * chunk_activity_target_soft),
    )
    return total, breakdown.as_dict()


def compute_aura_pa_loss(**kwargs):
    return compute_aura_pa_temporal_loss(**kwargs)
