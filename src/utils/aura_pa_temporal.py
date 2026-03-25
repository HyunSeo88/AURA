from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkGrid:
    num_samples: int
    sample_rate: int
    chunk_length_ms: int
    chunk_hop_ms: int
    chunk_length_samples: int
    chunk_hop_samples: int
    starts: torch.Tensor
    ends: torch.Tensor
    center_sec: torch.Tensor

    @property
    def num_chunks(self) -> int:
        return int(self.starts.numel())


@dataclass(frozen=True)
class AuraPATemporalTargets:
    speaker_match_target: torch.Tensor
    clip_activity_target: torch.Tensor
    chunk_activity_target_soft: torch.Tensor
    chunk_activity_weight: torch.Tensor
    chunk_eval_target_strict: torch.Tensor
    chunk_eval_mask_strict: torch.Tensor
    chunk_eval_target_tolerant: torch.Tensor
    chunk_eval_mask_tolerant: torch.Tensor
    chunk_active_seed_mask: torch.Tensor
    chunk_active_mask: torch.Tensor
    chunk_inactive_mask: torch.Tensor
    inactive_wave_weight: torch.Tensor
    chunk_center_sec: torch.Tensor
    silent_present_mask: torch.Tensor


def build_chunk_grid(
    num_samples: int,
    *,
    sample_rate: int = 16000,
    chunk_length_ms: int = 100,
    chunk_hop_ms: int = 50,
    device: torch.device | None = None,
) -> ChunkGrid:
    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if int(sample_rate) <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    chunk_length_samples = max(1, int(round(float(chunk_length_ms) * int(sample_rate) / 1000.0)))
    chunk_hop_samples = max(1, int(round(float(chunk_hop_ms) * int(sample_rate) / 1000.0)))

    if num_samples <= chunk_length_samples:
        starts = torch.tensor([0], device=device, dtype=torch.long)
        ends = torch.tensor([int(num_samples)], device=device, dtype=torch.long)
    else:
        max_start = int(num_samples) - chunk_length_samples
        starts = torch.arange(0, max_start + 1, chunk_hop_samples, device=device, dtype=torch.long)
        if int(starts[-1].item()) != max_start:
            starts = torch.cat([starts, torch.tensor([max_start], device=device, dtype=torch.long)], dim=0)
        ends = (starts + chunk_length_samples).clamp_max(int(num_samples))

    center_sec = ((starts.to(torch.float32) + ends.to(torch.float32)) * 0.5) / float(sample_rate)
    return ChunkGrid(
        num_samples=int(num_samples),
        sample_rate=int(sample_rate),
        chunk_length_ms=int(chunk_length_ms),
        chunk_hop_ms=int(chunk_hop_ms),
        chunk_length_samples=int(chunk_length_samples),
        chunk_hop_samples=int(chunk_hop_samples),
        starts=starts,
        ends=ends,
        center_sec=center_sec,
    )


def build_latent_chunk_ranges(
    *,
    num_samples: int,
    num_steps: int,
    grid: ChunkGrid,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if int(num_samples) != int(grid.num_samples):
        raise ValueError(
            f"num_samples mismatch between waveform and chunk grid: {num_samples} vs {grid.num_samples}"
        )

    starts = torch.floor(grid.starts.to(torch.float32) * float(num_steps) / float(num_samples)).to(torch.long)
    ends = torch.ceil(grid.ends.to(torch.float32) * float(num_steps) / float(num_samples)).to(torch.long)
    starts = starts.clamp(min=0, max=max(0, int(num_steps) - 1))
    ends = ends.clamp(min=1, max=int(num_steps))
    ends = torch.maximum(ends, starts + 1)
    if device is not None:
        starts = starts.to(device)
        ends = ends.to(device)
    return starts, ends


def chunk_mean_pool_sequence(
    sequence: torch.Tensor,
    *,
    step_starts: torch.Tensor,
    step_ends: torch.Tensor,
) -> torch.Tensor:
    if sequence.dim() != 3:
        raise ValueError(f"sequence must have shape [B, C, L], got {tuple(sequence.shape)}")
    if step_starts.dim() != 1 or step_ends.dim() != 1 or step_starts.shape != step_ends.shape:
        raise ValueError("step_starts and step_ends must be 1D tensors of the same shape")

    pooled = []
    for start, end in zip(step_starts.tolist(), step_ends.tolist()):
        chunk = sequence[..., int(start):int(end)]
        pooled.append(chunk.mean(dim=-1, keepdim=True))
    return torch.cat(pooled, dim=-1)


def overlap_normalized_wave_weight(
    chunk_mask: torch.Tensor,
    *,
    grid: ChunkGrid,
    num_samples: int,
) -> torch.Tensor:
    if chunk_mask.dim() != 2:
        raise ValueError(f"chunk_mask must have shape [B, S], got {tuple(chunk_mask.shape)}")
    if int(num_samples) != int(grid.num_samples):
        raise ValueError(f"num_samples mismatch: {num_samples} vs grid {grid.num_samples}")
    if chunk_mask.size(1) != grid.num_chunks:
        raise ValueError(
            f"chunk count mismatch: chunk_mask has {chunk_mask.size(1)}, grid has {grid.num_chunks}"
        )

    batch_size = int(chunk_mask.size(0))
    device = chunk_mask.device
    weight_sum = torch.zeros(batch_size, int(num_samples), device=device, dtype=torch.float32)
    overlap = torch.zeros(int(num_samples), device=device, dtype=torch.float32)
    for idx, (start, end) in enumerate(zip(grid.starts.tolist(), grid.ends.tolist())):
        overlap[int(start):int(end)] += 1.0
        weight_sum[:, int(start):int(end)] += chunk_mask[:, idx:idx + 1].to(torch.float32)
    normalized = weight_sum / overlap.clamp_min(1.0).unsqueeze(0)
    return normalized.clamp_(0.0, 1.0)


def _compute_chunk_energy(target_wave: torch.Tensor, grid: ChunkGrid) -> torch.Tensor:
    if target_wave.dim() != 2:
        raise ValueError(f"target_wave must have shape [B, T], got {tuple(target_wave.shape)}")
    energies = []
    for start, end in zip(grid.starts.tolist(), grid.ends.tolist()):
        chunk = target_wave[:, int(start):int(end)]
        energies.append(chunk.pow(2).mean(dim=-1, keepdim=True))
    return torch.cat(energies, dim=-1)


def _dilate_binary_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if int(radius) <= 0:
        return mask.to(torch.bool)
    if mask.dim() != 2:
        raise ValueError(f"mask must have shape [B, S], got {tuple(mask.shape)}")
    pad = int(radius)
    kernel = torch.ones(1, 1, (2 * pad) + 1, device=mask.device, dtype=torch.float32)
    expanded = torch.nn.functional.conv1d(
        mask.to(torch.float32).unsqueeze(1),
        kernel,
        padding=pad,
    )
    return expanded.squeeze(1) > 0.0


def build_chunk_activity_targets(
    *,
    target_wave: torch.Tensor,
    speaker_match_target: torch.Tensor,
    sample_rate: int = 16000,
    chunk_length_ms: int = 100,
    chunk_hop_ms: int = 50,
    ref_percentile: float = 95.0,
    positive_db: float = -20.0,
    negative_db: float = -35.0,
    temperature_db: float = 3.0,
    ambiguous_weight: float = 0.25,
    dilation_chunks: int = 1,
    min_active_chunks: int = 2,
    energy_floor_db: float = -80.0,
    eps: float = 1e-8,
) -> AuraPATemporalTargets:
    if target_wave.dim() != 2:
        raise ValueError(f"target_wave must have shape [B, T], got {tuple(target_wave.shape)}")
    if speaker_match_target.dim() == 2 and speaker_match_target.size(-1) == 1:
        speaker_match_target = speaker_match_target.squeeze(-1)
    if speaker_match_target.dim() != 1 or speaker_match_target.numel() != target_wave.size(0):
        raise ValueError(
            f"speaker_match_target must have shape [B], got {tuple(speaker_match_target.shape)} for batch_size={target_wave.size(0)}"
        )

    batch_size, num_samples = target_wave.shape
    device = target_wave.device
    speaker_match_target = speaker_match_target.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    grid = build_chunk_grid(
        num_samples=int(num_samples),
        sample_rate=int(sample_rate),
        chunk_length_ms=int(chunk_length_ms),
        chunk_hop_ms=int(chunk_hop_ms),
        device=device,
    )
    num_chunks = grid.num_chunks
    chunk_energy = _compute_chunk_energy(target_wave, grid)

    mid_db = 0.5 * (float(positive_db) + float(negative_db))
    energy_floor = 10.0 ** (float(energy_floor_db) / 10.0)

    clip_activity_target = torch.zeros(batch_size, device=device, dtype=torch.float32)
    chunk_activity_target_soft = torch.zeros(batch_size, num_chunks, device=device, dtype=torch.float32)
    chunk_activity_weight = torch.ones(batch_size, num_chunks, device=device, dtype=torch.float32)
    chunk_eval_target_strict = torch.zeros(batch_size, num_chunks, device=device, dtype=torch.bool)
    chunk_eval_mask_strict = torch.ones(batch_size, num_chunks, device=device, dtype=torch.bool)
    chunk_eval_target_tolerant = torch.zeros(batch_size, num_chunks, device=device, dtype=torch.bool)
    chunk_eval_mask_tolerant = torch.ones(batch_size, num_chunks, device=device, dtype=torch.bool)
    chunk_active_seed_mask = torch.zeros(batch_size, num_chunks, device=device, dtype=torch.bool)
    chunk_active_mask = torch.zeros(batch_size, num_chunks, device=device, dtype=torch.bool)
    chunk_inactive_mask = torch.ones(batch_size, num_chunks, device=device, dtype=torch.bool)
    silent_present_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)

    for batch_idx in range(batch_size):
        is_speaker_match = bool(float(speaker_match_target[batch_idx].item()) >= 0.5)
        if not is_speaker_match:
            continue

        energies = chunk_energy[batch_idx]
        valid_ref = energies[energies > float(energy_floor)]
        if valid_ref.numel() == 0:
            silent_present_mask[batch_idx] = True
            continue

        ref_energy = torch.quantile(valid_ref, float(ref_percentile) / 100.0)
        if not torch.isfinite(ref_energy) or float(ref_energy.item()) <= float(energy_floor):
            silent_present_mask[batch_idx] = True
            continue

        relative_db = 10.0 * torch.log10((energies + float(eps)) / (ref_energy + float(eps)))
        active_seed = relative_db >= float(positive_db)
        if int(active_seed.sum().item()) < int(min_active_chunks):
            silent_present_mask[batch_idx] = True
            continue

        chunk_activity_target_soft[batch_idx] = torch.sigmoid((relative_db - float(mid_db)) / float(temperature_db))
        ambiguous = (relative_db < float(positive_db)) & (relative_db > float(negative_db))
        chunk_activity_weight[batch_idx, ambiguous] = float(ambiguous_weight)

        tolerant_active = _dilate_binary_mask(active_seed.unsqueeze(0), int(dilation_chunks)).squeeze(0)
        negative_seed = relative_db <= float(negative_db)
        inactive_mask = negative_seed & (~tolerant_active)

        clip_activity_target[batch_idx] = 1.0
        chunk_active_seed_mask[batch_idx] = active_seed
        chunk_active_mask[batch_idx] = tolerant_active
        chunk_inactive_mask[batch_idx] = inactive_mask

        chunk_eval_target_strict[batch_idx] = active_seed
        chunk_eval_mask_strict[batch_idx] = active_seed | negative_seed
        chunk_eval_target_tolerant[batch_idx] = tolerant_active
        chunk_eval_mask_tolerant[batch_idx] = tolerant_active | negative_seed

    inactive_wave_weight = overlap_normalized_wave_weight(
        chunk_inactive_mask.to(torch.float32),
        grid=grid,
        num_samples=int(num_samples),
    )

    return AuraPATemporalTargets(
        speaker_match_target=speaker_match_target,
        clip_activity_target=clip_activity_target,
        chunk_activity_target_soft=chunk_activity_target_soft,
        chunk_activity_weight=chunk_activity_weight,
        chunk_eval_target_strict=chunk_eval_target_strict,
        chunk_eval_mask_strict=chunk_eval_mask_strict,
        chunk_eval_target_tolerant=chunk_eval_target_tolerant,
        chunk_eval_mask_tolerant=chunk_eval_mask_tolerant,
        chunk_active_seed_mask=chunk_active_seed_mask,
        chunk_active_mask=chunk_active_mask,
        chunk_inactive_mask=chunk_inactive_mask,
        inactive_wave_weight=inactive_wave_weight,
        chunk_center_sec=grid.center_sec,
        silent_present_mask=silent_present_mask,
    )


build_chunk_presence_targets = build_chunk_activity_targets
