import torch
import torch.nn.functional as F


def _validate_sample_weight(sample_weight, batch_size: int, device, dtype):
    if sample_weight is None:
        return None
    weight = sample_weight.to(device=device, dtype=dtype).reshape(-1)
    if weight.shape[0] != batch_size:
        raise ValueError(
            f"sample_weight batch mismatch: loss_batch={batch_size} "
            f"weight_batch={weight.shape[0]}"
        )
    return weight


def _weighted_mean(per_sample: torch.Tensor, sample_weight=None):
    if sample_weight is None:
        return torch.mean(per_sample)

    weight = _validate_sample_weight(
        sample_weight=sample_weight,
        batch_size=per_sample.shape[0],
        device=per_sample.device,
        dtype=per_sample.dtype,
    )
    denom = weight.sum().clamp_min(1.0)
    return torch.sum(per_sample * weight) / denom


def _si_snr_per_sample(preds, targets, eps=1e-8):
    preds = preds - torch.mean(preds, dim=-1, keepdim=True)
    targets = targets - torch.mean(targets, dim=-1, keepdim=True)

    target_energy = torch.sum(targets ** 2, dim=-1, keepdim=True) + eps
    projected = (torch.sum(preds * targets, dim=-1, keepdim=True) * targets) / target_energy
    noise = preds - projected

    snr = 10 * torch.log10(
        torch.sum(projected ** 2, dim=-1, keepdim=True)
        / (torch.sum(noise ** 2, dim=-1, keepdim=True) + eps)
        + eps
    )
    return -snr.squeeze(-1)


def si_snr_loss(preds, targets, sample_weight=None, eps=1e-8):
    per_sample = _si_snr_per_sample(preds, targets, eps=eps)
    return _weighted_mean(per_sample, sample_weight=sample_weight)


def l1_wave_loss(preds, targets, sample_weight=None):
    per_sample = torch.mean(torch.abs(preds - targets), dim=-1)
    return _weighted_mean(per_sample, sample_weight=sample_weight)


def pitch_l1_loss(pred_pitch, gt_pitch, voiced_mask=None):
    if voiced_mask is None:
        return F.l1_loss(pred_pitch, gt_pitch)

    mask = voiced_mask.float()
    denom = mask.sum().clamp_min(1.0)
    return torch.sum(torch.abs(pred_pitch - gt_pitch) * mask) / denom


def aura_loss(
    est_target,
    est_residual,
    gt_target,
    mix_input,
    target_present_mask=None,
    pred_pitch=None,
    gt_pitch=None,
    pitch_voiced_mask=None,
    gt_residual=None,
    w_target=1.0,
    w_consistency=0.1,
    w_pitch=0.2,
    w_residual=0.0,
    w_absent_silence=0.5,
    w_absent_residual=0.5,
):
    if target_present_mask is None:
        present_mask = est_target.new_ones(est_target.shape[0])
    else:
        present_mask = _validate_sample_weight(
            sample_weight=target_present_mask,
            batch_size=est_target.shape[0],
            device=est_target.device,
            dtype=est_target.dtype,
        )
    absent_mask = (1.0 - present_mask).clamp_min(0.0)

    # Present samples: optimize separation quality of target stream.
    loss_target = si_snr_loss(est_target, gt_target, sample_weight=present_mask)

    est_mix = est_target + est_residual
    loss_consist = l1_wave_loss(est_mix, mix_input, sample_weight=present_mask)

    if pred_pitch is not None and gt_pitch is not None:
        if pitch_voiced_mask is not None:
            pitch_mask = pitch_voiced_mask.float() * present_mask.unsqueeze(-1)
        else:
            pitch_mask = None
        loss_pitch = pitch_l1_loss(pred_pitch, gt_pitch, pitch_mask)
    else:
        loss_pitch = loss_target.new_zeros(())

    # Absent samples: suppress hallucinated target and preserve mix in residual.
    loss_abs_sil = l1_wave_loss(
        est_target,
        torch.zeros_like(est_target),
        sample_weight=absent_mask,
    )
    loss_abs_res = l1_wave_loss(
        est_residual,
        mix_input,
        sample_weight=absent_mask,
    )

    if w_residual > 0.0 and gt_residual is not None:
        loss_res = si_snr_loss(est_residual, gt_residual)
    else:
        loss_res = loss_target.new_zeros(())

    total_loss = (
        (w_target * loss_target)
        + (w_consistency * loss_consist)
        + (w_pitch * loss_pitch)
        + (w_residual * loss_res)
        + (w_absent_silence * loss_abs_sil)
        + (w_absent_residual * loss_abs_res)
    )

    return total_loss, {
        "target": loss_target,
        "consist": loss_consist,
        "pitch": loss_pitch,
        "res": loss_res,
        "abs_sil": loss_abs_sil,
        "abs_res": loss_abs_res,
    }
