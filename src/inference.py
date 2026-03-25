import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from aura_v1 import AuraTeacher


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def _resolve_config_path(checkpoint_path: Path, config_path: Optional[str]) -> Optional[Path]:
    if config_path:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"config not found: {p}")
        return p

    # Typical layout:
    # experiments/<exp_name>/checkpoints/best_model.pth
    # experiments/<exp_name>/config/resolved_config.json
    exp_dir = checkpoint_path.parent.parent
    candidate = exp_dir / "config" / "resolved_config.json"
    if candidate.exists():
        return candidate
    return None


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state_dict = ckpt_obj["model_state_dict"]
    elif isinstance(ckpt_obj, dict):
        state_dict = ckpt_obj
    else:
        raise ValueError("Unsupported checkpoint format")

    # Strip DDP prefix if present.
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _safe_mono_resampled(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim != 2:
        raise ValueError(f"invalid audio shape at {path}: {tuple(wav.shape)}")
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0).float()


def _resolve_path(base_dir: Path, raw_path: Optional[str]) -> Optional[Path]:
    if raw_path is None:
        return None
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _fit_embedding_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    x = x.flatten().float()
    if x.numel() == target_dim:
        out = x
    elif x.numel() > target_dim:
        out = x[:target_dim]
    else:
        out = F.pad(x, (0, target_dim - x.numel()))
    return F.normalize(out, p=2, dim=0, eps=1e-8)


def _load_enroll_embedding_from_row(row: Dict, audio_base_dir: Path, enroll_dim: int) -> Optional[torch.Tensor]:
    emb = row.get("enroll_embedding")
    if emb is not None:
        return _fit_embedding_dim(torch.tensor(emb, dtype=torch.float32), enroll_dim)

    emb_path_raw = row.get("enroll_emb_path")
    if emb_path_raw:
        emb_path = _resolve_path(audio_base_dir, emb_path_raw)
        if emb_path is None or not emb_path.exists():
            raise FileNotFoundError(f"enroll_emb_path not found: {emb_path_raw}")

        ext = emb_path.suffix.lower()
        if ext == ".npy":
            arr = np.load(str(emb_path))
            return _fit_embedding_dim(torch.from_numpy(arr), enroll_dim)
        if ext in {".pt", ".pth"}:
            t = torch.load(str(emb_path), map_location="cpu")
            if isinstance(t, dict):
                if "embedding" in t:
                    t = t["embedding"]
                elif "emb" in t:
                    t = t["emb"]
            if not torch.is_tensor(t):
                raise ValueError(f"unsupported embedding tensor format: {emb_path}")
            return _fit_embedding_dim(t, enroll_dim)
        raise ValueError(f"unsupported enroll_emb_path extension: {emb_path}")

    return None


def _build_spkrec_encoder(source: str, savedir: str, device: str):
    # Imported lazily to avoid overhead when manifest already has embeddings.
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source=source,
        savedir=savedir,
        run_opts={"device": device},
    )
    encoder.eval()
    return encoder


def _extract_enroll_embedding_from_wav(
    enroll_wav: torch.Tensor,
    enroll_dim: int,
    spkrec_source: str,
    spkrec_savedir: str,
    spkrec_device: str,
    fallback_device: str,
) -> torch.Tensor:
    encoder = _build_spkrec_encoder(spkrec_source, spkrec_savedir, spkrec_device)
    wav_2d = enroll_wav.unsqueeze(0)
    try:
        with torch.no_grad():
            wav_dev = wav_2d.to(spkrec_device)
            emb = encoder.encode_batch(wav_dev)
            emb = emb.squeeze().detach().float().cpu()
            return _fit_embedding_dim(emb, enroll_dim)
    except RuntimeError as exc:
        msg = str(exc)
        is_cuda_runtime = ("cuFFT" in msg) or ("CUDA" in msg)
        if not str(spkrec_device).startswith("cuda") or not is_cuda_runtime:
            raise

        fallback_encoder = _build_spkrec_encoder(spkrec_source, spkrec_savedir, fallback_device)
        with torch.no_grad():
            wav_fb = wav_2d.to(fallback_device)
            emb = fallback_encoder.encode_batch(wav_fb)
            emb = emb.squeeze().detach().float().cpu()
            return _fit_embedding_dim(emb, enroll_dim)


def _si_snr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred - pred.mean()
    target = target - target.mean()

    target_energy = (target ** 2).sum() + eps
    projected = ((pred * target).sum() * target) / target_energy
    noise = pred - projected
    ratio = (projected ** 2).sum() / ((noise ** 2).sum() + eps)
    return float(10.0 * torch.log10(ratio + eps).item())


def _sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    signal = float((target ** 2).sum().item())
    noise = float(((target - pred) ** 2).sum().item())
    return float(10.0 * math.log10((signal + eps) / (noise + eps)))


def _optional_stoi(pred: np.ndarray, target: np.ndarray, sample_rate: int) -> float:
    try:
        from pystoi import stoi
    except Exception:
        return float("nan")
    try:
        return float(stoi(target, pred, sample_rate, extended=False))
    except Exception:
        return float("nan")


def _optional_pesq(pred: np.ndarray, target: np.ndarray, sample_rate: int) -> float:
    if sample_rate not in (8000, 16000):
        return float("nan")
    try:
        from pesq import pesq
    except Exception:
        return float("nan")
    mode = "wb" if sample_rate == 16000 else "nb"
    try:
        return float(pesq(sample_rate, target, pred, mode))
    except Exception:
        return float("nan")


def _infer_chunked(
    model: AuraTeacher,
    mix: torch.Tensor,
    enroll_vec: torch.Tensor,
    segment_samples: int,
    hop_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert mix.ndim == 1
    length = int(mix.numel())
    if length <= 0:
        raise ValueError("empty mix waveform")

    if segment_samples <= 0:
        raise ValueError("segment_samples must be > 0")
    if hop_samples <= 0:
        raise ValueError("hop_samples must be > 0")

    window = torch.hann_window(segment_samples, periodic=False, device=device)
    window = window.clamp_min(1e-3)

    out_target = torch.zeros(length, dtype=torch.float32, device=device)
    out_residual = torch.zeros(length, dtype=torch.float32, device=device)
    out_pitch = torch.zeros(length, dtype=torch.float32, device=device)
    wsum = torch.zeros(length, dtype=torch.float32, device=device)
    pitch_wsum = torch.zeros(length, dtype=torch.float32, device=device)

    has_pitch = None

    with torch.no_grad():
        for start in range(0, length, hop_samples):
            end = start + segment_samples
            valid_end = min(end, length)
            valid_len = valid_end - start
            if valid_len <= 0:
                continue

            seg = torch.zeros(segment_samples, dtype=torch.float32, device=device)
            seg[:valid_len] = mix[start:valid_end]

            est_target, est_residual, aux = model(
                seg.unsqueeze(0),
                enroll_vec.unsqueeze(0),
                return_aux=True,
            )
            est_target = est_target.squeeze(0)[:segment_samples]
            est_residual = est_residual.squeeze(0)[:segment_samples]

            w = window[:valid_len]
            out_target[start:valid_end] += est_target[:valid_len] * w
            out_residual[start:valid_end] += est_residual[:valid_len] * w
            wsum[start:valid_end] += w

            pitch_pred = aux.get("pitch_pred")
            if pitch_pred is not None:
                has_pitch = True
                pitch_up = F.interpolate(
                    pitch_pred.unsqueeze(1),
                    size=segment_samples,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                out_pitch[start:valid_end] += pitch_up[:valid_len] * w
                pitch_wsum[start:valid_end] += w
            elif has_pitch is None:
                has_pitch = False

            if valid_end >= length:
                break

    out_target = out_target / wsum.clamp_min(1e-6)
    out_residual = out_residual / wsum.clamp_min(1e-6)

    if has_pitch:
        out_pitch = out_pitch / pitch_wsum.clamp_min(1e-6)
        return out_target, out_residual, out_pitch
    return out_target, out_residual, None


def _collect_input_from_manifest(args, sample_rate: int, enroll_dim: int):
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    rows = _load_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"empty manifest: {manifest_path}")

    selected = None
    if args.sample_id is not None:
        for row in rows:
            if str(row.get("id")) == str(args.sample_id):
                selected = row
                break
        if selected is None:
            raise ValueError(f"sample_id not found in manifest: {args.sample_id}")
    else:
        idx = int(args.sample_index)
        if idx < 0 or idx >= len(rows):
            raise IndexError(f"sample_index out of range: {idx} (rows={len(rows)})")
        selected = rows[idx]

    audio_base_dir = Path(args.audio_base_dir) if args.audio_base_dir else manifest_path.parent

    mix_path = _resolve_path(audio_base_dir, selected.get("mix_path"))
    if mix_path is None:
        raise ValueError("manifest row has null mix_path")
    if not mix_path.exists():
        raise FileNotFoundError(f"mix_path not found: {mix_path}")

    enroll_vec = _load_enroll_embedding_from_row(selected, audio_base_dir, enroll_dim)

    enroll_wav_path = _resolve_path(audio_base_dir, selected.get("enroll_path"))
    if enroll_vec is None:
        if enroll_wav_path is None or not enroll_wav_path.exists():
            raise ValueError(
                "Manifest row has no usable enrollment. "
                "Need enroll_embedding/enroll_emb_path or valid enroll_path."
            )
        enroll_wav = _safe_mono_resampled(enroll_wav_path, sample_rate)
        enroll_vec = _extract_enroll_embedding_from_wav(
            enroll_wav=enroll_wav,
            enroll_dim=enroll_dim,
            spkrec_source=args.spkrec_source,
            spkrec_savedir=args.spkrec_savedir,
            spkrec_device=args.spkrec_device,
            fallback_device=args.fallback_device,
        )

    target_path = _resolve_path(audio_base_dir, selected.get("target_path"))
    if target_path is not None and not target_path.exists():
        target_path = None

    mix = _safe_mono_resampled(mix_path, sample_rate)
    gt_target = _safe_mono_resampled(target_path, sample_rate) if target_path else None

    sample_name = str(selected.get("id", mix_path.stem))
    meta = {
        "mode": "manifest",
        "manifest_path": str(manifest_path),
        "sample_id": str(selected.get("id")),
        "mix_path": str(mix_path),
        "enroll_path": str(enroll_wav_path) if enroll_wav_path else None,
        "target_path": str(target_path) if target_path else None,
    }
    return mix, enroll_vec, gt_target, sample_name, meta


def _collect_input_from_direct(args, sample_rate: int, enroll_dim: int):
    mix_path = Path(args.mix_wav)
    if not mix_path.exists():
        raise FileNotFoundError(f"mix_wav not found: {mix_path}")
    mix = _safe_mono_resampled(mix_path, sample_rate)

    if args.enroll_emb_path:
        enroll_emb_path = Path(args.enroll_emb_path)
        if not enroll_emb_path.exists():
            raise FileNotFoundError(f"enroll_emb_path not found: {enroll_emb_path}")
        ext = enroll_emb_path.suffix.lower()
        if ext == ".npy":
            enroll_vec = _fit_embedding_dim(torch.from_numpy(np.load(str(enroll_emb_path))), enroll_dim)
        elif ext in {".pt", ".pth"}:
            emb = torch.load(str(enroll_emb_path), map_location="cpu")
            if isinstance(emb, dict):
                emb = emb.get("embedding", emb.get("emb", emb))
            if not torch.is_tensor(emb):
                raise ValueError(f"unsupported tensor format: {enroll_emb_path}")
            enroll_vec = _fit_embedding_dim(emb, enroll_dim)
        else:
            raise ValueError(f"unsupported enroll_emb_path extension: {enroll_emb_path}")
        enroll_path_for_meta = None
    else:
        if not args.enroll_wav:
            raise ValueError("direct mode requires either --enroll-wav or --enroll-emb-path")
        enroll_path = Path(args.enroll_wav)
        if not enroll_path.exists():
            raise FileNotFoundError(f"enroll_wav not found: {enroll_path}")
        enroll_wav = _safe_mono_resampled(enroll_path, sample_rate)
        enroll_vec = _extract_enroll_embedding_from_wav(
            enroll_wav=enroll_wav,
            enroll_dim=enroll_dim,
            spkrec_source=args.spkrec_source,
            spkrec_savedir=args.spkrec_savedir,
            spkrec_device=args.spkrec_device,
            fallback_device=args.fallback_device,
        )
        enroll_path_for_meta = str(enroll_path)

    gt_target = None
    if args.target_wav:
        target_path = Path(args.target_wav)
        if not target_path.exists():
            raise FileNotFoundError(f"target_wav not found: {target_path}")
        gt_target = _safe_mono_resampled(target_path, sample_rate)
    else:
        target_path = None

    sample_name = args.sample_name or mix_path.stem
    meta = {
        "mode": "direct",
        "mix_path": str(mix_path),
        "enroll_path": enroll_path_for_meta,
        "target_path": str(target_path) if target_path else None,
    }
    return mix, enroll_vec, gt_target, sample_name, meta


def _save_wav(path: Path, wav: torch.Tensor, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_2d = wav.detach().cpu().unsqueeze(0).clamp(-1.0, 1.0)
    torchaudio.save(str(path), wav_2d, sample_rate)


def _build_model(cfg: Dict, device: torch.device):
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model = AuraTeacher(
        enroll_dim=int(model_cfg.get("enroll_dim", data_cfg.get("enroll_dim", 128))),
        device=device,
        use_pitchnet=bool(model_cfg.get("use_pitchnet", True)),
        sepformer_source=str(model_cfg.get("sepformer_source", "speechbrain/sepformer-wsj02mix")),
        sepformer_savedir=str(
            model_cfg.get("sepformer_savedir", "pretrained_models/sepformer-wsj02mix")
        ),
    )
    model.eval()
    return model


def _default_out_dir(args, checkpoint_path: Path) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    exp_dir = checkpoint_path.parent.parent
    if exp_dir.exists():
        return exp_dir / "inference"
    return Path("inference")


def _parse_args():
    parser = argparse.ArgumentParser(description="AURA inference (real model, chunked long-form support)")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON (optional)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cuda:0/cpu")
    parser.add_argument("--strict-load", action="store_true", help="Use strict state_dict loading")

    # Direct mode inputs
    parser.add_argument("--mix-wav", type=str, default=None, help="Mixture wav path (direct mode)")
    parser.add_argument("--enroll-wav", type=str, default=None, help="Enrollment wav path (direct mode)")
    parser.add_argument(
        "--enroll-emb-path",
        type=str,
        default=None,
        help="Enrollment embedding path (.npy/.pt/.pth) for direct mode",
    )
    parser.add_argument("--target-wav", type=str, default=None, help="Optional GT target wav for metrics")

    # Manifest mode inputs
    parser.add_argument("--manifest", type=str, default=None, help="Manifest jsonl path (manifest mode)")
    parser.add_argument("--sample-id", type=str, default=None, help="Sample id in manifest")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index in manifest when --sample-id is not set",
    )
    parser.add_argument(
        "--audio-base-dir",
        type=str,
        default=None,
        help="Base dir for relative paths in manifest (default: manifest parent)",
    )

    parser.add_argument("--sample-rate", type=int, default=None, help="Override sample rate")
    parser.add_argument(
        "--segment-sec",
        type=float,
        default=None,
        help="Chunk size in seconds. Default uses config.data.duration_sec.",
    )
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=None,
        help="Chunk hop in seconds. Default: segment_sec * (1-overlap_ratio).",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.5,
        help="Used when --hop-sec is not provided. 0.5 => 50% overlap",
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Run single-pass full-length inference (not recommended for faster model)",
    )

    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--sample-name", type=str, default=None, help="Output basename in direct mode")

    parser.add_argument(
        "--spkrec-source",
        type=str,
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="Speaker encoder source for on-the-fly enroll embedding",
    )
    parser.add_argument(
        "--spkrec-savedir",
        type=str,
        default="/workspace/project/pretrained_models/spkrec-ecapa-voxceleb",
        help="Speaker encoder savedir",
    )
    parser.add_argument(
        "--spkrec-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Speaker encoder device",
    )
    parser.add_argument(
        "--fallback-device",
        type=str,
        default="cpu",
        help="Fallback speaker encoder device if CUDA runtime fails",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    config_path = _resolve_config_path(checkpoint_path, args.config)
    cfg = _load_json(config_path) if config_path else {}
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    sample_rate = int(args.sample_rate or data_cfg.get("sample_rate", 16000))
    segment_sec = float(args.segment_sec or data_cfg.get("duration_sec", 3.0))
    if segment_sec <= 0.0:
        raise ValueError(f"segment_sec must be > 0, got {segment_sec}")

    segment_samples = int(round(segment_sec * sample_rate))
    if args.hop_sec is not None:
        hop_sec = float(args.hop_sec)
        if hop_sec <= 0.0:
            raise ValueError(f"hop_sec must be > 0, got {hop_sec}")
    else:
        overlap = float(args.overlap_ratio)
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap}")
        hop_sec = segment_sec * (1.0 - overlap)
        hop_sec = max(hop_sec, 1.0 / sample_rate)
    hop_samples = int(round(hop_sec * sample_rate))
    if (not args.no_chunk) and hop_samples > segment_samples:
        raise ValueError(
            "hop is larger than segment. This can leave uncovered audio gaps. "
            "Set --hop-sec <= --segment-sec or use overlap-ratio <= 0.0."
        )

    enroll_dim = int(model_cfg.get("enroll_dim", data_cfg.get("enroll_dim", 128)))
    device = _resolve_device(args.device)

    if args.manifest:
        mix, enroll_vec, gt_target, sample_name, io_meta = _collect_input_from_manifest(
            args=args,
            sample_rate=sample_rate,
            enroll_dim=enroll_dim,
        )
    else:
        if not args.mix_wav:
            raise ValueError("direct mode requires --mix-wav (or use --manifest)")
        mix, enroll_vec, gt_target, sample_name, io_meta = _collect_input_from_direct(
            args=args,
            sample_rate=sample_rate,
            enroll_dim=enroll_dim,
        )

    out_root = _default_out_dir(args, checkpoint_path)
    out_dir = out_root / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(cfg, device)
    ckpt_obj = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = _resolve_state_dict(ckpt_obj)
    load_result = model.load_state_dict(state_dict, strict=bool(args.strict_load))
    if not args.strict_load:
        missing = getattr(load_result, "missing_keys", [])
        unexpected = getattr(load_result, "unexpected_keys", [])
        if missing:
            print(f"[Load] missing keys: {len(missing)}")
        if unexpected:
            print(f"[Load] unexpected keys: {len(unexpected)}")

    mix_device = mix.to(device)
    enroll_device = enroll_vec.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        if args.no_chunk:
            est_target, est_residual, aux = model(
                mix_device.unsqueeze(0),
                enroll_device.unsqueeze(0),
                return_aux=True,
            )
            est_target = est_target.squeeze(0).float()
            est_residual = est_residual.squeeze(0).float()
            pitch = aux.get("pitch_pred")
            if pitch is not None:
                pitch = pitch.squeeze(0).float()
        else:
            est_target, est_residual, pitch = _infer_chunked(
                model=model,
                mix=mix_device,
                enroll_vec=enroll_device,
                segment_samples=segment_samples,
                hop_samples=hop_samples,
                device=device,
            )
    infer_sec = time.perf_counter() - t0

    # Align with input length.
    length = int(mix.numel())
    est_target = est_target[:length].detach().cpu()
    est_residual = est_residual[:length].detach().cpu()
    recon = est_target + est_residual
    pitch_cpu = pitch[:length].detach().cpu().numpy() if pitch is not None else None

    target_out = out_dir / "target.wav"
    residual_out = out_dir / "residual.wav"
    recon_out = out_dir / "reconstructed_mix.wav"
    pitch_out = out_dir / "pitch_track.npy"

    _save_wav(target_out, est_target, sample_rate)
    _save_wav(residual_out, est_residual, sample_rate)
    _save_wav(recon_out, recon, sample_rate)
    if pitch_cpu is not None:
        np.save(str(pitch_out), pitch_cpu)

    duration_sec = float(length) / float(sample_rate)
    metrics = {
        "sample_rate": sample_rate,
        "num_samples": length,
        "duration_sec": duration_sec,
        "inference_sec": infer_sec,
        "rtf": (infer_sec / duration_sec) if duration_sec > 0 else float("nan"),
        "chunking": {
            "enabled": not bool(args.no_chunk),
            "segment_sec": segment_sec,
            "hop_sec": hop_sec,
            "segment_samples": segment_samples,
            "hop_samples": hop_samples,
        },
        "mix_reconstruction_l1": float(torch.mean(torch.abs(mix - recon)).item()),
        "mix_rms": float(torch.sqrt(torch.mean(mix ** 2)).item()),
        "target_rms": float(torch.sqrt(torch.mean(est_target ** 2)).item()),
        "residual_rms": float(torch.sqrt(torch.mean(est_residual ** 2)).item()),
    }

    if gt_target is not None:
        min_len = min(int(gt_target.numel()), int(est_target.numel()))
        gt_eval = gt_target[:min_len]
        pred_eval = est_target[:min_len]
        pred_np = pred_eval.numpy().astype(np.float64)
        gt_np = gt_eval.numpy().astype(np.float64)
        metrics["with_target"] = {
            "si_snr_db": _si_snr(pred_eval, gt_eval),
            "sdr_db": _sdr(pred_eval, gt_eval),
            "stoi": _optional_stoi(pred_np, gt_np, sample_rate),
            "pesq": _optional_pesq(pred_np, gt_np, sample_rate),
        }

    meta = {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path) if config_path else None,
        "output_dir": str(out_dir),
        "io": io_meta,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Inference] Output dir: {out_dir}")
    print(f"[Inference] target: {target_out}")
    print(f"[Inference] residual: {residual_out}")
    print(f"[Inference] recon: {recon_out}")
    if pitch_cpu is not None:
        print(f"[Inference] pitch: {pitch_out}")
    print(f"[Inference] metrics: {out_dir / 'metrics.json'}")
    print(
        "[Inference] runtime="
        f"{infer_sec:.3f}s, duration={duration_sec:.3f}s, rtf={metrics['rtf']:.4f}"
    )


if __name__ == "__main__":
    main()
