import math
import importlib
from typing import Dict, Iterable, Optional

import numpy as np
import torch

try:
    pesq_mod = importlib.import_module("pesq")
    pesq_fn = getattr(pesq_mod, "pesq", None)
except Exception:
    pesq_fn = None

try:
    pystoi_mod = importlib.import_module("pystoi")
    stoi_fn = getattr(pystoi_mod, "stoi", None)
except Exception:
    stoi_fn = None


class AverageMeter:
    def __init__(self):
        self._sum: Dict[str, float] = {}
        self._count: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            self._sum[key] = self._sum.get(key, 0.0) + numeric
            self._count[key] = self._count.get(key, 0) + 1

    def average(self) -> Dict[str, float]:
        if not self._count:
            return {}
        out: Dict[str, float] = {}
        for key, total in self._sum.items():
            cnt = self._count.get(key, 0)
            if cnt > 0:
                out[key] = total / cnt
        return out

    def state(self) -> Dict[str, Dict[str, float]]:
        return {
            "sum": dict(self._sum),
            "count": {key: float(value) for key, value in self._count.items()},
        }


def as_float_metrics(total_loss, loss_dict) -> Dict[str, float]:
    abs_sil = loss_dict.get("abs_sil")
    abs_res = loss_dict.get("abs_res")

    return {
        "total_loss": float(total_loss.item()),
        "target_loss": float(loss_dict["target"].item()),
        "consist_loss": float(loss_dict["consist"].item()),
        "pitch_loss": float(loss_dict["pitch"].item()),
        "residual_loss": float(loss_dict["res"].item()),
        "abs_silence_loss": float(abs_sil.item()) if abs_sil is not None else 0.0,
        "abs_residual_loss": float(abs_res.item()) if abs_res is not None else 0.0,
    }


def _safe_sdr(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    signal = float(np.sum(target ** 2))
    noise = float(np.sum((target - pred) ** 2))
    return 10.0 * math.log10((signal + eps) / (noise + eps))


def _mean_or_nan(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def compute_audio_metrics(
    est_target,
    gt_target,
    sample_rate: int,
    use_pesq: bool = True,
    use_stoi: bool = True,
    use_sdr: bool = True,
) -> Dict[str, float]:
    # Numpy does not support direct conversion from bfloat16 tensors.
    pred_np = est_target.detach().to(torch.float32).cpu().numpy()
    gt_np = gt_target.detach().to(torch.float32).cpu().numpy()

    pesq_scores = []
    stoi_scores = []
    sdr_scores = []

    for pred, target in zip(pred_np, gt_np):
        pred = pred.astype(np.float64)
        target = target.astype(np.float64)

        if use_sdr:
            sdr_scores.append(_safe_sdr(pred, target))

        if use_pesq and pesq_fn is not None and sample_rate in (8000, 16000):
            mode = "wb" if sample_rate == 16000 else "nb"
            try:
                pesq_scores.append(float(pesq_fn(sample_rate, target, pred, mode)))
            except Exception:
                pass

        if use_stoi and stoi_fn is not None:
            try:
                stoi_scores.append(float(stoi_fn(target, pred, sample_rate, extended=False)))
            except Exception:
                pass

    return {
        "pesq": _mean_or_nan(pesq_scores) if use_pesq else float("nan"),
        "stoi": _mean_or_nan(stoi_scores) if use_stoi else float("nan"),
        "sdr": _mean_or_nan(sdr_scores) if use_sdr else float("nan"),
    }
