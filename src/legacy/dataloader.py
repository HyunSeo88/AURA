import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


class AuraDataset(Dataset):
    """
    Manifest-based dataset for AURA teacher training.

    Expected manifest fields (from preprocess script):
      - id
      - speaker_id
      - is_target_present (0/1)
      - mix_path
      - enroll_path
      - target_path (nullable for absent samples)
      - pitch_gt_path (nullable)
      - pitch_voiced_mask_path (nullable)

        Enrollment note:
            This loader supports both fixed vector and variable-length sequence embeddings.
            Priority:
                1) enroll_emb_path: path to .npy/.pt embedding ([D] or [T, D])
                2) enroll_embedding: list[float] in manifest (interpreted as [D])
                3) fallback zero vector [D] (temporary default)
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        duration_sec: float = 6.0,
        enroll_dim: int = 128,
        audio_base_dir: Optional[str] = None,
        pitch_base_dir: Optional[str] = None,
        strict_audio_length: bool = False,
        use_zero_enroll_fallback: bool = True,
        random_crop: bool = True,
        partial_audio_load: bool = True,
        normalize_enroll: bool = False,
        enroll_cache_size: int = 4096,
        audio_info_cache_size: int = 16384,
        rows: Optional[List[Dict]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        if rows is None and not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")

        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_sec)
        self.enroll_dim = enroll_dim
        self.strict_audio_length = strict_audio_length
        self.use_zero_enroll_fallback = use_zero_enroll_fallback
        self.random_crop = random_crop
        self.partial_audio_load = partial_audio_load
        self.normalize_enroll = bool(normalize_enroll)
        self.enroll_cache_size = max(0, int(enroll_cache_size))
        self.audio_info_cache_size = max(0, int(audio_info_cache_size))

        manifest_dir = self.manifest_path.parent
        self.audio_base_dir = Path(audio_base_dir) if audio_base_dir else manifest_dir
        self.pitch_base_dir = Path(pitch_base_dir) if pitch_base_dir else manifest_dir

        self.rows = list(rows) if rows is not None else self._load_jsonl(self.manifest_path)
        if not self.rows:
            raise ValueError(f"empty manifest: {self.manifest_path}")

        self._enroll_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._audio_info_cache: OrderedDict[str, Tuple[int, int]] = OrderedDict()
        self._warned_zero_enroll = False

    @staticmethod
    def _lru_get(cache: OrderedDict, key):
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    @staticmethod
    def _lru_put(cache: OrderedDict, key, value, max_size: int):
        if max_size <= 0:
            return
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > max_size:
            cache.popitem(last=False)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict]:
        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_num}: {exc}") from exc
        return rows

    @staticmethod
    def _resolve_path(base_dir: Path, raw_path: Optional[str]) -> Optional[Path]:
        if raw_path is None:
            return None
        p = Path(raw_path)
        if p.is_absolute():
            return p
        return (base_dir / p).resolve()

    def _get_audio_info(self, path: Path) -> Tuple[int, int]:
        key = str(path)
        cached = self._lru_get(self._audio_info_cache, key)
        if cached is not None:
            return cached

        info = torchaudio.info(str(path))
        result = (int(info.sample_rate), int(info.num_frames))
        self._lru_put(self._audio_info_cache, key, result, self.audio_info_cache_size)
        return result

    def _load_audio_segment_1d(
        self,
        path: Path,
        frame_offset: int = 0,
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        kwargs = {}
        if frame_offset > 0:
            kwargs["frame_offset"] = int(frame_offset)
        if num_frames is not None:
            kwargs["num_frames"] = int(max(0, num_frames))

        wav, sr = torchaudio.load(str(path), **kwargs)
        if wav.ndim != 2:
            raise ValueError(f"invalid wav shape at {path}: {tuple(wav.shape)}")

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = wav.squeeze(0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)
        return wav.float()

    def _load_audio_raw_1d(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        if wav.ndim != 2:
            raise ValueError(f"invalid wav shape at {path}: {tuple(wav.shape)}")

        # Mono fold
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = wav.squeeze(0)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)

        return wav.float()

    def _crop_or_pad_wave(self, wav: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        if wav.numel() > self.num_samples:
            end_idx = start_idx + self.num_samples
            wav = wav[start_idx:end_idx]
        elif wav.numel() < self.num_samples:
            wav = F.pad(wav, (0, self.num_samples - wav.numel()))
        return wav

    def _select_crop_start(self, total_len: int) -> int:
        if total_len > self.num_samples:
            max_start = total_len - self.num_samples
            return random.randint(0, max_start) if self.random_crop else 0
        return 0

    def _crop_or_pad_pair(
        self, mix: torch.Tensor, target: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        if target is not None:
            total_len = min(mix.numel(), target.numel())
            mix = mix[:total_len]
            target = target[:total_len]
        else:
            total_len = mix.numel()

        if total_len > self.num_samples:
            max_start = total_len - self.num_samples
            start_idx = random.randint(0, max_start) if self.random_crop else 0
        else:
            start_idx = 0

        mix = self._crop_or_pad_wave(mix, start_idx=start_idx)
        if target is not None:
            target = self._crop_or_pad_wave(target, start_idx=start_idx)

        return mix, target, start_idx, total_len

    def _load_pair_with_optional_partial(
        self,
        mix_path: Path,
        target_path: Optional[Path],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        use_partial = False
        mix_sr = self.sample_rate
        mix_frames = 0
        if self.partial_audio_load:
            mix_sr, mix_frames = self._get_audio_info(mix_path)
            use_partial = mix_sr == self.sample_rate and mix_frames > 0

        if target_path is not None:
            target_sr = self.sample_rate
            target_frames = 0
            if use_partial:
                target_sr, target_frames = self._get_audio_info(target_path)
                use_partial = target_sr == self.sample_rate and target_frames > 0

            if use_partial:
                if self.strict_audio_length:
                    if mix_frames != self.num_samples:
                        raise ValueError(
                            f"audio length mismatch at {mix_path}: "
                            f"got={mix_frames} expected={self.num_samples}"
                        )
                    if target_frames != self.num_samples:
                        raise ValueError(
                            f"audio length mismatch at {target_path}: "
                            f"got={target_frames} expected={self.num_samples}"
                        )

                total_len = min(mix_frames, target_frames)
                start_idx = self._select_crop_start(total_len)
                seg_len = self.num_samples if total_len > self.num_samples else total_len
                if seg_len <= 0:
                    raise ValueError(f"empty audio segment after metadata lookup: {mix_path}")

                mix = self._load_audio_segment_1d(
                    mix_path,
                    frame_offset=start_idx,
                    num_frames=seg_len,
                )
                target = self._load_audio_segment_1d(
                    target_path,
                    frame_offset=start_idx,
                    num_frames=seg_len,
                )
                n = min(mix.numel(), target.numel())
                mix = self._crop_or_pad_wave(mix[:n])
                target = self._crop_or_pad_wave(target[:n])
                return mix, target, start_idx, total_len

            mix_raw = self._load_audio_raw_1d(mix_path)
            target_raw = self._load_audio_raw_1d(target_path)
            if self.strict_audio_length and mix_raw.numel() != self.num_samples:
                raise ValueError(
                    f"audio length mismatch at {mix_path}: "
                    f"got={mix_raw.numel()} expected={self.num_samples}"
                )
            if self.strict_audio_length and target_raw.numel() != self.num_samples:
                raise ValueError(
                    f"audio length mismatch at {target_path}: "
                    f"got={target_raw.numel()} expected={self.num_samples}"
                )
            return self._crop_or_pad_pair(mix_raw, target_raw)

        if use_partial:
            if self.strict_audio_length and mix_frames != self.num_samples:
                raise ValueError(
                    f"audio length mismatch at {mix_path}: "
                    f"got={mix_frames} expected={self.num_samples}"
                )

            total_len = mix_frames
            start_idx = self._select_crop_start(total_len)
            seg_len = self.num_samples if total_len > self.num_samples else total_len
            if seg_len <= 0:
                raise ValueError(f"empty audio segment after metadata lookup: {mix_path}")
            mix = self._load_audio_segment_1d(
                mix_path,
                frame_offset=start_idx,
                num_frames=seg_len,
            )
            mix = self._crop_or_pad_wave(mix)
            return mix, None, start_idx, total_len

        mix_raw = self._load_audio_raw_1d(mix_path)
        if self.strict_audio_length and mix_raw.numel() != self.num_samples:
            raise ValueError(
                f"audio length mismatch at {mix_path}: "
                f"got={mix_raw.numel()} expected={self.num_samples}"
            )
        return self._crop_or_pad_pair(mix_raw, None)

    def _fit_enroll_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        current_dim = int(x.size(-1))
        if current_dim != int(self.enroll_dim):
            raise ValueError(
                "Enrollment embedding dim mismatch: "
                f"embedding_dim={current_dim} config_enroll_dim={self.enroll_dim}. "
                "Set data.enroll_dim/model.enroll_dim to the saved embedding dim "
                "to avoid information loss."
            )
        return x

    def _normalize_enroll(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_enroll:
            return x
        return F.normalize(x, p=2, dim=-1, eps=1e-8)

    def _load_enroll_embedding(self, row: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = row.get("enroll_embedding")
        if emb is not None:
            out = torch.tensor(emb, dtype=torch.float32).flatten()
            out = self._fit_enroll_last_dim(out)
            out = self._normalize_enroll(out)
            mask = torch.ones(1, dtype=torch.bool)
            return out, mask

        emb_path_raw = row.get("enroll_emb_path")
        if emb_path_raw is not None:
            emb_path = self._resolve_path(self.audio_base_dir, emb_path_raw)
            if emb_path is None or not emb_path.exists():
                raise FileNotFoundError(f"enroll_emb_path not found: {emb_path_raw}")

            cache_key = str(emb_path)
            cached = self._lru_get(self._enroll_cache, cache_key)
            if cached is not None:
                return cached

            if emb_path.suffix.lower() == ".npy":
                arr = np.load(emb_path)
                out = torch.from_numpy(arr).float()
                if out.dim() == 1:
                    out = self._fit_enroll_last_dim(out)
                    out = self._normalize_enroll(out)
                    mask = torch.ones(1, dtype=torch.bool)
                elif out.dim() == 2:
                    out = self._fit_enroll_last_dim(out)
                    out = self._normalize_enroll(out)
                    mask = torch.ones(out.size(0), dtype=torch.bool)
                else:
                    raise ValueError(
                        f"unsupported enroll tensor rank in npy: path={emb_path} shape={tuple(out.shape)}"
                    )
                result = (out, mask)
                self._lru_put(self._enroll_cache, cache_key, result, self.enroll_cache_size)
                return result

            if emb_path.suffix.lower() in {".pt", ".pth"}:
                t = torch.load(emb_path, map_location="cpu")
                if isinstance(t, dict):
                    if "embedding" in t:
                        t = t["embedding"]
                    else:
                        raise ValueError(f"no 'embedding' key in tensor dict: {emb_path}")
                if not torch.is_tensor(t):
                    t = torch.tensor(t)
                out = t.float()
                if out.dim() == 1:
                    out = self._fit_enroll_last_dim(out)
                    out = self._normalize_enroll(out)
                    mask = torch.ones(1, dtype=torch.bool)
                elif out.dim() == 2:
                    out = self._fit_enroll_last_dim(out)
                    out = self._normalize_enroll(out)
                    mask = torch.ones(out.size(0), dtype=torch.bool)
                else:
                    raise ValueError(
                        f"unsupported enroll tensor rank in pt/pth: path={emb_path} shape={tuple(out.shape)}"
                    )
                result = (out, mask)
                self._lru_put(self._enroll_cache, cache_key, result, self.enroll_cache_size)
                return result

            raise ValueError(f"unsupported enroll_emb_path extension: {emb_path}")

        if self.use_zero_enroll_fallback:
            if not self._warned_zero_enroll:
                print(
                    "[AuraDataset] enroll_emb_path/enroll_embedding not found. "
                    "Using zero enrollment vectors (temporary fallback)."
                )
                self._warned_zero_enroll = True
            return torch.zeros(self.enroll_dim, dtype=torch.float32), torch.ones(1, dtype=torch.bool)

        raise ValueError(
            "Missing enrollment embedding. Provide 'enroll_emb_path' or 'enroll_embedding' in manifest."
        )

    def _load_pitch(
        self,
        row: Dict,
        audio_total_samples: Optional[int] = None,
        crop_start_samples: int = 0,
        crop_num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        p_path_raw = row.get("pitch_gt_path")
        m_path_raw = row.get("pitch_voiced_mask_path")

        if p_path_raw is None or m_path_raw is None:
            # Use 10ms frame step convention for fallback sequence length.
            frame_hop_samples = max(1, int(round(0.01 * self.sample_rate)))
            default_frames = max(1, int(round(self.num_samples / frame_hop_samples)))
            return {
                "pitch_gt": torch.zeros(default_frames, dtype=torch.float32),
                "pitch_voiced_mask": torch.zeros(default_frames, dtype=torch.float32),
                "has_pitch_gt": torch.tensor(0.0, dtype=torch.float32),
            }

        p_path = self._resolve_path(self.pitch_base_dir, p_path_raw)
        m_path = self._resolve_path(self.pitch_base_dir, m_path_raw)
        if p_path is None or m_path is None or (not p_path.exists()) or (not m_path.exists()):
            raise FileNotFoundError(f"pitch file not found: {p_path_raw}, {m_path_raw}")

        p = torch.from_numpy(np.load(p_path)).float().flatten()
        m = torch.from_numpy(np.load(m_path)).float().flatten()

        n = min(p.numel(), m.numel())
        p = p[:n]
        m = m[:n]
        if n == 0:
            return {
                "pitch_gt": torch.zeros(1, dtype=torch.float32),
                "pitch_voiced_mask": torch.zeros(1, dtype=torch.float32),
                "has_pitch_gt": torch.tensor(0.0, dtype=torch.float32),
            }

        if (
            audio_total_samples is not None
            and crop_num_samples is not None
            and audio_total_samples > 0
        ):
            start_ratio = float(crop_start_samples) / float(audio_total_samples)
            end_ratio = float(crop_start_samples + crop_num_samples) / float(audio_total_samples)

            start_idx = int(round(start_ratio * n))
            end_idx = int(round(end_ratio * n))

            start_idx = max(0, min(start_idx, n - 1))
            end_idx = max(start_idx + 1, min(end_idx, n))

            p = p[start_idx:end_idx]
            m = m[start_idx:end_idx]

        return {
            "pitch_gt": p,
            "pitch_voiced_mask": m,
            "has_pitch_gt": torch.tensor(1.0, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]

        mix_path = self._resolve_path(self.audio_base_dir, row.get("mix_path"))
        enroll_path = self._resolve_path(self.audio_base_dir, row.get("enroll_path"))
        target_path = self._resolve_path(self.audio_base_dir, row.get("target_path"))

        if mix_path is None or enroll_path is None:
            raise ValueError(f"missing required path fields at idx={idx}")

        is_present = int(row.get("is_target_present", 1))
        if is_present == 1:
            if target_path is None:
                raise ValueError(f"present sample has null target_path at idx={idx}")
            mix, gt_target, crop_start, total_len = self._load_pair_with_optional_partial(
                mix_path, target_path
            )
        else:
            mix, _, crop_start, total_len = self._load_pair_with_optional_partial(
                mix_path, None
            )
            gt_target = torch.zeros_like(mix)

        # Residual is generated on-the-fly by design.
        gt_residual = mix - gt_target

        enroll, enroll_mask = self._load_enroll_embedding(row)

        if is_present == 1:
            pitch_dict = self._load_pitch(
                row,
                audio_total_samples=total_len,
                crop_start_samples=crop_start,
                crop_num_samples=self.num_samples,
            )
        else:
            pitch_dict = self._load_pitch(row)

        sample = {
            "id": row.get("id", f"sample_{idx}"),
            "speaker_id": row.get("speaker_id", "unknown"),
            "is_target_present": torch.tensor(float(is_present), dtype=torch.float32),
            "mix": mix,
            "enroll": enroll,
            "enroll_mask": enroll_mask,
            "gt_target": gt_target,
            "gt_residual": gt_residual,
            "enroll_path": str(enroll_path),
        }
        sample.update(pitch_dict)
        return sample


def _collate_with_pitch_padding(batch: List[Dict]) -> Dict:
    # Wave-level tensors are fixed length by dataset policy.
    mix = torch.stack([b["mix"] for b in batch], dim=0)
    enroll_list = [b["enroll"] for b in batch]
    enroll_mask_list = [b["enroll_mask"].bool() for b in batch]

    is_sequence = any(x.dim() == 2 for x in enroll_list)
    if not is_sequence:
        enroll = torch.stack([x.flatten() for x in enroll_list], dim=0)
        enroll_mask = torch.ones(enroll.size(0), 1, dtype=torch.bool)
    else:
        seqs = [x if x.dim() == 2 else x.unsqueeze(0) for x in enroll_list]
        max_t = max(int(x.size(0)) for x in seqs)
        feat_dim = int(seqs[0].size(-1))
        padded = []
        masks = []
        for seq, mask in zip(seqs, enroll_mask_list):
            valid_t = int(seq.size(0))
            pad_t = max_t - valid_t
            if pad_t > 0:
                seq = F.pad(seq, (0, 0, 0, pad_t))
            padded.append(seq)

            mask_seq = torch.zeros(max_t, dtype=torch.bool)
            mask_seq[: min(valid_t, mask.numel())] = True
            masks.append(mask_seq)

        enroll = torch.stack(padded, dim=0)
        if enroll.size(-1) != feat_dim:
            raise ValueError("inconsistent enrollment feature dim in batch")
        enroll_mask = torch.stack(masks, dim=0)

    gt_target = torch.stack([b["gt_target"] for b in batch], dim=0)
    gt_residual = torch.stack([b["gt_residual"] for b in batch], dim=0)
    is_present = torch.stack([b["is_target_present"] for b in batch], dim=0)
    has_pitch_gt = torch.stack([b["has_pitch_gt"] for b in batch], dim=0)

    max_pitch_len = max(int(b["pitch_gt"].numel()) for b in batch)

    pitch_gt = []
    pitch_mask = []
    for b in batch:
        p = b["pitch_gt"]
        m = b["pitch_voiced_mask"]
        pad = max_pitch_len - p.numel()
        if pad > 0:
            p = F.pad(p, (0, pad))
            m = F.pad(m, (0, pad))
        # If pitch GT is missing, enforce zero supervision through has_pitch_gt.
        m = m * b["has_pitch_gt"]
        pitch_gt.append(p)
        pitch_mask.append(m)

    out = {
        "id": [b["id"] for b in batch],
        "speaker_id": [b["speaker_id"] for b in batch],
        "mix": mix,
        "enroll": enroll,
        "enroll_mask": enroll_mask,
        "gt_target": gt_target,
        "gt_residual": gt_residual,
        "is_target_present": is_present,
        "pitch_gt": torch.stack(pitch_gt, dim=0),
        "pitch_voiced_mask": torch.stack(pitch_mask, dim=0),
        "has_pitch_gt": has_pitch_gt,
        "enroll_path": [b["enroll_path"] for b in batch],
    }
    return out


def get_dataloader(
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    manifest_path: str = "data/processed/train_manifest.jsonl",
    sample_rate: int = 16000,
    duration_sec: float = 6.0,
    enroll_dim: int = 128,
    audio_base_dir: Optional[str] = None,
    pitch_base_dir: Optional[str] = None,
    strict_audio_length: bool = False,
    use_zero_enroll_fallback: bool = True,
    random_crop: bool = True,
    partial_audio_load: bool = True,
    normalize_enroll: bool = False,
    enroll_cache_size: int = 4096,
    audio_info_cache_size: int = 16384,
    prefetch_factor: int = 2,
    persistent_workers: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
):
    dataset = AuraDataset(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        enroll_dim=enroll_dim,
        audio_base_dir=audio_base_dir,
        pitch_base_dir=pitch_base_dir,
        strict_audio_length=strict_audio_length,
        use_zero_enroll_fallback=use_zero_enroll_fallback,
        random_crop=random_crop,
        partial_audio_load=partial_audio_load,
        normalize_enroll=normalize_enroll,
        enroll_cache_size=enroll_cache_size,
        audio_info_cache_size=audio_info_cache_size,
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": _collate_with_pitch_padding,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))

    return DataLoader(**loader_kwargs)


def get_train_val_dataloaders(
    batch_size: int = 4,
    num_workers: int = 0,
    manifest_path: str = "data/processed/train_manifest.jsonl",
    sample_rate: int = 16000,
    duration_sec: float = 6.0,
    enroll_dim: int = 128,
    audio_base_dir: Optional[str] = None,
    pitch_base_dir: Optional[str] = None,
    strict_audio_length: bool = False,
    use_zero_enroll_fallback: bool = True,
    random_crop_train: bool = True,
    random_crop_val: bool = False,
    partial_audio_load: bool = True,
    normalize_enroll: bool = False,
    enroll_cache_size: int = 4096,
    audio_info_cache_size: int = 16384,
    prefetch_factor: int = 2,
    persistent_workers: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    val_ratio: float = 0.1,
    split_seed: int = 42,
    split_mode: str = "random",
    distributed_train: bool = False,
    distributed_val: bool = False,
    world_size: int = 1,
    rank: int = 0,
):
    base_dataset = AuraDataset(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        enroll_dim=enroll_dim,
        audio_base_dir=audio_base_dir,
        pitch_base_dir=pitch_base_dir,
        strict_audio_length=strict_audio_length,
        use_zero_enroll_fallback=use_zero_enroll_fallback,
        random_crop=False,
        partial_audio_load=partial_audio_load,
        normalize_enroll=normalize_enroll,
        enroll_cache_size=enroll_cache_size,
        audio_info_cache_size=audio_info_cache_size,
    )

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    total = len(base_dataset)
    val_size = max(1, int(total * val_ratio))
    train_size = total - val_size
    if train_size <= 0:
        raise ValueError(
            f"Not enough samples for train/val split: total={total}, val_ratio={val_ratio}"
        )

    split_mode = split_mode.lower()
    if split_mode == "random":
        generator = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(total, generator=generator).tolist()
        train_indices = perm[:train_size]
        val_indices = perm[train_size:]
    elif split_mode == "speaker":
        speaker_to_indices: Dict[str, List[int]] = {}
        for idx, row in enumerate(base_dataset.rows):
            speaker = str(row.get("speaker_id", "unknown"))
            speaker_to_indices.setdefault(speaker, []).append(idx)

        speakers = list(speaker_to_indices.keys())
        rng = random.Random(split_seed)
        rng.shuffle(speakers)

        target_val_count = val_size
        running = 0
        val_speakers: List[str] = []
        for speaker in speakers:
            val_speakers.append(speaker)
            running += len(speaker_to_indices[speaker])
            if running >= target_val_count:
                break

        if len(val_speakers) == len(speakers):
            val_speakers = val_speakers[:-1]

        val_set = set(val_speakers)
        val_indices: List[int] = []
        train_indices: List[int] = []
        for speaker, indices in speaker_to_indices.items():
            if speaker in val_set:
                val_indices.extend(indices)
            else:
                train_indices.extend(indices)

        if len(train_indices) == 0 or len(val_indices) == 0:
            raise ValueError(
                "speaker split failed: empty train or val split. "
                "Try a different val_ratio or split_seed."
            )

    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}. Use 'random' or 'speaker'.")

    rows = base_dataset.rows
    train_base = AuraDataset(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        enroll_dim=enroll_dim,
        audio_base_dir=audio_base_dir,
        pitch_base_dir=pitch_base_dir,
        strict_audio_length=strict_audio_length,
        use_zero_enroll_fallback=use_zero_enroll_fallback,
        random_crop=random_crop_train,
        partial_audio_load=partial_audio_load,
        normalize_enroll=normalize_enroll,
        enroll_cache_size=enroll_cache_size,
        audio_info_cache_size=audio_info_cache_size,
        rows=rows,
    )
    val_base = AuraDataset(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        enroll_dim=enroll_dim,
        audio_base_dir=audio_base_dir,
        pitch_base_dir=pitch_base_dir,
        strict_audio_length=strict_audio_length,
        use_zero_enroll_fallback=use_zero_enroll_fallback,
        random_crop=random_crop_val,
        partial_audio_load=partial_audio_load,
        normalize_enroll=normalize_enroll,
        enroll_cache_size=enroll_cache_size,
        audio_info_cache_size=audio_info_cache_size,
        rows=rows,
    )
    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)

    train_sampler = None
    if distributed_train and world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    val_sampler = None
    if distributed_val and world_size > 1:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": _collate_with_pitch_padding,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = bool(persistent_workers)
        common_loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))

    train_loader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        **common_loader_kwargs,
    )
    return train_loader, val_loader
