from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from data_pipeline.common import iter_jsonl


@dataclass(frozen=True)
class ManifestRow:
    id: str
    split: str
    task_bucket: str
    target_speaker_id: str
    mix_path: str
    target_path: str
    scenario: str
    noise_types: list[str]
    sample_rate: int
    num_samples: int


@dataclass(frozen=True)
class SpeakerRegistryEntry:
    speaker_id: str
    split: str
    enroll_wav_paths: list[str]
    enroll_emb_paths: list[str]
    num_enroll_utts_raw: int
    num_enroll_utts: int
    embedding_model: str
    embedding_dim: int
    cache_version: str
    low_enrollment: bool
    missing_embedding_count: int
    eval_fixed_indices: list[int]


@dataclass(frozen=True)
class DatasetValidationReport:
    num_manifest_rows: int
    num_valid_rows: int
    num_missing_registry_rows: int
    num_missing_embedding_rows: int
    missing_registry_speakers: list[str]
    missing_embedding_speakers: list[str]


def load_speaker_registry(path: str | Path) -> dict[str, SpeakerRegistryEntry]:
    path = Path(path)
    registry: dict[str, SpeakerRegistryEntry] = {}
    for _, line in iter_jsonl(path):
        if not line.strip():
            continue
        payload = json.loads(line)
        speaker_id = str(payload["speaker_id"])
        registry[speaker_id] = SpeakerRegistryEntry(
            speaker_id=speaker_id,
            split=str(payload["split"]),
            enroll_wav_paths=[str(item) for item in payload.get("enroll_wav_paths", [])],
            enroll_emb_paths=[str(item) for item in payload.get("enroll_emb_paths", [])],
            num_enroll_utts_raw=int(payload.get("num_enroll_utts_raw", payload.get("num_enroll_utts", 0))),
            num_enroll_utts=int(payload.get("num_enroll_utts", 0)),
            embedding_model=str(payload.get("embedding_model", "unknown")),
            embedding_dim=int(payload.get("embedding_dim", 0)),
            cache_version=str(payload.get("cache_version", "unknown")),
            low_enrollment=bool(payload.get("low_enrollment", False)),
            missing_embedding_count=int(payload.get("missing_embedding_count", 0)),
            eval_fixed_indices=[int(idx) for idx in payload.get("eval_fixed_indices", [])],
        )
    return registry


class AuraManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        speaker_registry_path: str | Path,
        *,
        split: str,
        enroll_k: int = 8,
        sample_rate: int = 16000,
        duration_sec: float | None = 6.0,
        random_crop: bool | None = None,
        deterministic_eval_enroll: bool = True,
        strict_registry: bool = True,
        registry_embedding_model: str | None = None,
        seed: int = 20260318,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.speaker_registry_path = Path(speaker_registry_path)
        self.split = str(split)
        self.enroll_k = int(enroll_k)
        self.sample_rate = int(sample_rate)
        self.duration_sec = duration_sec
        self.random_crop = bool(random_crop) if random_crop is not None else self.split == "train"
        self.deterministic_eval_enroll = bool(deterministic_eval_enroll)
        self.strict_registry = bool(strict_registry)
        self.registry_embedding_model = registry_embedding_model
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

        self.registry = self._load_registry(self.speaker_registry_path)
        self.rows = self._load_manifest(self.manifest_path, split=self.split)
        self.validation_report = self._validate_rows()
        if self.strict_registry and (
            self.validation_report.num_missing_registry_rows > 0
            or self.validation_report.num_missing_embedding_rows > 0
        ):
            raise ValueError(
                "manifest/registry validation failed: "
                f"missing_registry_rows={self.validation_report.num_missing_registry_rows}, "
                f"missing_embedding_rows={self.validation_report.num_missing_embedding_rows}"
            )
        if not self.strict_registry:
            self.rows = [row for row in self.rows if self._row_is_usable(row)]

    @staticmethod
    def _load_manifest(path: Path, *, split: str) -> list[ManifestRow]:
        rows: list[ManifestRow] = []
        for _, line in iter_jsonl(path):
            if not line.strip():
                continue
            payload = json.loads(line)
            if str(payload["split"]) != split:
                continue
            rows.append(
                ManifestRow(
                    id=str(payload["id"]),
                    split=str(payload["split"]),
                    task_bucket=str(payload["task_bucket"]),
                    target_speaker_id=str(payload["target_speaker_id"]),
                    mix_path=str(payload["mix_path"]),
                    target_path=str(payload["target_path"]),
                    scenario=str(payload["scenario"]),
                    noise_types=[str(item) for item in payload.get("noise_types", [])],
                    sample_rate=int(payload["sample_rate"]),
                    num_samples=int(payload["num_samples"]),
                )
            )
        return rows

    @staticmethod
    def _load_registry(path: Path) -> dict[str, SpeakerRegistryEntry]:
        return load_speaker_registry(path)

    def _validate_rows(self) -> DatasetValidationReport:
        missing_registry_speakers = set()
        missing_embedding_speakers = set()
        valid_rows = 0
        for row in self.rows:
            entry = self.registry.get(row.target_speaker_id)
            if entry is None:
                missing_registry_speakers.add(row.target_speaker_id)
                continue
            if self.registry_embedding_model is not None and entry.embedding_model != self.registry_embedding_model:
                raise ValueError(
                    f"registry embedding model mismatch for {row.target_speaker_id}: "
                    f"expected {self.registry_embedding_model}, got {entry.embedding_model}"
                )
            if entry.num_enroll_utts <= 0:
                missing_embedding_speakers.add(row.target_speaker_id)
                continue
            valid_rows += 1
        return DatasetValidationReport(
            num_manifest_rows=len(self.rows),
            num_valid_rows=valid_rows,
            num_missing_registry_rows=len(missing_registry_speakers),
            num_missing_embedding_rows=len(missing_embedding_speakers),
            missing_registry_speakers=sorted(missing_registry_speakers),
            missing_embedding_speakers=sorted(missing_embedding_speakers),
        )

    def _row_is_usable(self, row: ManifestRow) -> bool:
        entry = self.registry.get(row.target_speaker_id)
        return entry is not None and entry.num_enroll_utts > 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self._rng = random.Random(self.seed + int(epoch))

    def _load_waveform(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if wav.dim() != 2:
            raise ValueError(f"invalid waveform shape at {path}: {tuple(wav.shape)}")
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.squeeze(0).to(torch.float32)

    def _resolve_target_num_samples(self, row: ManifestRow) -> int:
        if self.duration_sec is not None:
            return int(round(float(self.duration_sec) * self.sample_rate))
        return int(row.num_samples)

    def _fit_pair(self, mix: torch.Tensor, target: torch.Tensor, row: ManifestRow) -> tuple[torch.Tensor, torch.Tensor]:
        target_len = self._resolve_target_num_samples(row)
        common_len = min(mix.numel(), target.numel())
        if common_len >= target_len:
            max_start = common_len - target_len
            if self.random_crop and max_start > 0:
                start = self._rng.randint(0, max_start)
            else:
                start = max_start // 2
            mix = mix[start:start + target_len]
            target = target[start:start + target_len]
            return mix, target

        mix = mix[:common_len]
        target = target[:common_len]
        pad_amount = target_len - common_len
        if pad_amount > 0:
            mix = torch.nn.functional.pad(mix, (0, pad_amount))
            target = torch.nn.functional.pad(target, (0, pad_amount))
        return mix, target

    def _load_embedding(self, path: str) -> torch.Tensor:
        emb = np.load(path)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        return torch.from_numpy(emb)

    def _select_train_indices(self, num_embeddings: int) -> list[int]:
        if num_embeddings >= self.enroll_k:
            return self._rng.sample(list(range(num_embeddings)), self.enroll_k)
        return list(range(num_embeddings))

    def _select_eval_indices(self, entry: SpeakerRegistryEntry) -> list[int]:
        indices = [idx for idx in entry.eval_fixed_indices if 0 <= idx < entry.num_enroll_utts]
        if indices:
            return indices[: min(self.enroll_k, len(indices))]
        return list(range(min(self.enroll_k, entry.num_enroll_utts)))

    def _load_enrollment_sequence(self, speaker_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.registry.get(speaker_id)
        if entry is None:
            raise KeyError(f"speaker not found in registry: {speaker_id}")
        if entry.num_enroll_utts <= 0:
            raise ValueError(f"speaker has no enrollment embeddings: {speaker_id}")
        if self.split == "train":
            selected_indices = self._select_train_indices(entry.num_enroll_utts)
        elif self.deterministic_eval_enroll:
            selected_indices = self._select_eval_indices(entry)
        else:
            selected_indices = list(range(min(self.enroll_k, entry.num_enroll_utts)))

        embeddings = [self._load_embedding(entry.enroll_emb_paths[idx]) for idx in selected_indices]
        if not embeddings:
            raise ValueError(f"speaker yielded empty enrollment sequence: {speaker_id}")

        emb_dim = int(embeddings[0].numel())
        enroll_seq = torch.zeros(self.enroll_k, emb_dim, dtype=torch.float32)
        enroll_mask = torch.zeros(self.enroll_k, dtype=torch.bool)
        for pos, emb in enumerate(embeddings[: self.enroll_k]):
            enroll_seq[pos] = emb
            enroll_mask[pos] = True
        return enroll_seq, enroll_mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        mix = self._load_waveform(row.mix_path)
        target = self._load_waveform(row.target_path)
        mix, target = self._fit_pair(mix, target, row)
        enroll_seq, enroll_mask = self._load_enrollment_sequence(row.target_speaker_id)
        return {
            "sample_id": row.id,
            "speaker_id": row.target_speaker_id,
            "task_bucket": row.task_bucket,
            "scenario": row.scenario,
            "noise_types": list(row.noise_types),
            "mix": mix,
            "target": target,
            "enroll_seq": enroll_seq,
            "enroll_mask": enroll_mask,
        }


def aura_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty batch")
    mix = torch.stack([item["mix"] for item in batch], dim=0)
    target = torch.stack([item["target"] for item in batch], dim=0)
    enroll_seq = torch.stack([item["enroll_seq"] for item in batch], dim=0)
    enroll_mask = torch.stack([item["enroll_mask"] for item in batch], dim=0)
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "speaker_id": [item["speaker_id"] for item in batch],
        "task_bucket": [item["task_bucket"] for item in batch],
        "scenario": [item["scenario"] for item in batch],
        "noise_types": [item["noise_types"] for item in batch],
        "mix": mix,
        "target": target,
        "enroll_seq": enroll_seq,
        "enroll_mask": enroll_mask,
    }
