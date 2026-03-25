from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from data_pipeline.dataset import load_speaker_registry


RegistryInput = Mapping[str, Any] | str | Path


def _stable_hash_int(*parts: object) -> int:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _entry_split(entry: Any, default: str) -> str:
    if isinstance(entry, Mapping):
        return str(entry.get("split", default))
    return str(getattr(entry, "split", default))


def _entry_num_enroll_utts(entry: Any) -> int:
    if isinstance(entry, Mapping):
        return int(entry.get("num_enroll_utts", 0))
    return int(getattr(entry, "num_enroll_utts", 0))


def _entry_enroll_emb_paths(entry: Any) -> list[str]:
    if isinstance(entry, Mapping):
        return [str(path) for path in entry.get("enroll_emb_paths", [])]
    return [str(path) for path in getattr(entry, "enroll_emb_paths", [])]


def _entry_eval_fixed_indices(entry: Any) -> list[int]:
    if isinstance(entry, Mapping):
        return [int(idx) for idx in entry.get("eval_fixed_indices", [])]
    return [int(idx) for idx in getattr(entry, "eval_fixed_indices", [])]


def load_absent_registry(registry: RegistryInput) -> dict[str, Any]:
    if isinstance(registry, (str, Path)):
        return load_speaker_registry(registry)
    return {str(speaker_id): entry for speaker_id, entry in registry.items()}


def merge_absent_registries(*named_registries: tuple[str, RegistryInput | None]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    ownership: dict[str, str] = {}
    for name, registry in named_registries:
        if registry is None:
            continue
        resolved = load_absent_registry(registry)
        for speaker_id, entry in resolved.items():
            prev = ownership.get(speaker_id)
            if prev is not None:
                raise ValueError(
                    f"duplicate speaker_id across absent registries: speaker_id={speaker_id}, first={prev}, second={name}"
                )
            merged[speaker_id] = entry
            ownership[speaker_id] = name
    return merged


def resolve_absent_donor_registry(
    *,
    source: str,
    internal_registry: RegistryInput,
    external_registry: RegistryInput | None = None,
) -> dict[str, Any]:
    source_key = str(source).lower()
    if source_key == "internal":
        return load_absent_registry(internal_registry)
    if source_key == "external":
        if external_registry is None:
            raise ValueError("absent source=external requires data.external_absent.speaker_registry")
        return load_absent_registry(external_registry)
    if source_key == "hybrid":
        if external_registry is None:
            raise ValueError("absent source=hybrid requires data.external_absent.speaker_registry")
        return merge_absent_registries(
            ("internal", internal_registry),
            ("external", external_registry),
        )
    raise ValueError(f"unsupported absent source: {source}")


def build_absent_manager_from_source(
    *,
    source: str,
    split: str,
    internal_registry: RegistryInput,
    external_registry: RegistryInput | None,
    enroll_k: int,
    seed: int = 20260318,
    cache_embeddings: bool = True,
) -> "AuraPAOnlineAbsentManager":
    donor_registry = resolve_absent_donor_registry(
        source=source,
        internal_registry=internal_registry,
        external_registry=external_registry,
    )
    return AuraPAOnlineAbsentManager(
        donor_registry,
        split=split,
        enroll_k=enroll_k,
        seed=seed,
        cache_embeddings=cache_embeddings,
    )


class AuraPAOnlineAbsentManager:
    def __init__(
        self,
        registry: Mapping[str, Any],
        *,
        split: str,
        enroll_k: int,
        seed: int = 20260318,
        cache_embeddings: bool = True,
    ) -> None:
        self.split = str(split)
        self.enroll_k = int(enroll_k)
        self.seed = int(seed)
        self.cache_embeddings = bool(cache_embeddings)
        self.epoch = 0
        self._embedding_cache: dict[str, torch.Tensor] = {}

        self.registry = {
            speaker_id: entry
            for speaker_id, entry in load_absent_registry(registry).items()
            if _entry_split(entry, self.split) == self.split and _entry_num_enroll_utts(entry) > 0
        }
        self.speaker_ids = sorted(self.registry.keys())
        if len(self.speaker_ids) < 2:
            raise ValueError(
                f"AuraPAOnlineAbsentManager for split={self.split} requires at least two speakers with enrollment, "
                f"got {len(self.speaker_ids)}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_embedding(self, path: str) -> torch.Tensor:
        cached = self._embedding_cache.get(path)
        if cached is not None:
            return cached
        arr = np.load(path)
        emb = torch.from_numpy(np.asarray(arr, dtype=np.float32).reshape(-1))
        if self.cache_embeddings:
            self._embedding_cache[path] = emb
        return emb

    def _build_sequence_from_indices(self, speaker_id: str, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.registry[speaker_id]
        if not indices:
            raise ValueError(f"empty enrollment index set for speaker={speaker_id}")
        emb_paths = _entry_enroll_emb_paths(entry)
        embeddings = [self._load_embedding(emb_paths[idx]) for idx in indices[: self.enroll_k]]
        emb_dim = int(embeddings[0].numel())
        enroll_seq = torch.zeros(self.enroll_k, emb_dim, dtype=torch.float32)
        enroll_mask = torch.zeros(self.enroll_k, dtype=torch.bool)
        for pos, emb in enumerate(embeddings):
            enroll_seq[pos] = emb
            enroll_mask[pos] = True
        return enroll_seq, enroll_mask

    def _random_indices(self, speaker_id: str, rng: random.Random) -> list[int]:
        entry = self.registry[speaker_id]
        num_utts = _entry_num_enroll_utts(entry)
        if num_utts >= self.enroll_k:
            return rng.sample(list(range(num_utts)), self.enroll_k)
        return list(range(num_utts))

    def _deterministic_indices(self, speaker_id: str, *, sample_key: str) -> list[int]:
        entry = self.registry[speaker_id]
        num_utts = _entry_num_enroll_utts(entry)
        fixed = [int(idx) for idx in _entry_eval_fixed_indices(entry) if 0 <= int(idx) < num_utts]
        chosen: list[int] = []
        for idx in fixed:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= min(self.enroll_k, num_utts):
                return chosen

        remaining = [idx for idx in range(num_utts) if idx not in chosen]
        if not remaining:
            return chosen
        ordered = sorted(
            remaining,
            key=lambda idx: (_stable_hash_int(self.seed, self.split, speaker_id, sample_key, idx), idx),
        )
        for idx in ordered:
            chosen.append(idx)
            if len(chosen) >= min(self.enroll_k, num_utts):
                break
        return chosen

    def _random_donor_speaker(self, current_speaker: str, rng: random.Random) -> str:
        candidates = [speaker_id for speaker_id in self.speaker_ids if speaker_id != current_speaker]
        if not candidates:
            raise ValueError(f"no donor speaker available for split={self.split} and current speaker={current_speaker}")
        return rng.choice(candidates)

    def _deterministic_donor_speaker(self, current_speaker: str, *, sample_key: str) -> str:
        candidates = [speaker_id for speaker_id in self.speaker_ids if speaker_id != current_speaker]
        if not candidates:
            raise ValueError(f"no donor speaker available for split={self.split} and current speaker={current_speaker}")
        ordered = sorted(
            candidates,
            key=lambda speaker_id: (_stable_hash_int(self.seed, self.split, sample_key, current_speaker, speaker_id), speaker_id),
        )
        return ordered[0]

    @staticmethod
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

    def make_train_batch(
        self,
        batch: Mapping[str, Any],
        *,
        absent_ratio: float,
        step: int,
        rank: int = 0,
        ensure_min_present: bool = False,
        present_ratio_min: float = 0.0,
        absent_ratio_max: float = 1.0,
        source_type: str = "online_swap",
    ) -> dict[str, Any]:
        output = self._clone_batch(batch)
        batch_size = len(output["speaker_id"])
        rng = random.Random(self.seed + (self.epoch * 1_000_003) + (int(step) * 9_973) + (int(rank) * 31))

        target_presence = torch.ones(batch_size, dtype=torch.float32)
        enroll_speaker_id = list(output["speaker_id"])
        presence_source_type = ["present" for _ in range(batch_size)]
        absent_indices = [idx for idx in range(batch_size) if rng.random() < float(absent_ratio)]

        min_present_count = max(0, int(math.ceil(float(present_ratio_min) * float(batch_size))))
        max_absent_count = int(math.floor(float(absent_ratio_max) * float(batch_size)))
        max_absent_count = min(max_absent_count, max(0, batch_size - min_present_count))
        if ensure_min_present and batch_size > 1:
            max_absent_count = min(max_absent_count, batch_size - 1)
        max_absent_count = max(0, max_absent_count)
        if len(absent_indices) > max_absent_count:
            rng.shuffle(absent_indices)
            absent_indices = absent_indices[:max_absent_count]

        for idx in absent_indices:
            current_speaker = str(output["speaker_id"][idx])
            donor_speaker = self._random_donor_speaker(current_speaker, rng)
            donor_indices = self._random_indices(donor_speaker, rng)
            enroll_seq, enroll_mask = self._build_sequence_from_indices(donor_speaker, donor_indices)
            output["enroll_seq"][idx] = enroll_seq
            output["enroll_mask"][idx] = enroll_mask
            target_presence[idx] = 0.0
            enroll_speaker_id[idx] = donor_speaker
            presence_source_type[idx] = source_type

        output["target_presence"] = target_presence
        output["enroll_speaker_id"] = enroll_speaker_id
        output["presence_source_type"] = presence_source_type
        return output

    def make_eval_absent_batch(
        self,
        batch: Mapping[str, Any],
        *,
        absent_ratio: float = 1.0,
        source_type: str = "deterministic_swap",
    ) -> dict[str, Any]:
        output = self._clone_batch(batch)
        batch_size = len(output["speaker_id"])
        target_presence = torch.ones(batch_size, dtype=torch.float32)
        enroll_speaker_id = list(output["speaker_id"])
        presence_source_type = ["present" for _ in range(batch_size)]

        for idx in range(batch_size):
            sample_id = str(output["sample_id"][idx])
            current_speaker = str(output["speaker_id"][idx])
            draw = (_stable_hash_int(self.seed, self.split, sample_id, "select") % 1_000_000) / 1_000_000.0
            if draw >= float(absent_ratio):
                continue
            donor_speaker = self._deterministic_donor_speaker(current_speaker, sample_key=sample_id)
            donor_indices = self._deterministic_indices(donor_speaker, sample_key=sample_id)
            enroll_seq, enroll_mask = self._build_sequence_from_indices(donor_speaker, donor_indices)
            output["enroll_seq"][idx] = enroll_seq
            output["enroll_mask"][idx] = enroll_mask
            target_presence[idx] = 0.0
            enroll_speaker_id[idx] = donor_speaker
            presence_source_type[idx] = source_type

        output["target_presence"] = target_presence
        output["enroll_speaker_id"] = enroll_speaker_id
        output["presence_source_type"] = presence_source_type
        return output
