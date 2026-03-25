from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Iterator

from torch.utils.data import Sampler


DEFAULT_BUCKET_WEIGHTS = {
    "1spk": 0.60,
    "2spk": 0.25,
    "3spk": 0.15,
}


@dataclass(frozen=True)
class SamplerRecord:
    index: int
    speaker_id: str
    task_bucket: str


@dataclass(frozen=True)
class SamplerPlan:
    epoch: int
    total_size: int
    num_samples_per_rank: int
    normalized_bucket_weights: dict[str, float]
    available_bucket_counts: dict[str, int]


class BucketAwareSpeakerSampler(Sampler[int]):
    def __init__(
        self,
        records: Iterable[SamplerRecord],
        *,
        bucket_weights: dict[str, float] | None = None,
        epoch_size: int | None = None,
        seed: int = 20260318,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("sampler received no records")
        if world_size < 1:
            raise ValueError(f"invalid world_size: {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"invalid rank/world_size combination: rank={rank}, world_size={world_size}")

        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.bucket_weights = dict(bucket_weights or DEFAULT_BUCKET_WEIGHTS)
        self._bucket_to_speakers = self._build_bucket_index(self.records)
        self._available_bucket_counts = {
            bucket: sum(len(indices) for indices in speakers.values())
            for bucket, speakers in self._bucket_to_speakers.items()
            if speakers
        }
        self._normalized_bucket_weights = self._normalize_bucket_weights(self.bucket_weights, self._available_bucket_counts)
        base_epoch_size = int(epoch_size) if epoch_size is not None else len(self.records)
        self.num_samples = int(math.ceil(base_epoch_size / self.world_size))
        self.total_size = self.num_samples * self.world_size

    @staticmethod
    def _build_bucket_index(records: list[SamplerRecord]) -> dict[str, dict[str, list[int]]]:
        bucket_to_speakers: dict[str, dict[str, list[int]]] = {}
        for record in records:
            speakers = bucket_to_speakers.setdefault(record.task_bucket, {})
            speakers.setdefault(record.speaker_id, []).append(record.index)
        return bucket_to_speakers

    @staticmethod
    def _normalize_bucket_weights(
        requested_weights: dict[str, float],
        available_bucket_counts: dict[str, int],
    ) -> dict[str, float]:
        available_weights = {
            bucket: float(requested_weights.get(bucket, 0.0))
            for bucket, count in available_bucket_counts.items()
            if count > 0 and float(requested_weights.get(bucket, 0.0)) > 0.0
        }
        if available_weights:
            total = sum(available_weights.values())
            return {bucket: weight / total for bucket, weight in available_weights.items()}

        fallback = {bucket: 1.0 for bucket, count in available_bucket_counts.items() if count > 0}
        if not fallback:
            raise ValueError("no available buckets for sampling")
        total = sum(fallback.values())
        return {bucket: weight / total for bucket, weight in fallback.items()}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self) -> random.Random:
        return random.Random(self.seed + self.epoch)

    def _sample_bucket(self, rng: random.Random) -> str:
        draw = rng.random()
        cumulative = 0.0
        items = sorted(self._normalized_bucket_weights.items())
        for bucket, weight in items:
            cumulative += weight
            if draw <= cumulative:
                return bucket
        return items[-1][0]

    def _generate_global_indices(self) -> list[int]:
        rng = self._rng()
        global_indices: list[int] = []
        for _ in range(self.total_size):
            bucket = self._sample_bucket(rng)
            speaker_pool = self._bucket_to_speakers[bucket]
            speaker_id = rng.choice(sorted(speaker_pool.keys()))
            sample_index = rng.choice(speaker_pool[speaker_id])
            global_indices.append(int(sample_index))
        return global_indices

    def get_plan(self) -> SamplerPlan:
        return SamplerPlan(
            epoch=self.epoch,
            total_size=self.total_size,
            num_samples_per_rank=self.num_samples,
            normalized_bucket_weights=dict(self._normalized_bucket_weights),
            available_bucket_counts=dict(self._available_bucket_counts),
        )

    def __iter__(self) -> Iterator[int]:
        global_indices = self._generate_global_indices()
        rank_indices = global_indices[self.rank:self.total_size:self.world_size]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError(
                f"sampler shard length mismatch: expected {self.num_samples}, got {len(rank_indices)}"
            )
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples
