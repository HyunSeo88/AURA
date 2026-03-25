from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from data_pipeline.common import CanonicalSample


@dataclass
class SourceSpec:
    name: str
    adaptor: str
    root: Path
    enrollment_root: Optional[Path] = None
    meta_glob: str = "meta.jsonl"


@dataclass
class SourceStats:
    source_name: str
    adaptor_name: str
    root: str
    meta_files_seen: int = 0
    rows_seen: int = 0
    rows_emitted: int = 0
    malformed_meta_lines: int = 0
    dropped_rows: int = 0
    split_eligible_rows: int = 0
    split_ineligible_rows: int = 0
    duplicate_sample_ids: int = 0
    task_bucket_counts: Counter = field(default_factory=Counter)
    split_ineligible_reasons: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, object]:
        return {
            "source_name": self.source_name,
            "adaptor_name": self.adaptor_name,
            "root": self.root,
            "meta_files_seen": self.meta_files_seen,
            "rows_seen": self.rows_seen,
            "rows_emitted": self.rows_emitted,
            "malformed_meta_lines": self.malformed_meta_lines,
            "dropped_rows": self.dropped_rows,
            "split_eligible_rows": self.split_eligible_rows,
            "split_ineligible_rows": self.split_ineligible_rows,
            "duplicate_sample_ids": self.duplicate_sample_ids,
            "task_bucket_counts": dict(self.task_bucket_counts),
            "split_ineligible_reasons": dict(self.split_ineligible_reasons),
            "errors": list(self.errors),
        }


class BaseSourceAdaptor(ABC):
    adaptor_name = "base"

    def __init__(
        self,
        spec: SourceSpec,
        *,
        expected_sample_rate: int,
        expected_num_samples: int,
        audio_check_mode: str = "exists",
    ):
        self.spec = spec
        self.expected_sample_rate = int(expected_sample_rate)
        self.expected_num_samples = int(expected_num_samples)
        self.audio_check_mode = str(audio_check_mode)
        self.stats = SourceStats(
            source_name=spec.name,
            adaptor_name=self.adaptor_name,
            root=str(spec.root),
        )

    @abstractmethod
    def iter_samples(self) -> Iterator[CanonicalSample]:
        raise NotImplementedError

    def meta_files(self) -> Iterable[Path]:
        return sorted(self.spec.root.glob(self.spec.meta_glob))

    def _path_exists(self, path: Path) -> bool:
        return path.exists() if self.audio_check_mode in {"exists", "header"} else True
