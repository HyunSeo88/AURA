from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


@dataclass
class CanonicalSample:
    id: str
    source_dataset: str
    task_bucket: str
    num_interferers: int
    num_speakers_total: int
    target_speaker_id: str
    mix_path: str
    target_path: str
    scenario: str
    noise_types: list[str]
    sample_rate: int
    num_samples: int
    target_type: str
    mix_type: str
    split_eligible: bool
    split_ineligible_reason: Optional[str] = None
    room_dim: Optional[list[float]] = None
    mic_loc: Optional[list[float]] = None
    target_loc: Optional[list[float]] = None
    raw_meta_path: Optional[str] = None
    raw_meta_line: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class CanonicalizationReport:
    version: str
    generated_by: str
    sources: list[Dict[str, Any]]
    totals: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_all(self, rows: Iterable[Dict[str, Any]]) -> int:
        count = 0
        with self.path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        return count


class CounterJsonEncoder(json.JSONEncoder):
    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, Counter):
            return dict(obj)
        return super().default(obj)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, cls=CounterJsonEncoder)
        handle.write("\n")


def iter_jsonl(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            yield line_no, line.rstrip("\n")


def normalize_noise_types(raw: Any) -> list[str]:
    if raw is None:
        return []
    values: list[str] = []
    if isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    for item in items:
        if isinstance(item, dict):
            value = item.get("type") or item.get("name")
        else:
            value = item
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            values.append(value_str)
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def safe_relaxed_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
