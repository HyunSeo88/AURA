from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from data_pipeline.common import dump_json, iter_jsonl


@dataclass
class SpeakerStatsRow:
    speaker_id: str
    n_total: int
    n_1spk: int
    n_2spk: int
    n_3spk: int
    num_enroll_utts_raw: int
    split_eligible_samples: int
    ineligible_samples: int
    eligible_for_split: bool
    low_enrollment: bool

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpeakerStatsReport:
    canonical_path: str
    enrollment_root: str
    enroll_k: int
    num_speakers_total: int
    num_speakers_eligible: int
    num_speakers_low_enrollment: int
    num_speakers_missing_enrollment: int
    task_bucket_counts: Dict[str, int]

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def _count_enrollment_utts(enrollment_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not enrollment_root.exists():
        return counts
    for child in sorted(enrollment_root.iterdir()):
        if not child.is_dir():
            continue
        counts[child.name] = sum(1 for _ in child.glob("*.wav"))
    return counts


def build_speaker_stats(
    canonical_path: Path,
    enrollment_root: Path,
    output_dir: Path,
    *,
    enroll_k: int,
) -> Dict[str, str]:
    per_speaker = defaultdict(lambda: {
        "n_total": 0,
        "n_1spk": 0,
        "n_2spk": 0,
        "n_3spk": 0,
        "split_eligible_samples": 0,
        "ineligible_samples": 0,
    })
    bucket_counts = Counter()

    for _, line in iter_jsonl(canonical_path):
        if not line.strip():
            continue
        row = json.loads(line)
        speaker_id = str(row["target_speaker_id"])
        bucket = str(row["task_bucket"])
        entry = per_speaker[speaker_id]
        entry["n_total"] += 1
        if bucket in {"1spk", "2spk", "3spk"}:
            entry[f"n_{bucket}"] += 1
            bucket_counts[bucket] += 1
        if bool(row.get("split_eligible", True)):
            entry["split_eligible_samples"] += 1
        else:
            entry["ineligible_samples"] += 1

    enroll_counts = _count_enrollment_utts(enrollment_root)
    rows: list[SpeakerStatsRow] = []
    for speaker_id in sorted(per_speaker.keys()):
        entry = per_speaker[speaker_id]
        num_enroll_utts = int(enroll_counts.get(speaker_id, 0))
        eligible = entry["split_eligible_samples"] > 0 and num_enroll_utts > 0
        low_enrollment = 0 < num_enroll_utts < int(enroll_k)
        rows.append(
            SpeakerStatsRow(
                speaker_id=speaker_id,
                n_total=int(entry["n_total"]),
                n_1spk=int(entry["n_1spk"]),
                n_2spk=int(entry["n_2spk"]),
                n_3spk=int(entry["n_3spk"]),
                num_enroll_utts_raw=num_enroll_utts,
                split_eligible_samples=int(entry["split_eligible_samples"]),
                ineligible_samples=int(entry["ineligible_samples"]),
                eligible_for_split=eligible,
                low_enrollment=low_enrollment,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "speaker_stats.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")

    report = SpeakerStatsReport(
        canonical_path=str(canonical_path.resolve()),
        enrollment_root=str(enrollment_root.resolve()),
        enroll_k=int(enroll_k),
        num_speakers_total=len(rows),
        num_speakers_eligible=sum(1 for row in rows if row.eligible_for_split),
        num_speakers_low_enrollment=sum(1 for row in rows if row.low_enrollment),
        num_speakers_missing_enrollment=sum(1 for row in rows if row.num_enroll_utts_raw == 0),
        task_bucket_counts=dict(bucket_counts),
    )
    dump_json(output_dir / "speaker_stats_report.json", report.to_json())
    return {
        "speaker_stats_path": str(rows_path.resolve()),
        "speaker_stats_report": str((output_dir / "speaker_stats_report.json").resolve()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build speaker statistics for AURA data pipeline v1.")
    parser.add_argument("--canonical-path", type=str, required=True)
    parser.add_argument("--enrollment-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--enroll-k", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = build_speaker_stats(
        canonical_path=Path(args.canonical_path),
        enrollment_root=Path(args.enrollment_root),
        output_dir=Path(args.output_dir),
        enroll_k=int(args.enroll_k),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
