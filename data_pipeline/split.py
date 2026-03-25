from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

from data_pipeline.common import dump_json, iter_jsonl


BUCKET_KEYS = ("n_1spk", "n_2spk", "n_3spk")
SPLIT_KEYS = ("train", "val", "test")


@dataclass
class SpeakerRow:
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


@dataclass
class SplitSummary:
    split: str
    num_speakers: int
    speaker_quota: int
    n_total: int
    n_1spk: int
    n_2spk: int
    n_3spk: int
    low_enrollment_speakers: int

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def _load_rows(path: Path) -> list[SpeakerRow]:
    rows: list[SpeakerRow] = []
    for _, line in iter_jsonl(path):
        if not line.strip():
            continue
        rows.append(SpeakerRow(**json.loads(line)))
    return rows


def _speaker_quotas(num_speakers: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {split: num_speakers * ratios[split] for split in SPLIT_KEYS}
    quotas = {split: int(math.floor(raw[split])) for split in SPLIT_KEYS}
    remainder = num_speakers - sum(quotas.values())
    order = sorted(SPLIT_KEYS, key=lambda split: (raw[split] - quotas[split]), reverse=True)
    for split in order[:remainder]:
        quotas[split] += 1
    return quotas


def _target_totals(rows: list[SpeakerRow], ratios: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    total_all = sum(row.n_total for row in rows)
    bucket_all = {key: sum(getattr(row, key) for row in rows) for key in BUCKET_KEYS}
    quotas = _speaker_quotas(len(rows), ratios)
    return {
        split: {
            "n_total": total_all * ratios[split],
            "num_speakers": float(quotas[split]),
            **{key: bucket_all[key] * ratios[split] for key in BUCKET_KEYS},
        }
        for split in SPLIT_KEYS
    }


def _state_template() -> Dict[str, Dict[str, Any]]:
    return {
        split: {
            "speakers": [],
            "n_total": 0,
            "n_1spk": 0,
            "n_2spk": 0,
            "n_3spk": 0,
            "low_enrollment_speakers": 0,
        }
        for split in SPLIT_KEYS
    }


def _cost_for_split(
    state: Dict[str, Dict[str, Any]],
    row: SpeakerRow,
    split: str,
    target: Dict[str, Dict[str, float]],
    bucket_penalty: float,
    speaker_penalty: float,
) -> float:
    current = state[split]
    prospective = {
        "num_speakers": len(current["speakers"]) + 1,
        "n_total": current["n_total"] + row.n_total,
        "n_1spk": current["n_1spk"] + row.n_1spk,
        "n_2spk": current["n_2spk"] + row.n_2spk,
        "n_3spk": current["n_3spk"] + row.n_3spk,
    }
    target_split = target[split]

    total_cost = ((prospective["n_total"] - target_split["n_total"]) / max(1.0, target_split["n_total"])) ** 2
    speaker_cost = ((prospective["num_speakers"] - target_split["num_speakers"]) / max(1.0, target_split["num_speakers"])) ** 2
    bucket_cost = 0.0
    for key in BUCKET_KEYS:
        denom = max(1.0, target_split[key])
        bucket_cost += ((prospective[key] - target_split[key]) / denom) ** 2
    bucket_cost /= len(BUCKET_KEYS)

    overflow_cost = 0.0
    if prospective["n_total"] > target_split["n_total"]:
        overflow_cost += ((prospective["n_total"] - target_split["n_total"]) / max(1.0, target_split["n_total"])) ** 2

    return total_cost + (speaker_penalty * speaker_cost) + (bucket_penalty * bucket_cost) + overflow_cost


def build_speaker_split(
    speaker_stats_path: Path,
    output_dir: Path,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    bucket_penalty: float,
    speaker_penalty: float,
) -> Dict[str, str]:
    rows = [row for row in _load_rows(speaker_stats_path) if row.eligible_for_split]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows.sort(key=lambda row: row.n_total, reverse=True)

    ratios = {"train": float(train_ratio), "val": float(val_ratio), "test": float(test_ratio)}
    quotas = _speaker_quotas(len(rows), ratios)
    target = _target_totals(rows, ratios)
    state = _state_template()

    for row in rows:
        available_splits = [split for split in SPLIT_KEYS if len(state[split]["speakers"]) < quotas[split]]
        if not available_splits:
            available_splits = list(SPLIT_KEYS)
        split_scores = {
            split: _cost_for_split(state, row, split, target, bucket_penalty, speaker_penalty)
            for split in available_splits
        }
        best_split = min(available_splits, key=lambda split: (split_scores[split], split))
        state[best_split]["speakers"].append(row.speaker_id)
        state[best_split]["n_total"] += row.n_total
        state[best_split]["n_1spk"] += row.n_1spk
        state[best_split]["n_2spk"] += row.n_2spk
        state[best_split]["n_3spk"] += row.n_3spk
        if row.low_enrollment:
            state[best_split]["low_enrollment_speakers"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "version": "v1",
        "seed": int(seed),
        "policy": "speaker_balanced_greedy_with_bucket_penalty_and_quota",
        "ratios": ratios,
        "speaker_quotas": quotas,
        "penalties": {
            "bucket_penalty": float(bucket_penalty),
            "speaker_penalty": float(speaker_penalty),
        },
        "train": sorted(state["train"]["speakers"]),
        "val": sorted(state["val"]["speakers"]),
        "test": sorted(state["test"]["speakers"]),
    }
    dump_json(output_dir / "speaker_split.json", split_payload)

    summary = {
        split: SplitSummary(
            split=split,
            num_speakers=len(state[split]["speakers"]),
            speaker_quota=quotas[split],
            n_total=state[split]["n_total"],
            n_1spk=state[split]["n_1spk"],
            n_2spk=state[split]["n_2spk"],
            n_3spk=state[split]["n_3spk"],
            low_enrollment_speakers=state[split]["low_enrollment_speakers"],
        ).to_json()
        for split in SPLIT_KEYS
    }
    dump_json(
        output_dir / "split_summary.json",
        {
            "speaker_stats_path": str(speaker_stats_path.resolve()),
            "summary": summary,
        },
    )
    return {
        "speaker_split_path": str((output_dir / "speaker_split.json").resolve()),
        "split_summary_path": str((output_dir / "split_summary.json").resolve()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build speaker split for AURA data pipeline v1.")
    parser.add_argument("--speaker-stats-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--bucket-penalty", type=float, default=2.0)
    parser.add_argument("--speaker-penalty", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = build_speaker_split(
        speaker_stats_path=Path(args.speaker_stats_path),
        output_dir=Path(args.output_dir),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        bucket_penalty=float(args.bucket_penalty),
        speaker_penalty=float(args.speaker_penalty),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
