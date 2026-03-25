from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.common import dump_json, iter_jsonl


def _load_split(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {split: set(payload.get(split, [])) for split in ("train", "val", "test")}


def build_split_manifests(canonical_path: Path, split_path: Path, output_dir: Path) -> dict[str, str]:
    split_map = _load_split(split_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {split: (output_dir / f"{split}_manifest.jsonl").open("w", encoding="utf-8") for split in split_map}
    counts = {
        split: {"num_rows": 0, "bucket_counts": {"1spk": 0, "2spk": 0, "3spk": 0}}
        for split in split_map
    }
    dropped_ineligible = 0
    dropped_unassigned = 0

    try:
        for _, line in iter_jsonl(canonical_path):
            if not line.strip():
                continue
            row = json.loads(line)
            if not bool(row.get("split_eligible", True)):
                dropped_ineligible += 1
                continue
            speaker_id = str(row["target_speaker_id"])
            split = None
            for key, speakers in split_map.items():
                if speaker_id in speakers:
                    split = key
                    break
            if split is None:
                dropped_unassigned += 1
                continue
            payload = {
                "id": row["id"],
                "split": split,
                "task_bucket": row["task_bucket"],
                "target_speaker_id": speaker_id,
                "mix_path": row["mix_path"],
                "target_path": row["target_path"],
                "scenario": row["scenario"],
                "noise_types": row.get("noise_types", []),
                "sample_rate": row["sample_rate"],
                "num_samples": row["num_samples"],
            }
            writers[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts[split]["num_rows"] += 1
            bucket = str(row.get("task_bucket", ""))
            if bucket in counts[split]["bucket_counts"]:
                counts[split]["bucket_counts"][bucket] += 1
    finally:
        for handle in writers.values():
            handle.close()

    dump_json(
        output_dir / "manifest_report.json",
        {
            "canonical_path": str(canonical_path.resolve()),
            "speaker_split_path": str(split_path.resolve()),
            "dropped_split_ineligible_rows": dropped_ineligible,
            "dropped_unassigned_rows": dropped_unassigned,
            "split_counts": counts,
        },
    )
    return {split: str((output_dir / f"{split}_manifest.jsonl").resolve()) for split in split_map}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build split manifests for AURA data pipeline v1.")
    parser.add_argument("--canonical-path", type=str, required=True)
    parser.add_argument("--speaker-split-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = build_split_manifests(
        canonical_path=Path(args.canonical_path),
        split_path=Path(args.speaker_split_path),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
