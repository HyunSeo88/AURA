from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data_pipeline.common import dump_json, iter_jsonl


def _load_split(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {split: set(payload.get(split, [])) for split in ("train", "val", "test")}


def _load_registry(path: Path) -> dict[str, dict[str, Any]]:
    registry = {}
    for _, line in iter_jsonl(path):
        if not line.strip():
            continue
        row = json.loads(line)
        registry[str(row["speaker_id"])] = row
    return registry


def _count_manifest(path: Path) -> tuple[int, set[str], set[str], set[str], dict[str, int], set[str]]:
    num_rows = 0
    sample_ids = set()
    mix_paths = set()
    target_paths = set()
    speakers = set()
    buckets = {"1spk": 0, "2spk": 0, "3spk": 0}
    for _, line in iter_jsonl(path):
        if not line.strip():
            continue
        row = json.loads(line)
        num_rows += 1
        sample_ids.add(str(row["id"]))
        mix_paths.add(str(row["mix_path"]))
        target_paths.add(str(row["target_path"]))
        speakers.add(str(row["target_speaker_id"]))
        bucket = str(row["task_bucket"])
        if bucket in buckets:
            buckets[bucket] += 1
    return num_rows, sample_ids, mix_paths, target_paths, buckets, speakers


def _scan_canonical(path: Path) -> dict[str, Any]:
    reasons = Counter()
    total = 0
    eligible = 0
    for _, line in iter_jsonl(path):
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        if bool(row.get("split_eligible", True)):
            eligible += 1
        else:
            reasons[str(row.get("split_ineligible_reason") or "unknown")] += 1
    return {
        "num_canonical_rows": total,
        "num_split_eligible_rows": eligible,
        "num_split_ineligible_rows": total - eligible,
        "split_ineligible_reason_counts": dict(sorted(reasons.items())),
    }


def audit_pipeline(
    split_path: Path,
    registry_path: Path,
    manifest_dir: Path,
    output_dir: Path,
    *,
    enroll_k: int,
    canonical_path: Path | None,
) -> dict[str, str]:
    split_map = _load_split(split_path)
    registry = _load_registry(registry_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlaps = {
        "target_speaker_overlap_count": len(split_map["train"] & split_map["val"])
        + len(split_map["train"] & split_map["test"])
        + len(split_map["val"] & split_map["test"]),
    }
    manifest_stats = {}
    sample_sets = {}
    mix_sets = {}
    target_sets = {}

    for split in ("train", "val", "test"):
        manifest_path = manifest_dir / f"{split}_manifest.jsonl"
        num_rows, sample_ids, mix_paths, target_paths, buckets, manifest_speakers = _count_manifest(manifest_path)
        sample_sets[split] = sample_ids
        mix_sets[split] = mix_paths
        target_sets[split] = target_paths
        split_speakers = split_map[split]
        low_enrollment = sorted(
            speaker_id
            for speaker_id in split_speakers
            if speaker_id in registry and bool(registry[speaker_id].get("low_enrollment", False))
        )
        missing_registry = sorted(speaker_id for speaker_id in split_speakers if speaker_id not in registry)
        missing_embeddings = sorted(
            speaker_id
            for speaker_id in split_speakers
            if speaker_id in registry and int(registry[speaker_id].get("num_enroll_utts", 0)) == 0
        )
        registry_split_mismatches = sorted(
            speaker_id
            for speaker_id in split_speakers
            if speaker_id in registry and str(registry[speaker_id].get("split")) != split
        )
        unexpected_manifest_speakers = sorted(manifest_speakers - split_speakers)
        manifest_stats[split] = {
            "num_rows": num_rows,
            "num_speakers": len(split_speakers),
            "bucket_counts": buckets,
            "low_enrollment_speakers": low_enrollment,
            "low_enrollment_speaker_count": len(low_enrollment),
            "missing_registry_speakers": missing_registry,
            "missing_registry_speaker_count": len(missing_registry),
            "missing_embedding_speakers": missing_embeddings,
            "missing_embedding_speaker_count": len(missing_embeddings),
            "registry_split_mismatches": registry_split_mismatches,
            "unexpected_manifest_speakers": unexpected_manifest_speakers,
        }

    payload = {
        "rules": {
            "split_eligible_rule": "canonical row is split eligible iff target speaker id is present, task bucket is one of {1spk,2spk,3spk}, and both mix_path and target_path exist",
            "low_enrollment_rule": f"speaker is low enrollment iff 0 < num_enroll_utts < {int(enroll_k)}",
            "missing_embedding_rule": "speaker has missing enrollment embeddings iff num_enroll_utts == 0",
            "eval_enrollment_subset_rule": "val/test enrollment subset is deterministic via fixed indices when provided, otherwise via speaker-hash ordering",
        },
        **overlaps,
        "duplicate_sample_id_count": len((sample_sets["train"] & sample_sets["val"]) | (sample_sets["train"] & sample_sets["test"]) | (sample_sets["val"] & sample_sets["test"])),
        "mix_path_overlap_count": len((mix_sets["train"] & mix_sets["val"]) | (mix_sets["train"] & mix_sets["test"]) | (mix_sets["val"] & mix_sets["test"])),
        "target_path_overlap_count": len((target_sets["train"] & target_sets["val"]) | (target_sets["train"] & target_sets["test"]) | (target_sets["val"] & target_sets["test"])),
        "enroll_k": int(enroll_k),
        "split_stats": manifest_stats,
    }
    if canonical_path is not None:
        payload["canonical_summary"] = _scan_canonical(canonical_path)
        payload["canonical_path"] = str(canonical_path.resolve())
    dump_json(output_dir / "leak_audit.json", payload)
    return {"leak_audit_path": str((output_dir / "leak_audit.json").resolve())}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit AURA data pipeline v1 outputs.")
    parser.add_argument("--speaker-split-path", type=str, required=True)
    parser.add_argument("--speaker-registry-path", type=str, required=True)
    parser.add_argument("--manifest-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--canonical-path", type=str, default=None)
    parser.add_argument("--enroll-k", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = audit_pipeline(
        split_path=Path(args.speaker_split_path),
        registry_path=Path(args.speaker_registry_path),
        manifest_dir=Path(args.manifest_dir),
        output_dir=Path(args.output_dir),
        enroll_k=int(args.enroll_k),
        canonical_path=Path(args.canonical_path) if args.canonical_path else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
