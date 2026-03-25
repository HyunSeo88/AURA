from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data_pipeline.common import dump_json, iter_jsonl
from data_pipeline.enrollment import build_enrollment_cache
from data_pipeline.registry import build_speaker_registry


SPLIT_KEYS = ("train", "val", "test")


@dataclass(frozen=True)
class ExternalAbsentUtterance:
    speaker_id: str
    utterance_id: str
    wav_relpath: str
    source_dataset: str | None
    source_partition: str | None
    clean_only: bool | None
    contains_music: bool | None
    contains_overlap: bool | None
    sample_rate: int | None
    num_channels: int | None
    duration_sec: float | None


@dataclass(frozen=True)
class ExternalAbsentSpeakerStatsRow:
    speaker_id: str
    n_total: int
    num_enroll_utts_raw: int
    eligible_for_split: bool
    low_enrollment: bool
    source_dataset: str | None
    source_partition: str | None
    flagged_music_utterances: int
    flagged_overlap_utterances: int
    missing_files: int
    invalid_clean_only_utterances: int
    split_ineligible_reason: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_speaker_ids_from_registry(path: Path) -> set[str]:
    speaker_ids: set[str] = set()
    for _, line in iter_jsonl(path):
        if not line.strip():
            continue
        row = json.loads(line)
        speaker_ids.add(str(row["speaker_id"]))
    return speaker_ids


def _load_absent_utterances(absent_root: Path) -> tuple[list[ExternalAbsentUtterance], Path]:
    manifest_path = absent_root / "manifests" / "utterances.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"external absent utterance manifest not found: {manifest_path}")
    rows: list[ExternalAbsentUtterance] = []
    for _, line in iter_jsonl(manifest_path):
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            ExternalAbsentUtterance(
                speaker_id=str(payload["speaker_id"]),
                utterance_id=str(payload["utterance_id"]),
                wav_relpath=str(payload["wav_relpath"]),
                source_dataset=payload.get("source_dataset"),
                source_partition=payload.get("source_partition"),
                clean_only=payload.get("clean_only"),
                contains_music=payload.get("contains_music"),
                contains_overlap=payload.get("contains_overlap"),
                sample_rate=int(payload["sample_rate"]) if payload.get("sample_rate") is not None else None,
                num_channels=int(payload["num_channels"]) if payload.get("num_channels") is not None else None,
                duration_sec=float(payload["duration_sec"]) if payload.get("duration_sec") is not None else None,
            )
        )
    return rows, manifest_path


def _load_speakers_manifest(absent_root: Path) -> tuple[dict[str, dict[str, Any]], Path]:
    manifest_path = absent_root / "manifests" / "speakers.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"external absent speaker manifest not found: {manifest_path}")
    payloads: dict[str, dict[str, Any]] = {}
    for _, line in iter_jsonl(manifest_path):
        if not line.strip():
            continue
        row = json.loads(line)
        payloads[str(row["speaker_id"])] = row
    return payloads, manifest_path


def _speaker_quotas(num_speakers: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: num_speakers * ratios[split] for split in SPLIT_KEYS}
    quotas = {split: int(math.floor(raw[split])) for split in SPLIT_KEYS}
    remainder = num_speakers - sum(quotas.values())
    order = sorted(SPLIT_KEYS, key=lambda split: (raw[split] - quotas[split], split), reverse=True)
    for split in order[:remainder]:
        quotas[split] += 1
    return quotas


def _build_external_split(
    rows: list[ExternalAbsentSpeakerStatsRow],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    speaker_penalty: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    eligible_rows = [row for row in rows if row.eligible_for_split]
    ratios = {"train": float(train_ratio), "val": float(val_ratio), "test": float(test_ratio)}
    quotas = _speaker_quotas(len(eligible_rows), ratios)
    target_total = sum(row.n_total for row in eligible_rows)
    target = {
        split: {
            "num_speakers": float(quotas[split]),
            "n_total": float(target_total * ratios[split]),
        }
        for split in SPLIT_KEYS
    }

    state = {
        split: {"speakers": [], "n_total": 0, "low_enrollment_speakers": 0}
        for split in SPLIT_KEYS
    }
    rng = random.Random(int(seed))
    rng.shuffle(eligible_rows)
    eligible_rows.sort(key=lambda row: row.n_total, reverse=True)

    def split_cost(split: str, row: ExternalAbsentSpeakerStatsRow) -> float:
        cur = state[split]
        prospective_speakers = len(cur["speakers"]) + 1
        prospective_total = cur["n_total"] + row.n_total
        target_split = target[split]
        total_cost = ((prospective_total - target_split["n_total"]) / max(1.0, target_split["n_total"])) ** 2
        speaker_cost = ((prospective_speakers - target_split["num_speakers"]) / max(1.0, target_split["num_speakers"])) ** 2
        overflow = 0.0
        if prospective_total > target_split["n_total"]:
            overflow = ((prospective_total - target_split["n_total"]) / max(1.0, target_split["n_total"])) ** 2
        return total_cost + (float(speaker_penalty) * speaker_cost) + overflow

    for row in eligible_rows:
        available = [split for split in SPLIT_KEYS if len(state[split]["speakers"]) < quotas[split]]
        if not available:
            available = list(SPLIT_KEYS)
        best_split = min(available, key=lambda split: (split_cost(split, row), split))
        state[best_split]["speakers"].append(row.speaker_id)
        state[best_split]["n_total"] += row.n_total
        if row.low_enrollment:
            state[best_split]["low_enrollment_speakers"] += 1

    split_payload = {
        "version": "v1",
        "seed": int(seed),
        "policy": "external_absent_speaker_balanced_greedy_by_num_utts",
        "ratios": ratios,
        "speaker_quotas": quotas,
        "penalties": {"speaker_penalty": float(speaker_penalty)},
        "train": sorted(state["train"]["speakers"]),
        "val": sorted(state["val"]["speakers"]),
        "test": sorted(state["test"]["speakers"]),
    }
    summary = {
        split: {
            "split": split,
            "num_speakers": len(state[split]["speakers"]),
            "speaker_quota": quotas[split],
            "n_total": state[split]["n_total"],
            "low_enrollment_speakers": state[split]["low_enrollment_speakers"],
        }
        for split in SPLIT_KEYS
    }
    return split_payload, summary


def build_external_absent_pipeline(
    absent_root: Path,
    output_dir: Path,
    *,
    embedding_model: str,
    spkrec_source: str,
    spkrec_savedir: str,
    device: str,
    sample_rate: int,
    enroll_k: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    speaker_penalty: float,
    cache_version: str,
    fixed_indices_path: Path | None,
    reference_registry_path: Path | None,
    skip_enrollment_cache: bool,
) -> dict[str, Any]:
    absent_root = absent_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = output_dir / "config"
    speaker_stats_dir = output_dir / "speaker_stats"
    speaker_split_dir = output_dir / "speaker_split"
    cache_root = output_dir / "cache" / embedding_model
    registry_dir = output_dir / "registry"
    audit_dir = output_dir / "audit"
    reports_dir = output_dir / "reports"
    for path in (config_dir, speaker_stats_dir, speaker_split_dir, cache_root, registry_dir, audit_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    utterances, utterances_manifest_path = _load_absent_utterances(absent_root)
    speakers_manifest, speakers_manifest_path = _load_speakers_manifest(absent_root)
    dataset_info_path = absent_root / "manifests" / "dataset_info.json"
    dataset_info = _load_json(dataset_info_path) if dataset_info_path.exists() else {}
    enrollment_root = absent_root / "enrollment"

    by_speaker: dict[str, list[ExternalAbsentUtterance]] = defaultdict(list)
    speaker_sources: dict[str, tuple[str | None, str | None]] = {}
    missing_files = 0
    missing_speaker_manifest_refs = 0
    for row in utterances:
        by_speaker[row.speaker_id].append(row)
        if row.speaker_id not in speakers_manifest:
            missing_speaker_manifest_refs += 1
        if row.speaker_id not in speaker_sources:
            speaker_sources[row.speaker_id] = (row.source_dataset, row.source_partition)
        wav_path = absent_root / row.wav_relpath
        if not wav_path.exists():
            missing_files += 1

    reference_speakers = _load_speaker_ids_from_registry(reference_registry_path) if reference_registry_path is not None else set()
    overlap_with_reference = sorted(set(by_speaker.keys()) & reference_speakers)

    stats_rows: list[ExternalAbsentSpeakerStatsRow] = []
    low_enrollment_count = 0
    for speaker_id in sorted(by_speaker):
        rows = by_speaker[speaker_id]
        flagged_music = sum(1 for row in rows if bool(row.contains_music))
        flagged_overlap = sum(1 for row in rows if bool(row.contains_overlap))
        invalid_clean = sum(1 for row in rows if row.clean_only is False)
        missing = sum(1 for row in rows if not (absent_root / row.wav_relpath).exists())
        num_utts = len(rows)
        low_enrollment = 0 < num_utts < int(enroll_k)
        if low_enrollment:
            low_enrollment_count += 1
        ineligible_reason = None
        eligible = True
        if speaker_id in reference_speakers:
            eligible = False
            ineligible_reason = "speaker_overlap_with_reference_registry"
        elif missing > 0:
            eligible = False
            ineligible_reason = "missing_enrollment_files"
        elif invalid_clean > 0:
            eligible = False
            ineligible_reason = "clean_only_false_present"
        elif flagged_overlap > 0:
            eligible = False
            ineligible_reason = "contains_overlap_true_present"
        elif flagged_music > 0:
            eligible = False
            ineligible_reason = "contains_music_true_present"

        source_dataset = speakers_manifest.get(speaker_id, {}).get("source_dataset")
        source_partition = speakers_manifest.get(speaker_id, {}).get("source_partition")
        if source_dataset is None or source_partition is None:
            source_dataset, source_partition = speaker_sources.get(speaker_id, (source_dataset, source_partition))

        stats_rows.append(
            ExternalAbsentSpeakerStatsRow(
                speaker_id=speaker_id,
                n_total=num_utts,
                num_enroll_utts_raw=num_utts,
                eligible_for_split=eligible,
                low_enrollment=low_enrollment,
                source_dataset=source_dataset,
                source_partition=source_partition,
                flagged_music_utterances=flagged_music,
                flagged_overlap_utterances=flagged_overlap,
                missing_files=missing,
                invalid_clean_only_utterances=invalid_clean,
                split_ineligible_reason=ineligible_reason,
            )
        )

    stats_path = speaker_stats_dir / "speaker_stats.jsonl"
    with stats_path.open("w", encoding="utf-8") as handle:
        for row in stats_rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")

    dump_json(
        speaker_stats_dir / "speaker_stats_report.json",
        {
            "absent_root": str(absent_root),
            "utterances_manifest_path": str(utterances_manifest_path.resolve()),
            "speakers_manifest_path": str(speakers_manifest_path.resolve()),
            "dataset_info_path": str(dataset_info_path.resolve()) if dataset_info_path.exists() else None,
            "num_utterance_rows": len(utterances),
            "num_speakers": len(by_speaker),
            "num_reference_overlap_speakers": len(overlap_with_reference),
            "reference_overlap_speakers_sample": overlap_with_reference[:50],
            "num_missing_speaker_manifest_refs": missing_speaker_manifest_refs,
            "num_missing_wav_files": missing_files,
            "low_enrollment_rule": f"0 < num_enroll_utts < {int(enroll_k)}",
            "num_low_enrollment_speakers": low_enrollment_count,
            "dataset_info": dataset_info,
        },
    )

    split_payload, split_summary = _build_external_split(
        stats_rows,
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed),
        speaker_penalty=float(speaker_penalty),
    )
    dump_json(speaker_split_dir / "speaker_split.json", split_payload)
    dump_json(
        speaker_split_dir / "split_summary.json",
        {
            "speaker_stats_path": str(stats_path.resolve()),
            "summary": split_summary,
        },
    )

    if not skip_enrollment_cache:
        enrollment_cache_result = build_enrollment_cache(
            enrollment_root=enrollment_root,
            split_path=speaker_split_dir / "speaker_split.json",
            cache_root=cache_root,
            embedding_model=str(embedding_model),
            source=str(spkrec_source),
            savedir=str(spkrec_savedir),
            device=str(device),
            sample_rate=int(sample_rate),
            selected_splits=["train", "val", "test"],
            max_speakers=None,
            force=False,
        )
    else:
        enrollment_cache_result = {
            "cache_root": str(cache_root.resolve()),
            "cache_report": None,
            "skipped": True,
        }

    registry_result = build_speaker_registry(
        split_path=speaker_split_dir / "speaker_split.json",
        enrollment_root=enrollment_root,
        cache_root=cache_root,
        output_dir=registry_dir,
        enroll_k=int(enroll_k),
        seed=int(seed),
        cache_version=str(cache_version),
        embedding_model=str(embedding_model),
        fixed_indices_path=fixed_indices_path,
    )

    split_sets = {split: set(split_payload.get(split, [])) for split in SPLIT_KEYS}
    overlap_counts = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    ineligible_reason_counts = Counter(
        row.split_ineligible_reason or "eligible"
        for row in stats_rows
        if not row.eligible_for_split
    )
    audit_payload = {
        "rules": {
            "speaker_overlap_rule": "external absent speaker ids must not overlap the reference registry speaker ids",
            "low_enrollment_rule": f"speaker is low enrollment iff 0 < num_enroll_utts < {int(enroll_k)}",
            "wav_source_rule": "external absent enrollment wavs are resolved from absent_root/wav_relpath in utterances manifest",
        },
        "absent_root": str(absent_root),
        "reference_registry_path": str(reference_registry_path.resolve()) if reference_registry_path is not None else None,
        "reference_overlap_speaker_count": len(overlap_with_reference),
        "reference_overlap_speakers_sample": overlap_with_reference[:50],
        "split_overlap_counts": overlap_counts,
        "num_speakers_total": len(stats_rows),
        "num_speakers_eligible": sum(1 for row in stats_rows if row.eligible_for_split),
        "num_speakers_ineligible": sum(1 for row in stats_rows if not row.eligible_for_split),
        "split_ineligible_reason_counts": dict(sorted(ineligible_reason_counts.items())),
        "split_counts": {split: len(split_payload.get(split, [])) for split in SPLIT_KEYS},
    }
    dump_json(audit_dir / "external_absent_audit.json", audit_payload)

    run_report = {
        "absent_root": str(absent_root),
        "enrollment_root": str(enrollment_root.resolve()),
        "speaker_stats": str(stats_path.resolve()),
        "speaker_split": str((speaker_split_dir / "speaker_split.json").resolve()),
        "speaker_split_summary": str((speaker_split_dir / "split_summary.json").resolve()),
        "enrollment_cache": enrollment_cache_result,
        "speaker_registry": registry_result,
        "audit": str((audit_dir / "external_absent_audit.json").resolve()),
    }
    dump_json(reports_dir / "external_absent_run_report.json", run_report)
    dump_json(
        config_dir / "external_absent_context.json",
        {
            "absent_root": str(absent_root),
            "embedding_model": str(embedding_model),
            "spkrec_source": str(spkrec_source),
            "spkrec_savedir": str(spkrec_savedir),
            "device": str(device),
            "sample_rate": int(sample_rate),
            "enroll_k": int(enroll_k),
            "seed": int(seed),
            "reference_registry_path": str(reference_registry_path.resolve()) if reference_registry_path is not None else None,
        },
    )
    return run_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build external absent donor artifacts for AuraPA.")
    parser.add_argument("--absent-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--embedding-model-name", type=str, default="speechbrain_ecapa")
    parser.add_argument("--spkrec-source", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--spkrec-savedir", type=str, default="/workspace/project/pretrained_models/spkrec-ecapa-voxceleb")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--enroll-k", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--speaker-penalty", type=float, default=0.25)
    parser.add_argument("--cache-version", type=str, default="v1")
    parser.add_argument("--fixed-indices-path", type=str, default=None)
    parser.add_argument("--reference-registry-path", type=str, default=None)
    parser.add_argument("--skip-enrollment-cache", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = build_external_absent_pipeline(
        absent_root=Path(args.absent_root),
        output_dir=Path(args.output_dir),
        embedding_model=str(args.embedding_model_name),
        spkrec_source=str(args.spkrec_source),
        spkrec_savedir=str(args.spkrec_savedir),
        device=str(args.device),
        sample_rate=int(args.sample_rate),
        enroll_k=int(args.enroll_k),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        speaker_penalty=float(args.speaker_penalty),
        cache_version=str(args.cache_version),
        fixed_indices_path=Path(args.fixed_indices_path) if args.fixed_indices_path else None,
        reference_registry_path=Path(args.reference_registry_path) if args.reference_registry_path else None,
        skip_enrollment_cache=bool(args.skip_enrollment_cache),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
