from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from data_pipeline.audit import audit_pipeline
from data_pipeline.canonicalize import build_canonical_samples
from data_pipeline.enrollment import build_enrollment_cache
from data_pipeline.layout import prepare_run_layout
from data_pipeline.manifests import build_split_manifests
from data_pipeline.registry import build_speaker_registry
from data_pipeline.split import build_speaker_split
from data_pipeline.stats import build_speaker_stats
from data_pipeline.common import dump_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AURA data pipeline v1 into a clean structured artifact layout.")
    parser.add_argument("--artifact-root", type=str, default="/workspace/project/data_pipeline_artifacts")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--source-config", type=str, required=True)
    parser.add_argument("--enrollment-root", type=str, required=True)
    parser.add_argument("--embedding-model-name", type=str, default="speechbrain_ecapa")
    parser.add_argument("--spkrec-source", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--spkrec-savedir", type=str, default="/workspace/project/pretrained_models/spkrec-ecapa-voxceleb")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--expected-num-samples", type=int, default=96000)
    parser.add_argument("--audio-check-mode", type=str, default="exists", choices=["none", "exists"])
    parser.add_argument("--enroll-k", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--bucket-penalty", type=float, default=2.0)
    parser.add_argument("--speaker-penalty", type=float, default=0.25)
    parser.add_argument("--cache-version", type=str, default="v1")
    parser.add_argument("--fixed-indices-path", type=str, default=None)
    parser.add_argument("--skip-enrollment-cache", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--note", action="append", default=[])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    notes = {"note_lines": list(args.note)} if args.note else {}
    layout = prepare_run_layout(
        artifact_root=Path(args.artifact_root),
        run_name=str(args.run_name),
        embedding_model_name=str(args.embedding_model_name),
        source_config_path=Path(args.source_config),
        notes=notes,
        exist_ok=bool(args.exist_ok),
    )

    canonical_result = build_canonical_samples(
        Namespace(
            source_config=str(Path(args.source_config).resolve()),
            source_root=None,
            dataset_name="unused",
            adaptor="aura",
            enrollment_root=None,
            meta_glob="meta.jsonl",
            output_dir=layout.canonical_dir,
            expected_sample_rate=int(args.sample_rate),
            expected_num_samples=int(args.expected_num_samples),
            audio_check_mode=str(args.audio_check_mode),
        )
    )
    speaker_stats_result = build_speaker_stats(
        canonical_path=Path(canonical_result["canonical_path"]),
        enrollment_root=Path(args.enrollment_root),
        output_dir=Path(layout.speaker_stats_dir),
        enroll_k=int(args.enroll_k),
    )
    speaker_split_result = build_speaker_split(
        speaker_stats_path=Path(speaker_stats_result["speaker_stats_path"]),
        output_dir=Path(layout.speaker_split_dir),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        bucket_penalty=float(args.bucket_penalty),
        speaker_penalty=float(args.speaker_penalty),
    )

    if not args.skip_enrollment_cache:
        enrollment_cache_result = build_enrollment_cache(
            enrollment_root=Path(args.enrollment_root),
            split_path=Path(speaker_split_result["speaker_split_path"]),
            cache_root=Path(layout.cache_embedding_dir),
            embedding_model=str(args.embedding_model_name),
            source=str(args.spkrec_source),
            savedir=str(args.spkrec_savedir),
            device=str(args.device),
            sample_rate=int(args.sample_rate),
            selected_splits=["train", "val", "test"],
            max_speakers=None,
            force=False,
        )
    else:
        enrollment_cache_result = {
            "cache_root": str(Path(layout.cache_embedding_dir).resolve()),
            "cache_report": None,
            "skipped": True,
        }

    speaker_registry_result = build_speaker_registry(
        split_path=Path(speaker_split_result["speaker_split_path"]),
        enrollment_root=Path(args.enrollment_root),
        cache_root=Path(layout.cache_embedding_dir),
        output_dir=Path(layout.registry_dir),
        enroll_k=int(args.enroll_k),
        seed=int(args.seed),
        cache_version=str(args.cache_version),
        embedding_model=str(args.embedding_model_name),
        fixed_indices_path=Path(args.fixed_indices_path) if args.fixed_indices_path else None,
    )
    manifests_result = build_split_manifests(
        canonical_path=Path(canonical_result["canonical_path"]),
        split_path=Path(speaker_split_result["speaker_split_path"]),
        output_dir=Path(layout.manifests_dir),
    )
    audit_result = audit_pipeline(
        split_path=Path(speaker_split_result["speaker_split_path"]),
        registry_path=Path(speaker_registry_result["speaker_registry_path"]),
        manifest_dir=Path(layout.manifests_dir),
        output_dir=Path(layout.audit_dir),
        enroll_k=int(args.enroll_k),
        canonical_path=Path(canonical_result["canonical_path"]),
    )

    report = {
        "run_root": layout.run_root,
        "canonical": canonical_result,
        "speaker_stats": speaker_stats_result,
        "speaker_split": speaker_split_result,
        "enrollment_cache": enrollment_cache_result,
        "speaker_registry": speaker_registry_result,
        "manifests": manifests_result,
        "audit": audit_result,
    }
    dump_json(Path(layout.reports_dir) / "pipeline_run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
