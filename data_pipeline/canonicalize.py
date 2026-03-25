from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_pipeline.adaptors import AuraSourceAdaptor, BaseSourceAdaptor, SourceSpec
from data_pipeline.common import CanonicalSample, CanonicalizationReport, JsonlWriter, dump_json


ADAPTOR_REGISTRY = {
    "aura": AuraSourceAdaptor,
}


def _load_source_specs(args: argparse.Namespace) -> list[SourceSpec]:
    if args.source_config:
        with Path(args.source_config).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        specs = []
        for row in payload:
            specs.append(
                SourceSpec(
                    name=str(row["name"]),
                    adaptor=str(row.get("adaptor", "aura")),
                    root=Path(str(row["root"])).resolve(),
                    enrollment_root=(
                        Path(str(row["enrollment_root"])).resolve()
                        if row.get("enrollment_root")
                        else None
                    ),
                    meta_glob=str(row.get("meta_glob", "meta.jsonl")),
                )
            )
        return specs

    if not args.source_root:
        raise ValueError("Provide either --source-config or --source-root.")

    return [
        SourceSpec(
            name=str(args.dataset_name),
            adaptor=str(args.adaptor),
            root=Path(str(args.source_root)).resolve(),
            enrollment_root=(
                Path(str(args.enrollment_root)).resolve() if args.enrollment_root else None
            ),
            meta_glob=str(args.meta_glob),
        )
    ]


def _build_adaptor(spec: SourceSpec, args: argparse.Namespace) -> BaseSourceAdaptor:
    adaptor_cls = ADAPTOR_REGISTRY.get(spec.adaptor)
    if adaptor_cls is None:
        raise ValueError(f"Unsupported adaptor: {spec.adaptor}")
    return adaptor_cls(
        spec,
        expected_sample_rate=int(args.expected_sample_rate),
        expected_num_samples=int(args.expected_num_samples),
        audio_check_mode=str(args.audio_check_mode),
    )


def _sort_samples(samples: Iterable[CanonicalSample]) -> list[CanonicalSample]:
    return sorted(samples, key=lambda row: (row.source_dataset, row.task_bucket, row.id))


def build_canonical_samples(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_source_specs(args)
    all_samples: list[CanonicalSample] = []
    source_reports = []
    totals_bucket = Counter()
    totals_ineligible = Counter()

    for spec in specs:
        adaptor = _build_adaptor(spec, args)
        samples = list(adaptor.iter_samples())
        all_samples.extend(samples)
        totals_bucket.update(sample.task_bucket for sample in samples)
        totals_ineligible.update(
            sample.split_ineligible_reason
            for sample in samples
            if sample.split_ineligible_reason is not None
        )
        source_reports.append(adaptor.stats.to_json())

    sorted_samples = _sort_samples(all_samples)
    canonical_path = output_dir / "canonical_samples.jsonl"
    written = JsonlWriter(canonical_path).write_all(sample.to_json() for sample in sorted_samples)

    report = CanonicalizationReport(
        version="v1",
        generated_by="build_canonical_samples.py",
        sources=source_reports,
        totals={
            "num_sources": len(specs),
            "num_rows_written": written,
            "task_bucket_counts": dict(totals_bucket),
            "split_ineligible_reasons": dict(totals_ineligible),
        },
    )
    dump_json(output_dir / "canonicalization_report.json", report.to_json())
    return {
        "canonical_path": str(canonical_path),
        "report_path": str((output_dir / "canonicalization_report.json").resolve()),
        "num_rows_written": written,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical sample table for AURA data pipeline v1.")
    parser.add_argument("--source-config", type=str, default=None, help="JSON file with source specs.")
    parser.add_argument("--source-root", type=str, default=None, help="Single source root for shortcut mode.")
    parser.add_argument("--dataset-name", type=str, default="Aura_v2", help="Dataset name for shortcut mode.")
    parser.add_argument("--adaptor", type=str, default="aura", help="Adaptor name for shortcut mode.")
    parser.add_argument("--enrollment-root", type=str, default=None, help="Optional enrollment root for shortcut mode.")
    parser.add_argument("--meta-glob", type=str, default="meta.jsonl", help="Meta glob for shortcut mode.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write canonical artifacts.")
    parser.add_argument("--expected-sample-rate", type=int, default=16000, help="Canonical sample rate.")
    parser.add_argument("--expected-num-samples", type=int, default=96000, help="Canonical waveform length in samples.")
    parser.add_argument(
        "--audio-check-mode",
        type=str,
        default="exists",
        choices=["none", "exists"],
        help="Whether to check audio path existence during canonicalization.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = build_canonical_samples(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
