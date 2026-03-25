from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_pipeline.common import dump_json
from data_pipeline.enrollment import _embedding_output_path


@dataclass
class SpeakerRegistryRow:
    speaker_id: str
    split: str
    enroll_wav_paths: list[str]
    enroll_emb_paths: list[str]
    num_enroll_utts_raw: int
    num_enroll_utts: int
    embedding_model: str
    embedding_dim: int
    cache_version: str
    low_enrollment: bool
    missing_embedding_count: int
    eval_fixed_indices: list[int]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _load_split(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {split: list(payload.get(split, [])) for split in ("train", "val", "test")}


def _load_fixed_indices(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): [int(idx) for idx in value] for key, value in payload.items()}


def _hash_index_order(speaker_id: str, count: int, seed: int) -> list[int]:
    scored = []
    for idx in range(count):
        digest = hashlib.sha256(f"{seed}:{speaker_id}:{idx}".encode("utf-8")).hexdigest()
        scored.append((digest, idx))
    scored.sort()
    return [idx for _, idx in scored]


def _resolve_eval_fixed_indices(
    speaker_id: str,
    count: int,
    seed: int,
    enroll_k: int,
    fixed_indices_map: dict[str, list[int]],
) -> list[int]:
    requested = [idx for idx in fixed_indices_map.get(speaker_id, []) if 0 <= idx < count]
    if requested:
        return requested[: min(enroll_k, len(requested))]
    order = _hash_index_order(speaker_id=speaker_id, count=count, seed=seed)
    return order[: min(enroll_k, len(order))]


def build_speaker_registry(
    split_path: Path,
    enrollment_root: Path,
    cache_root: Path,
    output_dir: Path,
    *,
    enroll_k: int,
    seed: int,
    cache_version: str,
    embedding_model: str,
    fixed_indices_path: Path | None,
) -> dict[str, str]:
    split_map = _load_split(split_path)
    fixed_indices_map = _load_fixed_indices(fixed_indices_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[SpeakerRegistryRow] = []
    embedding_dims = set()
    split_counts = {split: 0 for split in split_map}

    for split, speakers in split_map.items():
        for speaker_id in sorted(speakers):
            wav_paths = sorted((enrollment_root / speaker_id).glob("*.wav"))
            aligned_wav_paths: list[str] = []
            aligned_emb_paths: list[str] = []
            sample_dim = 0

            for wav_path in wav_paths:
                emb_path = _embedding_output_path(cache_root, speaker_id, wav_path)
                if not emb_path.exists():
                    continue
                aligned_wav_paths.append(str(wav_path.resolve()))
                aligned_emb_paths.append(str(emb_path.resolve()))
                if sample_dim == 0:
                    sample_dim = int(np.load(emb_path, mmap_mode="r").shape[-1])

            if sample_dim > 0:
                embedding_dims.add(sample_dim)

            count = len(aligned_emb_paths)
            fixed_indices = _resolve_eval_fixed_indices(
                speaker_id=speaker_id,
                count=count,
                seed=seed,
                enroll_k=enroll_k,
                fixed_indices_map=fixed_indices_map,
            )
            rows.append(
                SpeakerRegistryRow(
                    speaker_id=speaker_id,
                    split=split,
                    enroll_wav_paths=aligned_wav_paths,
                    enroll_emb_paths=aligned_emb_paths,
                    num_enroll_utts_raw=len(wav_paths),
                    num_enroll_utts=count,
                    embedding_model=str(embedding_model),
                    embedding_dim=sample_dim,
                    cache_version=cache_version,
                    low_enrollment=0 < count < int(enroll_k),
                    missing_embedding_count=max(0, len(wav_paths) - count),
                    eval_fixed_indices=fixed_indices,
                )
            )
            split_counts[split] += 1

    rows_path = output_dir / "speaker_registry.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")

    dump_json(
        output_dir / "speaker_registry_report.json",
        {
            "speaker_split_path": str(split_path.resolve()),
            "enrollment_root": str(enrollment_root.resolve()),
            "cache_root": str(cache_root.resolve()),
            "embedding_model": str(embedding_model),
            "cache_version": str(cache_version),
            "fixed_indices_path": str(fixed_indices_path.resolve()) if fixed_indices_path is not None else None,
            "num_rows": len(rows),
            "split_counts": split_counts,
            "embedding_dims_seen": sorted(embedding_dims),
            "num_low_enrollment_rows": sum(1 for row in rows if row.low_enrollment),
            "num_missing_embedding_rows": sum(1 for row in rows if row.num_enroll_utts == 0),
            "low_enrollment_rule": f"0 < num_enroll_utts < {int(enroll_k)}",
            "missing_embedding_rule": "num_enroll_utts == 0",
        },
    )
    return {
        "speaker_registry_path": str(rows_path.resolve()),
        "speaker_registry_report": str((output_dir / "speaker_registry_report.json").resolve()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build speaker registry for AURA data pipeline v1.")
    parser.add_argument("--speaker-split-path", type=str, required=True)
    parser.add_argument("--enrollment-root", type=str, required=True)
    parser.add_argument("--cache-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--enroll-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--cache-version", type=str, default="v1")
    parser.add_argument("--embedding-model-name", type=str, default="speechbrain_ecapa")
    parser.add_argument("--fixed-indices-path", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = build_speaker_registry(
        split_path=Path(args.speaker_split_path),
        enrollment_root=Path(args.enrollment_root),
        cache_root=Path(args.cache_root),
        output_dir=Path(args.output_dir),
        enroll_k=int(args.enroll_k),
        seed=int(args.seed),
        cache_version=str(args.cache_version),
        embedding_model=str(args.embedding_model_name),
        fixed_indices_path=Path(args.fixed_indices_path) if args.fixed_indices_path else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
