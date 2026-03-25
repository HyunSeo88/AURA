from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from data_pipeline.common import dump_json


@dataclass
class EnrollmentCacheReport:
    embedding_model: str
    source: str
    savedir: str
    sample_rate: int
    device: str
    selected_splits: list[str]
    max_speakers: int | None
    num_speakers_requested: int
    num_speakers_processed: int
    num_utterances_seen: int
    num_embeddings_written: int
    num_embeddings_skipped: int
    errors: list[str]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _load_split_speakers(path: Path, selected_splits: list[str]) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    speakers = set()
    for split in selected_splits:
        speakers.update(payload.get(split, []))
    return sorted(speakers)


def _build_encoder(source: str, savedir: str, device: str):
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source=source,
        savedir=savedir,
        run_opts={"device": device},
    )
    encoder.eval()
    return encoder


def _load_wav(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim != 2:
        raise ValueError(f"invalid wav shape at {path}: {tuple(wav.shape)}")
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = embedding.astype(np.float32, copy=False).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if norm > 0.0:
        embedding = embedding / norm
    return embedding


def _embedding_output_path(cache_root: Path, speaker_id: str, wav_path: Path) -> Path:
    digest = hashlib.sha1(str(wav_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return cache_root / speaker_id / f"{wav_path.stem}__{digest}.npy"


def build_enrollment_cache(
    enrollment_root: Path,
    split_path: Path,
    cache_root: Path,
    *,
    embedding_model: str,
    source: str,
    savedir: str,
    device: str,
    sample_rate: int,
    selected_splits: list[str],
    max_speakers: int | None,
    force: bool,
) -> dict[str, str]:
    speakers = _load_split_speakers(split_path, selected_splits=selected_splits)
    num_requested = len(speakers)
    if max_speakers is not None:
        speakers = speakers[: max_speakers]
    encoder = _build_encoder(source=source, savedir=savedir, device=device)

    num_seen = 0
    num_written = 0
    num_skipped = 0
    errors: list[str] = []

    for speaker_id in speakers:
        speaker_dir = enrollment_root / speaker_id
        if not speaker_dir.exists():
            errors.append(f"missing_enrollment_dir:{speaker_id}")
            continue
        wav_paths = sorted(speaker_dir.glob("*.wav"))
        if not wav_paths:
            errors.append(f"empty_enrollment_dir:{speaker_id}")
            continue
        for wav_path in wav_paths:
            num_seen += 1
            out_path = _embedding_output_path(cache_root, speaker_id, wav_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not force:
                num_skipped += 1
                continue
            try:
                wav = _load_wav(wav_path, sample_rate=sample_rate)
                with torch.no_grad():
                    emb = encoder.encode_batch(wav.to(device)).detach().float().cpu().numpy()
                emb = _normalize_embedding(emb)
                np.save(out_path, emb)
                num_written += 1
            except Exception as exc:  # pragma: no cover - defensive reporting
                errors.append(f"embedding_failed:{speaker_id}:{wav_path.name}:{exc}")

    report = EnrollmentCacheReport(
        embedding_model=str(embedding_model),
        source=str(source),
        savedir=str(savedir),
        sample_rate=int(sample_rate),
        device=str(device),
        selected_splits=list(selected_splits),
        max_speakers=max_speakers,
        num_speakers_requested=num_requested,
        num_speakers_processed=len(speakers),
        num_utterances_seen=num_seen,
        num_embeddings_written=num_written,
        num_embeddings_skipped=num_skipped,
        errors=errors,
    )
    dump_json(cache_root / "cache_report.json", report.to_json())
    return {
        "cache_root": str(cache_root.resolve()),
        "cache_report": str((cache_root / "cache_report.json").resolve()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract utterance-level enrollment embeddings for AURA data pipeline v1.")
    parser.add_argument("--enrollment-root", type=str, required=True)
    parser.add_argument("--speaker-split-path", type=str, required=True)
    parser.add_argument("--cache-root", type=str, required=True)
    parser.add_argument("--embedding-model-name", type=str, default="speechbrain_ecapa")
    parser.add_argument("--spkrec-source", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--spkrec-savedir", type=str, default="/workspace/project/pretrained_models/spkrec-ecapa-voxceleb")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    selected_splits = [part.strip() for part in str(args.splits).split(",") if part.strip()]
    result = build_enrollment_cache(
        enrollment_root=Path(args.enrollment_root),
        split_path=Path(args.speaker_split_path),
        cache_root=Path(args.cache_root),
        embedding_model=str(args.embedding_model_name),
        source=str(args.spkrec_source),
        savedir=str(args.spkrec_savedir),
        device=str(args.device),
        sample_rate=int(args.sample_rate),
        selected_splits=selected_splits,
        max_speakers=args.max_speakers,
        force=bool(args.force),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
