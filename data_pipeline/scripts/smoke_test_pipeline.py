from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

from data_pipeline.dataset import AuraManifestDataset, aura_collate_fn
from data_pipeline.sampler import BucketAwareSpeakerSampler, SamplerRecord


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test AURA data pipeline v1 dataset + sampler + dataloader.")
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--speaker-registry-path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--enroll-k", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-sec", type=float, default=6.0)
    parser.add_argument("--epoch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260318)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = AuraManifestDataset(
        manifest_path=Path(args.manifest_path),
        speaker_registry_path=Path(args.speaker_registry_path),
        split=str(args.split),
        enroll_k=int(args.enroll_k),
        sample_rate=int(args.sample_rate),
        duration_sec=float(args.duration_sec),
        deterministic_eval_enroll=True,
        strict_registry=True,
        seed=int(args.seed),
    )
    records = [
        SamplerRecord(index=i, speaker_id=row.target_speaker_id, task_bucket=row.task_bucket)
        for i, row in enumerate(dataset.rows)
    ]
    sampler = BucketAwareSpeakerSampler(records, epoch_size=int(args.epoch_size), seed=int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=0,
        collate_fn=aura_collate_fn,
    )
    batch = next(iter(loader))
    bucket_counts = Counter(dataset.rows[idx].task_bucket for idx in list(iter(sampler)))
    payload = {
        "dataset_len": len(dataset),
        "validation_report": dataset.validation_report.__dict__,
        "sampler_plan": sampler.get_plan().__dict__,
        "sampled_bucket_counts": dict(bucket_counts),
        "batch_mix_shape": list(batch["mix"].shape),
        "batch_target_shape": list(batch["target"].shape),
        "batch_enroll_shape": list(batch["enroll_seq"].shape),
        "batch_enroll_mask_shape": list(batch["enroll_mask"].shape),
        "batch_enroll_mask_sum": [int(x) for x in batch["enroll_mask"].sum(dim=1).tolist()],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
