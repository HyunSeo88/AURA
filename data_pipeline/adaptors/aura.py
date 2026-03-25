from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from data_pipeline.adaptors.base import BaseSourceAdaptor
from data_pipeline.common import CanonicalSample, iter_jsonl, normalize_noise_types


class AuraSourceAdaptor(BaseSourceAdaptor):
    adaptor_name = "aura"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_dir = self._resolve_target_dir()

    def _resolve_target_dir(self) -> Path:
        for candidate in (self.spec.root / "target", self.spec.root / "clean"):
            if candidate.exists():
                return candidate
        return self.spec.root / "target"

    def _target_speaker_id(self, obj: Dict[str, Any]) -> Optional[str]:
        target = obj.get("target")
        if isinstance(target, dict):
            speaker_id = target.get("speaker_id")
            if speaker_id:
                return str(speaker_id)
        for key in ("speaker_id", "target_speaker_id"):
            value = obj.get(key)
            if value:
                return str(value)
        return None

    def _target_loc(self, obj: Dict[str, Any]):
        target = obj.get("target")
        if isinstance(target, dict):
            loc = target.get("loc")
            if isinstance(loc, list):
                return loc
        locations = obj.get("locations")
        if isinstance(locations, dict):
            loc = locations.get("teacher") or locations.get("target")
            if isinstance(loc, list):
                return loc
        return None

    def _num_interferers(self, obj: Dict[str, Any]) -> int:
        interferers = obj.get("interferers")
        if isinstance(interferers, list):
            return len(interferers)
        return 0

    def _task_bucket(self, num_interferers: int) -> str:
        mapping = {0: "1spk", 1: "2spk", 2: "3spk"}
        return mapping.get(num_interferers, "unsupported")

    def _noise_types(self, obj: Dict[str, Any]) -> list[str]:
        if "noises" in obj:
            return normalize_noise_types(obj.get("noises"))
        if "noise_sources" in obj:
            return normalize_noise_types(obj.get("noise_sources"))
        return []

    def iter_samples(self) -> Iterator[CanonicalSample]:
        meta_files = list(self.meta_files())
        self.stats.meta_files_seen = len(meta_files)
        if not meta_files:
            self.stats.errors.append("no meta files found")
            return

        seen_ids: set[str] = set()
        for meta_path in meta_files:
            for line_no, line in iter_jsonl(meta_path):
                stripped = line.strip()
                if not stripped:
                    continue
                self.stats.rows_seen += 1
                if not stripped.startswith("{"):
                    self.stats.malformed_meta_lines += 1
                    self.stats.dropped_rows += 1
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    self.stats.malformed_meta_lines += 1
                    self.stats.dropped_rows += 1
                    continue

                sample_id = str(obj.get("id") or "").strip()
                if not sample_id:
                    self.stats.dropped_rows += 1
                    self.stats.split_ineligible_reasons["missing_id"] += 1
                    continue
                if sample_id in seen_ids:
                    self.stats.duplicate_sample_ids += 1
                    self.stats.dropped_rows += 1
                    self.stats.split_ineligible_reasons["duplicate_id"] += 1
                    continue
                seen_ids.add(sample_id)

                target_speaker_id = self._target_speaker_id(obj)
                num_interferers = self._num_interferers(obj)
                task_bucket = self._task_bucket(num_interferers)
                mix_path = self.spec.root / "mix" / f"{sample_id}.wav"
                target_path = self.target_dir / f"{sample_id}.wav"

                split_eligible = True
                reason = None
                if not target_speaker_id:
                    split_eligible = False
                    reason = "missing_target_speaker_id"
                elif task_bucket == "unsupported":
                    split_eligible = False
                    reason = f"unsupported_num_interferers_{num_interferers}"
                elif not self._path_exists(mix_path):
                    split_eligible = False
                    reason = "missing_mix_path"
                elif not self._path_exists(target_path):
                    split_eligible = False
                    reason = "missing_target_path"

                sample = CanonicalSample(
                    id=sample_id,
                    source_dataset=self.spec.name,
                    task_bucket=task_bucket,
                    num_interferers=num_interferers,
                    num_speakers_total=num_interferers + 1,
                    target_speaker_id=str(target_speaker_id or "unknown"),
                    mix_path=str(mix_path.resolve()),
                    target_path=str(target_path.resolve()),
                    scenario=str(obj.get("scenario") or "unknown"),
                    noise_types=self._noise_types(obj),
                    sample_rate=self.expected_sample_rate,
                    num_samples=self.expected_num_samples,
                    target_type="clean_direct",
                    mix_type="reverb_noise_mix",
                    split_eligible=split_eligible,
                    split_ineligible_reason=reason,
                    room_dim=obj.get("room_dim"),
                    mic_loc=obj.get("mic_loc") or (obj.get("locations") or {}).get("mic"),
                    target_loc=self._target_loc(obj),
                    raw_meta_path=str(meta_path.resolve()),
                    raw_meta_line=line_no,
                )
                self.stats.rows_emitted += 1
                self.stats.task_bucket_counts[sample.task_bucket] += 1
                if split_eligible:
                    self.stats.split_eligible_rows += 1
                else:
                    self.stats.split_ineligible_rows += 1
                    if reason is not None:
                        self.stats.split_ineligible_reasons[reason] += 1
                yield sample
