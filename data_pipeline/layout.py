from __future__ import annotations

import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_pipeline.common import dump_json


RUN_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class PipelineRunLayout:
    artifact_root: str
    run_name: str
    run_root: str
    config_dir: str
    logs_dir: str
    canonical_dir: str
    speaker_stats_dir: str
    speaker_split_dir: str
    cache_root: str
    cache_embedding_dir: str
    registry_dir: str
    manifests_dir: str
    audit_dir: str
    reports_dir: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedRunContext:
    version: str
    generated_at_utc: str
    run_name: str
    embedding_model_name: str
    source_config_snapshot: str | None
    notes: dict[str, Any]
    layout: PipelineRunLayout

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layout"] = self.layout.to_json()
        return payload


def sanitize_run_name(run_name: str) -> str:
    normalized = RUN_NAME_PATTERN.sub("_", str(run_name).strip())
    normalized = normalized.strip("._-")
    if not normalized:
        raise ValueError("run_name must contain at least one alphanumeric character")
    return normalized


def build_run_layout(artifact_root: Path, run_name: str, embedding_model_name: str) -> PipelineRunLayout:
    clean_run_name = sanitize_run_name(run_name)
    clean_embedding_name = sanitize_run_name(embedding_model_name)
    run_root = artifact_root / "runs" / clean_run_name
    return PipelineRunLayout(
        artifact_root=str(artifact_root.resolve()),
        run_name=clean_run_name,
        run_root=str(run_root.resolve()),
        config_dir=str((run_root / "config").resolve()),
        logs_dir=str((run_root / "logs").resolve()),
        canonical_dir=str((run_root / "canonical").resolve()),
        speaker_stats_dir=str((run_root / "speaker_stats").resolve()),
        speaker_split_dir=str((run_root / "speaker_split").resolve()),
        cache_root=str((run_root / "cache").resolve()),
        cache_embedding_dir=str((run_root / "cache" / clean_embedding_name).resolve()),
        registry_dir=str((run_root / "registry").resolve()),
        manifests_dir=str((run_root / "manifests").resolve()),
        audit_dir=str((run_root / "audit").resolve()),
        reports_dir=str((run_root / "reports").resolve()),
    )


def prepare_run_layout(
    artifact_root: Path,
    run_name: str,
    *,
    embedding_model_name: str,
    source_config_path: Path | None = None,
    notes: dict[str, Any] | None = None,
    exist_ok: bool = False,
) -> PipelineRunLayout:
    artifact_root = artifact_root.resolve()
    layout = build_run_layout(artifact_root, run_name, embedding_model_name)
    run_root = Path(layout.run_root)
    if run_root.exists() and not exist_ok:
        raise FileExistsError(f"run root already exists: {run_root}")

    for path in (
        layout.config_dir,
        layout.logs_dir,
        layout.canonical_dir,
        layout.speaker_stats_dir,
        layout.speaker_split_dir,
        layout.cache_root,
        layout.cache_embedding_dir,
        layout.registry_dir,
        layout.manifests_dir,
        layout.audit_dir,
        layout.reports_dir,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)

    source_snapshot = None
    if source_config_path is not None:
        source_config_path = source_config_path.resolve()
        snapshot_path = Path(layout.config_dir) / "source_config.snapshot.json"
        shutil.copy2(source_config_path, snapshot_path)
        source_snapshot = str(snapshot_path.resolve())

    context = PreparedRunContext(
        version="v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_name=layout.run_name,
        embedding_model_name=str(embedding_model_name),
        source_config_snapshot=source_snapshot,
        notes=dict(notes or {}),
        layout=layout,
    )
    dump_json(Path(layout.config_dir) / "run_context.json", context.to_json())
    dump_json(Path(layout.reports_dir) / "layout_summary.json", layout.to_json())
    return layout
