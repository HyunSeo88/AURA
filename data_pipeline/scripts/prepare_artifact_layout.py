from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.layout import prepare_run_layout


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a clean artifact layout for an AURA data pipeline run.")
    parser.add_argument("--artifact-root", type=str, default="/workspace/project/data_pipeline_artifacts")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--embedding-model-name", type=str, default="speechbrain_ecapa")
    parser.add_argument("--source-config", type=str, default=None)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    notes = {"note_lines": list(args.note)} if args.note else {}
    layout = prepare_run_layout(
        artifact_root=Path(args.artifact_root),
        run_name=str(args.run_name),
        embedding_model_name=str(args.embedding_model_name),
        source_config_path=Path(args.source_config) if args.source_config else None,
        notes=notes,
        exist_ok=bool(args.exist_ok),
    )
    print(json.dumps(layout.to_json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
