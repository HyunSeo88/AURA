import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def create_experiment_dirs(project_root: str = "/workspace/project", exp_name: Optional[str] = None) -> Dict[str, Path]:
    root = Path(project_root)
    experiments_root = root / "experiments"
    if exp_name is None:
        exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = experiments_root / exp_name
    checkpoints_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    config_dir = run_dir / "config"
    logs_dir = run_dir / "logs"

    for p in [run_dir, checkpoints_dir, metrics_dir, config_dir, logs_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "experiments_root": experiments_root,
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "metrics_dir": metrics_dir,
        "config_dir": config_dir,
        "logs_dir": logs_dir,
        "exp_name": Path(exp_name),
    }


def setup_experiment_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("aura_train")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_experiment_config(config_path: Path, config: Dict):
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def append_epoch_metrics(jsonl_path: Path, csv_path: Path, row: Dict):
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
