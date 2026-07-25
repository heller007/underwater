"""Seeding, run folders, and reproducibility manifests."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .io import PROJECT_ROOT, is_kaggle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_revision() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "is_kaggle": is_kaggle(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        info["gpu_names"] = (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
    except ImportError:
        info["torch"] = None
    return info


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    seed: int
    experiment_id: str
    config: dict[str, Any] = field(default_factory=dict)
    env_name: str = "local"

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)


def create_run(
    experiment_id: str,
    seed: int,
    config: dict[str, Any],
    runs_root: Path | None = None,
    fold: str | None = None,
    model_tag: str = "raw",
) -> RunContext:
    runs_root = runs_root or (PROJECT_ROOT / "runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fold_part = (fold or "nofold").lower().replace(" ", "-")
    run_id = f"fold-{fold_part}_model-{model_tag}_exp-{experiment_id}_seed-{seed}_{ts}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "seed": seed,
        "fold": fold,
        "model_tag": model_tag,
        "created_utc": ts,
        "git_revision": git_revision(),
        "hardware": hardware_info(),
        "config": config,
        "argv": sys.argv,
    }
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        seed=seed,
        experiment_id=experiment_id,
        config=config,
        env_name=config.get("env_name", "local"),
    )


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def setup_logging(run_dir: Path | None = None, level: str = "INFO") -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(run_dir / "run.log", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("underwater")
