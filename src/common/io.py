"""Config loading, environment paths, and device resolution."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_device(device_cfg: str | int | None = "auto") -> str:
    """Return Ultralytics-compatible device string."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if device_cfg is None or device_cfg == "auto":
        if not torch.cuda.is_available():
            return "cpu"
        n = torch.cuda.device_count()
        if n >= 2:
            return "0,1"
        return "0"
    return str(device_cfg)


def discover_dataset_root(
    data_root: Path,
    candidates: list[str] | None,
    explicit: str | Path | None,
    markers: tuple[str, ...] = ("annotations", "images", "Annotations", "Images"),
) -> Path | None:
    """Find a dataset directory under data_root."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p

    if not data_root.exists():
        return None

    for name in candidates or []:
        cand = data_root / name
        if cand.exists():
            return cand

    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if any((child / m).exists() for m in markers):
            return child
        try:
            for sub in child.iterdir():
                if sub.is_dir() and any((sub / m).exists() for m in markers):
                    return sub
        except PermissionError:
            continue
    return None


@dataclass
class EnvPaths:
    name: str
    data_root: Path
    seaclear_root: Path | None
    trashcan_root: Path | None
    processed_root: Path
    manifests_root: Path
    reports_root: Path
    runs_root: Path
    paper_assets_root: Path
    device: str
    num_workers: int = 2
    pin_memory: bool = True

    def ensure_output_dirs(self) -> None:
        for p in (
            self.processed_root,
            self.manifests_root,
            self.reports_root,
            self.runs_root,
            self.paper_assets_root,
        ):
            p.mkdir(parents=True, exist_ok=True)


def load_env(env_name: str | None = None) -> EnvPaths:
    if env_name is None:
        env_name = "kaggle" if is_kaggle() else "local"
    cfg = load_yaml(PROJECT_ROOT / "configs" / "env" / f"{env_name}.yaml")

    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    seaclear = cfg.get("seaclear_root")
    trashcan = cfg.get("trashcan_root")
    if env_name == "kaggle" or is_kaggle():
        seaclear_path = discover_dataset_root(
            data_root,
            cfg.get("seaclear_candidates"),
            seaclear,
        )
        trashcan_path = discover_dataset_root(
            data_root,
            cfg.get("trashcan_candidates"),
            trashcan,
        )
    else:
        seaclear_path = Path(seaclear) if seaclear else None
        trashcan_path = Path(trashcan) if trashcan else None
        if seaclear_path and not seaclear_path.is_absolute():
            seaclear_path = PROJECT_ROOT / seaclear_path
        if trashcan_path and not trashcan_path.is_absolute():
            trashcan_path = PROJECT_ROOT / trashcan_path
        if seaclear_path and not seaclear_path.exists():
            seaclear_path = None
        if trashcan_path and not trashcan_path.exists():
            trashcan_path = None

    def _out(key: str) -> Path:
        p = Path(cfg[key])
        return p if p.is_absolute() else PROJECT_ROOT / p

    return EnvPaths(
        name=cfg.get("name", env_name),
        data_root=data_root,
        seaclear_root=seaclear_path,
        trashcan_root=trashcan_path,
        processed_root=_out("processed_root"),
        manifests_root=_out("manifests_root"),
        reports_root=_out("reports_root"),
        runs_root=_out("runs_root"),
        paper_assets_root=_out("paper_assets_root"),
        device=resolve_device(cfg.get("device", "auto")),
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=bool(cfg.get("pin_memory", True)),
    )
