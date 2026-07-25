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


def _looks_like_coco_json(path: Path, max_bytes: int = 2_000_000) -> bool:
    """Cheap check that a JSON file is COCO-format (has images + annotations)."""
    try:
        import json

        with open(path, encoding="utf-8") as f:
            # Read a prefix first for huge files; fall back to full load if needed
            head = f.read(max_bytes)
            if '"images"' not in head or '"annotations"' not in head:
                # might be beyond prefix — try full parse for modest files
                if path.stat().st_size > max_bytes:
                    f.seek(0)
                    data = json.load(f)
                else:
                    return False
            else:
                f.seek(0)
                data = json.load(f)
        return isinstance(data, dict) and "images" in data and "annotations" in data
    except Exception:
        return False


def find_coco_json_under(root: Path, max_files: int = 200) -> Path | None:
    """Search for a COCO annotation JSON under root (breadth-biased)."""
    if not root.exists():
        return None
    preferred_names = ("instances", "annotation", "coco", "labels")
    json_files: list[Path] = []
    try:
        for p in root.rglob("*.json"):
            json_files.append(p)
            if len(json_files) >= max_files:
                break
    except PermissionError:
        return None

    def score(p: Path) -> tuple[int, int]:
        name = p.name.lower()
        pref = 0 if any(k in name for k in preferred_names) else 1
        return (pref, len(p.parts))

    for p in sorted(json_files, key=score):
        if _looks_like_coco_json(p):
            return p
    return None


def discover_dataset_root(
    data_root: Path,
    candidates: list[str] | None,
    explicit: str | Path | None,
    markers: tuple[str, ...] = ("annotations", "images", "Annotations", "Images"),
    require_coco: bool = True,
) -> Path | None:
    """
    Find a dataset directory under data_root.

    SeaClear on Kaggle is often organized as site-camera folders + a COCO JSON,
    without top-level `images/` or `annotations/` dirs — so we also search for COCO.
    """
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p

    if not data_root.exists():
        return None

    # 1) Named candidate folders (verify COCO when required)
    for name in candidates or []:
        cand = data_root / name
        if not cand.exists():
            continue
        if not require_coco:
            return cand
        if find_coco_json_under(cand) is not None:
            return cand

    # 2) Any input folder whose name mentions seaclear / trashcan keywords
    keywords = []
    for name in candidates or []:
        keywords.append(name.lower().replace("-", "").replace("_", ""))
    keywords.extend(["seaclear", "trashcan", "marine debris", "marinedebris"])

    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        key = child.name.lower().replace("-", "").replace("_", "")
        if any(k.replace("-", "").replace("_", "") in key for k in keywords if k):
            if (not require_coco) or find_coco_json_under(child) is not None:
                return child

    # 3) Classic marker folders
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if any((child / m).exists() for m in markers):
            if (not require_coco) or find_coco_json_under(child) is not None:
                return child
        try:
            for sub in child.iterdir():
                if sub.is_dir() and any((sub / m).exists() for m in markers):
                    if (not require_coco) or find_coco_json_under(sub) is not None:
                        return sub
        except PermissionError:
            continue

    # 4) Last resort: any COCO JSON under data_root → use that JSON's parent tree root
    coco = find_coco_json_under(data_root)
    if coco is not None:
        # Prefer the Kaggle dataset slug dir (/kaggle/input/<slug>/...)
        parts = coco.parts
        if "input" in parts:
            idx = parts.index("input")
            if idx + 1 < len(parts):
                return Path(*parts[: idx + 2])
        return coco.parent

    return None


def list_input_datasets(data_root: Path | None = None) -> list[str]:
    """Human-readable listing of /kaggle/input (or local data_root) for error messages."""
    root = data_root or Path("/kaggle/input")
    if not root.exists():
        return [f"(missing) {root}"]
    lines = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            n_json = sum(1 for _ in child.rglob("*.json"))
            n_img = sum(
                1
                for _ in child.rglob("*")
                if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            lines.append(f"{child}  (json≈{n_json}, images≈{n_img})")
        else:
            lines.append(str(child))
    return lines or [f"(empty) {root}"]


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
        # Prefer env override from notebook: SEACLEAR_ROOT / TRASHCAN_ROOT
        seaclear = seaclear or os.environ.get("SEACLEAR_ROOT")
        trashcan = trashcan or os.environ.get("TRASHCAN_ROOT")
        seaclear_path = discover_dataset_root(
            data_root,
            cfg.get("seaclear_candidates"),
            seaclear,
            require_coco=True,
        )
        trashcan_path = discover_dataset_root(
            data_root,
            cfg.get("trashcan_candidates"),
            trashcan,
            require_coco=True,
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
