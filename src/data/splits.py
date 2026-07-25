"""Leakage-resistant LOSO splits by site, grouped by camera/sequence."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.io import load_yaml, PROJECT_ROOT
from src.common.run import save_json
from src.data.seaclear import ImageRecord, SeaClearDataset


SPLIT_NAMES = ("train", "gate", "val", "test")
# train = detector_train, gate = gate_train, val = calibration/validation, test = held-out site


def _stable_hash(s: str, seed: int) -> int:
    h = hashlib.md5(f"{seed}::{s}".encode()).hexdigest()
    return int(h[:8], 16)


def assign_group_split(group_id: str, seed: int, ratios: dict[str, float]) -> str:
    """Deterministic bucket into train/gate/val for source groups."""
    r_train = ratios.get("detector_train", 0.70)
    r_gate = ratios.get("gate_train", 0.15)
    # remainder -> calibration/val
    u = (_stable_hash(group_id, seed) % 10_000) / 10_000.0
    if u < r_train:
        return "train"
    if u < r_train + r_gate:
        return "gate"
    return "val"


def build_loso_fold(
    ds: SeaClearDataset,
    held_out_site: str,
    seed: int = 0,
    ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    ratios = ratios or {"detector_train": 0.70, "gate_train": 0.15, "calibration": 0.15}
    rows = []
    for im in ds.images:
        if im.site.lower() == held_out_site.lower():
            split = "test"
        else:
            split = assign_group_split(im.group_id, seed, ratios)
        rows.append(
            {
                "image_id": im.image_id,
                "file_name": im.file_name,
                "path": str(im.path),
                "site": im.site,
                "camera": im.camera,
                "sequence": im.sequence,
                "group_id": im.group_id,
                "width": im.width,
                "height": im.height,
                "split": split,
                "held_out_site": held_out_site,
                "fold": held_out_site,
            }
        )
    return pd.DataFrame(rows)


def contamination_check(df: pd.DataFrame) -> dict[str, Any]:
    """Ensure no group_id appears in more than one split."""
    g = df.groupby("group_id")["split"].nunique()
    bad = g[g > 1]
    # Also: test site images must all be test
    test_site = df["held_out_site"].iloc[0] if len(df) else None
    leakage_to_source = 0
    if test_site:
        leakage_to_source = int(
            ((df["site"].str.lower() == str(test_site).lower()) & (df["split"] != "test")).sum()
        )
    return {
        "n_contaminated_groups": int(len(bad)),
        "contaminated_groups_sample": bad.head(20).to_dict(),
        "test_site_leakage_count": leakage_to_source,
        "pass": len(bad) == 0 and leakage_to_source == 0,
        "split_counts": df["split"].value_counts().to_dict(),
        "site_by_split": df.groupby(["split", "site"]).size().to_dict(),
    }


def write_fold_manifests(
    df: pd.DataFrame,
    manifests_root: Path,
    held_out_site: str,
) -> dict[str, Path]:
    manifests_root.mkdir(parents=True, exist_ok=True)
    fold_dir = manifests_root / f"loso_{held_out_site.lower()}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    full = fold_dir / "manifest.csv"
    df.to_csv(full, index=False)
    paths["full"] = full
    for split in ("train", "gate", "val", "test"):
        p = fold_dir / f"{split}.csv"
        df[df["split"] == split].to_csv(p, index=False)
        paths[split] = p
    check = contamination_check(df)
    save_json(fold_dir / "contamination_check.json", check)
    if not check["pass"]:
        raise RuntimeError(f"Split contamination detected for fold {held_out_site}: {check}")
    return paths


def build_all_loso(
    ds: SeaClearDataset,
    sites: list[str] | None = None,
    seed: int = 0,
    ratios: dict[str, float] | None = None,
    manifests_root: Path | None = None,
) -> dict[str, Any]:
    manifests_root = manifests_root or (PROJECT_ROOT / "data" / "manifests")
    if sites is None:
        cfg = load_yaml(PROJECT_ROOT / "configs" / "data" / "seaclear.yaml")
        sites = cfg.get("sites")
    present = sorted({im.site for im in ds.images if im.site != "Unknown"})
    sites = [s for s in (sites or present) if s in present]
    if not sites:
        sites = present

    summary = {"folds": {}, "sites_used": sites}
    for site in sites:
        df = build_loso_fold(ds, site, seed=seed, ratios=ratios)
        paths = write_fold_manifests(df, manifests_root, site)
        summary["folds"][site] = {
            "manifest": str(paths["full"]),
            "split_counts": df["split"].value_counts().to_dict(),
            "n_images": len(df),
        }
    save_json(manifests_root / "loso_summary.json", summary)
    return summary
