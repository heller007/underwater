"""E4: mixed-path dataset — one deterministic action per image from a shortlist."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.common.quiet import progress
from src.detection.naive_enhance import _copy_or_link
from src.enhancement.transforms import get_transform


def _pick_action(stem: str, actions: list[str], seed: int) -> str:
    h = hashlib.md5(f"{seed}::{stem}".encode()).hexdigest()
    idx = int(h[:8], 16) % len(actions)
    return actions[idx]


def materialize_mixed_dataset(
    src_yolo_root: Path,
    out_root: Path,
    actions: list[str],
    seed: int = 0,
    splits: tuple[str, ...] = ("train", "val", "test"),
    progress_every: int = 200,
) -> tuple[Path, dict[str, Any]]:
    """
    For each image, assign one action uniformly (deterministic) and write T_k(x).
    Labels are linked/copied unchanged.
    Returns (data_yaml_path, assignment_stats).
    """
    src_yolo_root = Path(src_yolo_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    transforms = {a: get_transform(a) for a in actions}

    with open(src_yolo_root / "data.yaml", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    stats: dict[str, Any] = {"actions": actions, "seed": seed, "splits": {}}
    assignments_path = out_root / "action_assignments.csv"
    assign_rows = ["split,stem,action"]

    for split in splits:
        src_img = src_yolo_root / "images" / split
        src_lbl = src_yolo_root / "labels" / split
        dst_img = out_root / "images" / split
        dst_lbl = out_root / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        counts = {a: 0 for a in actions}
        if not src_img.exists():
            stats["splits"][split] = counts
            continue
        images = sorted(
            p
            for p in src_img.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        n = len(images)
        progress(f"[mixed] {split}: {n} images, actions={actions}")
        for i, img_path in enumerate(images, 1):
            action = _pick_action(img_path.stem, actions, seed)
            counts[action] += 1
            assign_rows.append(f"{split},{img_path.stem},{action}")
            out_img = dst_img / img_path.name
            lbl_src = src_lbl / f"{img_path.stem}.txt"
            lbl_dst = dst_lbl / f"{img_path.stem}.txt"
            if not out_img.exists():
                if action == "T0":
                    _copy_or_link(img_path, out_img)
                else:
                    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    enh = transforms[action](bgr)
                    cv2.imwrite(str(out_img), enh)
            if lbl_src.exists():
                _copy_or_link(lbl_src, lbl_dst)
            else:
                lbl_dst.write_text("", encoding="utf-8")
            if i % progress_every == 0 or i == n:
                progress(f"[mixed] {split}: {i}/{n}")
        stats["splits"][split] = counts

    assignments_path.write_text("\n".join(assign_rows) + "\n", encoding="utf-8")
    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": data_cfg.get("names", {0: "debris", 1: "bio", 2: "robot"}),
        "nc": data_cfg.get("nc", 3),
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    with open(out_root / "mixed_stats.json", "w", encoding="utf-8") as f:
        import json

        json.dump(stats, f, indent=2)
    progress(f"[mixed] ready: {yaml_path}")
    return yaml_path, stats
