"""Convert SeaClear fold manifests into Ultralytics YOLO datasets without copying images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.seaclear import SeaClearDataset, iter_valid_boxes

CLASS_NAMES = ["debris", "bio", "robot"]


def coco_xywh_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    # clip
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    if nw <= 0 or nh <= 0:
        return None
    return cx, cy, nw, nh


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    try:
        dst.symlink_to(src.resolve())
        return "symlink"
    except OSError:
        try:
            import os

            os.link(src, dst)
            return "hardlink"
        except OSError:
            import shutil

            shutil.copy2(src, dst)
            return "copy"


def prepare_yolo_fold(
    ds: SeaClearDataset,
    manifest_csv: Path,
    out_root: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
    use_symlinks: bool = True,
) -> dict[str, Any]:
    """
    Write YOLO labels and image links for requested splits.

    Ultralytics data.yaml points at absolute image directories; labels live in
    parallel labels/ tree with matching basenames.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest_csv)
    anns_by_image = ds.anns_by_image
    images_by_id = {im.image_id: im for im in ds.images}

    link_stats: dict[str, int] = {}
    counts: dict[str, int] = {}

    for split in splits:
        sub = df[df["split"] == split]
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for _, row in sub.iterrows():
            image_id = int(row["image_id"])
            im = images_by_id.get(image_id)
            if im is None:
                continue
            src = Path(row["path"]) if Path(str(row["path"])).exists() else im.path
            if not src.exists():
                continue
            # Unique name to avoid collisions across folders
            stem = f"{im.site}_{im.camera}_{src.stem}".replace(" ", "_").replace("|", "_")
            # Keep extension
            dst_img = img_dir / f"{stem}{src.suffix.lower()}"
            if use_symlinks:
                mode = _link_or_copy(src, dst_img)
            else:
                import shutil

                dst_img.parent.mkdir(parents=True, exist_ok=True)
                if not dst_img.exists():
                    shutil.copy2(src, dst_img)
                mode = "copy"
            link_stats[mode] = link_stats.get(mode, 0) + 1

            # labels
            w, h = im.width, im.height
            if w <= 0 or h <= 0:
                # try to read
                try:
                    from PIL import Image

                    with Image.open(src) as pil:
                        w, h = pil.size
                except Exception:
                    w, h = 1920, 1080

            lines = []
            for a in iter_valid_boxes(anns_by_image.get(image_id, [])):
                yolo = coco_xywh_to_yolo(a.bbox, w, h)
                if yolo is None or a.mapped_id is None:
                    continue
                cx, cy, nw, nh = yolo
                lines.append(f"{a.mapped_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n += 1
        counts[split] = n

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    meta = {
        "out_root": str(out_root),
        "data_yaml": str(yaml_path),
        "counts": counts,
        "link_stats": link_stats,
        "class_names": CLASS_NAMES,
        "manifest": str(manifest_csv),
    }
    with open(out_root / "prepare_meta.json", "w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, indent=2)
    return meta
