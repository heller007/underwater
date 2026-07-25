"""Dataset audit: statistics, duplicates, invalid annotations."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.run import save_json
from src.data.seaclear import SeaClearDataset, iter_valid_boxes


def _file_md5(path: Path, chunk: int = 1 << 20) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def _phash(path: Path) -> str | None:
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        return None
    if not path.exists():
        return None
    try:
        with Image.open(path) as im:
            return str(imagehash.phash(im))
    except Exception:
        return None


def audit_dataset(
    ds: SeaClearDataset,
    reports_root: Path,
    compute_hashes: bool = True,
    max_hash_images: int | None = 2000,
) -> dict[str, Any]:
    reports_root.mkdir(parents=True, exist_ok=True)
    anns_by_image = ds.anns_by_image

    site_counts: Counter[str] = Counter()
    camera_counts: Counter[str] = Counter()
    site_camera_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    box_areas: list[float] = []
    box_aspects: list[float] = []
    obj_counts: list[int] = []
    missing_files = 0
    empty_frames = 0
    invalid_boxes = 0
    ignored_anns = 0

    domain_rows = []
    for im in ds.images:
        site_counts[im.site] += 1
        camera_counts[im.camera] += 1
        sc = f"{im.site}|{im.camera}"
        site_camera_counts[sc] += 1
        exists = im.path.exists()
        if not exists:
            missing_files += 1
        anns = anns_by_image.get(im.image_id, [])
        valid = list(iter_valid_boxes(anns))
        ignored_anns += sum(1 for a in anns if a.mapped_id is None)
        for a in anns:
            x, y, w, h = (a.bbox + [0, 0, 0, 0])[:4]
            if w <= 0 or h <= 0:
                invalid_boxes += 1
        if not valid:
            empty_frames += 1
        obj_counts.append(len(valid))
        for a in valid:
            class_counts[a.mapped_class or "none"] += 1
            area = a.bbox[2] * a.bbox[3]
            box_areas.append(area)
            box_aspects.append(a.bbox[2] / a.bbox[3] if a.bbox[3] > 0 else 0.0)
        domain_rows.append(
            {
                "image_id": im.image_id,
                "file_name": im.file_name,
                "path": str(im.path),
                "exists": exists,
                "site": im.site,
                "camera": im.camera,
                "sequence": im.sequence,
                "group_id": im.group_id,
                "width": im.width,
                "height": im.height,
                "n_valid_boxes": len(valid),
                "n_raw_anns": len(anns),
            }
        )

    exact_dup_groups: dict[str, list[int]] = defaultdict(list)
    phash_groups: dict[str, list[int]] = defaultdict(list)
    if compute_hashes:
        to_hash = ds.images[: max_hash_images or len(ds.images)]
        for im in to_hash:
            md5 = _file_md5(im.path)
            if md5:
                exact_dup_groups[md5].append(im.image_id)
            ph = _phash(im.path)
            if ph:
                phash_groups[ph].append(im.image_id)

    exact_dups = {k: v for k, v in exact_dup_groups.items() if len(v) > 1}
    near_dups = {k: v for k, v in phash_groups.items() if len(v) > 1}

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {}
        s = pd.Series(vals)
        return {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
        }

    audit = {
        "n_images": len(ds.images),
        "n_annotations_raw": len(ds.annotations),
        "n_annotations_mapped": int(sum(class_counts.values())),
        "n_ignored_annotations": ignored_anns,
        "n_invalid_boxes": invalid_boxes,
        "n_empty_frames": empty_frames,
        "n_missing_files": missing_files,
        "sites": dict(site_counts),
        "cameras": dict(camera_counts),
        "site_camera": dict(site_camera_counts),
        "class_counts": dict(class_counts),
        "objects_per_image": _stats([float(x) for x in obj_counts]),
        "box_area": _stats(box_areas),
        "box_aspect": _stats(box_aspects),
        "n_exact_duplicate_groups": len(exact_dups),
        "n_phash_duplicate_groups": len(near_dups),
        "exact_duplicate_groups_sample": {
            k: v for i, (k, v) in enumerate(exact_dups.items()) if i < 20
        },
        "phash_duplicate_groups_sample": {
            k: v for i, (k, v) in enumerate(near_dups.items()) if i < 20
        },
        "class_map_report": ds.class_map_report,
        "pass": missing_files == 0 and len(ds.images) > 0,
        "warnings": [],
    }
    if missing_files:
        audit["warnings"].append(f"{missing_files} image paths missing on disk")
    if "Unknown" in site_counts:
        audit["warnings"].append(f"{site_counts['Unknown']} images with Unknown site")
    if empty_frames / max(len(ds.images), 1) > 0.5:
        audit["warnings"].append("More than 50% empty frames after mapping")

    save_json(reports_root / "data_audit.json", audit)
    save_json(reports_root / "class_map.json", ds.class_map_report)

    df = pd.DataFrame(domain_rows)
    domain_manifest = (
        df.groupby(["site", "camera", "sequence"], dropna=False)
        .agg(
            n_images=("image_id", "count"),
            n_boxes=("n_valid_boxes", "sum"),
            missing=("exists", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    domain_manifest.to_csv(reports_root / "domain_manifest.csv", index=False)
    df.to_csv(reports_root / "image_manifest.csv", index=False)
    return audit
