"""SeaClear COCO ingestion, domain parsing, and class mapping."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.common.io import load_yaml, PROJECT_ROOT


SITE_ALIASES = {
    "bistrina": "Bistrina",
    "jakljan": "Jakljan",
    "lokrum": "Lokrum",
    "slano": "Slano",
    "marseille": "Marseille",
}

SUPER_IDS = {"debris": 0, "bio": 1, "robot": 2}


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    path: Path
    width: int
    height: int
    site: str
    camera: str
    sequence: str
    group_id: str
    split_hint: str | None = None


@dataclass
class AnnotationRecord:
    ann_id: int
    image_id: int
    category_id: int
    category_name: str
    supercategory: str | None
    mapped_class: str | None  # debris/bio/robot or None if ignored
    mapped_id: int | None
    bbox: list[float]  # COCO xywh
    area: float
    iscrowd: int = 0


@dataclass
class SeaClearDataset:
    root: Path
    images: list[ImageRecord]
    annotations: list[AnnotationRecord]
    categories: list[dict[str, Any]]
    class_map_report: dict[str, Any] = field(default_factory=dict)

    @property
    def anns_by_image(self) -> dict[int, list[AnnotationRecord]]:
        out: dict[int, list[AnnotationRecord]] = defaultdict(list)
        for a in self.annotations:
            out[a.image_id].append(a)
        return dict(out)


def find_coco_json(root: Path) -> Path:
    candidates = list(root.rglob("*.json"))
    preferred = []
    for p in candidates:
        name = p.name.lower()
        if "coco" in name or "annotation" in name or "instances" in name:
            preferred.append(p)
    search = preferred or candidates
    for p in search:
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "images" in data and "annotations" in data:
                return p
        except Exception:
            continue
    raise FileNotFoundError(f"No COCO annotation JSON found under {root}")


def find_image_path(root: Path, file_name: str, cache: dict[str, Path] | None = None) -> Path | None:
    """Resolve image path; builds a basename index on first miss."""
    direct = root / file_name
    if direct.exists():
        return direct
    # Often file_name is relative like Site/cam/img.jpg
    parts = Path(file_name)
    if (root / parts).exists():
        return root / parts

    if cache is None:
        return _scan_for_file(root, Path(file_name).name)
    if not cache:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                cache[p.name] = p
    return cache.get(Path(file_name).name)


def _scan_for_file(root: Path, basename: str) -> Path | None:
    for p in root.rglob(basename):
        if p.is_file():
            return p
    return None


def parse_domain_from_path(path: Path, file_name: str) -> tuple[str, str, str]:
    """Return (site, camera, sequence) from path / filename heuristics."""
    text = str(path).replace("\\", "/")
    parts = [p for p in Path(file_name).parts] + [p for p in path.parts]

    site = "Unknown"
    for part in parts:
        key = re.sub(r"[^a-zA-Z]", "", part).lower()
        if key in SITE_ALIASES:
            site = SITE_ALIASES[key]
            break
        for alias, canon in SITE_ALIASES.items():
            if alias in part.lower():
                site = canon
                break

    camera = "unknown_cam"
    for part in parts:
        low = part.lower()
        if any(tok in low for tok in ("cam", "blue", "rov", "gopro", "tortuga", "camera")):
            camera = part
            break

    # Sequence: parent folder if not site
    sequence = path.parent.name if path.parent.name else "seq0"
    if sequence.lower() == site.lower():
        sequence = path.stem.rsplit("_", 1)[0] if "_" in path.stem else path.stem

    return site, camera, sequence


def load_class_map(path: str | Path | None = None) -> dict[str, Any]:
    path = path or (PROJECT_ROOT / "configs" / "data" / "class_map.yaml")
    return load_yaml(path)


def map_category(
    cat: dict[str, Any],
    class_map: dict[str, Any],
) -> tuple[str | None, int | None, str]:
    """Map a COCO category to supercategory. Returns (name, id, reason)."""
    name = str(cat.get("name", "")).strip()
    name_l = name.lower().replace("-", "_").replace(" ", "_")
    super_raw = str(cat.get("supercategory", "") or "").strip().lower()
    ignore = {x.lower() for x in class_map.get("ignore", [])}

    if name_l in ignore or super_raw in ignore:
        return None, None, f"ignored:{name}"

    fine = class_map.get("fine_to_super", {})
    if name_l in fine:
        mapped = fine[name_l]
        return mapped, SUPER_IDS[mapped], f"fine_map:{name}->{mapped}"

    sc_map = class_map.get("seaclear_supercategory_map", {})
    if super_raw in sc_map:
        mapped = sc_map[super_raw]
        return mapped, SUPER_IDS[mapped], f"supercategory:{super_raw}->{mapped}"

    # Heuristic keywords
    debris_kw = ("plastic", "bottle", "can", "bag", "tire", "rope", "net", "litter", "trash", "debris", "waste", "glass", "metal", "cloth", "wood", "paper", "styrofoam", "foam", "wrapper", "package")
    bio_kw = ("fish", "crab", "urchin", "seagrass", "algae", "plant", "animal", "bio", "starfish", "sponge", "coral", "vegetation", "seaweed")
    robot_kw = ("robot", "rov", "manipulator", "tether", "gripper", "arm", "vehicle", "cable")

    for kw in debris_kw:
        if kw in name_l or kw in super_raw:
            return "debris", 0, f"heuristic_debris:{name}"
    for kw in bio_kw:
        if kw in name_l or kw in super_raw:
            return "bio", 1, f"heuristic_bio:{name}"
    for kw in robot_kw:
        if kw in name_l or kw in super_raw:
            return "robot", 2, f"heuristic_robot:{name}"

    return None, None, f"unmapped_ignored:{name}"


def load_seaclear(
    root: Path,
    class_map: dict[str, Any] | None = None,
    max_images: int | None = None,
    held_out_site: str | None = None,
) -> SeaClearDataset:
    root = Path(root)
    class_map = class_map or load_class_map()
    coco_path = find_coco_json(root)
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    cats = {c["id"]: c for c in coco.get("categories", [])}
    map_reasons: dict[str, str] = {}
    cat_mapped: dict[int, tuple[str | None, int | None]] = {}
    for cid, cat in cats.items():
        mapped_name, mapped_id, reason = map_category(cat, class_map)
        cat_mapped[cid] = (mapped_name, mapped_id)
        map_reasons[str(cat.get("name"))] = reason

    # Build lightweight records first so max_images can be site-stratified
    raw_images = list(coco.get("images", []))
    prelim: list[tuple[dict[str, Any], str, str, str]] = []
    for im in raw_images:
        file_name = im["file_name"]
        # Domain from file_name before path resolve (faster / works offline)
        site, camera, sequence = parse_domain_from_path(Path(file_name), file_name)
        prelim.append((im, site, camera, sequence))

    if max_images is not None and max_images < len(prelim):
        by_site: dict[str, list[tuple]] = defaultdict(list)
        for item in prelim:
            by_site[item[1]].append(item)
        sites = sorted(by_site.keys())
        # Ensure held-out site is represented when present
        if held_out_site:
            for s in sites:
                if s.lower() == held_out_site.lower():
                    # move to front so it gets quota
                    sites = [s] + [x for x in sites if x != s]
                    break
        selected: list[tuple] = []
        # Round-robin until max_images
        idxs = {s: 0 for s in sites}
        while len(selected) < max_images and any(idxs[s] < len(by_site[s]) for s in sites):
            for s in sites:
                if len(selected) >= max_images:
                    break
                i = idxs[s]
                if i < len(by_site[s]):
                    selected.append(by_site[s][i])
                    idxs[s] = i + 1
        prelim = selected

    path_cache: dict[str, Path] = {}
    images: list[ImageRecord] = []
    for im, site, camera, sequence in prelim:
        file_name = im["file_name"]
        img_path = find_image_path(root, file_name, path_cache)
        if img_path is None:
            img_path = root / file_name
        # Re-parse from resolved path (more accurate)
        site, camera, sequence = parse_domain_from_path(img_path, file_name)
        group_id = f"{site}|{camera}|{sequence}"
        images.append(
            ImageRecord(
                image_id=int(im["id"]),
                file_name=file_name,
                path=img_path,
                width=int(im.get("width", 0) or 0),
                height=int(im.get("height", 0) or 0),
                site=site,
                camera=camera,
                sequence=sequence,
                group_id=group_id,
            )
        )

    keep_ids = {im.image_id for im in images}
    annotations: list[AnnotationRecord] = []
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in keep_ids:
            continue
        cid = int(ann["category_id"])
        cat = cats.get(cid, {"name": str(cid), "supercategory": None})
        mapped_name, mapped_id = cat_mapped.get(cid, (None, None))
        bbox = [float(x) for x in ann.get("bbox", [0, 0, 0, 0])]
        annotations.append(
            AnnotationRecord(
                ann_id=int(ann["id"]),
                image_id=image_id,
                category_id=cid,
                category_name=str(cat.get("name", "")),
                supercategory=cat.get("supercategory"),
                mapped_class=mapped_name,
                mapped_id=mapped_id,
                bbox=bbox,
                area=float(ann.get("area", bbox[2] * bbox[3] if len(bbox) == 4 else 0)),
                iscrowd=int(ann.get("iscrowd", 0)),
            )
        )

    report = {
        "coco_json": str(coco_path),
        "n_images_loaded": len(images),
        "n_annotations_loaded": len(annotations),
        "category_mapping": map_reasons,
        "supercategories": SUPER_IDS,
        "max_images": max_images,
        "held_out_site": held_out_site,
    }
    return SeaClearDataset(
        root=root,
        images=images,
        annotations=annotations,
        categories=list(cats.values()),
        class_map_report=report,
    )


def iter_valid_boxes(anns: list[AnnotationRecord]) -> Iterator[AnnotationRecord]:
    for a in anns:
        if a.mapped_id is None:
            continue
        if len(a.bbox) != 4:
            continue
        x, y, w, h = a.bbox
        if w <= 0 or h <= 0:
            continue
        yield a
