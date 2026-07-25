"""E2: apply transforms at test time on a frozen detector (no retraining)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import yaml
from tqdm import tqdm

from src.common.run import save_json
from src.detection.train import predict_split
from src.enhancement.transforms import TRANSFORMS, get_transform


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        try:
            import os

            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def materialize_enhanced_split(
    src_yolo_root: Path,
    out_root: Path,
    action_id: str,
    splits: tuple[str, ...] = ("val", "test"),
) -> Path:
    """
    Build a YOLO dataset whose images are T_k(x), labels copied/linked from source.
    Returns path to data.yaml.
    """
    src_yolo_root = Path(src_yolo_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tfm = get_transform(action_id)

    with open(src_yolo_root / "data.yaml", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    for split in splits:
        src_img = src_yolo_root / "images" / split
        src_lbl = src_yolo_root / "labels" / split
        dst_img = out_root / "images" / split
        dst_lbl = out_root / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        if not src_img.exists():
            continue
        images = sorted(
            [
                p
                for p in src_img.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        for img_path in tqdm(images, desc=f"{action_id}/{split}", leave=False):
            out_img = dst_img / img_path.name
            lbl_src = src_lbl / f"{img_path.stem}.txt"
            lbl_dst = dst_lbl / f"{img_path.stem}.txt"
            if action_id == "T0":
                _copy_or_link(img_path, out_img)
            else:
                if not out_img.exists():
                    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    enh = tfm(bgr)
                    cv2.imwrite(str(out_img), enh)
            if lbl_src.exists():
                _copy_or_link(lbl_src, lbl_dst)
            else:
                lbl_dst.write_text("", encoding="utf-8")

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": data_cfg.get("train", "images/train"),
        "val": "images/val",
        "test": "images/test",
        "names": data_cfg.get("names", {0: "debris", 1: "bio", 2: "robot"}),
        "nc": data_cfg.get("nc", 3),
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return yaml_path


def run_e2(
    weights: Path,
    src_yolo_root: Path,
    out_dir: Path,
    actions: list[str],
    splits: list[str],
    device: str | None = None,
    imgsz: int = 640,
    keep_enhanced: bool = True,
) -> dict[str, Any]:
    """
    For each action: materialize enhanced YOLO set, evaluate frozen weights, record metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "weights": str(weights),
        "src_yolo_root": str(src_yolo_root),
        "actions": {},
        "baseline_action": "T0",
    }

    for action in actions:
        if action not in TRANSFORMS:
            raise KeyError(f"Unknown action {action}")
        enh_root = out_dir / f"enhanced_{action}"
        data_yaml = materialize_enhanced_split(
            src_yolo_root,
            enh_root,
            action,
            splits=tuple(splits),
        )
        action_metrics: dict[str, Any] = {"data_yaml": str(data_yaml), "splits": {}}
        for split in splits:
            img_dir = enh_root / "images" / split
            if not img_dir.exists() or not any(img_dir.iterdir()):
                action_metrics["splits"][split] = {"skipped": True, "reason": "empty"}
                continue
            m = predict_split(
                weights=weights,
                data_yaml=data_yaml,
                split=split,
                out_dir=out_dir / f"eval_{action}",
                imgsz=imgsz,
                device=device,
            )
            action_metrics["splits"][split] = m.get("metrics", m)
        results["actions"][action] = action_metrics
        if not keep_enhanced and action != "T0":
            # Free disk: drop images after eval (keep metrics)
            shutil.rmtree(enh_root / "images", ignore_errors=True)

    # Deltas vs T0
    deltas: dict[str, Any] = {}
    t0 = results["actions"].get("T0", {}).get("splits", {})
    for action, am in results["actions"].items():
        if action == "T0":
            continue
        deltas[action] = {}
        for split, metrics in am.get("splits", {}).items():
            if not isinstance(metrics, dict) or "mAP50" not in metrics:
                continue
            base = t0.get(split, {})
            if "mAP50" not in base:
                continue
            deltas[action][split] = {
                "delta_mAP50": float(metrics["mAP50"] - base["mAP50"]),
                "delta_mAP50_95": float(metrics["mAP50_95"] - base["mAP50_95"]),
                "mAP50": metrics["mAP50"],
                "mAP50_95": metrics["mAP50_95"],
                "baseline_mAP50": base["mAP50"],
                "baseline_mAP50_95": base["mAP50_95"],
            }
    results["deltas_vs_T0"] = deltas

    # Shortlist hint: best action per split by mAP50-95
    shortlist: dict[str, Any] = {}
    for split in splits:
        ranked = []
        for action, am in results["actions"].items():
            m = am.get("splits", {}).get(split, {})
            if isinstance(m, dict) and "mAP50_95" in m:
                ranked.append((action, float(m["mAP50_95"]), float(m["mAP50"])))
        ranked.sort(key=lambda x: x[1], reverse=True)
        shortlist[split] = [
            {"action": a, "mAP50_95": m95, "mAP50": m50} for a, m95, m50 in ranked
        ]
    results["ranking"] = shortlist

    save_json(out_dir / "e2_results.json", results)
    return results
