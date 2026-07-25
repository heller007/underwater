"""YOLOv8 detector training helpers (Ultralytics)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io import deep_merge, load_yaml, resolve_device
from src.common.run import save_json


def load_detector_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_yaml(path)
    if overrides:
        # Flatten nested detector overrides from experiment yaml
        flat = {k: v for k, v in overrides.items() if k != "config"}
        cfg = deep_merge(cfg, flat)
    return cfg


def train_yolo(
    data_yaml: Path,
    cfg: dict[str, Any],
    run_dir: Path,
    device: str | None = None,
    resume: bool | str = False,
) -> dict[str, Any]:
    """
    Train YOLOv8 with Ultralytics. On dual T4, pass device='0,1'.
    Returns paths to best/last weights and metrics summary.
    """
    from ultralytics import YOLO

    device = resolve_device(device if device is not None else cfg.get("device", "auto"))
    model_name = cfg.get("model", "yolov8n.pt")
    model = YOLO(model_name)

    project = str(run_dir)
    name = "train"

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=int(cfg.get("epochs", 100)),
        patience=int(cfg.get("patience", 20)),
        batch=int(cfg.get("batch", 16)),
        imgsz=int(cfg.get("imgsz", 640)),
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=float(cfg.get("lr0", 1e-3)),
        lrf=float(cfg.get("lrf", 0.01)),
        weight_decay=float(cfg.get("weight_decay", 5e-4)),
        warmup_epochs=float(cfg.get("warmup_epochs", 3)),
        cos_lr=bool(cfg.get("cos_lr", True)),
        amp=bool(cfg.get("amp", True)),
        hsv_h=float(cfg.get("hsv_h", 0.0)),
        hsv_s=float(cfg.get("hsv_s", 0.0)),
        hsv_v=float(cfg.get("hsv_v", 0.0)),
        degrees=float(cfg.get("degrees", 0.0)),
        translate=float(cfg.get("translate", 0.1)),
        scale=float(cfg.get("scale", 0.5)),
        shear=float(cfg.get("shear", 0.0)),
        perspective=float(cfg.get("perspective", 0.0)),
        flipud=float(cfg.get("flipud", 0.0)),
        fliplr=float(cfg.get("fliplr", 0.5)),
        mosaic=float(cfg.get("mosaic", 1.0)),
        mixup=float(cfg.get("mixup", 0.0)),
        close_mosaic=int(cfg.get("close_mosaic", 10)),
        workers=int(cfg.get("workers", 2)),
        seed=int(cfg.get("seed", 0)),
        pretrained=bool(cfg.get("pretrained", True)),
        project=project,
        name=name,
        exist_ok=True,
        device=device,
        save=True,
        plots=True,
        verbose=True,
    )
    if resume:
        train_kwargs["resume"] = resume if isinstance(resume, str) else True

    results = model.train(**train_kwargs)
    save_dir = Path(project) / name
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"

    summary: dict[str, Any] = {
        "device": device,
        "save_dir": str(save_dir),
        "best_weights": str(best) if best.exists() else None,
        "last_weights": str(last) if last.exists() else None,
        "model": model_name,
        "data_yaml": str(data_yaml),
    }
    # Ultralytics results may expose maps
    try:
        if hasattr(results, "results_dict"):
            summary["train_results"] = dict(results.results_dict)
        elif isinstance(results, dict):
            summary["train_results"] = results
    except Exception:
        pass

    save_json(run_dir / "train_summary.json", summary)
    return summary


def predict_split(
    weights: Path,
    data_yaml: Path,
    split: str,
    out_dir: Path,
    imgsz: int = 640,
    device: str | None = None,
    conf: float = 0.001,
) -> dict[str, Any]:
    """Run validation-style evaluation on a named split via Ultralytics."""
    from ultralytics import YOLO

    device = resolve_device(device)
    model = YOLO(str(weights))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics val() uses data.yaml train/val; for test we temporarily point val->test
    metrics = model.val(
        data=str(data_yaml),
        split=split if split in ("train", "val", "test") else "val",
        imgsz=imgsz,
        device=device,
        conf=conf,
        plots=True,
        save_json=True,
        project=str(out_dir),
        name=f"val_{split}",
        exist_ok=True,
    )

    out: dict[str, Any] = {"split": split, "weights": str(weights)}
    try:
        out["metrics"] = {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        if hasattr(metrics.box, "maps"):
            out["metrics"]["per_class_mAP50_95"] = [float(x) for x in metrics.box.maps]
    except Exception as e:
        out["metrics_error"] = str(e)
    save_json(out_dir / f"metrics_{split}.json", out)
    return out
