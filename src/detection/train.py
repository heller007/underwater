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
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Train YOLOv8 with Ultralytics. On dual T4, pass device='0,1'.
    Returns paths to best/last weights and metrics summary.
    """
    from ultralytics import YOLO

    from src.common.quiet import progress, silence_ultralytics

    if quiet:
        silence_ultralytics()

    device = resolve_device(device if device is not None else cfg.get("device", "auto"))
    model_name = cfg.get("model", "yolov8n.pt")
    model = YOLO(model_name)

    project = str(run_dir)
    name = "train"
    epochs = int(cfg.get("epochs", 100))

    if quiet:

        def _on_fit_epoch_end(trainer):  # type: ignore[no-untyped-def]
            ep = int(getattr(trainer, "epoch", 0)) + 1
            # Minimal: print every 5 epochs + first + last
            if ep != 1 and ep % 5 != 0 and ep != epochs:
                return
            metrics = getattr(trainer, "metrics", {}) or {}
            map50 = metrics.get("metrics/mAP50(B)", metrics.get("mAP50"))
            map5095 = metrics.get("metrics/mAP50-95(B)", metrics.get("mAP50-95"))
            if map50 is not None and map5095 is not None:
                progress(
                    f"[train] epoch {ep}/{epochs}  mAP50={float(map50):.3f}  mAP50-95={float(map5095):.3f}"
                )
            else:
                progress(f"[train] epoch {ep}/{epochs}")

        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
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
        plots=not quiet,
        verbose=not quiet,
    )
    if resume:
        train_kwargs["resume"] = resume if isinstance(resume, str) else True

    if quiet:
        progress(f"[train] start epochs={epochs} device={device} (updates every 5 epochs)")
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
    try:
        if hasattr(results, "results_dict"):
            summary["train_results"] = dict(results.results_dict)
        elif isinstance(results, dict):
            summary["train_results"] = results
    except Exception:
        pass

    save_json(run_dir / "train_summary.json", summary)
    if quiet:
        progress(f"[train] done best={summary.get('best_weights')}")
    return summary


def predict_split(
    weights: Path,
    data_yaml: Path,
    split: str,
    out_dir: Path,
    imgsz: int = 640,
    device: str | None = None,
    conf: float = 0.001,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run validation-style evaluation on a named split via Ultralytics."""
    from ultralytics import YOLO

    from src.common.quiet import progress, silence_ultralytics

    if quiet:
        silence_ultralytics()
        progress(f"[eval] split={split}")

    device = resolve_device(device)
    model = YOLO(str(weights))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = model.val(
        data=str(data_yaml),
        split=split if split in ("train", "val", "test") else "val",
        imgsz=imgsz,
        device=device,
        conf=conf,
        plots=not quiet,
        save_json=True,
        project=str(out_dir),
        name=f"val_{split}",
        exist_ok=True,
        verbose=not quiet,
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
    if quiet and "metrics" in out:
        m = out["metrics"]
        progress(
            f"[eval] {split}: mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f}"
        )
    return out
