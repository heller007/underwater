#!/usr/bin/env python3
"""
E3: Fixed-path detectors — train and evaluate consistently on each shortlisted Tk.

Default shortlist from E2 (Lokrum): T0 (reuse E1), T2 (CLAHE), T4 (fusion).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import create_run, load_env, load_yaml, save_json, set_seed, setup_logging
from src.detection import load_detector_config, predict_split, train_yolo
from src.detection.naive_enhance import materialize_enhanced_split


def _find_e1_metrics(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    path = Path(path)
    if path.is_file() and path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    # directory: look for metrics_all.json
    for cand in [
        path / "train" / "eval" / "metrics_all.json",
        path / "eval" / "metrics_all.json",
        path / "metrics_all.json",
    ]:
        if cand.exists():
            with open(cand, encoding="utf-8") as f:
                return json.load(f)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="E3 fixed-path training")
    p.add_argument("--env", default=None)
    p.add_argument("--experiment", default="configs/experiments/e3_fixed_path.yaml")
    p.add_argument("--held-out-site", default=None)
    p.add_argument("--actions", default=None, help="Comma list, default from config")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument(
        "--reuse-t0-weights",
        default=None,
        help="E1 best.pt — skip T0 retrain and copy metrics if available",
    )
    p.add_argument(
        "--reuse-t0-metrics",
        default=None,
        help="Path to E1 metrics_all.json or E1 run folder",
    )
    p.add_argument(
        "--drop-enhanced",
        action="store_true",
        help="Delete materialized enhanced images after each action (keep weights)",
    )
    p.add_argument("--skip-actions", default="", help="Comma list to skip e.g. T0")
    args = p.parse_args()

    env = load_env(args.env)
    env.ensure_output_dirs()
    exp = load_yaml(args.experiment)
    seed = args.seed if args.seed is not None else int(exp.get("seed", 0))
    set_seed(seed)
    site = args.held_out_site or exp.get("held_out_site", "Lokrum")
    actions = (
        [a.strip() for a in args.actions.split(",") if a.strip()]
        if args.actions
        else list(exp.get("actions") or ["T0", "T2", "T4"])
    )
    skip = {a.strip() for a in args.skip_actions.split(",") if a.strip()}
    actions = [a for a in actions if a not in skip]

    src_yolo = env.processed_root / f"yolo_loso_{site.lower()}"
    if not (src_yolo / "data.yaml").exists():
        raise SystemExit(f"Missing YOLO root {src_yolo} — run prep first")

    det_overrides = dict(exp.get("detector") or {})
    det_cfg_path = det_overrides.pop("config", "configs/detector/yolov8n_raw.yaml")
    det_cfg = load_detector_config(det_cfg_path, det_overrides)
    if args.epochs is not None:
        det_cfg["epochs"] = args.epochs
    if args.batch is not None:
        det_cfg["batch"] = args.batch
    det_cfg["seed"] = seed

    parent = create_run(
        experiment_id=exp.get("experiment_id", "e3_fixed_path"),
        seed=seed,
        config={
            "env_name": env.name,
            "experiment": exp,
            "actions": actions,
            "held_out_site": site,
            "device": args.device or env.device,
            "detector": det_cfg,
        },
        runs_root=env.runs_root,
        fold=site,
        model_tag="fixed",
    )
    log = setup_logging(parent.run_dir)
    summary: dict = {"actions": {}, "held_out_site": site, "seed": seed}
    device = args.device or env.device

    for action in actions:
        log.info("===== E3 action %s =====", action)
        action_dir = parent.run_dir / action
        action_dir.mkdir(parents=True, exist_ok=True)

        # T0: optionally reuse E1
        if action == "T0" and args.reuse_t0_weights:
            w = Path(args.reuse_t0_weights)
            if not w.exists():
                raise SystemExit(f"reuse-t0-weights not found: {w}")
            metrics = _find_e1_metrics(
                Path(args.reuse_t0_metrics) if args.reuse_t0_metrics else w.parent.parent.parent
            )
            entry = {
                "reused_e1": True,
                "weights": str(w),
                "metrics": metrics,
            }
            # If no metrics file, evaluate T0 on raw yolo
            if metrics is None:
                log.info("No E1 metrics found; evaluating reused T0 weights on raw splits")
                eval_out = action_dir / "eval"
                metrics = {"splits": {}}
                for split in ("val", "test"):
                    m = predict_split(
                        weights=w,
                        data_yaml=src_yolo / "data.yaml",
                        split=split,
                        out_dir=eval_out,
                        device=device,
                        imgsz=int(det_cfg.get("imgsz", 640)),
                    )
                    metrics["splits"][split] = m.get("metrics", m)
                entry["metrics"] = metrics
            summary["actions"][action] = entry
            save_json(action_dir / "result.json", entry)
            continue

        # Materialize enhanced YOLO (train/val/test)
        enh_root = env.processed_root / f"yolo_loso_{site.lower()}_{action}"
        log.info("Materializing %s -> %s", action, enh_root)
        data_yaml = materialize_enhanced_split(
            src_yolo,
            enh_root,
            action,
            splits=("train", "val", "test"),
        )

        # Train
        train_dir = action_dir
        train_summary = train_yolo(
            data_yaml=data_yaml,
            cfg=det_cfg,
            run_dir=train_dir,
            device=device,
        )
        weights = Path(train_summary["best_weights"] or train_summary["last_weights"])
        log.info("%s training done: %s", action, weights)

        # Evaluate on same-path val/test
        eval_out = action_dir / "eval"
        split_metrics = {}
        for split in ("val", "test"):
            img_dir = enh_root / "images" / split
            if not img_dir.exists() or not any(img_dir.iterdir()):
                log.warning("Skip empty split %s for %s", split, action)
                continue
            m = predict_split(
                weights=weights,
                data_yaml=data_yaml,
                split=split,
                out_dir=eval_out,
                device=device,
                imgsz=int(det_cfg.get("imgsz", 640)),
            )
            split_metrics[split] = m.get("metrics", m)
            log.info("%s %s metrics=%s", action, split, split_metrics[split])

        entry = {
            "reused_e1": False,
            "weights": str(weights),
            "data_yaml": str(data_yaml),
            "train_summary": train_summary,
            "metrics": {"splits": split_metrics},
        }
        summary["actions"][action] = entry
        save_json(action_dir / "result.json", entry)

        if args.drop_enhanced and action != "T0":
            log.info("Dropping enhanced images for %s", action)
            shutil.rmtree(enh_root / "images", ignore_errors=True)

    # Comparison table vs T0 if present
    t0 = summary["actions"].get("T0", {}).get("metrics", {})
    t0_splits = t0.get("splits", t0) if isinstance(t0, dict) else {}
    comparison = {}
    for action, entry in summary["actions"].items():
        metrics = entry.get("metrics") or {}
        splits = metrics.get("splits", metrics)
        comparison[action] = {}
        for split, m in (splits or {}).items():
            if not isinstance(m, dict) or "mAP50_95" not in m:
                continue
            row = {
                "mAP50": m.get("mAP50"),
                "mAP50_95": m.get("mAP50_95"),
            }
            base = t0_splits.get(split, {})
            if isinstance(base, dict) and "mAP50_95" in base:
                row["delta_mAP50_95_vs_T0"] = float(m["mAP50_95"] - base["mAP50_95"])
                row["delta_mAP50_vs_T0"] = float(m["mAP50"] - base["mAP50"])
            comparison[action][split] = row
    summary["comparison_vs_T0"] = comparison

    # Rank by test mAP50-95
    ranked = []
    for action, entry in summary["actions"].items():
        m = (entry.get("metrics") or {}).get("splits", {}).get("test", {})
        if isinstance(m, dict) and "mAP50_95" in m:
            ranked.append(
                {
                    "action": action,
                    "mAP50_95": m["mAP50_95"],
                    "mAP50": m["mAP50"],
                }
            )
    ranked.sort(key=lambda x: x["mAP50_95"], reverse=True)
    summary["ranking_test"] = ranked

    save_json(parent.run_dir / "e3_results.json", summary)
    log.info("E3 ranking (test): %s", ranked)
    log.info("Wrote %s", parent.run_dir / "e3_results.json")
    print(parent.run_dir)


if __name__ == "__main__":
    main()
