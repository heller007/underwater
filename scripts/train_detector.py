#!/usr/bin/env python3
"""Train raw / fixed-path YOLOv8 detector (E1 baseline entrypoint)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (
    create_run,
    load_env,
    load_yaml,
    set_seed,
    setup_logging,
)
from src.detection import load_detector_config, train_yolo


def main() -> None:
    p = argparse.ArgumentParser(description="Train detector")
    p.add_argument("--env", default=None)
    p.add_argument("--experiment", default="configs/experiments/e1_baseline.yaml")
    p.add_argument("--data-yaml", default=None, help="YOLO data.yaml path")
    p.add_argument("--held-out-site", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help='e.g. "0,1" for dual T4')
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    env = load_env(args.env)
    env.ensure_output_dirs()
    exp = load_yaml(args.experiment)
    seed = args.seed if args.seed is not None else int(exp.get("seed", 0))
    set_seed(seed)

    site = args.held_out_site or exp.get("held_out_site", "Lokrum")
    det_overrides = dict(exp.get("detector") or {})
    det_cfg_path = det_overrides.pop("config", "configs/detector/yolov8n_raw.yaml")
    det_cfg = load_detector_config(det_cfg_path, det_overrides)
    if args.epochs is not None:
        det_cfg["epochs"] = args.epochs
    if args.batch is not None:
        det_cfg["batch"] = args.batch
    det_cfg["seed"] = seed

    if args.data_yaml:
        data_yaml = Path(args.data_yaml)
    else:
        data_yaml = env.processed_root / f"yolo_loso_{site.lower()}" / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml} (run prepare_yolo.py first)")

    config = {
        "env_name": env.name,
        "experiment": exp,
        "detector": det_cfg,
        "data_yaml": str(data_yaml),
        "held_out_site": site,
        "device": args.device or env.device,
    }
    run = create_run(
        experiment_id=exp.get("experiment_id", "e1_baseline"),
        seed=seed,
        config=config,
        runs_root=env.runs_root,
        fold=site,
        model_tag=exp.get("action", "T0").lower(),
    )
    log = setup_logging(run.run_dir)
    log.info("Run %s device=%s data=%s", run.run_id, config["device"], data_yaml)
    log.info("Hardware will be recorded in run_manifest.json")

    summary = train_yolo(
        data_yaml=data_yaml,
        cfg=det_cfg,
        run_dir=run.run_dir,
        device=config["device"],
        resume=args.resume,
    )
    log.info("Training complete: %s", summary.get("best_weights"))
    print(run.run_dir)


if __name__ == "__main__":
    main()
