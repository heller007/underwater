#!/usr/bin/env python3
"""
E4: Mixed-path detector — each train image gets one action from {T0, T4} (deterministic).

Quiet by default: cell shows progress lines; details go to run.log.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import create_run, load_env, load_yaml, save_json, set_seed, setup_logging
from src.common.quiet import progress, quiet_run
from src.detection import load_detector_config, predict_split, train_yolo
from src.detection.mixed_path import materialize_mixed_dataset
from src.detection.naive_enhance import materialize_enhanced_split


def main() -> None:
    p = argparse.ArgumentParser(description="E4 mixed-path training")
    p.add_argument("--env", default=None)
    p.add_argument("--experiment", default="configs/experiments/e4_mixed_path.yaml")
    p.add_argument("--held-out-site", default=None)
    p.add_argument("--actions", default=None, help="Default T0,T4")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--drop-enhanced", action="store_true")
    p.add_argument("--no-quiet", action="store_true", help="Spam full Ultralytics logs to cell")
    args = p.parse_args()

    quiet = not args.no_quiet
    env = load_env(args.env)
    env.ensure_output_dirs()
    exp = load_yaml(args.experiment)
    seed = args.seed if args.seed is not None else int(exp.get("seed", 0))
    set_seed(seed)
    site = args.held_out_site or exp.get("held_out_site", "Lokrum")
    actions = (
        [a.strip() for a in args.actions.split(",") if a.strip()]
        if args.actions
        else list(exp.get("actions") or ["T0", "T4"])
    )

    src_yolo = env.processed_root / f"yolo_loso_{site.lower()}"
    if not (src_yolo / "data.yaml").exists():
        raise SystemExit(f"Missing {src_yolo} — run: python scripts/run_stage.py --stage prep --env kaggle")

    det_overrides = dict(exp.get("detector") or {})
    det_cfg_path = det_overrides.pop("config", "configs/detector/yolov8n_raw.yaml")
    det_cfg = load_detector_config(det_cfg_path, det_overrides)
    if args.epochs is not None:
        det_cfg["epochs"] = args.epochs
    if args.batch is not None:
        det_cfg["batch"] = args.batch
    det_cfg["seed"] = seed
    device = args.device or env.device

    run = create_run(
        experiment_id=exp.get("experiment_id", "e4_mixed_path"),
        seed=seed,
        config={
            "env_name": env.name,
            "experiment": exp,
            "actions": actions,
            "held_out_site": site,
            "device": device,
            "detector": det_cfg,
        },
        runs_root=env.runs_root,
        fold=site,
        model_tag="mixed",
    )
    setup_logging(run.run_dir)
    log_path = run.run_dir / "e4_console.log"

    def _work() -> dict:
        progress(f"[e4] run={run.run_id}")
        progress(f"[e4] actions={actions} site={site} device={device}")

        mixed_root = env.processed_root / f"yolo_loso_{site.lower()}_mixed_{'_'.join(actions)}"
        data_yaml, stats = materialize_mixed_dataset(
            src_yolo,
            mixed_root,
            actions=actions,
            seed=seed,
            splits=("train", "val", "test"),
        )
        progress(f"[e4] mixed assignment stats: {stats.get('splits')}")

        train_summary = train_yolo(
            data_yaml=data_yaml,
            cfg=det_cfg,
            run_dir=run.run_dir,
            device=device,
            quiet=quiet,
        )
        weights = Path(train_summary["best_weights"] or train_summary["last_weights"])

        # Eval mixed detector on: mixed val/test, plus pure T0 and T4 test sets
        results: dict = {
            "weights": str(weights),
            "mixed_stats": stats,
            "train_summary": train_summary,
            "eval": {},
        }

        progress("[e4] eval on mixed val/test")
        for split in ("val", "test"):
            m = predict_split(
                weights=weights,
                data_yaml=data_yaml,
                split=split,
                out_dir=run.run_dir / "eval_mixed",
                device=device,
                quiet=quiet,
            )
            results["eval"][f"mixed_{split}"] = m.get("metrics", m)

        for action in actions:
            progress(f"[e4] eval on pure {action} test (consistent path)")
            pure_root = env.processed_root / f"yolo_loso_{site.lower()}_{action}_e4eval"
            pure_yaml = materialize_enhanced_split(
                src_yolo, pure_root, action, splits=("test",)
            )
            m = predict_split(
                weights=weights,
                data_yaml=pure_yaml,
                split="test",
                out_dir=run.run_dir / f"eval_{action}",
                device=device,
                quiet=quiet,
            )
            results["eval"][f"{action}_test"] = m.get("metrics", m)
            if args.drop_enhanced:
                shutil.rmtree(pure_root / "images", ignore_errors=True)

        if args.drop_enhanced:
            progress("[e4] dropping mixed images to save disk")
            shutil.rmtree(mixed_root / "images", ignore_errors=True)

        save_json(run.run_dir / "e4_results.json", results)
        progress(f"[e4] DONE results -> {run.run_dir / 'e4_results.json'}")
        progress(f"[e4] weights -> {weights}")
        return results

    if quiet:
        with quiet_run(log_path):
            # progress() still prints to cell; YOLO spam goes to log
            results = _work()
    else:
        results = _work()

    print(run.run_dir)  # final path always visible


if __name__ == "__main__":
    main()
