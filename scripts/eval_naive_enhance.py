#!/usr/bin/env python3
"""E2: Naive test-time enhancement on a frozen E1 detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import create_run, load_env, load_yaml, set_seed, setup_logging
from src.detection.naive_enhance import run_e2


def main() -> None:
    p = argparse.ArgumentParser(description="E2 naive test-time enhancement")
    p.add_argument("--env", default=None)
    p.add_argument("--experiment", default="configs/experiments/e2_naive_enhance.yaml")
    p.add_argument(
        "--weights",
        required=True,
        help="Path to E1 best.pt (frozen raw detector)",
    )
    p.add_argument("--held-out-site", default=None)
    p.add_argument("--data-yaml-root", default=None, help="YOLO root with images/{val,test}")
    p.add_argument("--actions", default=None, help="Comma list e.g. T0,T1,T2,T3,T4")
    p.add_argument("--splits", default="val,test")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--drop-enhanced",
        action="store_true",
        help="Delete enhanced images after each action eval (saves disk)",
    )
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
        else list(exp.get("actions") or ["T0", "T1", "T2", "T3", "T4"])
    )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.data_yaml_root:
        yolo_root = Path(args.data_yaml_root)
    else:
        yolo_root = env.processed_root / f"yolo_loso_{site.lower()}"
    if not (yolo_root / "data.yaml").exists():
        raise SystemExit(
            f"YOLO dataset not found at {yolo_root}. Run prep/e1 first "
            f"(scripts/prepare_yolo.py --held-out-site {site})."
        )

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    config = {
        "env_name": env.name,
        "experiment": exp,
        "weights": str(weights),
        "yolo_root": str(yolo_root),
        "actions": actions,
        "splits": splits,
        "held_out_site": site,
        "device": args.device or env.device,
    }
    run = create_run(
        experiment_id=exp.get("experiment_id", "e2_naive_enhance"),
        seed=seed,
        config=config,
        runs_root=env.runs_root,
        fold=site,
        model_tag="e1frozen",
    )
    log = setup_logging(run.run_dir)
    log.info("E2 start weights=%s actions=%s", weights, actions)

    results = run_e2(
        weights=weights,
        src_yolo_root=yolo_root,
        out_dir=run.run_dir / "e2",
        actions=actions,
        splits=splits,
        device=config["device"],
        keep_enhanced=not args.drop_enhanced,
    )
    log.info("Ranking (test): %s", results.get("ranking", {}).get("test"))
    log.info("Deltas vs T0: %s", results.get("deltas_vs_T0"))
    log.info("Wrote %s", run.run_dir / "e2" / "e2_results.json")
    print(run.run_dir)


if __name__ == "__main__":
    main()
