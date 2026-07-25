#!/usr/bin/env python3
"""E5: Oracle study on frozen E4 mixed detector (quiet cell progress)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import create_run, load_env, load_yaml, set_seed, setup_logging
from src.common.quiet import progress, quiet_run
from src.data import load_seaclear, prepare_yolo_fold
from src.detection.oracle import run_e5_oracle


def _ensure_splits(env, site: str, needed: list[str]) -> Path:
    yolo = env.processed_root / f"yolo_loso_{site.lower()}"
    missing = [s for s in needed if not (yolo / "images" / s).exists()]
    if not missing and (yolo / "data.yaml").exists():
        return yolo
    progress(f"[e5] preparing missing YOLO splits: {missing or needed}")
    manifest = env.manifests_root / f"loso_{site.lower()}" / "manifest.csv"
    if not manifest.exists():
        raise SystemExit("Missing manifests — run: python scripts/run_stage.py --stage prep --env kaggle")
    if env.seaclear_root is None:
        raise SystemExit("SeaClear root not found")
    ds = load_seaclear(env.seaclear_root)
    prepare_yolo_fold(ds, manifest, yolo, splits=tuple(sorted(set(needed) | {"train", "val", "test"})))
    return yolo


def main() -> None:
    p = argparse.ArgumentParser(description="E5 oracle study")
    p.add_argument("--env", default=None)
    p.add_argument("--experiment", default="configs/experiments/e5_oracle.yaml")
    p.add_argument("--weights", required=True, help="E4 mixed best.pt")
    p.add_argument("--held-out-site", default=None)
    p.add_argument("--actions", default=None)
    p.add_argument("--splits", default=None, help="Comma list, default gate,test")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lambda-cost", type=float, default=None)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--drop-enhanced", action="store_true")
    p.add_argument("--no-quiet", action="store_true")
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
    splits = (
        [s.strip() for s in args.splits.split(",") if s.strip()]
        if args.splits
        else list(exp.get("splits") or ["gate", "test"])
    )
    lam = args.lambda_cost if args.lambda_cost is not None else float(exp.get("lambda_cost", 0.05))
    delta = args.delta if args.delta is not None else float(exp.get("delta", 0.01))
    conf = args.conf if args.conf is not None else float(exp.get("conf", 0.25))
    device = args.device or env.device

    weights = Path(args.weights)
    if not weights.exists():
        # search under directory
        hits = list(weights.rglob("best.pt")) if weights.is_dir() else []
        if hits:
            weights = hits[0]
        else:
            raise SystemExit(f"Weights not found: {args.weights}")

    yolo = _ensure_splits(env, site, splits)

    run = create_run(
        experiment_id=exp.get("experiment_id", "e5_oracle"),
        seed=seed,
        config={
            "env_name": env.name,
            "experiment": exp,
            "weights": str(weights),
            "actions": actions,
            "splits": splits,
            "lambda": lam,
            "delta": delta,
            "conf": conf,
            "device": device,
            "held_out_site": site,
        },
        runs_root=env.runs_root,
        fold=site,
        model_tag="oracle",
    )
    setup_logging(run.run_dir)
    log_path = run.run_dir / "e5_console.log"

    def _work():
        progress(f"[e5] start weights={weights}")
        progress(f"[e5] actions={actions} splits={splits} λ={lam} δ={delta} conf={conf}")
        summary = run_e5_oracle(
            weights=weights,
            src_yolo_root=yolo,
            out_dir=run.run_dir / "e5",
            actions=actions,
            splits=splits,
            device=device,
            lam=lam,
            delta=delta,
            conf=conf,
            drop_enhanced=args.drop_enhanced,
        )
        decision = summary.get("primary_go_no_go", {})
        progress(f"[e5] DECISION: {decision.get('decision')}")
        progress(f"[e5] gap_mAP50={decision.get('oracle_minus_best_fixed_mAP50')}")
        progress(f"[e5] results={run.run_dir / 'e5' / 'e5_results.json'}")
        return summary

    if quiet:
        with quiet_run(log_path):
            _work()
    else:
        _work()
    progress(str(run.run_dir))


if __name__ == "__main__":
    main()
