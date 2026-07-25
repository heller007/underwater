#!/usr/bin/env python3
"""E6: UCIQE/UIQM/heuristic action selectors on frozen E4 detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import create_run, load_env, load_yaml, set_seed, setup_logging
from src.common.quiet import progress, quiet_run
from src.data import build_loso_fold, load_seaclear, prepare_yolo_fold, write_fold_manifests
from src.routing.quality_selectors import run_e6_quality_selectors


def _ensure_splits(env, site: str, needed: list[str]) -> Path:
    yolo = env.processed_root / f"yolo_loso_{site.lower()}"
    missing = [s for s in needed if not (yolo / "images" / s).exists()]
    if not missing and (yolo / "data.yaml").exists():
        return yolo
    progress(f"[e6] preparing missing YOLO splits: {missing or needed}")
    if env.seaclear_root is None:
        raise SystemExit("SeaClear root not found — attach SeaClear dataset")
    ds = load_seaclear(env.seaclear_root)
    manifest = env.manifests_root / f"loso_{site.lower()}" / "manifest.csv"
    if not manifest.exists():
        progress("[e6] building LOSO manifests...")
        df = build_loso_fold(ds, held_out_site=site, seed=0)
        write_fold_manifests(df, env.manifests_root, held_out_site=site)
    prepare_yolo_fold(
        ds,
        manifest,
        yolo,
        splits=tuple(sorted(set(needed) | {"train", "val", "test", "gate"})),
    )
    return yolo


def _resolve_weights(path: Path) -> Path:
    if path.is_file() and path.suffix == ".pt":
        return path
    hits = list(path.rglob("best.pt")) if path.exists() else []
    # prefer e4 naming
    for h in hits:
        if "e4" in h.name.lower() or "mixed" in str(h).lower():
            return h
    if hits:
        return hits[0]
    raise SystemExit(f"Weights not found: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="E6 quality-metric selectors")
    p.add_argument("--env", default=None)
    p.add_argument("--experiment", default="configs/experiments/e6_quality_selectors.yaml")
    p.add_argument("--weights", required=True, help="E4 mixed best.pt")
    p.add_argument("--held-out-site", default=None)
    p.add_argument("--actions", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--oracle-gate", default=None, help="oracle_table_gate.csv")
    p.add_argument("--oracle-test", default=None, help="oracle_table_test.csv")
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
    conf = args.conf if args.conf is not None else float(exp.get("conf", 0.25))
    device = args.device or env.device
    weights = _resolve_weights(Path(args.weights))
    yolo = _ensure_splits(env, site, splits)

    oracle_tables = {}
    if args.oracle_gate:
        oracle_tables["gate"] = Path(args.oracle_gate)
    if args.oracle_test:
        oracle_tables["test"] = Path(args.oracle_test)
    # default: look in repo root / cwd
    for split, name in (("gate", "oracle_table_gate.csv"), ("test", "oracle_table_test.csv")):
        if split not in oracle_tables:
            for cand in (ROOT / name, Path.cwd() / name, env.runs_root / name):
                if cand.exists():
                    oracle_tables[split] = cand
                    break

    run = create_run(
        experiment_id=exp.get("experiment_id", "e6_quality_selectors"),
        seed=seed,
        config={
            "env_name": env.name,
            "experiment": exp,
            "weights": str(weights),
            "actions": actions,
            "splits": splits,
            "conf": conf,
            "device": device,
            "held_out_site": site,
            "oracle_tables": {k: str(v) for k, v in oracle_tables.items()},
        },
        runs_root=env.runs_root,
        fold=site,
        model_tag="quality",
    )
    setup_logging(run.run_dir)
    log_path = run.run_dir / "e6_console.log"

    def _work():
        progress(f"[e6] start weights={weights}")
        summary = run_e6_quality_selectors(
            weights=weights,
            src_yolo_root=yolo,
            out_dir=run.run_dir / "e6",
            actions=actions,
            splits=splits,
            device=device,
            conf=conf,
            oracle_tables=oracle_tables,
            drop_enhanced=args.drop_enhanced,
        )
        for split, s in summary.get("splits", {}).items():
            board = {
                n: v.get("metrics", {}).get("mAP50")
                for n, v in s.get("selectors", {}).items()
            }
            progress(f"[e6] {split} mAP50={board}")
        progress(f"[e6] results={run.run_dir / 'e6' / 'e6_results.json'}")

    if quiet:
        with quiet_run(log_path):
            _work()
    else:
        _work()
    progress(str(run.run_dir))


if __name__ == "__main__":
    main()
