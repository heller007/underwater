#!/usr/bin/env python3
"""E7: train descriptor/CNN/combined utility gates on E5 oracle labels."""

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
from src.routing.gate import run_e7_gate


def _ensure_splits(env, site: str, needed: list[str]) -> Path:
    yolo = env.processed_root / f"yolo_loso_{site.lower()}"
    missing = [s for s in needed if not (yolo / "images" / s).exists()]
    if not missing and (yolo / "data.yaml").exists():
        return yolo
    progress(f"[e7] preparing missing YOLO splits: {missing or needed}")
    if env.seaclear_root is None:
        raise SystemExit("SeaClear root not found — attach SeaClear dataset")
    ds = load_seaclear(env.seaclear_root)
    manifest = env.manifests_root / f"loso_{site.lower()}" / "manifest.csv"
    if not manifest.exists():
        progress("[e7] building LOSO manifests...")
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
    for h in hits:
        if "e4" in h.name.lower() or "mixed" in str(h).lower():
            return h
    if hits:
        return hits[0]
    raise SystemExit(f"Weights not found: {path}")


def _find_oracle(explicit: str | None, names: list[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"Oracle table not found: {p}")
        return p
    for name in names:
        for cand in (ROOT / name, Path.cwd() / name):
            if cand.exists():
                return cand
    raise SystemExit(
        "Need oracle_table_gate.csv (from E5). Pass --oracle-gate or place it in the repo root."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="E7 learned utility gate")
    p.add_argument("--env", default=None)
    p.add_argument("--experiment", default="configs/experiments/e7_utility_gate.yaml")
    p.add_argument("--weights", required=True, help="E4 mixed best.pt")
    p.add_argument("--held-out-site", default=None)
    p.add_argument("--actions", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--oracle-gate", default=None)
    p.add_argument("--oracle-test", default=None)
    p.add_argument("--gates", default=None, help="descriptor,cnn,combined")
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
    conf = args.conf if args.conf is not None else float(exp.get("conf", 0.25))
    epochs = args.epochs if args.epochs is not None else int(exp.get("epochs", 40))
    batch = args.batch if args.batch is not None else int(exp.get("batch", 64))
    device = args.device or env.device
    kinds = (
        [g.strip() for g in args.gates.split(",") if g.strip()]
        if args.gates
        else list(exp.get("gates") or ["descriptor", "cnn", "combined"])
    )
    weights = _resolve_weights(Path(args.weights))
    yolo = _ensure_splits(env, site, ["gate", "test"])
    oracle_gate = _find_oracle(args.oracle_gate, ["oracle_table_gate.csv"])
    oracle_test = None
    try:
        oracle_test = _find_oracle(args.oracle_test, ["oracle_table_test.csv"])
    except SystemExit:
        progress("[e7] WARNING: no oracle_table_test.csv — skip test accuracy/regret")

    run = create_run(
        experiment_id=exp.get("experiment_id", "e7_utility_gate"),
        seed=seed,
        config={
            "env_name": env.name,
            "experiment": exp,
            "weights": str(weights),
            "actions": actions,
            "gates": kinds,
            "epochs": epochs,
            "batch": batch,
            "conf": conf,
            "device": device,
            "held_out_site": site,
            "oracle_gate": str(oracle_gate),
            "oracle_test": str(oracle_test) if oracle_test else None,
        },
        runs_root=env.runs_root,
        fold=site,
        model_tag="gate",
    )
    setup_logging(run.run_dir)
    log_path = run.run_dir / "e7_console.log"

    def _work():
        progress(f"[e7] start weights={weights} gates={kinds}")
        summary = run_e7_gate(
            weights=weights,
            src_yolo_root=yolo,
            out_dir=run.run_dir / "e7",
            actions=actions,
            oracle_gate=oracle_gate,
            oracle_test=oracle_test,
            device=device,
            gate_kinds=kinds,
            epochs=epochs,
            batch_size=batch,
            lr=float(exp.get("lr", 1e-3)),
            alpha=float(exp.get("alpha", 0.5)),
            temperature=float(exp.get("temperature", 1.0)),
            imgsz=int(exp.get("imgsz", 160)),
            conf=conf,
            drop_enhanced=args.drop_enhanced,
        )
        for kind, g in summary.get("gates", {}).items():
            m = g.get("test_split", {}).get("metrics", {})
            progress(
                f"[e7/{kind}] test mAP50={m.get('mAP50')} "
                f"acc={g.get('test_split', {}).get('action_accuracy_vs_oracle')}"
            )
        progress(f"[e7] results={run.run_dir / 'e7' / 'e7_results.json'}")

    if quiet:
        with quiet_run(log_path):
            _work()
    else:
        _work()
    progress(str(run.run_dir))


if __name__ == "__main__":
    main()
