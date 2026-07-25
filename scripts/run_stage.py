#!/usr/bin/env python3
"""
Stage runner for Kaggle / local.

Stages:
  smoke  - audit (optional) + tiny train
  prep   - audit + splits + yolo prepare
  e1     - prep (if needed) + full baseline train + eval
  e2     - naive test-time enhancement on frozen E1 weights
  e3     - fixed-path train/eval for shortlisted actions (T0/T2/T4)
  e4     - mixed-path detector (T0+T4); quiet progress by default
  e5     - oracle study on frozen mixed detector (go/no-go)
  e6     - UCIQE/UIQM/heuristic selectors on frozen E4
  e7     - learned utility gate (descriptor/CNN/combined)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd: list[str]) -> None:
    # Minimal: do not echo full command spam; details stay in stage logs
    subprocess.check_call(cmd, cwd=ROOT)


def _split_has_images(processed_root: Path, site: str, split: str) -> bool:
    img_dir = processed_root / f"yolo_loso_{site.lower()}" / "images" / split
    if not img_dir.exists():
        return False
    return any(img_dir.iterdir())


def _latest_e1_weights(runs_root: Path, site: str) -> Path | None:
    pattern = f"fold-{site.lower()}_model-t0_exp-e1_baseline_*"
    runs = sorted(runs_root.glob(pattern), key=lambda p: p.stat().st_mtime)
    for run_dir in reversed(runs):
        best = run_dir / "train" / "weights" / "best.pt"
        if best.exists():
            return best
        last = run_dir / "train" / "weights" / "last.pt"
        if last.exists():
            return last
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["smoke", "prep", "e1", "e2", "e3", "e4", "e5", "e6", "e7"], required=True)
    p.add_argument("--env", default=None)
    p.add_argument("--seaclear-root", default=None)
    p.add_argument("--held-out-site", default="Lokrum")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--weights",
        default=None,
        help="For e2/e3/e5/e6/e7: path to detector best.pt",
    )
    p.add_argument("--oracle-gate", default=None, help="E6/E7: oracle_table_gate.csv")
    p.add_argument("--oracle-test", default=None, help="E6/E7: oracle_table_test.csv")
    p.add_argument("--drop-enhanced", action="store_true")
    p.add_argument("--actions", default=None, help="For e3/e4/e5/e6/e7: e.g. T0,T4")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--no-quiet", action="store_true")
    args = p.parse_args()

    py = sys.executable
    env_args = ["--env", args.env] if args.env else []
    sc_args = ["--seaclear-root", args.seaclear_root] if args.seaclear_root else []
    site_args = ["--held-out-site", args.held_out_site]
    max_args = ["--max-images", str(args.max_images)] if args.max_images else []

    def _resolve_e4_weights(env, weights_arg):
        weights = Path(weights_arg) if weights_arg else None
        if weights is not None and Path(str(weights)).exists():
            w = Path(weights)
            if w.is_file():
                return w
            hits = list(w.rglob("best.pt"))
            for h in hits:
                if "e4" in h.name.lower():
                    return h
            if hits:
                return hits[0]
        runs = sorted(
            env.runs_root.glob(f"fold-{args.held_out_site.lower()}_model-mixed_*"),
            key=lambda p: p.stat().st_mtime,
        )
        for r in reversed(runs):
            b = r / "train" / "weights" / "best.pt"
            if b.exists():
                return b
        return None

    # ---- E6 quality selectors ----
    if args.stage == "e6":
        from src.common import load_env

        env = load_env(args.env)
        yolo = env.processed_root / f"yolo_loso_{args.held_out_site.lower()}" / "data.yaml"
        manifest = env.manifests_root / f"loso_{args.held_out_site.lower()}" / "manifest.csv"
        if not yolo.exists() or not manifest.exists():
            print("YOLO/manifests missing; running prep first...", flush=True)
            run(
                [
                    py,
                    "scripts/run_stage.py",
                    "--stage",
                    "prep",
                    *env_args,
                    *sc_args,
                    *site_args,
                ]
            )
        weights = _resolve_e4_weights(env, args.weights)
        if weights is None:
            raise SystemExit("E6 needs E4 mixed weights. Pass --weights /path/to/e4_best.pt")
        cmd = [
            py,
            "scripts/run_quality_selectors.py",
            *env_args,
            *site_args,
            "--weights",
            str(weights),
            "--drop-enhanced",
        ]
        if args.actions:
            cmd += ["--actions", args.actions]
        if args.device:
            cmd += ["--device", args.device]
        if args.oracle_gate:
            cmd += ["--oracle-gate", args.oracle_gate]
        if args.oracle_test:
            cmd += ["--oracle-test", args.oracle_test]
        if args.no_quiet:
            cmd += ["--no-quiet"]
        run(cmd)
        print("E6 complete.")
        return

    # ---- E7 utility gate ----
    if args.stage == "e7":
        from src.common import load_env

        env = load_env(args.env)
        yolo = env.processed_root / f"yolo_loso_{args.held_out_site.lower()}" / "data.yaml"
        manifest = env.manifests_root / f"loso_{args.held_out_site.lower()}" / "manifest.csv"
        if not yolo.exists() or not manifest.exists():
            print("YOLO/manifests missing; running prep first...", flush=True)
            run(
                [
                    py,
                    "scripts/run_stage.py",
                    "--stage",
                    "prep",
                    *env_args,
                    *sc_args,
                    *site_args,
                ]
            )
        weights = _resolve_e4_weights(env, args.weights)
        if weights is None:
            raise SystemExit("E7 needs E4 mixed weights. Pass --weights /path/to/e4_best.pt")
        cmd = [
            py,
            "scripts/train_gate.py",
            *env_args,
            *site_args,
            "--weights",
            str(weights),
            "--drop-enhanced",
        ]
        if args.actions:
            cmd += ["--actions", args.actions]
        if args.device:
            cmd += ["--device", args.device]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch is not None:
            cmd += ["--batch", str(args.batch)]
        if args.oracle_gate:
            cmd += ["--oracle-gate", args.oracle_gate]
        if args.oracle_test:
            cmd += ["--oracle-test", args.oracle_test]
        if args.no_quiet:
            cmd += ["--no-quiet"]
        run(cmd)
        print("E7 complete.")
        return

    # ---- E5 oracle ----
    if args.stage == "e5":
        from src.common import load_env

        env = load_env(args.env)
        # Fresh Save Version sessions have empty /kaggle/working — rebuild data first.
        yolo = env.processed_root / f"yolo_loso_{args.held_out_site.lower()}" / "data.yaml"
        manifest = env.manifests_root / f"loso_{args.held_out_site.lower()}" / "manifest.csv"
        if not yolo.exists() or not manifest.exists():
            print("YOLO/manifests missing; running prep first...", flush=True)
            run(
                [
                    py,
                    "scripts/run_stage.py",
                    "--stage",
                    "prep",
                    *env_args,
                    *sc_args,
                    *site_args,
                ]
            )
        weights = Path(args.weights) if args.weights else None
        if weights is None or not Path(str(weights)).exists():
            # try latest e4 mixed run
            runs = sorted(
                env.runs_root.glob(f"fold-{args.held_out_site.lower()}_model-mixed_*"),
                key=lambda p: p.stat().st_mtime,
            )
            weights = None
            for r in reversed(runs):
                b = r / "train" / "weights" / "best.pt"
                if b.exists():
                    weights = b
                    break
        if weights is None or not Path(weights).exists():
            raise SystemExit(
                "E5 needs E4 mixed weights. Pass --weights /path/to/e4_best.pt "
                "(not E3-T4). Attach your E4 dataset and set E4_WEIGHTS."
            )
        wname = Path(weights).name.lower()
        if "e3" in wname or "t4" in wname and "e4" not in wname and "mixed" not in wname:
            print(
                f"WARNING: weights look like E3/T4 ({weights}). "
                "E5 oracle should use E4 mixed best.pt.",
                flush=True,
            )
        cmd = [
            py,
            "scripts/run_oracle.py",
            *env_args,
            *site_args,
            "--weights",
            str(weights),
            "--drop-enhanced",
        ]
        if args.actions:
            cmd += ["--actions", args.actions]
        if args.device:
            cmd += ["--device", args.device]
        if args.no_quiet:
            cmd += ["--no-quiet"]
        run(cmd)
        print("E5 complete.")
        return

    # ---- E4 mixed-path ----
    if args.stage == "e4":
        from src.common import load_env

        env = load_env(args.env)
        yolo = env.processed_root / f"yolo_loso_{args.held_out_site.lower()}" / "data.yaml"
        if not yolo.exists():
            print("YOLO data missing; running prep first...", flush=True)
            run(
                [
                    py,
                    "scripts/run_stage.py",
                    "--stage",
                    "prep",
                    *env_args,
                    *sc_args,
                    *site_args,
                ]
            )
        cmd = [
            py,
            "scripts/run_mixed_path.py",
            *env_args,
            *site_args,
        ]
        if args.actions:
            cmd += ["--actions", args.actions]
        if args.device:
            cmd += ["--device", args.device]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch is not None:
            cmd += ["--batch", str(args.batch)]
        if args.drop_enhanced:
            cmd += ["--drop-enhanced"]
        if args.no_quiet:
            cmd += ["--no-quiet"]
        run(cmd)
        print("E4 complete.")
        return

    # ---- E3 fixed-path ----
    if args.stage == "e3":
        from src.common import load_env

        env = load_env(args.env)
        weights = Path(args.weights) if args.weights else _latest_e1_weights(
            env.runs_root, args.held_out_site
        )
        yolo = env.processed_root / f"yolo_loso_{args.held_out_site.lower()}" / "data.yaml"
        if not yolo.exists():
            print("YOLO data missing; running prep first...", flush=True)
            run(
                [
                    py,
                    "scripts/run_stage.py",
                    "--stage",
                    "prep",
                    *env_args,
                    *sc_args,
                    *site_args,
                ]
            )
        cmd = [
            py,
            "scripts/run_fixed_path.py",
            *env_args,
            *site_args,
        ]
        if weights and Path(weights).exists():
            cmd += ["--reuse-t0-weights", str(weights)]
            # metrics may sit next to uploaded weights dataset
            metrics_guess = Path(weights).parent.parent.parent / "train" / "eval" / "metrics_all.json"
            if not metrics_guess.exists():
                metrics_guess = Path(weights).parents[2] / "eval" / "metrics_all.json"
            if metrics_guess.exists():
                cmd += ["--reuse-t0-metrics", str(metrics_guess)]
        if args.actions:
            cmd += ["--actions", args.actions]
        if args.device:
            cmd += ["--device", args.device]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch is not None:
            cmd += ["--batch", str(args.batch)]
        if args.drop_enhanced:
            cmd += ["--drop-enhanced"]
        run(cmd)
        print("E3 complete.")
        return

    # ---- E2 only (no retraining) ----
    if args.stage == "e2":
        from src.common import load_env

        env = load_env(args.env)
        weights = Path(args.weights) if args.weights else _latest_e1_weights(
            env.runs_root, args.held_out_site
        )
        if weights is None or not Path(weights).exists():
            raise SystemExit(
                "E2 needs E1 weights. Pass --weights /kaggle/working/runs/.../best.pt"
            )
        cmd = [
            py,
            "scripts/eval_naive_enhance.py",
            *env_args,
            "--weights",
            str(weights),
            *site_args,
            "--splits",
            "val,test",
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.drop_enhanced:
            cmd += ["--drop-enhanced"]
        run(cmd)
        print("E2 complete.")
        return

    if args.stage in ("smoke", "prep", "e1") and not args.skip_audit:
        audit_cmd = [py, "scripts/audit_data.py", *env_args, *sc_args, *max_args, "--no-hashes"]
        if args.stage == "smoke" and not args.max_images:
            audit_cmd += ["--max-images", "100"]
        run(audit_cmd)

    if args.stage in ("smoke", "prep", "e1"):
        split_max = max_args
        if args.stage == "smoke" and not args.max_images:
            split_max = ["--max-images", "100"]
        run(
            [
                py,
                "scripts/build_splits.py",
                *env_args,
                *sc_args,
                *site_args,
                *split_max,
            ]
        )
        run(
            [
                py,
                "scripts/prepare_yolo.py",
                *env_args,
                *sc_args,
                *site_args,
                *split_max,
            ]
        )

    if args.stage == "prep":
        print("Prep complete.")
        return

    exp = "configs/experiments/smoke.yaml" if args.stage == "smoke" else "configs/experiments/e1_baseline.yaml"
    train_cmd = [
        py,
        "scripts/train_detector.py",
        *env_args,
        "--experiment",
        exp,
        *site_args,
    ]
    if args.device:
        train_cmd += ["--device", args.device]
    if args.stage == "smoke":
        train_cmd += ["--epochs", "2", "--batch", "8"]
    run(train_cmd)

    from src.common import load_env

    env = load_env(args.env)
    runs = sorted(
        env.runs_root.glob(f"fold-{args.held_out_site.lower()}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise SystemExit("No run directories found after training")
    run_dir = runs[-1]
    best = run_dir / "train" / "weights" / "best.pt"
    if not best.exists():
        best = run_dir / "train" / "weights" / "last.pt"
    if not best.exists():
        raise SystemExit(f"No weights found under {run_dir}")

    splits = []
    for split in ("val", "test"):
        if _split_has_images(env.processed_root, args.held_out_site, split):
            splits.append(split)
        else:
            print(f"Skipping empty split: {split}", flush=True)
    if not splits:
        print("No non-empty val/test splits to evaluate; training artifacts saved.", flush=True)
        print(f"Done. Artifacts: {run_dir}")
        return

    eval_cmd = [
        py,
        "scripts/evaluate.py",
        *env_args,
        "--weights",
        str(best),
        *site_args,
        "--splits",
        ",".join(splits),
    ]
    if args.device:
        eval_cmd += ["--device", args.device]
    run(eval_cmd)
    print(f"Done. Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
