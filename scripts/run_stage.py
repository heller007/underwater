#!/usr/bin/env python3
"""
Stage runner for Kaggle / local.

Stages:
  smoke  - audit (optional) + tiny train
  prep   - audit + splits + yolo prepare
  e1     - prep (if needed) + full baseline train + eval
  e2     - naive test-time enhancement on frozen E1 weights
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
    print("+", " ".join(cmd), flush=True)
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
    p.add_argument("--stage", choices=["smoke", "prep", "e1", "e2"], required=True)
    p.add_argument("--env", default=None)
    p.add_argument("--seaclear-root", default=None)
    p.add_argument("--held-out-site", default="Lokrum")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--weights",
        default=None,
        help="For e2: path to E1 best.pt (auto-discovers latest e1 run if omitted)",
    )
    p.add_argument("--drop-enhanced", action="store_true")
    args = p.parse_args()

    py = sys.executable
    env_args = ["--env", args.env] if args.env else []
    sc_args = ["--seaclear-root", args.seaclear_root] if args.seaclear_root else []
    site_args = ["--held-out-site", args.held_out_site]
    max_args = ["--max-images", str(args.max_images)] if args.max_images else []

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
