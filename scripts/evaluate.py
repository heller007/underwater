#!/usr/bin/env python3
"""Evaluate a trained detector on val/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_env, load_yaml, setup_logging
from src.detection import predict_split
from src.evaluation import aggregate_split_metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate detector")
    p.add_argument("--env", default=None)
    p.add_argument("--weights", required=True, help="Path to best.pt")
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--held-out-site", default=None)
    p.add_argument("--splits", default="val,test")
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default=None, help="Output directory (default: beside weights)")
    args = p.parse_args()

    env = load_env(args.env)
    env.ensure_output_dirs()
    log = setup_logging()

    site = args.held_out_site
    if args.data_yaml:
        data_yaml = Path(args.data_yaml)
    else:
        if not site:
            cfg = load_yaml(ROOT / "configs" / "data" / "seaclear.yaml")
            site = cfg.get("dev_held_out_site", "Lokrum")
        data_yaml = env.processed_root / f"yolo_loso_{site.lower()}" / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    weights = Path(args.weights)
    out = Path(args.out) if args.out else weights.parent.parent / "eval"
    device = args.device or env.device
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    results = []
    for split in splits:
        log.info("Evaluating split=%s", split)
        r = predict_split(
            weights=weights,
            data_yaml=data_yaml,
            split=split,
            out_dir=out,
            imgsz=args.imgsz,
            device=device,
        )
        results.append(r)
        log.info("split=%s metrics=%s", split, r.get("metrics"))

    agg = aggregate_split_metrics(results, out / "metrics_all.json")
    log.info("Wrote %s", out / "metrics_all.json")
    print(agg)


if __name__ == "__main__":
    main()
