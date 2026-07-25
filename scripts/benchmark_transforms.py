#!/usr/bin/env python3
"""Benchmark transform latency (CPU; optional GPU path for future)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_env, save_json, setup_logging
from src.enhancement import TRANSFORMS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None, help="Optional sample image path")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--size", type=int, default=640)
    args = p.parse_args()

    env = load_env()
    env.ensure_output_dirs()
    log = setup_logging()

    if args.image and Path(args.image).exists():
        img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    else:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    img = cv2.resize(img, (args.size, args.size))

    results = {}
    for tid, tfm in TRANSFORMS.items():
        for _ in range(args.warmup):
            _ = tfm(img)
        t0 = time.perf_counter()
        for _ in range(args.n):
            _ = tfm(img)
        dt = (time.perf_counter() - t0) / args.n * 1000.0
        results[tid] = {"name": tfm.name, "ms_per_image": dt, "params": tfm.params()}
        log.info("%s (%s): %.3f ms", tid, tfm.name, dt)

    out = env.reports_root / "transform_latency.json"
    save_json(out, {"n": args.n, "size": args.size, "results": results})
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
