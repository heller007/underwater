#!/usr/bin/env python3
"""Prepare YOLO dataset dirs for a LOSO fold (symlinks/hardlinks preferred)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_env, load_yaml, setup_logging
from src.data import load_seaclear, prepare_yolo_fold


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare YOLO fold")
    p.add_argument("--env", default=None)
    p.add_argument("--seaclear-root", default=None)
    p.add_argument("--held-out-site", default=None)
    p.add_argument("--manifest", default=None, help="Path to fold manifest.csv")
    p.add_argument("--out", default=None, help="Output processed dir")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--copy", action="store_true", help="Force copy instead of symlink/hardlink")
    p.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to materialize",
    )
    args = p.parse_args()

    env = load_env(args.env)
    env.ensure_output_dirs()
    log = setup_logging()
    cfg = load_yaml(ROOT / "configs" / "data" / "seaclear.yaml")
    site = args.held_out_site or cfg.get("dev_held_out_site", "Lokrum")

    root = Path(args.seaclear_root) if args.seaclear_root else env.seaclear_root
    if root is None or not Path(root).exists():
        log.error("SeaClear root not found: %s", root)
        sys.exit(1)

    if args.manifest:
        manifest = Path(args.manifest)
    else:
        manifest = env.manifests_root / f"loso_{site.lower()}" / "manifest.csv"
    if not manifest.exists():
        log.error("Manifest not found: %s (run build_splits.py first)", manifest)
        sys.exit(1)

    out = Path(args.out) if args.out else env.processed_root / f"yolo_loso_{site.lower()}"
    ds = load_seaclear(Path(root), max_images=args.max_images)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    meta = prepare_yolo_fold(
        ds,
        manifest,
        out,
        splits=splits,
        use_symlinks=not args.copy,
    )
    log.info("Prepared YOLO dataset: %s", meta)


if __name__ == "__main__":
    main()
