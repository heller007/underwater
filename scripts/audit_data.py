#!/usr/bin/env python3
"""E0: Audit SeaClear (and optionally TrashCan presence)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_env, save_json, setup_logging
from src.data import audit_dataset, load_seaclear


def main() -> None:
    p = argparse.ArgumentParser(description="Audit SeaClear dataset")
    p.add_argument("--env", default=None, help="local|kaggle (auto if omitted)")
    p.add_argument("--seaclear-root", default=None, help="Override dataset root")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--no-hashes", action="store_true")
    p.add_argument("--max-hash-images", type=int, default=2000)
    args = p.parse_args()

    env = load_env(args.env)
    env.ensure_output_dirs()
    log = setup_logging()
    root = Path(args.seaclear_root) if args.seaclear_root else env.seaclear_root
    if root is None or not Path(root).exists():
        log.error(
            "SeaClear root not found. Place data under data/raw/seaclear "
            "or attach a Kaggle Dataset and set configs/env/kaggle.yaml."
        )
        sys.exit(1)

    log.info("Loading SeaClear from %s", root)
    ds = load_seaclear(Path(root), max_images=args.max_images)
    log.info("Loaded %d images, %d annotations", len(ds.images), len(ds.annotations))
    audit = audit_dataset(
        ds,
        env.reports_root,
        compute_hashes=not args.no_hashes,
        max_hash_images=args.max_hash_images,
    )
    save_json(env.reports_root / "audit_summary.json", {"pass": audit["pass"], "warnings": audit["warnings"]})
    log.info("Audit pass=%s warnings=%s", audit["pass"], audit["warnings"])
    log.info("Wrote reports to %s", env.reports_root)
    if not audit["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
