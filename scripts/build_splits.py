#!/usr/bin/env python3
"""Build leakage-resistant LOSO split manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import list_input_datasets, load_env, load_yaml, setup_logging
from src.data import build_all_loso, build_loso_fold, load_seaclear, write_fold_manifests


def main() -> None:
    p = argparse.ArgumentParser(description="Build LOSO manifests")
    p.add_argument("--env", default=None)
    p.add_argument("--seaclear-root", default=None)
    p.add_argument("--held-out-site", default=None, help="If set, only this fold")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

    env = load_env(args.env)
    env.ensure_output_dirs()
    log = setup_logging()
    cfg = load_yaml(ROOT / "configs" / "data" / "seaclear.yaml")
    seed = args.seed if args.seed is not None else int(cfg.get("split_seed", 0))
    ratios = cfg.get("split_ratios")

    root = Path(args.seaclear_root) if args.seaclear_root else env.seaclear_root
    if root is None or not Path(root).exists():
        log.error("SeaClear root not found: %s", root)
        for line in list_input_datasets(env.data_root):
            log.error("  input: %s", line)
        log.error("Attach SeaClear via Add Data, or pass --seaclear-root.")
        sys.exit(1)

    held = args.held_out_site or cfg.get("dev_held_out_site")
    ds = load_seaclear(Path(root), max_images=args.max_images, held_out_site=held)

    if held:
        site = held
        log.info("Building single fold held_out=%s seed=%s", site, seed)
        df = build_loso_fold(ds, site, seed=seed, ratios=ratios)
        paths = write_fold_manifests(df, env.manifests_root, site)
        log.info("Wrote %s (%d images)", paths["full"], len(df))
    else:
        log.info("Building all LOSO folds seed=%s", seed)
        summary = build_all_loso(
            ds,
            sites=cfg.get("sites"),
            seed=seed,
            ratios=ratios,
            manifests_root=env.manifests_root,
        )
        log.info("Folds: %s", list(summary["folds"].keys()))


if __name__ == "__main__":
    main()
