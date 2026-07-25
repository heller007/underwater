"""E6: non-learned action selectors from UCIQE / UIQM / heuristics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
from scipy import stats

from src.common.quiet import progress
from src.common.run import save_json
from src.detection.train import predict_split
from src.enhancement.descriptors import basic_descriptors, uciqe, uiqm
from src.enhancement.transforms import get_transform
from src.routing.dataset_utils import (
    list_split_images,
    load_oracle_table,
    materialize_selected_split,
    summarize_selection,
    yolo_alias_for_split,
)


SelectorFn = Callable[[dict[str, dict[str, float]], dict[str, float]], str]


def _score_actions(
    bgr: np.ndarray,
    actions: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Per-action quality scores + raw-image descriptors."""
    raw_desc = basic_descriptors(bgr)
    raw_desc["uciqe"] = uciqe(bgr)
    raw_desc["uiqm"] = uiqm(bgr)
    per_action: dict[str, dict[str, float]] = {}
    for a in actions:
        img = bgr if a == "T0" else get_transform(a)(bgr)
        per_action[a] = {
            "uciqe": uciqe(img),
            "uiqm": uiqm(img),
            "contrast": basic_descriptors(img)["contrast"],
            "laplacian_var": basic_descriptors(img)["laplacian_var"],
        }
    return per_action, raw_desc


def selector_always_t0(
    per_action: dict[str, dict[str, float]], raw: dict[str, float]
) -> str:
    return "T0"


def selector_argmax_uciqe(
    per_action: dict[str, dict[str, float]], raw: dict[str, float]
) -> str:
    return max(per_action.keys(), key=lambda a: per_action[a]["uciqe"])


def selector_argmax_uiqm(
    per_action: dict[str, dict[str, float]], raw: dict[str, float]
) -> str:
    return max(per_action.keys(), key=lambda a: per_action[a]["uiqm"])


def make_heuristic_low_quality(
    contrast_thr: float = 0.12,
    uciqe_thr: float = 0.45,
    enhance_action: str = "T4",
) -> SelectorFn:
    def _fn(per_action: dict[str, dict[str, float]], raw: dict[str, float]) -> str:
        if raw.get("contrast", 1.0) < contrast_thr or raw.get("uciqe", 1.0) < uciqe_thr:
            return enhance_action if enhance_action in per_action else "T0"
        return "T0"

    return _fn


SELECTORS: dict[str, SelectorFn] = {
    "always_t0": selector_always_t0,
    "argmax_uciqe": selector_argmax_uciqe,
    "argmax_uiqm": selector_argmax_uiqm,
    "heuristic_low_quality": make_heuristic_low_quality(),
}


def build_quality_table(
    src_yolo_root: Path,
    split: str,
    actions: list[str],
) -> pd.DataFrame:
    rows = []
    images = list_split_images(src_yolo_root, split)
    for i, img_path in enumerate(images):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        per_action, raw = _score_actions(bgr, actions)
        row: dict[str, Any] = {
            "stem": img_path.stem,
            "raw_uciqe": raw["uciqe"],
            "raw_uiqm": raw["uiqm"],
            "raw_contrast": raw["contrast"],
            "raw_laplacian_var": raw["laplacian_var"],
        }
        for a in actions:
            row[f"uciqe_{a}"] = per_action[a]["uciqe"]
            row[f"uiqm_{a}"] = per_action[a]["uiqm"]
        for name, fn in SELECTORS.items():
            row[f"sel_{name}"] = fn(per_action, raw)
        rows.append(row)
        if (i + 1) % 200 == 0:
            progress(f"[e6] scored {split}: {i + 1}/{len(images)}")
    progress(f"[e6] scored {split}: done ({len(rows)})")
    return pd.DataFrame(rows)


def correlation_vs_utility(
    quality_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    actions: list[str],
) -> dict[str, Any]:
    """Correlate Δquality (enhance − raw) with Δutility on shared stems."""
    if len(actions) < 2 or "T0" not in actions:
        return {}
    enhance = [a for a in actions if a != "T0"]
    if not enhance:
        return {}
    a = enhance[0]
    merged = quality_df.merge(oracle_df, on="stem", how="inner")
    out: dict[str, Any] = {"n": int(len(merged)), "enhance_action": a}
    if f"utility_{a}" not in merged.columns or "utility_T0" not in merged.columns:
        return out
    du = merged[f"utility_{a}"] - merged["utility_T0"]
    for metric in ("uciqe", "uiqm"):
        col = f"{metric}_{a}"
        raw_col = f"raw_{metric}"
        if col not in merged.columns or raw_col not in merged.columns:
            continue
        dq = merged[col] - merged[raw_col]
        if len(dq) < 5 or dq.std() == 0 or du.std() == 0:
            out[f"pearson_{metric}_vs_utility"] = None
            out[f"spearman_{metric}_vs_utility"] = None
            continue
        pr = stats.pearsonr(dq, du)
        sr = stats.spearmanr(dq, du)
        out[f"pearson_{metric}_vs_utility"] = float(pr.statistic)
        out[f"pearson_{metric}_p"] = float(pr.pvalue)
        out[f"spearman_{metric}_vs_utility"] = float(sr.statistic)
        out[f"spearman_{metric}_p"] = float(sr.pvalue)
    return out


def run_e6_quality_selectors(
    weights: Path,
    src_yolo_root: Path,
    out_dir: Path,
    actions: list[str],
    splits: list[str],
    device: str,
    conf: float = 0.25,
    oracle_tables: dict[str, Path] | None = None,
    selectors: list[str] | None = None,
    drop_enhanced: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_names = selectors or list(SELECTORS.keys())
    oracle_tables = oracle_tables or {}

    summary: dict[str, Any] = {
        "weights": str(weights),
        "actions": actions,
        "selectors": sel_names,
        "splits": {},
    }

    for split in splits:
        progress(f"[e6] === split {split} ===")
        qdf = build_quality_table(src_yolo_root, split, actions)
        qpath = out_dir / f"quality_table_{split}.csv"
        qdf.to_csv(qpath, index=False)

        ora = None
        ora_path = oracle_tables.get(split)
        if ora_path and Path(ora_path).exists():
            ora = load_oracle_table(Path(ora_path))
            corr = correlation_vs_utility(qdf, ora, actions)
        else:
            corr = {}

        alias = yolo_alias_for_split(split)
        split_summary: dict[str, Any] = {
            "n_images": int(len(qdf)),
            "quality_table": str(qpath),
            "correlations": corr,
            "selectors": {},
        }

        for name in sel_names:
            col = f"sel_{name}"
            if col not in qdf.columns:
                continue
            stem_to_action = dict(zip(qdf["stem"], qdf[col]))
            counts = Counter(stem_to_action.values())
            progress(f"[e6] {split}/{name}: {dict(counts)}")
            sel_stats = summarize_selection(ora, stem_to_action, actions)

            root = out_dir / f"selected_{name}_{split}"
            data_yaml = materialize_selected_split(
                src_yolo_root, root, split, stem_to_action, yolo_alias=alias
            )
            metrics = predict_split(
                weights=weights,
                data_yaml=data_yaml,
                split=alias,
                out_dir=out_dir / f"eval_{name}_{split}",
                device=device,
                conf=conf,
                quiet=True,
            )
            split_summary["selectors"][name] = {
                **sel_stats,
                "metrics": metrics.get("metrics", metrics),
            }
            if drop_enhanced:
                import shutil

                shutil.rmtree(root / "images", ignore_errors=True)

        summary["splits"][split] = split_summary
        # quick leaderboard line
        board = {
            n: s.get("metrics", {}).get("mAP50")
            for n, s in split_summary["selectors"].items()
        }
        progress(f"[e6] {split} mAP50: {board}")

    save_json(out_dir / "e6_results.json", summary)
    return summary
