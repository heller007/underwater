"""Shared helpers: materialize action-selected YOLO splits; load oracle tables."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml

from src.enhancement.transforms import get_transform


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        try:
            import os

            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def list_split_images(src_yolo_root: Path, split: str) -> list[Path]:
    img_dir = Path(src_yolo_root) / "images" / split
    if not img_dir.exists():
        return []
    return sorted(
        p
        for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def yolo_alias_for_split(split: str) -> str:
    if split == "gate":
        return "val"
    if split in ("train", "val", "test"):
        return split
    return "val"


def load_oracle_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "stem" not in df.columns:
        raise ValueError(f"Oracle table missing 'stem': {path}")
    return df


def materialize_selected_split(
    src_yolo_root: Path,
    out_root: Path,
    split: str,
    stem_to_action: dict[str, str],
    *,
    yolo_alias: str | None = None,
) -> Path:
    """
    Build a YOLO dataset where each image is enhanced with its selected action.
    Writes under images/{yolo_alias} so Ultralytics can val/test it.
    """
    src_yolo_root = Path(src_yolo_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    alias = yolo_alias or yolo_alias_for_split(split)

    with open(src_yolo_root / "data.yaml", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    src_img = src_yolo_root / "images" / split
    src_lbl = src_yolo_root / "labels" / split
    dst_img = out_root / "images" / alias
    dst_lbl = out_root / "labels" / alias
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for img_path in list_split_images(src_yolo_root, split):
        stem = img_path.stem
        action = stem_to_action.get(stem, "T0")
        out_img = dst_img / img_path.name
        lbl_src = src_lbl / f"{stem}.txt"
        lbl_dst = dst_lbl / f"{stem}.txt"
        if not out_img.exists():
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            if action == "T0":
                cv2.imwrite(str(out_img), bgr)
            else:
                enh = get_transform(action)(bgr)
                cv2.imwrite(str(out_img), enh)
        if lbl_src.exists():
            _copy_or_link(lbl_src, lbl_dst)
        else:
            lbl_dst.write_text("", encoding="utf-8")

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": data_cfg.get("train", "images/train"),
        "val": "images/val",
        "test": "images/test",
        "names": data_cfg.get("names", {0: "debris", 1: "bio", 2: "robot"}),
        "nc": data_cfg.get("nc", 3),
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return yaml_path


def action_accuracy(pred: pd.Series, oracle: pd.Series) -> float:
    if len(pred) == 0:
        return 0.0
    return float((pred.values == oracle.values).mean())


def mean_utility_regret(
    table: pd.DataFrame,
    pred_actions: list[str] | pd.Series,
    actions: list[str],
) -> float:
    """Oracle utility minus selected-action utility (higher = worse)."""
    util_cols = {a: f"utility_{a}" for a in actions if f"utility_{a}" in table.columns}
    if not util_cols or "oracle" not in table.columns:
        return float("nan")
    if isinstance(pred_actions, pd.Series):
        preds = pred_actions.tolist()
    else:
        preds = list(pred_actions)
    if len(preds) != len(table):
        return float("nan")
    regrets = []
    for i, (_, row) in enumerate(table.iterrows()):
        a = preds[i]
        ora = row["oracle"]
        u_sel_col = util_cols.get(a)
        u_ora_col = util_cols.get(ora)
        if not u_sel_col or not u_ora_col:
            continue
        regrets.append(float(row[u_ora_col]) - float(row[u_sel_col]))
    return float(sum(regrets) / max(len(regrets), 1))


def harm_help_rates(
    table: pd.DataFrame,
    pred_actions: list[str] | pd.Series,
    enhance_action: str = "T4",
) -> dict[str, float]:
    """Among images where selector picks enhance_action, fraction that harm/help vs T0."""
    if "t4_harms" not in table.columns:
        return {}
    preds = pred_actions.tolist() if isinstance(pred_actions, pd.Series) else list(pred_actions)
    if len(preds) != len(table):
        return {}
    mask = [p == enhance_action for p in preds]
    n = int(sum(mask))
    if n == 0:
        return {"n_enhanced": 0, "harm_rate": 0.0, "help_rate": 0.0}
    sub = table.loc[mask]
    return {
        "n_enhanced": n,
        "harm_rate": float(sub["t4_harms"].mean()),
        "help_rate": float(sub["t4_helps"].mean()),
    }


def summarize_selection(
    table: pd.DataFrame | None,
    stem_to_action: dict[str, str],
    actions: list[str],
) -> dict[str, Any]:
    from collections import Counter

    counts = Counter(stem_to_action.values())
    n = max(sum(counts.values()), 1)
    out: dict[str, Any] = {
        "action_counts": dict(counts),
        "action_fractions": {k: v / n for k, v in counts.items()},
    }
    if table is None or "oracle" not in table.columns:
        return out
    t = table.set_index("stem")
    stems = [s for s in stem_to_action if s in t.index]
    if not stems:
        return out
    pred = [stem_to_action[s] for s in stems]
    ora = t.loc[stems, "oracle"].tolist()
    out["action_accuracy_vs_oracle"] = float(
        sum(int(a == b) for a, b in zip(pred, ora)) / max(len(pred), 1)
    )
    sub = t.loc[stems].reset_index()
    out["mean_utility_regret"] = mean_utility_regret(sub, pred, actions)
    out.update(harm_help_rates(sub, pred))
    return out
