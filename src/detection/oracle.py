"""E5 oracle study helpers: predict per action, utilities, oracle dataset, go/no-go."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image

from src.common.quiet import progress
from src.common.run import save_json
from src.detection.naive_enhance import materialize_enhanced_split
from src.detection.train import predict_split
from src.evaluation.image_error import (
    Box,
    image_matching_error,
    oracle_action,
    utility,
    yolo_line_to_xyxy,
)


DEFAULT_COSTS = {"T0": 0.0, "T1": 0.25, "T2": 0.45, "T3": 0.20, "T4": 1.0}


def _load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[Box]:
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append(yolo_line_to_xyxy(cls, cx, cy, w, h, img_w, img_h))
    return boxes


def _predict_folder(
    weights: Path,
    image_dir: Path,
    conf: float,
    device: str,
    imgsz: int = 640,
) -> dict[str, list[Box]]:
    """Run YOLO predict on a folder; return stem -> pred boxes."""
    from ultralytics import YOLO

    from src.common.quiet import silence_ultralytics

    silence_ultralytics()
    model = YOLO(str(weights))
    # stream=True to avoid huge RAM
    results = model.predict(
        source=str(image_dir),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
        stream=True,
    )
    out: dict[str, list[Box]] = {}
    n = 0
    for r in results:
        stem = Path(r.path).stem
        preds: list[Box] = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                preds.append(
                    Box(
                        cls=int(cls[i]),
                        conf=float(confs[i]),
                        xyxy=tuple(map(float, xyxy[i].tolist())),
                    )
                )
        out[stem] = preds
        n += 1
        if n % 300 == 0:
            progress(f"[predict] {image_dir.name}: {n}")
    progress(f"[predict] {image_dir.name}: done ({n})")
    return out


def build_oracle_table(
    stems: list[str],
    gt_dir: Path,
    img_dir_by_action: dict[str, Path],
    preds_by_action: dict[str, dict[str, list[Box]]],
    costs: dict[str, float],
    lam: float,
    delta: float,
    iou_thr: float = 0.5,
) -> pd.DataFrame:
    rows = []
    actions = list(preds_by_action.keys())
    for stem in stems:
        # size from first existing image
        img_w = img_h = 1920
        for a, d in img_dir_by_action.items():
            p = d / f"{stem}.jpg"
            if not p.exists():
                # try any suffix
                hits = list(d.glob(f"{stem}.*"))
                p = hits[0] if hits else p
            if p.exists():
                with Image.open(p) as im:
                    img_w, img_h = im.size
                break
        gts = _load_gt_boxes(gt_dir / f"{stem}.txt", img_w, img_h)
        utils = {}
        errors = {}
        details = {}
        for a in actions:
            preds = preds_by_action[a].get(stem, [])
            m = image_matching_error(preds, gts, iou_thr=iou_thr)
            errors[a] = m["error"]
            utils[a] = utility(m["error"], costs.get(a, 0.0), lam)
            details[a] = m
        ora = oracle_action(utils, raw_action="T0", delta=delta)
        row = {
            "stem": stem,
            "oracle": ora,
            "n_gt": len(gts),
        }
        for a in actions:
            row[f"error_{a}"] = errors[a]
            row[f"utility_{a}"] = utils[a]
            row[f"fp_{a}"] = details[a]["n_fp"]
            row[f"fn_{a}"] = details[a]["n_fn"]
            row[f"tp_{a}"] = details[a]["n_tp"]
        # harm: enhanced worse than raw
        if "T4" in errors and "T0" in errors:
            row["t4_harms"] = int(errors["T4"] > errors["T0"] + 1e-9)
            row["t4_helps"] = int(errors["T4"] < errors["T0"] - 1e-9)
        rows.append(row)
    return pd.DataFrame(rows)


def materialize_oracle_images(
    table: pd.DataFrame,
    src_img_by_action: dict[str, Path],
    src_lbl: Path,
    out_root: Path,
    split_name: str = "test",
) -> Path:
    """Build YOLO dataset where each image comes from its oracle action folder."""
    out_root = Path(out_root)
    img_out = out_root / "images" / split_name
    lbl_out = out_root / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    from src.detection.naive_enhance import _copy_or_link

    for _, row in table.iterrows():
        stem = row["stem"]
        action = row["oracle"]
        src_dir = src_img_by_action[action]
        hits = list(src_dir.glob(f"{stem}.*"))
        if not hits:
            continue
        src = hits[0]
        _copy_or_link(src, img_out / src.name)
        lbl = src_lbl / f"{stem}.txt"
        if lbl.exists():
            _copy_or_link(lbl, lbl_out / f"{stem}.txt")
        else:
            (lbl_out / f"{stem}.txt").write_text("", encoding="utf-8")

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": f"images/{split_name}",
        "val": f"images/{split_name}",
        "test": f"images/{split_name}",
        "names": {0: "debris", 1: "bio", 2: "robot"},
        "nc": 3,
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return yaml_path


def go_no_go(
    oracle_map50: float,
    best_fixed_map50: float,
    action_counts: dict[str, int],
    n_images: int,
) -> dict[str, Any]:
    """
    Softened for 2-action shortlist (T0/T4):
    - continue if oracle - best_fixed >= 0.02 mAP50 OR mean-error improvement compelling
    - both actions used on >= 10% of images
    - no action > 90%
    """
    gap = float(oracle_map50 - best_fixed_map50)
    fracs = {a: c / max(n_images, 1) for a, c in action_counts.items()}
    diversity_ok = all(f >= 0.10 for f in fracs.values()) if len(fracs) >= 2 else False
    not_collapsed = all(f <= 0.90 for f in fracs.values()) if fracs else False
    continue_routing = (gap >= 0.02) and diversity_ok and not_collapsed
    pivot = gap < 0.01
    return {
        "oracle_minus_best_fixed_mAP50": gap,
        "action_fractions": fracs,
        "diversity_ok_ge_10pct_each": diversity_ok,
        "not_collapsed_le_90pct": not_collapsed,
        "decision": "CONTINUE_ROUTING" if continue_routing else ("PIVOT_CALIBRATION" if pivot else "BORDERLINE"),
        "continue_routing": continue_routing,
        "notes": (
            "H1-style target: +2 mAP50 over best fixed on same detector; "
            "with 2 actions require both used on >=10% images."
        ),
    }


def run_e5_oracle(
    weights: Path,
    src_yolo_root: Path,
    out_dir: Path,
    actions: list[str],
    splits: list[str],
    device: str,
    lam: float = 0.05,
    delta: float = 0.01,
    conf: float = 0.25,
    costs: dict[str, float] | None = None,
    drop_enhanced: bool = True,
) -> dict[str, Any]:
    costs = costs or {a: DEFAULT_COSTS.get(a, 0.5) for a in actions}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_yolo_root = Path(src_yolo_root)

    summary: dict[str, Any] = {
        "weights": str(weights),
        "actions": actions,
        "lambda": lam,
        "delta": delta,
        "conf": conf,
        "costs": costs,
        "splits": {},
    }

    for split in splits:
        progress(f"[e5] === split {split} ===")
        # Materialize each action into a YOLO-compatible layout (alias non-standard
        # names like 'gate' to images/val for Ultralytics).
        yolo_alias = "val" if split == "gate" else ("test" if split == "test" else split)
        if yolo_alias not in ("train", "val", "test"):
            yolo_alias = "val"

        img_dirs: dict[str, Path] = {}
        preds: dict[str, dict[str, list[Box]]] = {}
        for a in actions:
            progress(f"[e5] materialize {a}/{split}")
            root = out_dir / f"enhanced_{a}_{split}"
            # Write files under the alias folder name Ultralytics understands
            materialize_enhanced_split(src_yolo_root, root, a, splits=(split,))
            # If source split name != alias, move/rename folder
            src_img = root / "images" / split
            dst_img = root / "images" / yolo_alias
            src_lbl = root / "labels" / split
            dst_lbl = root / "labels" / yolo_alias
            if split != yolo_alias and src_img.exists():
                if dst_img.exists():
                    import shutil

                    shutil.rmtree(dst_img, ignore_errors=True)
                    shutil.rmtree(dst_lbl, ignore_errors=True)
                src_img.rename(dst_img)
                if src_lbl.exists():
                    src_lbl.rename(dst_lbl)
            # Rewrite data.yaml for single-split eval
            data_yaml = {
                "path": str(root.resolve()),
                "train": f"images/{yolo_alias}",
                "val": f"images/{yolo_alias}",
                "test": f"images/{yolo_alias}",
                "names": {0: "debris", 1: "bio", 2: "robot"},
                "nc": 3,
            }
            with open(root / "data.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(data_yaml, f, sort_keys=False)
            img_dirs[a] = root / "images" / yolo_alias
            preds[a] = _predict_folder(weights, img_dirs[a], conf=conf, device=device)

        gt_dir = src_yolo_root / "labels" / split
        stems = sorted({p.stem for p in gt_dir.glob("*.txt")})
        for a, d in img_dirs.items():
            stems = sorted(
                set(stems)
                | {
                    p.stem
                    for p in d.glob("*.*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                }
            )

        progress(f"[e5] utilities for {len(stems)} images")
        table = build_oracle_table(
            stems,
            gt_dir,
            img_dirs,
            preds,
            costs=costs,
            lam=lam,
            delta=delta,
        )
        table_path = out_dir / f"oracle_table_{split}.csv"
        table.to_csv(table_path, index=False)

        counts = Counter(table["oracle"].tolist())
        harm_rate = float(table["t4_harms"].mean()) if "t4_harms" in table.columns else None
        help_rate = float(table["t4_helps"].mean()) if "t4_helps" in table.columns else None

        fixed_metrics = {}
        for a in actions:
            data_yaml = out_dir / f"enhanced_{a}_{split}" / "data.yaml"
            m = predict_split(
                weights=weights,
                data_yaml=data_yaml,
                split=yolo_alias,
                out_dir=out_dir / f"eval_fixed_{a}_{split}",
                device=device,
                quiet=True,
                conf=conf,
            )
            fixed_metrics[a] = m.get("metrics", m)

        progress(f"[e5] materialize oracle images ({split})")
        ora_root = out_dir / f"oracle_ds_{split}"
        ora_yaml = materialize_oracle_images(
            table, img_dirs, gt_dir, ora_root, split_name=yolo_alias
        )
        ora_m = predict_split(
            weights=weights,
            data_yaml=ora_yaml,
            split=yolo_alias,
            out_dir=out_dir / f"eval_oracle_{split}",
            device=device,
            quiet=True,
            conf=conf,
        )
        oracle_metrics = ora_m.get("metrics", ora_m)

        best_fixed = max(
            (
                fixed_metrics[a].get("mAP50", 0.0)
                for a in actions
                if isinstance(fixed_metrics.get(a), dict)
            ),
            default=0.0,
        )
        decision = go_no_go(
            oracle_map50=float(oracle_metrics.get("mAP50", 0.0)),
            best_fixed_map50=float(best_fixed),
            action_counts=dict(counts),
            n_images=len(table),
        )

        split_summary = {
            "n_images": len(table),
            "yolo_alias": yolo_alias,
            "oracle_action_counts": dict(counts),
            "oracle_action_fractions": {
                k: v / max(len(table), 1) for k, v in counts.items()
            },
            "mean_error": {a: float(table[f"error_{a}"].mean()) for a in actions},
            "mean_utility": {a: float(table[f"utility_{a}"].mean()) for a in actions},
            "t4_harm_rate": harm_rate,
            "t4_help_rate": help_rate,
            "fixed_metrics": fixed_metrics,
            "oracle_metrics": oracle_metrics,
            "go_no_go": decision,
            "table_csv": str(table_path),
        }
        summary["splits"][split] = split_summary
        progress(
            f"[e5] {split}: oracle mAP50={oracle_metrics.get('mAP50', 0):.4f} "
            f"best_fixed={best_fixed:.4f} gap={decision['oracle_minus_best_fixed_mAP50']:.4f} "
            f"decision={decision['decision']} counts={dict(counts)}"
        )

        if drop_enhanced:
            import shutil

            for a in actions:
                shutil.rmtree(out_dir / f"enhanced_{a}_{split}" / "images", ignore_errors=True)
            shutil.rmtree(ora_root / "images", ignore_errors=True)

    # Primary decision from test if present else first split
    primary = summary["splits"].get("test") or next(iter(summary["splits"].values()))
    summary["primary_go_no_go"] = primary["go_no_go"]
    save_json(out_dir / "e5_results.json", summary)
    return summary
