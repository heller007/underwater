"""Per-image detection error (LRP-inspired) and oracle utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Box:
    cls: int
    conf: float
    xyxy: tuple[float, float, float, float]  # x1,y1,x2,y2


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def yolo_line_to_xyxy(cls: int, cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Box:
    bw, bh = w * img_w, h * img_h
    x1 = (cx * img_w) - bw / 2.0
    y1 = (cy * img_h) - bh / 2.0
    x2 = x1 + bw
    y2 = y1 + bh
    return Box(cls=cls, conf=1.0, xyxy=(x1, y1, x2, y2))


def image_matching_error(
    preds: list[Box],
    gts: list[Box],
    iou_thr: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """
    Class-wise greedy matching at IoU >= iou_thr.
    e = (sum_TP (1-IoU) + N_FP + N_FN) / (N_TP + N_FP + N_FN + eps)

    Empty-frame convention: no GT and no FP -> error 0; no GT but has FP -> error 1.
    """
    if not gts and not preds:
        return {"error": 0.0, "n_tp": 0, "n_fp": 0, "n_fn": 0, "sum_one_minus_iou": 0.0}
    if not gts and preds:
        return {"error": 1.0, "n_tp": 0, "n_fp": len(preds), "n_fn": 0, "sum_one_minus_iou": 0.0}

    # Match per class
    classes = sorted({b.cls for b in gts} | {b.cls for b in preds})
    n_tp = n_fp = n_fn = 0
    sum_1miou = 0.0

    for c in classes:
        gt_c = [b for b in gts if b.cls == c]
        pr_c = sorted([b for b in preds if b.cls == c], key=lambda b: b.conf, reverse=True)
        matched_gt = set()
        for pb in pr_c:
            best_j, best_iou = -1, 0.0
            for j, gb in enumerate(gt_c):
                if j in matched_gt:
                    continue
                iou = iou_xyxy(pb.xyxy, gb.xyxy)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_thr:
                matched_gt.add(best_j)
                n_tp += 1
                sum_1miou += 1.0 - best_iou
            else:
                n_fp += 1
        n_fn += len(gt_c) - len(matched_gt)

    denom = n_tp + n_fp + n_fn + eps
    err = (sum_1miou + n_fp + n_fn) / denom
    return {
        "error": float(err),
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "sum_one_minus_iou": float(sum_1miou),
    }


def utility(error: float, cost_norm: float, lam: float) -> float:
    return float(1.0 - error - lam * cost_norm)


def oracle_action(
    utilities: dict[str, float],
    raw_action: str = "T0",
    delta: float = 0.01,
) -> str:
    """Pick max utility; prefer raw if best enhanced gain < delta."""
    if not utilities:
        return raw_action
    best = max(utilities.keys(), key=lambda a: utilities[a])
    if best == raw_action:
        return raw_action
    gain = utilities[best] - utilities.get(raw_action, utilities[best])
    if gain < delta:
        return raw_action
    return best
