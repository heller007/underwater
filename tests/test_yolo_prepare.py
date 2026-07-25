"""Tests for YOLO box conversion."""

from __future__ import annotations

from src.data.yolo_prepare import coco_xywh_to_yolo


def test_coco_to_yolo_center():
    # box covering full 100x100 image
    y = coco_xywh_to_yolo([0, 0, 100, 100], 100, 100)
    assert y is not None
    cx, cy, w, h = y
    assert abs(cx - 0.5) < 1e-6
    assert abs(cy - 0.5) < 1e-6
    assert abs(w - 1.0) < 1e-6
    assert abs(h - 1.0) < 1e-6


def test_invalid_box():
    assert coco_xywh_to_yolo([0, 0, 0, 10], 100, 100) is None
