"""Unit tests for image matching error / oracle tie rule."""

from __future__ import annotations

from src.evaluation.image_error import Box, image_matching_error, oracle_action, utility


def test_empty_frame_ok():
    m = image_matching_error([], [])
    assert m["error"] == 0.0


def test_empty_frame_fp():
    preds = [Box(0, 0.9, (0, 0, 10, 10))]
    m = image_matching_error(preds, [])
    assert m["error"] == 1.0


def test_perfect_match():
    gt = [Box(0, 1.0, (0, 0, 10, 10))]
    pr = [Box(0, 0.8, (0, 0, 10, 10))]
    m = image_matching_error(pr, gt)
    assert m["n_tp"] == 1
    assert m["error"] < 1e-6


def test_oracle_tie_to_raw():
    u = {"T0": 0.80, "T4": 0.805}
    assert oracle_action(u, delta=0.01) == "T0"
    assert oracle_action(u, delta=0.001) == "T4"


def test_utility_cost():
    assert utility(0.2, 1.0, 0.05) == 0.75
