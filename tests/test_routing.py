"""Smoke tests for E6 selectors and E7 gate forward pass."""

from __future__ import annotations

import numpy as np
import torch

from src.routing.gate import CNNGate, CombinedGate, DescriptorGate, build_gate
from src.routing.quality_selectors import (
    SELECTORS,
    selector_argmax_uciqe,
    selector_always_t0,
)


def test_selectors_prefer_higher_uciqe():
    per = {
        "T0": {"uciqe": 0.3, "uiqm": 1.0, "contrast": 0.1, "laplacian_var": 1.0},
        "T4": {"uciqe": 0.9, "uiqm": 0.2, "contrast": 0.2, "laplacian_var": 2.0},
    }
    raw = {"uciqe": 0.3, "uiqm": 1.0, "contrast": 0.1}
    assert selector_always_t0(per, raw) == "T0"
    assert selector_argmax_uciqe(per, raw) == "T4"
    assert "argmax_uiqm" in SELECTORS


def test_gate_forward_shapes():
    B, D, K = 4, 15, 2
    img = torch.randn(B, 3, 160, 160)
    desc = torch.randn(B, D)
    for kind in ("descriptor", "cnn", "combined"):
        m = build_gate(kind, D, K)
        out = m(img, desc)
        assert out.shape == (B, K)
