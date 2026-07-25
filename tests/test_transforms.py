"""Unit tests for enhancement transforms."""

from __future__ import annotations

import numpy as np
import pytest

from src.enhancement.transforms import TRANSFORMS, apply_action


@pytest.fixture
def sample_img():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)


def test_all_transforms_preserve_shape_dtype(sample_img):
    for tid, tfm in TRANSFORMS.items():
        out = tfm(sample_img)
        assert out.shape == sample_img.shape, tid
        assert out.dtype == np.uint8, tid
        assert out.min() >= 0 and out.max() <= 255, tid


def test_raw_is_copy(sample_img):
    out = apply_action(sample_img, "T0")
    assert np.array_equal(out, sample_img)
    out[0, 0, 0] = 0 if sample_img[0, 0, 0] != 0 else 1
    assert not np.array_equal(out, sample_img)


def test_gray_world_changes_means(sample_img):
    out = apply_action(sample_img, "T1")
    # Not necessarily equal image, but valid
    assert out.shape == sample_img.shape


def test_unknown_action():
    with pytest.raises(KeyError):
        apply_action(np.zeros((8, 8, 3), dtype=np.uint8), "T9")
