"""Deterministic underwater enhancement transforms (T0–T4)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class Transform(ABC):
    id: str
    name: str

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply transform. Image is BGR uint8 HxWx3 (OpenCV convention)."""

    def params(self) -> dict:
        return {"id": self.id, "name": self.name}


def _ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {image.shape}")
    return image


class RawTransform(Transform):
    id = "T0"
    name = "raw"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return _ensure_bgr_u8(image).copy()


class GrayWorldTransform(Transform):
    id = "T1"
    name = "gray_world"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = _ensure_bgr_u8(image).astype(np.float32)
        means = img.reshape(-1, 3).mean(axis=0)
        means = np.maximum(means, 1e-6)
        gray = float(means.mean())
        scale = gray / means
        out = img * scale.reshape(1, 1, 3)
        return np.clip(out, 0, 255).astype(np.uint8)


class LabClaheTransform(Transform):
    id = "T2"
    name = "lab_clahe"

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def params(self) -> dict:
        return {**super().params(), "clip_limit": self.clip_limit, "tile_grid_size": list(self.tile_grid_size)}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = _ensure_bgr_u8(image)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


class AdaptiveGammaTransform(Transform):
    id = "T3"
    name = "adaptive_gamma"

    def __init__(self, gamma_min: float = 0.6, gamma_max: float = 1.8, target_mean: float = 0.5):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.target_mean = target_mean
        self.last_gamma: float | None = None

    def params(self) -> dict:
        return {
            **super().params(),
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "target_mean": self.target_mean,
        }

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = _ensure_bgr_u8(image)
        # luminance from BGR
        y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mean = float(np.mean(y))
        mean = max(mean, 1e-6)
        # gamma such that mean^gamma ≈ target
        gamma = np.log(max(self.target_mean, 1e-6)) / np.log(mean)
        gamma = float(np.clip(gamma, self.gamma_min, self.gamma_max))
        self.last_gamma = gamma
        table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img, table)


class FusionTransform(Transform):
    """
    Lightweight fusion inspired by Ancuti et al. / SeaClear fusion idea:
    blend white-balanced and contrast-enhanced inputs with simple weights.
    Deterministic, no external MATLAB dependency.
    """

    id = "T4"
    name = "fusion"

    def __init__(self):
        self.wb = GrayWorldTransform()
        self.clahe = LabClaheTransform(clip_limit=2.0, tile_grid_size=(8, 8))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = _ensure_bgr_u8(image)
        wb = self.wb(img).astype(np.float32)
        ct = self.clahe(img).astype(np.float32)
        # Salience-like weights from local contrast (Laplacian energy)
        def weight(x: np.ndarray) -> np.ndarray:
            g = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            lap = cv2.Laplacian(g, cv2.CV_32F)
            w = np.abs(lap)
            w = cv2.GaussianBlur(w, (5, 5), 0) + 1e-6
            return w

        w1 = weight(wb)
        w2 = weight(ct)
        wsum = w1 + w2
        w1, w2 = w1 / wsum, w2 / wsum
        out = wb * w1[..., None] + ct * w2[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


TRANSFORMS: dict[str, Transform] = {
    "T0": RawTransform(),
    "T1": GrayWorldTransform(),
    "T2": LabClaheTransform(),
    "T3": AdaptiveGammaTransform(),
    "T4": FusionTransform(),
}


def get_transform(action_id: str) -> Transform:
    if action_id not in TRANSFORMS:
        raise KeyError(f"Unknown action {action_id}; expected one of {list(TRANSFORMS)}")
    return TRANSFORMS[action_id]


def apply_action(image: np.ndarray, action_id: str) -> np.ndarray:
    return get_transform(action_id)(image)


def list_actions() -> list[dict]:
    return [t.params() for t in TRANSFORMS.values()]
