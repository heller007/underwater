"""Image quality descriptors and underwater metrics (UCIQE / UIQM approximations)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def basic_descriptors(image: np.ndarray) -> dict[str, float]:
    """Cheap per-image descriptors for the utility gate."""
    img = image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    b, g, r = cv2.split(img)
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    means = rgb.reshape(-1, 3).mean(axis=0)
    stds = rgb.reshape(-1, 3).std(axis=0)
    eps = 1e-6
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1.0)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum()) if len(hist) else 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = float(hsv[:, :, 1].mean() / 255.0)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return {
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
        "rg_ratio": float(means[0] / (means[1] + eps)),
        "bg_ratio": float(means[2] / (means[1] + eps)),
        "luminance_mean": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "entropy": entropy,
        "saturation": sat,
        "laplacian_var": float(lap.var()),
    }


def uciqe(image: np.ndarray) -> float:
    """
    UCIQE (Yang & Sowmya): c1*σ_c + c2*con_l + c3*μ_s in Lab-ish / chroma space.
    Implemented in a commonly used OpenCV-friendly form.
    """
    img = image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    # chroma
    chroma = np.sqrt(a * a + b * b)
    sigma_c = float(np.std(chroma))
    # saturation proxy in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    mu_s = float(np.mean(sat))
    # contrast of luminance: difference of 1st/99th percentile
    l_norm = l / 255.0
    con_l = float(np.percentile(l_norm, 99) - np.percentile(l_norm, 1))
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return float(c1 * sigma_c + c2 * con_l + c3 * mu_s)


def _uicm(img: np.ndarray) -> float:
    rgb = _bgr_to_rgb(img).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = r - g
    yb = (r + g) / 2.0 - b
    # intensity contrast metrics
    mu_rg, sigma_rg = float(np.mean(rg)), float(np.std(rg))
    mu_yb, sigma_yb = float(np.mean(yb)), float(np.std(yb))
    return -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + 0.1586 * np.sqrt(sigma_rg**2 + sigma_yb**2)


def _uism(img: np.ndarray) -> float:
    # Enhancement measure of sharpness via Sobel on each channel
    rgb = _bgr_to_rgb(img).astype(np.float32)
    score = 0.0
    for c in range(3):
        ch = rgb[:, :, c]
        sx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        sobel = np.sqrt(sx * sx + sy * sy)
        score += float(np.mean(sobel))
    return score / 3.0 / 255.0


def _uiconm(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # PLIP-inspired local contrast on non-overlapping blocks
    k = 5
    h, w = gray.shape
    vals = []
    for i in range(0, h - k + 1, k):
        for j in range(0, w - k + 1, k):
            patch = gray[i : i + k, j : j + k]
            mx, mn = float(patch.max()), float(patch.min())
            if mx > mn:
                vals.append((mx - mn) / (mx + mn + 1e-6))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def uiqm(image: np.ndarray) -> float:
    """UIQM (Panetta et al.) linear combination of UICM, UISM, UIConM."""
    img = image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # Coefficients from literature
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return float(c1 * _uicm(img) + c2 * _uism(img) + c3 * _uiconm(img))


def all_descriptors(image: np.ndarray) -> dict[str, float]:
    d = basic_descriptors(image)
    d["uciqe"] = uciqe(image)
    d["uiqm"] = uiqm(image)
    return d


def descriptor_vector(image: np.ndarray) -> tuple[list[str], np.ndarray]:
    d = all_descriptors(image)
    keys = sorted(d.keys())
    return keys, np.array([d[k] for k in keys], dtype=np.float32)
