"""Synthetic shifts injected into copies of reference stems (positive control)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import ellipse, line_aa

SHIFT_NAMES = ("blur", "brightness", "voids", "cracks")


def _dark_level(img: np.ndarray, valid: np.ndarray, pct: float) -> float:
    return float(np.percentile(img[valid], pct))


def add_blur(views: dict[str, np.ndarray], sigma: float) -> dict[str, np.ndarray]:
    return {
        k: np.clip(np.rint(ndi.gaussian_filter(v.astype(np.float32), sigma)), 0, 255).astype(np.uint8)
        for k, v in views.items()
    }


def add_brightness(views: dict[str, np.ndarray], factor: float) -> dict[str, np.ndarray]:
    return {k: np.clip(np.rint(v.astype(np.float32) * factor), 0, 255).astype(np.uint8) for k, v in views.items()}


def void_mask(
    shape: tuple[int, int],
    valid: np.ndarray,
    target_frac: float,
    rng: np.random.Generator,
    axis_range: tuple[float, float],
) -> np.ndarray:
    """Union of random ellipses inside `valid` covering `target_frac` of valid area (new pixels only)."""
    mask = np.zeros(shape, dtype=bool)
    target = target_frac * float(valid.sum())
    h, w = shape
    for _ in range(100000):
        if mask.sum() >= target:
            break
        a = rng.uniform(*axis_range)
        b = a * rng.uniform(0.4, 1.0)
        cy, cx = rng.uniform(a, h - a), rng.uniform(a, w - a)
        rr, cc = ellipse(cy, cx, b, a, shape=shape, rotation=rng.uniform(0, math.pi))
        keep = valid[rr, cc]
        mask[rr[keep], cc[keep]] = True
    return mask


def crack_mask(
    shape: tuple[int, int],
    valid: np.ndarray,
    n_cracks: int,
    rng: np.random.Generator,
    length_range: tuple[float, float],
    width_px: int,
) -> np.ndarray:
    """Thin random line segments (width ~width_px) inside `valid`."""
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for _ in range(n_cracks):
        length = rng.uniform(*length_range)
        angle = rng.uniform(0, math.pi)
        y0, x0 = rng.uniform(0, h), rng.uniform(0, w)
        y1 = np.clip(y0 + length * math.sin(angle), 0, h - 1)
        x1 = np.clip(x0 + length * math.cos(angle), 0, w - 1)
        rr, cc, _ = line_aa(int(y0), int(x0), int(y1), int(x1))
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        mask[rr[ok], cc[ok]] = True
    if width_px > 1:
        mask = ndi.binary_dilation(mask, iterations=width_px // 2)
    return mask & valid


def paint_dark(
    views: dict[str, np.ndarray], valid: np.ndarray, mask: np.ndarray, pct: float, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Set masked pixels to each view's dark percentile with matched small noise."""
    out = {}
    for k, v in views.items():
        level = _dark_level(v, valid, pct)
        noise = rng.normal(0.0, 2.0, size=int(mask.sum()))
        img = v.copy()
        img[mask] = np.clip(np.rint(level + noise), 0, 255).astype(np.uint8)
        out[k] = img
    return out


def apply_shift(
    name: str,
    views: dict[str, np.ndarray],
    valid: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    pixel_size_nm: float = 25.0,
) -> dict[str, np.ndarray]:
    """Return shifted copies of co-registered views (same geometry applied to every view)."""
    rng = np.random.default_rng(seed)
    if name == "blur":
        return add_blur(views, float(cfg["blur_sigma"]))
    if name == "brightness":
        return add_brightness(views, 1.0 + float(cfg["brightness_delta"]))
    shape = next(iter(views.values())).shape
    if name == "voids":
        mask = void_mask(shape, valid, float(cfg["void_fraction_delta"]), rng,
                         tuple(cfg["void_axis_px"]))
        return paint_dark(views, valid, mask, float(cfg["dark_percentile"]), rng)
    if name == "cracks":
        area_mm2 = float(valid.sum()) * (pixel_size_nm / 1000.0) ** 2 / 1e6
        n = max(1, int(round(float(cfg["cracks_per_mm2"]) * area_mm2)))
        mask = crack_mask(shape, valid, n, rng, tuple(cfg["crack_length_px"]), int(cfg["crack_width_px"]))
        return paint_dark(views, valid, mask, float(cfg["dark_percentile"]), rng)
    raise ValueError(f"unknown shift {name}")
