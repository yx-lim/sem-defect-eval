"""Image loading and crop/context rendering for the review app and VLM."""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from sem.qc.io import list_stems, load_stem


class StemImageLoader:
    """Lazily discover stems once and serve co-registered view arrays."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        self._records: dict[str, Any] | None = None
        self._get = functools.lru_cache(maxsize=6)(self._load_view)

    def _record(self, stem: str):
        if self._records is None:
            self._records = {
                rec.stem: rec for rec in list_stems(self.data_root)
            }
        if stem not in self._records:
            raise KeyError(f"Unknown stem {stem!r} under {self.data_root}")
        return self._records[stem]

    def _load_view(self, stem: str, view: str) -> np.ndarray:
        rec = self._record(stem)
        return load_stem(rec, views=(view,))[view]

    def get(self, stem: str, view: str) -> np.ndarray:
        return self._get(stem, view)


def crop(img: np.ndarray, tile: dict[str, int]) -> np.ndarray:
    """Return the tile sub-array (y0:y0+h, x0:x0+w)."""
    x0, y0 = int(tile["x0"]), int(tile["y0"])
    w, h = int(tile["w"]), int(tile["h"])
    return img[y0 : y0 + h, x0 : x0 + w]


def render_context(
    bse: np.ndarray,
    tile: dict[str, int],
    polygon: list[list[float]] | None = None,
    scale: float = 3.0,
    max_px: int = 512,
) -> np.ndarray:
    """Wider BSE context (RGB) with a red tile rectangle and yellow polygon."""
    img_h, img_w = bse.shape[:2]
    tw, th = int(tile["w"]), int(tile["h"])
    cx, cy = tile["x0"] + tw / 2.0, tile["y0"] + th / 2.0
    rw = max(1024.0, scale * tw)
    rh = max(1024.0, scale * th)
    rx0 = int(round(cx - rw / 2.0))
    ry0 = int(round(cy - rh / 2.0))
    rx0 = min(max(0, rx0), max(0, img_w - int(rw)))
    ry0 = min(max(0, ry0), max(0, img_h - int(rh)))
    rx1 = min(img_w, rx0 + int(rw))
    ry1 = min(img_h, ry0 + int(rh))
    region = bse[ry0:ry1, rx0:rx1]
    factor = min(1.0, max_px / max(region.shape[0], region.shape[1]))
    if factor < 1.0:
        new_w = max(1, int(math.ceil(region.shape[1] * factor)))
        new_h = max(1, int(math.ceil(region.shape[0] * factor)))
        region = np.asarray(
            Image.fromarray(region, mode="L").resize(
                (new_w, new_h), resample=Image.LANCZOS
            )
        )
    rgb = np.stack([region] * 3, axis=-1).astype(np.uint8)
    canvas = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(canvas)

    def _pt(x: float, y: float) -> tuple[float, float]:
        return (x - rx0) * factor, (y - ry0) * factor

    x0, y0 = _pt(tile["x0"], tile["y0"])
    x1, y1 = _pt(tile["x0"] + tw, tile["y0"] + th)
    for offset in (0, 1):
        draw.rectangle(
            [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
            outline=(255, 0, 0),
        )
    if polygon and len(polygon) >= 3:
        pts = [_pt(float(p[0]), float(p[1])) for p in polygon]
        draw.line(pts + [pts[0]], fill=(255, 255, 0), width=1)
    return np.asarray(canvas, dtype=np.uint8)
