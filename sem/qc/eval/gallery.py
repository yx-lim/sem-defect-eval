"""FP/FN galleries: per class top-k unmatched regions by area (spec §4).

Each row: BSE crop | GT overlay | prediction overlay, with the region outlined (ring just outside it).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from sem.qc.eval.evaluate import ErrorRegion
from sem.qc.io import StemRecord, load_stem
from sem.qc.schema import IGNORE_LABEL

CLASS_COLORS = {
    1: (70, 130, 180), 2: (255, 215, 0), 3: (230, 25, 75), 4: (245, 130, 48),
    5: (240, 50, 230), 6: (70, 240, 240), 7: (60, 180, 75), IGNORE_LABEL: (40, 40, 40),
}
ALPHA = 0.45
CAPTION_PX = 14


class BSESource:
    """Loads full BSE views lazily from the data root (one stem cached)."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self._stem: str | None = None
        self._image: np.ndarray | None = None
        self.missing: set[str] = set()

    def get(self, stem: str) -> np.ndarray | None:
        if stem == self._stem:
            return self._image
        self._stem, self._image = stem, None
        paths = sorted(self.data_root.glob(f"Batch_*/img_{stem}_BSE.tif"))
        if not paths:
            self.missing.add(stem)
            return None
        rec = StemRecord(stem, paths[0].parent.name, "", {"BSE": paths[0]}, 0, 0, 0.0)
        try:
            self._image = load_stem(rec, views=("BSE",))["BSE"]
        except Exception:  # LZW without imagecodecs: Pillow reads these TIFFs
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(paths[0]) as img:
                arr = np.array(img)
            self._image = arr[..., 0] if arr.ndim == 3 else arr
        return self._image


def colorize(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(float)
    for class_id, color in CLASS_COLORS.items():
        mask = labels == class_id
        rgb[mask] = (1 - ALPHA) * rgb[mask] + ALPHA * np.array(color, dtype=float)
    return rgb.clip(0, 255).astype(np.uint8)


def _outline(rgb: np.ndarray, mask: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    # Ring just outside the region so thin objects keep their overlay colour.
    edge = ndi.binary_dilation(mask, structure=np.ones((3, 3), bool)) & ~mask
    out = rgb.copy()
    out[edge] = color
    return out


def _window(region: ErrorRegion, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = region.bbox
    pad = max(16, int(0.25 * max(y1 - y0, x1 - x0)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = max((y1 - y0) // 2 + pad, (x1 - x0) // 2 + pad, 24)
    wy0, wx0 = max(0, cy - half), max(0, cx - half)
    wy1, wx1 = min(shape[0], cy + half), min(shape[1], cx + half)
    return wy0, wy1, wx0, wx1


def render_row(
    region: ErrorRegion, gt: np.ndarray, pred: np.ndarray, bse_tile: np.ndarray | None,
    panel_px: int,
) -> Image.Image:
    wy0, wy1, wx0, wx1 = _window(region, gt.shape)
    gray = (
        bse_tile[wy0:wy1, wx0:wx1]
        if bse_tile is not None
        else np.full((wy1 - wy0, wx1 - wx0), 128, dtype=np.uint8)
    )
    mask = np.zeros(gt.shape, dtype=bool)
    y0, y1, x0, x1 = region.bbox
    mask[y0:y1, x0:x1] = region.mask
    mask = mask[wy0:wy1, wx0:wx1]
    panels = [
        _outline(np.repeat(gray[..., None], 3, axis=2), mask),
        _outline(colorize(gray, gt[wy0:wy1, wx0:wx1]), mask),
        _outline(colorize(gray, pred[wy0:wy1, wx0:wx1]), mask),
    ]
    scale = panel_px / max(gray.shape)
    size = (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale)))
    row = Image.new("RGB", (3 * panel_px + 8, panel_px + CAPTION_PX), (0, 0, 0))
    for i, panel in enumerate(panels):
        row.paste(Image.fromarray(panel).resize(size, Image.NEAREST), (i * (panel_px + 4), CAPTION_PX))
    caption = (
        f"{region.kind} {region.class_name} | {region.stem} tile({region.tile_xy[0]},"
        f"{region.tile_xy[1]}) | area {region.area_px}px | best IoU {region.best_iou:.2f}"
        + ("" if bse_tile is not None else " | BSE unavailable")
    )
    ImageDraw.Draw(row).text((2, 1), caption, fill=(255, 255, 255))
    return row


def render_galleries(
    errors: list[ErrorRegion],
    tile_maps: dict[str, tuple[np.ndarray, np.ndarray]],
    data_root: str | Path,
    out_dir: str | Path,
    top_k: int = 10,
    panel_px: int = 200,
) -> list[dict]:
    """Write galleries/<class>_<FP|FN>.png; return an index of rendered regions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[ErrorRegion]] = defaultdict(list)
    for err in errors:
        groups[(err.class_name, err.kind)].append(err)
    selected = {
        key: sorted(regs, key=lambda r: (-r.area_px, r.stem, r.item_id, r.bbox))[:top_k]
        for key, regs in groups.items()
    }
    source = BSESource(data_root)
    rows: dict[tuple[str, str], list[tuple[ErrorRegion, Image.Image]]] = defaultdict(list)
    order = sorted(
        ((key, r) for key, regs in selected.items() for r in regs), key=lambda kr: kr[1].stem
    )
    for key, region in order:
        gt, pred = tile_maps[region.item_id]
        full = source.get(region.stem)
        tx, ty = region.tile_xy
        bse_tile = None
        if full is not None:
            bse_tile = full[ty : ty + gt.shape[0], tx : tx + gt.shape[1]]
            if bse_tile.shape != gt.shape:
                bse_tile = None
        rows[key].append((region, render_row(region, gt, pred, bse_tile, panel_px)))
    index = []
    for (class_name, kind), items in sorted(rows.items()):
        items.sort(key=lambda it: (-it[0].area_px, it[0].stem, it[0].item_id, it[0].bbox))
        height = sum(img.height + 4 for _, img in items)
        sheet = Image.new("RGB", (items[0][1].width, height + 16), (0, 0, 0))
        ImageDraw.Draw(sheet).text(
            (2, 2), f"{kind} {class_name}: BSE | GT overlay | prediction overlay", fill=(255, 255, 0)
        )
        y = 16
        for region, img in items:
            sheet.paste(img, (0, y))
            y += img.height + 4
        path = out_dir / f"{class_name}_{kind}.png"
        sheet.save(path)
        for rank, (region, _) in enumerate(items, start=1):
            index.append({
                "file": path.name, "rank": rank, "kind": kind, "class": class_name,
                "stem": region.stem, "item_id": region.item_id,
                "tile_x0": region.tile_xy[0], "tile_y0": region.tile_xy[1],
                "bbox_y0": region.bbox[0], "bbox_y1": region.bbox[1],
                "bbox_x0": region.bbox[2], "bbox_x1": region.bbox[3],
                "area_px": region.area_px, "best_iou": region.best_iou,
            })
    if source.missing:
        index.append({"file": "", "kind": "warning",
                      "class": f"BSE missing for stems {sorted(source.missing)}"})
    return index
