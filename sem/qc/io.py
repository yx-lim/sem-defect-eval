"""Reading SEM images and their shared acquisition metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import tifffile
from scipy import ndimage as ndi

from sem.qc.config import load_config


_FILE_PATTERN = re.compile(r"^img_(?P<stem>.+)_(?P<view>BSE|Inlens|ETD|SE)\.tif$")


@dataclass(frozen=True)
class StemRecord:
    stem: str
    batch: str
    detector_set: str
    views: Mapping[str, Path]
    height: int
    width: int
    pixel_size_nm: float


def _resolution_value(value: object) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        return float(numerator) / float(denominator)
    return float(value)


def parse_pixel_size_nm(
    path: str | Path,
    bounds_nm: tuple[float, float] | None = None,
) -> float:
    """Parse TIFF horizontal resolution and assert the configured pixel size."""
    config = load_config()
    bounds = bounds_nm or (
        float(config["pixel_size_nm"]["min"]),
        float(config["pixel_size_nm"]["max"]),
    )
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        x_resolution = page.tags.get("XResolution")
        resolution_unit = page.tags.get("ResolutionUnit")
        if x_resolution is None or resolution_unit is None:
            raise ValueError(f"TIFF is missing XResolution/ResolutionUnit: {path}")
        pixels_per_unit = _resolution_value(x_resolution.value)
        unit = int(resolution_unit.value)
    if pixels_per_unit <= 0:
        raise ValueError(f"Invalid XResolution in {path}: {pixels_per_unit}")
    if unit == 2:  # inch
        pixel_size_nm = 25_400_000.0 / pixels_per_unit
    elif unit == 3:  # centimeter
        pixel_size_nm = 10_000_000.0 / pixels_per_unit
    else:
        raise ValueError(f"Unsupported TIFF ResolutionUnit {unit} in {path}")
    if not bounds[0] <= pixel_size_nm <= bounds[1]:
        raise AssertionError(
            f"Pixel size {pixel_size_nm:.6g} nm in {path} is outside "
            f"the required {bounds[0]}–{bounds[1]} nm bounds"
        )
    return pixel_size_nm


def _image_shape(path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(path) as tif:
        shape = tif.pages[0].shape
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[-1] in (3, 4):
        return int(shape[0]), int(shape[1])
    raise ValueError(f"Unsupported TIFF image shape {shape} in {path}")


def list_stems(root: str | Path) -> list[StemRecord]:
    """Discover acquisition stems and validate their co-registered views."""
    root = Path(root)
    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    for path in sorted(root.glob("Batch_*/*.tif")):
        match = _FILE_PATTERN.match(path.name)
        if match is None:
            continue
        batch = path.parent.name
        grouped.setdefault((batch, match.group("stem")), {})[
            match.group("view")
        ] = path

    records = []
    for (batch, stem), views in sorted(grouped.items()):
        if not {"BSE", "Inlens"}.issubset(views):
            raise ValueError(f"{batch}/{stem} is missing a common BSE/Inlens view")
        detector_views = set(views) & {"ETD", "SE"}
        if len(detector_views) != 1:
            raise ValueError(
                f"{batch}/{stem} must have exactly one ETD/SE view; "
                f"found {sorted(detector_views)}"
            )
        bse_shape = _image_shape(views["BSE"])
        pixel_size_nm = parse_pixel_size_nm(views["BSE"])
        for view, path in views.items():
            if _image_shape(path) != bse_shape:
                raise ValueError(f"View {path} is not co-registered with its BSE")
            view_pixel_size = parse_pixel_size_nm(path)
            if not np.isclose(view_pixel_size, pixel_size_nm, atol=1e-6):
                raise ValueError(f"View {path} has a different pixel size")
        records.append(
            StemRecord(
                stem=stem,
                batch=batch,
                detector_set=next(iter(detector_views)),
                views=views,
                height=bse_shape[0],
                width=bse_shape[1],
                pixel_size_nm=pixel_size_nm,
            )
        )
    return records


def _read_rgb(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 TIFF, found {image.dtype}: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected an HxW gray or RGB image in {path}")
    return np.ascontiguousarray(image)


def load_stem(
    rec: StemRecord,
    views: Sequence[str] = ("BSE", "Inlens"),
) -> dict[str, np.ndarray]:
    """Load selected views as uint8 HxW arrays using RGB channel zero."""
    missing = set(views) - set(rec.views)
    if missing:
        raise ValueError(f"{rec.stem} is missing requested views {sorted(missing)}")
    return {view: _read_rgb(rec.views[view]) for view in views}


def valid_mask(rec: StemRecord) -> np.ndarray:
    """Return the shared valid mask, excluding color columns and an 8px border."""
    image = tifffile.imread(rec.views["BSE"])
    if image.ndim == 3:
        color_pixels = np.any(image[..., 1:] != image[..., :1], axis=-1)
    elif image.ndim == 2:
        color_pixels = np.zeros(image.shape, dtype=bool)
    else:
        raise ValueError(f"Expected an HxW gray or RGB image in {rec.views['BSE']}")

    config = load_config()["valid_mask"]
    radius = int(config["color_dilation_radius_px"])
    if radius:
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        structure = (xx * xx + yy * yy) <= radius * radius
        color_pixels = ndi.binary_dilation(color_pixels, structure=structure)
    mask = ~color_pixels
    border = int(config["border_px"])
    if border:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False
    return mask
