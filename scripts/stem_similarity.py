#!/usr/bin/env python3
"""Find pairwise field overlap using zero-padded phase correlation."""

from __future__ import annotations

import csv
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import fft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sem.qc.config import load_config
from sem.qc.io import StemRecord, list_stems, valid_mask


def _quarter_image(record: StemRecord, scale: float) -> tuple[np.ndarray, np.ndarray]:
    raw = tifffile.imread(record.views["BSE"])
    gray = raw[..., 0] if raw.ndim == 3 else raw
    mask = valid_mask(record)
    size = (
        max(1, int(round(gray.shape[1] * scale))),
        max(1, int(round(gray.shape[0] * scale))),
    )
    quarter = np.asarray(
        Image.fromarray(gray).resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    quarter_mask = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            size, Image.Resampling.NEAREST
        )
    ) > 0
    return quarter, quarter_mask


def _phase_correlation_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_valid: np.ndarray,
    moving_valid: np.ndarray,
) -> tuple[int, int]:
    """Return the integer (dy, dx) shift to apply to moving to align reference."""
    reference_values = reference[reference_valid]
    moving_values = moving[moving_valid]
    if reference_values.size == 0 or moving_values.size == 0:
        return 0, 0
    ref_mean = float(reference_values.mean())
    mov_mean = float(moving_values.mean())
    ref_std = float(reference_values.std())
    mov_std = float(moving_values.std())
    ref = np.where(reference_valid, (reference - ref_mean) / max(ref_std, 1e-6), 0)
    mov = np.where(moving_valid, (moving - mov_mean) / max(mov_std, 1e-6), 0)
    shape = (
        fft.next_fast_len(reference.shape[0] + moving.shape[0] - 1),
        fft.next_fast_len(reference.shape[1] + moving.shape[1] - 1),
    )
    ref_fft = fft.rfftn(ref, shape)
    mov_fft = fft.rfftn(mov, shape)
    cross_power = ref_fft * np.conj(mov_fft)
    magnitude = np.abs(cross_power)
    cross_power /= np.maximum(magnitude, np.finfo(np.float32).eps)
    correlation = fft.irfftn(cross_power, shape)
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), shape)
    shift_y = int(peak_y) - shape[0] if peak_y >= reference.shape[0] else int(peak_y)
    shift_x = int(peak_x) - shape[1] if peak_x >= reference.shape[1] else int(peak_x)
    return shift_y, shift_x


def _overlap_slices(
    ref_shape: tuple[int, int],
    moving_shape: tuple[int, int],
    shift: tuple[int, int],
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    dy, dx = shift
    ref_y0, ref_y1 = max(0, dy), min(ref_shape[0], moving_shape[0] + dy)
    ref_x0, ref_x1 = max(0, dx), min(ref_shape[1], moving_shape[1] + dx)
    mov_y0, mov_y1 = ref_y0 - dy, ref_y1 - dy
    mov_x0, mov_x1 = ref_x0 - dx, ref_x1 - dx
    if ref_y1 <= ref_y0 or ref_x1 <= ref_x0:
        return (slice(0, 0), slice(0, 0)), (slice(0, 0), slice(0, 0))
    return (
        (slice(ref_y0, ref_y1), slice(ref_x0, ref_x1)),
        (slice(mov_y0, mov_y1), slice(mov_x0, mov_x1)),
    )


def _refined_ncc(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_valid: np.ndarray,
    moving_valid: np.ndarray,
    shift: tuple[int, int],
) -> tuple[float, float]:
    """Compute valid-pixel NCC and overlap fraction at the phase-correlation shift."""
    ref_slices, mov_slices = _overlap_slices(
        reference.shape, moving.shape, shift
    )
    overlap_valid = (
        reference_valid[ref_slices] & moving_valid[mov_slices]
    )
    overlap_pixels = int(np.count_nonzero(overlap_valid))
    denominator = min(
        int(reference_valid.sum()), int(moving_valid.sum())
    )
    overlap_fraction = overlap_pixels / denominator if denominator else 0.0
    if overlap_pixels < 2:
        return 0.0, overlap_fraction
    ref_values = reference[ref_slices][overlap_valid].astype(np.float64)
    mov_values = moving[mov_slices][overlap_valid].astype(np.float64)
    ref_values -= ref_values.mean()
    mov_values -= mov_values.mean()
    norm = float(np.linalg.norm(ref_values) * np.linalg.norm(mov_values))
    ncc = float(np.dot(ref_values, mov_values) / norm) if norm else 0.0
    return ncc, overlap_fraction


def _histogram_similarity(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_valid: np.ndarray,
    moving_valid: np.ndarray,
) -> float:
    ref_hist = np.histogram(
        reference[reference_valid], bins=256, range=(0, 256)
    )[0].astype(np.float64)
    mov_hist = np.histogram(
        moving[moving_valid], bins=256, range=(0, 256)
    )[0].astype(np.float64)
    if ref_hist.sum() == 0 or mov_hist.sum() == 0:
        return 0.0
    ref_hist /= ref_hist.sum()
    mov_hist /= mov_hist.sum()
    return float(np.sqrt(ref_hist * mov_hist).sum())


def _heatmap(path: Path, stems: list[str], matrix: np.ndarray) -> None:
    cell = 20
    left = 48
    top = 48
    size = left + cell * len(stems) + 16
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for row in range(len(stems)):
        draw.text((12, top + row * cell + 4), str(row), fill=(30, 30, 30))
        draw.text((left + row * cell + 5, 14), str(row), fill=(30, 30, 30))
        for col in range(len(stems)):
            value = float(np.clip(matrix[row, col], -1.0, 1.0))
            if value >= 0:
                color = (
                    int(255 - 215 * value),
                    int(255 - 220 * value),
                    int(255 - 220 * value),
                )
            else:
                strength = -value
                color = (
                    int(255 - 225 * strength),
                    int(255 - 175 * strength),
                    255,
                )
            x0 = left + col * cell
            y0 = top + row * cell
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), fill=color)
    image.save(path)


def run() -> list[dict[str, Any]]:
    config = load_config()
    data_root = Path(config["paths"]["data_root"])
    output_root = Path(config["paths"]["work_root"]) / "similarity"
    output_root.mkdir(parents=True, exist_ok=True)
    scale = float(config["tile_sizes"]["similarity_scale"])
    records = list_stems(data_root)
    if len(records) != 31:
        raise RuntimeError(f"Expected 31 stems in {data_root}, found {len(records)}")

    arrays = [_quarter_image(record, scale) for record in records]
    rows = []
    pair_rows = []
    for (index_a, record_a), (index_b, record_b) in combinations(
        enumerate(records), 2
    ):
        image_a, valid_a = arrays[index_a]
        image_b, valid_b = arrays[index_b]
        coarse_shift = _phase_correlation_shift(image_a, image_b, valid_a, valid_b)
        ncc, overlap_fraction = _refined_ncc(
            image_a, image_b, valid_a, valid_b, coarse_shift
        )
        pair_rows.append(
            {
                "stem_a": record_a.stem,
                "batch_a": record_a.batch,
                "stem_b": record_b.stem,
                "batch_b": record_b.batch,
                "ncc": ncc,
                "overlap_fraction": overlap_fraction,
                "shift_y_px": int(round(coarse_shift[0] / scale)),
                "shift_x_px": int(round(coarse_shift[1] / scale)),
                "histogram_similarity": _histogram_similarity(
                    image_a, image_b, valid_a, valid_b
                ),
            }
        )
    pair_rows.sort(key=lambda row: row["ncc"], reverse=True)
    csv_path = output_root / "pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    matrix = np.eye(len(records), dtype=np.float32)
    index_by_key = {(record.batch, record.stem): i for i, record in enumerate(records)}
    for pair in pair_rows:
        index_a = index_by_key[(pair["batch_a"], pair["stem_a"])]
        index_b = index_by_key[(pair["batch_b"], pair["stem_b"])]
        matrix[index_a, index_b] = pair["ncc"]
        matrix[index_b, index_a] = pair["ncc"]
    stems = [f"{record.batch}/{record.stem}" for record in records]
    _heatmap(output_root / "heatmap.png", stems, matrix)
    with (output_root / "heatmap_stems.json").open(
        "w", encoding="utf-8"
    ) as stems_file:
        json.dump(stems, stems_file, indent=2)
    return pair_rows


if __name__ == "__main__":
    for row in run()[:15]:
        print(
            f"{row['stem_a']} {row['stem_b']} NCC={row['ncc']:.4f} "
            f"overlap={row['overlap_fraction']:.3f} "
            f"shift=({row['shift_x_px']},{row['shift_y_px']})"
        )
