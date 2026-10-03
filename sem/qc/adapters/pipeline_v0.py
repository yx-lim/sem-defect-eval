"""Convert pipeline_v0 proposals into QC prediction objects."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import rankdata
from skimage import measure

from sem.qc.schema import Instance, Prediction


CLASSES = [
    "background",
    "crack_intra",
    "crack_inter",
    "void",
    "agglomerate",
    "curtaining",
    "edge_bloom",
    "other_anomaly",
]
EXTRA_LABELS = ["normal", "uncertain"]
SOURCES = {
    "tophat_crack",
    "dark_void",
    "fft_curtain",
    "edge_band",
    "anomaly_peak",
    "microsam",
    "random",
}

CLASS_MAP = {
    "background": (None, None),
    "crack_intra": ("crack_intraparticle", None),
    "crack_inter": ("interparticle_gap", None),
    "void": ("pore", None),
    "agglomerate": ("agglomerate", None),
    "curtaining": ("artifact", "curtaining"),
    "edge_bloom": ("artifact", "charging"),
    "other_anomaly": ("unmapped", None),
    "normal": (None, None),
    "uncertain": ("unmapped", None),
}

SOURCE_MAP = {
    "tophat_crack": ("crack_intraparticle", None),
    "dark_void": ("pore", None),
    "fft_curtain": ("artifact", "curtaining"),
    "edge_band": ("artifact", "charging"),
    "anomaly_peak": ("unmapped", None),
    "random": ("unmapped", None),
    "microsam": ("unmapped", None),
}

_PIXEL_IDS = {
    "pore": 3,
    "crack_intraparticle": 5,
    "interparticle_gap": 6,
    "artifact": 7,
}


def _decode_compressed_counts(value: str | bytes) -> list[int]:
    encoded = value.decode("ascii") if isinstance(value, bytes) else value
    counts = []
    position = 0
    while position < len(encoded):
        number = 0
        shift = 0
        more = True
        while more:
            code = ord(encoded[position]) - 48
            position += 1
            number |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            if not more and code & 0x10:
                number |= -1 << (5 * (shift + 1))
            shift += 1
        if len(counts) > 2:
            number += counts[-2]
        counts.append(number)
    return counts


def decode_mask_rle(mask_rle: Mapping[str, Any]) -> np.ndarray:
    """Decode COCO RLE, including its compressed differential count string."""
    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("COCO RLE must include size=[height, width]")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError("COCO RLE dimensions must be positive")
    if isinstance(counts, (str, bytes)):
        runs = _decode_compressed_counts(counts)
    elif isinstance(counts, (list, tuple)):
        runs = [int(run) for run in counts]
    else:
        raise ValueError("COCO RLE counts must be a list or compressed string")
    flat = np.zeros(height * width, dtype=bool)
    offset = 0
    foreground = False
    for run in runs:
        if run < 0 or offset + run > flat.size:
            raise ValueError("COCO RLE run lengths are invalid")
        if foreground:
            flat[offset : offset + run] = True
        offset += run
        foreground = not foreground
    if offset != flat.size:
        raise ValueError("COCO RLE counts do not cover the requested mask")
    return flat.reshape((height, width), order="F")


def _polygon_from_mask(mask: np.ndarray, x0: int, y0: int) -> list[list[float]]:
    contours = measure.find_contours(np.pad(mask.astype(np.uint8), 1), 0.5)
    if not contours:
        return []
    contour = measure.approximate_polygon(max(contours, key=len), tolerance=1.0)
    return [[float(col - 1 + x0), float(row - 1 + y0)] for row, col in contour]


def _shape_and_valid(
    value: Any, separate_valid: Mapping[str, np.ndarray] | None, stem: str
) -> tuple[tuple[int, int], np.ndarray | None]:
    valid = separate_valid.get(stem) if separate_valid is not None else None
    if isinstance(value, Mapping):
        shape = value.get("shape")
        if shape is None:
            shape = (value["height"], value["width"])
        valid = value.get("valid", valid)
    elif isinstance(value, np.ndarray) and value.dtype == bool:
        shape, valid = value.shape, value
    else:
        shape = value
    if len(shape) != 2:
        raise ValueError(f"Invalid image shape for stem {stem}: {shape}")
    return (int(shape[0]), int(shape[1])), valid


def convert_proposals(
    proposals: list[dict[str, Any]],
    stem_shapes: Mapping[str, Any],
    valid: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Prediction]:
    """Convert a run of proposals and percentile-normalize scores per source."""
    shapes = {
        stem: _shape_and_valid(value, valid, stem)
        for stem, value in stem_shapes.items()
    }
    predictions = {
        stem: Prediction(
            semantic=np.full(shape, 255, dtype=np.uint8),
            instances=[],
            uncertainty=None,
        )
        for stem, (shape, _) in shapes.items()
    }
    raw_by_source: dict[str, list[float]] = defaultdict(list)
    for proposal in proposals:
        raw_by_source[str(proposal["source"])].append(
            float(proposal.get("score", proposal.get("raw_score", 0.0)))
        )
    percentile_by_source: dict[str, dict[float, float]] = {}
    for source, scores in raw_by_source.items():
        ranks = rankdata(scores, method="average")
        percentiles = (
            np.ones(len(scores), dtype=np.float64)
            if len(scores) == 1
            else (ranks - 1.0) / (len(scores) - 1.0)
        )
        percentile_by_source[source] = {}
        for raw, percentile in zip(scores, percentiles):
            percentile_by_source[source].setdefault(float(raw), float(percentile))

    for proposal in proposals:
        stem = str(proposal["group_id"])
        if stem not in predictions:
            raise KeyError(f"No shape supplied for proposal stem {stem}")
        source = str(proposal["source"])
        class_name, subtype = SOURCE_MAP.get(source, ("unmapped", None))
        raw_score = float(proposal.get("score", proposal.get("raw_score", 0.0)))
        score = percentile_by_source[source][raw_score]
        x0, y0, x1, y1 = [int(value) for value in proposal["bbox"]]
        shape, valid_mask = shapes[stem]
        bbox_polygon = [
            [float(x0), float(y0)],
            [float(x1), float(y0)],
            [float(x1), float(y1)],
            [float(x0), float(y1)],
        ]
        rle = proposal.get("mask_rle")
        mask = decode_mask_rle(rle) if rle else None
        if mask is not None:
            expected = (max(0, y1 - y0), max(0, x1 - x0))
            if mask.shape != expected:
                raise ValueError(
                    f"RLE shape {mask.shape} does not match proposal bbox {expected}"
                )
            polygon = _polygon_from_mask(mask, x0, y0)
        else:
            polygon = bbox_polygon
        predictions[stem].instances.append(
            Instance(
                class_name=class_name,
                subtype=subtype,
                bbox=[x0, y0, x1, y1],
                polygon=polygon,
                score=score,
                source=f"pipeline_v0:{source}",
            )
        )
        semantic_id = _PIXEL_IDS.get(class_name)
        if mask is not None and semantic_id is not None:
            height, width = shape
            left, top = max(x0, 0), max(y0, 0)
            right, bottom = min(x1, width), min(y1, height)
            if right > left and bottom > top:
                patch = mask[top - y0 : bottom - y0, left - x0 : right - x0].copy()
                if valid_mask is not None:
                    patch &= np.asarray(valid_mask, dtype=bool)[top:bottom, left:right]
                predictions[stem].semantic[top:bottom, left:right][patch] = semantic_id
    return predictions


def write_proposals_map(
    path: str | Path,
    proposals: list[dict[str, Any]],
    predictions: Mapping[str, Prediction],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_ranks: dict[str, dict[float, float]] = defaultdict(dict)
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    for proposal in proposals:
        grouped_scores[str(proposal["source"])].append(
            float(proposal.get("score", proposal.get("raw_score", 0.0)))
        )
    for source, scores in grouped_scores.items():
        ranks = rankdata(scores, method="average")
        values = np.ones(len(scores)) if len(scores) == 1 else (
            ranks - 1.0
        ) / (len(scores) - 1.0)
        for raw, value in zip(scores, values):
            source_ranks[source].setdefault(float(raw), float(value))
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "proposal_id",
                "stem",
                "source",
                "raw_score",
                "score_pct",
                "class_name",
                "subtype",
            ],
        )
        writer.writeheader()
        for proposal in proposals:
            class_name, subtype = SOURCE_MAP.get(
                str(proposal["source"]), ("unmapped", None)
            )
            writer.writerow(
                {
                    "proposal_id": proposal.get("proposal_id", ""),
                    "stem": proposal["group_id"],
                    "source": proposal["source"],
                    "raw_score": proposal.get("score", proposal.get("raw_score", 0.0)),
                    "score_pct": source_ranks[str(proposal["source"])][
                        float(proposal.get("score", proposal.get("raw_score", 0.0)))
                    ],
                    "class_name": class_name,
                    "subtype": subtype or "",
                }
            )
