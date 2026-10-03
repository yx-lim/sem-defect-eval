"""Pure metric functions for QC evaluation (spec §4).

Conventions for empty cases: a ratio whose denominator is zero is NaN, never 0.
So a class with no GT and no prediction has precision = recall = F1 = IoU = NaN;
GT but no prediction gives recall 0, F1 0, IoU 0, precision NaN; prediction but
no GT gives precision 0, F1 0, IoU 0, recall NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from sem.qc.schema import CLASS_IDS, IGNORE_LABEL

N_CLASSES = 8
PORE = CLASS_IDS["pore"]
SUBSURFACE = CLASS_IDS["subsurface_uncertain"]
OBJECT_CLASSES = (
    CLASS_IDS["bright_particle"],
    CLASS_IDS["pore"],
    CLASS_IDS["crack_intraparticle"],
    CLASS_IDS["interparticle_gap"],
)
WILSON_Z95 = 1.959963984540054


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def prf_iou(tp: float, fp: float, fn: float) -> dict[str, float]:
    """Precision, recall, F1 and IoU from pooled counts (NaN on 0/0)."""
    return {
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": safe_div(tp, tp + fp + fn),
    }


# --------------------------------------------------------------------------- pixel
def confusion_matrix(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, int]:
    """8x8 raw confusion (GT rows, pred cols) over GT-labelled pixels.

    GT pixels equal to 255 are ignored. GT-labelled pixels predicted as 255 (outside
    the method's valid area) do not fit the 8x8 matrix; their count is returned
    separately and they are scored as false negatives of their GT class.
    """
    if gt.shape != pred.shape:
        raise ValueError(f"GT {gt.shape} and prediction {pred.shape} shapes differ")
    labelled = gt != IGNORE_LABEL
    if np.any(gt[labelled] >= N_CLASSES):
        raise ValueError("GT contains labels outside 0-7/255")
    g = gt[labelled].astype(np.int64)
    p = pred[labelled].astype(np.int64)
    in_range = p < N_CLASSES
    if np.any((p >= N_CLASSES) & (p != IGNORE_LABEL)):
        raise ValueError("Prediction contains labels outside 0-7/255")
    cm = np.bincount(
        g[in_range] * N_CLASSES + p[in_range], minlength=N_CLASSES * N_CLASSES
    ).reshape(N_CLASSES, N_CLASSES)
    return cm, int(np.count_nonzero(~in_range))


def pixel_counts(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-class TP/FP/FN, shape (8, 3).

    GT 255 is ignored everywhere. For the pore class (3), pixels whose GT is
    subsurface_uncertain (4) are excluded: a pore prediction there is neither TP
    nor FP. Prediction 255 on GT-labelled pixels counts as FN of the GT class.
    """
    if gt.shape != pred.shape:
        raise ValueError(f"GT {gt.shape} and prediction {pred.shape} shapes differ")
    labelled = gt != IGNORE_LABEL
    counts = np.zeros((N_CLASSES, 3), dtype=np.int64)
    for c in range(N_CLASSES):
        scored = labelled if c != PORE else labelled & (gt != SUBSURFACE)
        g = (gt == c) & scored
        p = (pred == c) & scored
        counts[c, 0] = np.count_nonzero(g & p)
        counts[c, 1] = np.count_nonzero(p & ~g)
        counts[c, 2] = np.count_nonzero(g & ~p)
    return counts


def row_normalize(cm: np.ndarray) -> np.ndarray:
    sums = cm.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sums > 0, cm / np.where(sums > 0, sums, 1), np.nan)


# --------------------------------------------------------------------------- objects
@dataclass
class ObjectMatch:
    tp: int
    fp: int
    fn: int
    pairs: list[tuple[int, int, float]] = field(default_factory=list)
    unmatched_gt: list[int] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)


def components(mask: np.ndarray, min_area_px: int = 1) -> list[np.ndarray]:
    """8-connected components as boolean masks (same shape as input)."""
    labels, n = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return []
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    return [labels == i for i in range(1, n + 1) if areas[i] >= min_area_px]


def iou_matrix(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(gt_masks), len(pred_masks)), dtype=float)
    if not gt_masks or not pred_masks:
        return out
    # Masks may overlap (agglomerate polygons): pairwise, restricted to bboxes.
    pred_area = np.array([int(m.sum()) for m in pred_masks])
    gt_area = np.array([int(m.sum()) for m in gt_masks])
    pred_boxes = [_bbox(m) for m in pred_masks]
    for i, g in enumerate(gt_masks):
        gb = _bbox(g)
        for j, p in enumerate(pred_masks):
            pb = pred_boxes[j]
            if gb is None or pb is None or not _boxes_overlap(gb, pb):
                continue
            y0, y1 = max(gb[0], pb[0]), min(gb[1], pb[1])
            x0, x1 = max(gb[2], pb[2]), min(gb[3], pb[3])
            inter = int(np.count_nonzero(g[y0:y1, x0:x1] & p[y0:y1, x0:x1]))
            union = gt_area[i] + pred_area[j] - inter
            out[i, j] = inter / union if union else 0.0
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def match_objects(iou: np.ndarray, threshold: float) -> ObjectMatch:
    """One-to-one Hungarian matching maximizing total IoU over eligible pairs.

    Pairs with IoU < threshold are ineligible (weight 0) and never count as matches.
    """
    n_gt, n_pred = iou.shape
    pairs: list[tuple[int, int, float]] = []
    if n_gt and n_pred:
        weights = np.where(iou >= threshold, iou, 0.0)
        rows, cols = linear_sum_assignment(weights, maximize=True)
        pairs = [
            (int(r), int(c), float(iou[r, c]))
            for r, c in zip(rows, cols)
            if iou[r, c] >= threshold and iou[r, c] > 0
        ]
    matched_gt = {p[0] for p in pairs}
    matched_pred = {p[1] for p in pairs}
    return ObjectMatch(
        tp=len(pairs),
        fp=n_pred - len(pairs),
        fn=n_gt - len(pairs),
        pairs=pairs,
        unmatched_gt=[i for i in range(n_gt) if i not in matched_gt],
        unmatched_pred=[j for j in range(n_pred) if j not in matched_pred],
    )


# --------------------------------------------------------------------------- candidates
def wilson_interval(successes: float, n: float, z: float = WILSON_Z95) -> tuple[float, float]:
    """Wilson score interval; (NaN, NaN) when n == 0. Accepts non-integer (effective) n."""
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def weighted_proportion(positive: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Inclusion-weighted proportion with Wilson CI on the Kish effective n."""
    positive = np.asarray(positive, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if positive.size == 0 or weights.sum() <= 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_eff": 0.0}
    w_sum = weights.sum()
    estimate = float((weights * positive).sum() / w_sum)
    n_eff = float(w_sum**2 / (weights**2).sum())
    low, high = wilson_interval(estimate * n_eff, n_eff)
    return {"estimate": estimate, "ci_low": low, "ci_high": high, "n_eff": n_eff}


# --------------------------------------------------------------------------- QC quantities
def quantity_errors(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Error statistics of pred vs GT values; pairs with any NaN are dropped.

    bias = mean(pred - gt); mae = mean(|pred - gt|);
    rel_error = sum|pred - gt| / sum|gt|; rel_bias = sum(pred - gt) / sum|gt|;
    spearman = Spearman rho across units (NaN if fewer than 3 pairs or constant).
    """
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    keep = np.isfinite(gt) & np.isfinite(pred)
    gt, pred = gt[keep], pred[keep]
    n = int(gt.size)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "bias": nan, "mae": nan, "rel_error": nan, "rel_bias": nan,
                "spearman": nan, "gt_mean": nan, "pred_mean": nan}
    diff = pred - gt
    gt_abs = float(np.abs(gt).sum())
    rho = float("nan")
    if n >= 3 and np.ptp(gt) > 0 and np.ptp(pred) > 0:
        rho = float(spearmanr(gt, pred).statistic)
    return {
        "n": n,
        "bias": float(diff.mean()),
        "mae": float(np.abs(diff).mean()),
        "rel_error": safe_div(float(np.abs(diff).sum()), gt_abs),
        "rel_bias": safe_div(float(diff.sum()), gt_abs),
        "spearman": rho,
        "gt_mean": float(gt.mean()),
        "pred_mean": float(pred.mean()),
    }
