import itertools
import math

import numpy as np
import pytest
from skimage.draw import disk

from sem.qc.eval.bootstrap import bootstrap_ci, stem_multiplicities
from sem.qc.eval.metrics import (
    components,
    confusion_matrix,
    iou_matrix,
    match_objects,
    pixel_counts,
    prf_iou,
    quantity_errors,
    weighted_proportion,
    wilson_interval,
)


def test_perfect_pixel_prediction_scores_one():
    gt = np.zeros((20, 20), np.uint8)
    gt[2:8, 2:8] = 3
    gt[10:15, 10:15] = 2
    counts = pixel_counts(gt, gt.copy())
    for c in (0, 2, 3):
        assert prf_iou(*counts[c]) == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}


def test_shifted_square_hand_computed_pixel_and_object_iou():
    gt = np.zeros((30, 30), np.uint8)
    pred = np.zeros_like(gt)
    gt[5:15, 5:15] = 3
    pred[5:15, 10:20] = 3  # shifted 5 px: intersection 50, union 150
    tp, fp, fn = pixel_counts(gt, pred)[3]
    assert (tp, fp, fn) == (50, 50, 50)
    assert prf_iou(tp, fp, fn) == pytest.approx(
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "iou": 1 / 3}
    )
    iou = iou_matrix(components(gt == 3), components(pred == 3))
    assert iou[0, 0] == pytest.approx(1 / 3)
    assert match_objects(iou, 0.5).tp == 0  # pore/bright threshold
    assert match_objects(iou, 0.3).tp == 1  # crack/gap/agglomerate threshold


def test_shifted_disc_iou_matches_pixel_count_and_lens_area():
    gt = np.zeros((80, 80), bool)
    pred = np.zeros_like(gt)
    r, d = 15, 6
    gt[disk((40, 40), r)] = True
    pred[disk((40, 40 + d), r)] = True
    iou = iou_matrix([gt], [pred])[0, 0]
    assert iou == pytest.approx((gt & pred).sum() / (gt | pred).sum())
    lens = 2 * r * r * math.acos(d / (2 * r)) - d / 2 * math.sqrt(4 * r * r - d * d)
    analytic = lens / (2 * math.pi * r * r - lens)
    assert iou == pytest.approx(analytic, abs=0.02)


def test_empty_class_edge_cases_are_nan_not_zero():
    nan_all = prf_iou(0, 0, 0)
    assert all(math.isnan(v) for v in nan_all.values())
    gt_only = prf_iou(0, 0, 7)
    assert math.isnan(gt_only["precision"]) and gt_only["recall"] == 0
    assert gt_only["f1"] == 0 and gt_only["iou"] == 0
    pred_only = prf_iou(0, 4, 0)
    assert pred_only["precision"] == 0 and math.isnan(pred_only["recall"])
    empty = match_objects(np.zeros((0, 0)), 0.5)
    assert (empty.tp, empty.fp, empty.fn) == (0, 0, 0)


def test_confusion_sums_and_ignore_handling():
    rng = np.random.default_rng(1)
    gt = rng.integers(0, 8, (40, 40)).astype(np.uint8)
    gt[:5] = 255
    pred = rng.integers(0, 8, (40, 40)).astype(np.uint8)
    pred[10:12] = 255
    cm, outside = confusion_matrix(gt, pred)
    labelled = gt != 255
    assert cm.sum() + outside == labelled.sum()
    assert outside == np.count_nonzero(labelled & (pred == 255))
    in_range = labelled & (pred != 255)
    for c in range(8):
        assert cm[c].sum() == np.count_nonzero(in_range & (gt == c))
    assert cm.trace() == np.count_nonzero(in_range & (gt == pred))


def test_gt_subsurface_excluded_from_pore_scoring_only():
    gt = np.zeros((10, 10), np.uint8)
    gt[0:4, 0:4] = 4
    gt[5:7, 5:7] = 3
    pred = np.zeros_like(gt)
    pred[0:4, 0:4] = 3
    pred[5:7, 5:7] = 3
    counts = pixel_counts(gt, pred)
    assert tuple(counts[3]) == (4, 0, 0)  # pore on GT-4 is neither TP nor FP
    assert tuple(counts[4]) == (0, 0, 16)  # GT-4 still scored for its own class


def _brute_force(weights, threshold):
    n_gt, n_pred = weights.shape
    best = (0, 0.0)
    for k in range(min(n_gt, n_pred) + 1):
        for rows in itertools.combinations(range(n_gt), k):
            for cols in itertools.permutations(range(n_pred), k):
                vals = [weights[r, c] for r, c in zip(rows, cols)]
                if all(v >= threshold for v in vals):
                    best = max(best, (k, float(sum(vals))), key=lambda b: (b[1], b[0]))
    return best


@pytest.mark.parametrize("seed", range(25))
def test_hungarian_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
    iou = rng.random(shape) * (rng.random(shape) < 0.7)
    for threshold in (0.3, 0.5):
        result = match_objects(iou, threshold)
        n, total = _brute_force(iou, threshold)
        assert sum(p[2] for p in result.pairs) == pytest.approx(total)
        assert result.tp == n
        assert result.tp + result.fp == shape[1] and result.tp + result.fn == shape[0]


def test_wilson_known_values():
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson_interval(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert wilson_interval(81, 263) == pytest.approx((0.2553, 0.3662), abs=1e-4)
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_weighted_proportion_inclusion_weights():
    unweighted = weighted_proportion(np.array([1, 1, 0, 0]), np.ones(4))
    assert unweighted["estimate"] == 0.5 and unweighted["n_eff"] == 4
    assert (unweighted["ci_low"], unweighted["ci_high"]) == pytest.approx(wilson_interval(2, 4))
    # positives sampled at 1/3 inclusion probability (weight 3), negatives at 1
    weighted = weighted_proportion(np.array([1, 1, 0, 0]), np.array([3, 3, 1, 1]))
    assert weighted["estimate"] == pytest.approx(6 / 8)
    assert weighted["n_eff"] == pytest.approx(64 / 20)


def test_bootstrap_resamples_stems_not_tiles():
    # Stem A: 10 tiles all TP; stem B: 10 tiles all FP. Pooled precision 0.5.
    tp = np.array([10, 0])
    fp = np.array([0, 10])

    def stat(mult):
        return {"precision": (mult @ tp) / (mult @ (tp + fp))}

    ci = bootstrap_ci(2, stat, n_reps=2000, seed=0)["precision"]
    # Stem resampling draws {A,A} or {B,B} with p=1/4 each -> CI spans [0, 1].
    assert ci["ci_low"] == 0.0 and ci["ci_high"] == 1.0
    # Tile resampling of the 20 tiles would give a narrow CI around 0.5.
    rng = np.random.default_rng(0)
    tiles = np.r_[np.ones(10), np.zeros(10)]
    tile_reps = [tiles[rng.integers(0, 20, 20)].mean() for _ in range(2000)]
    lo, hi = np.percentile(tile_reps, [2.5, 97.5])
    assert lo > 0.2 and hi < 0.8

    mult = stem_multiplicities(5, 100, 0)
    assert mult.shape == (100, 5) and (mult.sum(axis=1) == 5).all()


def test_bootstrap_reports_nan_fraction():
    ci = bootstrap_ci(2, lambda m: {"x": float("nan") if m[0] == 2 else 1.0}, 400, 0)["x"]
    assert 0.1 < ci["nan_frac"] < 0.4 and ci["ci_low"] == 1.0


def test_quantity_error_signs_and_definitions():
    gt = np.array([0.10, 0.20, 0.30, np.nan])
    pred = np.array([0.12, 0.25, 0.33, 0.5])
    errs = quantity_errors(gt, pred)
    assert errs["n"] == 3
    assert errs["bias"] == pytest.approx(0.1 / 3)  # over-estimate -> positive bias
    assert errs["mae"] == pytest.approx(0.1 / 3)
    assert errs["rel_error"] == pytest.approx(0.1 / 0.6)
    assert errs["spearman"] == pytest.approx(1.0)
    under = quantity_errors(gt[:3], gt[:3] - 0.05)
    assert under["bias"] == pytest.approx(-0.05) and under["rel_bias"] < 0
    assert math.isnan(quantity_errors([], [])["bias"])
