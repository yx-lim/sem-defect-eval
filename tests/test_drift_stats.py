import math

import numpy as np
import pytest

from sem.qc.drift import stats


def test_robust_z_mad_std_and_degenerate():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z, kind = stats.robust_z(np.array([6.0, 6.0]), ref)
    assert kind == "mad"
    assert z == pytest.approx((6 - 3) / (1.4826 * 1.0))
    ref_mad0 = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    z, kind = stats.robust_z(np.array([2.0]), ref_mad0)
    assert kind == "std"
    assert z == pytest.approx(1.0 / np.std(ref_mad0, ddof=1))
    z, kind = stats.robust_z(np.array([2.0]), np.ones(5))
    assert kind == "degenerate" and math.isnan(z)
    z, kind = stats.robust_z(np.array([np.nan]), ref)
    assert kind == "degenerate"


def test_hedges_g_known_value():
    x = np.array([2.0, 4.0, 6.0])
    y = np.array([1.0, 2.0, 3.0])
    sp = math.sqrt((2 * 4.0 + 2 * 1.0) / 4)
    j = 1 - 3 / (4 * 6 - 9)
    assert stats.hedges_g(x, y) == pytest.approx((4 - 2) / sp * j)


def test_holm_hand_values_and_nan():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert stats.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    out = stats.holm([0.02, np.nan, 0.5])
    assert out[0] == pytest.approx(0.04) and math.isnan(out[1]) and out[2] == pytest.approx(0.5)
    assert stats.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])
    assert stats.holm([0.2, 0.01]) == pytest.approx([0.2, 0.02])


def test_bootstrap_ci_contains_point_and_deterministic():
    rng = np.random.default_rng(1)
    x, y = rng.normal(1, 1, 9), rng.normal(0, 1, 7)
    a = stats.bootstrap_median_diff_ci(x, y, 500, 0)
    b = stats.bootstrap_median_diff_ci(x, y, 500, 0)
    assert a == b
    assert a[1] <= a[0] <= a[2]


def test_permutation_p_bounds_and_shift_detected():
    n_perm = 999
    x = np.arange(7) + 100.0
    y = np.arange(7, dtype=float)
    p = stats.permutation_p_median_diff(x, y, n_perm, 0)
    assert 1 / (n_perm + 1) <= p <= 1
    assert p < 0.05
    p_same = stats.permutation_p_median_diff(y, y, n_perm, 0)
    assert p_same == pytest.approx(1.0)


def test_rank_permutation_full_resolution_7v7():
    # complete separation: only 2 of C(14,7)=3432 assignments are as extreme
    p = stats.permutation_p_rank(np.arange(7) + 100.0, np.arange(7, dtype=float), 10000, 0)
    assert p < 0.002
    p_med = stats.permutation_p_median_diff(np.arange(7) + 100.0, np.arange(7, dtype=float), 10000, 0)
    assert p_med == pytest.approx(40 / 3432, abs=0.004)


def test_permutation_p_roughly_uniform_under_null():
    rng = np.random.default_rng(42)
    ps = np.array([
        fn(rng.normal(size=7), rng.normal(size=7), 199, seed=i)
        for i in range(200)
        for fn in (stats.permutation_p_rank, stats.permutation_p_median_diff)
    ])
    assert ((ps >= 1 / 200) & (ps <= 1)).all()
    assert 0.0 <= np.mean(ps < 0.05) <= 0.12
    assert 0.35 <= np.mean(ps) <= 0.65


def test_mmd_identical_near_zero_and_shift_larger():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = rng.normal(size=(200, 3))
    shifted = y + 2.0
    bw = stats.median_heuristic_bandwidth(np.vstack([x, y]))
    same = stats.mmd2_unbiased(x, y, bw)
    diff = stats.mmd2_unbiased(x, shifted, bw)
    assert abs(same) < 0.02
    assert diff > 10 * abs(same) and diff > 0.1
    assert abs(stats.mmd2_unbiased(x, x.copy(), bw)) < 0.02


def test_mmd_permutation_test_detects_and_deterministic():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(7, 4))
    inc = rng.normal(size=(7, 4)) + 3
    a = stats.mmd_permutation_test(ref, inc, 999, 0)
    b = stats.mmd_permutation_test(ref, inc, 999, 0)
    assert a == b
    assert a["p"] < 0.01
    null = stats.mmd_permutation_test(ref, rng.normal(size=(7, 4)), 999, 0)
    assert null["p"] > 0.01


def test_ledoit_wolf_mahalanobis_finite_with_n_lt_p():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(7, 18))
    inc = rng.normal(size=(5, 18))
    ref[:, 3] = 1.0  # degenerate column is dropped
    res = stats.mahalanobis_test(ref, inc)
    assert np.isfinite(res["d2_incoming"]).all()
    assert np.isfinite(res["d2_null_loo"]).all() and len(res["d2_null_loo"]) == 7
    assert res["n_features_used"] == 17
    far = stats.mahalanobis_test(ref, inc + 50)
    assert far["frac_above_null"] == 1.0


def test_block_permutation_keeps_stem_tiles_together():
    tile_stem = np.repeat(np.arange(6), [3, 1, 4, 2, 5, 2])
    stem_is_inc = np.array([0, 0, 0, 1, 1, 1], bool)
    labels = stats.block_permutation_labels(tile_stem, stem_is_inc, 300, 0)
    for row in labels:
        for s in range(6):
            assert len(set(row[tile_stem == s])) == 1
        per_stem = np.array([row[tile_stem == s][0] for s in range(6)])
        assert per_stem.sum() == 3
    assert len({tuple(r) for r in labels}) > 5


def test_tile_mmd_block_detects_shift_and_deterministic():
    rng = np.random.default_rng(3)
    stems = np.repeat(np.arange(8), 6)
    is_inc = np.array([0] * 4 + [1] * 4, bool)
    tiles = rng.normal(size=(48, 3)) + is_inc[stems][:, None] * 3.0
    a = stats.tile_mmd_block_test(tiles, stems, is_inc, 500, 0)
    assert a == stats.tile_mmd_block_test(tiles, stems, is_inc, 500, 0)
    # only C(8,4)=70 block assignments; observed is the most extreme pair (with its mirror)
    assert a["p"] < 0.06
