"""Statistical primitives for stem-level batch change detection (spec §5)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.covariance import LedoitWolf

MAD_SCALE = 1.4826


def robust_scale(values: np.ndarray) -> tuple[float, str]:
    """Return (scale, kind) with kind in {"mad", "std", "degenerate"}."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, "degenerate"
    mad = MAD_SCALE * float(np.median(np.abs(values - np.median(values))))
    if mad > 0:
        return mad, "mad"
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if std > 0:
        return std, "std"
    return math.nan, "degenerate"


def robust_z(incoming: np.ndarray, reference: np.ndarray) -> tuple[float, str]:
    """(median_inc - median_ref) / (1.4826 * MAD_ref); MAD=0 -> std; both 0 -> degenerate."""
    incoming = np.asarray(incoming, dtype=float)
    reference = np.asarray(reference, dtype=float)
    incoming = incoming[np.isfinite(incoming)]
    reference = reference[np.isfinite(reference)]
    if incoming.size == 0 or reference.size == 0:
        return math.nan, "degenerate"
    scale, kind = robust_scale(reference)
    if kind == "degenerate":
        return math.nan, kind
    return (float(np.median(incoming)) - float(np.median(reference))) / scale, kind


def hedges_g(incoming: np.ndarray, reference: np.ndarray) -> float:
    """Bias-corrected standardized mean difference (incoming - reference)."""
    x = np.asarray(incoming, dtype=float)
    y = np.asarray(reference, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        return math.nan
    pooled = ((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (
        n1 + n2 - 2
    )
    if pooled <= 0:
        return math.nan
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float((x.mean() - y.mean()) / math.sqrt(pooled) * correction)


def bootstrap_median_diff_ci(
    incoming: np.ndarray,
    reference: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Median difference with a percentile CI; stems resampled within each group."""
    x = np.asarray(incoming, dtype=float)
    y = np.asarray(reference, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    bx = np.median(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1)
    by = np.median(y[rng.integers(0, y.size, size=(n_boot, y.size))], axis=1)
    diffs = bx - by
    alpha = (1.0 - level) / 2.0
    return (
        float(np.median(x) - np.median(y)),
        float(np.quantile(diffs, alpha)),
        float(np.quantile(diffs, 1.0 - alpha)),
    )


def permutation_label_matrix(
    n_ref: int, n_inc: int, n_perm: int, seed: int = 0
) -> np.ndarray:
    """Boolean (n_perm, n_ref+n_inc) matrix; True marks units labelled incoming."""
    rng = np.random.default_rng(seed)
    n = n_ref + n_inc
    order = np.argsort(rng.random((n_perm, n)), axis=1)
    labels = np.zeros((n_perm, n), dtype=bool)
    rows = np.repeat(np.arange(n_perm), n_inc)
    labels[rows, order[:, :n_inc].ravel()] = True
    return labels


def permutation_p_median_diff(
    incoming: np.ndarray,
    reference: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided stem-label permutation p for |median_inc - median_ref|.

    p = (1 + #{T* >= T}) / (1 + n_perm), so p lies in [1/(n_perm+1), 1].
    """
    x = np.asarray(incoming, dtype=float)
    y = np.asarray(reference, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return math.nan
    pooled = np.concatenate([y, x])
    observed = abs(float(np.median(x) - np.median(y)))
    labels = permutation_label_matrix(y.size, x.size, n_perm, seed)
    values = np.broadcast_to(pooled, labels.shape)
    inc = np.sort(values[labels].reshape(n_perm, x.size), axis=1)
    ref = np.sort(values[~labels].reshape(n_perm, y.size), axis=1)
    stats = np.abs(np.median(inc, axis=1) - np.median(ref, axis=1))
    exceed = int(np.count_nonzero(stats >= observed - 1e-12 * max(1.0, observed)))
    return (1.0 + exceed) / (1.0 + n_perm)


def permutation_p_rank(
    incoming: np.ndarray,
    reference: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided stem-label permutation p for |mean rank_inc - mean rank_ref| (Mann-Whitney form).

    Average ranks for ties; p = (1 + #{T* >= T}) / (1 + n_perm).
    """
    from scipy.stats import rankdata

    x = np.asarray(incoming, dtype=float)
    y = np.asarray(reference, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return math.nan
    ranks = rankdata(np.concatenate([y, x]))
    observed = abs(ranks[y.size:].mean() - ranks[: y.size].mean())
    labels = permutation_label_matrix(y.size, x.size, n_perm, seed).astype(float)
    inc_mean = labels @ ranks / x.size
    ref_mean = (1.0 - labels) @ ranks / y.size
    stats_ = np.abs(inc_mean - ref_mean)
    exceed = int(np.count_nonzero(stats_ >= observed - 1e-9))
    return (1.0 + exceed) / (1.0 + n_perm)


def permutation_p(
    incoming: np.ndarray,
    reference: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
    statistic: str = "rank_sum",
) -> float:
    if statistic == "rank_sum":
        return permutation_p_rank(incoming, reference, n_perm, seed)
    if statistic == "median_diff":
        return permutation_p_median_diff(incoming, reference, n_perm, seed)
    raise ValueError(f"unknown permutation statistic {statistic}")


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values; NaNs stay NaN and are not counted."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    if m == 0:
        return adjusted
    order = finite[np.argsort(p[finite], kind="mergesort")]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def reference_standardizer(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column reference median and robust scale; returns (center, scale, usable)."""
    reference = np.asarray(reference, dtype=float)
    center = np.nanmedian(reference, axis=0)
    scales = np.array([robust_scale(reference[:, j])[0] for j in range(reference.shape[1])])
    usable = np.isfinite(scales) & np.isfinite(center)
    return center, scales, usable


def _impute(z: np.ndarray) -> np.ndarray:
    z = np.array(z, dtype=float)
    z[~np.isfinite(z)] = 0.0
    return z


def mahalanobis_d2(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """D² of query rows w.r.t. reference rows.

    Columns are standardized by reference median/MAD (MAD=0 -> std); degenerate
    columns are dropped. Missing values are imputed at the reference median (z=0).
    Covariance: sklearn LedoitWolf on the standardized reference rows (finite for n<p).
    """
    reference = np.asarray(reference, dtype=float)
    query = np.atleast_2d(np.asarray(query, dtype=float))
    center, scale, usable = reference_standardizer(reference)
    cols = list(np.flatnonzero(usable))
    if not cols:
        return np.full(query.shape[0], np.nan), cols
    zr = _impute((reference[:, cols] - center[cols]) / scale[cols])
    zq = _impute((query[:, cols] - center[cols]) / scale[cols])
    lw = LedoitWolf().fit(zr)
    return lw.mahalanobis(zq), cols


def mahalanobis_test(reference: np.ndarray, incoming: np.ndarray, pct: float = 95.0) -> dict:
    """Incoming D², leave-one-reference-stem-out null D² and its percentile threshold."""
    reference = np.asarray(reference, dtype=float)
    incoming = np.atleast_2d(np.asarray(incoming, dtype=float))
    d2_inc, cols = mahalanobis_d2(reference, incoming)
    null = []
    for i in range(reference.shape[0]):
        rest = np.delete(reference, i, axis=0)
        d2, _ = mahalanobis_d2(rest, reference[i : i + 1])
        null.append(float(d2[0]))
    null_arr = np.asarray(null)
    threshold = float(np.nanpercentile(null_arr, pct)) if np.isfinite(null_arr).any() else math.nan
    above = d2_inc > threshold
    return {
        "d2_incoming": d2_inc.tolist(),
        "d2_null_loo": null_arr.tolist(),
        "null_threshold": threshold,
        "null_pct": pct,
        "above_null": above.tolist(),
        "frac_above_null": float(np.mean(above)) if above.size else math.nan,
        "n_features_used": len(cols),
        "feature_indices_used": [int(c) for c in cols],
    }


def pooled_standardize(x: np.ndarray) -> np.ndarray:
    """Label-free standardization (pooled median/MAD), NaN imputed to 0."""
    x = np.asarray(x, dtype=float)
    center = np.nanmedian(x, axis=0)
    scale = np.array([robust_scale(x[:, j])[0] for j in range(x.shape[1])])
    keep = np.isfinite(scale) & np.isfinite(center)
    return _impute((x[:, keep] - center[keep]) / scale[keep])


def median_heuristic_bandwidth(z: np.ndarray, max_points: int = 2000, seed: int = 0) -> float:
    z = np.asarray(z, dtype=float)
    if z.shape[0] > max_points:
        z = z[np.random.default_rng(seed).choice(z.shape[0], max_points, replace=False)]
    d2 = _sq_dists(z)
    iu = np.triu_indices(z.shape[0], k=1)
    med = float(np.sqrt(np.median(d2[iu]))) if iu[0].size else 0.0
    return med if med > 0 else 1.0


def _sq_dists(z: np.ndarray) -> np.ndarray:
    sq = np.sum(z * z, axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * z @ z.T, 0.0)


def rbf_kernel(z: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-_sq_dists(z) / (2.0 * bandwidth * bandwidth))


def mmd2_from_kernel(kernel: np.ndarray, is_inc: np.ndarray) -> np.ndarray:
    """Unbiased MMD² for one (n,) or many (P, n) boolean label vectors."""
    labels = np.atleast_2d(is_inc).astype(float)
    a, b = labels, 1.0 - labels
    m, n = a.sum(1), b.sum(1)
    diag = np.diag(kernel)
    ka, kb = a @ kernel, b @ kernel
    if np.any(m < 2) or np.any(n < 2):
        raise ValueError("unbiased MMD² needs >=2 units in each group")
    xx = ((ka * a).sum(1) - a @ diag) / (m * (m - 1))
    yy = ((kb * b).sum(1) - b @ diag) / (n * (n - 1))
    xy = (ka * b).sum(1) / (m * n)
    out = xx + yy - 2.0 * xy
    return out if np.ndim(is_inc) > 1 else out[:1]


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, bandwidth: float | None = None) -> float:
    z = np.vstack([np.asarray(x, float), np.asarray(y, float)])
    bw = median_heuristic_bandwidth(z) if bandwidth is None else bandwidth
    labels = np.zeros(z.shape[0], bool)
    labels[: len(x)] = True
    return float(mmd2_from_kernel(rbf_kernel(z, bw), labels)[0])


def mmd_permutation_test(
    reference: np.ndarray, incoming: np.ndarray, n_perm: int = 10000, seed: int = 0
) -> dict:
    """Stem-level unbiased MMD² (RBF, median heuristic on pooled standardized vectors)."""
    z = pooled_standardize(np.vstack([reference, incoming]))
    n_ref = len(reference)
    bw = median_heuristic_bandwidth(z)
    kernel = rbf_kernel(z, bw)
    observed_labels = np.zeros(z.shape[0], bool)
    observed_labels[n_ref:] = True
    observed = float(mmd2_from_kernel(kernel, observed_labels)[0])
    perms = permutation_label_matrix(n_ref, len(incoming), n_perm, seed)
    null = mmd2_from_kernel(kernel, perms)
    p = (1.0 + np.count_nonzero(null >= observed - 1e-12)) / (1.0 + n_perm)
    return {"mmd2": observed, "p": float(p), "bandwidth": bw, "n_perm": n_perm,
            "n_ref": n_ref, "n_inc": len(incoming)}


def block_permutation_labels(
    tile_stem_index: np.ndarray, stem_is_inc: np.ndarray, n_perm: int, seed: int = 0
) -> np.ndarray:
    """Permute stem->group labels; every tile inherits its stem's label."""
    stem_is_inc = np.asarray(stem_is_inc, bool)
    perms = permutation_label_matrix(
        int((~stem_is_inc).sum()), int(stem_is_inc.sum()), n_perm, seed
    )
    return perms[:, np.asarray(tile_stem_index, int)]


def tile_mmd_block_test(
    tiles: np.ndarray,
    tile_stem_index: np.ndarray,
    stem_is_inc: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> dict:
    """Tile-level unbiased MMD² with stem-block permutation."""
    z = pooled_standardize(tiles)
    bw = median_heuristic_bandwidth(z, seed=seed)
    kernel = rbf_kernel(z, bw)
    stem_is_inc = np.asarray(stem_is_inc, bool)
    observed_labels = stem_is_inc[np.asarray(tile_stem_index, int)]
    observed = float(mmd2_from_kernel(kernel, observed_labels)[0])
    labels = block_permutation_labels(tile_stem_index, stem_is_inc, n_perm, seed)
    null = np.concatenate([mmd2_from_kernel(kernel, chunk) for chunk in np.array_split(labels, max(1, n_perm // 1000))])
    p = (1.0 + np.count_nonzero(null >= observed - 1e-12)) / (1.0 + n_perm)
    return {"mmd2": observed, "p": float(p), "bandwidth": bw, "n_perm": n_perm,
            "n_tiles": int(z.shape[0])}
