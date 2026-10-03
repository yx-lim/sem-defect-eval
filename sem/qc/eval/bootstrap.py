"""Stem-level bootstrap (spec §4): resample STEMS with replacement, never tiles."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np


def stem_multiplicities(n_stems: int, n_reps: int, seed: int) -> np.ndarray:
    """(n_reps, n_stems) counts of how often each stem is drawn per replicate."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_stems, size=(n_reps, n_stems))
    out = np.zeros((n_reps, n_stems), dtype=np.int64)
    np.add.at(out, (np.repeat(np.arange(n_reps), n_stems), draws.ravel()), 1)
    return out


def bootstrap_ci(
    n_stems: int,
    statistic: Callable[[np.ndarray], Mapping[str, float]],
    n_reps: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Percentile CI for every key returned by ``statistic(multiplicity_vector)``.

    NaN replicates (e.g. a resample with no GT of a class) are dropped; the share of
    NaN replicates is reported as ``nan_frac`` so it is never hidden.
    """
    if n_stems == 0:
        return {}
    reps = stem_multiplicities(n_stems, n_reps, seed)
    values: dict[str, list[float]] = {}
    for mult in reps:
        for key, value in statistic(mult).items():
            values.setdefault(key, []).append(float(value))
    out = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        else:
            low = high = float("nan")
        out[key] = {
            "ci_low": float(low),
            "ci_high": float(high),
            "nan_frac": float(1 - finite.size / arr.size),
        }
    return out
