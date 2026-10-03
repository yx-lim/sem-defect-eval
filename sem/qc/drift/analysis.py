"""Group comparison, flag rule, confound checks and controls (spec §5)."""

from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sem.qc.drift import stats
from sem.qc.drift.features import ALL_FEATURES, COVARIATE_FEATURES, MATERIAL_FEATURES

STATUS_DIFFERENT = "different from reference"
STATUS_INVESTIGATE = "investigate"
STATUS_WITHIN = "within observed reference variation"
FEATURE_SETS = {"material": MATERIAL_FEATURES, "all": ALL_FEATURES}


def kpi_shifts(
    ref: pd.DataFrame, inc: pd.DataFrame, n_perm: int, n_boot: int, seed_perm: int, seed_boot: int,
    statistic: str = "rank_sum",
) -> pd.DataFrame:
    """Per-feature robust z, Hedges' g, median diff + bootstrap CI, permutation p, Holm per family."""
    rows = []
    for family, names in (("material", MATERIAL_FEATURES), ("covariate", COVARIATE_FEATURES)):
        for name in names:
            r = ref[name].to_numpy(float)
            x = inc[name].to_numpy(float)
            z, kind = stats.robust_z(x, r)
            diff, lo, hi = stats.bootstrap_median_diff_ci(x, r, n_boot, seed_boot)
            rows.append({
                "family": family, "feature": name,
                "n_ref": int(np.isfinite(r).sum()), "n_inc": int(np.isfinite(x).sum()),
                "median_ref": float(np.nanmedian(r)) if np.isfinite(r).any() else math.nan,
                "median_inc": float(np.nanmedian(x)) if np.isfinite(x).any() else math.nan,
                "robust_z": z, "z_scale": kind,
                "hedges_g": stats.hedges_g(x, r),
                "median_diff": diff, "median_diff_ci_lo": lo, "median_diff_ci_hi": hi,
                "p_raw": stats.permutation_p(x, r, n_perm, seed_perm, statistic),
                "p_raw_median_stat": stats.permutation_p_median_diff(x, r, n_perm, seed_perm),
            })
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for family in ("material", "covariate"):
        m = df["family"] == family
        df.loc[m, "p_holm"] = stats.holm(df.loc[m, "p_raw"].to_numpy())
    return df


def multivariate(
    ref: pd.DataFrame, inc: pd.DataFrame, ref_tiles: pd.DataFrame | None, inc_tiles: pd.DataFrame | None,
    n_perm: int, seed: int, null_pct: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {"mahalanobis": {}, "mmd": {}, "tile_mmd": {}}
    for set_name, names in FEATURE_SETS.items():
        cols = list(names)
        maha = stats.mahalanobis_test(ref[cols].to_numpy(float), inc[cols].to_numpy(float), null_pct)
        maha["incoming_stems"] = inc["stem"].tolist()
        maha["reference_stems"] = ref["stem"].tolist()
        maha["features_used"] = [cols[i] for i in maha.pop("feature_indices_used")]
        out["mahalanobis"][set_name] = maha
        out["mmd"][set_name] = stats.mmd_permutation_test(
            ref[cols].to_numpy(float), inc[cols].to_numpy(float), n_perm, seed
        )
        if ref_tiles is not None and inc_tiles is not None:
            tiles = pd.concat([ref_tiles, inc_tiles], ignore_index=True)
            stems = list(ref["stem"]) + list(inc["stem"])
            index = {s: i for i, s in enumerate(stems)}
            tiles = tiles[tiles["stem"].isin(index)]
            out["tile_mmd"][set_name] = stats.tile_mmd_block_test(
                tiles[cols].to_numpy(float),
                tiles["stem"].map(index).to_numpy(int),
                np.array([False] * len(ref) + [True] * len(inc)),
                n_perm, seed,
            )
    return out


def criteria(kpi: pd.DataFrame, multi: dict[str, Any], cfg: dict[str, Any], feature_set: str = "material") -> dict[str, bool]:
    alpha = float(cfg["alpha"])
    majority = float(cfg["mahalanobis_majority_frac"])
    mat = kpi[kpi["family"] == "material"]
    return {
        "mmd_p_lt_alpha": bool(multi["mmd"][feature_set]["p"] < alpha),
        "mahalanobis_majority_over_null": bool(
            multi["mahalanobis"][feature_set]["frac_above_null"] > majority
        ),
        "material_kpi_holm_lt_alpha": bool((mat["p_holm"] < alpha).any()),
    }


def flag_status(crit: dict[str, bool]) -> str:
    """Spec §5 flag rule."""
    multi = crit["mmd_p_lt_alpha"] or crit["mahalanobis_majority_over_null"]
    if multi and crit["material_kpi_holm_lt_alpha"]:
        return STATUS_DIFFERENT
    if multi or crit["material_kpi_holm_lt_alpha"]:
        return STATUS_INVESTIGATE
    return STATUS_WITHIN


def compare(
    ref: pd.DataFrame, inc: pd.DataFrame, cfg: dict[str, Any], n_perm: int, n_boot: int,
    seed_perm: int = 0, seed_boot: int = 0,
    ref_tiles: pd.DataFrame | None = None, inc_tiles: pd.DataFrame | None = None,
) -> dict[str, Any]:
    kpi = kpi_shifts(ref, inc, n_perm, n_boot, seed_perm, seed_boot, cfg.get("permutation_statistic", "rank_sum"))
    multi = multivariate(ref, inc, ref_tiles, inc_tiles, n_perm, seed_perm, float(cfg["mahalanobis_null_pct"]))
    crit = criteria(kpi, multi, cfg, "material")
    crit_all = criteria(kpi, multi, cfg, "all")
    return {
        "kpi": kpi, "multi": multi,
        "criteria": crit, "status": flag_status(crit),
        "criteria_all_features": crit_all, "status_all_features": flag_status(crit_all),
    }


def method_fires(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, bool]:
    """Per-method decision used for control false-flag / detection rates."""
    alpha = float(cfg["alpha"])
    majority = float(cfg["mahalanobis_majority_frac"])
    kpi, multi = result["kpi"], result["multi"]
    out = {
        "kpi_material_any_holm": bool((kpi.loc[kpi.family == "material", "p_holm"] < alpha).any()),
        "kpi_covariate_any_holm": bool((kpi.loc[kpi.family == "covariate", "p_holm"] < alpha).any()),
    }
    for s in FEATURE_SETS:
        out[f"mahalanobis_{s}"] = bool(multi["mahalanobis"][s]["frac_above_null"] > majority)
        out[f"mmd_stem_{s}"] = bool(multi["mmd"][s]["p"] < alpha)
        if s in multi["tile_mmd"]:
            out[f"mmd_tile_{s}"] = bool(multi["tile_mmd"][s]["p"] < alpha)
    out["flag_different"] = result["status"] == STATUS_DIFFERENT
    out["flag_investigate_or_different"] = result["status"] != STATUS_WITHIN
    return out


def changed_material_kpis(kpi: pd.DataFrame, alpha: float) -> list[str]:
    m = (kpi["family"] == "material") & (kpi["p_holm"] < alpha)
    return kpi.loc[m, "feature"].tolist()


def confound_check(
    stems: pd.DataFrame, ref: pd.DataFrame, inc: pd.DataFrame, kpi: pd.DataFrame,
    cfg: dict[str, Any], n_perm: int, n_boot: int, seed_perm: int, seed_boot: int,
) -> dict[str, Any]:
    """Covariate Holm results, KPI-covariate Spearman across all stems, detector_set-matched tests."""
    alpha = float(cfg["alpha"])
    rho_thr = float(cfg["confound_spearman_abs_rho"])
    changed = changed_material_kpis(kpi, alpha)
    cov = kpi[kpi.family == "covariate"]
    shifted_covs = cov.loc[cov.p_holm < alpha, "feature"].tolist()
    targets = changed if changed else kpi[kpi.family == "material"].sort_values("p_raw")["feature"].head(3).tolist()
    corr = []
    for k in targets:
        for c in COVARIATE_FEATURES:
            a, b = stems[k].to_numpy(float), stems[c].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 4 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
                rho, p = math.nan, math.nan
            else:
                rho, p = spearmanr(a[ok], b[ok])
            corr.append({"kpi": k, "covariate": c, "rho": float(rho), "p": float(p),
                         "abs_rho_over_threshold": bool(abs(rho) > rho_thr) if np.isfinite(rho) else False})
    corr_df = pd.DataFrame(corr)
    matched = []
    for det in sorted(set(ref.detector_set) & set(inc.detector_set)):
        r, x = ref[ref.detector_set == det], inc[inc.detector_set == det]
        if len(r) < 2 or len(x) < 2:
            matched.append({"detector_set": det, "n_ref": len(r), "n_inc": len(x), "skipped": "fewer than 2 stems per group"})
            continue
        km = kpi_shifts(r, x, n_perm, n_boot, seed_perm, seed_boot, cfg.get("permutation_statistic", "rank_sum"))
        km = km[km.family == "material"]
        for _, row in km.iterrows():
            matched.append({"detector_set": det, "n_ref": len(r), "n_inc": len(x), "feature": row.feature,
                            "robust_z": row.robust_z, "median_diff": row.median_diff,
                            "p_raw": row.p_raw, "p_holm": row.p_holm})
    matched_df = pd.DataFrame(matched)
    det_ref = ref.detector_set.value_counts().to_dict()
    det_inc = inc.detector_set.value_counts().to_dict()

    reasons = []
    explained = []
    for k in changed:
        sub = corr_df[(corr_df.kpi == k) & corr_df.abs_rho_over_threshold]
        if len(sub):
            explained.append(k)
            reasons.append(f"{k}: |Spearman rho| > {rho_thr} with " + ", ".join(
                f"{r.covariate} (rho={r.rho:+.2f})" for r in sub.itertuples()))
        else:
            reasons.append(f"{k}: no covariate with |Spearman rho| > {rho_thr}")
    if shifted_covs:
        reasons.append("covariates shifted (Holm p<{:.2f}): {}".format(alpha, ", ".join(shifted_covs)))
    else:
        reasons.append("no acquisition covariate shifted at Holm p<{:.2f}".format(alpha))
    held = []
    if changed and len(matched_df) and "feature" in matched_df:
        for k in changed:
            sub = matched_df[matched_df.get("feature") == k]
            held.append(bool(len(sub)) and bool((sub.p_raw < alpha).any()))
            if len(sub):
                reasons.append(f"{k} detector_set-matched: " + "; ".join(
                    f"{r.detector_set} (n={r.n_ref}v{r.n_inc}) z={r.robust_z:+.2f}, p_raw={r.p_raw:.3g}" for r in sub.itertuples()))
    if det_ref != det_inc and set(det_ref) != set(det_inc):
        reasons.append(f"detector_set composition differs: reference {det_ref}, incoming {det_inc}")
    if not changed:
        verdict = "not applicable (no material KPI changed at Holm p<0.05)"
    elif len(explained) == len(changed) and shifted_covs:
        verdict = "yes"
    elif not explained and held and all(held):
        verdict = "no"
    else:
        verdict = "unclear"
    return {
        "changed_material_kpis": changed,
        "shifted_covariates": shifted_covs,
        "spearman": corr_df,
        "detector_matched": matched_df,
        "detector_set_ref": det_ref, "detector_set_inc": det_inc,
        "acquisition_artifact_could_explain": verdict,
        "reasons": reasons,
    }


def splits(n: int, k: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(n), k))


def negative_control(
    ref: pd.DataFrame, ref_tiles: pd.DataFrame, cfg: dict[str, Any], n_perm: int, n_boot: int, seed: int = 0
) -> dict[str, Any]:
    """Every pseudo-reference(4) vs pseudo-incoming(3) split of the reference stems."""
    n_ref_pseudo, n_inc_pseudo = (int(v) for v in cfg["negative_control_split"])
    ref = ref.reset_index(drop=True)
    rows = []
    for inc_idx in splits(len(ref), n_inc_pseudo):
        inc = ref.iloc[list(inc_idx)]
        pref = ref.drop(index=list(inc_idx))
        res = compare(pref, inc, cfg, n_perm, n_boot, seed, seed,
                      ref_tiles[ref_tiles.stem.isin(pref.stem)], ref_tiles[ref_tiles.stem.isin(inc.stem)])
        fires = method_fires(res, cfg)
        fires["pseudo_incoming"] = ";".join(inc.stem)
        fires["min_material_p_raw"] = float(res["kpi"].loc[res["kpi"].family == "material", "p_raw"].min())
        rows.append(fires)
    df = pd.DataFrame(rows)
    methods = [c for c in df.columns if c not in ("pseudo_incoming", "min_material_p_raw")]
    return {
        "n_splits": len(df), "split": f"{n_ref_pseudo} pseudo-reference vs {n_inc_pseudo} pseudo-incoming",
        "false_flag_rate": {m: float(df[m].mean()) for m in methods},
        "min_achievable_perm_p_note": (
            f"only {math.comb(len(ref), n_inc_pseudo)} distinct label assignments exist; "
            f"smallest attainable permutation p is about {1 / math.comb(len(ref), n_inc_pseudo):.3f}, "
            f"so Holm over {len(MATERIAL_FEATURES)} material KPIs cannot reach 0.05"
        ),
        "per_split": df,
    }


def positive_control(
    ref: pd.DataFrame, ref_tiles: pd.DataFrame, shifted: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    cfg: dict[str, Any], n_perm: int, n_boot: int, seed: int = 0,
) -> dict[str, Any]:
    """Detection per shift: (a) all 7 shifted copies vs 7 originals; (b) every 4-original vs 3-shifted split."""
    n_inc_pseudo = int(cfg["negative_control_split"][1])
    ref = ref.reset_index(drop=True)
    out: dict[str, Any] = {}
    for name, (s_stems, s_tiles) in shifted.items():
        s_stems = s_stems.set_index("stem").loc[ref.stem].reset_index()
        full = compare(ref, s_stems, cfg, n_perm, n_boot, seed, seed, ref_tiles, s_tiles)
        rows = []
        for inc_idx in splits(len(ref), n_inc_pseudo):
            inc_stems = ref.stem.iloc[list(inc_idx)]
            inc = s_stems[s_stems.stem.isin(inc_stems)]
            pref = ref[~ref.stem.isin(inc_stems)]
            res = compare(pref, inc, cfg, n_perm, n_boot, seed, seed,
                          ref_tiles[ref_tiles.stem.isin(pref.stem)], s_tiles[s_tiles.stem.isin(inc.stem)])
            rows.append(method_fires(res, cfg))
        df = pd.DataFrame(rows)
        kpi = full["kpi"]
        out[name] = {
            "full_7v7": {"fires": method_fires(full, cfg), "status": full["status"],
                          "kpi": kpi[["family", "feature", "robust_z", "median_diff", "p_raw", "p_holm"]]},
            "splits_4v3_detection_rate": {m: float(df[m].mean()) for m in df.columns},
            "n_splits": len(df),
        }
    return out
