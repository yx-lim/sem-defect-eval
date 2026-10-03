import numpy as np
import pandas as pd
import pytest

from sem.qc.config import load_config
from sem.qc.drift import analysis, features, report, shifts
from sem.qc.drift.features import ALL_FEATURES, MATERIAL_FEATURES
from sem.qc.schema import Instance, Prediction

CFG = load_config()["drift"]


def _synthetic_stem(seed, size=128, dark=0.05):
    rng = np.random.default_rng(seed)
    bse = rng.normal(120, 10, (size, size)).clip(0, 255).astype(np.uint8)
    inlens = rng.normal(100, 10, (size, size)).clip(0, 255).astype(np.uint8)
    sem = np.zeros((size, size), np.uint8)
    sem[rng.random((size, size)) < dark] = 3
    sem[10:20, 10:20] = 2
    sem[40:50, 40:50] = 2
    return bse, inlens, sem


def test_tile_grid_valid_fraction():
    valid = np.ones((100, 130), bool)
    valid[:, :20] = False
    grid = features.tile_grid(valid, 50, 0.9)
    assert [(x, y) for x, y, _ in grid] == [(50, 0), (50, 50)]
    assert all(f >= 0.9 for *_, f in grid)


def test_tile_features_and_stem_vector_median():
    bse, inlens, sem = _synthetic_stem(0)
    valid = np.ones(sem.shape, bool)
    agg = Instance("agglomerate", [5, 5, 30, 30], [[5, 5], [30, 5], [30, 30], [5, 30]])
    rows = features.tile_features("s1", "Batch_1", "ETD", bse, inlens, valid,
                                  Prediction(sem, [agg]), 25.0, 3, CFG, 64)
    assert len(rows) == 4
    df = pd.DataFrame(rows)
    assert set(ALL_FEATURES) <= set(df.columns)
    assert df.loc[(df.tile_x0 == 0) & (df.tile_y0 == 0), "agglomerate_per_mm2"].iloc[0] > 0
    assert (df.edge_column_px == 3).all()
    stem = features.stem_vectors(df)
    assert stem.n_tiles.iloc[0] == 4
    assert stem.void_fraction.iloc[0] == pytest.approx(df.void_fraction.median())


def test_covariates_respond_to_blur_and_brightness():
    rng = np.random.default_rng(0)
    img = rng.normal(120, 20, (128, 128)).clip(0, 255).astype(np.uint8)
    valid = np.ones(img.shape, bool)
    cfg = dict(CFG, curtain_subtile_px=64)
    base = features.acquisition_covariates(img, img, valid, 0, cfg)
    blurred = shifts.add_blur({"BSE": img}, 1.5)["BSE"]
    bright = shifts.add_brightness({"BSE": img}, 1.1)["BSE"]
    assert features.acquisition_covariates(blurred, img, valid, 0, cfg)["bse_focus"] < base["bse_focus"]
    assert features.acquisition_covariates(blurred, img, valid, 0, cfg)["bse_noise"] < base["bse_noise"]
    assert features.acquisition_covariates(bright, img, valid, 0, cfg)["bse_mean"] > base["bse_mean"]


def test_curtaining_score_higher_for_vertical_stripes():
    rng = np.random.default_rng(0)
    img = rng.normal(120, 5, (128, 128))
    striped = img + 30 * np.sin(np.arange(128) / 2.0)[None, :]
    assert features.fft_curtain_score(striped, 64) > features.fft_curtain_score(img, 64)


def test_void_shift_hits_target_fraction_and_is_deterministic():
    valid = np.ones((400, 600), bool)
    valid[:, :10] = False
    m1 = shifts.void_mask(valid.shape, valid, 0.02, np.random.default_rng(0), (5, 20))
    m2 = shifts.void_mask(valid.shape, valid, 0.02, np.random.default_rng(0), (5, 20))
    assert np.array_equal(m1, m2)
    assert 0.02 <= m1.sum() / valid.sum() < 0.03
    assert not (m1 & ~valid).any()
    rng = np.random.default_rng(0)
    views = {"BSE": rng.normal(120, 10, valid.shape).astype(np.uint8),
             "Inlens": rng.normal(90, 10, valid.shape).astype(np.uint8)}
    out = shifts.apply_shift("cracks", views, valid, CFG["positive_control"] | {"cracks_per_mm2": 1e5}, 0)
    changed_bse = out["BSE"] != views["BSE"]
    changed_inl = out["Inlens"] != views["Inlens"]
    assert changed_bse.sum() > 100
    assert (changed_bse & changed_inl).sum() > 0.8 * changed_bse.sum()
    assert out["BSE"][changed_bse].mean() < views["BSE"][changed_bse].mean() - 10


def _stem_table(n, batch, shift=0.0, seed=0, det="ETD"):
    rng = np.random.default_rng(seed)
    data = {name: rng.normal(1.0, 0.1, n) for name in ALL_FEATURES}
    for name in MATERIAL_FEATURES[:2]:
        data[name] = data[name] + shift
    df = pd.DataFrame(data)
    df.insert(0, "stem", [f"{batch}_{i}" for i in range(n)])
    df.insert(1, "batch", batch)
    df.insert(2, "detector_set", det)
    df.insert(3, "n_tiles", 4)
    return df


def _tiles_from(stems, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in stems.itertuples():
        for t in range(4):
            row = {"stem": r.stem, "batch": r.batch, "detector_set": r.detector_set,
                   "tile_x0": t * 1024, "tile_y0": 0, "tile_px": 1024}
            for name in ALL_FEATURES:
                row[name] = getattr(r, name) + rng.normal(0, 0.02)
            rows.append(row)
    return pd.DataFrame(rows)


def test_flag_rule_and_wording():
    ref = _stem_table(7, "Batch_1", seed=1)
    far = _stem_table(7, "Batch_2", shift=2.0, seed=2)
    same = _stem_table(7, "Batch_3", seed=3)
    res = analysis.compare(ref, far, CFG, 999, 200, 0, 0, _tiles_from(ref), _tiles_from(far))
    assert res["status"] == analysis.STATUS_DIFFERENT
    assert set(analysis.changed_material_kpis(res["kpi"], 0.05)) >= {"void_fraction"}
    res2 = analysis.compare(ref, same, CFG, 999, 200, 0, 0)
    assert res2["status"] in (analysis.STATUS_WITHIN, analysis.STATUS_INVESTIGATE)
    assert analysis.flag_status({"mmd_p_lt_alpha": True, "mahalanobis_majority_over_null": False,
                                 "material_kpi_holm_lt_alpha": False}) == analysis.STATUS_INVESTIGATE
    assert analysis.flag_status({"mmd_p_lt_alpha": False, "mahalanobis_majority_over_null": False,
                                 "material_kpi_holm_lt_alpha": False}) == analysis.STATUS_WITHIN
    for status in (analysis.STATUS_DIFFERENT, analysis.STATUS_INVESTIGATE, analysis.STATUS_WITHIN):
        assert report.check_wording(status) == []
    assert report.check_wording("this batch is defective") == ["defective", "defect"]


def test_compare_is_deterministic():
    ref = _stem_table(7, "Batch_1", seed=1)
    inc = _stem_table(5, "Batch_2", shift=0.3, seed=2)
    a = analysis.compare(ref, inc, CFG, 499, 100, 0, 0)
    b = analysis.compare(ref, inc, CFG, 499, 100, 0, 0)
    pd.testing.assert_frame_equal(a["kpi"], b["kpi"])
    assert a["multi"]["mmd"] == b["multi"]["mmd"]


def test_confound_check_marks_covariate_correlation():
    ref = _stem_table(7, "Batch_1", seed=1)
    inc = _stem_table(7, "Batch_2", seed=2, det="SE")
    inc["bse_focus"] = inc["bse_focus"] - 1.0
    inc["void_fraction"] = inc["void_fraction"] + 1.0
    stems = pd.concat([ref, inc], ignore_index=True)
    stems["void_fraction"] = -stems["bse_focus"] + np.linspace(0, 0.01, len(stems))
    ref, inc = stems.iloc[:7], stems.iloc[7:]
    res = analysis.compare(ref, inc, CFG, 999, 100)
    conf = analysis.confound_check(stems, ref, inc, res["kpi"], CFG, 999, 100, 0, 0)
    assert "void_fraction" in conf["changed_material_kpis"]
    assert conf["acquisition_artifact_could_explain"] == "yes"
    assert any("detector_set composition differs" in r for r in conf["reasons"])


def test_negative_control_split_count_and_rates():
    ref = _stem_table(7, "Batch_1", seed=1)
    neg = analysis.negative_control(ref, _tiles_from(ref), CFG, 199, 50)
    assert neg["n_splits"] == 35
    assert all(0 <= v <= 1 for v in neg["false_flag_rate"].values())
    assert neg["false_flag_rate"]["flag_different"] == 0.0
