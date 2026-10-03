"""End-to-end batch change detection run (spec §5)."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sem.qc import io as qcio
from sem.qc.config import load_config
from sem.qc.drift import analysis, features, report, shifts
from sem.qc.drift.features import ALL_FEATURES, COVARIATE_FEATURES, MATERIAL_FEATURES
from sem.qc.schema import IGNORE_LABEL

PALETTE = np.array([
    [120, 120, 120], [40, 90, 200], [250, 220, 40], [230, 30, 30],
    [240, 120, 220], [0, 220, 220], [40, 200, 40], [255, 140, 0],
], dtype=np.uint8)


def get_classical_method():
    """Locate the foundation's classical_v1 Method implementation."""
    from sem.qc.classical import ClassicalV1

    return ClassicalV1(load_config()["classical_v1"])


def _stem_tiles(args) -> list[dict[str, Any]]:
    rec, pred_dir, cfg, tile_px = args
    views = qcio.load_stem(rec, ("BSE", "Inlens"))
    valid = qcio.valid_mask(rec)
    pred = features.load_prediction(pred_dir, rec.stem)
    edge = features.read_edge_columns(rec.views["BSE"])
    return features.tile_features(rec.stem, rec.batch, rec.detector_set, views["BSE"], views["Inlens"],
                                  valid, pred, rec.pixel_size_nm, edge, cfg, tile_px)


def _shifted_stem_tiles(args) -> list[dict[str, Any]]:
    rec, shift, cfg, pc_cfg, tile_px, seed = args
    method = get_classical_method()
    views = qcio.load_stem(rec, ("BSE", "Inlens"))
    valid = qcio.valid_mask(rec)
    edge = features.read_edge_columns(rec.views["BSE"])
    shifted = shifts.apply_shift(shift, views, valid, pc_cfg, seed, rec.pixel_size_nm)
    pred = method.predict(shifted, valid)
    rows = features.tile_features(rec.stem, rec.batch, rec.detector_set, shifted["BSE"], shifted["Inlens"],
                                  valid, pred, rec.pixel_size_nm, edge, cfg, tile_px)
    for r in rows:
        r["shift"] = shift
    return rows


def _pmap(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs))


def compute_tile_table(records, pred_dir: Path, cfg, tile_px: int, workers: int) -> pd.DataFrame:
    missing = [r.stem for r in records if not (pred_dir / f"{r.stem}_semantic.png").exists()]
    if missing:
        raise FileNotFoundError(f"classical_v1 predictions missing in {pred_dir} for stems: {missing}")
    rows = _pmap(_stem_tiles, [(r, pred_dir, cfg, tile_px) for r in records], workers)
    return pd.DataFrame([row for stem_rows in rows for row in stem_rows])


def compute_shifted_tables(records, cfg, tile_px, workers, cache_dir: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    pc = cfg["positive_control"]
    out = {}
    for shift in pc["shifts"]:
        cache = cache_dir / f"tile_features_{shift}.csv"
        if cache.exists():
            tiles = pd.read_csv(cache)
        else:
            jobs = [(r, shift, cfg, pc, tile_px, int(pc["seed"]) + i) for i, r in enumerate(records)]
            rows = _pmap(_shifted_stem_tiles, jobs, workers)
            tiles = pd.DataFrame([row for stem_rows in rows for row in stem_rows])
            cache.parent.mkdir(parents=True, exist_ok=True)
            tiles.to_csv(cache, index=False)
        out[shift] = (features.stem_vectors(tiles), tiles)
    return out


def _tile_loader(records_by_stem, pred_dir):
    def load(stem, x0, y0, size):
        rec = records_by_stem[stem]
        bse = qcio.load_stem(rec, ("BSE",))["BSE"][y0 : y0 + size, x0 : x0 + size]
        sem = features.load_prediction(pred_dir, stem).semantic[y0 : y0 + size, x0 : x0 + size]
        overlay = np.stack([bse] * 3, -1).astype(np.float32)
        known = sem != IGNORE_LABEL
        color = PALETTE[np.where(known, sem, 0).clip(0, 7)].astype(np.float32)
        overlay[known] = 0.45 * overlay[known] + 0.55 * color[known]
        return bse, overlay.astype(np.uint8)
    return load


def _json_default(v):
    if isinstance(v, pd.DataFrame):
        return v.to_dict(orient="records")
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v))


def run(
    reference: str,
    incoming: list[str],
    run_name: str | None = None,
    data_root: str | None = None,
    work_root: str | None = None,
    method: str = "classical_v1",
    workers: int | None = None,
    n_perm: int | None = None,
    n_boot: int | None = None,
    skip_positive: bool = False,
) -> Path:
    full_cfg = load_config()
    cfg = full_cfg["drift"]
    data_root = data_root or full_cfg["paths"]["data_root"]
    work_root = Path(work_root or full_cfg["paths"]["work_root"])
    n_perm = n_perm or int(full_cfg["evaluation"]["permutation_replicates"])
    n_boot = n_boot or int(full_cfg["evaluation"]["bootstrap_replicates"])
    seed_perm = int(full_cfg["seeds"]["permutation"])
    seed_boot = int(full_cfg["seeds"]["bootstrap"])
    tile_px = int(full_cfg["tile_sizes"]["drift_px"])
    workers = workers or min(4, os.cpu_count() or 1)
    run_name = run_name or f"ref-{reference}__inc-{'-'.join(incoming)}"
    out = work_root / "drift" / run_name
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    pred_dir = work_root / "preds" / method

    records = [r for r in qcio.list_stems(data_root) if r.batch in {reference, *incoming}]
    by_stem = {r.stem: r for r in records}
    tiles = compute_tile_table(records, pred_dir, cfg, tile_px, workers)
    tiles.to_csv(out / "tile_features.csv", index=False)
    stems = features.stem_vectors(tiles)
    stems.to_csv(out / "stem_features.csv", index=False)

    ref = stems[stems.batch == reference].reset_index(drop=True)
    ref_tiles = tiles[tiles.batch == reference]
    results: dict[str, Any] = {}
    kpi_rows, maha_rows, mmd_json = [], [], {}
    for batch in incoming:
        inc = stems[stems.batch == batch].reset_index(drop=True)
        inc_tiles = tiles[tiles.batch == batch]
        res = analysis.compare(ref, inc, cfg, n_perm, n_boot, seed_perm, seed_boot, ref_tiles, inc_tiles)
        res["confound"] = analysis.confound_check(stems, ref, inc, res["kpi"], cfg, n_perm, n_boot, seed_perm, seed_boot)
        results[batch] = res
        kpi_rows.append(res["kpi"].assign(incoming_batch=batch))
        for fs, m in res["multi"]["mahalanobis"].items():
            for stem, d2, above in zip(m["incoming_stems"], m["d2_incoming"], m["above_null"]):
                maha_rows.append({"incoming_batch": batch, "feature_set": fs, "role": "incoming", "stem": stem,
                                  "d2": d2, "null_threshold": m["null_threshold"], "above_null": above})
            for stem, d2 in zip(m["reference_stems"], m["d2_null_loo"]):
                maha_rows.append({"incoming_batch": batch, "feature_set": fs, "role": "reference_loo", "stem": stem,
                                  "d2": d2, "null_threshold": m["null_threshold"], "above_null": d2 > m["null_threshold"]})
        mmd_json[batch] = {"stem": res["multi"]["mmd"], "tile": res["multi"]["tile_mmd"]}
    pd.concat(kpi_rows).to_csv(out / "kpi_shifts.csv", index=False)
    pd.DataFrame(maha_rows).to_csv(out / "mahalanobis.csv", index=False)
    (out / "mmd.json").write_text(json.dumps(mmd_json, indent=2, default=_json_default))

    neg = analysis.negative_control(ref, ref_tiles, cfg, n_perm, n_boot, seed_perm)
    pos = None
    if not skip_positive:
        shifted = compute_shifted_tables([by_stem[s] for s in ref.stem], cfg, tile_px, workers,
                                         work_root / "drift" / "_cache" / f"positive_{method}_{reference}")
        pos = analysis.positive_control(ref, ref_tiles, shifted, cfg, n_perm, n_boot, seed_perm)
    (out / "controls.json").write_text(json.dumps({"negative": neg, "positive": pos}, indent=2, default=_json_default))

    viz = {"pca": report.embedding_plot(stems, out / "pca.png", "pca")}
    try:
        viz["umap"] = report.embedding_plot(stems, out / "umap.png", "umap")
    except Exception as exc:  # optional visualization
        viz["umap"] = {"error": f"{type(exc).__name__}: {exc}"}

    loader = _tile_loader(by_stem, pred_dir)
    evidence: dict[str, list] = {}
    for batch, res in results.items():
        kpi = res["kpi"]
        changed = analysis.changed_material_kpis(kpi, float(cfg["alpha"]))
        note = ""
        if not changed:
            mat = kpi[(kpi.family == "material") & np.isfinite(kpi.robust_z)]
            changed = mat.reindex(mat.robust_z.abs().sort_values(ascending=False).index).feature.head(2).tolist()
            note = " (largest |robust z|; NOT a significant change)"
        evidence[batch] = []
        for feat in changed:
            ref_med = float(np.nanmedian(ref[feat]))
            p = out / "evidence" / f"{batch}_{feat}.png"
            meta = report.evidence_panel(feat, tiles[tiles.batch == batch], ref_tiles, ref_med, loader, p,
                                         int(cfg["evidence_tiles"]), note)
            evidence[batch].append({"feature": feat, "path": f"evidence/{p.name}", "tiles": meta, "note": note})

    text = write_report(out, reference, incoming, stems, results, neg, pos, viz, evidence, cfg, n_perm, n_boot, method)
    bad = report.check_wording(text)
    if bad:
        raise AssertionError(f"report.md contains forbidden wording: {bad}")
    return out


def _sig(p, alpha=0.05):
    return "**" if isinstance(p, float) and math.isfinite(p) and p < alpha else ""


def write_report(out, reference, incoming, stems, results, neg, pos, viz, evidence, cfg, n_perm, n_boot, method) -> str:
    alpha = float(cfg["alpha"])
    f = report._fmt
    L = [f"# Batch change detection: reference {reference} vs incoming {', '.join(incoming)}", ""]
    L.append(f"Unit = stem. Material features from `{method}` predictions via `sem/qc/kpi.py`; "
             f"tiles {int(stems.n_tiles.sum())} total (1024 px, >=90% valid), stem vector = median over tiles. "
             f"Permutations {n_perm}, bootstrap {n_boot}. Model outputs are not ground truth; all material values are "
             f"`{method}` estimates.")
    L.append("")
    L.append("| batch | stems | detector_set counts | tiles |\n|---|---|---|---|")
    for b, g in stems.groupby("batch"):
        L.append(f"| {b} | {len(g)} | {g.detector_set.value_counts().to_dict()} | {int(g.n_tiles.sum())} |")
    L.append("")
    L.append("## Summary\n")
    L.append("| incoming | flag (material features, spec rule) | flag (all features) | changed material KPIs (Holm p<0.05) | acquisition artifact could explain |\n|---|---|---|---|---|")
    for b, r in results.items():
        L.append(f"| {b} | {r['status']} | {r['status_all_features']} | {', '.join(analysis.changed_material_kpis(r['kpi'], alpha)) or 'none'} | {r['confound']['acquisition_artifact_could_explain']} |")
    L.append("")
    L.append("Flag rule (spec §5): \"different from reference\" if (stem MMD p<0.05 or >50% incoming stems over the "
             "leave-one-reference-stem-out Mahalanobis 95th pct) AND >=1 material KPI Holm p<0.05; \"investigate\" if any one; "
             "else \"within observed reference variation\". The MMD/Mahalanobis criteria use the material feature set; "
             "the all-features variant is reported alongside.")
    L.append("")
    for b, r in results.items():
        L.append(f"## {b} vs {reference}\n")
        L.append(f"**Flag: {r['status']}** (criteria: " + ", ".join(f"{k}={v}" for k, v in r["criteria"].items()) + ")")
        L.append(f"All-features variant: {r['status_all_features']} (" + ", ".join(f"{k}={v}" for k, v in r["criteria_all_features"].items()) + ")\n")
        L.append("### Per-KPI shifts\n")
        L.append(report.kpi_table(r["kpi"]))
        L.append("robust z = (median_inc - median_ref)/(1.4826 MAD_ref); `(std)` = MAD was 0, std used. "
                 "CI = stem bootstrap percentile. p = two-sided stem-label permutation on |median difference|; "
                 "Holm within material and within covariate family.\n")
        L.append("### Multivariate\n")
        L.append("| feature set | Mahalanobis: frac incoming over null 95th | null threshold D² | incoming D² (median) | stem MMD² | stem MMD p | tile MMD² | tile MMD p (stem-block perm) |\n|---|---|---|---|---|---|---|---|")
        for fs in ("material", "all"):
            m = r["multi"]["mahalanobis"][fs]
            mm = r["multi"]["mmd"][fs]
            tm = r["multi"]["tile_mmd"].get(fs, {})
            L.append(f"| {fs} | {f(m['frac_above_null'])} ({sum(m['above_null'])}/{len(m['above_null'])}) | {f(m['null_threshold'])} | "
                     f"{f(float(np.median(m['d2_incoming'])))} | {f(mm['mmd2'])} | {_sig(mm['p'])}{f(mm['p'])}{_sig(mm['p'])} | "
                     f"{f(tm.get('mmd2'))} | {_sig(tm.get('p'))}{f(tm.get('p'))}{_sig(tm.get('p'))} |")
        L.append("\nMahalanobis: reference median/MAD standardization, Ledoit-Wolf covariance on the reference stems, "
                 f"null = {len(r['multi']['mahalanobis']['material']['d2_null_loo'])} leave-one-reference-stem-out D² values "
                 "(the 95th percentile of so few values is close to their maximum). MMD: unbiased MMD², RBF, median-heuristic "
                 "bandwidth on pooled median/MAD-standardized vectors (label-free so permutations are exchangeable).\n")
        c = r["confound"]
        L.append("### Confound assessment\n")
        L.append(f"**Acquisition artifact could explain: {c['acquisition_artifact_could_explain']}**\n")
        for reason in c["reasons"]:
            L.append(f"- {reason}")
        L.append(f"\nSpearman rho across all {len(stems)} stems (KPIs: {', '.join(sorted(set(c['spearman'].kpi))) if len(c['spearman']) else 'none'}; "
                 f"|rho|>{cfg['confound_spearman_abs_rho']} marked):\n")
        if len(c["spearman"]):
            piv = c["spearman"].pivot(index="kpi", columns="covariate", values="rho")
            L.append("| kpi | " + " | ".join(piv.columns) + " |\n|---|" + "---|" * len(piv.columns))
            for k, row in piv.iterrows():
                L.append(f"| {k} | " + " | ".join(
                    (f"**{v:+.2f}**" if abs(v) > float(cfg['confound_spearman_abs_rho']) else f"{v:+.2f}") if np.isfinite(v) else "n/a"
                    for v in row) + " |")
        dm = c["detector_matched"]
        L.append("\nMaterial tests repeated on stems matched by detector_set:\n")
        if len(dm) and "feature" in dm:
            dm2 = dm.dropna(subset=["feature"])
            L.append("| detector_set | n ref v inc | feature | robust z | median diff | p raw | p Holm |\n|---|---|---|---|---|---|---|")
            for x in dm2.itertuples():
                L.append(f"| {x.detector_set} | {x.n_ref}v{x.n_inc} | {x.feature} | {f(x.robust_z)} | {f(x.median_diff)} | {f(x.p_raw)} | {f(x.p_holm)} |")
        if len(dm) and "skipped" in dm:
            for x in dm.dropna(subset=["skipped"]).itertuples():
                L.append(f"- {x.detector_set}: skipped ({x.skipped}; n={x.n_ref}v{x.n_inc})")
        if not len(dm):
            L.append("- no detector_set present in both groups")
        L.append("\n### Evidence\n")
        for e in evidence.get(b, []):
            L.append(f"![{e['feature']}]({e['path']})\n")
            L.append(f"{e['feature']}{e['note']}: " + "; ".join(
                f"{t['role']} {t['stem']} ({t['tile_x0']},{t['tile_y0']}) = {f(t['value'])}" for t in e["tiles"]) + "\n")
    L.append("## Controls\n")
    L.append(f"### Negative control ({neg['n_splits']} splits, {neg['split']} of {reference})\n")
    L.append("| method | false-flag rate |\n|---|---|")
    for k, v in neg["false_flag_rate"].items():
        L.append(f"| {k} | {f(v)} |")
    L.append(f"\nNote: {neg['min_achievable_perm_p_note']}.\n")
    if pos:
        L.append("### Positive control (synthetic shifts in copies of reference stems, re-run through classical_v1)\n")
        methods = list(next(iter(pos.values()))["splits_4v3_detection_rate"].keys())
        L.append("Detection rate over 35 splits (4 original reference stems vs shifted copies of the other 3):\n")
        L.append("| method | " + " | ".join(pos) + " |\n|---|" + "---|" * len(pos))
        for m in methods:
            L.append(f"| {m} | " + " | ".join(f(pos[s]["splits_4v3_detection_rate"][m]) for s in pos) + " |")
        L.append("\nFull design (all 7 shifted copies vs the 7 originals; paired copies, so optimistic):\n")
        L.append("| method | " + " | ".join(pos) + " |\n|---|" + "---|" * len(pos))
        for m in methods:
            L.append(f"| {m} | " + " | ".join(str(pos[s]["full_7v7"]["fires"].get(m)) for s in pos) + " |")
        L.append("| status | " + " | ".join(pos[s]["full_7v7"]["status"] for s in pos) + " |")
        L.append("\nMaterial KPI robust z in the full design (shifted vs original):\n")
        L.append("| feature | " + " | ".join(pos) + " |\n|---|" + "---|" * len(pos))
        for feat in list(MATERIAL_FEATURES) + list(COVARIATE_FEATURES):
            vals = []
            for s in pos:
                k = pos[s]["full_7v7"]["kpi"]
                row = k[k.feature == feat].iloc[0]
                vals.append(f"{f(row.robust_z)}{' *' if row.p_holm < alpha else ''}")
            L.append(f"| {feat} | " + " | ".join(vals) + " |")
        L.append("\n`*` = Holm p<0.05 within family.\n")
    else:
        L.append("Positive control skipped (--skip-positive).\n")
    L.append("## Visualization (not a decision input)\n")
    L.append("![pca](pca.png)\n")
    pca = viz["pca"]
    L.append(f"PCA explained variance: {', '.join(f'{v:.0%}' for v in pca['explained_variance_ratio'])}; "
             f"largest |PC1| loadings: {', '.join(pca.get('top_loadings_pc1', []))}.")
    if "error" not in viz.get("umap", {"error": ""}):
        L.append("\n![umap](umap.png)\n")
    L.append("\n## Limitations\n")
    L.append(f"- {len(results and next(iter(results.values()))['multi']['mahalanobis']['material']['reference_stems'])} reference stems: "
             "permutation p-values have a hard floor and the Mahalanobis null has only as many values as reference stems; low power.")
    L.append(f"- Material values are `{method}` model outputs, not ground truth; any change may be a segmentation response to acquisition differences.")
    L.append("- All stems come from one physical sample; differences between batches are expected to be acquisition-driven.")
    text = "\n".join(L) + "\n"
    (out / "report.md").write_text(text)
    return text
