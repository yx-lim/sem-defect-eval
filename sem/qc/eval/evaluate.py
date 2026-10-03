"""Evaluation of one method against human GT (spec §4)."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sem.qc.eval.bootstrap import bootstrap_ci
from sem.qc.eval.gt import (
    CandidateDecision,
    GroundTruthError,
    TileGT,
    load_ground_truth,
    load_manifest,
)
from sem.qc.eval.metrics import (
    N_CLASSES,
    OBJECT_CLASSES,
    PORE,
    SUBSURFACE,
    components,
    confusion_matrix,
    iou_matrix,
    match_objects,
    pixel_counts,
    prf_iou,
    quantity_errors,
    safe_div,
    weighted_proportion,
    wilson_interval,
)
from sem.qc.eval.preds import PredictionStore
from sem.qc.kpi import _polygon_mask, compute_kpis
from sem.qc.schema import CLASS_NAMES, IGNORE_LABEL, Instance

KPI_NAMES = (
    "void_fraction",
    "void_fraction_incl_uncertain",
    "bright_particle_ecd_count",
    "bright_particle_ecd_median_um",
    "bright_particle_ecd_p90_um",
    "crack_density_um_per_mm2",
    "gap_density_um_per_mm2",
    "agglomerate_per_mm2",
    "agglomerate_area_frac",
)
AGGLOMERATE_KPIS = ("agglomerate_per_mm2", "agglomerate_area_frac")
OBJECT_KINDS = tuple(CLASS_NAMES[c] for c in OBJECT_CLASSES) + ("agglomerate",)
ERROR_STATS = ("bias", "mae", "rel_error", "spearman")
STRATA = ("random", "uncertainty", "all")


@dataclass
class EvalSettings:
    method: str
    split: str
    stratum: str
    work_root: Path
    data_root: Path
    manifest_path: Path
    frozen_manifest_sha256: str | None
    n_boot: int
    seed: int
    match_iou: dict[str, float]
    object_min_area_px: int = 1
    pred_object_max_ignored_frac: float = 0.5
    min_gt_objects: int = 5
    gallery_top_k: int = 10

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides: Any) -> "EvalSettings":
        ev = config["qc_eval"]
        values = dict(
            split=ev["headline_split"],
            stratum=ev["headline_stratum"],
            work_root=Path(config["paths"]["work_root"]),
            data_root=Path(config["paths"]["data_root"]),
            manifest_path=Path(__file__).resolve().parents[3] / "data/splits/manifest.csv",
            frozen_manifest_sha256=config["split"].get("frozen_manifest_sha256"),
            n_boot=int(config["evaluation"]["bootstrap_replicates"]),
            seed=int(config["seeds"]["bootstrap"]),
            match_iou={k: float(v) for k, v in ev["match_iou"].items()},
            object_min_area_px=int(ev["object_min_area_px"]),
            pred_object_max_ignored_frac=float(ev["pred_object_max_ignored_frac"]),
            min_gt_objects=int(ev["min_gt_objects"]),
            gallery_top_k=int(ev["gallery_top_k"]),
        )
        values.update({k: v for k, v in overrides.items() if v is not None})
        for key in ("work_root", "data_root", "manifest_path"):
            values[key] = Path(values[key])
        return cls(**values)


@dataclass
class ErrorRegion:
    """An unmatched object (FP = predicted only, FN = GT only) for galleries."""

    kind: str  # "FP" | "FN"
    class_name: str
    stem: str
    item_id: str
    tile_xy: tuple[int, int]
    bbox: tuple[int, int, int, int]  # tile-local y0, y1, x0, x1
    area_px: int
    best_iou: float
    mask: np.ndarray = field(repr=False)  # bbox-cropped bool


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _local_instances(polygons: list[list[list[float]]], x0: int, y0: int) -> list[Instance]:
    out = []
    for poly in polygons:
        pts = [[p[0] - x0, p[1] - y0] for p in poly]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append(Instance("agglomerate", [min(xs), min(ys), max(xs), max(ys)], pts))
    return out


def _filter_pred_masks(
    masks: list[np.ndarray], excluded: np.ndarray, max_frac: float
) -> list[np.ndarray]:
    """Drop predicted objects lying mostly on unscored GT pixels."""
    keep = []
    for mask in masks:
        area = mask.sum()
        if area and np.count_nonzero(mask & excluded) / area < max_frac:
            keep.append(mask)
    return keep


def _record_errors(
    errors: list[ErrorRegion], kind: str, class_name: str, tile: TileGT,
    masks: list[np.ndarray], indices: list[int], iou: np.ndarray, axis: int,
) -> None:
    for idx in indices:
        mask = masks[idx]
        y0, y1, x0, x1 = _bbox(mask)
        row = iou[idx, :] if axis == 0 else iou[:, idx]
        errors.append(
            ErrorRegion(
                kind, class_name, tile.stem, tile.item_id, (tile.x0, tile.y0),
                (y0, y1, x0, x1), int(mask.sum()),
                float(row.max()) if row.size else 0.0, mask[y0:y1, x0:x1].copy(),
            )
        )


def evaluate_objects(
    gt: np.ndarray, pred: np.ndarray, tile: TileGT, pred_agglomerates: list[Instance],
    settings: EvalSettings, errors: list[ErrorRegion] | None = None,
) -> dict[str, tuple[int, int, int] | None]:
    """Per object kind (tp, fp, fn); None for agglomerates when not annotated."""
    ignored = gt == IGNORE_LABEL
    out: dict[str, tuple[int, int, int] | None] = {}
    for class_id in OBJECT_CLASSES:
        name = CLASS_NAMES[class_id]
        excluded = ignored | (gt == SUBSURFACE) if class_id == PORE else ignored
        gt_masks = components(gt == class_id, settings.object_min_area_px)
        pred_masks = _filter_pred_masks(
            components(pred == class_id, settings.object_min_area_px),
            excluded, settings.pred_object_max_ignored_frac,
        )
        out[name] = _match(gt_masks, pred_masks, name, tile, settings, errors)
    if tile.agglomerates is None:
        out["agglomerate"] = None
    else:
        shape = gt.shape
        gt_masks = [m for m in (
            _polygon_mask(i.polygon, shape)
            for i in _local_instances(tile.agglomerates, tile.x0, tile.y0)
        ) if m.any()]
        pred_masks = _filter_pred_masks(
            [m for m in (_polygon_mask(i.polygon, shape) for i in pred_agglomerates) if m.any()],
            ignored, settings.pred_object_max_ignored_frac,
        )
        out["agglomerate"] = _match(gt_masks, pred_masks, "agglomerate", tile, settings, errors)
    return out


def _match(gt_masks, pred_masks, name, tile, settings, errors):
    iou = iou_matrix(gt_masks, pred_masks)
    result = match_objects(iou, settings.match_iou[name])
    if errors is not None:
        _record_errors(errors, "FN", name, tile, gt_masks, result.unmatched_gt, iou, 0)
        _record_errors(errors, "FP", name, tile, pred_masks, result.unmatched_pred, iou, 1)
    return result.tp, result.fp, result.fn


def _kpis_or_nan(semantic, instances, valid, px_nm, agglomerates_annotated):
    try:
        values = compute_kpis(semantic, instances, valid, px_nm)
    except ValueError:
        return {k: math.nan for k in KPI_NAMES}, False
    out = {k: (math.nan if values[k] is None else float(values[k])) for k in KPI_NAMES}
    if not agglomerates_annotated:
        for k in AGGLOMERATE_KPIS:
            out[k] = math.nan
    return out, True


def _mosaic(blocks: list[tuple[np.ndarray, list[Instance]]]):
    """Stack tiles vertically with 1-px 255 separators so kpi.py pools them exactly."""
    width = max(b[0].shape[1] for b in blocks)
    height = sum(b[0].shape[0] for b in blocks) + len(blocks) - 1
    mosaic = np.full((height, width), IGNORE_LABEL, dtype=np.uint8)
    instances, y = [], 0
    for label, insts in blocks:
        h, w = label.shape
        mosaic[y : y + h, :w] = label
        for inst in insts:
            pts = [[p[0], p[1] + y] for p in inst.polygon]
            instances.append(Instance(inst.class_name, inst.bbox, pts))
        y += h + 1
    return mosaic, instances


def _pixel_stats(counts: np.ndarray) -> dict[str, float]:
    out = {}
    for c in range(N_CLASSES):
        for k, v in prf_iou(*counts[c]).items():
            out[f"{CLASS_NAMES[c]}.{k}"] = v
    return out


def _object_stats(counts: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for name, arr in counts.items():
        tp, fp, fn = arr
        for k, v in prf_iou(tp, fp, fn).items():
            if k != "iou":
                out[f"{name}.{k}"] = v
    return out


def select_tiles(tiles: list[TileGT], split: str, stratum: str) -> list[TileGT]:
    return [
        t for t in tiles
        if t.split == split and (stratum == "all" or t.sampling_method == stratum)
    ]


def evaluate(settings: EvalSettings, with_errors: bool = True) -> dict[str, Any]:
    if settings.split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    if settings.stratum not in STRATA:
        raise ValueError(f"stratum must be one of {STRATA}")
    manifest = load_manifest(settings.manifest_path, settings.frozen_manifest_sha256)
    review_dir = settings.work_root / "review"
    all_tiles, all_candidates = load_ground_truth(review_dir, settings.work_root, manifest)
    tiles = select_tiles(all_tiles, settings.split, settings.stratum)
    candidates = [
        c for c in all_candidates
        if c.method == settings.method and c.split == settings.split
        and (settings.stratum == "all" or c.sampling_method == settings.stratum)
    ]
    stems = sorted({t.stem for t in tiles} | {c.stem for c in candidates})
    stem_index = {s: i for i, s in enumerate(stems)}
    n_stems = len(stems)

    warnings: list[str] = []
    if not manifest.frozen:
        warnings.append(
            "Split manifest is NOT frozen (split.frozen_manifest_sha256 is null); "
            "hash check skipped. Results are provisional until checkpoint 1."
        )

    errors: list[ErrorRegion] = []
    pixel = np.zeros((n_stems, N_CLASSES, 3), dtype=np.int64)
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    pred_outside = 0
    obj = {k: np.zeros((n_stems, 3), dtype=np.int64) for k in OBJECT_KINDS}
    obj_tiles = {k: 0 for k in OBJECT_KINDS}
    gt_components = np.zeros((n_stems, N_CLASSES), dtype=np.int64)
    tile_rows: list[dict[str, Any]] = []
    stem_blocks: dict[str, dict[str, list]] = defaultdict(lambda: {"gt": [], "pred": [], "agg": []})

    tile_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    store = PredictionStore(settings.work_root, settings.method) if tiles else None
    for tile in sorted(tiles, key=lambda t: (t.stem, t.y0, t.x0)):
        s = stem_index[tile.stem]
        pred_full = store.get(tile.stem)
        sem = pred_full.semantic
        if tile.y0 + tile.h > sem.shape[0] or tile.x0 + tile.w > sem.shape[1]:
            raise GroundTruthError(
                f"Tile {tile.item_id} exceeds prediction map {sem.shape} for {tile.stem}"
            )
        gt = tile.label
        pred = sem[tile.y0 : tile.y0 + tile.h, tile.x0 : tile.x0 + tile.w].copy()
        tile_maps[tile.item_id] = (gt, pred)
        pixel[s] += pixel_counts(gt, pred)
        cm, outside = confusion_matrix(gt, pred)
        confusion += cm
        pred_outside += outside
        for c in range(N_CLASSES):
            gt_components[s, c] += len(components(gt == c, settings.object_min_area_px))

        pred_agg_global = [
            Instance("agglomerate", i.bbox, i.polygon) for i in pred_full.instances
            if i.class_name == "agglomerate"
        ]
        pred_agg = _local_instances([i.polygon for i in pred_agg_global], tile.x0, tile.y0)
        pred_agg = [i for i in pred_agg if _polygon_mask(i.polygon, gt.shape).any()]
        objects = evaluate_objects(gt, pred, tile, pred_agg, settings,
                                   errors if with_errors else None)
        for name, value in objects.items():
            if value is not None:
                obj[name][s] += value
                obj_tiles[name] += 1

        valid = gt != IGNORE_LABEL
        pred_masked = np.where(valid, pred, IGNORE_LABEL).astype(np.uint8)
        annotated = tile.agglomerates is not None
        gt_inst = _local_instances(tile.agglomerates or [], tile.x0, tile.y0)
        px_nm = float(manifest.rows[tile.stem].get("pixel_size_nm") or "nan")
        if not np.isfinite(px_nm):
            raise GroundTruthError(f"Manifest has no pixel_size_nm for {tile.stem}")
        gt_k, ok_g = _kpis_or_nan(gt, gt_inst, valid, px_nm, annotated)
        pred_k, ok_p = _kpis_or_nan(pred_masked, pred_agg, valid, px_nm, annotated)
        if not (ok_g and ok_p):
            warnings.append(f"Tile {tile.item_id}: no valid non-artifact area; KPIs NaN")
        tile_rows.append({
            "item_id": tile.item_id, "stem": tile.stem, "x0": tile.x0, "y0": tile.y0,
            **{f"gt.{k}": v for k, v in gt_k.items()},
            **{f"pred.{k}": v for k, v in pred_k.items()},
        })
        blocks = stem_blocks[tile.stem]
        blocks["gt"].append((gt, gt_inst))
        blocks["pred"].append((pred_masked, pred_agg))
        blocks["agg"].append(annotated)

    stem_rows = []
    for stem in stems:
        if stem not in stem_blocks:
            continue
        b = stem_blocks[stem]
        annotated = all(b["agg"])
        px_nm = float(manifest.rows[stem]["pixel_size_nm"])
        gt_m, gt_i = _mosaic(b["gt"])
        pred_m, pred_i = _mosaic(b["pred"])
        valid = gt_m != IGNORE_LABEL
        gt_k, _ = _kpis_or_nan(gt_m, gt_i, valid, px_nm, annotated)
        pred_k, _ = _kpis_or_nan(pred_m, pred_i, valid, px_nm, annotated)
        stem_rows.append({
            "stem": stem, "n_tiles": len(b["gt"]),
            **{f"gt.{k}": v for k, v in gt_k.items()},
            **{f"pred.{k}": v for k, v in pred_k.items()},
        })

    tile_stem = np.array([stem_index[r["stem"]] for r in tile_rows], dtype=np.int64)
    stem_ids = np.array([stem_index[r["stem"]] for r in stem_rows], dtype=np.int64)

    def kpi_arrays(rows):
        return {k: (np.array([r[f"gt.{k}"] for r in rows], float),
                    np.array([r[f"pred.{k}"] for r in rows], float)) for k in KPI_NAMES}

    tile_kpi, stem_kpi = kpi_arrays(tile_rows), kpi_arrays(stem_rows)

    cand_groups = _candidate_groups(candidates, stem_index)

    def statistic(mult: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        if tiles:
            out.update({f"pixel.{k}": v for k, v in
                        _pixel_stats(np.tensordot(mult, pixel, axes=1)).items()})
            out.update({f"object.{k}": v for k, v in _object_stats(
                {n: mult @ a for n, a in obj.items()}).items()})
            t_idx = np.repeat(np.arange(len(tile_rows)), mult[tile_stem])
            s_idx = np.repeat(np.arange(len(stem_rows)), mult[stem_ids])
            for k in KPI_NAMES:
                for level, arrays, idx in (("tile", tile_kpi, t_idx), ("stem", stem_kpi, s_idx)):
                    g, p = arrays[k]
                    errs = quantity_errors(g[idx], p[idx])
                    for stat in ERROR_STATS:
                        out[f"kpi.{level}.{k}.{stat}"] = errs[stat]
        for key, group in cand_groups.items():
            w = group["w"] * mult[group["stem"]]
            out[f"candidate.{key}.precision"] = safe_div((w * group["pos"]).sum(), w.sum())
        return out

    point = statistic(np.ones(n_stems, dtype=np.int64)) if n_stems else {}
    cis = bootstrap_ci(n_stems, statistic, settings.n_boot, settings.seed) if n_stems else {}

    def with_ci(key: str) -> dict[str, float]:
        ci = cis.get(key, {"ci_low": math.nan, "ci_high": math.nan, "nan_frac": math.nan})
        return {"value": point.get(key, math.nan), **ci}

    n_tiles = len(tiles)
    pixel_total = pixel.sum(axis=0)
    gt_comp_total = gt_components.sum(axis=0)
    pixel_table = []
    for c in range(N_CLASSES):
        name = CLASS_NAMES[c]
        n_obj = int(gt_comp_total[c])
        stems_with = int(np.count_nonzero(gt_components[:, c]))
        row = {"class": name, "tp": int(pixel_total[c, 0]), "fp": int(pixel_total[c, 1]),
               "fn": int(pixel_total[c, 2]), "n_tiles": n_tiles, "n_stems": n_stems,
               "n_gt_objects": n_obj, "n_stems_with_gt": stems_with,
               "insufficient_n": n_obj < settings.min_gt_objects}
        for metric in ("precision", "recall", "f1", "iou"):
            row[metric] = with_ci(f"pixel.{name}.{metric}")
        pixel_table.append(row)

    object_table = []
    for name in OBJECT_KINDS:
        tp, fp, fn = (int(v) for v in obj[name].sum(axis=0))
        row = {"class": name, "tp": tp, "fp": fp, "fn": fn, "n_gt_objects": tp + fn,
               "n_pred_objects": tp + fp, "n_tiles": obj_tiles[name],
               "n_stems": int(np.count_nonzero(obj[name].sum(axis=1) > 0)) if n_tiles else 0,
               "match_iou": settings.match_iou[name],
               "insufficient_n": tp + fn < settings.min_gt_objects}
        for metric in ("precision", "recall", "f1"):
            row[metric] = with_ci(f"object.{name}.{metric}")
        object_table.append(row)

    candidate_table = []
    for key, group in sorted(cand_groups.items()):
        source, class_name, sampling = key.split("|")
        pos, w = group["pos"], group["w"]
        if sampling == "random":
            est = weighted_proportion(pos, w)
        else:
            k, n = float(pos.sum()), float(pos.size)
            low, high = wilson_interval(k, n)
            est = {"estimate": safe_div(k, n), "ci_low": low, "ci_high": high, "n_eff": n}
        candidate_table.append({
            "method": settings.method, "source": source, "class": class_name,
            "sampling": sampling, "weighted": sampling == "random",
            "n_positive": int(pos.sum()), "n_negative": int(pos.size - pos.sum()),
            "n_uncertain": group["n_uncertain"],
            "n_stems": int(len(set(group["stem"].tolist()))),
            "precision": est["estimate"], "wilson_low": est["ci_low"],
            "wilson_high": est["ci_high"], "n_eff": est["n_eff"],
            "bootstrap": with_ci(f"candidate.{key}.precision"),
            "insufficient_n": pos.size < settings.min_gt_objects,
        })
    for (source, class_name, sampling), n_unc in sorted(_uncertain_only(candidates, cand_groups).items()):
        candidate_table.append({
            "method": settings.method, "source": source, "class": class_name,
            "sampling": sampling, "weighted": sampling == "random", "n_positive": 0,
            "n_negative": 0, "n_uncertain": n_unc, "n_stems": 0,
            "precision": math.nan, "wilson_low": math.nan, "wilson_high": math.nan,
            "n_eff": 0.0, "bootstrap": with_ci("__none__"), "insufficient_n": True,
        })

    kpi_table = []
    for level, rows, arrays in (("tile", tile_rows, tile_kpi), ("stem", stem_rows, stem_kpi)):
        for k in KPI_NAMES:
            g, p = arrays[k]
            errs = quantity_errors(g, p)
            row = {"level": level, "kpi": k, "n_pairs": errs["n"],
                   "n_stems": len({r["stem"] for r, gv, pv in zip(rows, g, p)
                                   if np.isfinite(gv) and np.isfinite(pv)}),
                   "gt_mean": errs["gt_mean"], "pred_mean": errs["pred_mean"],
                   "rel_bias": errs["rel_bias"],
                   "insufficient_n": errs["n"] < settings.min_gt_objects}
            for stat in ERROR_STATS:
                row[stat] = with_ci(f"kpi.{level}.{k}.{stat}")
            kpi_table.append(row)

    rownorm = confusion / np.where(confusion.sum(1, keepdims=True) > 0,
                                   confusion.sum(1, keepdims=True), 1)
    rownorm = np.where(confusion.sum(1, keepdims=True) > 0, rownorm, np.nan)
    return {
        "method": settings.method,
        "split": settings.split,
        "stratum": settings.stratum,
        "headline": settings.split == "test" and settings.stratum == "random",
        "manifest": {"path": str(settings.manifest_path), "sha256": manifest.sha256,
                     "frozen": manifest.frozen},
        "bootstrap": {"replicates": settings.n_boot, "seed": settings.seed,
                      "unit": "stem", "ci": "percentile 95%"},
        "settings": {"match_iou": settings.match_iou,
                     "object_min_area_px": settings.object_min_area_px,
                     "pred_object_max_ignored_frac": settings.pred_object_max_ignored_frac,
                     "min_gt_objects": settings.min_gt_objects},
        "counts": {"n_tiles": n_tiles, "n_stems": n_stems, "stems": stems,
                   "n_candidates_decided": len(candidates),
                   "n_gt_tiles_all_splits": len(all_tiles),
                   "n_candidates_all_methods_splits": len(all_candidates),
                   "gt_pixels_predicted_ignore": int(pred_outside)},
        "pixel": pixel_table,
        "confusion": {"classes": [CLASS_NAMES[c] for c in range(N_CLASSES)],
                      "raw": confusion.tolist(), "row_normalized": rownorm.tolist()},
        "object": object_table,
        "candidate": candidate_table,
        "kpi": kpi_table,
        "kpi_per_tile": tile_rows,
        "kpi_per_stem": stem_rows,
        "warnings": warnings,
        "_errors": errors,
        "_tile_maps": tile_maps,
    }


def _candidate_groups(
    candidates: list[CandidateDecision], stem_index: dict[str, int]
) -> dict[str, dict[str, Any]]:
    """Group decided candidates by source|class|sampling; uncertain only counted.

    Random-stratum items carry their design weight (sampling.weight, interpreted as
    1/inclusion probability); other strata are unweighted.
    """
    raw: dict[str, dict[str, list]] = defaultdict(lambda: {"pos": [], "w": [], "stem": [], "unc": 0})
    for c in candidates:
        key = f"{c.source}|{c.class_name}|{c.sampling_method}"
        if c.outcome == "uncertain":
            raw[key]["unc"] += 1
            continue
        raw[key]["pos"].append(1.0 if c.outcome == "positive" else 0.0)
        raw[key]["w"].append(c.weight if c.sampling_method == "random" else 1.0)
        raw[key]["stem"].append(stem_index[c.stem])
    return {
        key: {"pos": np.array(v["pos"]), "w": np.array(v["w"]),
              "stem": np.array(v["stem"], dtype=np.int64), "n_uncertain": v["unc"]}
        for key, v in raw.items() if v["pos"]
    }


def _uncertain_only(candidates, groups) -> dict[tuple[str, str, str], int]:
    out: dict[tuple[str, str, str], int] = defaultdict(int)
    for c in candidates:
        key = f"{c.source}|{c.class_name}|{c.sampling_method}"
        if key not in groups and c.outcome == "uncertain":
            out[(c.source, c.class_name, c.sampling_method)] += 1
    return out
