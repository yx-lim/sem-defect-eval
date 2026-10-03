"""Build the human review set: exhaustive tiles plus per-class candidates."""

from __future__ import annotations

import hashlib
import json
import math

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_polygon
from skimage.measure import approximate_polygon, find_contours

from sem.qc.schema import CLASS_IDS, IGNORE_LABEL, make_item_id, write_jsonl
from sem.qc.split import load_manifest as _frozen_load_manifest


def _round_half_down(x: float) -> int:
    """n_uncertainty = round(n * frac) with halves rounding down (25*0.3 -> 7)."""
    return int(math.ceil(x - 0.5))


def _stratum_rng(seed: int, stratum: str) -> np.random.Generator:
    digest = int(hashlib.sha256(stratum.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng([int(seed), digest])


def load_manifest(
    path: str | Path, expected_sha256: str | None = None
) -> dict[str, dict[str, Any]]:
    """Load the split manifest, keeping only val/test rows keyed by stem."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest is required but missing: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for row in _frozen_load_manifest(path, expected_sha256):
        if row.get("split") not in {"val", "test"}:
            continue
        rows[row["stem"]] = row
    return rows


def load_instances(path: str | Path) -> list[dict[str, Any]]:
    """Load an instances JSON file (list or {"instances": [...]})."""
    with Path(path).open(encoding="utf-8") as instances_file:
        data = json.load(instances_file)
    if isinstance(data, dict):
        data = data.get("instances", [])
    return [dict(instance) for instance in data]


def list_methods(preds_root: str | Path) -> list[str]:
    root = Path(preds_root)
    if not root.is_dir():
        return []
    return sorted(child.name for child in root.iterdir() if child.is_dir())


def _load_semantic(preds_root: Path, method: str, stem: str) -> np.ndarray | None:
    path = preds_root / method / f"{stem}_semantic.png"
    if not path.exists():
        return None
    return np.asarray(Image.open(path), dtype=np.uint8)


def _load_uncertainty(preds_root: Path, method: str, stem: str) -> np.ndarray | None:
    path = preds_root / method / f"{stem}_uncertainty.png"
    if not path.exists():
        return None
    return np.asarray(Image.open(path), dtype=np.uint8).astype(np.float32) / 255.0


def _stem_size(manifest_row: dict[str, Any]) -> tuple[int, int]:
    return int(manifest_row["height"]), int(manifest_row["width"])


def _tiles_on_grid(
    height: int, width: int, tile_px: int, stride: int
) -> Iterable[tuple[int, int]]:
    for y0 in range(0, height - tile_px + 1, stride):
        for x0 in range(0, width - tile_px + 1, stride):
            yield x0, y0


def _valid_fraction(semantic: np.ndarray, x0: int, y0: int, tile_px: int) -> float:
    window = semantic[y0 : y0 + tile_px, x0 : x0 + tile_px]
    return float(np.mean(window != IGNORE_LABEL))


def _largest_remainder(
    budget: int, demand: dict[str, int]
) -> dict[str, int]:
    """Hamilton allocation: floors then leftover seats by (-remainder, name)."""
    keys = sorted(key for key, size in demand.items() if size > 0)
    total = sum(demand[key] for key in keys)
    alloc = {key: 0 for key in demand}
    if not keys or budget <= 0:
        return alloc
    quotas = {key: budget * demand[key] / total for key in keys}
    for key in keys:
        alloc[key] = min(int(quotas[key]), demand[key])
    order = sorted(
        keys, key=lambda k: (-(quotas[k] - math.floor(quotas[k])), k)
    )
    while sum(alloc.values()) < budget:
        progressed = False
        for key in order:
            if sum(alloc.values()) >= budget:
                break
            if alloc[key] < demand[key]:
                alloc[key] += 1
                progressed = True
        if not progressed:
            break
    return alloc


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _rasterize(
    instance: dict[str, Any], region: tuple[int, int, int, int]
) -> np.ndarray:
    """Boolean mask of the instance within region (x0, y0, x1, y1)."""
    rx0, ry0, rx1, ry1 = region
    height = max(1, ry1 - ry0)
    width = max(1, rx1 - rx0)
    mask = np.zeros((height, width), dtype=bool)
    poly = instance.get("polygon") or []
    if len(poly) >= 3:
        rr, cc = draw_polygon(
            [float(p[1]) - ry0 for p in poly],
            [float(p[0]) - rx0 for p in poly],
            shape=(height, width),
        )
        mask[rr, cc] = True
    else:
        bx0, by0, bx1, by1 = [float(v) for v in instance["bbox"]]
        ix0 = max(0, int(math.floor(bx0)) - rx0)
        iy0 = max(0, int(math.floor(by0)) - ry0)
        ix1 = min(width, int(math.ceil(bx1)) - rx0)
        iy1 = min(height, int(math.ceil(by1)) - ry0)
        if ix1 > ix0 and iy1 > iy0:
            mask[iy0:iy1, ix0:ix1] = True
    return mask


def _bbox_int(instance: dict[str, Any]) -> tuple[int, int, int, int]:
    bx0, by0, bx1, by1 = [float(v) for v in instance["bbox"]]
    return (
        int(math.floor(bx0)),
        int(math.floor(by0)),
        int(math.ceil(bx1)),
        int(math.ceil(by1)),
    )


def _polygon_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0, ay0, ax1, ay1 = _bbox_int(a)
    bx0, by0, bx1, by1 = _bbox_int(b)
    region = (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))
    if region[2] - region[0] > 4096 or region[3] - region[1] > 4096:
        region = (
            min(ax0, bx0),
            min(ay0, by0),
            min(ax0, bx0) + 4096,
            min(ay0, by0) + 4096,
        )
    mask_a = _rasterize(a, region)
    mask_b = _rasterize(b, region)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return _rect_intersects(a, b)


def _iou_cached(
    a: tuple[tuple[int, int, int, int], np.ndarray, int],
    b: tuple[tuple[int, int, int, int], np.ndarray, int],
) -> float:
    """IoU from bbox-local boolean masks; identical to _polygon_iou modulo the
    dropped >4096 px region clamp."""
    (ax0, ay0, ax1, ay1), mask_a, area_a = a
    (bx0, by0, bx1, by1), mask_b, area_b = b
    ox0, oy0 = max(ax0, bx0), max(ay0, by0)
    ox1, oy1 = min(ax1, bx1), min(ay1, by1)
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0
    inter = int(
        (
            mask_a[oy0 - ay0 : oy1 - ay0, ox0 - ax0 : ox1 - ax0]
            & mask_b[oy0 - by0 : oy1 - by0, ox0 - bx0 : ox1 - bx0]
        ).sum()
    )
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _dedup_instances(
    instances: list[dict[str, Any]], iou_threshold: float
) -> tuple[list[dict[str, Any]], int]:
    """Greedy IoU dedup with a uniform-grid bbox prefilter."""
    ordered = sorted(
        instances,
        key=lambda inst: (
            -float(inst.get("score") or 0.0),
            str(inst.get("source") or ""),
            str(inst.get("item_id") or ""),
        ),
    )
    cell_px = 512
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    kept: list[dict[str, Any]] = []
    kept_order: list[int] = []
    merged = 0
    mask_cache: dict[int, tuple[tuple[int, int, int, int], np.ndarray, int]] = {}

    def _mask_entry(
        order_idx: int,
    ) -> tuple[tuple[int, int, int, int], np.ndarray, int]:
        entry = mask_cache.get(order_idx)
        if entry is None:
            box = _bbox_int(ordered[order_idx])
            mask = _rasterize(ordered[order_idx], box)
            entry = (box, mask, int(mask.sum()))
            mask_cache[order_idx] = entry
        return entry

    for order_idx, inst in enumerate(ordered):
        bbox = _bbox_int(inst)
        cx0, cy0 = bbox[0] // cell_px, bbox[1] // cell_px
        cx1, cy1 = bbox[2] // cell_px, bbox[3] // cell_px
        duplicate_of = None
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                for kept_idx in buckets[(cy, cx)]:
                    other = kept[kept_idx]
                    other_box = mask_cache.get(kept_order[kept_idx], (_bbox_int(other),))[0]
                    if not _bboxes_overlap(bbox, other_box):
                        continue
                    if (
                        _iou_cached(
                            _mask_entry(order_idx),
                            _mask_entry(kept_order[kept_idx]),
                        )
                        > iou_threshold
                    ):
                        duplicate_of = other
                        break
                if duplicate_of is not None:
                    break
            if duplicate_of is not None:
                break
        if duplicate_of is not None:
            merged += 1
            duplicate_of.setdefault("duplicates", []).append(
                {
                    "source": inst.get("source"),
                    "class_name": inst.get("class_name"),
                    "subtype": inst.get("subtype"),
                    "score": inst.get("score"),
                    "bbox": inst.get("bbox"),
                }
            )
            continue
        inst.setdefault("duplicates", [])
        kept.append(inst)
        kept_order.append(order_idx)
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                buckets[(cy, cx)].append(len(kept) - 1)
    return kept, merged


def _instance_uncertainty(
    instance: dict[str, Any], uncertainty: np.ndarray | None
) -> float:
    if uncertainty is None:
        return 1.0 - float(instance.get("score") or 0.0)
    x0, y0, x1, y1 = _bbox_int(instance)
    x0 = max(0, min(x0, uncertainty.shape[1] - 1))
    y0 = max(0, min(y0, uncertainty.shape[0] - 1))
    x1 = max(x0 + 1, min(x1, uncertainty.shape[1]))
    y1 = max(y0 + 1, min(y1, uncertainty.shape[0]))
    mask = _rasterize(instance, (x0, y0, x1, y1))
    region = uncertainty[y0:y1, x0:x1]
    if not mask.any():
        return float(region.mean())
    return float(region[mask].mean())


def _cc_polygon(mask_crop: np.ndarray, x0: int, y0: int) -> list[list[float]]:
    """Longest outer contour of a component crop, as full-res [x, y] floats."""
    padded = np.pad(mask_crop.astype(float), 1)
    contours = find_contours(padded, 0.5)
    if contours:
        contour = max(contours, key=len)
        approx = approximate_polygon(contour, tolerance=0.5)
        points = [
            [float(col) - 1 + x0, float(row) - 1 + y0]
            for row, col in approx
        ]
        if len(points) >= 3:
            return points
    h, w = mask_crop.shape
    return [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]


def _semantic_cc_instances(
    semantic: np.ndarray,
    u_map: np.ndarray | None,
    method: str,
    stem: str,
    cc_min_area: dict[str, int],
) -> list[dict[str, Any]]:
    """Connected-component candidate records from a semantic prediction map."""
    records = []
    structure = np.ones((3, 3))
    for class_name in sorted(cc_min_area):
        min_area = int(cc_min_area[class_name])
        class_id = CLASS_IDS[class_name]
        labels, n_labels = ndi.label(semantic == class_id, structure=structure)
        if n_labels == 0:
            continue
        counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
        kept = np.nonzero(counts >= min_area)[0]
        kept = kept[kept > 0]
        sums = None
        if u_map is not None:
            sums = np.bincount(
                labels.ravel(), weights=u_map.ravel(), minlength=n_labels + 1
            )
        slices = ndi.find_objects(labels)
        for label_id in kept:
            sl = slices[label_id - 1]
            area = int(counts[label_id])
            mean_u = (
                float(sums[label_id] / area) if sums is not None else None
            )
            x0, x1 = int(sl[1].start), int(sl[1].stop)
            y0, y1 = int(sl[0].start), int(sl[0].stop)
            records.append(
                {
                    "class_name": class_name,
                    "subtype": None,
                    "score": None,
                    "source": f"{method}:semantic_cc",
                    "method": method,
                    "stem": stem,
                    "bbox": [x0, y0, x1, y1],
                    "polygon": _cc_polygon(labels[sl] == label_id, x0, y0),
                    "area_px": area,
                    "uncertainty": mean_u,
                }
            )
    return records


def _candidate_tile(
    instance: dict[str, Any],
    height: int,
    width: int,
    min_context_px: int,
    max_tile_px: int = 1024,
) -> dict[str, Any]:
    bx0, by0, bx1, by1 = [float(v) for v in instance["bbox"]]
    bw, bh = bx1 - bx0, by1 - by0
    pad_x = max(16.0, 0.25 * bw)
    pad_y = max(16.0, 0.25 * bh)
    w = min(
        width,
        min(max_tile_px, max(min_context_px, int(math.ceil(bw + 2 * pad_x)))),
    )
    h = min(
        height,
        min(max_tile_px, max(min_context_px, int(math.ceil(bh + 2 * pad_y)))),
    )
    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x0 = min(max(0, x0), width - w)
    y0 = min(max(0, y0), height - h)
    return {
        "x0": x0,
        "y0": y0,
        "w": w,
        "h": h,
        "truncated": bw > w or bh > h,
    }


def build_review_set(
    manifest_path: str | Path,
    preds_root: str | Path,
    out_dir: str | Path,
    review_config: dict[str, Any],
    tile_px: int = 512,
    manifest_sha256: str | None = None,
    use_vlm: bool = False,
    image_loader: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """Sample review items and write items.jsonl, summary.json, prefill PNGs."""
    preds_root = Path(preds_root)
    out_dir = Path(out_dir)
    items_path = out_dir / "items.jsonl"
    decisions_path = out_dir / "decisions.jsonl"
    if (
        items_path.exists()
        and decisions_path.exists()
        and decisions_path.stat().st_size > 0
        and not force
    ):
        raise RuntimeError(
            f"Refusing to overwrite {items_path}: {decisions_path} is non-empty "
            "(pass force=True / --force to rebuild anyway)"
        )

    seed = int(review_config["seed"])
    prefill = str(review_config["prefill_method"])
    n_exhaustive = int(review_config["n_exhaustive"])
    n_unc_tiles = _round_half_down(
        n_exhaustive * float(review_config["exhaustive_uncertainty_frac"])
    )
    n_random_tiles = n_exhaustive - n_unc_tiles
    min_valid = float(review_config["exhaustive_min_valid_frac"])
    unc_stride = int(review_config["exhaustive_uncertainty_stride_px"])
    n_candidates = int(review_config["n_candidates"])
    n_unc_candidates = _round_half_down(
        n_candidates * float(review_config["candidate_uncertainty_frac"])
    )
    n_random_candidates = n_candidates - n_unc_candidates
    min_per_class = int(review_config["candidate_min_per_class"])
    candidate_classes = set(review_config["candidate_classes"])
    dedup_iou = float(review_config["dedup_iou"])
    min_context_px = int(review_config["candidate_min_context_px"])
    max_tile_px = int(review_config.get("candidate_max_tile_px", 1024))

    manifest = load_manifest(manifest_path, manifest_sha256)
    methods = list_methods(preds_root)
    if prefill not in methods:
        raise FileNotFoundError(
            f"Prefill method {prefill!r} has no directory under {preds_root}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    prefill_dir = out_dir / "prefill"
    prefill_dir.mkdir(exist_ok=True)

    # ---- per-stem prediction assets -------------------------------------
    semantics: dict[str, np.ndarray] = {}
    uncertainties: dict[str, np.ndarray | None] = {}
    prefill_instances: dict[str, list[dict[str, Any]]] = {}
    skipped: list[dict[str, str]] = []
    used_stems: list[str] = []
    for stem in sorted(manifest):
        semantic = _load_semantic(preds_root, prefill, stem)
        if semantic is None:
            skipped.append({"stem": stem, "reason": f"missing {prefill} semantic"})
            continue
        used_stems.append(stem)
        semantics[stem] = semantic
        uncertainties[stem] = _load_uncertainty(preds_root, prefill, stem)
        inst_path = preds_root / prefill / f"{stem}_instances.json"
        prefill_instances[stem] = (
            load_instances(inst_path) if inst_path.exists() else []
        )

    # ---- exhaustive tiles: random part ----------------------------------
    strata_population: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tile_lookup: dict[str, tuple[str, int, int]] = {}
    for stem in used_stems:
        row = manifest[stem]
        height, width = _stem_size(row)
        stratum = f"{row['split']}|{row['batch']}"
        semantic = semantics[stem]
        for x0, y0 in _tiles_on_grid(height, width, tile_px, tile_px):
            if _valid_fraction(semantic, x0, y0, tile_px) < min_valid:
                continue
            item_id = make_item_id(
                stem, "exhaustive_tile", x0, y0, tile_px, tile_px, prefill
            )
            tile_lookup[item_id] = (stem, x0, y0)
            strata_population[stratum].append(
                {"item_id": item_id, "stem": stem, "x0": x0, "y0": y0}
            )
    for entries in strata_population.values():
        entries.sort(key=lambda e: e["item_id"])

    strata_sizes = {s: len(p) for s, p in strata_population.items()}
    random_alloc = _largest_remainder(n_random_tiles, strata_sizes)
    chosen_tiles: dict[str, dict[str, Any]] = {}
    exhaustive_stats: dict[str, dict[str, int]] = {}
    chosen_rects: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for stratum in sorted(strata_population):
        population = strata_population[stratum]
        n_s = min(random_alloc[stratum], len(population))
        rng = _stratum_rng(seed, stratum)
        picks = (
            rng.choice(len(population), size=n_s, replace=False)
            if n_s
            else np.array([], dtype=int)
        )
        for idx in sorted(int(i) for i in picks):
            entry = population[idx]
            item_id = entry["item_id"]
            chosen_tiles[item_id] = {
                **entry,
                "sampling": {
                    "stratum": stratum,
                    "method": "random",
                    "weight": len(population) / n_s,
                },
            }
            chosen_rects[entry["stem"]].append(
                (entry["x0"], entry["y0"], entry["x0"] + tile_px, entry["y0"] + tile_px)
            )
        exhaustive_stats[stratum] = {"population": len(population), "n_random": n_s}

    # ---- exhaustive tiles: uncertainty part -----------------------------
    unc_windows: list[dict[str, Any]] = []
    for stem in used_stems:
        u_map = uncertainties[stem]
        if u_map is None:
            continue
        row = manifest[stem]
        height, width = _stem_size(row)
        stratum = f"{row['split']}|{row['batch']}"
        semantic = semantics[stem]
        for x0, y0 in _tiles_on_grid(height, width, tile_px, unc_stride):
            if _valid_fraction(semantic, x0, y0, tile_px) < min_valid:
                continue
            window_u = u_map[y0 : y0 + tile_px, x0 : x0 + tile_px]
            window_v = semantics[stem][y0 : y0 + tile_px, x0 : x0 + tile_px]
            valid = window_v != IGNORE_LABEL
            score = float(window_u[valid].mean()) if valid.any() else 0.0
            item_id = make_item_id(
                stem, "exhaustive_tile", x0, y0, tile_px, tile_px, prefill
            )
            unc_windows.append(
                {
                    "item_id": item_id,
                    "stem": stem,
                    "x0": x0,
                    "y0": y0,
                    "score": score,
                    "stratum": stratum,
                }
            )
    unc_windows.sort(key=lambda w: (-w["score"], w["item_id"]))
    n_unc_used = 0
    for window in unc_windows:
        if n_unc_used >= n_unc_tiles:
            break
        rect = (
            window["x0"],
            window["y0"],
            window["x0"] + tile_px,
            window["y0"] + tile_px,
        )
        if window["item_id"] in chosen_tiles:
            continue
        if any(_rect_intersects(rect, r) for r in chosen_rects[window["stem"]]):
            continue
        chosen_tiles[window["item_id"]] = {
            **{k: window[k] for k in ("item_id", "stem", "x0", "y0")},
            "sampling": {
                "stratum": window["stratum"],
                "method": "uncertainty",
                "weight": None,
            },
            "mean_uncertainty": window["score"],
        }
        chosen_rects[window["stem"]].append(rect)
        n_unc_used += 1

    # ---- candidates: gather pool ----------------------------------------
    pool: list[dict[str, Any]] = []
    instances_by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for method in methods:
        for stem in used_stems:
            inst_path = preds_root / method / f"{stem}_instances.json"
            if not inst_path.exists():
                continue
            for inst in load_instances(inst_path):
                if inst.get("class_name") not in candidate_classes:
                    continue
                record = dict(inst)
                record["source"] = method
                record["method"] = method
                record["stem"] = stem
                pool.append(record)

    stem_sizes = {stem: _stem_size(manifest[stem]) for stem in used_stems}

    # semantic connected-component candidates from every method's semantic map
    cc_min_area = review_config.get("cc_min_area_px") or {}
    if cc_min_area:
        for method in methods:
            for stem in used_stems:
                cc_semantic = (
                    semantics[stem]
                    if method == prefill
                    else _load_semantic(preds_root, method, stem)
                )
                if cc_semantic is None:
                    continue
                cc_u = _load_uncertainty(preds_root, method, stem)
                pool.extend(
                    _semantic_cc_instances(
                        cc_semantic, cc_u, method, stem, cc_min_area
                    )
                )
    pool_size_before = len(pool)

    for inst in pool:
        instances_by_stem[inst["stem"]].append(inst)

    deduped: list[dict[str, Any]] = []
    merged_total = 0
    for stem in sorted(instances_by_stem):
        stem_deduped, merged = _dedup_instances(instances_by_stem[stem], dedup_iou)
        merged_total += merged
        deduped.extend(stem_deduped)

    # candidate item_ids (depend only on stem/source/tile)
    id_collisions = 0
    seen_ids: dict[str, dict[str, Any]] = {}
    final_pool: list[dict[str, Any]] = []
    for inst in deduped:
        height, width = stem_sizes[inst["stem"]]
        tile = _candidate_tile(inst, height, width, min_context_px, max_tile_px)
        item_id = make_item_id(
            inst["stem"], "candidate", tile["x0"], tile["y0"], tile["w"], tile["h"],
            inst["source"],
        )
        if item_id in seen_ids:
            id_collisions += 1
            seen_ids[item_id].setdefault("duplicates", []).append(
                {
                    "source": inst.get("source"),
                    "class_name": inst.get("class_name"),
                    "subtype": inst.get("subtype"),
                    "score": inst.get("score"),
                    "bbox": inst.get("bbox"),
                    "reason": "same_review_tile",
                }
            )
            continue
        seen_ids[item_id] = inst
        inst["item_id"] = item_id
        inst["tile"] = tile
        final_pool.append(inst)

    # uncertainty per instance from its own method's map (skip already set)
    method_u: dict[tuple[str, str], np.ndarray | None] = {}
    for inst in final_pool:
        if "uncertainty" in inst:
            continue
        key = (inst["method"], inst["stem"])
        if key not in method_u:
            method_u[key] = _load_uncertainty(preds_root, inst["method"], inst["stem"])
        inst["uncertainty"] = _instance_uncertainty(inst, method_u[key])

    # ---- candidates: random allocation ----------------------------------
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for inst in final_pool:
        by_stratum[f"{inst['class_name']}|{inst['source']}"].append(inst)
    for entries in by_stratum.values():
        entries.sort(key=lambda e: e["item_id"])
    by_class: dict[str, list[str]] = defaultdict(list)
    for stratum, entries in by_stratum.items():
        by_class[stratum.split("|")[0]].append(stratum)

    alloc = {s: 0 for s in by_stratum}
    budget_left = n_random_candidates
    for class_name in sorted(by_class):
        strata = by_class[class_name]
        n_c = sum(len(by_stratum[s]) for s in strata)
        floor_c = min(n_c, min_per_class)
        demand = {s: len(by_stratum[s]) for s in strata}
        sub = _largest_remainder(min(floor_c, budget_left), demand)
        for s, n in sub.items():
            alloc[s] += n
            budget_left -= n
    while budget_left > 0:
        remaining = {s: len(by_stratum[s]) - alloc[s] for s in by_stratum}
        if not any(v > 0 for v in remaining.values()):
            break
        extra = _largest_remainder(budget_left, remaining)
        gained = sum(extra.values())
        for s, n in extra.items():
            alloc[s] += n
        budget_left -= gained
        if gained == 0:
            break

    chosen_candidates: dict[str, dict[str, Any]] = {}
    candidate_strata_stats: dict[str, dict[str, Any]] = {}
    n_random_used = 0
    for stratum in sorted(by_stratum):
        population = by_stratum[stratum]
        n_s = min(alloc[stratum], len(population))
        rng = _stratum_rng(seed, stratum)
        picks = (
            rng.choice(len(population), size=n_s, replace=False)
            if n_s
            else np.array([], dtype=int)
        )
        for idx in sorted(int(i) for i in picks):
            inst = population[idx]
            chosen_candidates[inst["item_id"]] = {
                "inst": inst,
                "sampling": {
                    "stratum": stratum,
                    "method": "random",
                    "weight": len(population) / n_s,
                },
            }
        candidate_strata_stats[stratum] = {
            "population": len(population),
            "n_random": n_s,
            "weight": (len(population) / n_s) if n_s else None,
        }
        n_random_used += n_s

    # ---- candidates: uncertainty part -----------------------------------
    remaining_pool = [
        inst for inst in final_pool if inst["item_id"] not in chosen_candidates
    ]
    n_unc_wanted = n_unc_candidates + (n_random_candidates - n_random_used)
    # round-robin by class: each round, every class takes its most-uncertain
    # remaining item (items with uncertainty None are never picked)
    remaining_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for inst in remaining_pool:
        if inst.get("uncertainty") is not None:
            remaining_by_class[inst["class_name"]].append(inst)
    for entries in remaining_by_class.values():
        entries.sort(key=lambda i: (-float(i["uncertainty"]), i["item_id"]))
    unc_index = {cls: 0 for cls in remaining_by_class}
    n_unc_used_cand = 0
    while n_unc_used_cand < n_unc_wanted:
        progressed = False
        for cls in sorted(remaining_by_class):
            if n_unc_used_cand >= n_unc_wanted:
                break
            idx = unc_index[cls]
            if idx >= len(remaining_by_class[cls]):
                continue
            inst = remaining_by_class[cls][idx]
            unc_index[cls] += 1
            stratum = f"{inst['class_name']}|{inst['source']}"
            chosen_candidates[inst["item_id"]] = {
                "inst": inst,
                "sampling": {
                    "stratum": stratum,
                    "method": "uncertainty",
                    "weight": None,
                },
            }
            n_unc_used_cand += 1
            progressed = True
        if not progressed:
            break

    # ---- blank-start anchoring control ------------------------------------
    n_blank = int(review_config.get("blank_start_tiles", 0))
    random_tile_ids = sorted(
        item_id
        for item_id, t in chosen_tiles.items()
        if t["sampling"]["method"] == "random"
    )
    blank_ids: set[str] = set()
    if n_blank > 0 and random_tile_ids:
        rng = _stratum_rng(seed, "blank_start")
        picks = rng.choice(
            len(random_tile_ids),
            size=min(n_blank, len(random_tile_ids)),
            replace=False,
        )
        blank_ids = {random_tile_ids[int(i)] for i in picks}

    # ---- assemble items --------------------------------------------------
    items: list[dict[str, Any]] = []
    for item_id, tile in chosen_tiles.items():
        stem, x0, y0 = tile["stem"], tile["x0"], tile["y0"]
        row = manifest[stem]
        crop = semantics[stem][y0 : y0 + tile_px, x0 : x0 + tile_px]
        rel = Path("prefill") / f"{item_id}.png"
        Image.fromarray(crop, mode="L").save(out_dir / rel)
        is_blank = item_id in blank_ids
        u_map = uncertainties[stem]
        if u_map is not None:
            window_u = u_map[y0 : y0 + tile_px, x0 : x0 + tile_px]
            valid = crop != IGNORE_LABEL
            mean_u = float(window_u[valid].mean()) if valid.any() else None
        else:
            mean_u = None
        insts = [
            inst
            for inst in prefill_instances.get(stem, [])
            if inst.get("class_name") in {"agglomerate", "artifact"}
            and _rect_intersects(
                _bbox_int(inst), (x0, y0, x0 + tile_px, y0 + tile_px)
            )
        ]
        items.append(
            {
                "item_id": item_id,
                "stem": stem,
                "batch": row["batch"],
                "split": row["split"],
                "kind": "exhaustive_tile",
                "tile": {"x0": x0, "y0": y0, "w": tile_px, "h": tile_px},
                "sampling": tile["sampling"],
                "prefill": "blank" if is_blank else prefill,
                "proposal": {
                    "source": prefill,
                    "class_name": None,
                    "subtype": None,
                    "polygon": None,
                    "semantic_png": None if is_blank else str(rel),
                    "reference_semantic_png": str(rel) if is_blank else None,
                    "score": None,
                    "uncertainty": mean_u,
                    "instances": [] if is_blank else insts,
                },
                "vlm_suggestion": None,
                "human": None,
            }
        )
    for item_id, entry in chosen_candidates.items():
        inst = entry["inst"]
        stem = inst["stem"]
        row = manifest[stem]
        items.append(
            {
                "item_id": item_id,
                "stem": stem,
                "batch": row["batch"],
                "split": row["split"],
                "kind": "candidate",
                "tile": inst["tile"],
                "sampling": entry["sampling"],
                "proposal": {
                    "source": inst["source"],
                    "class_name": inst.get("class_name"),
                    "subtype": inst.get("subtype"),
                    "polygon": inst.get("polygon"),
                    "bbox": inst.get("bbox"),
                    "semantic_png": None,
                    "score": inst.get("score"),
                    "uncertainty": inst["uncertainty"],
                    "duplicates": inst.get("duplicates", []),
                    "method": inst.get("method"),
                    "area_px": inst.get("area_px"),
                },
                "vlm_suggestion": None,
                "human": None,
            }
        )

    order = {"exhaustive_tile": 0, "candidate": 1}
    items.sort(
        key=lambda it: (
            order[it["kind"]],
            it["sampling"]["method"],
            it["sampling"]["stratum"],
            it["item_id"],
        )
    )

    vlm_stats = {"requested": 0, "ok": 0, "invalid": 0, "errors": 0, "skipped_reason": None}
    if use_vlm:
        from sem.qc.review import vlm as vlm_module

        vlm_stats = vlm_module.annotate_candidates(
            items, image_loader, review_config.get("vlm", {}), out_dir
        )

    per_class: dict[str, dict[str, Any]] = {}
    for class_name in sorted(candidate_classes):
        strata = by_class.get(class_name, [])
        n_pool = sum(len(by_stratum[s]) for s in strata)
        n_r = sum(
            candidate_strata_stats[s]["n_random"] for s in strata
        )
        n_u = sum(
            1
            for e in chosen_candidates.values()
            if e["sampling"]["method"] == "uncertainty"
            and e["sampling"]["stratum"].split("|")[0] == class_name
        )
        per_class[class_name] = {
            "pool": n_pool,
            "n_random": n_r,
            "n_uncertainty": n_u,
            "shortfall_vs_min": max(0, min_per_class - n_pool),
        }
    per_method = {m: 0 for m in methods}
    for inst in final_pool:
        per_method[inst["source"]] = per_method.get(inst["source"], 0) + 1
    pool_per_source: dict[str, int] = defaultdict(int)
    for inst in final_pool:
        pool_per_source[inst["source"]] += 1

    summary = {
        "seed": seed,
        "config": review_config,
        "methods": methods,
        "stems_used": used_stems,
        "stems_skipped": skipped,
        "exhaustive": {
            "strata": exhaustive_stats,
            "n_uncertainty": n_unc_used,
            "n_random_requested": n_random_tiles,
            "n_uncertainty_requested": n_unc_tiles,
        },
        "candidates": {
            "pool_before_dedup": pool_size_before,
            "pool_after_dedup": len(deduped),
            "pool_final": len(final_pool),
            "duplicates_merged": merged_total,
            "strata": candidate_strata_stats,
            "per_class": per_class,
            "per_method": per_method,
            "id_collisions": id_collisions,
            "pool_per_source": dict(pool_per_source),
            "n_random_requested": n_random_candidates,
            "n_uncertainty_requested": n_unc_candidates,
            "shortfall": max(0, n_candidates - len(chosen_candidates)),
        },
        "vlm": vlm_stats,
    }

    write_jsonl(items_path, items)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_counts_md(out_dir / "counts.md", items, summary)
    return {"items": items, "summary": summary}


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _write_counts_md(
    path: Path, items: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    """Deterministic markdown tables describing the review set."""
    lines = [
        "# Review set counts",
        "",
        f"Total items: {len(items)}",
        "",
        "## Per stem x kind x sampling method",
        "",
        "| stem | split | kind | method | n |",
        "|---|---|---|---|---|",
    ]
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    meta: dict[str, str] = {}
    for item in items:
        counts[(item["stem"], item["kind"], item["sampling"]["method"])] += 1
        meta[item["stem"]] = item["split"]
    for (stem, kind, method), n in sorted(counts.items()):
        lines.append(f"| {stem} | {meta[stem]} | {kind} | {method} | {n} |")

    lines += [
        "",
        "## Per class x source x sampling method (candidates)",
        "",
        "| class | source | random | uncertainty | pool |",
        "|---|---|---|---|---|",
    ]
    cand = summary["candidates"]
    strata = cand["strata"]
    picked: dict[tuple[str, str, str], int] = defaultdict(int)
    for item in items:
        if item["kind"] == "candidate":
            key = (
                item["proposal"]["class_name"],
                item["proposal"]["source"],
                item["sampling"]["method"],
            )
            picked[key] += 1
    for stratum in sorted(strata):
        class_name, source = stratum.split("|", 1)
        lines.append(
            f"| {class_name} | {source} "
            f"| {picked.get((class_name, source, 'random'), 0)} "
            f"| {picked.get((class_name, source, 'uncertainty'), 0)} "
            f"| {strata[stratum]['population']} |"
        )
    for class_name in sorted(cand["per_class"]):
        if not any(
            s.split("|", 1)[0] == class_name for s in strata
        ):
            lines.append(f"| {class_name} | — | 0 | 0 | 0 |")

    lines += [
        "",
        "## Per stratum",
        "",
        "| stratum | population | n_random | weight |",
        "|---|---|---|---|",
    ]
    for stratum, stats in sorted(summary["exhaustive"]["strata"].items()):
        lines.append(
            f"| {_md_cell(stratum)} (exhaustive) | {stats['population']} "
            f"| {stats['n_random']} | — |"
        )
    for stratum, stats in sorted(strata.items()):
        lines.append(
            f"| {_md_cell(stratum)} | {stats['population']} "
            f"| {stats['n_random']} | {stats['weight']} |"
        )

    lines += ["", "## Shortfalls", ""]
    for class_name, stats in sorted(cand["per_class"].items()):
        if stats["shortfall_vs_min"]:
            lines.append(
                f"- {class_name}: pool {stats['pool']} "
                f"(shortfall {stats['shortfall_vs_min']})"
            )
    if cand["shortfall"]:
        lines.append(f"- total candidates shortfall: {cand['shortfall']}")
    if lines[-1] == "":
        lines.append("- none")

    lines += [
        "",
        "## Pools",
        "",
        f"- pool_before_dedup: {cand['pool_before_dedup']}",
        f"- pool_after_dedup: {cand['pool_after_dedup']}",
        f"- pool_final: {cand['pool_final']}",
        f"- duplicates_merged: {cand['duplicates_merged']}",
        f"- id_collisions: {cand['id_collisions']}",
    ]
    prefill_counts: dict[str, int] = defaultdict(int)
    for item in items:
        if item["kind"] == "exhaustive_tile":
            prefill_counts[item.get("prefill") or "classical_v1"] += 1
    lines += ["", "## Prefill mode (exhaustive tiles)", ""]
    for mode in sorted(prefill_counts):
        lines.append(f"- {mode}: {prefill_counts[mode]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
