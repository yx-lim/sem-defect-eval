"""Tests for sem.qc.review.sampler (synthetic fixtures only)."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sem.qc.review.sampler import (
    _bbox_int,
    _iou_cached,
    _polygon_iou,
    _rasterize,
    build_review_set,
)
from sem.qc.schema import read_jsonl
from conftest import (
    CLASSES,
    H,
    _instance,
    REVIEW_CONFIG,
    STEMS,
    W,
    write_manifest,
    write_method_preds,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build(synthetic):
    return build_review_set(
        manifest_path=synthetic["manifest"],
        preds_root=synthetic["preds"],
        out_dir=synthetic["out"],
        review_config=synthetic["config"],
        manifest_sha256=_sha(synthetic["manifest"]),
    )


def _items(synthetic):
    return read_jsonl(synthetic["out"] / "items.jsonl")


def test_counts_and_kinds(synthetic):
    _build(synthetic)
    items = _items(synthetic)
    exhaustive = [i for i in items if i["kind"] == "exhaustive_tile"]
    candidates = [i for i in items if i["kind"] == "candidate"]
    assert len(exhaustive) == 25
    assert len(candidates) == 60
    by_method = lambda its: {
        m: sum(1 for i in its if i["sampling"]["method"] == m)
        for m in ("random", "uncertainty")
    }
    assert by_method(exhaustive) == {"random": 18, "uncertainty": 7}
    assert by_method(candidates) == {"random": 42, "uncertainty": 18}


def test_only_val_test_stems(synthetic):
    _build(synthetic)
    items = _items(synthetic)
    splits = {i["stem"]: i["split"] for i in items}
    assert set(splits.values()) <= {"val", "test"}
    train_stems = {s for s, _b, sp in STEMS if sp == "train"}
    assert not (set(splits) & train_stems)


def test_random_weights_sum_to_population(synthetic):
    _build(synthetic)
    items = _items(synthetic)
    weights = {}
    for item in items:
        if item["sampling"]["method"] != "random":
            continue
        weights.setdefault(item["sampling"]["stratum"], []).append(
            item["sampling"]["weight"]
        )
    summary = json.loads((synthetic["out"] / "summary.json").read_text())
    populations = {
        **{k: v["population"] for k, v in summary["exhaustive"]["strata"].items()},
        **{k: v["population"] for k, v in summary["candidates"]["strata"].items()},
    }
    for stratum, stratum_weights in weights.items():
        assert sum(stratum_weights) == pytest.approx(populations[stratum])


def test_at_least_min_per_class(synthetic):
    _build(synthetic)
    items = _items(synthetic)
    counts = {}
    for item in items:
        if item["kind"] != "candidate" or item["sampling"]["method"] != "random":
            continue
        cls = item["proposal"]["class_name"]
        counts[cls] = counts.get(cls, 0) + 1
    for cls in CLASSES:
        assert counts.get(cls, 0) >= 6


def test_shortfall_reported(tmp_path):
    stems = [("a", "Batch_1", "val"), ("b", "Batch_1", "test")]
    manifest = write_manifest(tmp_path / "manifest.csv", stems=stems)
    preds = tmp_path / "preds"
    write_method_preds(preds, "classical_v1", stems, method_idx=0)
    # delete every pore instance so the pore class pool < 6
    inst_path = preds / "classical_v1" / "a_instances.json"
    inst_path.write_text(json.dumps([]))
    inst_path = preds / "classical_v1" / "b_instances.json"
    data = [i for i in json.loads(inst_path.read_text()) if i["class_name"] != "pore"]
    inst_path.write_text(json.dumps(data))
    out = tmp_path / "review"
    build_review_set(manifest, preds, out, dict(REVIEW_CONFIG), manifest_sha256=_sha(manifest))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["candidates"]["per_class"]["pore"]["shortfall_vs_min"] > 0


def test_determinism(tmp_path):
    results = []
    for run in range(2):
        out = tmp_path / f"review{run}"
        manifest = write_manifest(tmp_path / f"manifest{run}.csv")
        preds = tmp_path / f"preds{run}"
        write_method_preds(preds, "classical_v1", STEMS, method_idx=0)
        write_method_preds(
            preds, "unet_pseudo_v1", STEMS, method_idx=1, duplicate_of="x"
        )
        build_review_set(manifest, preds, out, dict(REVIEW_CONFIG), manifest_sha256=_sha(manifest))
        results.append(
            (
                (out / "items.jsonl").read_bytes(),
                (out / "summary.json").read_bytes(),
            )
        )
    # stems/methods identical -> identical bytes
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]


def test_no_overlapping_exhaustive_tiles(synthetic):
    _build(synthetic)
    items = _items(synthetic)
    rects = {}
    for item in items:
        if item["kind"] != "exhaustive_tile":
            continue
        t = item["tile"]
        rect = (t["x0"], t["y0"], t["x0"] + t["w"], t["y0"] + t["h"])
        for other in rects.get(item["stem"], []):
            ox0, oy0, ox1, oy1 = other
            assert not (
                rect[0] < ox1 and ox0 < rect[2] and rect[1] < oy1 and oy0 < rect[3]
            ), f"overlap on {item['stem']}"
        rects.setdefault(item["stem"], []).append(rect)


def test_no_candidate_duplicates_over_iou(synthetic):
    _build(synthetic)
    items = _items(synthetic)
    for item in items:
        if item["kind"] == "candidate":
            # dedup merged twins are recorded, not sampled as separate items
            assert "duplicates" in item["proposal"]
    ids = [i["item_id"] for i in items]
    assert len(ids) == len(set(ids))


def test_tiles_min_valid(synthetic):
    _build(synthetic)
    items = _items(synthetic)
    prefill = synthetic["config"]["prefill_method"]
    for item in items:
        if item["kind"] != "exhaustive_tile":
            continue
        sem = np.asarray(
            Image.open(
                synthetic["preds"] / prefill / f"{item['stem']}_semantic.png"
            )
        )
        t = item["tile"]
        window = sem[t["y0"] : t["y0"] + t["h"], t["x0"] : t["x0"] + t["w"]]
        assert np.mean(window != 255) >= 0.9


def test_prefill_png_matches_semantic_crop(synthetic):
    _build(synthetic)
    items = _items(synthetic)
    prefill = synthetic["config"]["prefill_method"]
    for item in items:
        if item["kind"] != "exhaustive_tile":
            continue
        t = item["tile"]
        sem = np.asarray(
            Image.open(
                synthetic["preds"] / prefill / f"{item['stem']}_semantic.png"
            )
        )
        expected = sem[t["y0"] : t["y0"] + t["h"], t["x0"] : t["x0"] + t["w"]]
        actual = np.asarray(
            Image.open(synthetic["out"] / item["proposal"]["semantic_png"])
        )
        assert np.array_equal(expected, actual)


def test_works_with_one_and_three_methods(tmp_path):
    for n_methods in (1, 3):
        manifest = write_manifest(tmp_path / f"m{n_methods}.csv")
        preds = tmp_path / f"preds{n_methods}"
        write_method_preds(preds, "classical_v1", STEMS, method_idx=0)
        for m in range(1, n_methods):
            write_method_preds(preds, f"m{m}", STEMS, method_idx=m)
        out = tmp_path / f"review{n_methods}"
        build_review_set(manifest, preds, out, dict(REVIEW_CONFIG), manifest_sha256=_sha(manifest))
        items = read_jsonl(out / "items.jsonl")
        assert items


def test_dedup_performance():
    from sem.qc.review.sampler import _dedup_instances

    rng = np.random.default_rng(0)
    instances = []
    for i in range(20000):
        x0, y0 = rng.integers(0, 6000, size=2)
        instances.append(
            {
                "class_name": "bright_particle",
                "subtype": None,
                "bbox": [int(x0), int(y0), int(x0) + 12, int(y0) + 10],
                "polygon": [
                    [int(x0), int(y0)],
                    [int(x0) + 12, int(y0)],
                    [int(x0) + 12, int(y0) + 10],
                    [int(x0), int(y0) + 10],
                ],
                "score": float(rng.random()),
                "source": "classical_v1",
                "item_id": f"i{i}",
            }
        )
    start = time.monotonic()
    kept, merged = _dedup_instances(instances, 0.5)
    assert time.monotonic() - start < 5.0
    assert kept and merged == 20000 - len(kept)


def test_refuse_overwrite_with_decisions(synthetic):
    _build(synthetic)
    decisions = synthetic["out"] / "decisions.jsonl"
    decisions.write_text('{"item_id": "x", "human": {"status": "accepted"}}\n')
    with pytest.raises(RuntimeError, match="Refusing"):
        _build(synthetic)
    _build_forced = build_review_set(
        manifest_path=synthetic["manifest"],
        preds_root=synthetic["preds"],
        out_dir=synthetic["out"],
        review_config=synthetic["config"],
        manifest_sha256=_sha(synthetic["manifest"]),
        force=True,
    )
    assert _build_forced["items"]


def test_frozen_manifest_hash_mismatch(synthetic):
    with pytest.raises(AssertionError, match="hash mismatch"):
        build_review_set(
            manifest_path=synthetic["manifest"],
            preds_root=synthetic["preds"],
            out_dir=synthetic["out"],
            review_config=synthetic["config"],
            manifest_sha256="0" * 64,
        )


def test_candidate_tile_scan_streak():
    from sem.qc.review.sampler import _candidate_tile

    inst = {"bbox": [0, 136, 7000, 137]}
    tile = _candidate_tile(inst, 2080, 7000, 128, 1024)
    assert tile["w"] == 1024
    assert tile["h"] == 128
    assert tile["x0"] + tile["w"] / 2 == pytest.approx(3500)
    assert tile["x0"] == 2988
    assert tile["y0"] <= 136 and tile["y0"] + tile["h"] >= 137
    assert tile["truncated"] is True


def test_candidate_tile_small_blob():
    from sem.qc.review.sampler import _candidate_tile

    tile = _candidate_tile({"bbox": [100, 100, 110, 110]}, 2080, 7000, 128, 1024)
    assert tile["w"] == 128 and tile["h"] == 128
    assert tile["truncated"] is False


def _streak_instance(y, source="classical_v1"):
    return {
        "class_name": "artifact",
        "subtype": "scan_streak",
        "bbox": [0, y, 7000, y + 1],
        "polygon": [],
        "score": 0.5,
        "source": source,
    }


def test_distinct_streaks_get_distinct_tiles(tmp_path):
    from sem.qc.review.sampler import _candidate_tile

    t1 = _candidate_tile(_streak_instance(136), 2080, 7000, 128, 1024)
    t2 = _candidate_tile(_streak_instance(600), 2080, 7000, 128, 1024)
    assert (t1["x0"], t1["y0"], t1["w"], t1["h"]) != (
        t2["x0"], t2["y0"], t2["w"], t2["h"]
    )


def test_same_tile_collision_merged_not_dropped(tmp_path):
    # two instances whose bboxes produce the identical tile -> one item,
    # the second recorded under duplicates with reason same_review_tile
    import json as _json
    from conftest import write_manifest, write_method_preds, REVIEW_CONFIG
    from sem.qc.review.sampler import build_review_set, _dedup_instances
    from sem.qc.schema import make_item_id

    stems = [("a", "Batch_1", "val"), ("b", "Batch_1", "test")]
    manifest = write_manifest(tmp_path / "manifest.csv", stems=stems)
    preds = tmp_path / "preds"
    write_method_preds(preds, "classical_v1", stems, method_idx=0)
    inst_path = preds / "classical_v1" / "a_instances.json"
    # same bbox -> identical tile & item_id, but disjoint polygons -> IoU 0,
    # so dedup keeps both and the item_id collision path fires
    i1 = {"class_name": "pore", "subtype": None,
          "bbox": [500, 500, 620, 615],
          "polygon": [[500, 500], [510, 500], [505, 510]],
          "score": 0.9, "source": "classical_v1"}
    i2 = {"class_name": "pore", "subtype": None,
          "bbox": [500, 500, 620, 615],
          "polygon": [[610, 605], [620, 605], [615, 615]],
          "score": 0.8, "source": "classical_v1"}
    inst_path.write_text(_json.dumps([i1, i2]))
    inst_path = preds / "classical_v1" / "b_instances.json"
    inst_path.write_text(_json.dumps([]))
    out = tmp_path / "review"
    cfg = dict(REVIEW_CONFIG)
    r = build_review_set(manifest, preds, out, cfg, manifest_sha256=_sha(manifest))
    items = r["items"]
    cand = [i for i in items if i["kind"] == "candidate"]
    assert len(cand) == 1
    dups = cand[0]["proposal"]["duplicates"]
    assert any(d.get("reason") == "same_review_tile" for d in dups)
    s = r["summary"]["candidates"]
    assert s["pool_before_dedup"] == s["pool_final"] + s["duplicates_merged"] + s["id_collisions"]
    assert s["id_collisions"] == 1


def test_largest_remainder_real_strata():
    from sem.qc.review.sampler import _largest_remainder

    demand = {
        "test|Batch_1": 52, "test|Batch_2": 52, "test|Batch_3": 195,
        "val|Batch_1": 52, "val|Batch_2": 52, "val|Batch_3": 208,
    }
    assert _largest_remainder(18, demand) == {
        "test|Batch_1": 2, "test|Batch_2": 2, "test|Batch_3": 6,
        "val|Batch_1": 1, "val|Batch_2": 1, "val|Batch_3": 6,
    }


def test_largest_remainder_properties():
    from sem.qc.review.sampler import _largest_remainder

    cases = [
        ({"a": 1, "b": 1, "c": 100}, 10),
        ({"a": 5, "b": 5, "c": 5}, 7),
        ({"x": 3, "y": 30, "z": 60}, 20),
        ({"a": 2, "b": 2}, 3),
    ]
    for demand, budget in cases:
        total = sum(demand.values())
        alloc = _largest_remainder(budget, demand)
        assert sum(alloc.values()) == min(budget, total)
        for k, n in demand.items():
            assert 0 <= alloc[k] <= n
            if total:
                quota = budget * n / total
                if alloc[k] < n:
                    assert abs(alloc[k] - quota) < 1 or alloc[k] == int(quota) + 1


def test_largest_remainder_budget_exceeds_demand():
    from sem.qc.review.sampler import _largest_remainder

    demand = {"a": 3, "b": 2, "c": 0}
    assert _largest_remainder(10, demand) == {"a": 3, "b": 2, "c": 0}


# ---- semantic connected-component candidates -----------------------------

def _write_cc_preds(preds: Path, stem: str, components, uncertainty=128):
    """Write a semantic png with drawn components + flat uncertainty map."""
    from conftest import H, W, BORDER

    method_dir = preds / "classical_v1"
    method_dir.mkdir(parents=True, exist_ok=True)
    sem = np.zeros((H, W), dtype=np.uint8)
    sem[:BORDER, :] = 255
    sem[-BORDER:, :] = 255
    sem[:, :BORDER] = 255
    sem[:, -BORDER:] = 255
    for class_id, (y0, x0, h, w) in components:
        sem[y0 : y0 + h, x0 : x0 + w] = class_id
    Image.fromarray(sem, mode="L").save(method_dir / f"{stem}_semantic.png")
    u = np.full((H, W), uncertainty, dtype=np.uint8)
    Image.fromarray(u, mode="L").save(method_dir / f"{stem}_uncertainty.png")
    (method_dir / f"{stem}_instances.json").write_text("[]")


def test_cc_candidates_extracted(tmp_path):
    stems = [("a", "Batch_1", "val"), ("b", "Batch_1", "test")]
    manifest = write_manifest(tmp_path / "m.csv", stems=stems)
    preds = tmp_path / "preds"
    # class ids: pore=3, crack=5
    comps = [
        (3, (100, 100, 10, 10)),   # pore 100px keep
        (3, (300, 100, 10, 10)),   # pore 100px keep
        (3, (500, 100, 10, 10)),   # pore 100px keep
        (3, (700, 100, 5, 6)),     # pore 30px dropped
        (5, (100, 600, 1, 40)),    # crack 40px keep
        (5, (300, 600, 2, 20)),    # crack 40px keep
    ]
    for stem, _b, _s in stems:
        _write_cc_preds(preds, stem, comps)
    cfg = dict(REVIEW_CONFIG)
    cfg["cc_min_area_px"] = {
        "pore": 64, "bright_particle": 64,
        "crack_intraparticle": 40, "interparticle_gap": 40,
    }
    out = tmp_path / "review"
    r = build_review_set(manifest, preds, out, cfg, manifest_sha256=_sha(manifest))
    summary = r["summary"]
    strata = summary["candidates"]["strata"]
    assert strata["pore|classical_v1:semantic_cc"]["population"] == 6
    assert strata["crack_intraparticle|classical_v1:semantic_cc"]["population"] == 4
    items = r["items"]
    cc = [i for i in items if i["kind"] == "candidate"
          and i["proposal"]["source"].endswith(":semantic_cc")]
    assert cc, "expected cc candidates"
    for item in cc:
        assert item["proposal"]["score"] is None
        assert item["proposal"]["method"] == "classical_v1"
        assert item["proposal"]["area_px"] >= 40
        assert item["proposal"]["uncertainty"] == pytest.approx(128 / 255)
        assert item["sampling"]["weight"] == pytest.approx(
            strata[item["sampling"]["stratum"]]["population"]
            / summary["candidates"]["strata"][item["sampling"]["stratum"]]["n_random"]
        )
    bboxes = {tuple(i["proposal"]["bbox"]) for i in cc}
    assert (100, 100, 110, 110) in bboxes  # exact component bbox


def test_uncertainty_picks_round_robin(tmp_path):
    from collections import Counter

    stems = [("a", "Batch_1", "val"), ("b", "Batch_1", "test")]
    manifest = write_manifest(tmp_path / "m.csv", stems=stems)
    preds = tmp_path / "preds"
    comps = []
    for i in range(20):  # 20 pores 64px
        comps.append((3, (20 + i * 30, 20, 10, 10)))
    for i in range(20):  # 20 cracks 40px
        comps.append((5, (20 + i * 30, 400, 1, 40)))
    for i in range(20):  # 20 bright 64px
        comps.append((2, (20 + i * 30, 700, 10, 10)))
    for stem, _b, _s in stems:
        _write_cc_preds(preds, stem, comps)
    cfg = dict(REVIEW_CONFIG)
    cfg["cc_min_area_px"] = {
        "pore": 64, "bright_particle": 64,
        "crack_intraparticle": 40, "interparticle_gap": 40,
    }
    out = tmp_path / "review"
    r = build_review_set(manifest, preds, out, cfg, manifest_sha256=_sha(manifest))
    per_class = r["summary"]["candidates"]["per_class"]
    unc = {c: per_class[c]["n_uncertainty"] for c in ("pore", "crack_intraparticle", "bright_particle")}
    assert sum(v > 0 for v in unc.values()) >= 2
    assert max(unc.values()) - min(unc.values()) <= 1
    # every class hit its 6-item floor via random picks
    for c in ("pore", "crack_intraparticle", "bright_particle"):
        assert per_class[c]["n_random"] >= 6


def test_blank_start_tiles(synthetic):
    cfg = dict(synthetic["config"])
    cfg["blank_start_tiles"] = 2
    build_review_set(
        manifest_path=synthetic["manifest"],
        preds_root=synthetic["preds"],
        out_dir=synthetic["out"],
        review_config=cfg,
        manifest_sha256=_sha(synthetic["manifest"]),
    )
    items = _items(synthetic)
    tiles = [i for i in items if i["kind"] == "exhaustive_tile"]
    blanks = [i for i in tiles if i["prefill"] == "blank"]
    assert len(blanks) == 2
    for item in blanks:
        assert item["sampling"]["method"] == "random"
        assert item["proposal"]["semantic_png"] is None
        assert item["proposal"]["instances"] == []
        ref = item["proposal"]["reference_semantic_png"]
        assert ref and (synthetic["out"] / ref).exists()
    assert all(i["prefill"] == "blank" for i in blanks)
    assert all(
        i["sampling"]["method"] == "random" or i["prefill"] != "blank"
        for i in tiles
    )
    non_blank = [i for i in tiles if i["prefill"] != "blank"]
    assert all(i["prefill"] == "classical_v1" for i in non_blank)

    # deterministic blank selection
    out2 = synthetic["out"].parent / "review2"
    build_review_set(
        manifest_path=synthetic["manifest"],
        preds_root=synthetic["preds"],
        out_dir=out2,
        review_config=cfg,
        manifest_sha256=_sha(synthetic["manifest"]),
    )
    items2 = read_jsonl(out2 / "items.jsonl")
    assert [
        i["item_id"] for i in items2 if i.get("prefill") == "blank"
    ] == [i["item_id"] for i in blanks]


def test_no_blank_when_zero(synthetic):
    _build(synthetic)
    items = _items(synthetic)
    tiles = [i for i in items if i["kind"] == "exhaustive_tile"]
    assert all(i["prefill"] == "classical_v1" for i in tiles)


def test_cached_iou_matches_polygon_iou():
    import random

    rng = random.Random(1234)
    pairs = []
    for _ in range(30):
        pts_a = [
            (rng.uniform(0, 200), rng.uniform(0, 200)) for _ in range(rng.randint(3, 8))
        ]
        # hull-sort around centroid so the polygon is simple
        cx = sum(p[0] for p in pts_a) / len(pts_a)
        cy = sum(p[1] for p in pts_a) / len(pts_a)
        pts_a.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        shift = rng.uniform(0, 150)
        pts_b = [(x + shift, y + rng.uniform(-40, 40)) for x, y in pts_a]
        pairs.append((pts_a, pts_b))
    # off-by-fraction coordinates
    pairs.append(
        (
            [(10.4, 10.9), (50.999, 10.1), (50.5, 60.7), (10.2, 60.3)],
            [(20.3, 15.6), (60.999, 14.9), (59.4, 70.1), (19.8, 69.999)],
        )
    )

    def _inst(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return {
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "polygon": [[float(x), float(y)] for x, y in pts],
        }

    for pts_a, pts_b in pairs:
        a, b = _inst(pts_a), _inst(pts_b)
        expected = _polygon_iou(a, b)
        entry_a = (_bbox_int(a), _rasterize(a, _bbox_int(a)), 0)
        entry_b = (_bbox_int(b), _rasterize(b, _bbox_int(b)), 0)
        entry_a = (entry_a[0], entry_a[1], int(entry_a[1].sum()))
        entry_b = (entry_b[0], entry_b[1], int(entry_b[1].sum()))
        assert _iou_cached(entry_a, entry_b) == pytest.approx(expected)


def _write_method_semantic(preds: Path, method: str, stem: str,
                           components, uncertainty=None):
    from conftest import H, W, BORDER

    method_dir = preds / method
    method_dir.mkdir(parents=True, exist_ok=True)
    sem = np.zeros((H, W), dtype=np.uint8)
    sem[:BORDER, :] = 255
    sem[-BORDER:, :] = 255
    sem[:, :BORDER] = 255
    sem[:, -BORDER:] = 255
    for class_id, (y0, x0, h, w) in components:
        sem[y0 : y0 + h, x0 : x0 + w] = class_id
    Image.fromarray(sem, mode="L").save(method_dir / f"{stem}_semantic.png")
    if uncertainty is not None:
        u = np.full((H, W), uncertainty, dtype=np.uint8)
        Image.fromarray(u, mode="L").save(
            method_dir / f"{stem}_uncertainty.png"
        )


def test_cc_exclude_methods(tmp_path):
    stems = [("a", "Batch_1", "val"), ("b", "Batch_1", "test")]
    manifest = write_manifest(tmp_path / "m.csv", stems=stems)
    preds = tmp_path / "preds"
    comps = [(3, (100, 100, 10, 10))]  # one 100px pore component
    for stem, _b, _s in stems:
        _write_method_semantic(preds, "classical_v1", stem, comps)
        (preds / "classical_v1" / f"{stem}_instances.json").write_text("[]")
        _write_method_semantic(preds, "excluded_m", stem, comps)
        (preds / "excluded_m" / f"{stem}_instances.json").write_text(
            json.dumps(
                [
                    _instance("pore", 0, 0, 0, "excluded_m:orig"),
                    _instance("unmapped", 0, 0, 1, "excluded_m"),
                ]
            )
        )
    cfg = dict(REVIEW_CONFIG)
    cfg["cc_min_area_px"] = {
        "pore": 64, "bright_particle": 64,
        "crack_intraparticle": 40, "interparticle_gap": 40,
    }
    cfg["cc_exclude_methods"] = ["excluded_m"]
    out = tmp_path / "review"
    r = build_review_set(manifest, preds, out, cfg, manifest_sha256=_sha(manifest))
    summary = r["summary"]
    assert summary["candidates"]["cc_methods"] == ["classical_v1"]
    strata = summary["candidates"]["strata"]
    assert not any(
        s.startswith("pore|excluded_m:semantic_cc") for s in strata
    )
    pool_sources = summary["candidates"]["pool_per_source"]
    assert pool_sources.get("classical_v1:semantic_cc", 0) > 0
    # its instances.json records still enter the pool
    assert pool_sources.get("excluded_m", 0) == 2
    # 'unmapped' filtered by candidate_classes
    assert not any(
        i["kind"] == "candidate" and i["proposal"]["class_name"] == "unmapped"
        for i in r["items"]
    )


def test_blank_start_five(synthetic):
    cfg = dict(synthetic["config"])
    cfg["blank_start_tiles"] = 5
    build_review_set(
        manifest_path=synthetic["manifest"],
        preds_root=synthetic["preds"],
        out_dir=synthetic["out"],
        review_config=cfg,
        manifest_sha256=_sha(synthetic["manifest"]),
    )
    items = _items(synthetic)
    blanks = [i for i in items if i.get("prefill") == "blank"]
    assert len(blanks) == 5
    assert all(i["kind"] == "exhaustive_tile" for i in blanks)
    assert all(i["sampling"]["method"] == "random" for i in blanks)
    out2 = synthetic["out"].parent / "review2"
    build_review_set(
        manifest_path=synthetic["manifest"],
        preds_root=synthetic["preds"],
        out_dir=out2,
        review_config=cfg,
        manifest_sha256=_sha(synthetic["manifest"]),
    )
    ids2 = [i["item_id"] for i in read_jsonl(out2 / "items.jsonl")
            if i.get("prefill") == "blank"]
    assert ids2 == [i["item_id"] for i in blanks]


def test_allocate_random_min_per_stratum():
    from sem.qc.review.sampler import _allocate_random_candidates as alloc_fn

    alloc, warn = alloc_fn(
        {"pore|m1": 2, "pore|m2": 5000, "crack_intraparticle|m1": 300},
        n_random=20, min_per_class=6, min_per_stratum=3,
    )
    assert alloc["pore|m1"] == 2
    assert all(alloc[s] >= min(3, n) for s, n in
               {"pore|m1": 2, "pore|m2": 5000, "crack_intraparticle|m1": 300}.items())
    assert sum(alloc.values()) == 20
    assert warn is None


def test_allocate_random_oversubscribed():
    from sem.qc.review.sampler import _allocate_random_candidates as alloc_fn

    sizes = {f"c|s{i}": 100 for i in range(5)}
    alloc, warn = alloc_fn(sizes, n_random=10, min_per_class=1,
                           min_per_stratum=3)
    assert all(v >= 2 for v in alloc.values())
    assert sum(alloc.values()) == 10
    assert warn["applied_min"] == 2
    assert warn["requested_total"] == 15

    sizes7 = {f"c|s{i}": 100 for i in range(7)}
    alloc7, warn7 = alloc_fn(sizes7, n_random=10, min_per_class=1,
                             min_per_stratum=3)
    assert min(alloc7.values()) == 1
    assert sum(alloc7.values()) == 10
    assert warn7["applied_min"] == 1
    assert warn7["requested_total"] == 21


def test_uncertainty_percentiles_and_ranking():
    from sem.qc.review.sampler import _uncertainty_percentiles

    pool = [
        {"source": "m", "uncertainty": 0.5},
        {"source": "m", "uncertainty": 0.5},
        {"source": "m", "uncertainty": 1.0},
        {"source": "m", "uncertainty": None},
    ]
    _uncertainty_percentiles(pool)
    assert [i["uncertainty_pct"] for i in pool] == [0.5, 0.5, 1.0, None]


def _build_two_source_class(tmp_path):
    """Two methods each emitting one class with opposite uncertainty scales."""
    stems = [("a", "Batch_1", "val"), ("b", "Batch_1", "test")]
    manifest = write_manifest(tmp_path / "m.csv", stems=stems)
    preds = tmp_path / "preds"
    for s_idx, (stem, _b, _s) in enumerate(stems):
        insts_hi = [
            dict(_instance("pore", s_idx, 0, j, "m_hi"),
                 score=0.05 + j * 0.001)
            for j in range(8)
        ]
        insts_lo = [
            dict(_instance("pore", s_idx, 1, j, "m_lo"),
                 score=0.95 - j * 0.001)
            for j in range(8)
        ]
        (preds / "m_hi").mkdir(parents=True, exist_ok=True)
        (preds / "m_lo").mkdir(parents=True, exist_ok=True)
        (preds / "m_hi" / f"{stem}_instances.json").write_text(
            json.dumps(insts_hi))
        (preds / "m_lo" / f"{stem}_instances.json").write_text(
            json.dumps(insts_lo))
        _write_method_semantic(preds, "classical_v1", stem, [])
        (preds / "classical_v1" / f"{stem}_instances.json").write_text("[]")
    cfg = dict(REVIEW_CONFIG)
    cfg["n_candidates"] = 16
    out = tmp_path / "review"
    return build_review_set(
        manifest, preds, out, cfg, manifest_sha256=_sha(manifest))


def test_uncertainty_picks_span_sources(tmp_path):
    r = _build_two_source_class(tmp_path)
    unc = [
        i for i in r["items"]
        if i["kind"] == "candidate"
        and i["sampling"]["method"] == "uncertainty"
        and i["proposal"]["class_name"] == "pore"
    ]
    sources = {i["proposal"]["source"] for i in unc}
    assert unc, "expected pore uncertainty picks"
    assert len(sources) >= 2, sources


def test_min_random_per_stratum_end_to_end(tmp_path):
    stems = [("a", "Batch_1", "val"), ("b", "Batch_1", "test")]
    manifest = write_manifest(tmp_path / "m.csv", stems=stems)
    preds = tmp_path / "preds"
    for s_idx, (stem, _b, _s) in enumerate(stems):
        (preds / "m_small").mkdir(parents=True, exist_ok=True)
        (preds / "m_big").mkdir(parents=True, exist_ok=True)
        # pool of exactly 2 in m_small
        (preds / "m_small" / f"{stem}_instances.json").write_text(
            json.dumps([_instance("pore", s_idx, 0, 0, "m_small")]))
        (preds / "m_big" / f"{stem}_instances.json").write_text(
            json.dumps(
                [_instance("pore", s_idx, 1, j, "m_big") for j in range(30)]
            )
        )
        _write_method_semantic(preds, "classical_v1", stem, [])
        (preds / "classical_v1" / f"{stem}_instances.json").write_text("[]")
    cfg = dict(REVIEW_CONFIG)
    cfg["candidate_min_random_per_stratum"] = 3
    out = tmp_path / "review"
    r = build_review_set(manifest, preds, out, cfg, manifest_sha256=_sha(manifest))
    strata = r["summary"]["candidates"]["strata"]
    small = [
        i for i in r["items"]
        if i["kind"] == "candidate"
        and i["sampling"]["stratum"] == "pore|m_small"
        and i["sampling"]["method"] == "random"
    ]
    assert len(small) == 2
    assert all(i["sampling"]["weight"] == 1.0 for i in small)
    for item in r["items"]:
        if item["kind"] == "candidate" and item["sampling"]["method"] == "random":
            st = strata[item["sampling"]["stratum"]]
            assert item["sampling"]["weight"] == pytest.approx(
                st["population"] / st["n_random"])
