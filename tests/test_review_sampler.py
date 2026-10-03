"""Tests for sem.qc.review.sampler (synthetic fixtures only)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sem.qc.review.sampler import build_review_set
from sem.qc.schema import read_jsonl
from conftest import (
    CLASSES,
    H,
    REVIEW_CONFIG,
    STEMS,
    W,
    write_manifest,
    write_method_preds,
)


def _build(synthetic):
    return build_review_set(
        manifest_path=synthetic["manifest"],
        preds_root=synthetic["preds"],
        out_dir=synthetic["out"],
        review_config=synthetic["config"],
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
    build_review_set(manifest, preds, out, dict(REVIEW_CONFIG))
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
        build_review_set(manifest, preds, out, dict(REVIEW_CONFIG))
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
        build_review_set(manifest, preds, out, dict(REVIEW_CONFIG))
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
        force=True,
    )
    assert _build_forced["items"]
