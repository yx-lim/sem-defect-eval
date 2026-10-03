import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sem.qc.config import load_config
from sem.qc.eval.compare import compare
from sem.qc.eval.evaluate import EvalSettings, evaluate
from sem.qc.eval.gallery import render_galleries
from sem.qc.eval.gt import GroundTruthError, load_manifest
from sem.qc.eval.report import write_outputs
from sem.qc.eval.synthetic import REVIEWER, make_study
from sem.qc.schema import make_item_id, read_jsonl, write_jsonl

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    paths = make_study(root, {"perfect": 0.0, "degraded": 1.0},
                       n_stems={"test": 3, "val": 1, "train": 1},
                       shape=(256, 256), tile=128, seed=3, candidates_per_stem=8)
    over = paths["work_root"] / "preds" / "overvoid"
    over.mkdir(parents=True)
    for png in (paths["work_root"] / "preds" / "perfect").glob("*_semantic.png"):
        pred = np.array(Image.open(png))
        block = pred[40:80, 40:80]
        block[block == 0] = 3
        Image.fromarray(pred).save(over / png.name)
        name = png.name.replace("_semantic.png", "_instances.json")
        (over / name).write_text((png.parent / name).read_text())
    return paths


def settings(paths, method, **kw):
    cfg = load_config()
    base = dict(method=method, work_root=paths["work_root"], data_root=paths["data_root"],
                manifest_path=paths["manifest"], n_boot=50, split="test", stratum="all")
    base.update(kw)
    return EvalSettings.from_config(cfg, **base)


def test_perfect_prediction_scores_one_everywhere(study):
    res = evaluate(settings(study, "perfect"))
    assert res["counts"]["n_tiles"] == 12 and res["counts"]["n_stems"] == 3
    scored = [r for r in res["pixel"] if r["tp"] + r["fn"] > 0]
    assert scored
    for row in scored:
        for m in ("precision", "recall", "f1", "iou"):
            assert row[m]["value"] == pytest.approx(1.0), (row["class"], m)
            assert row[m]["ci_low"] == pytest.approx(1.0)
        assert row["fp"] == 0 and row["fn"] == 0
    objs = [r for r in res["object"] if r["n_gt_objects"] > 0]
    assert {r["class"] for r in objs} >= {"bright_particle", "pore", "agglomerate"}
    for row in objs:
        assert row["fp"] == 0 and row["fn"] == 0
        assert row["precision"]["value"] == row["recall"]["value"] == 1.0
    for row in res["kpi"]:
        if row["n_pairs"]:
            assert row["bias"]["value"] == pytest.approx(0.0, abs=1e-12)
            assert row["mae"]["value"] == pytest.approx(0.0, abs=1e-12)
    cm = np.array(res["confusion"]["raw"])
    assert cm.sum() == cm.trace()
    assert res["counts"]["gt_pixels_predicted_ignore"] == 0


def test_degraded_is_worse_and_reports_counts(study):
    res = evaluate(settings(study, "degraded", stratum="random"))
    assert res["counts"]["n_tiles"] == 9  # last tile per stem is the uncertainty stratum
    pore = next(r for r in res["pixel"] if r["class"] == "pore")
    assert pore["iou"]["value"] < 1.0
    assert pore["iou"]["ci_low"] <= pore["iou"]["value"] <= pore["iou"]["ci_high"]
    for row in res["pixel"] + res["object"]:
        assert "n_tiles" in row and "n_stems" in row and "insufficient_n" in row
    assert next(r for r in res["object"] if r["class"] == "interparticle_gap")["match_iou"] == 0.3
    assert next(r for r in res["object"] if r["class"] == "pore")["match_iou"] == 0.5


def test_kpi_error_sign_overestimated_voids(study):
    res = evaluate(settings(study, "overvoid"))
    for level in ("tile", "stem"):
        row = next(r for r in res["kpi"] if r["level"] == level and r["kpi"] == "void_fraction")
        assert row["bias"]["value"] > 0
        assert row["mae"]["value"] == pytest.approx(row["bias"]["value"])
        assert row["rel_error"]["value"] > 0
        assert row["pred_mean"] > row["gt_mean"]
    assert len(res["kpi_per_stem"]) == 3


def test_candidate_precision_no_recall_and_uncertain_separate(study):
    res = evaluate(settings(study, "degraded"))
    rows = res["candidate"]
    assert rows
    for row in rows:
        assert not any("recall" in k for k in row)
        assert row["weighted"] == (row["sampling"] == "random")
        if row["n_positive"] + row["n_negative"]:
            assert row["wilson_low"] <= row["precision"] <= row["wilson_high"]
    items = {i["item_id"]: i for i in read_jsonl(study["review"] / "items.jsonl")}
    decs = read_jsonl(study["review"] / "decisions.jsonl")
    n_unc = sum(1 for d in decs if d["human"]["status"] == "uncertain"
                and items[d["item_id"]]["kind"] == "candidate"
                and items[d["item_id"]]["split"] == "test"
                and items[d["item_id"]]["proposal"]["source"].startswith("degraded:"))
    assert sum(r["n_uncertain"] for r in rows) == n_unc


def _append_reviewed(paths, stem, split, tmp_path):
    shutil.copytree(paths["work_root"], tmp_path / "work")
    review = tmp_path / "work" / "review"
    items = read_jsonl(paths["review"] / "items.jsonl")
    decisions = read_jsonl(paths["review"] / "decisions.jsonl")
    item_id = make_item_id(stem, "exhaustive_tile", 0, 0, 16, 16, "classical_v1")
    items.append({"item_id": item_id, "stem": stem, "batch": "Batch_1", "split": split,
                  "kind": "exhaustive_tile", "tile": {"x0": 0, "y0": 0, "w": 16, "h": 16},
                  "sampling": {"stratum": "x", "method": "random", "weight": 1.0},
                  "proposal": {"source": "classical_v1", "class_name": None,
                               "semantic_png": None},
                  "vlm_suggestion": None, "human": None})
    decisions.append({"item_id": item_id, "human": {"status": "accepted",
                                                     "reviewer_id": REVIEWER}})
    write_jsonl(review / "items.jsonl", items)
    write_jsonl(review / "decisions.jsonl", decisions)
    return {**paths, "work_root": tmp_path / "work"}


def _stem_of_split(paths, split):
    with open(paths["manifest"], newline="") as fh:
        return next(r["stem"] for r in csv.DictReader(fh) if r["split"] == split)


def test_train_stem_gt_is_rejected(study, tmp_path):
    bad = _append_reviewed(study, _stem_of_split(study, "train"), "train", tmp_path)
    with pytest.raises(GroundTruthError, match="TRAIN stem"):
        evaluate(settings(bad, "perfect"))


def test_item_split_disagreeing_with_manifest_is_rejected(study, tmp_path):
    bad = _append_reviewed(study, _stem_of_split(study, "val"), "test", tmp_path)
    with pytest.raises(GroundTruthError, match="disagrees with manifest"):
        evaluate(settings(bad, "perfect"))


def test_undecided_train_item_is_ignored_not_gt(study, tmp_path):
    ok = _append_reviewed(study, _stem_of_split(study, "train"), "train", tmp_path)
    decs = read_jsonl(ok["work_root"] / "review" / "decisions.jsonl")[:-1]
    write_jsonl(ok["work_root"] / "review" / "decisions.jsonl", decs)
    assert evaluate(settings(ok, "perfect"))["counts"]["n_tiles"] == 12


def test_manifest_hash_mismatch_fails_loudly(study):
    with pytest.raises(GroundTruthError, match="hash check failed"):
        load_manifest(study["manifest"], "0" * 64)
    with pytest.raises(GroundTruthError, match="hash check failed"):
        evaluate(settings(study, "perfect", frozen_manifest_sha256="0" * 64))
    digest = load_manifest(study["manifest"], None).sha256
    assert load_manifest(study["manifest"], digest).frozen


def test_unfrozen_manifest_is_flagged(study):
    res = evaluate(settings(study, "perfect"))
    assert not res["manifest"]["frozen"]
    assert any("NOT frozen" in w for w in res["warnings"])


def test_model_reviewer_is_never_gt(study, tmp_path):
    bad = _append_reviewed(study, _stem_of_split(study, "test"), "test", tmp_path)
    decs = read_jsonl(bad["work_root"] / "review" / "decisions.jsonl")
    decs[-1]["human"]["reviewer_id"] = "classical_v1"
    write_jsonl(bad["work_root"] / "review" / "decisions.jsonl", decs)
    with pytest.raises(GroundTruthError, match="never ground truth"):
        evaluate(settings(bad, "perfect"))


def test_reports_galleries_and_comparison(study, tmp_path):
    files = []
    for method in ("perfect", "degraded"):
        res = evaluate(settings(study, method, stratum="random"))
        out = tmp_path / method
        index = render_galleries(res["_errors"], res["_tile_maps"], study["data_root"],
                                 out / "galleries", top_k=3)
        report = write_outputs(res, out, index)
        assert "insufficient n" in report.read_text() or "| " in report.read_text()
        metrics = json.loads((out / "metrics.json").read_text())
        assert "_errors" not in metrics
        files.append(out / "metrics.json")
        if method == "perfect":
            assert index == []
        else:
            assert index and all((out / "galleries" / r["file"]).is_file() for r in index)
            assert all(r["rank"] <= 3 and r["kind"] in ("FP", "FN") for r in index)
            assert all(sum(1 for r in index if r["file"] == f) <= 3
                       for f in {r["file"] for r in index})
    md, csv_path = compare(files, tmp_path / "cmp")
    text = md.read_text()
    assert "perfect" in text and "degraded" in text
    rows = list(csv.DictReader(csv_path.open()))
    assert {r["method"] for r in rows} == {"perfect", "degraded"}
    other = evaluate(settings(study, "degraded", stratum="uncertainty"))
    write_outputs(other, tmp_path / "unc")
    with pytest.raises(ValueError, match="different split/stratum"):
        compare(files + [tmp_path / "unc" / "metrics.json"], tmp_path / "cmp2")


def test_cli_end_to_end(study, tmp_path):
    common = ["--work-root", str(study["work_root"]), "--data-root", str(study["data_root"]),
              "--manifest", str(study["manifest"]), "--n-boot", "20"]
    out = tmp_path / "cli"
    subprocess.run([sys.executable, str(REPO / "scripts/evaluate.py"), "--method", "degraded",
                    "--split", "test", "--stratum", "random", "--out", str(out), *common],
                   check=True, capture_output=True)
    assert (out / "report.md").is_file() and (out / "tables" / "pixel_metrics.csv").is_file()
    assert math.isfinite(json.loads((out / "metrics.json").read_text())["pixel"][0]["tp"])
