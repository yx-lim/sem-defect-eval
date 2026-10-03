"""Tests for the FastAPI review backend (synthetic, no network)."""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sem.qc.review.app import create_app
from sem.qc.review.sampler import build_review_set
from sem.qc.schema import read_jsonl
from conftest import FakeLoader


@pytest.fixture
def client(synthetic, tmp_path):
    build_review_set(
        manifest_path=synthetic["manifest"],
        preds_root=synthetic["preds"],
        out_dir=synthetic["out"],
        review_config=synthetic["config"],
        manifest_sha256=hashlib.sha256(
            synthetic["manifest"].read_bytes()
        ).hexdigest(),
    )
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>")
    app = create_app(
        review_dir=synthetic["out"],
        image_loader=FakeLoader(),
        static_dir=static_dir,
    )
    return TestClient(app)


def _some_item(client, kind=None):
    items = client.get("/api/items").json()["items"]
    for item in items:
        if kind is None or item["kind"] == kind:
            return item
    raise AssertionError(f"no item of kind {kind}")


def test_index_and_no_store(client):
    resp = client.get("/")
    assert resp.status_code == 200
    resp = client.get("/api/config")
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert body["ignore_label"] == 255
    assert body["palette"]["7"] == [227, 119, 194]
    assert body["exhaustive_tile_px"] == 512


def test_items_filters(client):
    items = client.get("/api/items").json()["items"]
    assert items
    candidates = client.get("/api/items?kind=candidate").json()["items"]
    assert all(i["kind"] == "candidate" for i in candidates)
    cls = candidates[0]["class_name"]
    filtered = client.get(f"/api/items?class_name={cls}").json()["items"]
    assert filtered and all(i["class_name"] == cls for i in filtered)
    unreviewed = client.get("/api/items?status=unreviewed").json()["items"]
    assert len(unreviewed) == len(items)


def test_decision_append_latest_wins_and_survives_restart(
    client, synthetic, tmp_path
):
    item = _some_item(client, "candidate")
    r = client.post(
        f"/api/items/{item['item_id']}/decision",
        json={"status": "rejected", "reviewer_id": "ann"},
    )
    assert r.status_code == 200
    r = client.post(
        f"/api/items/{item['item_id']}/decision",
        json={"status": "accepted", "reviewer_id": "bob"},
    )
    assert r.status_code == 200
    assert client.get(f"/api/items/{item['item_id']}").json()["human"][
        "status"
    ] == "accepted"
    static_dir = tmp_path / "static"
    app2 = create_app(
        review_dir=synthetic["out"],
        image_loader=FakeLoader(),
        static_dir=static_dir,
    )
    client2 = TestClient(app2)
    assert client2.get(f"/api/items/{item['item_id']}").json()["human"][
        "status"
    ] == "accepted"


def test_reset_via_null_status(client):
    item = _some_item(client)
    client.post(
        f"/api/items/{item['item_id']}/decision",
        json={"status": "accepted", "reviewer_id": "ann"},
    )
    client.post(
        f"/api/items/{item['item_id']}/decision",
        json={"status": None, "reviewer_id": "ann"},
    )
    assert client.get(f"/api/items/{item['item_id']}").json()["human"][
        "status"
    ] is None


@pytest.mark.parametrize(
    "body",
    [
        {"status": "bogus", "reviewer_id": "ann"},
        {"status": "accepted"},
        {"status": "accepted", "reviewer_id": "ann", "class_name": "pore",
         "subtype": "curtaining"},
        {"status": "relabeled", "reviewer_id": "ann"},
        {"status": "redrawn", "reviewer_id": "ann"},
    ],
)
def test_decision_validation_422(client, body):
    item = _some_item(client)
    resp = client.post(f"/api/items/{item['item_id']}/decision", json=body)
    assert resp.status_code == 422, resp.text


def test_mask_roundtrip_and_validation(client):
    item = _some_item(client, "exhaustive_tile")
    full = client.get(f"/api/items/{item['item_id']}").json()
    w, h = full["tile"]["w"], full["tile"]["h"]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[0, 0] = 7
    mask[-1, -1] = 255
    resp = client.post(
        f"/api/items/{item['item_id']}/mask",
        content=mask.tobytes(),
        headers={"content-type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    assert resp.json()["semantic_png"] == f"masks/{item['item_id']}.png"
    got = client.get(f"/api/items/{item['item_id']}/mask.png")
    assert got.status_code == 200
    decoded = np.asarray(Image.open(io.BytesIO(got.content)))
    assert np.array_equal(decoded, mask)
    assert client.get(f"/api/items/{item['item_id']}").json()["has_mask"]

    bad_len = client.post(
        f"/api/items/{item['item_id']}/mask", content=b"\x00" * 10
    )
    assert bad_len.status_code == 422
    bad_val = np.full((h, w), 9, dtype=np.uint8)
    bad = client.post(
        f"/api/items/{item['item_id']}/mask", content=bad_val.tobytes()
    )
    assert bad.status_code == 422


def test_accepted_defaults_filled(client):
    cand = _some_item(client, "candidate")
    resp = client.post(
        f"/api/items/{cand['item_id']}/decision",
        json={"status": "accepted", "reviewer_id": "ann"},
    )
    human = resp.json()["human"]
    assert human["class_name"] == cand["class_name"]
    assert human["timestamp"].endswith("Z")

    exh = _some_item(client, "exhaustive_tile")
    resp = client.post(
        f"/api/items/{exh['item_id']}/decision",
        json={"status": "accepted", "reviewer_id": "ann"},
    )
    human = resp.json()["human"]
    assert human["semantic_png"] == f"prefill/{exh['item_id']}.png"


def test_export_only_ground_truth(client, synthetic):
    items = client.get("/api/items").json()["items"][:5]
    statuses = ["accepted", "rejected", "relabeled", "uncertain", "redrawn"]
    for item, status in zip(items, statuses):
        body = {"status": status, "reviewer_id": "ann"}
        if status == "relabeled":
            body["class_name"] = "pore"
        if status == "redrawn":
            body["polygons"] = [
                {"class_name": "pore", "subtype": None,
                 "points": [[1, 1], [5, 1], [5, 5]]}
            ]
        resp = client.post(f"/api/items/{item['item_id']}/decision", json=body)
        assert resp.status_code == 200, resp.text
    resp = client.post("/api/export")
    exported = read_jsonl(resp.json()["path"])
    got_statuses = {e["human"]["status"] for e in exported}
    assert got_statuses == {"accepted", "relabeled", "redrawn"}


def test_progress_counts(client):
    before = client.get("/api/progress").json()
    item = _some_item(client)
    client.post(
        f"/api/items/{item['item_id']}/decision",
        json={"status": "uncertain", "reviewer_id": "ann"},
    )
    after = client.get("/api/progress").json()
    assert after["reviewed"] == before["reviewed"] + 1
    assert after["by_status"]["uncertain"] == before["by_status"]["uncertain"] + 1


def test_next_unreviewed_wraps(client):
    items = client.get("/api/items").json()["items"]
    first = client.get("/api/next_unreviewed").json()["item_id"]
    assert first == items[0]["item_id"]
    nxt = client.get(
        f"/api/next_unreviewed?after={items[-1]['item_id']}"
    ).json()["item_id"]
    assert nxt == first


def test_crop_context_prefill_pngs(client):
    item = _some_item(client, "exhaustive_tile")
    full = client.get(f"/api/items/{item['item_id']}").json()
    resp = client.get(f"/api/items/{item['item_id']}/crop/BSE.png")
    assert resp.status_code == 200
    arr = np.asarray(Image.open(io.BytesIO(resp.content)))
    assert arr.shape == (full["tile"]["h"], full["tile"]["w"])
    ctx = client.get(f"/api/items/{item['item_id']}/context.png")
    assert ctx.status_code == 200
    ctx_arr = np.asarray(Image.open(io.BytesIO(ctx.content)))
    assert ctx_arr.ndim == 3 and max(ctx_arr.shape[:2]) <= 512
    pre = client.get(f"/api/items/{item['item_id']}/prefill.png")
    assert pre.status_code == 200

    cand = _some_item(client, "candidate")
    assert client.get(f"/api/items/{cand['item_id']}/prefill.png").status_code == 404
    assert client.get(f"/api/items/{cand['item_id']}/mask.png").status_code == 404
    assert client.get("/api/items/nonexistent").status_code == 404


def _blank_client(synthetic, tmp_path, n_blank=1):
    import hashlib as _hl
    from fastapi.testclient import TestClient as TC
    cfg = dict(synthetic["config"])
    cfg["blank_start_tiles"] = n_blank
    build_review_set(
        manifest_path=synthetic["manifest"],
        preds_root=synthetic["preds"],
        out_dir=synthetic["out"],
        review_config=cfg,
        manifest_sha256=_hl.sha256(
            synthetic["manifest"].read_bytes()
        ).hexdigest(),
    )
    static_dir = tmp_path / "static_b"
    static_dir.mkdir(exist_ok=True)
    (static_dir / "index.html").write_text("<html>ok</html>")
    app = create_app(
        review_dir=synthetic["out"],
        image_loader=FakeLoader(),
        static_dir=static_dir,
    )
    return TC(app)


def _tile(client, prefill=None):
    for item in client.get("/api/items?kind=exhaustive_tile").json()["items"]:
        full = client.get(f"/api/items/{item['item_id']}").json()
        if prefill is None or full.get("prefill") == prefill:
            return full
    raise AssertionError(f"no tile with prefill={prefill}")


def test_changed_px_frac_classical_tile(client):
    item = _tile(client, "classical_v1")
    w, h = item["tile"]["w"], item["tile"]["h"]
    mask = np.zeros((h, w), dtype=np.uint8)
    k = 17
    mask.ravel()[:k] = 3
    client.post(
        f"/api/items/{item['item_id']}/mask", content=mask.tobytes()
    )
    r = client.post(
        f"/api/items/{item['item_id']}/decision",
        json={
            "status": "redrawn",
            "reviewer_id": "ann",
            "semantic_png": f"masks/{item['item_id']}.png",
        },
    )
    assert r.status_code == 200, r.text
    human = r.json()["human"]
    assert human["changed_px_frac"] > 0
    # expected: fraction of the mask differing from the classical prefill crop
    pre = client.get(f"/api/items/{item['item_id']}/prefill.png")
    ref = np.asarray(Image.open(io.BytesIO(pre.content)))
    expected = float(np.mean(mask != ref))
    assert human["changed_px_frac"] == pytest.approx(expected)
    assert human["changed_px_frac_vs_classical"] == pytest.approx(expected)

    r = client.post(
        f"/api/items/{item['item_id']}/decision",
        json={"status": "accepted", "reviewer_id": "ann"},
    )
    human = r.json()["human"]
    assert human["changed_px_frac"] == 0.0
    assert human["changed_px_frac_vs_classical"] == 0.0


def test_blank_tile_decisions(synthetic, tmp_path):
    client = _blank_client(synthetic, tmp_path, n_blank=1)
    blank = _tile(client, "blank")
    iid = blank["item_id"]
    w, h = blank["tile"]["w"], blank["tile"]["h"]
    r = client.post(
        f"/api/items/{iid}/decision",
        json={"status": "accepted", "reviewer_id": "ann"},
    )
    assert r.status_code == 422
    m = 25
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask.ravel()[:m] = 1
    client.post(f"/api/items/{iid}/mask", content=mask.tobytes())
    r = client.post(
        f"/api/items/{iid}/decision",
        json={
            "status": "redrawn",
            "reviewer_id": "ann",
            "semantic_png": f"masks/{iid}.png",
            "vlm_viewed": True,
        },
    )
    assert r.status_code == 200, r.text
    human = r.json()["human"]
    assert human["changed_px_frac"] == pytest.approx(m / (w * h))
    # vs classical reference (the hidden prefill crop still on disk)
    ref_rel = blank["proposal"]["reference_semantic_png"]
    import pathlib
    ref = np.asarray(
        Image.open(pathlib.Path(synthetic["out"]) / ref_rel)
    )
    assert human["changed_px_frac_vs_classical"] == pytest.approx(
        float(np.mean(mask != ref))
    )
    assert human["vlm_viewed"] is True


def test_vlm_viewed_validation(client):
    item = _some_item(client)
    r = client.post(
        f"/api/items/{item['item_id']}/decision",
        json={"status": "accepted", "reviewer_id": "ann"},
    )
    assert r.json()["human"]["vlm_viewed"] is False
    r = client.post(
        f"/api/items/{item['item_id']}/decision",
        json={"status": "accepted", "reviewer_id": "ann",
              "vlm_viewed": "yes"},
    )
    assert r.status_code == 422
