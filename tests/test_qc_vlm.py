import base64
import hashlib
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from sem.qc.models.vlm import (
    DEFAULT_PROMPT_PATH,
    classify_item,
    load_prompt,
    parse_response,
    prompt_sha256,
    render_user,
    render_views,
    resolve_model_id,
)


def test_prompt_version_and_template_rendering():
    docs_prompt = Path("docs/vlm_prompt_v1.txt").read_bytes()
    assert DEFAULT_PROMPT_PATH.read_bytes() == docs_prompt
    assert prompt_sha256() == hashlib.sha256(docs_prompt).hexdigest()
    system, template = load_prompt()
    rendered = render_user(template, "source-x", "pore (subtype)")
    assert "source-x" in rendered
    assert "pore (subtype)" in rendered
    assert "{source}" not in rendered
    assert "{proposed_class}" not in rendered
    assert '{"class_name"' in rendered
    assert "You are assisting" in system


def test_parse_response_variants_and_validation():
    valid = (
        '{"class_name":"artifact","artifact_subtype":"curtaining",'
        '"confidence":0.8,"rationale":"Parallel bands are visible."}'
    )
    parsed = parse_response(valid)
    assert parsed["parse_ok"]
    assert parsed["artifact_subtype"] == "curtaining"
    assert parse_response(f"```json\n{valid}\n```")["parse_ok"]
    assert parse_response(f"Result follows: {valid}")["parse_ok"]
    assert not parse_response("not structured").get("parse_ok")
    unknown = parse_response('{"class_name":"alien","confidence":0.8}')
    assert unknown["class_name"] == "uncertain"
    invalid_confidence = parse_response(
        '{"class_name":"pore","confidence":1.5}'
    )
    assert invalid_confidence["class_name"] == "uncertain"
    other_subtype = parse_response(
        '{"class_name":"pore","artifact_subtype":"curtaining","confidence":0.5}'
    )
    assert other_subtype["artifact_subtype"] is None


def test_render_views_sizes_border_shift_and_outline():
    bse = np.full((240, 360), 70, dtype=np.uint8)
    inlens = np.full_like(bse, 120)
    crops = render_views(bse, inlens, [0, 0, 20, 12])
    assert crops[0].mode == "RGB"
    assert crops[1].mode == "RGB"
    assert max(crops[0].size) == 384
    assert max(crops[1].size) == 384
    assert max(crops[2].size) == 768
    rgb = np.asarray(crops[2])
    assert np.any((rgb[..., 0] > 200) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80))


class _ModelPage:
    def __init__(self, data, has_more=False, last_id=None):
        self.data = data
        self.has_more = has_more
        self.last_id = last_id


class _Model:
    def __init__(self, model_id):
        self.id = model_id


class _TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ThinkingBlock:
    type = "thinking"

    def __init__(self):
        self.thinking = "private reasoning"


class _Response:
    stop_reason = "end_turn"
    usage = {"input_tokens": 4, "output_tokens": 5}

    def __init__(self, text):
        self.content = [_ThinkingBlock(), _TextBlock(text)]


class _FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.models = types.SimpleNamespace(
            list=lambda **kwargs: _ModelPage([_Model("claude-opus-5-5")])
        )
        self.messages = types.SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)


def _item():
    return {
        "item_id": "it-1",
        "stem": "sample",
        "kind": "candidate",
        "tile": {"x0": 10, "y0": 12, "w": 20, "h": 16},
        "proposal": {
            "source": "dark_void",
            "class_name": "pore",
            "subtype": None,
            "polygon": None,
        },
    }


def _valid_text():
    return (
        '{"class_name":"pore","artifact_subtype":null,'
        '"confidence":0.9,"rationale":"The region is uniformly dark."}'
    )


def test_classify_item_filters_thinking_and_sends_images_in_order():
    client = _FakeClient([_Response(_valid_text())])
    item = _item()
    result = classify_item(
        client,
        "claude-opus-5-5",
        item,
        {
            "BSE": np.full((240, 360), 30, dtype=np.uint8),
            "Inlens": np.full((240, 360), 150, dtype=np.uint8),
        },
    )
    assert result["class_name"] == "pore"
    assert result["parse_ok"]
    assert "private reasoning" not in result["raw_text"]
    request = client.requests[0]
    assert request["max_tokens"] == 2000
    assert "temperature" not in request
    blocks = request["messages"][0]["content"]
    assert [block["type"] for block in blocks] == ["image", "image", "image", "text"]
    images = [
        Image.open(io.BytesIO(base64.b64decode(block["source"]["data"])))
        for block in blocks[:3]
    ]
    assert [max(image.size) for image in images] == [384, 384, 768]
    assert np.asarray(images[0]).mean() < np.asarray(images[1]).mean()
    assert abs(np.asarray(images[2]).mean() - np.asarray(images[0]).mean()) < 0.2


def test_malformed_reply_retries_once():
    client = _FakeClient([_Response("no object"), _Response(_valid_text())])
    result = classify_item(
        client,
        "claude-opus-5-5",
        _item(),
        {
            "BSE": np.zeros((160, 160), dtype=np.uint8),
            "Inlens": np.zeros((160, 160), dtype=np.uint8),
        },
    )
    assert len(client.requests) == 2
    assert result["parse_ok"]


def test_model_resolution_lists_available_ids():
    client = _FakeClient([])
    assert resolve_model_id(client, "claude-opus-5-5") == "claude-opus-5-5"


def test_cli_classifies_candidate_from_synthetic_tiffs(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "data"
    batch = data_root / "Batch_1"
    batch.mkdir(parents=True)
    for view, value in (("BSE", 30), ("Inlens", 150), ("ETD", 200)):
        tifffile.imwrite(
            batch / f"img_sample_{view}.tif",
            np.full((32, 40), value, dtype=np.uint8),
            resolution=(1_016_000, 1_016_000),
            resolutionunit="INCH",
        )
    config = tmp_path / "qc.yaml"
    config.write_text(
        "paths:\n  data_root: " + str(data_root) + "\n  work_root: " + str(tmp_path / "work")
        + "\nvlm_claude_crops:\n  model_preferred: claude-opus-5-5\n",
        encoding="utf-8",
    )
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps(_item()) + "\n", encoding="utf-8")
    output = tmp_path / "preds" / "candidate_predictions.jsonl"
    client = _FakeClient([_Response(_valid_text())])
    monkeypatch.setitem(
        sys.modules, "anthropic", types.SimpleNamespace(Anthropic=lambda: client)
    )
    from scripts.vlm_classify import main

    main(
        [
            "--items",
            str(items),
            "--out",
            str(output),
            "--config",
            str(config),
        ]
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["item_id"] == "it-1"
    assert rows[0]["class_name"] == "pore"
    output_text = capsys.readouterr().out
    assert "n_items_sent=1" in output_text
    assert "input_tokens=4" in output_text
    assert "output_tokens=5" in output_text


def test_candidate_view_iterator_caches_one_stem_and_preserves_stable_order(
    monkeypatch,
):
    import scripts.vlm_classify as vlm_classify

    items = [
        {"item_id": "b-first", "stem": "b"},
        {"item_id": "a-first", "stem": "a"},
        {"item_id": "b-second", "stem": "b"},
    ]
    records = {"a": object(), "b": object()}
    loaded = []

    def fake_load_stem(record):
        loaded.append(record)
        return {"BSE": np.zeros((1, 1), dtype=np.uint8)}

    monkeypatch.setattr(vlm_classify, "load_stem", fake_load_stem)
    batches = list(vlm_classify._items_with_cached_views(items, records))
    assert [item["item_id"] for item, _ in batches] == [
        "a-first",
        "b-first",
        "b-second",
    ]
    assert loaded == [records["a"], records["b"]]
    assert batches[1][1] is batches[2][1]


def test_cost_summary_separates_api_errors_from_parse_failures(capsys):
    from scripts.vlm_classify import _print_cost_summary

    _print_cost_summary(
        [
            {
                "parse_ok": False,
                "error": "API unavailable",
                "usage": None,
            }
        ]
    )

    output_text = capsys.readouterr().out
    assert "n_parse_failures=0" in output_text
    assert "n_api_errors=1" in output_text
