"""Tests for sem.qc.review.vlm with a fake injected client (no network)."""

from __future__ import annotations

import json

import pytest

from sem.qc.review import vlm
from conftest import FakeLoader


GOOD_JSON = json.dumps(
    {
        "class_name": "pore",
        "artifact_subtype": None,
        "confidence": 0.8,
        "rationale": "uniformly black region with no internal structure",
    }
)


class _Block:
    def __init__(self, text, block_type="text"):
        self.type = block_type
        self.text = text


class _Response:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason


class _Models:
    def __init__(self, ids):
        self._ids = ids

    def list(self):
        page = type("Page", (), {})()
        page.data = [type("M", (), {"id": i})() for i in self._ids]
        page.has_next_page = False
        return page


class _Messages:
    def __init__(self, script):
        self.script = script  # list of responses or exceptions
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            return _Response(GOOD_JSON)
        effect = self.script.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class FakeClient:
    def __init__(self, ids, script=None):
        self.models = _Models(ids)
        self.messages = _Messages(script or [])


def _items():
    return [
        {
            "item_id": "c1",
            "kind": "candidate",
            "stem": "s2",
            "tile": {"x0": 10, "y0": 10, "w": 128, "h": 128},
            "proposal": {
                "source": "classical_v1",
                "class_name": "pore",
                "subtype": None,
                "polygon": [[10, 10], [30, 10], [30, 25], [10, 25]],
            },
            "vlm_suggestion": None,
            "human": None,
        },
        {
            "item_id": "c2",
            "kind": "candidate",
            "stem": "s3",
            "tile": {"x0": 10, "y0": 10, "w": 128, "h": 128},
            "proposal": {
                "source": "unet_pseudo_v1",
                "class_name": "artifact",
                "subtype": "curtaining",
                "polygon": None,
            },
            "vlm_suggestion": None,
            "human": None,
        },
        {
            "item_id": "t1",
            "kind": "exhaustive_tile",
            "stem": "s2",
            "tile": {"x0": 0, "y0": 0, "w": 512, "h": 512},
            "proposal": {"source": "classical_v1", "class_name": None},
            "vlm_suggestion": None,
            "human": None,
        },
    ]


CFG = {
    "model_id": "fake-model-1",
    "prompt_path": None,
    "prompt_version": "vlm_prompt_v1",
    "max_tokens": 2000,
}


@pytest.fixture
def prompt_file(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text(
        "=== SYSTEM ===\nsys text\n=== USER ===\n"
        '{source} -> {proposed_class} returns {"class_name": x}\n'
    )
    return path


def test_prompt_split_and_fill(prompt_file):
    system, user = vlm.load_prompt(prompt_file)
    assert system == "sys text"
    filled = vlm.fill_user(user, "m1", "pore (subtype: x)")
    assert "{source}" not in filled and "{proposed_class}" not in filled
    assert '{"class_name": x}' in filled  # literal braces preserved


def test_annotate_happy_path(prompt_file, tmp_path):
    cfg = dict(CFG, prompt_path=str(prompt_file))
    client = FakeClient(ids={"fake-model-1"})
    items = _items()
    stats = vlm.annotate_candidates(
        items, FakeLoader(), cfg, tmp_path, client=client, sleep_fn=lambda s: None
    )
    assert stats["requested"] == 2
    assert stats["ok"] == 2
    suggestion = items[0]["vlm_suggestion"]
    assert suggestion["class_name"] == "pore"
    assert suggestion["label"] == "VLM suggestion — not ground truth"
    assert suggestion["prompt_version"] == "vlm_prompt_v1"
    assert items[0]["human"] is None
    # subtype appended to proposed_class text
    sent = client.messages.calls[-1]["messages"][0]["content"][-1]["text"]
    assert "artifact (subtype: curtaining)" in sent
    assert (tmp_path / "vlm_raw.jsonl").exists()


def test_model_not_listed_skips(prompt_file, tmp_path):
    cfg = dict(CFG, prompt_path=str(prompt_file))
    client = FakeClient(ids={"other-model"})
    items = _items()
    stats = vlm.annotate_candidates(
        items, FakeLoader(), cfg, tmp_path, client=client
    )
    assert stats["requested"] == 0
    assert "not in client.models.list()" in stats["skipped_reason"]


def test_parse_fenced_and_invalid():
    assert vlm.parse_response(f"```json\n{GOOD_JSON}\n```") is not None
    assert vlm.parse_response(GOOD_JSON) is not None
    bad_class = json.loads(GOOD_JSON)
    bad_class["class_name"] = "nope"
    assert vlm.parse_response(json.dumps(bad_class)) is None
    assert vlm.parse_response(GOOD_JSON + " trailing") is None
    bad_conf = json.loads(GOOD_JSON)
    bad_conf["confidence"] = 1.5
    assert vlm.parse_response(json.dumps(bad_conf)) is None
    extra = json.loads(GOOD_JSON)
    extra["extra"] = 1
    assert vlm.parse_response(json.dumps(extra)) is None


def test_invalid_response_counted(prompt_file, tmp_path):
    cfg = dict(CFG, prompt_path=str(prompt_file))
    client = FakeClient(
        ids={"fake-model-1"}, script=[_Response("not json at all")]
    )
    items = _items()
    stats = vlm.annotate_candidates(
        items, FakeLoader(), cfg, tmp_path, client=client, sleep_fn=lambda s: None
    )
    assert stats["invalid"] == 1
    assert items[0]["vlm_suggestion"] is None
    raw = (tmp_path / "vlm_raw.jsonl").read_text()
    assert "not json at all" in raw


def test_api_error_continues(prompt_file, tmp_path):
    cfg = dict(CFG, prompt_path=str(prompt_file))
    client = FakeClient(
        ids={"fake-model-1"},
        script=[RuntimeError("boom"), RuntimeError("boom")],
    )
    items = _items()
    stats = vlm.annotate_candidates(
        items, FakeLoader(), cfg, tmp_path, client=client, sleep_fn=lambda s: None
    )
    assert stats["errors"] == 1
    assert stats["ok"] == 1
    assert items[0]["vlm_suggestion"] is None
    assert items[1]["vlm_suggestion"]["class_name"] == "pore"


def test_max_tokens_retry(prompt_file, tmp_path):
    cfg = dict(CFG, prompt_path=str(prompt_file))
    client = FakeClient(
        ids={"fake-model-1"}, script=[_Response("truncated", stop_reason="max_tokens")]
    )
    items = _items()[:1]
    stats = vlm.annotate_candidates(
        items, FakeLoader(), cfg, tmp_path, client=client, sleep_fn=lambda s: None
    )
    assert stats["ok"] == 1
    assert client.messages.calls[1]["max_tokens"] == 4000
