import hashlib

from sem.qc.schema import (
    HumanDecision,
    ReviewItem,
    is_ground_truth,
    make_item_id,
    read_jsonl,
    resolve_review_items,
    write_jsonl,
)


def _item():
    return ReviewItem(
        item_id="abc123",
        stem="sample",
        batch="Batch_1",
        split="test",
        kind="candidate",
        tile={"x0": 0, "y0": 0, "w": 512, "h": 512},
        sampling={"stratum": "random", "method": "random", "weight": 1.0},
        proposal={"source": "classical_v1", "class_name": "pore", "score": 0.8},
    )


def test_latest_human_decision_wins_without_mutating_proposal():
    item = _item()
    resolved = resolve_review_items(
        [item],
        [
            {"item_id": item.item_id, "human": {"status": "rejected"}},
            {
                "item_id": item.item_id,
                "human": {"status": "relabeled", "class_name": "interparticle_gap"},
            },
        ],
    )

    assert resolved[0].human == {
        "status": "relabeled",
        "class_name": "interparticle_gap",
    }
    assert resolved[0].proposal["class_name"] == "pore"
    assert item.human is None


def test_is_ground_truth_status_truth_table():
    expected = {
        None: False,
        "accepted": True,
        "rejected": False,
        "relabeled": True,
        "redrawn": True,
        "uncertain": False,
    }
    for status, is_gt in expected.items():
        assert is_ground_truth({"human": {"status": status}}) is is_gt
    assert is_ground_truth({"human": HumanDecision(status="accepted")})
    assert is_ground_truth(_item()) is False


def test_jsonl_round_trip_models_and_dicts(tmp_path):
    path = tmp_path / "review" / "items.jsonl"
    write_jsonl(path, [_item(), {"item_id": "second", "human": None}])

    assert [row["item_id"] for row in read_jsonl(path)] == ["abc123", "second"]


def test_review_id_is_stable_and_uses_specified_sha1_input():
    expected = hashlib.sha1(b"s|candidate|1,2,3,4|m").hexdigest()[:12]
    assert make_item_id("s", "candidate", 1, 2, 3, 4, "m") == expected
