import hashlib
from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest
import tifffile

from sem.qc.io import StemRecord
from sem.qc.split import (
    assert_frozen,
    assign_splits,
    manifest_rows,
    normalize_group_overrides,
)


def _records(n=10):
    return [
        StemRecord(
            stem=f"stem-{index}",
            batch="Batch_1",
            detector_set="ETD",
            views={"BSE": f"stem-{index}-BSE.tif", "Inlens": f"stem-{index}-Inlens.tif"},
            height=20,
            width=30,
            pixel_size_nm=25.0,
        )
        for index in range(n)
    ]


def test_split_is_deterministic_and_keeps_group_members_together():
    records = _records()
    overrides = {"merged": ["stem-0", "stem-1"]}
    first, groups = assign_splits(records, groups_override=overrides)
    second, second_groups = assign_splits(records, groups_override=overrides)

    assert first == second
    assert groups == second_groups
    assert first["stem-0"] == first["stem-1"]
    assert len(set(first.values())) > 1
    assert len({first[stem] for stem in overrides["merged"]}) == 1


def test_stratum_with_at_least_three_groups_has_test_and_val():
    records = _records(5)

    splits, _ = assign_splits(records, seed=20261003)

    assert sum(split == "test" for split in splits.values()) >= 1
    assert sum(split == "val" for split in splits.values()) >= 1


def test_all_views_of_a_stem_share_one_manifest_split(tmp_path):
    record = _records(1)[0]
    views = {}
    for view in ("BSE", "Inlens"):
        path = tmp_path / f"{view}.tif"
        tifffile.imwrite(path, np.zeros((20, 30, 3), dtype=np.uint8))
        views[view] = path
    record = replace(record, views=views)

    splits, groups = assign_splits([record])
    rows = manifest_rows([record], splits, groups)

    assert len(rows) == 1
    assert rows[0]["views"] == "BSE;Inlens"
    assert rows[0]["split"] == splits[record.stem]


def test_a_group_that_would_cross_strata_assignments_is_rejected():
    records = _records(8)
    for index in range(1, len(records), 2):
        records[index] = replace(records[index], detector_set="SE")
    base_splits, _ = assign_splits(records)
    cross_stratum_pair = next(
        (left.stem, right.stem)
        for left, right in combinations(records, 2)
        if left.detector_set != right.detector_set
        and base_splits[left.stem] != base_splits[right.stem]
    )

    with pytest.raises(ValueError, match="spans strata"):
        assign_splits(
            records,
            groups_override={"cross_detector": list(cross_stratum_pair)},
        )


def test_override_validation_and_manifest_freeze_hash(tmp_path):
    with pytest.raises(ValueError, match="Unknown stems"):
        normalize_group_overrides(["known"], {"bad_group": ["known", "unknown"]})

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("stem,split\ns,train\n", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="not hash-frozen"):
        assert_frozen(manifest, None)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert assert_frozen(manifest, digest) == digest
