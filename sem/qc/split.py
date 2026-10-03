"""Deterministic batch/detector-stratified stem splits."""

from __future__ import annotations

import csv
import hashlib
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
import yaml

from sem.qc.io import StemRecord


SPLIT_ORDER = ("train", "val", "test")
VIEW_ORDER = ("BSE", "Inlens", "ETD", "SE")


def normalize_group_overrides(
    stems: Iterable[str], overrides: dict[str, Any] | None
) -> dict[str, str]:
    """Map each known stem to itself or to a manually defined group id."""
    known = set(stems)
    group_by_stem = {stem: stem for stem in known}
    overrides = overrides or {}
    if not isinstance(overrides, dict):
        raise ValueError("groups_override.yaml must contain a mapping")
    claimed_members: dict[str, str] = {}
    for key, value in overrides.items():
        if isinstance(value, (list, tuple)):
            members = [str(member) for member in value]
            if len(members) < 2:
                raise ValueError(f"Group {key!r} must contain at least two stems")
            unknown = set(members) - known
            if unknown:
                raise ValueError(f"Unknown stems in group {key!r}: {sorted(unknown)}")
            for member in members:
                if member in claimed_members and claimed_members[member] != str(key):
                    raise ValueError(
                        f"Stem {member!r} appears in multiple group overrides"
                    )
                claimed_members[member] = str(key)
                group_by_stem[member] = str(key)
        elif isinstance(value, str):
            stem = str(key)
            if stem not in known:
                raise ValueError(f"Unknown stem in group override: {stem}")
            group_by_stem[stem] = value
        else:
            raise ValueError(
                "Group overrides must map group ids to stem lists or stems to ids"
            )
    return group_by_stem


def assign_splits(
    records: list[StemRecord],
    seed: int = 20261003,
    groups_override: dict[str, Any] | None = None,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign whole groups independently within each batch × detector stratum."""
    stems = [record.stem for record in records]
    if len(stems) != len(set(stems)):
        raise ValueError("Stem ids must be unique across batches")
    group_by_stem = normalize_group_overrides(stems, groups_override)
    groups_by_stratum: dict[tuple[str, str], set[str]] = defaultdict(set)
    strata_by_group: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        group_id = group_by_stem[record.stem]
        stratum = (record.batch, record.detector_set)
        groups_by_stratum[stratum].add(group_id)
        strata_by_group[group_id].add(stratum)
    group_splits: dict[str, str] = {}
    for stratum in sorted(groups_by_stratum):
        groups = sorted(
            groups_by_stratum[stratum],
            key=lambda group_id: hashlib.sha256(
                f"{seed}:{group_id}".encode("utf-8")
            ).hexdigest(),
        )
        n = len(groups)
        n_test = int(round(test_fraction * n))
        n_val = int(round(val_fraction * n))
        if n >= 3:
            n_test = max(n_test, 1)
            n_val = max(n_val, 1)
        assigned = {
            group_id: (
                "test"
                if index < n_test
                else "val"
                if index < n_test + n_val
                else "train"
            )
            for index, group_id in enumerate(groups)
        }
        for group_id, split_name in assigned.items():
            previous = group_splits.get(group_id)
            if previous is not None and previous != split_name:
                conflicting_strata = sorted(strata_by_group[group_id])
                raise ValueError(
                    f"Group {group_id!r} spans strata {conflicting_strata} "
                    "whose deterministic assignments disagree"
                )
            group_splits[group_id] = split_name
    stem_splits = {
        stem: group_splits[group_by_stem[stem]] for stem in stems
    }
    return stem_splits, group_by_stem


def assert_frozen(
    manifest_path: str | Path, expected_sha256: str | None
) -> str | None:
    """Verify an approved manifest hash; a null hash warns before approval."""
    if expected_sha256 is None:
        warnings.warn(
            "Split manifest is not hash-frozen; checkpoint 1 approval is pending.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    actual = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise AssertionError(
            f"Frozen split manifest hash mismatch: expected {expected_sha256}, "
            f"found {actual}"
        )
    return actual


def load_manifest(
    manifest_path: str | Path, expected_sha256: str | None
) -> list[dict[str, str]]:
    """Load a manifest only after checking its configured freeze hash."""
    assert_frozen(manifest_path, expected_sha256)
    with Path(manifest_path).open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nongray_pixel_count(record: StemRecord) -> int:
    bse = tifffile.imread(record.views["BSE"])
    if bse.ndim != 3:
        return 0
    return int(np.count_nonzero(np.any(bse[..., 1:] != bse[..., :1], axis=-1)))


def manifest_rows(
    records: list[StemRecord],
    stem_splits: dict[str, str],
    group_by_stem: dict[str, str],
) -> list[dict[str, Any]]:
    """Collect stable manifest data and content hashes for every view."""
    rows = []
    for record in records:
        view_names = [view for view in VIEW_ORDER if view in record.views]
        hashes = [_sha256_file(record.views[view]) for view in view_names]
        rows.append(
            {
                "stem": record.stem,
                "group_id": group_by_stem[record.stem],
                "batch": record.batch,
                "detector_set": record.detector_set,
                "split": stem_splits[record.stem],
                "views": ";".join(view_names),
                "height": record.height,
                "width": record.width,
                "pixel_size_nm": f"{record.pixel_size_nm:.6f}",
                "n_nongray_px": _nongray_pixel_count(record),
                "files_sha256": ";".join(hashes),
            }
        )
    return rows


def write_manifest(
    rows: list[dict[str, Any]],
    manifest_path: str | Path,
    summary_path: str | Path,
    seed: int,
) -> str:
    manifest_path = Path(manifest_path)
    summary_path = Path(summary_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stem",
        "group_id",
        "batch",
        "detector_set",
        "split",
        "views",
        "height",
        "width",
        "pixel_size_nm",
        "n_nongray_px",
        "files_sha256",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(_summary_markdown(rows, seed), encoding="utf-8")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _summary_markdown(rows: list[dict[str, Any]], seed: int) -> str:
    counts = Counter(
        (row["split"], row["batch"], row["detector_set"]) for row in rows
    )
    batches = sorted({row["batch"] for row in rows})
    detector_sets = sorted({row["detector_set"] for row in rows})
    lines = [
        "# Split manifest summary",
        "",
        f"Split seed: `{seed}`. Counts are stems (one row per stem).",
        "",
        "| split | batch | detector_set | stems |",
        "|---|---|---|---:|",
    ]
    for split_name in SPLIT_ORDER:
        for batch in batches:
            for detector_set in detector_sets:
                count = counts[(split_name, batch, detector_set)]
                lines.append(
                    f"| {split_name} | {batch} | {detector_set} | {count} |"
                )
    lines.append("")
    return "\n".join(lines)


def load_group_overrides(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as override_file:
        value = yaml.safe_load(override_file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Group overrides must be a YAML mapping: {path}")
    return value
