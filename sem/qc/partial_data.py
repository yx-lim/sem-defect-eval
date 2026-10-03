"""Helpers for evaluating complete stems when an acquisition directory is partial."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from sem.qc.io import StemRecord, list_stems


_IMAGE_NAME = re.compile(r"^img_(?P<stem>.+)_(?P<view>BSE|Inlens|ETD|SE)\.tif$")


def list_complete_stems(
    data_root: str | Path,
    work_root: str | Path,
    manifest_path: str | Path,
) -> tuple[list[StemRecord], list[str]]:
    data_root = Path(data_root)
    filtered_root = Path(work_root) / "data_complete"
    filtered_root.mkdir(parents=True, exist_ok=True)
    with Path(manifest_path).open(newline="", encoding="utf-8") as manifest_file:
        expected = {
            row["stem"]
            for row in csv.DictReader(manifest_file)
            if row.get("stem")
        }

    for batch_dir in filtered_root.glob("Batch_*"):
        for link in batch_dir.glob("*.tif"):
            if link.is_symlink():
                link.unlink()

    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    for path in sorted(data_root.glob("Batch_*/*.tif")):
        match = _IMAGE_NAME.match(path.name)
        if match is None:
            continue
        key = (path.parent.name, match.group("stem"))
        grouped.setdefault(key, {})[match.group("view")] = path

    for (batch, stem), views in grouped.items():
        detector_views = set(views) & {"ETD", "SE"}
        if not {"BSE", "Inlens"}.issubset(views) or len(detector_views) != 1:
            continue
        destination_dir = filtered_root / batch
        destination_dir.mkdir(parents=True, exist_ok=True)
        for path in views.values():
            destination = destination_dir / path.name
            destination.symlink_to(path.resolve())

    records = list_stems(filtered_root)
    found = {record.stem for record in records}
    return records, sorted(expected - found)
