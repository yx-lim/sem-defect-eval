#!/usr/bin/env python3
"""Generate the deterministic train/validation/test stem manifest."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sem.qc.config import load_config
from sem.qc.io import list_stems
from sem.qc.split import (
    assign_splits,
    load_group_overrides,
    manifest_rows,
    write_manifest,
)


def main() -> None:
    config = load_config()
    root = Path(__file__).resolve().parents[1]
    records = list_stems(config["paths"]["data_root"])
    if len(records) != 31:
        raise RuntimeError(f"Expected 31 stems, found {len(records)}")
    overrides_path = Path(config["paths"]["groups_override"])
    if not overrides_path.is_absolute():
        overrides_path = root / overrides_path
    overrides = load_group_overrides(overrides_path)
    split_config = config["split"]
    stem_splits, group_by_stem = assign_splits(
        records,
        seed=int(config["seeds"]["split"]),
        groups_override=overrides,
        test_fraction=float(split_config["test_fraction"]),
        val_fraction=float(split_config["val_fraction"]),
    )
    rows = manifest_rows(records, stem_splits, group_by_stem)
    split_dir = root / "data" / "splits"
    manifest_path = split_dir / "manifest.csv"
    summary_path = split_dir / "manifest_summary.md"
    digest = write_manifest(
        rows, manifest_path, summary_path, int(config["seeds"]["split"])
    )
    print(f"Wrote {manifest_path} ({len(rows)} stems)")
    print(f"Manifest SHA256: {digest}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
