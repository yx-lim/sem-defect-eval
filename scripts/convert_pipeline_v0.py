"""Convert pipeline_v0 proposal JSONL into QC candidate predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sem.qc.adapters.pipeline_v0 import (
    convert_proposals,
    write_proposals_map,
)
from sem.qc.config import load_config
from sem.qc.io import list_stems, valid_mask
from sem.qc.models.pred_store import write_prediction
from sem.qc.schema import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--config", default="configs/qc.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = args.data_root or config["paths"]["data_root"]
    work_root = Path(os.environ.get("SEM_WORK_ROOT", config["paths"]["work_root"]))
    output_dir = Path(
        args.out_dir or work_root / "preds" / "pipeline_v0_proposals"
    )
    proposal_rows = read_jsonl(args.proposals)
    records = list_stems(data_root)
    stem_shapes = {
        record.stem: {
            "shape": (record.height, record.width),
            "valid": valid_mask(record),
        }
        for record in records
    }
    predictions = convert_proposals(proposal_rows, stem_shapes)
    for stem, prediction in predictions.items():
        write_prediction(output_dir, stem, prediction, include_uncertainty=False)
    write_proposals_map(output_dir / "proposals_map.csv", proposal_rows, predictions)
    print(json.dumps({"stems_seen": sorted(predictions)}, sort_keys=True))


if __name__ == "__main__":
    main()
