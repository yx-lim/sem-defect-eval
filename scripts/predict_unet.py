"""Run a trained unet_pseudo_v1 checkpoint on selected stems."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from sem.qc.config import load_config
from sem.qc.io import list_stems, load_stem, valid_mask
from sem.qc.models.pred_store import write_prediction
from sem.qc.models.unet import UNetPseudoV1
from sem.qc.partial_data import list_complete_stems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stems", default="all")
    parser.add_argument("--config", default="configs/qc.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    work_root = Path(os.environ.get("SEM_WORK_ROOT", config["paths"]["work_root"]))
    missing_stems = []
    if args.allow_partial:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "splits"
            / "manifest.csv"
        )
        records, missing_stems = list_complete_stems(
            config["paths"]["data_root"], work_root, manifest_path
        )
    else:
        records = list_stems(config["paths"]["data_root"])
    selected = set(record.stem for record in records) if args.stems == "all" else set(
        args.stems.split(",")
    )
    method = UNetPseudoV1.from_run_dir(args.run_dir, device=args.device)
    run_dir = Path(args.run_dir)
    weights_sha = hashlib.sha256((run_dir / "weights.pt").read_bytes()).hexdigest()
    output_dir = work_root / "preds" / method.name
    if args.allow_partial:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "missing_stems.json").write_text(
            json.dumps(missing_stems, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Missing stems ({len(missing_stems)}): {missing_stems}", flush=True)
    per_stem = {}
    started = time.monotonic()
    for record in records:
        if record.stem not in selected:
            continue
        stem_started = time.monotonic()
        prediction = method.predict(load_stem(record), valid_mask(record))
        write_prediction(output_dir, record.stem, prediction)
        per_stem[record.stem] = time.monotonic() - stem_started
    missing = selected - set(per_stem)
    if missing:
        raise ValueError(f"Unknown requested stems: {sorted(missing)}")
    run_log = {
        "method": method.name,
        "seed": json.loads((run_dir / "config.json").read_text())["seed"],
        "weights_sha256": weights_sha,
        "per_stem_wall_s": per_stem,
        "total_wall_s": time.monotonic() - started,
    }
    if args.allow_partial:
        run_log["missing_stems"] = missing_stems
    (output_dir / "run_log.json").write_text(
        json.dumps(
            run_log,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
