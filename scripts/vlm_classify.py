"""Build review candidates and request optional Claude crop suggestions."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from sem.qc.config import load_config
from sem.qc.io import list_stems, load_stem
from sem.qc.models.vlm import (
    DEFAULT_PROMPT_PATH,
    classify_item,
    load_prompt,
    render_views,
    resolve_model_id,
)
from sem.qc.schema import make_item_id, read_jsonl, write_jsonl


def _load_splits(root: Path, config: dict) -> dict[str, str]:
    manifest_path = root / "data" / "splits" / "manifest.csv"
    try:
        from sem.qc.split import load_manifest
    except ImportError:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            import csv

            return {
                row["stem"]: row["split"]
                for row in csv.DictReader(manifest_file)
            }
    rows = load_manifest(
        manifest_path, config.get("split", {}).get("frozen_manifest_sha256")
    )
    return {str(row["stem"]): str(row["split"]) for row in rows}


def make_items_from_predictions(
    preds_dir: str | Path,
    splits: set[str],
    n: int,
    seed: int,
    config: dict,
) -> list[dict]:
    from sem.qc.io import list_stems

    preds_dir = Path(preds_dir)
    manifest = _load_splits(Path(__file__).resolve().parents[1], config)
    records = {
        record.stem: record
        for record in list_stems(config["paths"]["data_root"])
        if manifest.get(record.stem) in splits
    }
    by_class: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(preds_dir.glob("*_instances.json")):
        stem = path.name.removesuffix("_instances.json")
        if stem not in records:
            continue
        instances = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(instances, dict):
            instances = instances.get("instances", [])
        for instance in instances:
            box = instance["bbox"]
            x0, y0, x1, y1 = [int(round(float(value))) for value in box]
            by_class[str(instance["class_name"])].append(
                {
                    "stem": stem,
                    "record": records[stem],
                    "instance": instance,
                    "tile": {
                        "x0": x0,
                        "y0": y0,
                        "w": max(1, x1 - x0),
                        "h": max(1, y1 - y0),
                    },
                }
            )
    if not by_class:
        return []
    rng = np.random.default_rng(seed)
    classes = sorted(by_class)
    allocation = {name: n // len(classes) for name in classes}
    for name in classes[: n % len(classes)]:
        allocation[name] += 1
    selected = []
    leftovers = []
    for name in classes:
        candidates = by_class[name]
        order = rng.permutation(len(candidates))
        count = min(allocation[name], len(candidates))
        selected.extend(candidates[index] for index in order[:count])
        leftovers.extend(candidates[index] for index in order[count:])
    if len(selected) < min(n, sum(len(rows) for rows in by_class.values())):
        order = rng.permutation(len(leftovers))
        selected.extend(
            leftovers[index]
            for index in order[: n - len(selected)]
        )
    selected = selected[:n]
    total_by_class = {name: len(rows) for name, rows in by_class.items()}
    selected_by_class: dict[str, int] = defaultdict(int)
    for row in selected:
        selected_by_class[str(row["instance"]["class_name"])] += 1
    method = preds_dir.name
    items = []
    for row in selected:
        instance = row["instance"]
        stem = row["stem"]
        tile = row["tile"]
        source = str(instance.get("source", method))
        class_name = str(instance["class_name"])
        items.append(
            {
                "item_id": make_item_id(
                    stem,
                    "candidate",
                    tile["x0"],
                    tile["y0"],
                    tile["w"],
                    tile["h"],
                    source,
                ),
                "stem": stem,
                "batch": row["record"].batch,
                "split": manifest[stem],
                "kind": "candidate",
                "tile": tile,
                "sampling": {
                    "stratum": class_name,
                    "method": "random",
                    "weight": total_by_class[class_name]
                    / max(1, selected_by_class[class_name]),
                },
                "proposal": {
                    "source": source,
                    "class_name": class_name,
                    "subtype": instance.get("subtype"),
                    "polygon": instance.get("polygon"),
                    "semantic_png": str(
                        preds_dir / f"{stem}_semantic.png"
                    ) if (preds_dir / f"{stem}_semantic.png").exists() else None,
                    "score": float(instance.get("score", 0.0)),
                    "uncertainty": None,
                },
                "vlm_suggestion": None,
                "human": None,
            }
        )
    return items


def _views_for_item(item: dict, records: dict[str, object]) -> dict[str, np.ndarray]:
    record = records[item["stem"]]
    return load_stem(record)


def _dry_run(
    candidates: list[dict],
    records: dict[str, object],
    output_path: Path,
    prompt_path: str | Path,
) -> None:
    system, template = load_prompt(prompt_path)
    dry_dir = output_path.parent / "dry_run"
    dry_dir.mkdir(parents=True, exist_ok=True)
    for item in candidates:
        proposal = item["proposal"]
        if proposal.get("polygon"):
            points = np.asarray(proposal["polygon"], dtype=np.float64)
            bbox = (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
        else:
            tile = item["tile"]
            bbox = (
                tile["x0"],
                tile["y0"],
                tile["x0"] + tile["w"],
                tile["y0"] + tile["h"],
            )
        views = _views_for_item(item, records)
        crops = render_views(views["BSE"], views["Inlens"], bbox, proposal.get("polygon"))
        item_dir = dry_dir / item["item_id"]
        item_dir.mkdir(parents=True, exist_ok=True)
        for name, crop in zip(("bse_crop", "inlens_crop", "bse_context"), crops):
            crop.save(item_dir / f"{name}.png")
        proposed_class = str(proposal.get("class_name", ""))
        if proposal.get("subtype") is not None:
            proposed_class = f"{proposed_class} ({proposal['subtype']})"
        rendered = template.replace("{source}", str(proposal.get("source", "")))
        rendered = rendered.replace("{proposed_class}", proposed_class)
        (item_dir / "prompt.txt").write_text(
            f"=== SYSTEM ===\n{system}\n\n=== USER ===\n{rendered}\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items")
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--make-items-from")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--items-out")
    parser.add_argument("--config", default="configs/qc.yaml")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT_PATH))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.make_items_from:
        if not args.items_out:
            parser.error("--items-out is required with --make-items-from")
        split_values = {
            value for entry in args.splits for value in entry.split(",") if value
        }
        items = make_items_from_predictions(
            args.make_items_from, split_values, args.n, args.seed, config
        )
        write_jsonl(args.items_out, items)
        print(f"wrote {len(items)} candidate items to {args.items_out}")
        return
    if not args.items or not args.out:
        parser.error("--items and --out are required for classification")

    all_items = read_jsonl(args.items)
    candidates = [item for item in all_items if item.get("kind") == "candidate"]
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output_path) if output_path.exists() else []
    existing_ids = {row["item_id"] for row in existing}
    candidates = [item for item in candidates if item["item_id"] not in existing_ids]
    if args.limit is not None:
        candidates = candidates[: max(0, args.limit)]
    records = {
        record.stem: record for record in list_stems(config["paths"]["data_root"])
    }
    if args.dry_run:
        _dry_run(candidates, records, output_path, args.prompt)
        return

    import anthropic

    client = anthropic.Anthropic()
    preferred = str(config.get("vlm_claude_crops", {}).get("model_preferred", ""))
    model_id = resolve_model_id(client, preferred)
    results = list(existing)
    for item in candidates:
        results.append(
            classify_item(
                client,
                model_id,
                item,
                _views_for_item(item, records),
                prompt_path=args.prompt,
            )
        )
        write_jsonl(output_path, results)
    if not candidates and not output_path.exists():
        write_jsonl(output_path, [])


if __name__ == "__main__":
    main()
