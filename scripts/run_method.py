#!/usr/bin/env python3
"""Run a QC method over every discovered stem and save predictions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sem.qc.classical import ClassicalV1
from sem.qc.config import load_config
from sem.qc.io import list_stems, load_stem, valid_mask
from sem.qc.schema import CLASS_NAMES, Instance, Prediction


_PALETTE = {
    0: (100, 100, 100),
    1: (70, 120, 220),
    2: (255, 190, 0),
    3: (30, 30, 240),
    4: (180, 80, 200),
    5: (255, 30, 30),
    6: (255, 80, 160),
    7: (0, 220, 220),
}


def _serialize_instance(instance: Instance) -> dict:
    return {
        "class_name": instance.class_name,
        "subtype": instance.subtype,
        "bbox": instance.bbox,
        "polygon": instance.polygon,
        "score": instance.score,
        "source": instance.source,
    }


def _save_overlay(
    path: Path, bse: np.ndarray, prediction: Prediction, scale: float
) -> None:
    base = Image.fromarray(bse).convert("RGB")
    output_size = (
        max(1, int(round(base.width * scale))),
        max(1, int(round(base.height * scale))),
    )
    base = base.resize(output_size, Image.Resampling.BILINEAR)
    labels = prediction.semantic
    color = np.zeros((*labels.shape, 4), dtype=np.uint8)
    for label, rgb in _PALETTE.items():
        mask = labels == label
        color[mask, :3] = rgb
        color[mask, 3] = 100
    overlay = Image.fromarray(color, mode="RGBA").resize(
        output_size, Image.Resampling.NEAREST
    )
    base = Image.alpha_composite(base.convert("RGBA"), overlay)
    canvas = Image.new("RGB", (base.width, base.height + 56), (25, 25, 25))
    canvas.paste(base.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    x = 8
    y = base.height + 10
    legend = list(CLASS_NAMES.items()) + [(255, "ignore")]
    for label, name in legend:
        swatch = _PALETTE.get(label, (225, 225, 225))
        draw.rectangle(
            (x, y, x + 12, y + 12),
            fill=swatch,
            outline=(110, 110, 110) if label == 255 else None,
        )
        draw.text((x + 16, y - 2), name, fill=(245, 245, 245))
        x += 20 + int(draw.textlength(name))
        if x > canvas.width - 100:
            x = 8
            y += 22
    canvas.save(path)


def run(method_name: str) -> list[tuple[str, float]]:
    if method_name != "classical_v1":
        raise ValueError(f"Unsupported method {method_name!r}")
    config = load_config()
    data_root = Path(config["paths"]["data_root"])
    work_root = Path(config["paths"]["work_root"])
    output_root = work_root / "preds" / method_name
    output_root.mkdir(parents=True, exist_ok=True)
    overlay_root = output_root / "overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)
    method = ClassicalV1(config["classical_v1"])
    records = list_stems(data_root)
    if len(records) != 31:
        raise RuntimeError(f"Expected 31 stems in {data_root}, found {len(records)}")

    runtimes = []
    batch_overlays = set()
    scale = float(config["tile_sizes"]["similarity_scale"])
    for record in records:
        views = load_stem(record)
        mask = valid_mask(record)
        start = time.perf_counter()
        prediction = method.predict(views, mask)
        elapsed = time.perf_counter() - start
        runtimes.append((record.stem, elapsed))

        Image.fromarray(prediction.semantic, mode="L").save(
            output_root / f"{record.stem}_semantic.png"
        )
        if prediction.uncertainty is not None:
            uncertainty = np.rint(prediction.uncertainty * 255).astype(np.uint8)
            Image.fromarray(uncertainty, mode="L").save(
                output_root / f"{record.stem}_uncertainty.png"
            )
        with (output_root / f"{record.stem}_instances.json").open(
            "w", encoding="utf-8"
        ) as output_file:
            json.dump(
                [_serialize_instance(instance) for instance in prediction.instances],
                output_file,
                indent=2,
                sort_keys=True,
            )
        if record.batch not in batch_overlays:
            _save_overlay(
                overlay_root / f"{record.batch}.png",
                views["BSE"],
                prediction,
                scale,
            )
            batch_overlays.add(record.batch)
    return runtimes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("classical_v1",))
    args = parser.parse_args()
    runtimes = run(args.method)
    for stem, seconds in runtimes:
        print(f"{stem}: {seconds:.3f}s")
    print(f"Mean runtime: {np.mean([value for _, value in runtimes]):.3f}s/image")


if __name__ == "__main__":
    main()
