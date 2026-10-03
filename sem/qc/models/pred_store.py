"""Private persistence helpers for method prediction files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from sem.qc.schema import Prediction


def write_prediction(
    output_dir: str | Path,
    stem: str,
    prediction: Prediction,
    include_uncertainty: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.semantic, mode="L").save(
        output_dir / f"{stem}_semantic.png"
    )
    if include_uncertainty and prediction.uncertainty is not None:
        uncertainty = np.rint(
            np.clip(prediction.uncertainty, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        Image.fromarray(uncertainty, mode="L").save(
            output_dir / f"{stem}_uncertainty.png"
        )
    instances = [
        {
            "class_name": instance.class_name,
            "subtype": instance.subtype,
            "bbox": instance.bbox,
            "polygon": instance.polygon,
            "score": instance.score,
            "source": instance.source,
        }
        for instance in prediction.instances
    ]
    with (output_dir / f"{stem}_instances.json").open("w", encoding="utf-8") as file:
        json.dump(instances, file, indent=2, sort_keys=True)
