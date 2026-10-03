"""Shared synthetic fixtures for review-workstream tests (never real data)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

H, W = 1024, 1536
BORDER = 8

STEMS = [
    ("s1", "Batch_1", "train"),
    ("s2", "Batch_1", "val"),
    ("s3", "Batch_1", "test"),
    ("s4", "Batch_2", "val"),
    ("s5", "Batch_2", "test"),
    ("s6", "Batch_3", "val"),
    ("s7", "Batch_3", "test"),
    ("s8", "Batch_3", "train"),
]
CLASSES = [
    "pore",
    "crack_intraparticle",
    "interparticle_gap",
    "agglomerate",
    "artifact",
    "bright_particle",
]

REVIEW_CONFIG = {
    "seed": 20261003,
    "n_exhaustive": 25,
    "exhaustive_uncertainty_frac": 0.3,
    "exhaustive_min_valid_frac": 0.9,
    "exhaustive_uncertainty_stride_px": 256,
    "prefill_method": "classical_v1",
    "n_candidates": 60,
    "candidate_uncertainty_frac": 0.3,
    "candidate_min_per_class": 6,
    "candidate_classes": CLASSES,
    "dedup_iou": 0.5,
    "candidate_min_context_px": 128,
    "context_scale": 3.0,
    "context_max_px": 512,
    "vlm": {
        "model_id": "fake-model-1",
        "prompt_path": "unused",
        "prompt_version": "vlm_prompt_v1",
        "max_tokens": 2000,
    },
}


def write_manifest(path: Path, stems=STEMS, h=H, w=W) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["stem", "batch", "split", "height", "width"]
        )
        writer.writeheader()
        for stem, batch, split in stems:
            writer.writerow(
                {
                    "stem": stem,
                    "batch": batch,
                    "split": split,
                    "height": h,
                    "width": w,
                }
            )
    return path


def semantic_array(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 8, size=(H, W), dtype=np.uint8)
    arr[:BORDER, :] = 255
    arr[-BORDER:, :] = 255
    arr[:, :BORDER] = 255
    arr[:, -BORDER:] = 255
    return arr


def _instance(class_name, stem_idx, method_idx, inst_idx, source, offset=0):
    x0 = 100 + stem_idx * 13 + inst_idx * 37 + offset
    y0 = 60 + method_idx * 11 + inst_idx * 29 + offset
    poly = [
        [x0, y0],
        [x0 + 20, y0],
        [x0 + 20, y0 + 15],
        [x0, y0 + 15],
    ]
    return {
        "class_name": class_name,
        "subtype": "curtaining" if class_name == "artifact" else None,
        "bbox": [x0, y0, x0 + 20, y0 + 15],
        "polygon": poly,
        "score": 0.4 + 0.01 * ((stem_idx + inst_idx) % 50),
        "source": source,
    }


def write_method_preds(
    preds_root: Path,
    method: str,
    stems,
    method_idx: int = 0,
    with_uncertainty: bool = True,
    duplicate_of: str | None = None,
) -> None:
    method_dir = preds_root / method
    method_dir.mkdir(parents=True, exist_ok=True)
    for s_idx, (stem, _batch, split) in enumerate(stems):
        if split == "train":
            continue
        arr = semantic_array(seed=s_idx * 100 + method_idx)
        Image.fromarray(arr, mode="L").save(method_dir / f"{stem}_semantic.png")
        if with_uncertainty:
            rng = np.random.default_rng(s_idx * 7 + method_idx)
            u = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
            Image.fromarray(u, mode="L").save(
                method_dir / f"{stem}_uncertainty.png"
            )
        instances = []
        for c_idx, class_name in enumerate(CLASSES):
            instances.append(
                _instance(class_name, s_idx, method_idx, c_idx, method)
            )
        instances.append(
            _instance("graphite_particle", s_idx, method_idx, 9, method)
        )
        if duplicate_of is not None:
            for c_idx, class_name in enumerate(CLASSES):
                instances.append(
                    _instance(class_name, s_idx, 0, c_idx, method)
                )
        (method_dir / f"{stem}_instances.json").write_text(json.dumps(instances))


@pytest.fixture
def synthetic(tmp_path):
    """Manifest + preds for classical_v1 and unet_pseudo_v1."""
    manifest = write_manifest(tmp_path / "manifest.csv")
    preds = tmp_path / "preds"
    write_method_preds(preds, "classical_v1", STEMS, method_idx=0)
    write_method_preds(
        preds, "unet_pseudo_v1", STEMS, method_idx=1, duplicate_of="classical_v1"
    )
    return {
        "manifest": manifest,
        "preds": preds,
        "out": tmp_path / "review",
        "config": dict(REVIEW_CONFIG),
    }


class FakeLoader:
    """Deterministic image loader; no filesystem access."""

    def __init__(self, h=H, w=W):
        self.h = h
        self.w = w

    def get(self, stem: str, view: str) -> np.ndarray:
        seed = sum(ord(c) for c in f"{stem}|{view}") % 255
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(self.h, self.w), dtype=np.uint8)
