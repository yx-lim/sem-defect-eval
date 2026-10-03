"""Read method predictions from ``$SEM_WORK_ROOT/preds/<method>/`` (spec §2.2)."""

from __future__ import annotations

import json
from dataclasses import fields
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from sem.qc.schema import Instance, Prediction

_INSTANCE_FIELDS = {f.name for f in fields(Instance)}


class PredictionError(RuntimeError):
    pass


def _instance(record: dict) -> Instance:
    return Instance(**{k: v for k, v in record.items() if k in _INSTANCE_FIELDS})


def read_instances(path: Path) -> list[Instance]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("instances", [])
    return [_instance(r) for r in data]


class PredictionStore:
    """Lazy per-stem reader with a small cache (full-res maps are ~16 MB each)."""

    def __init__(self, work_root: str | Path, method: str):
        self.method = method
        self.root = Path(work_root) / "preds" / method
        if not self.root.is_dir():
            raise PredictionError(f"No predictions for method {method!r} at {self.root}")
        self._load = lru_cache(maxsize=2)(self._load_uncached)

    def has(self, stem: str) -> bool:
        return (self.root / f"{stem}_semantic.png").is_file()

    def _load_uncached(self, stem: str) -> Prediction:
        path = self.root / f"{stem}_semantic.png"
        if not path.is_file():
            raise PredictionError(f"Missing prediction {path}")
        with Image.open(path) as image:
            semantic = np.array(image, dtype=np.uint8)
        if semantic.ndim != 2:
            raise PredictionError(f"{path} must be a single-channel label map")
        return Prediction(semantic, read_instances(self.root / f"{stem}_instances.json"))

    def get(self, stem: str) -> Prediction:
        return self._load(stem)
