"""Shared class, prediction, and human-review data models."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


CLASS_NAMES = {
    0: "matrix_other",
    1: "graphite_particle",
    2: "bright_particle",
    3: "pore",
    4: "subsurface_uncertain",
    5: "crack_intraparticle",
    6: "interparticle_gap",
    7: "artifact",
}
CLASS_IDS = {name: class_id for class_id, name in CLASS_NAMES.items()}
IGNORE_LABEL = 255
INSTANCE_CLASS_NAMES = frozenset((*CLASS_IDS, "agglomerate"))
ARTIFACT_SUBTYPES = frozenset(
    {"curtaining", "scan_streak", "charging", "redeposition", "edge_column", "other"}
)
GROUND_TRUTH_STATUSES = frozenset({"accepted", "relabeled", "redrawn"})


def make_item_id(
    stem: str,
    kind: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
    source: str,
) -> str:
    """Create the stable review identifier specified by the QC data contract."""
    value = f"{stem}|{kind}|{x0},{y0},{width},{height}|{source}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


@dataclass
class Instance:
    class_name: str
    bbox: list[float]
    polygon: list[list[float]]
    subtype: str | None = None
    score: float = 1.0
    source: str = ""


@dataclass
class Prediction:
    semantic: np.ndarray
    instances: list[Instance]
    uncertainty: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.semantic.dtype != np.uint8 or self.semantic.ndim != 2:
            raise ValueError("Prediction.semantic must be a 2D uint8 array")
        if self.uncertainty is not None:
            if self.uncertainty.shape != self.semantic.shape:
                raise ValueError("Uncertainty and semantic maps must have equal shapes")
            if not np.isfinite(self.uncertainty).all() or np.any(
                (self.uncertainty < 0) | (self.uncertainty > 1)
            ):
                raise ValueError("Uncertainty values must be finite and in [0, 1]")


@runtime_checkable
class Method(Protocol):
    name: str

    def predict(
        self, views: dict[str, np.ndarray], valid: np.ndarray
    ) -> Prediction: ...


HumanStatus = Literal[
    "accepted", "rejected", "relabeled", "redrawn", "uncertain"
]


@dataclass
class HumanDecision:
    status: HumanStatus | None = None
    class_name: str | None = None
    subtype: str | None = None
    polygons: list[dict[str, Any]] | None = None
    semantic_png: str | None = None
    notes: str | None = None
    reviewer_id: str | None = None
    timestamp: str | None = None


@dataclass
class ReviewItem:
    item_id: str
    stem: str
    batch: str
    split: str
    kind: Literal["exhaustive_tile", "candidate"]
    tile: dict[str, int]
    sampling: dict[str, Any]
    proposal: dict[str, Any]
    vlm_suggestion: dict[str, Any] | None = None
    human: HumanDecision | dict[str, Any] | None = None


def is_ground_truth(item: ReviewItem | dict[str, Any]) -> bool:
    """Return true only for accepted or explicitly human-corrected items."""
    human = item.human if isinstance(item, ReviewItem) else item.get("human")
    status = (
        human.status if isinstance(human, HumanDecision) else (human or {}).get("status")
    )
    return status in GROUND_TRUTH_STATUSES


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON Lines, ignoring blank lines."""
    records = []
    with Path(path).open(encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def write_jsonl(path: str | Path, records: list[Any]) -> None:
    """Write records as UTF-8 JSON Lines, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(
                json.dumps(record, default=_to_jsonable, sort_keys=True) + "\n"
            )


def latest_decisions(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Resolve append-only decisions so the last row per item_id wins."""
    resolved = {}
    for decision in decisions:
        if "item_id" not in decision:
            raise ValueError("Every decision must contain item_id")
        resolved[decision["item_id"]] = decision.get("human", decision)
    return resolved


def resolve_review_items(
    items: list[ReviewItem | dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> list[ReviewItem | dict[str, Any]]:
    """Apply each item's latest human decision without mutating proposal data."""
    by_item_id = latest_decisions(decisions)
    resolved = []
    for item in items:
        item_id = item.item_id if isinstance(item, ReviewItem) else item["item_id"]
        if item_id not in by_item_id:
            resolved.append(item)
        elif isinstance(item, ReviewItem):
            resolved.append(replace(item, human=by_item_id[item_id]))
        else:
            resolved.append({**item, "human": by_item_id[item_id]})
    return resolved
