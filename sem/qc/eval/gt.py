"""Ground-truth loading and split validation for evaluation (spec §2.3, §3, §4).

GT is selected exclusively with ``sem.qc.schema.is_ground_truth``. Any human-reviewed
item on a train stem, on a stem missing from the manifest, or with a split that
disagrees with the manifest raises ``GroundTruthError``.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import warnings

import numpy as np
from PIL import Image

from sem.qc.schema import (
    CLASS_IDS,
    GROUND_TRUTH_STATUSES,
    IGNORE_LABEL,
    is_ground_truth,
    read_jsonl,
    resolve_review_items,
)
from sem.qc.split import assert_frozen

EVAL_SPLITS = ("val", "test")
DECIDED_STATUSES = GROUND_TRUTH_STATUSES | {"rejected", "uncertain"}


class GroundTruthError(RuntimeError):
    """Raised when GT violates split rules or cannot be loaded unambiguously."""


@dataclass(frozen=True)
class Manifest:
    path: Path
    sha256: str
    frozen_sha256: str | None
    rows: dict[str, dict[str, str]]

    @property
    def frozen(self) -> bool:
        return self.frozen_sha256 is not None

    def split_of(self, stem: str) -> str:
        if stem not in self.rows:
            raise GroundTruthError(f"Stem {stem!r} is not in manifest {self.path}")
        return self.rows[stem]["split"]


@dataclass
class TileGT:
    item_id: str
    stem: str
    batch: str
    split: str
    x0: int
    y0: int
    w: int
    h: int
    sampling_method: str
    weight: float
    status: str
    label: np.ndarray
    # Full-resolution polygons; None means agglomerates were not annotated on this tile.
    agglomerates: list[list[list[float]]] | None


@dataclass
class CandidateDecision:
    item_id: str
    stem: str
    batch: str
    split: str
    method: str
    source: str
    class_name: str
    status: str
    human_class: str | None
    sampling_method: str
    weight: float
    is_ground_truth: bool

    @property
    def outcome(self) -> str:
        """'positive', 'negative' or 'uncertain' for the precision formula."""
        if self.status == "uncertain":
            return "uncertain"
        if self.status == "rejected":
            return "negative"
        if self.status in ("relabeled", "redrawn") and self.human_class not in (
            None,
            self.class_name,
        ):
            return "negative"
        return "positive"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path: str | Path, frozen_sha256: str | None) -> Manifest:
    """Read the split manifest and assert its hash when a frozen hash is configured."""
    path = Path(path)
    if not path.is_file():
        raise GroundTruthError(f"Split manifest not found: {path}")
    digest = file_sha256(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # unfrozen: reported by eval
            assert_frozen(path, frozen_sha256)
    except AssertionError as exc:
        raise GroundTruthError(
            f"Manifest hash check failed for {path}: {exc}. "
            "The split must not change after approval."
        ) from exc
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["stem"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise GroundTruthError(f"Manifest {path} has no rows")
    bad = {r["split"] for r in rows.values()} - {"train", "val", "test"}
    if bad:
        raise GroundTruthError(f"Manifest {path} has unknown splits {sorted(bad)}")
    return Manifest(path, digest, frozen_sha256, rows)


def _human(item: dict[str, Any]) -> dict[str, Any]:
    human = item.get("human") or {}
    if not isinstance(human, dict):
        human = dict(vars(human))
    return human


def load_reviewed_items(review_dir: str | Path, manifest: Manifest) -> list[dict[str, Any]]:
    """Resolve items+decisions and validate every human-decided item against the split."""
    review_dir = Path(review_dir)
    items_path = review_dir / "items.jsonl"
    if not items_path.is_file():
        raise GroundTruthError(f"Review items not found: {items_path}")
    items = read_jsonl(items_path)
    decisions_path = review_dir / "decisions.jsonl"
    decisions = read_jsonl(decisions_path) if decisions_path.is_file() else []
    known = {item["item_id"] for item in items}
    orphan = sorted({d["item_id"] for d in decisions} - known)
    if orphan:
        raise GroundTruthError(f"Decisions reference unknown item_ids: {orphan[:5]}")
    resolved = resolve_review_items(items, decisions)
    reviewed = []
    for item in resolved:
        status = _human(item).get("status")
        if status is None:
            continue
        if status not in DECIDED_STATUSES:
            raise GroundTruthError(f"Item {item['item_id']} has unknown status {status!r}")
        _validate_split(item, manifest)
        _validate_reviewer(item)
        reviewed.append(item)
    return reviewed


def _validate_split(item: dict[str, Any], manifest: Manifest) -> None:
    stem = item["stem"]
    manifest_split = manifest.split_of(stem)
    if manifest_split == "train":
        raise GroundTruthError(
            f"Human-reviewed item {item['item_id']} is on TRAIN stem {stem!r}; "
            "GT may only come from val/test stems."
        )
    if item.get("split") != manifest_split:
        raise GroundTruthError(
            f"Item {item['item_id']} split {item.get('split')!r} disagrees with "
            f"manifest split {manifest_split!r} for stem {stem!r}"
        )


def _validate_reviewer(item: dict[str, Any]) -> None:
    """Model/VLM outputs are never GT: reject decisions attributed to a model id."""
    reviewer = _human(item).get("reviewer_id")
    if reviewer is None:
        return
    model_ids = {
        (item.get("vlm_suggestion") or {}).get("model_id"),
        (item.get("proposal") or {}).get("source"),
    } - {None, ""}
    if reviewer in model_ids:
        raise GroundTruthError(
            f"Item {item['item_id']} decision reviewer_id {reviewer!r} is a model "
            "identifier; model outputs are never ground truth."
        )


def _resolve_path(path: str, review_dir: Path, work_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for base in (review_dir, work_root):
        if (base / candidate).is_file():
            return base / candidate
    return review_dir / candidate


def _read_label_png(path: Path, shape: tuple[int, int], item_id: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("L", "P"):
            raise GroundTruthError(f"Mask {path} must be single-channel, got {image.mode}")
        label = np.array(image, dtype=np.uint8)
    if label.shape != shape:
        raise GroundTruthError(
            f"Mask {path} for item {item_id} has shape {label.shape}, tile is {shape}"
        )
    bad = set(np.unique(label).tolist()) - set(range(8)) - {IGNORE_LABEL}
    if bad:
        raise GroundTruthError(f"Mask {path} contains invalid labels {sorted(bad)}")
    return label


def tile_label_path(item: dict[str, Any], review_dir: Path, work_root: Path) -> Path:
    """Mask precedence: review/masks/<item_id>.png, human.semantic_png, and for
    status 'accepted' only, the pre-filled proposal.semantic_png."""
    store = review_dir / "masks" / f"{item['item_id']}.png"
    if store.is_file():
        return store
    human = _human(item)
    if human.get("semantic_png"):
        return _resolve_path(human["semantic_png"], review_dir, work_root)
    proposal_png = (item.get("proposal") or {}).get("semantic_png")
    if human.get("status") == "accepted" and proposal_png:
        return _resolve_path(proposal_png, review_dir, work_root)
    raise GroundTruthError(
        f"GT tile {item['item_id']} (status {human.get('status')}) has no label mask"
    )


def _agglomerates(human: dict[str, Any]) -> list[list[list[float]]] | None:
    polygons = human.get("polygons")
    if polygons is None:
        return None
    return [
        [list(map(float, pt)) for pt in poly["points"]]
        for poly in polygons
        if poly.get("class_name") == "agglomerate"
    ]


def _sampling(item: dict[str, Any]) -> tuple[str, float]:
    sampling = item.get("sampling") or {}
    method = sampling.get("method")
    if method not in ("random", "uncertainty"):
        raise GroundTruthError(f"Item {item['item_id']} has sampling.method {method!r}")
    weight = float(sampling.get("weight", 1.0))
    if not np.isfinite(weight) or weight <= 0:
        raise GroundTruthError(f"Item {item['item_id']} has invalid weight {weight}")
    return method, weight


def candidate_method(proposal: dict[str, Any]) -> str:
    """Method of a candidate: proposal['method'] if present, else the source prefix
    before the first ':' or '/' (e.g. 'classical_v1:thin' -> 'classical_v1')."""
    if proposal.get("method"):
        return str(proposal["method"])
    source = str(proposal.get("source", ""))
    for sep in (":", "/"):
        source = source.split(sep, 1)[0]
    return source


def load_ground_truth(
    review_dir: str | Path,
    work_root: str | Path,
    manifest: Manifest,
) -> tuple[list[TileGT], list[CandidateDecision]]:
    """Return GT exhaustive tiles and human-decided candidates (all eval splits)."""
    review_dir, work_root = Path(review_dir), Path(work_root)
    tiles: list[TileGT] = []
    candidates: list[CandidateDecision] = []
    for item in load_reviewed_items(review_dir, manifest):
        human = _human(item)
        method, weight = _sampling(item)
        if item["kind"] == "exhaustive_tile":
            if not is_ground_truth(item):
                continue
            tile = item["tile"]
            shape = (int(tile["h"]), int(tile["w"]))
            label = _read_label_png(
                tile_label_path(item, review_dir, work_root), shape, item["item_id"]
            )
            tiles.append(
                TileGT(
                    item_id=item["item_id"], stem=item["stem"], batch=item["batch"],
                    split=item["split"], x0=int(tile["x0"]), y0=int(tile["y0"]),
                    w=shape[1], h=shape[0], sampling_method=method, weight=weight,
                    status=human["status"], label=label,
                    agglomerates=_agglomerates(human),
                )
            )
        elif item["kind"] == "candidate":
            proposal = item.get("proposal") or {}
            class_name = proposal.get("class_name")
            if class_name not in (*CLASS_IDS, "agglomerate"):
                raise GroundTruthError(
                    f"Candidate {item['item_id']} has unknown class {class_name!r}"
                )
            candidates.append(
                CandidateDecision(
                    item_id=item["item_id"], stem=item["stem"], batch=item["batch"],
                    split=item["split"], method=candidate_method(proposal),
                    source=str(proposal.get("source", "")), class_name=class_name,
                    status=human["status"], human_class=human.get("class_name"),
                    sampling_method=method, weight=weight,
                    is_ground_truth=is_ground_truth(item),
                )
            )
        else:
            raise GroundTruthError(f"Item {item['item_id']} has unknown kind {item['kind']!r}")
    return tiles, candidates
