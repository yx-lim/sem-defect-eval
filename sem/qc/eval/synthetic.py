"""Synthetic study generator for tests and the end-to-end demo.

Everything here is FAKE: synthetic label maps, BSE-like images rendered from them,
a synthetic split manifest, fixture "human" decisions (reviewer_id
'synthetic_fixture') and degraded copies of the GT posing as method predictions.
Nothing produced here is a real measurement or real ground truth.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.draw import disk, ellipse, line

from sem.qc.eval.metrics import components
from sem.qc.schema import CLASS_NAMES, IGNORE_LABEL, make_item_id, write_jsonl

INTENSITY = {0: 120, 1: 80, 2: 225, 3: 18, 4: 55, 5: 30, 6: 35, 7: 170}
REVIEWER = "synthetic_fixture"
BORDER = 8


@dataclass
class SyntheticStem:
    stem: str
    batch: str
    split: str
    label: np.ndarray
    agglomerates: list[list[list[float]]]


def make_label_map(shape: tuple[int, int], rng: np.random.Generator):
    h, w = shape
    label = np.zeros(shape, dtype=np.uint8)
    density = h * w / (512 * 512)
    for _ in range(int(14 * density)):  # graphite flakes
        rr, cc = ellipse(rng.integers(0, h), rng.integers(0, w), rng.integers(12, 30),
                         rng.integers(30, 80), shape=shape, rotation=rng.uniform(0, np.pi))
        label[rr, cc] = 1
    flakes = components(label == 1, 200)
    for flake in flakes[: int(6 * density)]:  # intraparticle cracks inside flakes
        ys, xs = np.nonzero(ndi.binary_erosion(flake, iterations=4))
        if len(ys) < 2:
            continue
        i, j = rng.choice(len(ys), 2, replace=False)
        rr, cc = line(ys[i], xs[i], ys[j], xs[j])
        keep = flake[rr, cc]
        label[rr[keep], cc[keep]] = 5
    edge = (label == 1) & ~ndi.binary_erosion(label == 1, iterations=1)
    gap_lab, n = ndi.label(edge, structure=np.ones((3, 3), bool))
    for k in rng.choice(np.arange(1, n + 1), size=min(n, int(4 * density)), replace=False):
        comp = gap_lab == k
        ys, xs = np.nonzero(comp)
        sel = np.argsort(xs)[: max(10, len(xs) // 3)]
        label[ys[sel], xs[sel]] = 6
    for _ in range(int(8 * density)):  # pores, some with sub-surface material inside
        cy, cx, r = rng.integers(20, h - 20), rng.integers(20, w - 20), rng.integers(6, 16)
        rr, cc = disk((cy, cx), r, shape=shape)
        label[rr, cc] = 3
        if rng.random() < 0.4:
            rr, cc = disk((cy, cx), max(2, r // 2), shape=shape)
            label[rr, cc] = 4
    agglomerates = []
    for _ in range(int(16 * density)):  # bright particles
        cy, cx = rng.integers(15, h - 15), rng.integers(15, w - 15)
        rr, cc = disk((cy, cx), rng.integers(3, 8), shape=shape)
        label[rr, cc] = 2
    for _ in range(max(1, int(1.5 * density))):  # agglomerates: clusters of 3 particles
        cy, cx = rng.integers(40, h - 40), rng.integers(40, w - 40)
        for dy, dx in ((0, 0), (0, 12), (11, 6)):
            rr, cc = disk((cy + dy, cx + dx), 5, shape=shape)
            label[rr, cc] = 2
        agglomerates.append([[cx - 6, cy - 6], [cx + 18, cy - 6], [cx + 12, cy + 17],
                             [cx, cy + 17]])
    label[:, BORDER : BORDER + 3] = 7  # edge column artifact
    label[:BORDER, :] = IGNORE_LABEL
    label[-BORDER:, :] = IGNORE_LABEL
    label[:, :BORDER] = IGNORE_LABEL
    label[:, -BORDER:] = IGNORE_LABEL
    return label, [[list(map(float, p)) for p in poly] for poly in agglomerates]


def render_bse(label: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = np.vectorize(lambda v: INTENSITY.get(int(v), 0))(label).astype(float)
    image = ndi.gaussian_filter(base, 0.8) + rng.normal(0, 8, label.shape)
    return image.clip(0, 255).astype(np.uint8)


def degrade(label: np.ndarray, agglomerates, severity: float, rng: np.random.Generator):
    """A 'classical_v1-style' prediction: GT with systematic and random errors."""
    pred = label.copy()
    shift = int(round(2 * severity))
    bright = label == 2
    pred[bright] = 0
    pred[np.roll(bright, (shift, shift), axis=(0, 1))] = 2
    pores = label == 3
    if shift:
        pred[pores & ~ndi.binary_erosion(pores, iterations=shift)] = 0
    sub = label == 4
    lab, n = ndi.label(sub)
    for k in range(1, n + 1):
        if rng.random() < 0.6 * severity:
            pred[lab == k] = 3  # sub-surface region called pore
    for cls in (5, 6):
        lab, n = ndi.label(label == cls, structure=np.ones((3, 3), bool))
        for k in range(1, n + 1):
            if rng.random() < 0.5 * severity:
                pred[lab == k] = 1 if cls == 5 else 0
    h, w = label.shape
    for _ in range(int(10 * severity * h * w / 512**2)):  # spurious pores
        rr, cc = disk((rng.integers(0, h), rng.integers(0, w)), rng.integers(3, 9), shape=label.shape)
        pred[rr, cc] = 3
    for _ in range(int(4 * severity * h * w / 512**2)):  # spurious cracks
        y, x = rng.integers(20, h - 20), rng.integers(20, w - 40)
        rr, cc = line(y, x, y + rng.integers(-10, 10), x + 30)
        pred[rr, cc] = 5
    pred[label == IGNORE_LABEL] = IGNORE_LABEL
    pred_agg = []
    for poly in agglomerates:
        if rng.random() < 0.3 * severity:
            continue
        pred_agg.append([[float(p[0] + 2 * shift), float(p[1] + 2 * shift)] for p in poly])
    if severity > 0:
        cy, cx = float(rng.integers(60, h - 60)), float(rng.integers(60, w - 60))
        pred_agg.append([[cx, cy], [cx + 25, cy], [cx + 25, cy + 25], [cx, cy + 25]])
    return pred, pred_agg


def _instances_json(polys, source):
    out = []
    for poly in polys:
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        out.append({"class_name": "agglomerate", "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "polygon": poly, "subtype": None, "score": 0.8, "source": source})
    return out


def _write_tif(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, image, resolution=(1_016_000, 1_016_000), resolutionunit="INCH")


def _candidate_decision(mask: np.ndarray, cls: int, label: np.ndarray, rng) -> dict:
    """Fixture reviewer: accept if mask mostly lies on GT of the same class."""
    if rng.random() < 0.08:
        return {"status": "uncertain"}
    values = label[mask]
    values = values[values != IGNORE_LABEL]
    if values.size == 0:
        return {"status": "uncertain"}
    counts = np.bincount(values, minlength=8)
    if counts[cls] / values.size >= 0.5:
        return {"status": "accepted"}
    other = int(np.argmax(counts))
    if other != 0 and counts[other] / values.size >= 0.5:
        return {"status": "relabeled", "class_name": CLASS_NAMES[other]}
    return {"status": "rejected"}


def make_study(
    root: str | Path,
    methods: dict[str, float],
    n_stems: dict[str, int] | None = None,
    shape: tuple[int, int] = (1024, 1024),
    tile: int = 512,
    seed: int = 0,
    candidates_per_stem: int = 12,
) -> dict[str, Path]:
    """Write a complete fake study under ``root``; returns its paths."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    n_stems = n_stems or {"test": 4, "val": 2, "train": 2}
    data_root, work_root = root / "data", root / "work"
    review = work_root / "review"
    (review / "masks").mkdir(parents=True, exist_ok=True)
    stems: list[SyntheticStem] = []
    k = 0
    for split, n in n_stems.items():
        for _ in range(n):
            label, agg = make_label_map(shape, rng)
            stems.append(SyntheticStem(f"syn{k:02d}", f"Batch_{1 + k % 3}", split, label, agg))
            k += 1

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stem", "group_id", "batch", "detector_set", "split", "views",
                         "height", "width", "pixel_size_nm", "n_nongray_px", "files_sha256"])
        for s in stems:
            writer.writerow([s.stem, s.stem, s.batch, "ETD", s.split, "BSE;Inlens;ETD",
                             shape[0], shape[1], 25.0, 0, ""])

    items, decisions = [], []
    for s in stems:
        bse = render_bse(s.label, rng)
        for view in ("BSE", "Inlens", "ETD"):
            _write_tif(data_root / s.batch / f"img_{s.stem}_{view}.tif", bse)
        preds = {}
        for method, severity in methods.items():
            pred, pred_agg = degrade(s.label, s.agglomerates, severity, rng)
            out = work_root / "preds" / method
            out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred).save(out / f"{s.stem}_semantic.png")
            (out / f"{s.stem}_instances.json").write_text(
                json.dumps(_instances_json(pred_agg, method)), encoding="utf-8")
            preds[method] = pred
        if s.split == "train":
            continue
        positions = [(x, y) for y in range(0, shape[0] - tile + 1, tile)
                     for x in range(0, shape[1] - tile + 1, tile)]
        for t_index, (x0, y0) in enumerate(positions):
            sampling = "uncertainty" if t_index == len(positions) - 1 else "random"
            item_id = make_item_id(s.stem, "exhaustive_tile", x0, y0, tile, tile, "classical_v1")
            crop = s.label[y0 : y0 + tile, x0 : x0 + tile]
            prefill = review / "prefill" / f"{item_id}.png"
            prefill.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop).save(prefill)
            items.append({
                "item_id": item_id, "stem": s.stem, "batch": s.batch, "split": s.split,
                "kind": "exhaustive_tile", "tile": {"x0": x0, "y0": y0, "w": tile, "h": tile},
                "sampling": {"stratum": f"{s.batch}/ETD", "method": sampling, "weight": 1.0},
                "proposal": {"source": "classical_v1", "class_name": None, "subtype": None,
                             "polygon": None, "semantic_png": f"prefill/{item_id}.png",
                             "score": None, "uncertainty": None},
                "vlm_suggestion": None, "human": None,
            })
            polys = [{"class_name": "agglomerate", "subtype": None, "points": p}
                     for p in s.agglomerates
                     if x0 <= p[0][0] < x0 + tile and y0 <= p[0][1] < y0 + tile]
            if t_index % 2 == 0:
                human = {"status": "accepted", "polygons": polys}
            else:
                Image.fromarray(crop).save(review / "masks" / f"{item_id}.png")
                human = {"status": "redrawn", "polygons": polys,
                         "semantic_png": f"masks/{item_id}.png"}
            human.update({"reviewer_id": REVIEWER, "timestamp": "2026-10-03T00:00:00Z",
                          "notes": "synthetic fixture"})
            decisions.append({"item_id": item_id, "human": human})
        for method, pred in preds.items():
            pool = []
            for cls in (3, 5, 6):
                pool += [(cls, m) for m in components(pred == cls, 4)]
            if not pool:
                continue
            sizes = {cls: sum(1 for c, _ in pool if c == cls) for cls in (3, 5, 6)}
            chosen = rng.choice(len(pool), size=min(candidates_per_stem, len(pool)), replace=False)
            for n_c, idx in enumerate(chosen):
                cls, mask = pool[idx]
                ys, xs = np.nonzero(mask)
                x0, y0 = int(xs.min()), int(ys.min())
                w, h = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
                sampling = "uncertainty" if n_c % 4 == 3 else "random"
                source = f"{method}:{'thin' if cls in (5, 6) else 'dark'}"
                item_id = make_item_id(s.stem, "candidate", x0, y0, w, h, source)
                weight = float(sizes[cls]) / max(1, candidates_per_stem) if sampling == "random" else 1.0
                items.append({
                    "item_id": item_id, "stem": s.stem, "batch": s.batch, "split": s.split,
                    "kind": "candidate", "tile": {"x0": x0, "y0": y0, "w": w, "h": h},
                    "sampling": {"stratum": CLASS_NAMES[cls], "method": sampling,
                                 "weight": weight},
                    "proposal": {"source": source, "class_name": CLASS_NAMES[cls],
                                 "subtype": None, "polygon": None, "semantic_png": None,
                                 "score": 0.5, "uncertainty": None},
                    "vlm_suggestion": None, "human": None,
                })
                human = _candidate_decision(mask, cls, s.label, rng)
                human.update({"reviewer_id": REVIEWER, "timestamp": "2026-10-03T00:00:00Z"})
                decisions.append({"item_id": item_id, "human": human})
    write_jsonl(review / "items.jsonl", items)
    write_jsonl(review / "decisions.jsonl", decisions)
    return {"root": root, "data_root": data_root, "work_root": work_root,
            "manifest": manifest, "review": review}
