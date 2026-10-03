"""Tiling and per-tile / per-stem feature extraction for batch change detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi

from sem.qc.kpi import compute_kpis
from sem.qc.schema import IGNORE_LABEL, Instance, Prediction

MATERIAL_FEATURES = (
    "void_fraction",
    "void_fraction_incl_uncertain",
    "area_frac_bright_particle",
    "area_frac_graphite_particle",
    "bright_particle_ecd_median_um",
    "bright_particle_ecd_p90_um",
    "crack_density_um_per_mm2",
    "gap_density_um_per_mm2",
    "agglomerate_per_mm2",
)
COVARIATE_FEATURES = (
    "bse_mean",
    "bse_std",
    "inlens_mean",
    "inlens_std",
    "bse_focus",
    "bse_noise",
    "curtaining_score",
    "scan_streak_score",
    "edge_column_px",
)
ALL_FEATURES = MATERIAL_FEATURES + COVARIATE_FEATURES
ID_COLUMNS = ("stem", "batch", "detector_set")


def tile_grid(
    valid: np.ndarray, tile_px: int = 1024, min_valid_frac: float = 0.9
) -> list[tuple[int, int, float]]:
    """Non-overlapping tile origins (x0, y0, valid_frac) with valid_frac >= min_valid_frac."""
    h, w = valid.shape
    out = []
    for y0 in range(0, h - tile_px + 1, tile_px):
        for x0 in range(0, w - tile_px + 1, tile_px):
            frac = float(valid[y0 : y0 + tile_px, x0 : x0 + tile_px].mean())
            if frac >= min_valid_frac:
                out.append((x0, y0, frac))
    return out


def fft_curtain_score(img: np.ndarray, tile: int = 512, band: int = 2) -> float:
    """pipeline_v0 definition: mean per-subtile spectral energy ratio in the |fy|<=band row band."""
    h, w = img.shape
    ratios = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            t = img[y : y + tile, x : x + tile].astype(np.float32)
            spectrum = np.fft.fftshift(np.abs(np.fft.fft2(t)))
            cy, cx = tile // 2, tile // 2
            bandmask = np.zeros_like(spectrum, dtype=bool)
            bandmask[cy - band : cy + band + 1, :] = True
            bandmask[cy, cx] = False
            total = spectrum.sum() - spectrum[cy, cx]
            ratios.append(float(spectrum[bandmask].sum() / total) if total > 0 else 0.0)
    return float(np.mean(ratios)) if ratios else 0.0


def acquisition_covariates(
    bse: np.ndarray,
    inlens: np.ndarray,
    valid: np.ndarray,
    edge_column_px: float,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Image-statistics covariates on one tile (valid pixels only where pointwise)."""
    b = bse.astype(np.float64)
    i = inlens.astype(np.float64)
    bv = b[valid]
    var_b = float(bv.var())
    lap = ndi.laplace(b)
    interior = ndi.binary_erosion(valid, iterations=1)
    focus = float(lap[interior].var() / var_b) if var_b > 0 and interior.any() else float("nan")
    resid = (b - ndi.gaussian_filter(b, sigma=float(cfg["noise_gaussian_sigma"])))[interior]
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) if resid.size else float("nan")
    highpass = b - ndi.gaussian_filter(b, sigma=float(cfg["scan_streak_highpass_sigma"]))
    hp = np.where(valid, highpass, np.nan)
    row_means = np.nanmean(hp, axis=1)
    std_b = float(bv.std())
    streak = float(np.nanstd(row_means) / std_b) if std_b > 0 else float("nan")
    return {
        "bse_mean": float(bv.mean()),
        "bse_std": std_b,
        "inlens_mean": float(i[valid].mean()),
        "inlens_std": float(i[valid].std()),
        "bse_focus": focus,
        "bse_noise": noise,
        "curtaining_score": fft_curtain_score(
            bse, tile=int(cfg["curtain_subtile_px"]), band=int(cfg["curtain_band"])
        ),
        "scan_streak_score": streak,
        "edge_column_px": float(edge_column_px),
    }


def count_edge_columns(rgb_or_gray: np.ndarray) -> int:
    """Number of image columns that contain any non-gray (channel-differing) pixel."""
    if rgb_or_gray.ndim != 3:
        return 0
    colored = np.any(rgb_or_gray[..., 1:] != rgb_or_gray[..., :1], axis=-1)
    return int(np.count_nonzero(colored.any(axis=0)))


def _shift_instance(inst: Instance, x0: int, y0: int, size: int) -> Instance | None:
    bx0, by0, bx1, by1 = inst.bbox
    if bx1 < x0 or by1 < y0 or bx0 >= x0 + size or by0 >= y0 + size:
        return None
    return Instance(
        class_name=inst.class_name,
        bbox=[bx0 - x0, by0 - y0, bx1 - x0, by1 - y0],
        polygon=[[px - x0, py - y0] for px, py in inst.polygon],
        subtype=inst.subtype,
        score=inst.score,
        source=inst.source,
    )


def material_features(
    semantic: np.ndarray, instances: list[Instance], valid: np.ndarray, pixel_size_nm: float
) -> dict[str, float]:
    k = compute_kpis(semantic, instances, valid, pixel_size_nm)
    return {name: (np.nan if k.get(name) is None else float(k[name])) for name in MATERIAL_FEATURES}


def tile_features(
    stem: str,
    batch: str,
    detector_set: str,
    bse: np.ndarray,
    inlens: np.ndarray,
    valid: np.ndarray,
    prediction: Prediction,
    pixel_size_nm: float,
    edge_column_px: float,
    cfg: dict[str, Any],
    tile_px: int,
) -> list[dict[str, Any]]:
    """One row per accepted tile: ids, tile geometry, material and covariate features."""
    rows = []
    semantic = prediction.semantic
    agglomerates = [i for i in prediction.instances if i.class_name == "agglomerate"]
    for x0, y0, frac in tile_grid(valid, tile_px, float(cfg["min_valid_frac"])):
        sl = (slice(y0, y0 + tile_px), slice(x0, x0 + tile_px))
        tv = valid[sl] & (semantic[sl] != IGNORE_LABEL)
        sem_tile = np.where(tv, semantic[sl], IGNORE_LABEL).astype(np.uint8)
        inst = [s for s in (_shift_instance(a, x0, y0, tile_px) for a in agglomerates) if s]
        row: dict[str, Any] = {
            "stem": stem, "batch": batch, "detector_set": detector_set,
            "tile_x0": x0, "tile_y0": y0, "tile_px": tile_px, "valid_frac": frac,
        }
        try:
            row.update(material_features(sem_tile, inst, tv, pixel_size_nm))
        except ValueError:
            row.update({name: np.nan for name in MATERIAL_FEATURES})
        row.update(acquisition_covariates(bse[sl], inlens[sl], valid[sl], edge_column_px, cfg))
        rows.append(row)
    return rows


def stem_vectors(tiles: pd.DataFrame) -> pd.DataFrame:
    """Stem vector = median over the stem's tiles (NaN-aware)."""
    grouped = tiles.groupby(list(ID_COLUMNS), sort=False)
    med = grouped[list(ALL_FEATURES)].median()
    med.insert(0, "n_tiles", grouped.size())
    return med.reset_index()


def load_prediction(pred_dir: str | Path, stem: str) -> Prediction:
    """Read `<stem>_semantic.png`, `<stem>_instances.json`, optional `<stem>_uncertainty.png`."""
    pred_dir = Path(pred_dir)
    Image.MAX_IMAGE_PIXELS = None
    semantic = np.asarray(Image.open(pred_dir / f"{stem}_semantic.png"), dtype=np.uint8)
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    instances: list[Instance] = []
    inst_path = pred_dir / f"{stem}_instances.json"
    if inst_path.exists():
        raw = json.loads(inst_path.read_text())
        if isinstance(raw, dict):
            raw = raw.get("instances", [])
        allowed = set(Instance.__dataclass_fields__)
        instances = [Instance(**{k: v for k, v in r.items() if k in allowed}) for r in raw]
    return Prediction(semantic=np.ascontiguousarray(semantic), instances=instances)


def read_edge_columns(path: str | Path) -> int:
    return count_edge_columns(tifffile.imread(path))


def records_by_batch(records: Iterable, batches: Iterable[str]) -> list:
    wanted = set(batches)
    return [r for r in records if r.batch in wanted]
