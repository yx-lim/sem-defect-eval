"""PCA/UMAP visualization, evidence panels and report.md (no semantic labels)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from sem.qc.drift import stats  # noqa: E402
from sem.qc.drift.features import ALL_FEATURES  # noqa: E402

FORBIDDEN_WORDS = ("defective", "defect", "reject", "faulty", "bad batch", "caused by", "due to")
BATCH_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
MARKERS = {"ETD": "o", "SE": "^"}


def embedding_plot(stems: pd.DataFrame, path: Path, method: str = "pca") -> dict[str, Any]:
    """2-D embedding of pooled-standardized stem vectors (visualization only)."""
    z = stats.pooled_standardize(stems[list(ALL_FEATURES)].to_numpy(float))
    info: dict[str, Any] = {"method": method}
    if method == "pca":
        pca = PCA(n_components=2, random_state=0)
        xy = pca.fit_transform(z)
        info["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
        comp = pd.DataFrame(pca.components_.T, index=[c for c in ALL_FEATURES][: z.shape[1]] if z.shape[1] == len(ALL_FEATURES) else None,
                            columns=["PC1", "PC2"])
        info["top_loadings_pc1"] = comp["PC1"].abs().sort_values(ascending=False).head(4).index.tolist() if comp.index is not None else []
        labels = ("PC1 ({:.0%})".format(info["explained_variance_ratio"][0]),
                  "PC2 ({:.0%})".format(info["explained_variance_ratio"][1]))
    else:
        import umap  # optional

        xy = umap.UMAP(n_neighbors=min(10, len(z) - 1), random_state=0).fit_transform(z)
        labels = ("UMAP-1", "UMAP-2")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for bi, batch in enumerate(sorted(stems.batch.unique())):
        for det, marker in MARKERS.items():
            m = ((stems.batch == batch) & (stems.detector_set == det)).to_numpy()
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], c=BATCH_COLORS[bi % 5], marker=marker, s=55,
                           edgecolor="k", linewidth=0.5, label=f"{batch} / {det}")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(f"Stem vectors ({method.upper()}, visualization only)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return info


def evidence_panel(
    feature: str,
    inc_tiles: pd.DataFrame,
    ref_tiles: pd.DataFrame,
    ref_median: float,
    load_tile: Callable[[str, int, int, int], tuple[np.ndarray, np.ndarray | None]],
    path: Path,
    n_inc: int = 3,
    title_note: str = "",
) -> list[dict[str, Any]]:
    """3 incoming tiles farthest from the reference median + 1 typical reference tile."""
    inc = inc_tiles[np.isfinite(inc_tiles[feature])].copy()
    ref = ref_tiles[np.isfinite(ref_tiles[feature])].copy()
    if inc.empty or ref.empty:
        return []
    inc["dist"] = (inc[feature] - ref_median).abs()
    ref["dist"] = (ref[feature] - ref_median).abs()
    chosen = [("incoming", r) for _, r in inc.sort_values("dist", ascending=False).head(n_inc).iterrows()]
    chosen.append(("reference (typical)", ref.sort_values("dist").iloc[0]))
    fig, axes = plt.subplots(2, len(chosen), figsize=(3.4 * len(chosen), 7.2), squeeze=False)
    meta = []
    for col, (role, row) in enumerate(chosen):
        bse, overlay = load_tile(row["stem"], int(row["tile_x0"]), int(row["tile_y0"]), int(row["tile_px"]))
        axes[0, col].imshow(bse, cmap="gray", vmin=0, vmax=255)
        if overlay is not None:
            axes[1, col].imshow(overlay)
        for r in range(2):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])
        axes[0, col].set_title(
            f"{role}\n{row['batch']} {row['stem']} ({row['detector_set']})\n"
            f"x0={int(row['tile_x0'])} y0={int(row['tile_y0'])}\n{feature}={row[feature]:.4g}",
            fontsize=8,
        )
        meta.append({"role": role, "stem": row["stem"], "batch": row["batch"], "tile_x0": int(row["tile_x0"]),
                     "tile_y0": int(row["tile_y0"]), "value": float(row[feature])})
    axes[1, 0].set_ylabel("classical_v1 classes", fontsize=8)
    axes[0, 0].set_ylabel("BSE", fontsize=8)
    fig.suptitle(f"{feature}: reference stem median = {ref_median:.4g}{title_note}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return meta


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}g}"
    return str(v)


def kpi_table(kpi: pd.DataFrame) -> str:
    head = ("| family | feature | median ref | median inc | robust z | median diff [95% CI] | Hedges g | p raw | p Holm |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in kpi.itertuples():
        z = _fmt(r.robust_z) + (f" ({r.z_scale})" if r.z_scale != "mad" else "")
        lines.append(
            f"| {r.family} | {r.feature} | {_fmt(r.median_ref)} | {_fmt(r.median_inc)} | {z} | "
            f"{_fmt(r.median_diff)} [{_fmt(r.median_diff_ci_lo)}, {_fmt(r.median_diff_ci_hi)}] | "
            f"{_fmt(r.hedges_g)} | {_fmt(r.p_raw)} | {_fmt(r.p_holm)} |"
        )
    return head + "\n".join(lines) + "\n"


def check_wording(text: str) -> list[str]:
    low = text.lower()
    return [w for w in FORBIDDEN_WORDS if w in low]
