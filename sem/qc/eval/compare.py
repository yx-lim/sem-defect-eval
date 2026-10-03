"""Cross-method results table (markdown + csv) from per-method metrics.json."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from sem.qc.eval.report import fmt

PIXEL_METRICS = ("f1", "iou", "precision", "recall")
OBJECT_METRICS = ("f1", "precision", "recall")
KPI_METRICS = ("bias", "mae", "rel_error", "spearman")


def run_dir(eval_root: Path, method: str, split: str, stratum: str, headline: tuple[str, str]) -> Path:
    base = eval_root / method
    return base if (split, stratum) == headline else base / f"{split}_{stratum}"


def _cell(value, low, high) -> dict[str, Any]:
    return {"value": value, "ci_low": low, "ci_high": high}


def long_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    method, rows = metrics["method"], []
    n_tiles, n_stems = metrics["counts"]["n_tiles"], metrics["counts"]["n_stems"]
    for r in metrics["pixel"]:
        for m in PIXEL_METRICS:
            rows.append({"method": method, "family": "pixel", "target": r["class"], "metric": m,
                         **r[m], "n_tiles": n_tiles, "n_stems": n_stems,
                         "n": r["n_gt_objects"], "insufficient_n": r["insufficient_n"]})
    for r in metrics["object"]:
        for m in OBJECT_METRICS:
            rows.append({"method": method, "family": "object", "target": r["class"], "metric": m,
                         **r[m], "n_tiles": r["n_tiles"], "n_stems": r["n_stems"],
                         "n": r["n_gt_objects"], "insufficient_n": r["insufficient_n"]})
    for r in metrics["candidate"]:
        rows.append({"method": method, "family": f"candidate[{r['sampling']}]",
                     "target": f"{r['source']}:{r['class']}", "metric": "precision",
                     **_cell(r["precision"], r["wilson_low"], r["wilson_high"]),
                     "n_tiles": None, "n_stems": r["n_stems"],
                     "n": r["n_positive"] + r["n_negative"], "insufficient_n": r["insufficient_n"]})
    for r in metrics["kpi"]:
        for m in KPI_METRICS:
            rows.append({"method": method, "family": f"kpi[{r['level']}]", "target": r["kpi"],
                         "metric": m, **{k: r[m][k] for k in ("value", "ci_low", "ci_high")},
                         "n_tiles": n_tiles, "n_stems": r["n_stems"], "n": r["n_pairs"],
                         "insufficient_n": r["insufficient_n"]})
    return rows


def compare(metrics_files: list[Path], out_dir: Path) -> tuple[Path, Path]:
    all_metrics = [json.loads(p.read_text(encoding="utf-8")) for p in metrics_files]
    if not all_metrics:
        raise FileNotFoundError("No metrics.json files to compare")
    runs = {(m["split"], m["stratum"]) for m in all_metrics}
    if len(runs) != 1:
        raise ValueError(f"Refusing to compare different split/stratum runs: {sorted(runs)}")
    rows = [row for m in all_metrics for row in long_rows(m)]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["family", "target", "metric", "method", "value", "ci_low", "ci_high",
                  "n_tiles", "n_stems", "n", "insufficient_n"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("NaN" if row.get(k) is None and k in ("value", "ci_low", "ci_high")
                                 else row.get(k)) for k in fields})

    methods = [m["method"] for m in all_metrics]
    keys: list[tuple[str, str, str]] = []
    cells: dict[tuple[str, str, str, str], str] = {}
    for row in rows:
        key = (row["family"], row["target"], row["metric"])
        if key not in keys:
            keys.append(key)
        flag = " ⚠ insufficient n" if row["insufficient_n"] else ""
        cells[(*key, row["method"])] = (
            f"{fmt(row['value'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] "
            f"(n={row['n']}, stems={row['n_stems']}){flag}"
        )
    split, stratum = next(iter(runs))
    lines = [
        f"# Method comparison — split {split}, stratum {stratum}",
        "",
        "Cells: `point [95% CI] (n = GT objects / decided candidates / KPI pairs, stems)`. "
        "Pixel/object/KPI CIs: stem bootstrap; candidate CIs: Wilson. NaN = undefined (0/0).",
        "",
        "| family | target | metric | " + " | ".join(methods) + " |",
        "|---|---|---|" + "---|" * len(methods),
    ]
    for key in keys:
        lines.append("| " + " | ".join(key) + " | "
                     + " | ".join(cells.get((*key, m), "—") for m in methods) + " |")
    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, csv_path
