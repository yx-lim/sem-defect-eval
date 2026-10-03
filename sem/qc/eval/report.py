"""Write metrics.json, tables/*.csv and report.md for one evaluation run."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

CI_FIELDS = ("value", "ci_low", "ci_high")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return _clean(value.item())
    return value


def fmt(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "NaN"
    if isinstance(x, float):
        if x != 0 and (abs(x) >= 1e4 or abs(x) < 10 ** -digits):
            return f"{x:.{digits}g}"
        return f"{x:.{digits}f}"
    return str(x)


def fmt_ci(cell: dict[str, float]) -> str:
    return f"{fmt(cell['value'])} [{fmt(cell['ci_low'])}, {fmt(cell['ci_high'])}]"


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, dict) and "value" in value:
            out[key] = value["value"]
            out[f"{key}_ci_low"] = value["ci_low"]
            out[f"{key}_ci_high"] = value["ci_high"]
        elif isinstance(value, dict) and "ci_low" in value:
            out[f"{key}_ci_low"] = value["ci_low"]
            out[f"{key}_ci_high"] = value["ci_high"]
        else:
            out[key] = value
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten(r) for r in rows]
    fieldnames: list[str] = []
    for row in rows:
        fieldnames += [k for k in row if k not in fieldnames]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("NaN" if isinstance(v, float) and math.isnan(v) else v)
                             for k, v in row.items()})


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _flag(row: dict[str, Any]) -> str:
    return "insufficient n" if row.get("insufficient_n") else ""


def write_outputs(results: dict[str, Any], out_dir: str | Path,
                  gallery_index: list[dict] | None = None) -> Path:
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(_clean(results), indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(tables / "pixel_metrics.csv", results["pixel"])
    write_csv(tables / "object_metrics.csv", results["object"])
    write_csv(tables / "candidate_precision.csv", results["candidate"])
    write_csv(tables / "kpi_errors.csv", results["kpi"])
    write_csv(tables / "kpi_per_tile.csv", results["kpi_per_tile"])
    write_csv(tables / "kpi_per_stem.csv", results["kpi_per_stem"])
    classes = results["confusion"]["classes"]
    for kind in ("raw", "row_normalized"):
        write_csv(tables / f"confusion_{kind}.csv", [
            {"gt\\pred": classes[i], **dict(zip(classes, row))}
            for i, row in enumerate(results["confusion"][kind])
        ])
    if gallery_index is not None:
        write_csv(tables / "gallery_index.csv", gallery_index)
    report = render_report(results, gallery_index)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return out_dir / "report.md"


def render_report(results: dict[str, Any], gallery_index: list[dict] | None) -> str:
    c = results["counts"]
    parts = [
        f"# Evaluation report — `{results['method']}`",
        "",
        f"- Split: **{results['split']}**, stratum: **{results['stratum']}**"
        + (" (headline)" if results["headline"] else " (NOT headline; reported separately)"),
        f"- n GT tiles = {c['n_tiles']}, n stems = {c['n_stems']}, "
        f"n decided candidates = {c['n_candidates_decided']}",
        f"- Manifest: `{results['manifest']['path']}` sha256 `{results['manifest']['sha256'][:16]}…`"
        f" frozen = {results['manifest']['frozen']}",
        f"- CIs: stem-level bootstrap, {results['bootstrap']['replicates']} replicates, "
        f"seed {results['bootstrap']['seed']}, 95% percentile. Values are "
        "`point [ci_low, ci_high]`. NaN = undefined (0/0, e.g. no GT and no prediction).",
        f"- 'insufficient n' = fewer than {results['settings']['min_gt_objects']} GT objects "
        "(or decided candidates / KPI pairs).",
        "- Ground truth = human-reviewed items only (`is_ground_truth`). Model/VLM outputs are "
        "never used as GT.",
    ]
    if results["warnings"]:
        parts += ["", "## Warnings", ""] + [f"- {w}" for w in results["warnings"]]

    parts += ["", "## Pixel metrics (pooled TP/FP/FN over tiles)", "",
              "GT 255 ignored; GT subsurface_uncertain pixels excluded from pore scoring. "
              f"GT-labelled pixels predicted as 255: {c['gt_pixels_predicted_ignore']} "
              "(counted as FN).", ""]
    parts.append(_md_table(
        ["class", "precision", "recall", "F1", "IoU", "TP", "FP", "FN", "n tiles", "n stems",
         "n GT objects", "flag"],
        [[r["class"], fmt_ci(r["precision"]), fmt_ci(r["recall"]), fmt_ci(r["f1"]),
          fmt_ci(r["iou"]), str(r["tp"]), str(r["fp"]), str(r["fn"]), str(r["n_tiles"]),
          str(r["n_stems"]), str(r["n_gt_objects"]), _flag(r)] for r in results["pixel"]],
    ))

    classes = results["confusion"]["classes"]
    short = [n[:10] for n in classes]
    parts += ["", "## Confusion matrix (rows = GT, cols = prediction)", "", "Row-normalized:", ""]
    parts.append(_md_table(["GT \\ pred"] + short, [
        [classes[i]] + [fmt(v, 2) for v in row]
        for i, row in enumerate(results["confusion"]["row_normalized"])
    ]))
    parts += ["", "Raw pixel counts:", ""]
    parts.append(_md_table(["GT \\ pred"] + short, [
        [classes[i]] + [str(v) for v in row] for i, row in enumerate(results["confusion"]["raw"])
    ]))

    parts += ["", "## Object metrics (Hungarian matching on IoU)", ""]
    parts.append(_md_table(
        ["class", "IoU thr", "precision", "recall", "F1", "TP", "FP", "FN", "n tiles",
         "n stems", "n GT objects", "flag"],
        [[r["class"], fmt(r["match_iou"], 1), fmt_ci(r["precision"]), fmt_ci(r["recall"]),
          fmt_ci(r["f1"]), str(r["tp"]), str(r["fp"]), str(r["fn"]), str(r["n_tiles"]),
          str(r["n_stems"]), str(r["n_gt_objects"]), _flag(r)] for r in results["object"]],
    ))
    parts += ["", "Agglomerates are scored only on tiles whose human decision includes a "
              "`polygons` list (n tiles column).", ""]

    parts += ["## Candidate precision", "",
              "precision = accepted / (accepted + rejected + relabeled-to-different-class); "
              "'uncertain' excluded and counted. Random stratum is inclusion-weighted "
              "(Wilson CI on Kish effective n). Recall is not estimable from candidates.", ""]
    if results["candidate"]:
        parts.append(_md_table(
            ["source", "class", "sampling", "precision", "Wilson 95%", "bootstrap 95%",
             "pos", "neg", "uncertain", "n_eff", "n stems", "flag"],
            [[r["source"], r["class"], r["sampling"], fmt(r["precision"]),
              f"[{fmt(r['wilson_low'])}, {fmt(r['wilson_high'])}]",
              f"[{fmt(r['bootstrap']['ci_low'])}, {fmt(r['bootstrap']['ci_high'])}]",
              str(r["n_positive"]), str(r["n_negative"]), str(r["n_uncertain"]),
              fmt(r["n_eff"], 1), str(r["n_stems"]), _flag(r)] for r in results["candidate"]],
        ))
    else:
        parts.append("No decided candidate items for this method/split/stratum.")

    parts += ["", "## QC-quantity errors (sem/qc/kpi.py on GT vs prediction)", "",
              "bias = mean(pred − GT); MAE = mean|pred − GT|; rel. error = Σ|pred − GT| / Σ|GT|; "
              "ρ = Spearman across units. Stem level pools a stem's tiles (kpi.py on a "
              "255-separated mosaic).", ""]
    parts.append(_md_table(
        ["level", "KPI", "GT mean", "pred mean", "bias", "MAE", "rel. error", "Spearman ρ",
         "n pairs", "n stems", "flag"],
        [[r["level"], r["kpi"], fmt(r["gt_mean"]), fmt(r["pred_mean"]), fmt_ci(r["bias"]),
          fmt_ci(r["mae"]), fmt_ci(r["rel_error"]), fmt_ci(r["spearman"]), str(r["n_pairs"]),
          str(r["n_stems"]), _flag(r)] for r in results["kpi"]],
    ))

    if gallery_index is not None:
        files = sorted({g["file"] for g in gallery_index if g.get("file")})
        parts += ["", "## FP/FN galleries (top by area; BSE | GT overlay | prediction overlay)", ""]
        parts += [f"- `galleries/{f}`" for f in files] or ["- none (no unmatched objects)"]
        parts += [f"- WARNING: {g['class']}" for g in gallery_index if g.get("kind") == "warning"]
    parts.append("")
    return "\n".join(parts)
