"""Evaluate one method against human GT: scripts/evaluate.py --method <name>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sem.qc.config import load_config  # noqa: E402
from sem.qc.eval.compare import run_dir  # noqa: E402
from sem.qc.eval.evaluate import STRATA, EvalSettings, evaluate  # noqa: E402
from sem.qc.eval.gallery import render_galleries  # noqa: E402
from sem.qc.eval.report import write_outputs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", choices=("test", "val"), default=None,
                        help="default: qc_eval.headline_split (test)")
    parser.add_argument("--stratum", choices=STRATA, default=None,
                        help="default: qc_eval.headline_stratum (random)")
    parser.add_argument("--work-root", help="default: paths.work_root ($SEM_WORK_ROOT)")
    parser.add_argument("--data-root", help="default: paths.data_root ($SEM_DATA_ROOT)")
    parser.add_argument("--manifest", help="default: data/splits/manifest.csv")
    parser.add_argument("--n-boot", type=int, help="default: evaluation.bootstrap_replicates")
    parser.add_argument("--out", help="default: <work_root>/eval/<method>[/<split>_<stratum>]")
    parser.add_argument("--no-galleries", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    settings = EvalSettings.from_config(
        config, method=args.method, split=args.split, stratum=args.stratum,
        work_root=args.work_root, data_root=args.data_root,
        manifest_path=args.manifest, n_boot=args.n_boot,
    )
    headline = (config["qc_eval"]["headline_split"], config["qc_eval"]["headline_stratum"])
    out = Path(args.out) if args.out else run_dir(
        settings.work_root / "eval", settings.method, settings.split, settings.stratum, headline
    )
    results = evaluate(settings, with_errors=not args.no_galleries)
    index = None
    if not args.no_galleries:
        index = render_galleries(
            results["_errors"], results["_tile_maps"], settings.data_root, out / "galleries",
            settings.gallery_top_k, int(config["qc_eval"]["gallery_panel_px"]),
        )
    report = write_outputs(results, out, index)
    print(f"{settings.method} {settings.split}/{settings.stratum}: "
          f"{results['counts']['n_tiles']} tiles, {results['counts']['n_stems']} stems, "
          f"{results['counts']['n_candidates_decided']} candidates -> {report}")
    for warning in results["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
