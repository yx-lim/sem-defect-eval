"""One results table across methods: scripts/compare_methods.py [--methods a b ...]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sem.qc.config import load_config  # noqa: E402
from sem.qc.eval.compare import compare, run_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", help="default: every method under eval/")
    parser.add_argument("--split", choices=("test", "val"), default=None)
    parser.add_argument("--stratum", choices=("random", "uncertainty", "all"), default=None)
    parser.add_argument("--work-root")
    parser.add_argument("--out", help="default: <work_root>/eval/comparison[_<split>_<stratum>]")
    args = parser.parse_args(argv)

    config = load_config()
    ev = config["qc_eval"]
    headline = (ev["headline_split"], ev["headline_stratum"])
    split, stratum = args.split or headline[0], args.stratum or headline[1]
    eval_root = Path(args.work_root or config["paths"]["work_root"]) / "eval"
    methods = args.methods or sorted(
        p.parent.name for p in eval_root.glob("*/metrics.json")
    )
    files = [run_dir(eval_root, m, split, stratum, headline) / "metrics.json" for m in methods]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise SystemExit(f"Missing metrics.json: {missing}")
    suffix = "" if (split, stratum) == headline else f"_{split}_{stratum}"
    out = Path(args.out) if args.out else eval_root / f"comparison{suffix}"
    md, csv_path = compare(files, out)
    print(f"Compared {methods} ({split}/{stratum}) -> {md}, {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
