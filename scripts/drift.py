"""Batch change detection CLI (spec §5).

Example: python scripts/drift.py --reference Batch_1 --incoming Batch_2 Batch_3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sem.qc.drift.run import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default="Batch_1")
    ap.add_argument("--incoming", nargs="+", default=["Batch_2", "Batch_3"])
    ap.add_argument("--run", default=None, help="run name under $SEM_WORK_ROOT/drift/")
    ap.add_argument("--method", default="classical_v1")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-root", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--skip-positive", action="store_true")
    a = ap.parse_args()
    out = run(a.reference, a.incoming, a.run, a.data_root, a.work_root, a.method, a.workers,
              a.n_perm, a.n_boot, a.skip_positive)
    print(out / "report.md")


if __name__ == "__main__":
    main()
