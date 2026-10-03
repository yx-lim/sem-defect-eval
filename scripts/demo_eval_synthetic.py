"""Synthetic end-to-end demo of the eval harness (FAKE data, not a real result).

Generates a fake study (labels, BSE-like images, manifest, fixture decisions, two
degraded classical_v1-style predictions), then runs evaluate.py and
compare_methods.py on it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sem.qc.eval.synthetic import make_study  # noqa: E402

METHODS = {"synthetic_classical_v1_mild": 0.4, "synthetic_classical_v1_strong": 1.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    work = os.environ.get("SEM_WORK_ROOT", "/home/ubuntu/work/qc")
    parser.add_argument("--out", default=f"{work}/demo_eval_synthetic")
    parser.add_argument("--n-boot", type=int, default=None)
    args = parser.parse_args()
    paths = make_study(args.out, METHODS)
    common = ["--work-root", str(paths["work_root"]), "--data-root", str(paths["data_root"]),
              "--manifest", str(paths["manifest"])]
    if args.n_boot:
        common += ["--n-boot", str(args.n_boot)]
    py = sys.executable
    for method in METHODS:
        for stratum in ("random", "uncertainty"):
            subprocess.run([py, str(REPO / "scripts/evaluate.py"), "--method", method,
                            "--stratum", stratum, *common], check=True)
    for stratum in ("random", "uncertainty"):
        subprocess.run([py, str(REPO / "scripts/compare_methods.py"), "--work-root",
                        str(paths["work_root"]), "--stratum", stratum], check=True)
    print(f"Demo outputs under {paths['work_root'] / 'eval'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
