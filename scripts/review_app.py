"""Serve the human review UI + API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    import uvicorn

    from sem.qc.config import load_config
    from sem.qc.review.app import create_app

    config = load_config()
    work_root = Path(config["paths"]["work_root"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--review-dir", default=str(work_root / "review"))
    parser.add_argument("--data-root", default=config["paths"]["data_root"])
    args = parser.parse_args()

    app = create_app(review_dir=args.review_dir, data_root=args.data_root)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
