"""Build the human review set (exhaustive tiles + candidates)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sem.qc.config import load_config
from sem.qc.review.sampler import build_review_set


def main() -> None:
    config = load_config()
    work_root = Path(config["paths"]["work_root"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/splits/manifest.csv")
    parser.add_argument("--preds-root", default=str(work_root / "preds"))
    parser.add_argument("--out", default=str(work_root / "review"))
    parser.add_argument("--data-root", default=config["paths"]["data_root"])
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    image_loader = None
    if args.vlm:
        from sem.qc.review.images import StemImageLoader

        image_loader = StemImageLoader(args.data_root)

    result = build_review_set(
        manifest_path=args.manifest,
        preds_root=args.preds_root,
        out_dir=args.out,
        review_config=config["review"],
        tile_px=int(config["tile_sizes"]["exhaustive_review_px"]),
        use_vlm=args.vlm,
        image_loader=image_loader,
        force=args.force,
    )
    summary = result["summary"]
    n_items = len(result["items"])
    print(f"items: {n_items}")
    print(f"exhaustive strata: {summary['exhaustive']['strata']}")
    print(f"candidates pool: {summary['candidates']['pool_after_dedup']}")
    print(f"vlm: {summary['vlm']}")
    print(f"wrote {Path(args.out) / 'items.jsonl'}")


if __name__ == "__main__":
    main()
