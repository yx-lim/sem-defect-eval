"""FastAPI backend serving the review UI, items, images and decisions."""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from sem.qc.config import load_config
from sem.qc.review.images import StemImageLoader, crop, render_context
from sem.qc.schema import (
    ARTIFACT_SUBTYPES,
    CLASS_NAMES,
    IGNORE_LABEL,
    INSTANCE_CLASS_NAMES,
    is_ground_truth,
    latest_decisions,
    read_jsonl,
    resolve_review_items,
    write_jsonl,
)

STATUSES = ["accepted", "rejected", "relabeled", "redrawn", "uncertain"]
PALETTE = {
    "0": [128, 128, 128],
    "1": [31, 119, 180],
    "2": [255, 215, 0],
    "3": [214, 39, 40],
    "4": [148, 103, 189],
    "5": [255, 127, 14],
    "6": [23, 190, 207],
    "7": [227, 119, 194],
    "255": [0, 0, 0],
}
_VALID_MASK_VALUES = set(range(8)) | {IGNORE_LABEL}


def _png_response(array: np.ndarray) -> Response:
    buffer = io.BytesIO()
    mode = "RGB" if array.ndim == 3 else "L"
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


def create_app(
    review_dir: str | Path,
    data_root: str | Path | None = None,
    image_loader: Any = None,
    static_dir: str | Path | None = None,
) -> FastAPI:
    review_dir = Path(review_dir)
    config = load_config()
    review_config = config.get("review", {})
    if image_loader is None:
        root = data_root or config["paths"]["data_root"]
        image_loader = StemImageLoader(root)
    if static_dir is None:
        static_dir = Path(__file__).resolve().parent / "static"
    static_dir = Path(static_dir)

    items_path = review_dir / "items.jsonl"
    decisions_path = review_dir / "decisions.jsonl"
    masks_dir = review_dir / "masks"
    cache_dir = review_dir / "cache"
    masks_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(items_path)
    by_id = {item["item_id"]: item for item in items}

    def _decisions() -> dict[str, dict[str, Any]]:
        if not decisions_path.exists():
            return {}
        return latest_decisions(read_jsonl(decisions_path))

    app = FastAPI(title="SEM QC review")

    @app.middleware("http")
    async def no_store_api(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> Response:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(404, "index.html not found")
        return FileResponse(index_path)

    if static_dir.is_dir():
        app.mount(
            "/static", StaticFiles(directory=str(static_dir)), name="static"
        )

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        return {
            "classes": {str(k): v for k, v in CLASS_NAMES.items()},
            "ignore_label": IGNORE_LABEL,
            "instance_classes": sorted(INSTANCE_CLASS_NAMES),
            "candidate_classes": list(review_config.get("candidate_classes", [])),
            "artifact_subtypes": sorted(ARTIFACT_SUBTYPES),
            "statuses": STATUSES,
            "palette": PALETTE,
            "exhaustive_tile_px": int(
                config.get("tile_sizes", {}).get("exhaustive_review_px", 512)
            ),
        }

    def _latest(item: dict[str, Any]) -> dict[str, Any] | None:
        return _decisions().get(item["item_id"])

    @app.get("/api/items")
    def api_items(
        kind: str | None = None,
        class_name: str | None = None,
        status: str | None = None,
        split: str | None = None,
    ) -> dict[str, Any]:
        decisions = _decisions()
        rows = []
        for item in items:
            if kind and item["kind"] != kind:
                continue
            if split and item["split"] != split:
                continue
            proposal = item.get("proposal") or {}
            if class_name and proposal.get("class_name") != class_name:
                continue
            human = decisions.get(item["item_id"])
            item_status = (human or {}).get("status")
            if status == "unreviewed" and item_status is not None:
                continue
            if status == "reviewed" and item_status is None:
                continue
            if status and status not in {"unreviewed", "reviewed"}:
                if item_status != status:
                    continue
            rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": item["kind"],
                    "stem": item["stem"],
                    "batch": item["batch"],
                    "split": item["split"],
                    "class_name": proposal.get("class_name"),
                    "subtype": proposal.get("subtype"),
                    "source": proposal.get("source"),
                    "sampling_method": item["sampling"]["method"],
                    "stratum": item["sampling"]["stratum"],
                    "status": item_status,
                    "reviewer_id": (human or {}).get("reviewer_id"),
                }
            )
        return {"items": rows}

    @app.get("/api/items/{item_id}")
    def api_item(item_id: str) -> dict[str, Any]:
        if item_id not in by_id:
            raise HTTPException(404, "unknown item_id")
        item = dict(by_id[item_id])
        item["human"] = _decisions().get(item_id)
        item["has_mask"] = (masks_dir / f"{item_id}.png").exists()
        return item

    def _get_item(item_id: str) -> dict[str, Any]:
        if item_id not in by_id:
            raise HTTPException(404, "unknown item_id")
        return by_id[item_id]

    @app.get("/api/items/{item_id}/crop/{view}.png")
    def api_crop(item_id: str, view: str) -> Response:
        item = _get_item(item_id)
        if view not in {"BSE", "Inlens"}:
            raise HTTPException(404, "unknown view")
        cache_path = cache_dir / f"{item_id}_{view}.png"
        if cache_path.exists():
            return FileResponse(cache_path)
        img = image_loader.get(item["stem"], view)
        tile_crop = crop(img, item["tile"])
        Image.fromarray(tile_crop, mode="L").save(cache_path)
        return _png_response(tile_crop)

    @app.get("/api/items/{item_id}/context.png")
    def api_context(item_id: str) -> Response:
        item = _get_item(item_id)
        cache_path = cache_dir / f"{item_id}_context.png"
        if cache_path.exists():
            return FileResponse(cache_path)
        bse = image_loader.get(item["stem"], "BSE")
        context = render_context(
            bse,
            item["tile"],
            (item.get("proposal") or {}).get("polygon"),
            scale=float(review_config.get("context_scale", 3.0)),
            max_px=int(review_config.get("context_max_px", 512)),
        )
        Image.fromarray(context, mode="RGB").save(cache_path)
        return _png_response(context)

    @app.get("/api/items/{item_id}/prefill.png")
    def api_prefill(item_id: str) -> Response:
        item = _get_item(item_id)
        rel = (item.get("proposal") or {}).get("semantic_png")
        if not rel:
            raise HTTPException(404, "no prefill semantic for this item")
        path = review_dir / rel
        if not path.exists():
            raise HTTPException(404, "prefill file missing")
        return FileResponse(path)

    @app.get("/api/items/{item_id}/mask.png")
    def api_mask(item_id: str) -> Response:
        _get_item(item_id)
        path = masks_dir / f"{item_id}.png"
        if not path.exists():
            raise HTTPException(404, "no mask for this item")
        return FileResponse(path)

    @app.post("/api/items/{item_id}/mask")
    async def api_post_mask(item_id: str, request: Request) -> dict[str, str]:
        item = _get_item(item_id)
        body = await request.body()
        tile = item["tile"]
        expected = int(tile["w"]) * int(tile["h"])
        if len(body) != expected:
            raise HTTPException(
                422, f"mask payload must be {expected} bytes, got {len(body)}"
            )
        array = np.frombuffer(body, dtype=np.uint8)
        if not set(np.unique(array).tolist()) <= _VALID_MASK_VALUES:
            raise HTTPException(422, "mask values must be in 0..7 or 255")
        out = masks_dir / f"{item_id}.png"
        tmp = masks_dir / f".{item_id}.png.tmp"
        Image.fromarray(array.reshape(tile["h"], tile["w"]), mode="L").save(
            tmp, format="PNG"
        )
        os.replace(tmp, out)
        return {"semantic_png": f"masks/{item_id}.png"}

    def _validate_decision(item: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        item_id = item["item_id"]
        status = body.get("status")
        if status is not None and status not in STATUSES:
            raise HTTPException(422, "invalid status")
        reviewer_id = body.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise HTTPException(422, "reviewer_id is required")
        class_name = body.get("class_name")
        if class_name is not None and class_name not in INSTANCE_CLASS_NAMES:
            raise HTTPException(422, "invalid class_name")
        subtype = body.get("subtype")
        if subtype is not None and subtype not in ARTIFACT_SUBTYPES:
            raise HTTPException(422, "invalid subtype")
        if subtype is not None and class_name != "artifact":
            raise HTTPException(422, "subtype is only allowed for artifact")
        polygons = body.get("polygons") or []
        if not isinstance(polygons, list):
            raise HTTPException(422, "polygons must be a list")
        for poly in polygons:
            if not isinstance(poly, dict):
                raise HTTPException(422, "polygon entries must be objects")
            if poly.get("class_name") not in INSTANCE_CLASS_NAMES:
                raise HTTPException(422, "polygon class_name invalid")
            if poly.get("subtype") is not None and (
                poly["subtype"] not in ARTIFACT_SUBTYPES
                or poly["class_name"] != "artifact"
            ):
                raise HTTPException(422, "polygon subtype invalid")
            points = poly.get("points")
            if (
                not isinstance(points, list)
                or len(points) < 3
                or not all(
                    isinstance(p, (list, tuple))
                    and len(p) == 2
                    and all(isinstance(v, (int, float)) for v in p)
                    for p in points
                )
            ):
                raise HTTPException(422, "polygon needs >=3 [x, y] points")
        semantic_png = body.get("semantic_png")
        if semantic_png is not None:
            if semantic_png != f"masks/{item_id}.png":
                raise HTTPException(422, "semantic_png must be masks/<item_id>.png")
            if not (masks_dir / f"{item_id}.png").exists():
                raise HTTPException(422, "referenced mask file does not exist")
        if status == "relabeled" and not class_name:
            raise HTTPException(422, "relabeled requires class_name")
        if status == "redrawn" and not (semantic_png or polygons):
            raise HTTPException(422, "redrawn requires semantic_png or polygons")
        proposal = item.get("proposal") or {}
        if status == "accepted":
            if item["kind"] == "candidate":
                if class_name is None:
                    class_name = proposal.get("class_name")
                if subtype is None:
                    subtype = proposal.get("subtype")
            else:
                if semantic_png is None:
                    semantic_png = proposal.get("semantic_png")
                if not polygons:
                    polygons = [
                        {
                            "class_name": inst.get("class_name"),
                            "subtype": inst.get("subtype"),
                            "points": inst.get("polygon") or [],
                        }
                        for inst in proposal.get("instances") or []
                    ]
        return {
            "status": status,
            "class_name": class_name,
            "subtype": subtype,
            "polygons": polygons or [],
            "semantic_png": semantic_png,
            "notes": body.get("notes") or "",
            "reviewer_id": reviewer_id,
            "timestamp": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }

    @app.post("/api/items/{item_id}/decision")
    async def api_decision(item_id: str, request: Request) -> dict[str, Any]:
        item = _get_item(item_id)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "body must be JSON")
        if not isinstance(body, dict):
            raise HTTPException(422, "body must be a JSON object")
        human = _validate_decision(item, body)
        line = json.dumps(
            {"item_id": item_id, "human": human}, sort_keys=True
        )
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with decisions_path.open("a", encoding="utf-8") as decisions_file:
            decisions_file.write(line + "\n")
            decisions_file.flush()
            os.fsync(decisions_file.fileno())
        return {"item_id": item_id, "human": human}

    @app.get("/api/progress")
    def api_progress() -> dict[str, Any]:
        decisions = _decisions()
        progress = {
            "total": len(items),
            "reviewed": 0,
            "by_kind": {},
            "by_class": {},
            "by_status": {s: 0 for s in STATUSES},
        }
        progress["by_status"]["unreviewed"] = 0
        for item in items:
            status = (decisions.get(item["item_id"]) or {}).get("status")
            kind = item["kind"]
            cls = (
                (item.get("proposal") or {}).get("class_name")
                if kind == "candidate"
                else "exhaustive_tile"
            )
            for bucket, key in (("by_kind", kind), ("by_class", cls)):
                entry = progress[bucket].setdefault(key, {"total": 0, "reviewed": 0})
                entry["total"] += 1
                if status is not None:
                    entry["reviewed"] += 1
            if status is None:
                progress["by_status"]["unreviewed"] += 1
            else:
                progress["by_status"][status] += 1
                progress["reviewed"] += 1
        return progress

    @app.get("/api/next_unreviewed")
    def api_next(
        after: str | None = None,
        kind: str | None = None,
        class_name: str | None = None,
    ) -> dict[str, Any]:
        decisions = _decisions()

        def _matches(item: dict[str, Any]) -> bool:
            if kind and item["kind"] != kind:
                return False
            if class_name and (item.get("proposal") or {}).get(
                "class_name"
            ) != class_name:
                return False
            return (decisions.get(item["item_id"]) or {}).get("status") is None

        start = 0
        if after is not None:
            for idx, item in enumerate(items):
                if item["item_id"] == after:
                    start = idx + 1
                    break
        n = len(items)
        for offset in range(n):
            item = items[(start + offset) % n]
            if _matches(item):
                return {"item_id": item["item_id"]}
        return {"item_id": None}

    @app.post("/api/export")
    def api_export() -> dict[str, Any]:
        decisions = read_jsonl(decisions_path) if decisions_path.exists() else []
        resolved = resolve_review_items(items, decisions)
        gt = [item for item in resolved if is_ground_truth(item)]
        out_path = review_dir / "export_gt.jsonl"
        write_jsonl(out_path, gt)
        return {"path": str(out_path), "count": len(gt)}

    return app
