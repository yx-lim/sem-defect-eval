"""Claude crop suggestions for human review candidates."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from sem.qc.schema import ARTIFACT_SUBTYPES


PROMPT_VERSION = "v1"
VLM_CLASSES = (
    "pore",
    "subsurface_uncertain",
    "crack_intraparticle",
    "interparticle_gap",
    "bright_particle",
    "graphite_particle",
    "agglomerate",
    "artifact",
    "matrix_other",
    "uncertain",
)
DEFAULT_PROMPT_PATH = Path(__file__).with_name("vlm_prompt_v1.txt")


def load_prompt(
    path: str | Path = DEFAULT_PROMPT_PATH,
) -> tuple[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    system_marker = "=== SYSTEM ==="
    user_marker = "=== USER ==="
    if text.count(system_marker) != 1 or text.count(user_marker) != 1:
        raise ValueError("Prompt must contain one SYSTEM and one USER marker")
    before, remainder = text.split(system_marker, 1)
    system, user = remainder.split(user_marker, 1)
    if before.strip():
        raise ValueError("Unexpected content before SYSTEM prompt marker")
    return system.strip("\n"), user.strip("\n")


def prompt_sha256(path: str | Path = DEFAULT_PROMPT_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_user(template: str, source: str, proposed_class: str) -> str:
    return template.replace("{source}", source).replace(
        "{proposed_class}", proposed_class
    )


def parse_response(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    start = cleaned.find("{")
    if start < 0:
        return _parse_failure("no_json_object")
    depth = 0
    quoted = False
    escaped = False
    end = None
    for position in range(start, len(cleaned)):
        character = cleaned[position]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                end = position + 1
                break
    if end is None:
        return _parse_failure("unclosed_json_object")
    try:
        payload = json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return _parse_failure("invalid_json")
    if not isinstance(payload, dict):
        return _parse_failure("json_not_object")
    class_name = payload.get("class_name")
    confidence = payload.get("confidence")
    if class_name not in VLM_CLASSES:
        return _parse_failure("unknown_class_name")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return _parse_failure("invalid_confidence")
    subtype = payload.get("artifact_subtype")
    if subtype not in ARTIFACT_SUBTYPES:
        subtype = None
    if class_name != "artifact":
        subtype = None
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return {
        "class_name": class_name,
        "artifact_subtype": subtype,
        "confidence": float(confidence),
        "rationale": rationale,
        "parse_ok": True,
        "parse_error": None,
    }


def _parse_failure(error: str) -> dict[str, Any]:
    return {
        "class_name": "uncertain",
        "artifact_subtype": None,
        "confidence": 0.0,
        "rationale": "",
        "parse_ok": False,
        "parse_error": error,
    }


def _as_rgb(image: np.ndarray) -> Image.Image:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("SEM views must be grayscale or RGB images")
    return Image.fromarray(np.asarray(array[:, :, :3], dtype=np.uint8), mode="RGB")


def _centered_window(
    center_x: float,
    center_y: float,
    desired_width: int,
    desired_height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    width = min(image_width, max(1, desired_width))
    height = min(image_height, max(1, desired_height))
    x0 = int(round(center_x - width / 2.0))
    y0 = int(round(center_y - height / 2.0))
    x0 = min(max(0, x0), image_width - width)
    y0 = min(max(0, y0), image_height - height)
    return x0, y0, x0 + width, y0 + height


def _resize_long_side(image: Image.Image, target: int) -> Image.Image:
    width, height = image.size
    scale = target / max(width, height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(resized, Image.Resampling.LANCZOS)


def _bbox_from_polygon(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("polygon must be a non-empty list of [x, y] points")
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def render_views(
    bse: np.ndarray,
    inlens: np.ndarray,
    bbox: tuple[float, float, float, float] | list[float],
    polygon: list[list[float]] | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Create the paired crop views and outlined wider BSE context."""
    bse_image = _as_rgb(bse)
    inlens_image = _as_rgb(inlens)
    if bse_image.size != inlens_image.size:
        raise ValueError("BSE and Inlens views must have identical dimensions")
    x0, y0, x1, y1 = [float(value) for value in bbox]
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    image_width, image_height = bse_image.size
    crop_width = max(128, int(np.ceil(x1 - x0)))
    crop_height = max(128, int(np.ceil(y1 - y0)))
    crop_window = _centered_window(
        center_x,
        center_y,
        crop_width,
        crop_height,
        image_width,
        image_height,
    )
    bse_crop = _resize_long_side(bse_image.crop(crop_window), 384)
    inlens_crop = _resize_long_side(inlens_image.crop(crop_window), 384)

    context_width = max(crop_window[2] - crop_window[0], int(np.ceil(2 * (x1 - x0))))
    context_height = max(
        crop_window[3] - crop_window[1], int(np.ceil(2 * (y1 - y0)))
    )
    context_window = _centered_window(
        center_x,
        center_y,
        context_width,
        context_height,
        image_width,
        image_height,
    )
    context = _resize_long_side(bse_image.crop(context_window), 768)
    scale_x = context.width / (context_window[2] - context_window[0])
    scale_y = context.height / (context_window[3] - context_window[1])
    local_points = polygon or [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ]
    points = [
        (
            round((point[0] - context_window[0]) * scale_x),
            round((point[1] - context_window[1]) * scale_y),
        )
        for point in local_points
    ]
    ImageDraw.Draw(context).line(points + [points[0]], fill=(255, 0, 0), width=3)
    return bse_crop, inlens_crop, context


def resolve_model_id(client: Any, preferred: str) -> str:
    available: list[str] = []
    after_id = None
    while True:
        kwargs = {"limit": 100}
        if after_id:
            kwargs["after_id"] = after_id
        page = client.models.list(**kwargs)
        for model in getattr(page, "data", page):
            model_id = getattr(model, "id", None)
            if model_id is not None:
                available.append(str(model_id))
        if not getattr(page, "has_more", False):
            break
        next_id = getattr(page, "last_id", None)
        if not next_id or next_id == after_id:
            break
        after_id = next_id
    if preferred in available:
        return preferred
    raise ValueError(
        f"Preferred model {preferred!r} is unavailable; available model ids: "
        f"{', '.join(available) if available else '(none returned)'}"
    )


def _png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _jsonable_usage(usage: Any) -> Any:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: value
        for key, value in vars(usage).items()
        if not key.startswith("_")
    }


def _uncertain_record(
    item: dict[str, Any],
    model_id: str,
    prompt_digest: str,
    error: str,
) -> dict[str, Any]:
    return {
        "item_id": item["item_id"],
        "stem": item["stem"],
        "method": "vlm_claude_crops",
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_digest,
        **_parse_failure("api_error"),
        "error": error,
        "raw_text": "",
        "proposal_source": item.get("proposal", {}).get("source"),
        "proposal_class": item.get("proposal", {}).get("class_name"),
        "stop_reason": None,
        "usage": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def classify_item(
    client: Any,
    model_id: str,
    item: dict[str, Any],
    views: dict[str, np.ndarray],
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
) -> dict[str, Any]:
    system, user_template = load_prompt(prompt_path)
    digest = prompt_sha256(prompt_path)
    proposal = item.get("proposal") or {}
    if proposal.get("polygon"):
        bbox = _bbox_from_polygon(proposal["polygon"])
    else:
        tile = item["tile"]
        bbox = (
            tile["x0"],
            tile["y0"],
            tile["x0"] + tile["w"],
            tile["y0"] + tile["h"],
        )
    crop_bse, crop_inlens, context = render_views(
        views["BSE"], views["Inlens"], bbox, proposal.get("polygon")
    )
    source = str(proposal.get("source", ""))
    proposed_class = str(proposal.get("class_name", ""))
    if proposal.get("subtype") is not None:
        proposed_class = f"{proposed_class} ({proposal['subtype']})"
    user_text = render_user(user_template, source, proposed_class)
    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _png_base64(image),
            },
        }
        for image in (crop_bse, crop_inlens, context)
    ]
    content.append({"type": "text", "text": user_text})
    stop_reason = None
    usage = None
    raw_text = ""
    parsed = _parse_failure("no_response")
    for attempt in range(2):
        for api_attempt in range(4):
            try:
                response = client.messages.create(
                    model=model_id,
                    max_tokens=2000,
                    system=system,
                    messages=[{"role": "user", "content": content}],
                )
                break
            except Exception as exc:
                if api_attempt == 3:
                    return _uncertain_record(
                        item, model_id, digest, f"{type(exc).__name__}: {exc}"
                    )
                time.sleep(2**api_attempt)
        stop_reason = getattr(response, "stop_reason", None)
        usage = _jsonable_usage(getattr(response, "usage", None))
        raw_text = "\n".join(
            str(getattr(block, "text", ""))
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        )
        parsed = parse_response(raw_text)
        if attempt == 0 and (
            stop_reason == "max_tokens" or not parsed["parse_ok"]
        ):
            continue
        break
    return {
        "item_id": item["item_id"],
        "stem": item["stem"],
        "method": "vlm_claude_crops",
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": digest,
        **parsed,
        "error": None,
        "raw_text": raw_text,
        "proposal_source": source,
        "proposal_class": proposal.get("class_name"),
        "stop_reason": stop_reason,
        "usage": usage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
