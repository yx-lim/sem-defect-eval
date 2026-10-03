"""VLM suggestions for candidate items via the Anthropic API (never GT)."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sem.qc.review.images import crop, render_context
from sem.qc.schema import ARTIFACT_SUBTYPES

VLM_CLASS_NAMES = {
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
}
VLM_LABEL = "VLM suggestion — not ground truth"


def load_prompt(path: str | Path) -> tuple[str, str]:
    """Split a prompt file on '=== SYSTEM ===' / '=== USER ===' lines."""
    system_lines: list[str] = []
    user_lines: list[str] = []
    section = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() == "=== SYSTEM ===":
            section = "system"
            continue
        if line.strip() == "=== USER ===":
            section = "user"
            continue
        if section == "system":
            system_lines.append(line)
        elif section == "user":
            user_lines.append(line)
    return "\n".join(system_lines).strip(), "\n".join(user_lines).strip()


def fill_user(user_template: str, source: str, proposed_class: str) -> str:
    """Replace placeholders without str.format (text has literal JSON braces)."""
    return user_template.replace("{source}", source).replace(
        "{proposed_class}", proposed_class
    )


def _png_b64(array: np.ndarray) -> str:
    mode = "RGB" if array.ndim == 3 else "L"
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _downscale(array: np.ndarray, max_side: int = 1024) -> np.ndarray:
    if max(array.shape[:2]) <= max_side:
        return array
    factor = max_side / max(array.shape[:2])
    size = (
        max(1, int(round(array.shape[1] * factor))),
        max(1, int(round(array.shape[0] * factor))),
    )
    mode = "RGB" if array.ndim == 3 else "L"
    return np.asarray(
        Image.fromarray(array, mode=mode).resize(size, resample=Image.LANCZOS)
    )


def _model_ids(client: Any) -> set[str]:
    ids: set[str] = set()
    page = client.models.list()
    while True:
        ids.update(model.id for model in page.data)
        hnp = getattr(page, "has_next_page", None)
        more = hnp() if callable(hnp) else bool(hnp)
        if not more:
            return ids
        page = page.get_next_page()


def parse_response(text: str) -> dict[str, Any] | None:
    """Strictly parse the single JSON object the VLM must return."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if set(data) != {"class_name", "artifact_subtype", "confidence", "rationale"}:
        return None
    if data["class_name"] not in VLM_CLASS_NAMES:
        return None
    if data["artifact_subtype"] is not None and (
        data["artifact_subtype"] not in ARTIFACT_SUBTYPES
    ):
        return None
    confidence = data["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return None
    if not 0.0 <= float(confidence) <= 1.0:
        return None
    if not isinstance(data["rationale"], str):
        return None
    return data


def annotate_candidates(
    items: list[dict[str, Any]],
    image_loader: Any,
    vlm_config: dict[str, Any],
    review_dir: str | Path,
    client: Any = None,
    sleep_fn: Any = time.sleep,
) -> dict[str, int | str | None]:
    """Fill vlm_suggestion on candidate items; append raw log to vlm_raw.jsonl."""
    stats: dict[str, Any] = {
        "requested": 0,
        "ok": 0,
        "invalid": 0,
        "errors": 0,
        "skipped_reason": None,
    }
    review_dir = Path(review_dir)
    model_id = str(vlm_config.get("model_id", ""))
    prompt_path = vlm_config.get("prompt_path")
    prompt_version = str(vlm_config.get("prompt_version", ""))
    max_tokens = int(vlm_config.get("max_tokens", 2000))

    if client is None:
        try:
            import anthropic
        except ImportError:
            stats["skipped_reason"] = "anthropic package not installed"
            return stats
        if not os.environ.get("ANTHROPIC_API_KEY"):
            stats["skipped_reason"] = "ANTHROPIC_API_KEY is not set"
            return stats
        client = anthropic.Anthropic()

    try:
        ids = _model_ids(client)
    except Exception as exc:  # noqa: BLE001
        stats["skipped_reason"] = f"model listing failed: {exc}"
        return stats
    if model_id not in ids:
        stats["skipped_reason"] = f"model {model_id!r} not in client.models.list()"
        return stats

    system, user_template = load_prompt(prompt_path)
    raw_log = review_dir / "vlm_raw.jsonl"

    for item in items:
        if item.get("kind") != "candidate":
            continue
        stats["requested"] += 1
        proposal = item["proposal"]
        proposed = proposal.get("class_name") or "uncertain"
        if proposal.get("subtype"):
            proposed += f" (subtype: {proposal['subtype']})"
        user_text = fill_user(user_template, proposal.get("source") or "", proposed)

        tile = item["tile"]
        bse = _downscale(crop(image_loader.get(item["stem"], "BSE"), tile))
        inlens = _downscale(crop(image_loader.get(item["stem"], "Inlens"), tile))
        context = render_context(
            image_loader.get(item["stem"], "BSE"),
            tile,
            proposal.get("polygon"),
        )
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _png_b64(bse),
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _png_b64(inlens),
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _png_b64(context),
                },
            },
            {"type": "text", "text": user_text},
        ]
        request_meta = {
            "model_id": model_id,
            "prompt_version": prompt_version,
            "max_tokens": max_tokens,
        }
        response_text = ""
        tokens = max_tokens
        try:
            response = None
            api_retried = False
            tokens_retried = False
            while True:
                try:
                    response = client.messages.create(
                        model=model_id,
                        max_tokens=tokens,
                        system=system,
                        messages=[{"role": "user", "content": content}],
                    )
                except Exception:
                    if not api_retried:
                        api_retried = True
                        sleep_fn(0.5)
                        continue
                    raise
                if (
                    getattr(response, "stop_reason", None) == "max_tokens"
                    and not tokens_retried
                ):
                    tokens_retried = True
                    tokens = max_tokens * 2
                    continue
                break
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            with raw_log.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    json.dumps(
                        {
                            "item_id": item["item_id"],
                            "request": request_meta,
                            "error": str(exc),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            continue
        text_blocks = [
            getattr(block, "text", "")
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        response_text = "\n".join(text_blocks)
        with raw_log.open("a", encoding="utf-8") as log_file:
            log_file.write(
                json.dumps(
                    {
                        "item_id": item["item_id"],
                        "request": request_meta,
                        "response_text": response_text,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        parsed = parse_response(response_text)
        if parsed is None:
            stats["invalid"] += 1
            continue
        stats["ok"] += 1
        item["vlm_suggestion"] = {
            "model_id": model_id,
            "class_name": parsed["class_name"],
            "artifact_subtype": parsed["artifact_subtype"],
            "confidence": float(parsed["confidence"]),
            "rationale": parsed["rationale"],
            "prompt_version": prompt_version,
            "label": VLM_LABEL,
        }
    return stats
