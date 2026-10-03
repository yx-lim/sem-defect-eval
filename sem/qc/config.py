"""Configuration loading for the QC pipeline."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any, environ: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item, environ) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item, environ) for item in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda match: environ.get(match.group(1), match.group(2) or ""),
            value,
        )
    return value


def load_config(
    path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load YAML configuration and expand ``${VAR:-default}`` values."""
    if path is None:
        path = os.environ.get(
            "SEM_QC_CONFIG",
            Path(__file__).resolve().parents[2] / "configs" / "qc.yaml",
        )
    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    return _expand_env(config, dict(os.environ if environ is None else environ))
