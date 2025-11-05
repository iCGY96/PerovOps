"""Shared serialization helpers used across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def to_serializable(value: Any) -> Any:
    """Convert nested structures returned by agents into JSON/YAML-safe data.

    - Supports Pydantic-style objects (`model_dump`)
    - Supports dataclasses (DocBundle/DocSection/DocTable/DocFigure, etc.)
    - Supports Path, datetime, and nested containers
    """
    if hasattr(value, "model_dump"):
        return to_serializable(value.model_dump())

    if is_dataclass(value):
        return to_serializable(asdict(value))

    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    return value


__all__ = ["to_serializable"]
