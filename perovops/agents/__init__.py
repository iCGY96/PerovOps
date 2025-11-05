"""Agent package exports with lazy imports.

This module avoids importing heavy optional dependencies (e.g., LangChain) at
import time so the test suite can run in minimal environments. Submodules are
loaded on first access via :func:`__getattr__`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ingestion", "parser"]

_LAZY_MODULES = {
    "ingestion": "ingestion",
    "parser": "parser",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial delegation
    """Lazily import agent modules on demand."""
    target = _LAZY_MODULES.get(name)
    if target:
        return import_module(f"{__name__}.{target}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:  # pragma: no cover - trivial delegation
    return sorted(set(globals()) | set(__all__))
