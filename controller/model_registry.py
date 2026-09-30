"""
Model Registry — what is actually running, and what it honestly is.

Reads ``models/registry.json``. A result may only cite a ``model_version``
that appears here, so "which model produced this?" always has an answer that
can be checked against the code rather than against a claim in a slide.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "models" / "registry.json"

UNAVAILABLE = "unavailable"


@lru_cache(maxsize=1)
def _load() -> dict[str, dict[str, Any]]:
    if not _REGISTRY_PATH.exists():
        return {}
    models = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8")).get("models", [])
    return {m["task"]: m for m in models}


def entry_for_task(task: str) -> dict[str, Any] | None:
    """Return the registry entry backing a task/tool name, if any."""
    return _load().get(task)


def version_for_task(task: str) -> str:
    """``name@version (kind)`` for a task, or ``'unavailable'`` if nothing backs it."""
    entry = entry_for_task(task)
    if entry is None:
        return UNAVAILABLE
    return f"{entry['name']}@{entry['version']} ({entry['kind']})"


def all_models() -> list[dict[str, Any]]:
    """Every registered model, for the ``/models`` endpoint and reports."""
    return list(_load().values())
