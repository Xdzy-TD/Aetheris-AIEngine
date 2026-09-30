"""
SAT-OS Feedback Store — JSONL-backed feedback + run comparison.

Stores feedback separately from the audit log (implementation.md §Engineering
Requirement). Learning/update steps must be explicit — no auto-retraining.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from pipeline.schemas.sat_schemas import FeedbackEntry

_DEFAULT_PATH = Path("logs/sat_feedback.jsonl")


class FeedbackStore:
    """Append-only JSONL feedback store with run comparison."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def submit(self, entry: FeedbackEntry) -> FeedbackEntry:
        """Append a feedback entry."""
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        return entry

    def get_for_run(self, run_id: str) -> list[FeedbackEntry]:
        """All feedback entries for a specific run."""
        return [e for e in self._read_all() if e.run_id == run_id]

    def get_all(self) -> list[FeedbackEntry]:
        return self._read_all()

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        """Compare feedback between two runs — shows whether the system improved."""
        a_entries = self.get_for_run(run_id_a)
        b_entries = self.get_for_run(run_id_b)

        a_categories = [e.category for e in a_entries]
        b_categories = [e.category for e in b_entries]

        return {
            "run_a": {"run_id": run_id_a, "feedback_count": len(a_entries),
                      "categories": dict(_count(a_categories))},
            "run_b": {"run_id": run_id_b, "feedback_count": len(b_entries),
                      "categories": dict(_count(b_categories))},
            "improved": len(b_entries) < len(a_entries),
            "delta_feedback_count": len(b_entries) - len(a_entries),
        }

    def _read_all(self) -> list[FeedbackEntry]:
        if not self._path.exists():
            return []
        entries: list[FeedbackEntry] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(FeedbackEntry.model_validate_json(line))
        return entries


def _count(items: list[str]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return sorted(counts.items())
