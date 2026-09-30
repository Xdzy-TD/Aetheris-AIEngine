"""
M15 — CSV Dataset Analysis.

Real, computed summary statistics for an uploaded CSV: row/column counts,
per-column inferred type, and min/max/mean/missing for numeric columns.
Stdlib only (csv + statistics) — no pandas dependency added for this.
"""

from __future__ import annotations

import csv
import io
import statistics
from typing import Any

from pipeline.config import MAX_UPLOAD_BYTES
from pipeline.schemas.contracts import ModuleID, ModuleResult

MAX_ROWS_SCANNED = 200_000  # guard against pathological files


def _coerce_numeric(value: str) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def analyze_csv(data: bytes, filename: str) -> ModuleResult:
    """Parse CSV bytes and return real per-column statistics."""
    if len(data) > MAX_UPLOAD_BYTES:
        return ModuleResult.failure(
            ModuleID.M15, "FILE_TOO_LARGE",
            f"CSV exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit.",
        )
    try:
        text = data.decode("utf-8-sig", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
    except Exception as exc:
        return ModuleResult.failure(ModuleID.M15, "PARSE_ERROR", str(exc))

    if not rows:
        return ModuleResult.failure(ModuleID.M15, "EMPTY_FILE", "CSV has no rows.")

    header, body = rows[0], rows[1 : MAX_ROWS_SCANNED + 1]
    columns: list[dict[str, Any]] = []
    for i, name in enumerate(header):
        values = [r[i] for r in body if i < len(r)]
        missing = sum(1 for v in values if v.strip() == "")
        numeric = [_coerce_numeric(v) for v in values]
        numeric = [n for n in numeric if n is not None]
        is_numeric = len(numeric) > 0 and len(numeric) >= (len(values) - missing) * 0.9
        col: dict[str, Any] = {
            "name": name or f"column_{i+1}",
            "type": "numeric" if is_numeric else "text",
            "missing": missing,
            "non_null": len(values) - missing,
        }
        if is_numeric:
            col.update(
                min=round(min(numeric), 4),
                max=round(max(numeric), 4),
                mean=round(statistics.fmean(numeric), 4),
            )
        else:
            distinct = len({v for v in values if v.strip() != ""})
            col["distinct"] = distinct
        columns.append(col)

    return ModuleResult.success(
        ModuleID.M15,
        {
            "filename": filename,
            "row_count": len(body),
            "column_count": len(header),
            "columns": columns,
            "preview": body[:5],
            "truncated": len(rows) - 1 > MAX_ROWS_SCANNED,
        },
    )
