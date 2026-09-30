"""Benchmark harness.

Runs a labeled JSONL dataset through ``AgenticPlanner.process_query`` end to
end and reports routing/answer accuracy plus latency. Per ``docs/STATUS.md``,
no numbers are invented: an unscored item (no ``expected`` block) counts
toward latency only, and ``accuracy`` is ``None`` when nothing was scorable.

``run_comparison`` runs the same dataset a second way: for each item's task
family (vqa_caption, grounding, change_detection), every *bound* alternative
model for that family — classical baseline vs. pretrained/learned, see
``models/registry.json`` — is called directly and scored identically, giving
one accuracy/latency row per model instead of one row per query. That is the
quantitative baseline comparison ``run_benchmark`` alone doesn't produce.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from controller.planner import AgenticPlanner
from controller.schemas import ExecutionResult, Query, ToolCall
from controller.tool_registry import ToolRegistry

# Task families with more than one bound implementation worth comparing.
_FAMILIES: dict[str, list[str]] = {
    "vqa_caption": ["vqa_caption", "vlm_vqa_caption"],
    "grounding": ["grounding", "vlm_grounding"],
    "change_detection": ["change_detection", "vlm_change_detection", "fusion_change_detection"],
}


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSON object per line: ``{"query": {...}, "expected": {...}}``.

    ``expected`` is optional. Blank lines and ``#``-comments are skipped.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in (s.strip() for s in lines) if ln and not ln.startswith("#")]


def _score_keywords(answer: str, keywords: list[str] | None) -> bool | None:
    """``True``/``False`` if every keyword (case-insensitive) is in ``answer``, else ``None``."""
    if not keywords:
        return None
    lowered = answer.lower()
    return all(kw.lower() in lowered for kw in keywords)


def _score(result: ExecutionResult, expected: dict[str, Any]) -> bool | None:
    """``True``/``False`` if ``expected`` gives something checkable, else ``None``.

    ``keywords``: every keyword must appear in the final answer (case-insensitive).
    ``tool``: the named tool must appear among the executed outputs.
    Both may be given; both must hold for the item to score ``True``.
    """
    if not expected:
        return None
    checks: list[bool] = []
    if "keywords" in expected:
        checks.append(bool(_score_keywords(result.final_answer, expected["keywords"])))
    if "tool" in expected:
        checks.append(expected["tool"] in {o.tool_name for o in result.outputs})
    return all(checks) if checks else None


def _accuracy(scored: list[bool | None]) -> float | None:
    """Fraction correct among the scorable items, or ``None`` if nothing was scorable."""
    hits = [s for s in scored if s is not None]
    return (sum(hits) / len(hits)) if hits else None


def _latency_stats(latencies: list[float]) -> dict[str, float | None]:
    if not latencies:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": statistics.fmean(latencies),
        "p50": statistics.median(latencies),
        "p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
    }


async def run_benchmark(dataset_path: str | Path, planner: AgenticPlanner) -> dict[str, Any]:
    """Run every item in ``dataset_path`` through ``planner`` and aggregate metrics."""
    rows: list[dict[str, Any]] = []
    for item in load_dataset(dataset_path):
        query = Query(**item["query"])
        t0 = time.perf_counter()
        result = await planner.process_query(query)
        rows.append({
            "query_id": result.query_id,
            "text": query.text,
            "latency_s": time.perf_counter() - t0,
            "correct": _score(result, item.get("expected", {})),
            "final_answer": result.final_answer,
        })

    return {
        "dataset": str(dataset_path),
        "total": len(rows),
        "scored": len([r for r in rows if r["correct"] is not None]),
        "accuracy": _accuracy([r["correct"] for r in rows]),
        "latency_s": _latency_stats([r["latency_s"] for r in rows]),
        "rows": rows,
    }


def _answer_text(result: dict[str, Any]) -> str:
    """Best-effort natural-language text from a raw specialist result, for keyword scoring.

    Mirrors the exact phrasing ``AgenticPlanner._refuse`` uses so a dataset's
    ``keywords: ["unavailable"]`` check works the same way here as it does
    through the full pipeline.
    """
    if result.get("analysis_unavailable"):
        return f"Analysis unavailable — {result.get('reason', '')}."
    return str(result.get("answer") or result.get("change_summary") or "")


def _families_for(query: Query) -> list[str]:
    """Task families a dataset item exercises, mirroring the same image-count/
    modality heuristic ``AgenticPlanner._plan_rule_based`` uses to pick a default route."""
    n = len(query.images)
    if n >= 2 and len({img.modality for img in query.images}) == 1:
        return ["change_detection"]
    if n == 1:
        return ["vqa_caption", "grounding"]
    return []


def run_comparison(dataset_path: str | Path, registry: ToolRegistry) -> dict[str, Any]:
    """Compare every bound alternative model for each item's task family.

    Calls each model directly through ``registry.call_tool`` (bypassing the
    planner's own routing choice) so every candidate for the same family sees
    the same input. A model with no bound implementation (e.g. the optional
    ``vlm_*`` specialists without ``torch``/``transformers`` installed) is
    skipped rather than scored ``0`` — its absence isn't a wrong answer.
    """
    per_tool: dict[str, list[dict[str, Any]]] = {}
    for item in load_dataset(dataset_path):
        query = Query(**item["query"])
        keywords = item.get("expected", {}).get("keywords")
        for family in _families_for(query):
            base_args = (
                {"text_prompt": query.text} if family == "grounding" else {"question": query.text}
            )
            for tool in _FAMILIES[family]:
                if registry.get_tool(tool).implementation is None:
                    continue
                args = AgenticPlanner._resolve_image_args(
                    ToolCall(tool_name=tool, arguments=dict(base_args)), query, {}
                )
                t0 = time.perf_counter()
                try:
                    result = registry.call_tool(tool, **args)
                except Exception as exc:  # a bad call is a comparison result, not a crash
                    result = {"analysis_unavailable": True, "reason": str(exc)}
                per_tool.setdefault(tool, []).append({
                    "latency_s": time.perf_counter() - t0,
                    "correct": _score_keywords(_answer_text(result), keywords),
                })

    return {
        tool: {
            "n": len(rows),
            "accuracy": _accuracy([r["correct"] for r in rows]),
            "latency_s": _latency_stats([r["latency_s"] for r in rows]),
        }
        for tool, rows in per_tool.items()
    }


def write_report(report: dict[str, Any], out_path: str | Path) -> None:
    """Write ``report`` as pretty JSON, creating the parent directory if needed."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
