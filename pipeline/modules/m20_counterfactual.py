"""
M20 — Counterfactual / Self-Checking Agent.

Attempts to falsify each specialist's own conclusion instead of reporting it
at face value:

- Where a genuinely independent method for the same task already exists in
  this repo (a different specialist, not a resample of the same one — see
  the ``_*_FAMILY`` tuples below), that alternate route is re-run on the
  same images and its verdict compared against the primary one.
- Where the executed tool was already resampled for self-consistency
  (``confidence/self_consistency.py``, the ``vlm_*`` specialists), that
  agreement ratio is itself a falsification signal and is surfaced here
  rather than re-run.
- A tool with neither is reported as untested — silently skipping it would
  look like every conclusion had been checked when most hadn't.

A failed counterfactual call is recorded as inconclusive; it never raises
out of this module, so a best-effort secondary check can't take down an
answer the primary pipeline already produced.
"""

from __future__ import annotations

from typing import Any

from pipeline.schemas.contracts import ModuleID, ModuleResult

# Tools that measure the same real-world question through a genuinely
# different method already in this repo. Kept as an explicit tuple, not
# derived from the schemas (unlike M19's pixel check) — "measures the same
# thing" is domain knowledge a JSON schema can't express.
_CHANGE_FAMILY = ("change_detection", "vlm_change_detection", "fusion_change_detection")
_GROUNDING_FAMILY = ("grounding", "vlm_grounding")

# change_detection's own "minimal change" cutoff
# (specialists/change_detection/model.py::_generate_summary) — reused here
# as the shared changed/not-changed verdict boundary instead of a new one.
_CHANGE_VERDICT_PCT = 2.0
# Below this bounding-box IoU, two grounding routes are treated as pointing
# at different regions rather than the same one.
_BBOX_IOU_AGREEMENT_MIN = 0.3
# Below this self-consistency agreement ratio, resampling the same model
# already contradicts its own reported answer.
_RESAMPLE_AGREEMENT_MIN = 0.6


def _bbox_iou(a: Any, b: Any) -> float | None:
    if not (isinstance(a, list) and isinstance(b, list) and len(a) == 4 and len(b) == 4):
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else None


def _call_alternate(registry: Any, tool: str, call_args: dict[str, Any]) -> dict[str, Any] | None:
    """Best-effort call to an alternate route. ``None`` means it couldn't run at all."""
    try:
        result = registry.call_tool(tool, **call_args)
    except Exception:  # noqa: BLE001 - a failed counterfactual must never break the primary answer
        return None
    return result if isinstance(result, dict) else None


def _check_change(tool: str, result: dict[str, Any], registry: Any, images: list[dict[str, Any]]) -> dict[str, Any]:
    alt_tool = next(t for t in _CHANGE_FAMILY if t != tool)
    if len(images) < 2:
        return {"tool": tool, "method": "alternate_route", "alt_tool": alt_tool,
                "attempted": False, "reason": "fewer than two images to re-run against"}
    call_args = {
        "image_t1_path": images[0].get("path"), "image_t2_path": images[1].get("path"),
        "modality": images[0].get("modality", "optical"),
    }
    alt = _call_alternate(registry, alt_tool, call_args)
    if alt is None or alt.get("analysis_unavailable"):
        reason = alt.get("reason") if alt else "alternate route unavailable"
        return {"tool": tool, "method": "alternate_route", "alt_tool": alt_tool,
                "attempted": True, "corroborated": None, "reason": reason}
    primary_changed = result.get("changed_area_pct", 0.0) >= _CHANGE_VERDICT_PCT
    alt_changed = alt.get("changed_area_pct", 0.0) >= _CHANGE_VERDICT_PCT
    return {
        "tool": tool, "method": "alternate_route", "alt_tool": alt_tool, "attempted": True,
        "primary_changed_area_pct": result.get("changed_area_pct"),
        "alt_changed_area_pct": alt.get("changed_area_pct"),
        "corroborated": primary_changed == alt_changed,
    }


def _check_grounding(
    tool: str, result: dict[str, Any], registry: Any, images: list[dict[str, Any]], step_args: dict[str, Any]
) -> dict[str, Any]:
    alt_tool = next(t for t in _GROUNDING_FAMILY if t != tool)
    if not images:
        return {"tool": tool, "method": "alternate_route", "alt_tool": alt_tool,
                "attempted": False, "reason": "no image to re-run against"}
    call_args = {"image_path": images[0].get("path"), "text_prompt": step_args.get("text_prompt", "")}
    alt = _call_alternate(registry, alt_tool, call_args)
    if alt is None or alt.get("analysis_unavailable"):
        reason = alt.get("reason") if alt else "alternate route unavailable"
        return {"tool": tool, "method": "alternate_route", "alt_tool": alt_tool,
                "attempted": True, "corroborated": None, "reason": reason}
    iou = _bbox_iou(result.get("bbox"), alt.get("bbox"))
    return {
        "tool": tool, "method": "alternate_route", "alt_tool": alt_tool, "attempted": True,
        "bbox_iou": round(iou, 4) if iou is not None else None,
        "corroborated": iou is not None and iou >= _BBOX_IOU_AGREEMENT_MIN,
    }


def run_counterfactual_check(
    outputs: list[dict[str, Any]],
    plan_steps: list[dict[str, Any]],
    images: list[dict[str, Any]],
    registry: Any,
) -> ModuleResult:
    """Attempt to falsify each specialist's conclusion, where a check exists for it."""
    if not outputs:
        return ModuleResult.skipped(ModuleID.M20, "No specialist outputs to check")

    args_by_tool = {s.get("tool_name"): s.get("arguments", {}) for s in plan_steps}
    checks: list[dict[str, Any]] = []

    for out in outputs:
        tool = out.get("tool_name", "")
        result = out.get("result") or {}
        if result.get("analysis_unavailable"):
            continue

        sc = result.get("self_consistency")
        if isinstance(sc, dict):
            ratio = sc.get("agreement_ratio", 0.0)
            checks.append({
                "tool": tool, "method": "resample_agreement", "attempted": True,
                "agreement_ratio": ratio, "corroborated": ratio >= _RESAMPLE_AGREEMENT_MIN,
            })

        if tool in _CHANGE_FAMILY:
            checks.append(_check_change(tool, result, registry, images))
        elif tool in _GROUNDING_FAMILY:
            checks.append(_check_grounding(tool, result, registry, images, args_by_tool.get(tool, {})))
        elif not isinstance(sc, dict):
            checks.append({"tool": tool, "method": "none", "attempted": False,
                            "reason": "no independent check exists for this task"})

    attempted = [c for c in checks if c["attempted"]]
    contradicted = [c for c in checks if c.get("corroborated") is False]
    inconclusive = [c for c in attempted if c.get("corroborated") is None]
    return ModuleResult.success(
        ModuleID.M20,
        {
            "checks": checks,
            "falsification_attempted": len(attempted),
            "falsification_succeeded": bool(contradicted),
            "contradicted_tools": [c["tool"] for c in contradicted],
            "inconclusive_tools": [c["tool"] for c in inconclusive],
        },
    )
