"""
SAT-OS Falsification — flood/change-specific hypothesis testing.

Extends the existing m20 counterfactual module with domain-specific checks
for the MVP flood path: registration error, cloud/shadow artifacts, seasonal
water, sensor artifacts, image-quality problems.
"""

from __future__ import annotations

from typing import Any

from pipeline.schemas.sat_schemas import FalsificationCheck, FalsificationReport


def _check_registration_error(outputs: list[dict[str, Any]]) -> FalsificationCheck:
    """Did co-registration fail or produce an implausible shift?"""
    for out in outputs:
        r = out.get("result", {})
        if out.get("tool_name") in ("change_detection", "vlm_change_detection", "fusion_change_detection"):
            if r.get("coregistration_reliable") is False:
                return FalsificationCheck(
                    hypothesis="Detected change is a registration artifact",
                    tested=True, supported=True,
                    detail=f"Co-registration rejected shift {r.get('applied_shift_px')} as implausible"
                )
            shift = r.get("applied_shift_px", [0, 0])
            if isinstance(shift, list) and len(shift) == 2:
                if abs(shift[0]) > 20 or abs(shift[1]) > 20:
                    return FalsificationCheck(
                        hypothesis="Detected change is a registration artifact",
                        tested=True, supported=None,
                        detail=f"Large shift applied: {shift}px — change may be partly registration error"
                    )
            return FalsificationCheck(
                hypothesis="Detected change is a registration artifact",
                tested=True, supported=False,
                detail=f"Co-registration reliable, shift={shift}px"
            )
    return FalsificationCheck(hypothesis="Detected change is a registration artifact", tested=False)


def _check_cloud_shadow(outputs: list[dict[str, Any]], observability_pct: float | None) -> FalsificationCheck:
    """Could cloud/shadow explain the detected change?"""
    if observability_pct is not None and observability_pct < 70:
        return FalsificationCheck(
            hypothesis="Change is a cloud/shadow artifact",
            tested=True, supported=None,
            detail=f"Only {observability_pct:.0f}% observable — cloud/shadow may affect results"
        )
    if observability_pct is not None:
        return FalsificationCheck(
            hypothesis="Change is a cloud/shadow artifact",
            tested=True, supported=False,
            detail=f"{observability_pct:.0f}% observable — cloud/shadow unlikely to dominate"
        )
    return FalsificationCheck(hypothesis="Change is a cloud/shadow artifact", tested=False,
                              detail="No observability data available")


def _check_seasonal_water(outputs: list[dict[str, Any]]) -> FalsificationCheck:
    """For flood detection: could the 'flood' be seasonal water variation?"""
    for out in outputs:
        r = out.get("result", {})
        if out.get("tool_name") in ("change_detection", "vlm_change_detection"):
            pct = r.get("changed_area_pct", 0)
            if isinstance(pct, (int, float)) and 2 <= pct <= 15:
                return FalsificationCheck(
                    hypothesis="Detected water change is seasonal variation, not flood",
                    tested=True, supported=None,
                    detail=f"Moderate change ({pct:.1f}%) — could be seasonal water; temporal context needed"
                )
            if isinstance(pct, (int, float)) and pct > 15:
                return FalsificationCheck(
                    hypothesis="Detected water change is seasonal variation, not flood",
                    tested=True, supported=False,
                    detail=f"Large change ({pct:.1f}%) — unlikely to be purely seasonal"
                )
    return FalsificationCheck(hypothesis="Detected water change is seasonal variation, not flood", tested=False)


def _check_image_quality(outputs: list[dict[str, Any]]) -> FalsificationCheck:
    """Check if low confidence suggests image quality problems."""
    for out in outputs:
        r = out.get("result", {})
        if r.get("analysis_unavailable"):
            continue
        conf = r.get("confidence")
        if isinstance(conf, (int, float)) and conf < 0.3:
            return FalsificationCheck(
                hypothesis="Result is unreliable due to poor image quality",
                tested=True, supported=True,
                detail=f"{out.get('tool_name')}: confidence={conf:.2f} suggests quality issues"
            )
    return FalsificationCheck(
        hypothesis="Result is unreliable due to poor image quality",
        tested=True, supported=False, detail="All tools reported adequate confidence"
    )


def run_falsification(
    outputs: list[dict[str, Any]],
    observability_pct: float | None = None,
) -> FalsificationReport:
    """Run all flood/change-specific falsification checks."""
    checks = [
        _check_registration_error(outputs),
        _check_cloud_shadow(outputs, observability_pct),
        _check_seasonal_water(outputs),
        _check_image_quality(outputs),
    ]
    any_contradiction = any(c.supported is True for c in checks)
    return FalsificationReport(checks=checks, any_contradiction=any_contradiction)
