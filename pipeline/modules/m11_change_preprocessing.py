"""
M11 — Change-Detection Preprocessing.

Validates a bi-temporal image pair *before* it reaches the old
``specialists.change_detection`` model, and co-registers it when needed.

Differencing an unaligned or CRS-mismatched pair reports the misalignment
itself as "change" (see ``specialists/change_detection/model.py`` docstring).
M11 is the guard in front of that: it checks CRS/resolution compatibility
(via M09), estimates the pixel shift between the two scenes, and — if the
shift exceeds a tolerance — runs co-registration through the old
``preprocessing`` package before handing the pair onward.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pipeline.adapters.imagery_adapter import ImageryAdapter
from pipeline.adapters.preprocessing_adapter import PreprocessingAdapter
from pipeline.modules.m09_geospatial import check_crs_compatibility
from pipeline.schemas.contracts import ModuleID, ModuleResult

# Above this many pixels of estimated misalignment, co-registration is
# required rather than merely recommended — differencing would otherwise
# mostly measure the shift, not real change.
_SHIFT_TOLERANCE_PX = 3
_MAX_NODATA_FRACTION = 0.5
_THUMBNAIL_SIZE = (256, 256)


def _grayscale_thumbnail(path: str) -> np.ndarray | None:
    """Small grayscale array for cheap shift estimation. Best-effort."""
    try:
        pil_img = ImageryAdapter.load_pil(path, mode="L", size=_THUMBNAIL_SIZE)  # type: ignore[attr-defined]
        return np.asarray(pil_img, dtype=np.float64) / 255.0
    except AttributeError:
        # ImageryAdapter has no load_pil wrapper (older layer version) —
        # fall back to load_rgb and average the channels ourselves.
        try:
            rgb = ImageryAdapter.load_rgb(path)
            arr = np.asarray(rgb, dtype=np.float64)
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)
            return arr
        except Exception:
            return None
    except Exception:
        return None


def validate_change_pair(
    path_a: str,
    path_b: str,
    auto_coregister: bool = False,
) -> ModuleResult:
    """Validate (and optionally co-register) a bi-temporal image pair.

    Returns a success result carrying the pair's readiness for change
    detection, or a failure result naming the reason it isn't ready
    (mirrors the ``optical_sar_fusion`` refusal path in the old policy
    engine: a bad pair should be refused here, not silently differenced).
    """
    t0 = time.perf_counter()
    issues: list[str] = []
    warnings: list[str] = []

    meta_a = ImageryAdapter.probe_metadata(path_a)
    meta_b = ImageryAdapter.probe_metadata(path_b)

    if meta_a.get("error"):
        issues.append(f"Image A unreadable: {meta_a['error']}")
    if meta_b.get("error"):
        issues.append(f"Image B unreadable: {meta_b['error']}")
    if issues:
        return ModuleResult.failure(ModuleID.M11, "PAIR_UNREADABLE", "; ".join(issues))

    # 1. CRS compatibility (delegates to M09 — same rule used everywhere else)
    crs_result = check_crs_compatibility(meta_a.get("crs"), meta_b.get("crs"))
    reprojection_needed = crs_result.data.get("reprojection_needed", False)
    if crs_result.status.value == "failure":
        issues.append(crs_result.errors[0].message if crs_result.errors else "CRS mismatch")

    # 2. Resolution compatibility
    res_a = meta_a.get("resolution_m")
    res_b = meta_b.get("resolution_m")
    if res_a and res_b and res_a > 0 and res_b > 0:
        ratio = max(res_a, res_b) / min(res_a, res_b)
        if ratio > 1.5:
            warnings.append(
                f"Resolution differs: {res_a}m vs {res_b}m — consider resampling "
                "to a common grid before differencing"
            )

    # 3. NoData fraction — an image mostly NoData can't support real change
    #    detection, and would otherwise get silently differenced anyway.
    for label, meta in (("A", meta_a), ("B", meta_b)):
        frac = meta.get("nodata_fraction") or 0.0
        if frac > _MAX_NODATA_FRACTION:
            issues.append(f"Image {label} is {frac*100:.1f}% NoData — insufficient valid pixels")

    if issues:
        result = ModuleResult.failure(ModuleID.M11, "PAIR_VALIDATION_FAILED", "; ".join(issues))
        result.warnings = warnings
        result.execution_time_s = round(time.perf_counter() - t0, 4)
        return result

    # 4. Estimate misalignment via cheap FFT phase correlation
    shift: tuple[int, int] | None = None
    thumb_a, thumb_b = _grayscale_thumbnail(path_a), _grayscale_thumbnail(path_b)
    if thumb_a is not None and thumb_b is not None and thumb_a.shape == thumb_b.shape:
        try:
            shift = PreprocessingAdapter.estimate_shift(thumb_a, thumb_b)
        except Exception as exc:
            warnings.append(f"Shift estimation failed: {exc}")
    else:
        warnings.append("Could not build comparable thumbnails for shift estimation")

    shift_magnitude = None
    coregistration_needed = False
    if shift is not None:
        shift_magnitude = float(np.hypot(*shift))
        coregistration_needed = shift_magnitude > _SHIFT_TOLERANCE_PX or reprojection_needed
        if coregistration_needed:
            warnings.append(
                f"Estimated misalignment {shift_magnitude:.1f}px exceeds "
                f"{_SHIFT_TOLERANCE_PX}px tolerance — co-registration recommended"
            )

    coregistered_path = None
    if coregistration_needed and auto_coregister:
        coreg_result = PreprocessingAdapter.coregister(path_a, path_b)
        if coreg_result.status.value == "success":
            coregistered_path = coreg_result.data.get("output_path")
        else:
            issues.append(
                "Co-registration required but failed: "
                + (coreg_result.errors[0].message if coreg_result.errors else "unknown error")
            )

    elapsed = round(time.perf_counter() - t0, 4)
    if issues:
        result = ModuleResult.failure(ModuleID.M11, "COREGISTRATION_REQUIRED", "; ".join(issues))
        result.warnings = warnings
        result.execution_time_s = elapsed
        return result

    result = ModuleResult.success(
        ModuleID.M11,
        {
            "pair_valid": True,
            "crs_compatible": crs_result.data.get("compatible"),
            "estimated_shift_px": list(shift) if shift is not None else None,
            "shift_magnitude_px": shift_magnitude,
            "coregistration_needed": coregistration_needed,
            "coregistration_applied": coregistered_path is not None,
            "coregistered_path": coregistered_path,
            "metadata_a": meta_a,
            "metadata_b": meta_b,
        },
        exec_time=elapsed,
    )
    result.warnings = warnings
    return result


def get_preprocessing_summary() -> dict[str, Any]:
    """Return which co-registration/despeckle backends are available."""
    return {
        "arosics_available": PreprocessingAdapter.is_arosics_available(),
        "snap_available": PreprocessingAdapter.is_snap_available(),
        "fft_fallback_available": True,
        "shift_tolerance_px": _SHIFT_TOLERANCE_PX,
        "max_nodata_fraction": _MAX_NODATA_FRACTION,
    }
