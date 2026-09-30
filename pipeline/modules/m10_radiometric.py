"""
M10 — NoData & Radiometric Correctness.

Validates NoData handling, NaN/Inf detection, dtype preservation, numeric
ranges, and radiometric transformation provenance.  Reuses the old
``load_raster`` metadata for nodata, dtype, and nodata_fraction.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pipeline.adapters.imagery_adapter import ImageryAdapter
from pipeline.schemas.contracts import ModuleID, ModuleResult


def validate_radiometric(path: str) -> ModuleResult:
    """Full radiometric validation of a raster file."""
    t0 = time.perf_counter()
    issues: list[str] = []
    warnings: list[str] = []
    provenance: dict[str, Any] = {}

    # Check rasterio availability
    if not ImageryAdapter.check_rasterio_available():
        return ModuleResult.success(
            ModuleID.M10,
            {
                "validated": False,
                "rasterio_available": False,
                "note": "Rasterio not available — radiometric validation limited",
            },
            exec_time=round(time.perf_counter() - t0, 4),
        )

    # Load raster with metadata
    try:
        arr, meta = ImageryAdapter.load_raster(path)
    except Exception as exc:
        return ModuleResult.failure(
            ModuleID.M10,
            "RASTER_LOAD_ERROR",
            f"Failed to load raster: {exc}",
        )

    # 1. NoData validation
    nodata = meta.get("nodata")
    nodata_fraction = meta.get("nodata_fraction", 0.0)
    provenance["nodata_declared"] = nodata
    provenance["nodata_fraction"] = nodata_fraction

    if nodata is not None:
        if nodata_fraction > 0.9:
            issues.append(
                f"Excessive NoData: {nodata_fraction*100:.1f}% of pixels are NoData"
            )
        elif nodata_fraction > 0.5:
            warnings.append(
                f"High NoData fraction: {nodata_fraction*100:.1f}% of pixels"
            )
    else:
        if meta.get("georeferenced"):
            warnings.append("No NoData value declared in georeferenced raster")

    # 2. NaN/Inf detection
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    total_pixels = arr.size

    provenance["nan_count"] = nan_count
    provenance["inf_count"] = inf_count

    if nan_count > 0:
        nan_pct = nan_count / total_pixels * 100
        if nan_pct > 10:
            issues.append(f"Excessive NaN values: {nan_count} ({nan_pct:.1f}%)")
        else:
            warnings.append(f"Contains {nan_count} NaN values ({nan_pct:.2f}%)")

    if inf_count > 0:
        issues.append(f"Contains {inf_count} Inf values — radiometric error")

    # 3. Dtype preservation
    dtype = meta.get("dtype", str(arr.dtype))
    provenance["dtype"] = dtype
    provenance["array_dtype"] = str(arr.dtype)

    # 4. Numeric range checks
    valid_mask = ~(np.isnan(arr) | np.isinf(arr))
    if valid_mask.any():
        vmin = float(np.min(arr[valid_mask]))
        vmax = float(np.max(arr[valid_mask]))
        vmean = float(np.mean(arr[valid_mask]))
        vstd = float(np.std(arr[valid_mask]))
    else:
        vmin = vmax = vmean = vstd = 0.0
        issues.append("No valid (non-NaN, non-Inf) pixels found")

    provenance["value_range"] = {"min": round(vmin, 6), "max": round(vmax, 6)}
    provenance["statistics"] = {
        "mean": round(vmean, 6),
        "std": round(vstd, 6),
    }

    # Check if values are in expected [0, 1] range (after load_raster normalization)
    if vmax > 1.0 or vmin < 0.0:
        warnings.append(
            f"Values outside [0, 1] range after normalization: [{vmin:.4f}, {vmax:.4f}]"
        )

    # 5. Band consistency
    if arr.ndim == 3:
        band_count = arr.shape[-1]
        band_means = [float(np.mean(arr[..., i])) for i in range(band_count)]
        provenance["band_means"] = [round(m, 4) for m in band_means]

        # Check for dead bands (all zero or constant)
        for i, mean in enumerate(band_means):
            band_std = float(np.std(arr[..., i]))
            if band_std < 1e-8:
                warnings.append(f"Band {i} appears constant (std={band_std:.2e})")

    # 6. Spatial extent
    provenance["shape"] = list(arr.shape)
    provenance["total_pixels"] = total_pixels

    elapsed = round(time.perf_counter() - t0, 4)

    if issues:
        result = ModuleResult.failure(
            ModuleID.M10,
            "RADIOMETRIC_ISSUES",
            "; ".join(issues),
        )
        result.data = {
            "validated": False,
            "issues": issues,
            "provenance": provenance,
        }
        result.warnings = warnings
        result.execution_time_s = elapsed
        return result

    result = ModuleResult.success(
        ModuleID.M10,
        {
            "validated": True,
            "provenance": provenance,
        },
        exec_time=elapsed,
    )
    result.warnings = warnings
    return result


def get_radiometric_summary(path: str) -> dict[str, Any]:
    """Get a radiometric summary without full validation."""
    meta = ImageryAdapter.probe_metadata(path)
    return {
        "path": path,
        "nodata": meta.get("nodata"),
        "nodata_fraction": meta.get("nodata_fraction", 0.0),
        "dtype": meta.get("dtype"),
        "band_count": meta.get("band_count"),
        "reader": meta.get("reader"),
    }
