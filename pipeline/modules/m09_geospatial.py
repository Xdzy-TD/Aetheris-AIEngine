"""
M09 — CRS & Geospatial Foundation.

Validates coordinate reference systems, spatial metadata, bounds, and
resolution using the old rasterio-based utilities in ``specialists._imagery``.
"""

from __future__ import annotations

import time
from typing import Any

from pipeline.adapters.imagery_adapter import ImageryAdapter
from pipeline.schemas.contracts import ModuleID, ModuleResult


# Common EPSG codes for satellite imagery
COMMON_CRS = {
    "EPSG:4326": "WGS 84 (Geographic)",
    "EPSG:3857": "Web Mercator",
    "EPSG:32643": "WGS 84 / UTM zone 43N",
    "EPSG:32644": "WGS 84 / UTM zone 44N",
    "EPSG:32645": "WGS 84 / UTM zone 45N",
}

# Reasonable bounds for satellite imagery
_MAX_RESOLUTION_M = 1000.0  # 1km maximum
_MIN_RESOLUTION_M = 0.1     # 10cm minimum


def validate_crs(path: str) -> ModuleResult:
    """Validate the CRS of a raster file."""
    t0 = time.perf_counter()

    # Check rasterio availability
    if not ImageryAdapter.check_rasterio_available():
        return ModuleResult.success(
            ModuleID.M09,
            {
                "crs_valid": None,
                "rasterio_available": False,
                "note": "Rasterio not available — CRS validation skipped",
            },
            exec_time=round(time.perf_counter() - t0, 4),
        )

    # Probe metadata using old imagery code
    meta = ImageryAdapter.probe_metadata(path)
    elapsed = round(time.perf_counter() - t0, 4)

    if meta.get("error"):
        return ModuleResult.failure(
            ModuleID.M09,
            "CRS_PROBE_ERROR",
            f"Failed to read raster metadata: {meta['error']}",
        )

    crs = meta.get("crs")
    georef = meta.get("georeferenced", False)

    if not georef:
        return ModuleResult.success(
            ModuleID.M09,
            {
                "crs_valid": False,
                "georeferenced": False,
                "crs": None,
                "note": "File has no geospatial reference — plain image",
                "metadata": meta,
            },
            exec_time=elapsed,
        )

    # Validate CRS
    warnings = []
    crs_info = COMMON_CRS.get(crs, "Unknown CRS")
    if crs and crs not in COMMON_CRS:
        warnings.append(f"CRS '{crs}' is not in common satellite imagery CRS list")

    # Validate resolution
    resolution = meta.get("resolution_m")
    if resolution is not None:
        if resolution < _MIN_RESOLUTION_M:
            warnings.append(f"Resolution {resolution}m is unusually fine (< {_MIN_RESOLUTION_M}m)")
        if resolution > _MAX_RESOLUTION_M:
            warnings.append(f"Resolution {resolution}m is unusually coarse (> {_MAX_RESOLUTION_M}m)")

    result = ModuleResult.success(
        ModuleID.M09,
        {
            "crs_valid": True,
            "georeferenced": True,
            "crs": crs,
            "crs_description": crs_info,
            "transform": meta.get("transform"),
            "resolution_m": resolution,
            "bounds": meta.get("bounds"),
            "band_count": meta.get("band_count"),
            "bands": meta.get("bands", []),
            "reader": meta.get("reader"),
        },
        exec_time=elapsed,
    )
    result.warnings = warnings
    return result


def check_crs_compatibility(crs1: str | None, crs2: str | None) -> ModuleResult:
    """Check if two CRS are compatible for analysis."""
    t0 = time.perf_counter()

    if crs1 is None and crs2 is None:
        return ModuleResult.success(
            ModuleID.M09,
            {"compatible": True, "note": "Both images lack CRS — treated as same space"},
            exec_time=round(time.perf_counter() - t0, 4),
        )

    if crs1 is None or crs2 is None:
        return ModuleResult.failure(
            ModuleID.M09,
            "CRS_MISMATCH",
            f"One image has CRS ({crs1 or crs2}) but the other does not",
        )

    if crs1 == crs2:
        return ModuleResult.success(
            ModuleID.M09,
            {"compatible": True, "crs": crs1, "note": "CRS match"},
            exec_time=round(time.perf_counter() - t0, 4),
        )

    # Different CRS — may need reprojection
    result = ModuleResult.success(
        ModuleID.M09,
        {
            "compatible": False,
            "crs1": crs1,
            "crs2": crs2,
            "note": "CRS differ — reprojection recommended before analysis",
            "reprojection_needed": True,
        },
        exec_time=round(time.perf_counter() - t0, 4),
    )
    result.warnings = [f"CRS mismatch: {crs1} vs {crs2}"]
    return result


def get_spatial_summary(path: str) -> dict[str, Any]:
    """Get a comprehensive spatial summary of a raster file."""
    meta = ImageryAdapter.probe_metadata(path)
    return {
        "path": path,
        "georeferenced": meta.get("georeferenced", False),
        "crs": meta.get("crs"),
        "transform": meta.get("transform"),
        "resolution_m": meta.get("resolution_m"),
        "bounds": meta.get("bounds"),
        "band_count": meta.get("band_count"),
        "bands": meta.get("bands", []),
        "nodata": meta.get("nodata"),
        "dtype": meta.get("dtype"),
        "reader": meta.get("reader"),
    }
