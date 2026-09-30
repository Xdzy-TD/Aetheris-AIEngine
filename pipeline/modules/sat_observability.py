"""
SAT-OS Spatial Observability — generates an observability mask.

Classifies every pixel as observable / partially_observable / unobservable
based on nodata, cloud/quality bands, and valid-data coverage. Pure numpy
over existing raster I/O — no new deps.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.schemas.sat_schemas import ObservabilityStats

# Observability class values in the output mask
OBS_OBSERVABLE = 255
OBS_PARTIAL = 128
OBS_UNOBSERVABLE = 0


def _load_array(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load raster via existing specialists._imagery or PIL fallback."""
    try:
        from specialists._imagery import load_pil_with_meta
        img, meta = load_pil_with_meta(path)
        return np.asarray(img, dtype=np.float64), meta
    except Exception:
        from PIL import Image
        return np.asarray(Image.open(path), dtype=np.float64), {"georeferenced": False, "crs": None}


def compute_observability(
    file_path: str,
    nodata_value: float | None = None,
    cloud_threshold: float = 0.9,
    output_dir: str | None = None,
) -> ObservabilityStats:
    """Generate an observability raster/mask and return statistics.

    Observable: valid data, not cloud/saturated.
    Partially observable: valid but near-edge quality (high brightness, low contrast).
    Unobservable: nodata, NaN, or cloud-like.
    """
    arr, meta = _load_array(file_path)
    h, w = arr.shape[:2]

    # Flatten to grayscale mean if multi-band
    if arr.ndim == 3:
        gray = arr.mean(axis=-1)
    else:
        gray = arr

    # Normalize to [0,1]
    vmin, vmax = np.nanmin(gray), np.nanmax(gray)
    if vmax > vmin:
        norm = (gray - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(gray)

    # Build mask
    mask = np.full((h, w), OBS_OBSERVABLE, dtype=np.uint8)

    # Unobservable: NaN, Inf, nodata
    invalid = np.isnan(gray) | np.isinf(gray)
    if nodata_value is not None:
        invalid |= (gray == nodata_value)
    mask[invalid] = OBS_UNOBSERVABLE

    # Cloud proxy: pixels near saturation in optical imagery
    cloud_like = (norm > cloud_threshold) & ~invalid
    mask[cloud_like] = OBS_UNOBSERVABLE

    # Partially observable: near-threshold pixels (edge quality)
    edge_quality = (norm > cloud_threshold * 0.8) & (norm <= cloud_threshold) & ~invalid
    mask[edge_quality] = OBS_PARTIAL

    total_pixels = h * w
    observable_pct = round(float(np.sum(mask == OBS_OBSERVABLE)) / total_pixels * 100, 2)
    partial_pct = round(float(np.sum(mask == OBS_PARTIAL)) / total_pixels * 100, 2)
    unobservable_pct = round(float(np.sum(mask == OBS_UNOBSERVABLE)) / total_pixels * 100, 2)

    # Save mask
    mask_path = None
    if output_dir:
        from PIL import Image
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(file_path).stem
        mask_path = str(out / f"{stem}_observability.png")
        Image.fromarray(mask).save(mask_path)

    return ObservabilityStats(
        observable_pct=observable_pct,
        partially_observable_pct=partial_pct,
        unobservable_pct=unobservable_pct,
        mask_path=mask_path,
    )
