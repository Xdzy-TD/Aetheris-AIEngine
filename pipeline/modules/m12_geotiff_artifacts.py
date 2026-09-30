"""
M12 — GeoTIFF Artifact Preservation.

Every derived raster a specialist produces (change map, fusion overlay,
grounding mask, despeckled SAR) is written by the old
``controller.artifacts.save_image`` as a plain PNG — PNG has no field for
CRS/transform, so an artifact derived from a georeferenced scene loses its
georeferencing the moment it's saved and can't be laid back on a map. That
write path is left untouched (old code, used as-is); M12 instead writes a
*sibling* GeoTIFF next to it whenever the source image carried a CRS, with
the transform rescaled to match the artifact's (often resized) pixel grid.
"""

from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image

from pipeline.adapters.artifact_adapter import ArtifactAdapter
from pipeline.adapters.imagery_adapter import ImageryAdapter
from pipeline.schemas.contracts import ModuleID, ModuleResult


def preserve_geotiff(
    artifact_path: str,
    source_path: str,
    artifact_type: str = "derived_artifact",
    model_version: str = "m12@1.0",
) -> ModuleResult:
    """Write a georeferenced GeoTIFF sibling of ``artifact_path``, if warranted.

    Returns success with ``preserved: False`` (not a failure) when the source
    has no CRS or rasterio is unavailable — nothing was lost in that case,
    PNG already kept everything the source had.
    """
    t0 = time.perf_counter()
    if not ImageryAdapter.check_rasterio_available():
        return ModuleResult.success(
            ModuleID.M12, {"preserved": False, "note": "rasterio unavailable"},
            exec_time=round(time.perf_counter() - t0, 4),
        )

    meta = ImageryAdapter.probe_metadata(source_path)
    if not meta.get("georeferenced") or not meta.get("transform") or not meta.get("crs"):
        return ModuleResult.success(
            ModuleID.M12,
            {"preserved": False, "note": "source has no CRS/transform to carry over"},
            exec_time=round(time.perf_counter() - t0, 4),
        )

    try:
        import rasterio
        from affine import Affine
        from rasterio.crs import CRS
        from rasterio.io import MemoryFile

        with rasterio.open(source_path) as src:
            src_w, src_h = src.width, src.height

        arr = np.asarray(Image.open(artifact_path))
        out_h, out_w = arr.shape[0], arr.shape[1]
        band_count = 1 if arr.ndim == 2 else arr.shape[-1]
        data = arr if arr.ndim == 2 else np.moveaxis(arr, -1, 0)

        # The artifact is frequently a resized copy of the source (e.g. the
        # 256x256 change-detection grid) — rescale the affine transform onto
        # the artifact's own grid rather than assuming a 1:1 pixel match.
        transform = Affine(*meta["transform"]) * Affine.scale(src_w / out_w, src_h / out_h)

        buf = io.BytesIO()
        with MemoryFile() as mem:
            with mem.open(
                driver="GTiff", height=out_h, width=out_w, count=band_count,
                dtype=str(arr.dtype), crs=CRS.from_string(meta["crs"]), transform=transform,
            ) as dst:
                if band_count == 1:
                    dst.write(data, 1)
                else:
                    dst.write(data)
            buf.write(mem.read())

        record = ArtifactAdapter().store_artifact(
            buf.getvalue(), f"{artifact_type}_geotiff", model_version, suffix=".tif",
        )
        return ModuleResult.success(
            ModuleID.M12,
            {"preserved": True, "crs": meta["crs"], "geotiff_artifact": record},
            exec_time=round(time.perf_counter() - t0, 4),
        )
    except Exception as exc:
        result = ModuleResult.failure(ModuleID.M12, "GEOTIFF_WRITE_FAILED", str(exc))
        result.execution_time_s = round(time.perf_counter() - t0, 4)
        return result
