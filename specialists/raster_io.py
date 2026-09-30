"""
Raster I/O — safe, geo-aware image loading for all specialists.

Extracted from ``specialists/_imagery.py`` so raster loading, geo-metadata
probing, and GeoTIFF guards live in one place that both ``controller/``
(upload boundary, artifacts) and ``specialists/`` (pixel analysis) can import
without a layering violation.

Every specialist reads pixels through :func:`load_raster` / :func:`load_pil`,
so there is no side door back to a bare ``Image.open``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

try:  # optional: the geospatial extra
    import rasterio
except ImportError:  # pragma: no cover - exercised by the no-rasterio path
    rasterio = None  # type: ignore[assignment]

# Pillow modes that are already 8-bit-per-channel, so `.convert()` is lossless
# in the ways that matter here. Anything else ("I", "I;16", "F") is a
# higher-bit-depth raster whose values `.convert()` would crush into 0-255.
_EIGHT_BIT_MODES = frozenset({"1", "L", "LA", "P", "PA", "RGB", "RGBA"})


def open_checked(path: str) -> Image.Image:
    """Open an image, refusing rasters these 8-bit heuristics would misread.

    The thresholds in this module are calibrated for 8-bit RGB. Handing them a
    16-bit reflectance product or a multispectral stack does not fail loudly on
    its own — Pillow quietly rescales or drops bands and every downstream
    statistic then looks perfectly plausible while being wrong. A raster we
    cannot read correctly must raise, not answer.
    """
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Image not found: {path!r}")
    img = Image.open(path)
    if img.mode not in _EIGHT_BIT_MODES:
        raise ValueError(
            f"{path!r} has Pillow mode {img.mode!r} (deeper than 8-bit). Converting it "
            f"would compress its radiometry into 0-255 and yield confident nonsense. "
            f"Read it with rasterio/GDAL and scale the bands explicitly first."
        )
    if len(img.getbands()) > 3:
        warnings.warn(
            f"{path!r} has {len(img.getbands())} bands; only the first 3 are used. "
            f"If band 4 is NIR, any vegetation/water index computed here is invalid.",
            stacklevel=3,
        )
    return img


# TIFF tags that carry georeferencing. If any is present, the file is a
# GeoTIFF and Pillow is the wrong reader: it can return the pixels but drops
# CRS/transform/NoData entirely, and a lost CRS is not recoverable downstream.
_GEO_TIFF_TAGS = (33550, 33922, 34264, 34735)  # PixelScale, Tiepoint, Transform, GeoKeyDir

# Little- and big-endian TIFF magic. Sniffing the header rather than trusting
# the extension matters because the upload filename is client-supplied: a
# GeoTIFF sent as "scene.png" must not slip past the guard below.
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")


def _is_tiff(path: str) -> bool:
    """Whether the file's own header says TIFF. False if it cannot be read."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) in _TIFF_MAGIC
    except OSError:
        return False  # let the reader below raise the specific error


def load_raster(path: str) -> tuple[NDArray[np.floating], dict[str, Any]]:
    """Load a raster as ``(H, W, 3)`` float in [0, 1] plus its geospatial metadata.

    Rasterio/GDAL is the native reader: it preserves CRS, transform,
    resolution, bounds, band count and NoData, which are returned alongside
    the pixels so downstream steps can georeference their outputs. Pillow is
    used only for ordinary 8-bit images that carry no georeferencing to lose.

    Raises:
        ValueError: for a georeferenced raster when rasterio is unavailable.
            Reading it with Pillow would succeed while silently discarding the
            CRS, so refusing is the only honest option.
    """
    is_tiff = _is_tiff(path)

    if is_tiff and rasterio is not None:
        return _load_with_rasterio(path)

    if is_tiff:
        with Image.open(path) as probe:
            geo_tags = sorted(t for t in _GEO_TIFF_TAGS if t in getattr(probe, "tag_v2", {}))
        if geo_tags:
            raise ValueError(
                f"{path!r} is a GeoTIFF (tags {geo_tags}) but rasterio is not installed. "
                f"Pillow would read the pixels and silently drop the CRS, transform and "
                f"NoData mask. Install the geospatial extra: pip install -e '.[geo]'."
            )

    return _load_pillow(path), _no_geo_meta("pillow")


def _load_pillow(path: str) -> NDArray[np.floating]:
    """8-bit read for images with no georeferencing to preserve."""
    img = open_checked(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def _no_geo_meta(reader: str) -> dict[str, Any]:
    """Metadata record for a raster that carries no georeferencing at all."""
    return {
        "georeferenced": False, "reader": reader, "crs": None, "transform": None,
        "resolution_m": None, "bounds": None, "band_count": None, "bands": [],
        "nodata": None, "nodata_fraction": 0.0, "dtype": None,
    }


def _load_with_rasterio(path: str) -> tuple[NDArray[np.floating], dict[str, Any]]:
    """Read via rasterio, preserving every geospatial field the file declares."""
    with rasterio.open(path) as src:
        raw = src.read(indexes=list(range(1, min(src.count, 3) + 1)))
        crs = src.crs
        meta: dict[str, Any] = {
            "georeferenced": crs is not None,
            "reader": "rasterio",
            "crs": crs.to_string() if crs else None,
            "transform": list(src.transform)[:6],
            "resolution_m": abs(src.transform.a) if crs and crs.is_projected else None,
            "bounds": list(src.bounds),
            "band_count": src.count,
            "bands": [d for d in src.descriptions if d] or [f"band_{i}" for i in src.indexes],
            "nodata": src.nodata,
            "dtype": str(src.dtypes[0]),
        }

    arr = raw.astype(np.float64)
    # NoData must not be averaged in as if it were a real reflectance value.
    if meta["nodata"] is not None:
        invalid = raw == meta["nodata"]
        meta["nodata_fraction"] = round(float(invalid.mean()), 4)
        arr[invalid] = 0.0
    else:
        meta["nodata_fraction"] = 0.0

    # Scale to [0, 1] using the declared integer range, or the scene's own range
    # for float products. Explicit, unlike Pillow's implicit 0-255 crush.
    info = np.iinfo(raw.dtype) if np.issubdtype(raw.dtype, np.integer) else None
    if info is not None and info.max > 0:
        arr /= float(info.max)
    elif arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())

    arr = np.moveaxis(arr, 0, -1)
    while arr.shape[-1] < 3:  # 1-band, or 2-band dual-pol SAR (VV+VH) → pad
        arr = np.concatenate([arr, arr[..., -1:]], axis=-1)
    return np.clip(arr[..., :3], 0.0, 1.0), meta


def load_rgb(path: str) -> NDArray[np.floating]:
    """Load a raster as an (H, W, 3) float array in [0, 1]. Raises on failure.

    Thin wrapper over :func:`load_raster`, so callers that do not need the
    metadata still cannot accidentally read a GeoTIFF through Pillow. Use
    :func:`load_raster` directly when the CRS matters.
    """
    return load_raster(path)[0]


def load_pil(path: str, mode: str = "RGB", size: tuple[int, int] | None = None) -> Image.Image:
    """Geo-safe Pillow image, for the call sites that need ``resize``/``convert``.

    Routes through :func:`load_raster` first, so a georeferenced raster is
    never handed to Pillow behind the caller's back. For ordinary 8-bit input
    this round-trips the pixels unchanged.
    """
    arr = load_raster(path)[0]
    img = Image.fromarray((arr * 255.0).round().clip(0, 255).astype(np.uint8), mode="RGB")
    if mode != "RGB":
        img = img.convert(mode)
    return img.resize(size) if size else img


def load_pil_with_meta(path: str, mode: str = "RGB") -> tuple[Image.Image, dict[str, Any]]:
    """Same as :func:`load_pil`, plus the geospatial metadata ``load_pil`` drops.

    Callers that only need pixels (grounding, VQA, SAR bridging) keep using
    ``load_pil``; callers that must carry CRS/transform through to an output
    raster (bi-temporal change detection) use this instead, so a GeoTIFF's
    georeferencing survives into the produced change map.
    """
    arr, meta = load_raster(path)
    img = Image.fromarray((arr * 255.0).round().clip(0, 255).astype(np.uint8), mode="RGB")
    return (img if mode == "RGB" else img.convert(mode)), meta


def probe_raster_metadata(path: str) -> dict[str, Any]:
    """Read a raster's geospatial header without loading its pixels.

    Used at the upload boundary. Returns the ``_no_geo_meta`` record for plain
    images, unreadable files, or a GeoTIFF with no rasterio to read it — the
    caller gets ``georeferenced: False`` rather than a fabricated CRS.
    """
    if rasterio is None or not _is_tiff(path):
        return _no_geo_meta("none")
    try:
        with rasterio.open(path) as src:
            crs = src.crs
            return {
                "georeferenced": crs is not None,
                "reader": "rasterio",
                "crs": crs.to_string() if crs else None,
                "transform": list(src.transform)[:6],
                "resolution_m": abs(src.transform.a) if crs and crs.is_projected else None,
                "bounds": list(src.bounds),
                "band_count": src.count,
                "bands": [d for d in src.descriptions if d] or [],
                "nodata": src.nodata,
                "nodata_fraction": 0.0,
                "dtype": str(src.dtypes[0]),
            }
    except Exception as exc:  # unreadable/corrupt: say nothing rather than guess
        warnings.warn(f"could not read raster metadata from {path!r}: {exc}", stacklevel=2)
        return _no_geo_meta("none")
