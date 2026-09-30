"""
Shared real pixel-analysis helpers — pure NumPy/Pillow, no ML weights.

Every specialist that needs to "look at" an image uses these functions so
outputs are actually derived from the pixels handed in, instead of
keyword-matched canned text. Heuristics are intentionally simple
(spectral-index style thresholds) rather than fabricated: they are honest
about being classical, explainable proxies, not a trained classifier.

Raster I/O (loading, geo-probing, GeoTIFF guards) lives in
:mod:`specialists.raster_io` and is re-exported here so existing callers
(every specialist, ``controller/artifacts.py``, ``interfaces/api/app.py``)
keep working with zero import changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------- Re-export all raster I/O from the new home ----------
from specialists.raster_io import (  # noqa: F401 — re-exports
    load_pil,
    load_pil_with_meta,
    load_raster,
    load_rgb,
    open_checked,
    probe_raster_metadata,
)




def estimate_shift(
    ref: NDArray[np.floating], tgt: NDArray[np.floating]
) -> tuple[int, int]:
    """Integer ``(dy, dx)`` roll that best aligns 2-D ``tgt`` onto ``ref``.

    FFT phase correlation. ``preprocessing.arosics_coreg`` does the real,
    georeferenced job but every one of its backends needs arosics or OpenCV;
    this is the always-available fallback for the in-memory case, on nothing
    but NumPy's own FFT.
    """
    fr, ft = np.fft.fft2(ref), np.fft.fft2(tgt)
    cross = fr * np.conj(ft)
    corr = np.abs(np.fft.ifft2(cross / (np.abs(cross) + 1e-12)))
    dy, dx = (int(v) for v in np.unravel_index(int(np.argmax(corr)), corr.shape))
    h, w = ref.shape
    return (dy - h if dy > h // 2 else dy, dx - w if dx > w // 2 else dx)


_MAX_SHIFT_FRACTION = 0.4  # near the wrap-around limit estimate_shift can return


def validate_shift(dy: int, dx: int, shape: tuple[int, int]) -> bool:
    """Whether an ``estimate_shift`` result is plausible enough to apply.

    Phase correlation always returns *some* peak, even for two genuinely
    unrelated scenes (e.g. cloud cover, seasonal change, sensor noise) — it
    has no way to say "no reliable alignment exists". A shift close to half
    the image (near ``estimate_shift``'s own wrap-around limit) is far more
    likely to be a false correlation peak than a real co-registration error,
    and rolling by it would introduce misalignment, not fix it. Callers
    should skip applying the shift (dy=dx=0) when this returns ``False``.
    """
    h, w = shape[:2]
    return abs(dy) <= _MAX_SHIFT_FRACTION * h and abs(dx) <= _MAX_SHIFT_FRACTION * w


def sar_landcover_masks(
    sar_gray: NDArray[np.floating],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Water/vegetation/urban masks from SAR backscatter, via per-scene percentile bands.

    Real SAR domain physics, not a trained classifier: calm water is a
    specular reflector and returns almost no energy to the sensor (dark);
    urban double-bounce reflection between building walls and the ground
    returns unusually strong energy (bright); vegetation's volume scattering
    sits in between the two. Percentile thresholds keep this adaptive to
    each scene's own dynamic range rather than a fixed cutoff.
    """
    lo, hi = np.quantile(sar_gray, (0.2, 0.8))
    water = sar_gray <= lo
    urban = sar_gray >= hi
    veg = (~water) & (~urban)
    return water, veg, urban


def band_stats(arr: NDArray[np.floating]) -> dict[str, float]:
    """Per-channel mean/std plus overall brightness, from real pixel values."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return {
        "mean_r": round(float(r.mean()), 4),
        "mean_g": round(float(g.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "brightness": round(float(arr.mean()), 4),
        "contrast_std": round(float(arr.std()), 4),
    }


def classify_pixels(arr: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Classify each pixel into 0=water, 1=vegetation, 2=urban/bare, 3=other.

    Classical spectral-index thresholds (excess-green for vegetation,
    blue-dominance for water, high-brightness/low-saturation for
    built-up/bare surfaces). Deliberately simple and explainable rather
    than a trained model with no weights to load.
    """
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = arr.mean(axis=-1)
    sat = arr.max(axis=-1) - arr.min(axis=-1)

    exg = 2 * g - r - b
    water_idx = b - (r + g) / 2.0

    is_water = (water_idx > 0.03) & (b > 0.15)
    is_veg = (~is_water) & (exg > 0.02)
    is_urban = (~is_water) & (~is_veg) & (brightness > 0.35) & (sat < 0.18)

    labels = np.full(brightness.shape, 3, dtype=np.uint8)
    labels[is_urban] = 2
    labels[is_veg] = 1
    labels[is_water] = 0
    return labels


_CLASS_NAMES = {0: "water", 1: "vegetation", 2: "urban/built-up", 3: "bare/other"}


def land_cover_fractions(arr: NDArray[np.floating]) -> dict[str, float]:
    """Real percentage of pixels in each land-cover class."""
    labels = classify_pixels(arr)
    total = labels.size
    return {
        _CLASS_NAMES[k]: round(float(np.sum(labels == k)) / total * 100, 2)
        for k in _CLASS_NAMES
    }


_PROMPT_TO_CLASS = {
    "water": 0, "river": 0, "lake": 0, "flood": 0, "reservoir": 0,
    "vegetation": 1, "forest": 1, "crop": 1, "agricult": 1, "tree": 1,
    "urban": 2, "building": 2, "built": 2, "settlement": 2, "road": 2,
    # Exposed/disturbed ground (class 3, "bare/other"): burn scars, landslide
    # debris and mining/quarry earth all read as low-vegetation, low-water
    # surface in this classifier, so they share the bucket rather than
    # inventing per-hazard classes with no distinct pixel signature here.
    "bare": 3, "landslide": 3, "burn": 3,
    # Coastline/shoreline is the water/land boundary — same class as "water".
    "coast": 0, "shoreline": 0,
}


def mask_for_prompt(arr: NDArray[np.floating], prompt: str) -> tuple[NDArray[np.uint8], int]:
    """Binary mask (0/255) for the land-cover class matching keywords in prompt.

    Returns (mask, class_id). Falls back to an intensity threshold (brightest
    quartile) when no keyword matches, still derived from real pixel values.
    """
    labels = classify_pixels(arr)
    prompt_lower = prompt.lower()
    class_id = next((c for kw, c in _PROMPT_TO_CLASS.items() if kw in prompt_lower), None)

    if class_id is not None:
        mask = (labels == class_id).astype(np.uint8) * 255
        return mask, class_id

    brightness = arr.mean(axis=-1)
    thresh = np.quantile(brightness, 0.75)
    mask = (brightness >= thresh).astype(np.uint8) * 255
    return mask, -1


def detect_prompt_classes(prompt: str) -> list[int]:
    """All distinct land-cover classes named anywhere in ``prompt``.

    Reused by the custom mission generator (``controller/missions.py``) to
    compose one ``grounding`` call per concept in a free-text objective,
    instead of duplicating ``_PROMPT_TO_CLASS``.
    """
    prompt_lower = prompt.lower()
    return sorted({c for kw, c in _PROMPT_TO_CLASS.items() if kw in prompt_lower})


def class_label(class_id: int) -> str:
    """Public accessor for ``_CLASS_NAMES`` — a class's canonical prompt text."""
    return _CLASS_NAMES[class_id]


