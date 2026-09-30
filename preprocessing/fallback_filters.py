"""
Fallback speckle filters — pure NumPy / SciPy implementations.

These are classical adaptive filters for SAR speckle noise reduction.
They serve as a zero-dependency fallback when ESA SNAP is not installed
or when a lightweight, in-process solution is preferred.

References
----------
Lee, J.-S. (1980). "Digital Image Enhancement and Noise Filtering by
    Use of Local Statistics." IEEE TPAMI.
Frost, V.S. et al. (1982). "A Model for Radar Images and Its
    Application to Adaptive Digital Filtering." IEEE TPAMI.
Kuan, D.T. et al. (1985). "Adaptive Noise Smoothing Filter for Images
    with Signal-Dependent Noise." IEEE TPAMI.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import structlog
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

logger = structlog.get_logger(__name__)


def lee_filter(
    image: NDArray[np.floating],
    window_size: int = 7,
    num_looks: int = 1,
) -> NDArray[np.floating]:
    """Apply the Lee adaptive speckle filter.

    The Lee filter estimates the local mean and variance within a sliding
    window to weight between the pixel value and the local mean.  Regions
    with high variance (edges / features) are preserved; homogeneous
    regions are smoothed.

    Args:
        image:       2-D array of SAR intensity values (linear scale).
        window_size: Side length of the square filter window (must be odd).
        num_looks:   Number of looks (controls expected speckle variance).

    Returns:
        Filtered image of the same shape and dtype family.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")
    window_size = _ensure_odd(window_size)

    img = image.astype(np.float64)

    # Local statistics via uniform (box) filter
    local_mean = uniform_filter(img, size=window_size)
    local_sq_mean = uniform_filter(img ** 2, size=window_size)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0.0)  # numerical safety

    # Noise variance estimate (multiplicative speckle model)
    noise_var = local_mean ** 2 / max(num_looks, 1)

    # Weighting factor
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(
            local_var > 0,
            np.clip(1.0 - noise_var / local_var, 0.0, 1.0),
            0.0,
        )

    filtered = local_mean + weight * (img - local_mean)
    logger.debug("lee_filter_applied", window=window_size, num_looks=num_looks)
    return cast("NDArray[np.floating]", filtered.astype(image.dtype))


def frost_filter(
    image: NDArray[np.floating],
    window_size: int = 7,
    damping_factor: float = 2.0,
) -> NDArray[np.floating]:
    """Apply the Frost exponentially-damped filter.

    The Frost filter uses an exponential weighting kernel whose decay
    rate adapts to the local coefficient of variation (ratio of standard
    deviation to mean).  High-variation areas receive less smoothing.

    Args:
        image:          2-D array of SAR intensity values.
        window_size:    Side length of the square filter window (odd).
        damping_factor: Controls the rate of exponential decay (higher →
                        more smoothing in homogeneous areas).

    Returns:
        Filtered image.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")
    window_size = _ensure_odd(window_size)
    half = window_size // 2

    img = image.astype(np.float64)
    rows, cols = img.shape
    filtered = np.zeros_like(img)

    # Pad image with reflection to handle borders
    padded = np.pad(img, half, mode="reflect")

    # Pre-compute local mean and variance for the padded image
    local_mean = uniform_filter(padded, size=window_size)
    local_sq_mean = uniform_filter(padded ** 2, size=window_size)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    # Coefficient of variation squared
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_sq = np.where(local_mean > 0, local_var / (local_mean ** 2), 0.0)

    # Distance kernel (Euclidean distance from center)
    y_coords, x_coords = np.mgrid[-half:half + 1, -half:half + 1]
    dist = np.sqrt(x_coords ** 2 + y_coords ** 2).astype(np.float64)

    for i in range(rows):
        for j in range(cols):
            pi, pj = i + half, j + half
            alpha = damping_factor * cv_sq[pi, pj]
            kernel = np.exp(-alpha * dist)
            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                window = padded[pi - half:pi + half + 1, pj - half:pj + half + 1]
                filtered[i, j] = np.sum(kernel * window) / kernel_sum
            else:
                filtered[i, j] = img[i, j]

    logger.debug("frost_filter_applied", window=window_size, damping=damping_factor)
    return cast("NDArray[np.floating]", filtered.astype(image.dtype))


def kuan_filter(
    image: NDArray[np.floating],
    window_size: int = 7,
    num_looks: int = 1,
) -> NDArray[np.floating]:
    """Apply the Kuan adaptive filter.

    Similar to the Lee filter but uses a different weighting formulation
    based on the equivalent number of looks (ENL).  It is unbiased for
    the multiplicative speckle model.

    Args:
        image:       2-D array of SAR intensity values.
        window_size: Side length of the square filter window (odd).
        num_looks:   Number of looks.

    Returns:
        Filtered image.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")
    window_size = _ensure_odd(window_size)

    img = image.astype(np.float64)

    local_mean = uniform_filter(img, size=window_size)
    local_sq_mean = uniform_filter(img ** 2, size=window_size)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    # ENL-based noise variance
    enl = max(float(num_looks), 1.0)
    cu_sq = 1.0 / enl  # coefficient of variation of speckle noise squared

    with np.errstate(divide="ignore", invalid="ignore"):
        ci_sq = np.where(
            local_mean > 0,
            local_var / (local_mean ** 2),
            0.0,
        )

    # Kuan weight
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(
            ci_sq > 0,
            np.clip((1.0 - cu_sq / ci_sq) / (1.0 + cu_sq), 0.0, 1.0),
            0.0,
        )

    filtered = local_mean + weight * (img - local_mean)
    logger.debug("kuan_filter_applied", window=window_size, num_looks=num_looks)
    return cast("NDArray[np.floating]", filtered.astype(image.dtype))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_odd(n: int) -> int:
    """Round up to the nearest odd integer if even."""
    return n if n % 2 == 1 else n + 1
