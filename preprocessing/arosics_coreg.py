"""
Optical ↔ SAR co-registration.

Primary method: ``arosics`` (pip-installable, purpose-built for remote
sensing imagery co-registration using phase correlation).

Fallback: OpenCV ``findTransformECC`` or SIFT/ORB + homography when
arosics is unavailable (e.g., missing GDAL bindings).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def is_arosics_available() -> bool:
    """Check whether ``arosics`` is importable."""
    try:
        import arosics  # noqa: F401
        return True
    except ImportError:
        return False


def _is_opencv_available() -> bool:
    """Check whether ``cv2`` is importable."""
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def coregister(
    reference_path: str | Path,
    target_path: str | Path,
    output_path: str | Path | None = None,
    method: Literal["auto", "arosics", "ecc", "feature"] = "auto",
    max_shift: int = 50,
) -> Path:
    """Co-register a target image to a reference image.

    Args:
        reference_path: Path to the reference (master) image.
        target_path:    Path to the target (slave) image to warp.
        output_path:    Where to write the co-registered result.
                        Defaults to ``<target_stem>_coreg.tif``.
        method:         ``"auto"`` tries arosics → ECC → feature matching.
                        Specify explicitly to force a method.
        max_shift:      Maximum allowed shift in pixels (safety guard).

    Returns:
        Path to the co-registered output image.

    Raises:
        RuntimeError: If no method succeeded.
    """
    reference_path = Path(reference_path)
    target_path = Path(target_path)
    if output_path is None:
        output_path = target_path.with_name(f"{target_path.stem}_coreg.tif")
    output_path = Path(output_path)

    if method == "auto":
        # Try methods in order of preference
        for m in ("arosics", "ecc", "feature"):
            try:
                return _dispatch(m, reference_path, target_path, output_path, max_shift)
            except Exception as exc:
                logger.warning("coreg_method_failed", method=m, error=str(exc))
        raise RuntimeError("All co-registration methods failed.")
    else:
        return _dispatch(method, reference_path, target_path, output_path, max_shift)


def _dispatch(
    method: str,
    ref: Path,
    tgt: Path,
    out: Path,
    max_shift: int,
) -> Path:
    """Dispatch to the chosen co-registration backend."""
    if method == "arosics":
        return _coreg_arosics(ref, tgt, out, max_shift)
    elif method == "ecc":
        return _coreg_ecc(ref, tgt, out, max_shift)
    elif method == "feature":
        return _coreg_feature(ref, tgt, out, max_shift)
    else:
        raise ValueError(f"Unknown co-registration method: {method!r}")


# ---------------------------------------------------------------------------
# AROSICS backend
# ---------------------------------------------------------------------------

def _coreg_arosics(ref: Path, tgt: Path, out: Path, max_shift: int) -> Path:
    """Co-register using AROSICS local co-registration."""
    if not is_arosics_available():
        raise ImportError("arosics is not installed")

    from arosics import COREG

    cr = COREG(
        str(ref),
        str(tgt),
        path_out=str(out),
        max_shift=max_shift,
        fmt_out="GTiff",
    )
    cr.correct_shifts()

    logger.info("coreg_arosics_complete", output=str(out))
    return out


# ---------------------------------------------------------------------------
# OpenCV ECC backend
# ---------------------------------------------------------------------------

def _coreg_ecc(ref: Path, tgt: Path, out: Path, max_shift: int) -> Path:
    """Co-register using OpenCV Enhanced Correlation Coefficient."""
    if not _is_opencv_available():
        raise ImportError("OpenCV (cv2) is not installed")

    import cv2

    # Load as grayscale
    ref_img = cv2.imread(str(ref), cv2.IMREAD_GRAYSCALE)
    tgt_img: cv2.typing.MatLike | None = cv2.imread(str(tgt), cv2.IMREAD_GRAYSCALE)

    if ref_img is None or tgt_img is None:
        raise FileNotFoundError(f"Could not read images: {ref}, {tgt}")

    # Resize target to match reference if needed
    if ref_img.shape != tgt_img.shape:
        tgt_img = cv2.resize(tgt_img, (ref_img.shape[1], ref_img.shape[0]))

    # ECC alignment (affine)
    warp_matrix: cv2.typing.MatLike = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,  # max iterations
        1e-6,  # epsilon
    )

    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_img, tgt_img, warp_matrix, cv2.MOTION_AFFINE, criteria
        )
    except cv2.error as e:
        raise RuntimeError(f"ECC alignment failed: {e}")

    # Apply warp
    aligned = cv2.warpAffine(
        tgt_img, warp_matrix,
        (ref_img.shape[1], ref_img.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    cv2.imwrite(str(out), aligned)
    logger.info("coreg_ecc_complete", output=str(out))
    return out


# ---------------------------------------------------------------------------
# OpenCV feature-matching backend
# ---------------------------------------------------------------------------

def _coreg_feature(ref: Path, tgt: Path, out: Path, max_shift: int) -> Path:
    """Co-register using SIFT/ORB feature matching + homography."""
    if not _is_opencv_available():
        raise ImportError("OpenCV (cv2) is not installed")

    import cv2

    ref_img = cv2.imread(str(ref), cv2.IMREAD_GRAYSCALE)
    tgt_img = cv2.imread(str(tgt), cv2.IMREAD_GRAYSCALE)

    if ref_img is None or tgt_img is None:
        raise FileNotFoundError(f"Could not read images: {ref}, {tgt}")

    # Try SIFT first, fall back to ORB
    try:
        detector = cv2.SIFT_create(nfeatures=2000)  # type: ignore[attr-defined]
    except AttributeError:
        detector = cv2.ORB_create(nfeatures=2000)  # type: ignore[attr-defined]

    kp1, des1 = detector.detectAndCompute(ref_img, None)
    kp2, des2 = detector.detectAndCompute(tgt_img, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("Not enough keypoints for feature matching")

    # BFMatcher
    if des1.dtype == np.float32:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 4:
        raise RuntimeError(f"Only {len(good_matches)} good matches — need at least 4")

    src_pts = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(
        -1, 1, 2
    )
    dst_pts = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(
        -1, 1, 2
    )

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if homography is None:
        raise RuntimeError("Homography estimation failed")

    aligned = cv2.warpPerspective(tgt_img, homography, (ref_img.shape[1], ref_img.shape[0]))
    cv2.imwrite(str(out), aligned)
    logger.info(
        "coreg_feature_complete",
        output=str(out),
        good_matches=len(good_matches),
        inliers=int(mask.sum()) if mask is not None else 0,
    )
    return out
