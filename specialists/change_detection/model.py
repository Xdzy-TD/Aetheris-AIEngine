"""
Change Detection Specialist — bi-temporal analysis via real pixel differencing.

No Siamese transformer checkpoint exists for this domain; an untrained one
would output random noise. This specialist instead co-registers the pair,
resizes both to a common grid and thresholds the absolute pixel difference —
a classical, fully explainable change-detection baseline that genuinely
reflects what changed between the two acquisitions.

Co-registration is not optional polish. Differencing an unaligned pair
reports the misalignment itself as change, so the baseline would fabricate
"land-use conversion" from a scene that never moved. A trained model on
unaligned inputs would be wrong the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from PIL import Image

from controller.artifacts import save_geotiff
from specialists._imagery import estimate_shift, load_pil, load_pil_with_meta, validate_shift
from specialists.base import BaseSpecialist

logger = structlog.get_logger(__name__)

_TARGET_SIZE = (256, 256)
_DIFF_THRESHOLD = 0.12  # in [0,1] pixel-intensity space


class ChangeDetectionSpecialist(BaseSpecialist):
    """Bi-temporal change detection via thresholded absolute pixel difference."""

    name = "change_detection"

    def __init__(self, **_: Any) -> None:
        super().__init__()

    def load_model(self) -> None:
        """Nothing to load — pure NumPy/Pillow."""

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Expected kwargs: image_t1_path, image_t2_path, modality (default 'optical')."""
        self.ensure_loaded()

        t1_path = kwargs.get("image_t1_path", "")
        t2_path = kwargs.get("image_t2_path", "")
        modality = kwargs.get("modality", "optical")

        try:
            src1, geo_meta = load_pil_with_meta(t1_path)
            src2 = load_pil(t2_path)
        except Exception as exc:
            logger.warning("change_detection_image_unavailable", error=str(exc))
            return self._refusal(f"Could not load one or both images: {exc}")

        # Two rasters describe the same ground only if they cover the same
        # extent. Resizing mismatched extents onto a common grid compares
        # different places and reports the mismatch as "change" — an identical
        # scene clipped differently scores ~23%.
        ar1, ar2 = src1.width / src1.height, src2.width / src2.height
        if abs(ar1 - ar2) > 0.01 * max(ar1, ar2):
            logger.warning("change_detection_extent_mismatch", t1=src1.size, t2=src2.size)
            return self._refusal(
                f"Refusing to difference an unmatched pair: T1 is {src1.width}x{src1.height}, "
                f"T2 is {src2.width}x{src2.height}. Clip both to a common extent first."
            )

        # Co-register before differencing. An unregistered pair otherwise
        # reports its own misalignment as change: shifting an unchanged scene
        # by 10 px scored ~11% "significant change" before this step existed.
        # Estimate on the native grid, not the resized one — correcting a 1 px
        # shift after downsampling leaves a half-pixel residual that still
        # scores ~2%.
        dy = dx = 0
        coregistration_reliable = True
        native1, native2 = np.asarray(src1), np.asarray(src2)
        if native1.shape == native2.shape:
            dy, dx = estimate_shift(
                native1.mean(axis=-1, dtype=np.float64),
                native2.mean(axis=-1, dtype=np.float64),
            )
            # A phase-correlation peak always exists, even for two unrelated
            # scenes; an implausibly large shift is more likely a false peak
            # than a real registration error, so it is not applied.
            coregistration_reliable = validate_shift(dy, dx, native1.shape)
            if coregistration_reliable:
                src2 = Image.fromarray(np.roll(native2, (dy, dx), axis=(0, 1)))
            else:
                logger.warning("change_detection_shift_rejected", shift=[dy, dx])
                dy = dx = 0

        arr1 = np.asarray(src1.resize(_TARGET_SIZE), dtype=np.float64) / 255.0
        arr2 = np.asarray(src2.resize(_TARGET_SIZE), dtype=np.float64) / 255.0

        diff = np.abs(arr1 - arr2).mean(axis=-1)  # (H, W) in [0,1]

        # np.roll wraps, so the wrapped margin holds no real data. Drop it from
        # scoring, converted to target-grid pixels and rounded outwards.
        h, w = diff.shape
        my = int(np.ceil(abs(dy) * h / max(native1.shape[0], 1)))
        mx = int(np.ceil(abs(dx) * w / max(native1.shape[1], 1)))
        valid = np.zeros((h, w), dtype=bool)
        valid[my:h - my, mx:w - mx] = True
        change_mask = ((diff > _DIFF_THRESHOLD) & valid).astype(np.uint8) * 255

        # A georeferenced T1 (real CRS + transform) produces a georeferenced
        # change map instead of a plain PNG — save_geotiff rescales the
        # transform to this 256x256 grid itself; PNG fallback otherwise.
        artifact = save_geotiff(
            change_mask, geo_meta, native1.shape, "change_map", "change_detection"
        )

        changed_pct = float(np.mean(change_mask[valid] > 0) * 100) if valid.any() else 0.0
        changed_area_m2 = (
            round(changed_pct / 100 * valid.sum() * geo_meta["resolution_m"] ** 2, 1)
            if geo_meta.get("resolution_m")
            else None
        )

        # Confidence from the pixels themselves, not invented: how far the
        # valid diff values sit from the decision threshold on average, i.e.
        # how decisively each pixel was classified rather than borderline.
        confidence = (
            round(float(np.clip(np.mean(np.abs(diff[valid] - _DIFF_THRESHOLD)) / _DIFF_THRESHOLD, 0, 1)), 2)
            if valid.any()
            else 0.0
        )

        # Real, if simple, "embedding": mean diff over an 8x4 grid — a
        # genuine compressed signature of *where* the change happened.
        h, w = diff.shape
        gh, gw = 8, 4
        cropped = diff[: h - h % gh, : w - w % gw]
        block_means = cropped.reshape(gh, h // gh, gw, w // gw).mean(axis=(1, 3))
        embedding = [round(float(v), 4) for v in block_means.flatten()]

        return {
            "change_map_path": artifact["path"],
            "artifact": artifact,
            "change_summary": self._generate_summary(changed_pct, modality),
            "changed_area_pct": round(changed_pct, 2),
            "changed_area_m2": changed_area_m2,
            "confidence": confidence,
            "change_embedding": embedding,
            "applied_shift_px": [dy, dx],
            "coregistration_reliable": coregistration_reliable,
            "geo_metadata": {"georeferenced": geo_meta["georeferenced"], "crs": geo_meta["crs"]},
        }

    @staticmethod
    def _refusal(reason: str) -> dict[str, Any]:
        """Uniform "no comparable answer" result, so callers get one shape."""
        return {
            "analysis_unavailable": True,
            "reason": reason,
            "change_map_path": "",
            "change_summary": reason,
            "changed_area_pct": 0.0,
            "changed_area_m2": None,
            "confidence": 0.0,
            "change_embedding": [],
            "applied_shift_px": [0, 0],
            "coregistration_reliable": False,
            "geo_metadata": {"georeferenced": False, "crs": None},
        }

    @staticmethod
    def _generate_summary(changed_pct: float, modality: str) -> str:
        if changed_pct < 2:
            desc = "Minimal change detected between the two acquisitions."
        elif changed_pct < 10:
            desc = (
                f"Moderate changes detected affecting approximately "
                f"{changed_pct:.1f}% of the scene, concentrated in discrete regions."
            )
        elif changed_pct < 30:
            desc = (
                f"Significant changes detected across {changed_pct:.1f}% "
                f"of the scene, possibly indicating land-use conversion, "
                f"construction activity, or environmental change."
            )
        else:
            desc = (
                f"Major transformation detected — {changed_pct:.1f}% of "
                f"the scene has changed. This may indicate large-scale "
                f"events such as flooding, deforestation, or urban expansion."
            )
        return f"[{modality.upper()} pair] {desc}"
