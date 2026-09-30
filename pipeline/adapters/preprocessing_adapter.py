"""
Preprocessing Adapter — wraps ``preprocessing.*`` and
``specialists._imagery.estimate_shift`` without modifying old code.

Backs M11 (change-detection preprocessing): temporal-pair co-registration
and despeckling ahead of the old ``change_detection`` specialist.
"""

from __future__ import annotations

from typing import Any

from pipeline.schemas.contracts import ModuleID, ModuleResult


class PreprocessingAdapter:
    """Adapter around the existing preprocessing package."""

    @staticmethod
    def is_arosics_available() -> bool:
        try:
            from preprocessing.arosics_coreg import is_arosics_available
            return is_arosics_available()
        except Exception:
            return False

    @staticmethod
    def is_snap_available() -> bool:
        try:
            from preprocessing.snap_pipeline import is_snap_available
            return is_snap_available()
        except Exception:
            return False

    @staticmethod
    def estimate_shift(ref: Any, tgt: Any) -> tuple[int, int]:
        """Cheap in-memory FFT phase-correlation shift estimate (dy, dx).

        Always available (pure NumPy) — used as a fast pre-flight check
        before deciding whether the slower, georeferenced ``coregister``
        path is worth invoking.
        """
        from specialists._imagery import estimate_shift
        return estimate_shift(ref, tgt)

    @staticmethod
    def coregister(
        reference_path: str,
        target_path: str,
        output_path: str | None = None,
        method: str = "auto",
        max_shift: int = 50,
    ) -> ModuleResult:
        """Co-register target onto reference using the old preprocessing pipeline."""
        try:
            from preprocessing.arosics_coreg import coregister
            out = coregister(
                reference_path, target_path,
                output_path=output_path, method=method, max_shift=max_shift,
            )
            return ModuleResult.success(
                ModuleID.M11,
                {"coregistered": True, "output_path": str(out), "method": method},
            )
        except Exception as exc:
            return ModuleResult.failure(
                ModuleID.M11,
                "COREGISTRATION_FAILED",
                str(exc),
            )

    @staticmethod
    def despeckle(arr: Any, method: str = "lee", **kwargs: Any) -> Any:
        """Run a SAR despeckle filter from the fallback filter bank."""
        from preprocessing.fallback_filters import frost_filter, kuan_filter, lee_filter
        filters = {"lee": lee_filter, "frost": frost_filter, "kuan": kuan_filter}
        if method not in filters:
            raise ValueError(f"Unknown despeckle method: {method!r}")
        return filters[method](arr, **kwargs)
