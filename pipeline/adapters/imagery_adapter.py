"""
Imagery Adapter — wraps ``specialists._imagery`` without modifying it.

Provides safe access to raster metadata probing, loading, and validation
for M06, M09, and M10.
"""

from __future__ import annotations

from typing import Any

from pipeline.schemas.contracts import ModuleID, ModuleResult


class ImageryAdapter:
    """Adapter around existing imagery utilities."""

    @staticmethod
    def probe_metadata(path: str) -> dict[str, Any]:
        """Probe raster metadata using the old ``probe_raster_metadata``."""
        try:
            from specialists._imagery import probe_raster_metadata
            return probe_raster_metadata(path)
        except Exception as exc:
            return {
                "georeferenced": False,
                "reader": "error",
                "error": str(exc),
                "crs": None,
                "transform": None,
                "resolution_m": None,
                "bounds": None,
                "band_count": None,
                "bands": [],
                "nodata": None,
                "nodata_fraction": 0.0,
                "dtype": None,
            }

    @staticmethod
    def load_raster(path: str) -> tuple[Any, dict[str, Any]]:
        """Load a raster using the old ``load_raster``."""
        from specialists._imagery import load_raster
        return load_raster(path)

    @staticmethod
    def load_rgb(path: str) -> Any:
        """Load RGB array using the old ``load_rgb``."""
        from specialists._imagery import load_rgb
        return load_rgb(path)

    @staticmethod
    def check_rasterio_available() -> bool:
        """Check if rasterio is available."""
        try:
            import rasterio  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def validate_image_content(path: str) -> ModuleResult:
        """Validate image content is readable via the old imagery pipeline."""
        try:
            from specialists._imagery import open_checked
            open_checked(path)
            return ModuleResult.success(
                ModuleID.M06,
                {"content_valid": True, "path": path},
            )
        except (FileNotFoundError, ValueError) as exc:
            return ModuleResult.failure(
                ModuleID.M06,
                "CONTENT_VALIDATION_FAILED",
                str(exc),
            )
