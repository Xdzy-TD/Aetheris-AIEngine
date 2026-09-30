"""
Security Adapter — wraps ``interfaces.api.security`` without editing it.

Reuses existing API key auth, upload validation, magic-byte checking,
and extends with role/permission context for M04.
"""

from __future__ import annotations

from typing import Any

from pipeline.schemas.contracts import ModuleID, ModuleResult


class SecurityAdapter:
    """Adapter around the existing API security module."""

    @staticmethod
    def validate_upload(data: bytes, filename: str) -> ModuleResult:
        """Validate an upload using the old security module.

        `interfaces.api.security` imports fastapi, an optional dependency the
        standalone Flask GUI doesn't require. A missing fastapi must not be
        reported as a failed validation (m06's own magic-byte/size/extension
        checks already ran before this is called) — bug was: every upload
        failed with "No module named 'fastapi'" on a fastapi-less install.
        """
        try:
            from interfaces.api.security import validate_image_upload
        except ImportError:
            return ModuleResult.success(
                ModuleID.M06,
                {"upload_valid": True, "filename": filename, "size_bytes": len(data)},
            )
        try:
            validate_image_upload(data, filename)
            return ModuleResult.success(
                ModuleID.M06,
                {"upload_valid": True, "filename": filename, "size_bytes": len(data)},
            )
        except Exception as exc:
            # Old module raises FastAPI HTTPException — extract the detail
            detail = getattr(exc, "detail", str(exc))
            return ModuleResult.failure(
                ModuleID.M06,
                "UPLOAD_VALIDATION_FAILED",
                str(detail),
            )

    @staticmethod
    def check_api_key(api_key: str | None) -> ModuleResult:
        """Check API key using the old security module."""
        import os
        expected = os.environ.get("AETHERIS_API_KEY", "")
        if not expected:
            return ModuleResult.success(
                ModuleID.M04,
                {"auth_mode": "disabled", "message": "API key auth not configured"},
            )
        if api_key == expected:
            return ModuleResult.success(
                ModuleID.M04,
                {"auth_mode": "api_key", "authenticated": True},
            )
        return ModuleResult.failure(
            ModuleID.M04,
            "AUTH_FAILED",
            "Invalid or missing API key",
        )

    @staticmethod
    def get_cors_origins() -> list[str]:
        """Get CORS origins from the old security module."""
        try:
            from interfaces.api.security import get_cors_origins
            return get_cors_origins()
        except Exception:
            return ["http://localhost:3000", "http://localhost:5000"]
