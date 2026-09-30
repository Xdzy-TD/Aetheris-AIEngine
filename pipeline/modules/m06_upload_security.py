"""
M06 — Upload Security.

Secure upload handling with filename sanitization, path traversal
prevention, format validation, size checks, and content verification.
Reuses old image validation from ``interfaces.api.security`` and
``specialists._imagery``.
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pipeline.adapters.imagery_adapter import ImageryAdapter
from pipeline.adapters.security_adapter import SecurityAdapter
from pipeline.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, UPLOAD_DIR
from pipeline.schemas.contracts import ModuleID, ModuleResult

# Magic bytes for content-type sniffing (reuses the old security.py concept)
_MAGIC_BYTES: dict[bytes, str] = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"II\x2a\x00": "image/tiff",
    b"MM\x00\x2a": "image/tiff",
    b"RIFF": "image/webp",
}


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and injection."""
    # Remove path separators
    name = filename.replace("\\", "/").split("/")[-1]
    # Remove dangerous characters
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "", name)
    # Remove leading dots (hidden files)
    name = name.lstrip(".")
    # Limit length
    if len(name) > 200:
        stem, ext = os.path.splitext(name)
        name = stem[:200 - len(ext)] + ext
    return name or "upload"


def _check_path_traversal(filename: str) -> bool:
    """Check for path traversal attempts."""
    dangerous = ["..", "~", "//", "\\\\", "%2e", "%2f", "%5c"]
    lower = filename.lower()
    return any(d in lower for d in dangerous)


def validate_upload(
    data: bytes,
    filename: str,
    allowed_extensions: set[str] | None = None,
    max_size: int | None = None,
) -> ModuleResult:
    """Full upload security validation."""
    exts = allowed_extensions or ALLOWED_EXTENSIONS
    max_bytes = max_size or MAX_UPLOAD_BYTES
    issues: list[str] = []
    warnings: list[str] = []

    # 1. Check for path traversal
    if _check_path_traversal(filename):
        return ModuleResult.failure(
            ModuleID.M06,
            "PATH_TRAVERSAL_DETECTED",
            f"Potential path traversal in filename: {filename!r}",
        )

    # 2. Sanitize filename
    safe_name = _sanitize_filename(filename)
    if safe_name != filename:
        warnings.append(f"Filename sanitized: {filename!r} -> {safe_name!r}")

    # 3. Check file extension
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in exts:
        issues.append(f"Extension '{ext}' not in allowed set: {sorted(exts)}")

    # 4. Check file size
    if len(data) > max_bytes:
        issues.append(
            f"File size {len(data) // (1024*1024)} MB exceeds "
            f"limit {max_bytes // (1024*1024)} MB"
        )

    if len(data) < 8:
        issues.append("File too small to be a valid image (< 8 bytes)")

    # 5. Magic byte validation (content-type sniffing)
    if len(data) >= 4:
        header = data[:4]
        detected_type = None
        for magic, mime in _MAGIC_BYTES.items():
            if not header.startswith(magic):
                continue
            # RIFF is a generic container (WAV, AVI, ... also start with it) —
            # WebP additionally requires the "WEBP" form-type at bytes 8:12.
            if magic == b"RIFF" and data[8:12] != b"WEBP":
                continue
            detected_type = mime
            break
        if detected_type is None:
            issues.append("File header does not match any known image format")
    else:
        detected_type = None

    # 6. Delegate to old security module
    old_result = SecurityAdapter.validate_upload(data, safe_name)
    if old_result.status.value == "failure":
        for err in old_result.errors:
            issues.append(f"Old validation: {err.message}")

    if issues:
        result = ModuleResult.failure(
            ModuleID.M06,
            "UPLOAD_VALIDATION_FAILED",
            "; ".join(issues),
        )
        result.warnings = warnings
        return result

    return ModuleResult.success(
        ModuleID.M06,
        {
            "valid": True,
            "sanitized_filename": safe_name,
            "original_filename": filename,
            "size_bytes": len(data),
            "detected_type": detected_type,
            "extension": ext,
        },
    )


def save_upload(data: bytes, filename: str) -> ModuleResult:
    """Save an upload to a secure temporary location with unique ID."""
    # Validate first
    validation = validate_upload(data, filename)
    if validation.status.value != "success":
        return validation

    safe_name = _sanitize_filename(filename)
    unique_id = uuid.uuid4().hex[:16]
    ext = os.path.splitext(safe_name)[1]
    dest_name = f"{unique_id}{ext}"

    # Ensure upload directory exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / dest_name
    dest.write_bytes(data)

    # Probe metadata if possible
    metadata = ImageryAdapter.probe_metadata(str(dest))

    return ModuleResult.success(
        ModuleID.M06,
        {
            "upload_id": unique_id,
            "path": str(dest),
            "sanitized_filename": safe_name,
            "size_bytes": len(data),
            "raster_metadata": metadata,
        },
    )


def cleanup_upload(path: str) -> None:
    """Remove an uploaded file."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
