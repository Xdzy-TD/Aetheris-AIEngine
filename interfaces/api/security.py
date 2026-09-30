"""
Security utilities for the Aetheris API.

- API key authentication (optional — only enforced when AETHERIS_API_KEY is set)
- Upload validation (magic-byte + size check)
- Temp file cleanup context manager
- Error sanitization
"""

from __future__ import annotations

import hmac
import os
import tempfile
from contextlib import contextmanager
from collections.abc import Generator
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("AETHERIS_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """Dependency that enforces API key auth when AETHERIS_API_KEY is set.

    If the env var is empty/unset, all requests pass through (local dev mode).
    """
    if not _API_KEY:
        return None  # auth disabled
    # Plain != leaks the key's correct-prefix length via response timing.
    if not api_key or not hmac.compare_digest(api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# Magic bytes for common image formats
_MAGIC = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"II\x2a\x00": "image/tiff",  # little-endian TIFF
    b"MM\x00\x2a": "image/tiff",  # big-endian TIFF
    b"RIFF": "image/webp",
}


async def read_upload_limited(upload: Any, chunk_size: int = 1024 * 1024) -> bytes:
    """Read an UploadFile in chunks, aborting as soon as it exceeds the limit.

    Avoids buffering an unbounded upload fully into memory before the size
    check in ``validate_image_upload`` ever runs.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(chunk_size):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def validate_image_upload(data: bytes, filename: str) -> None:
    """Validate that uploaded bytes are a real image within size limits.

    Raises HTTPException on failure.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit",
        )
    if len(data) < 4:
        raise HTTPException(status_code=400, detail="Upload too small to be a valid image")

    header = data[:4]
    # RIFF is a generic container (WAV, AVI, ... also start with it) — WebP
    # additionally requires the "WEBP" form-type at bytes 8:12.
    is_valid = any(header.startswith(magic) for magic in _MAGIC if magic != b"RIFF")
    if header.startswith(b"RIFF") and data[8:12] == b"WEBP":
        is_valid = True
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format for '{filename}'. "
            "Accepted: PNG, JPEG, TIFF, WebP.",
        )


# ---------------------------------------------------------------------------
# Temp file cleanup
# ---------------------------------------------------------------------------

@contextmanager
def temp_upload(data: bytes, suffix: str = ".tif") -> Generator[str, None, None]:
    """Write upload data to a temp file and delete it after use."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="aetheris_")
    try:
        tmp.write(data)
        tmp.close()
        yield tmp.name
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Error sanitization
# ---------------------------------------------------------------------------

_DEBUG = os.environ.get("AETHERIS_DEBUG", "").lower() in ("1", "true", "yes")


def safe_error_detail(exc: Exception) -> str:
    """Return a safe error message — full traceback only in debug mode."""
    if _DEBUG:
        return str(exc)
    return "An internal error occurred. Check server logs for details."


# ---------------------------------------------------------------------------
# CORS origins
# ---------------------------------------------------------------------------

def get_cors_origins() -> list[str]:
    """Return allowed CORS origins from env, defaulting to localhost."""
    raw = os.environ.get("AETHERIS_CORS_ORIGINS", "")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://localhost:3000", "http://localhost:8000"]
