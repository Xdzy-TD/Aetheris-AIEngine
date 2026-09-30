"""
Central configuration for the AETHERIS M01–M20 pipeline layer.

Reads from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
import secrets as _secrets
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # e:\AETHERIS\AETHERIS
PIPELINE_ROOT = Path(__file__).resolve().parent
# Kept as alias so existing code that reads M01_M10_ROOT still works.
M01_M10_ROOT = PIPELINE_ROOT
UPLOAD_DIR = PIPELINE_ROOT / "uploads"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Demo mode (must be defined before Auth, which reads it)
# ---------------------------------------------------------------------------
# Seeds the admin/analyst/viewer demo accounts in m04_auth.py. Off by default —
# a deployment must opt in, not opt out, to shipping known credentials.
DEMO_MODE = os.environ.get("AETHERIS_DEMO_MODE", "false").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("AETHERIS_API_KEY", "")
_JWT_DEFAULT = "aetheris-hackathon-secret-key"
JWT_SECRET = os.environ.get("AETHERIS_JWT_SECRET", "")
if not JWT_SECRET:
    if DEMO_MODE:
        JWT_SECRET = _JWT_DEFAULT  # demo only — never ships to production
    else:
        JWT_SECRET = _secrets.token_hex(32)  # random per-process
AUTH_ENABLED = os.environ.get("AETHERIS_AUTH_ENABLED", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
RATE_LIMIT_RPM = int(os.environ.get("AETHERIS_RATE_LIMIT_RPM", "60"))
MAX_CONCURRENT_JOBS = int(os.environ.get("AETHERIS_MAX_CONCURRENT_JOBS", "4"))

# ---------------------------------------------------------------------------
# Upload security
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.environ.get("AETHERIS_MAX_UPLOAD_MB", "50")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp"}

# ---------------------------------------------------------------------------
# Session / Cache
# ---------------------------------------------------------------------------
SESSION_TTL_SECONDS = int(os.environ.get("AETHERIS_SESSION_TTL", "3600"))
MAX_CACHE_ENTRIES = int(os.environ.get("AETHERIS_MAX_CACHE", "512"))

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("AETHERIS_OLLAMA_URL", "http://localhost:11434")
USE_LLM = os.environ.get("AETHERIS_NO_LLM", "").lower() not in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
GUI_HOST = os.environ.get("AETHERIS_GUI_HOST", "0.0.0.0")
GUI_PORT = int(os.environ.get("AETHERIS_GUI_PORT", "5000"))
GUI_DEBUG = os.environ.get("AETHERIS_GUI_DEBUG", "false").lower() in ("1", "true", "yes")

# HTTPS: set both AETHERIS_GUI_SSL_CERT + AETHERIS_GUI_SSL_KEY to serve with a
# real certificate, or AETHERIS_GUI_SSL_ADHOC=true for a local self-signed one
# (dev/testing only — browsers show a warning; needs `pip install pyOpenSSL`).
# Leave all unset (the default) for plain HTTP, unchanged from before.
GUI_SSL_CERT = os.environ.get("AETHERIS_GUI_SSL_CERT", "")
GUI_SSL_KEY = os.environ.get("AETHERIS_GUI_SSL_KEY", "")
GUI_SSL_ADHOC = os.environ.get("AETHERIS_GUI_SSL_ADHOC", "false").lower() in ("1", "true", "yes")
