FROM python:3.11-slim AS builder

WORKDIR /app

# Build tools only — GDAL is no longer a dependency (arosics/rasterio/GDAL
# were removed; co-registration falls back to pure-Python OpenCV if present).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md .
COPY controller/ controller/
COPY specialists/ specialists/
COPY confidence/ confidence/
COPY preprocessing/ preprocessing/
COPY interfaces/ interfaces/
COPY pipeline/ pipeline/
COPY models/ models/
COPY run.py .
# NOTE: logs/ is intentionally NOT copied — that's the developer's local
# audit trail (execution_chain.jsonl), not part of the product. Every
# deployment starts its own chain via the empty dir created below.

# Install Python dependencies. geo_rag pulls in the real qdrant-client/
# sentence-transformers path in specialists/geo_rag/indexer.py; voice pulls
# in faster-whisper/piper-tts (currently unwired into the main pipeline but
# real, gated optional code) — kept here so the image can exercise them.
# The GUI layer's Flask dependency lives in pipeline/requirements.txt,
# separate from the base package's pyproject.toml.
RUN pip install --no-cache-dir ".[geo_rag,voice]" -r pipeline/requirements.txt

# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

COPY deploy/cloud_demo/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Non-root user. Runtime writes only ever go to logs/, artifacts/,
# pipeline/uploads/, data/, and /tmp — everything else is read-only
# at the container level (see docker-compose.yml's read_only + tmpfs).
RUN groupadd -r aetheris && useradd -r -g aetheris -d /app -s /usr/sbin/nologin aetheris \
    && mkdir -p /app/logs /app/artifacts /app/pipeline/uploads /app/data \
    && chown -R aetheris:aetheris /app

ENV HOME=/app
USER aetheris

# 8000 = FastAPI service (default). 5000 = Flask GUI, opt in with
# AETHERIS_SERVICE=gui — see entrypoint.sh.
EXPOSE 8000 5000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "\
import os, urllib.request; \
gui = os.environ.get('AETHERIS_SERVICE') == 'gui'; \
port = os.environ.get('AETHERIS_GUI_PORT', '5000') if gui else '8000'; \
path = '/api/health' if gui else '/health'; \
urllib.request.urlopen(f'http://localhost:{port}{path}')" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
