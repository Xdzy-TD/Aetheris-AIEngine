#!/usr/bin/env sh
# Serves either the original FastAPI app (default) or the new GUI, chosen by
# AETHERIS_SERVICE=gui. Host/port for each come from their own existing env
# vars (--host/--port for FastAPI, AETHERIS_GUI_HOST/AETHERIS_GUI_PORT read
# by pipeline/config.py for the GUI) — nothing new to configure.
set -e

if [ "$AETHERIS_SERVICE" = "gui" ]; then
    exec python -m pipeline.run gui
else
    exec python run.py serve --host "${AETHERIS_HOST:-0.0.0.0}" --port "${AETHERIS_PORT:-8000}"
fi
