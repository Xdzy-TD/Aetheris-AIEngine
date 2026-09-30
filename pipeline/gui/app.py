"""
Flask GUI — legacy web front end for the M01-M20 pipeline.

Serves templates/index.html and the /api/* routes it calls. Mounted into
the FastAPI app at /gui (see interfaces/api/app.py) via a2wsgi, and
runnable standalone through ``python -m pipeline.run gui``.

Every route here is a thin wrapper around code that already exists
elsewhere (pipeline.services, pipeline.modules, controller.artifacts,
interfaces.voice) — this file only does HTTP plumbing and the API-key
guard described in config.py's Auth section.
"""

from __future__ import annotations

import hmac
import tempfile
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

import structlog

from controller.artifacts import default_store
from interfaces.api.deps import get_audit
from interfaces.voice.stt import SpeechToText
from interfaces.voice.tts import TextToSpeech
from pipeline import config
from pipeline.gui.job_manager import JobManager
from pipeline.modules.m01_baseline import run_health_check
from pipeline.modules.m06_upload_security import cleanup_upload, save_upload
from pipeline.modules.m15_csv_analysis import analyze_csv
from pipeline.schemas.contracts import PipelineRequest
from pipeline.services.orchestrator import build_pipeline_dag
from pipeline.services.pipeline import AetherisPipeline

logger = structlog.get_logger(__name__)

app = Flask(__name__)
# Two images per analysis request, plus headroom for form fields.
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES * 2 + 1_048_576

# Copied from config at import time (not referenced as config.API_KEY) so
# tests can monkeypatch pipeline.gui.app.API_KEY / .AUTH_ENABLED directly,
# mirroring interfaces/api/security.py's require_api_key.
API_KEY = config.API_KEY
AUTH_ENABLED = config.AUTH_ENABLED

config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_pipeline = AetherisPipeline()
_jobs = JobManager(max_concurrent=config.MAX_CONCURRENT_JOBS)
_stt = SpeechToText()
_tts = TextToSpeech(output_dir=config.UPLOAD_DIR)


@app.before_request
def _enforce_api_key():
    """Gate /api/* the same way interfaces/api/security.py gates FastAPI.

    /api/health stays open so uptime checks work even when a key is set;
    everything outside /api/* (the SPA shell, static assets) is never
    gated here — that matches the FastAPI side, which only puts
    Depends(require_api_key) on specific routes, not the whole app.
    """
    if not request.path.startswith("/api/") or request.path == "/api/health":
        return None
    if not AUTH_ENABLED or not API_KEY:
        return None
    supplied = request.headers.get("X-API-Key", "")
    if not supplied or not hmac.compare_digest(supplied, API_KEY):
        return jsonify(error="Invalid or missing API key"), 401
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def api_health():
    return jsonify(run_health_check().model_dump(mode="json"))


@app.route("/api/pipeline/status")
def api_pipeline_status():
    return jsonify({
        "concurrency": _jobs.get_status(),
        "rate_limiter": _pipeline.rate_limiter.get_status(),
        "sessions": _pipeline.session_mgr.get_status(),
        "planner": _pipeline.planner.get_planner_status(),
    })


@app.route("/api/jobs")
def api_jobs():
    return jsonify(_jobs.list_jobs())


@app.route("/api/dag")
def api_dag():
    return jsonify(build_pipeline_dag().get_state())


@app.route("/api/audit/recent")
def api_audit_recent():
    n = request.args.get("n", 20, type=int)
    entries = get_audit().get_last_n(n)
    return jsonify([e.model_dump(mode="json") for e in entries])


@app.route("/api/audit/verify")
def api_audit_verify():
    valid, errors = get_audit().verify_chain()
    return jsonify({"valid": valid, "errors": errors})


@app.route("/api/artifacts/<artifact_id>")
def api_artifact(artifact_id: str):
    store = default_store()
    record = store.get_record(artifact_id)
    if record is None:
        return jsonify(error="Unknown artifact_id"), 404
    return jsonify({**record, "sha256_matches": store.verify(artifact_id)})


@app.route("/api/artifacts/<artifact_id>/image")
def api_artifact_image(artifact_id: str):
    store = default_store()
    record = store.get_record(artifact_id)
    if record is None:
        abort(404)
    return send_file(record["path"])


@app.route("/api/csv/analyze", methods=["POST"])
def api_csv_analyze():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file provided"), 400
    result = analyze_csv(f.read(), f.filename or "upload.csv")
    return jsonify(result.model_dump(mode="json"))


def _persist(f) -> str | None:
    """Save an uploaded FileStorage via M06's validated save path.

    Returns None (rather than raising) on validation failure — the
    caller passes that through as no file_path, and AetherisPipeline's
    own M06 check on file_data/file_name produces the real error in the
    pipeline result, instead of this layer duplicating that decision.
    """
    saved = save_upload(f.read(), f.filename or "upload")
    return saved.data.get("path") if saved.status.value == "success" else None


@app.route("/api/pipeline/upload", methods=["POST"])
def api_pipeline_upload():
    image1 = request.files.get("image")
    if not image1:
        return jsonify(error="No image provided"), 400
    image2 = request.files.get("image2")

    job_id = _jobs.submit()["job_id"]
    paths: list[str] = []
    try:
        data1 = image1.read()
        image1.seek(0)
        path1 = _persist(image1)
        if path1:
            paths.append(path1)

        path2 = None
        if image2 and image2.filename:
            path2 = _persist(image2)
            if path2:
                paths.append(path2)

        pipeline_request = PipelineRequest(
            question=request.form.get("question", ""),
            modality=request.form.get("modality", "optical"),
        )
        result = _pipeline.execute(
            pipeline_request,
            file_data=data1,
            file_name=image1.filename or "",
            file_path=path1,
            file2_path=path2,
            file2_name=(image2.filename or "") if image2 else "",
        )
        _jobs.complete(job_id, result={"pipeline_id": result.pipeline_id, "status": result.status.value})
        return jsonify(result.model_dump(mode="json"))
    except Exception as exc:
        logger.exception("pipeline_upload_failed")
        _jobs.complete(job_id, error=str(exc))
        return jsonify(error="Internal error processing upload"), 500
    finally:
        for p in paths:
            cleanup_upload(p)


@app.route("/api/stt/transcribe", methods=["POST"])
def api_stt_transcribe():
    """Offline dictation fallback — see interfaces/voice/stt.py.

    Audio isn't imagery, so this bypasses M06's image-only
    validate_upload (extension allowlist is .png/.jpg/.tiff/.webp) and
    just writes the bytes to a temp file, capped at the same
    MAX_UPLOAD_BYTES limit used everywhere else.
    """
    f = request.files.get("audio")
    if not f:
        return jsonify(error="No audio provided"), 400
    data = f.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        return jsonify(error=f"Audio exceeds {config.MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit."), 400
    suffix = Path(f.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(dir=config.UPLOAD_DIR, suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return jsonify(_stt.transcribe(path))
    except Exception as exc:
        logger.exception("stt_transcribe_failed")
        return jsonify(error=str(exc)), 500
    finally:
        cleanup_upload(path)


@app.route("/api/tts/speak", methods=["POST"])
def api_tts_speak():
    """Offline speech fallback — see interfaces/voice/tts.py."""
    text = (request.get_json(silent=True) or {}).get("text", "")
    if not text.strip():
        return jsonify(error="No text provided"), 400
    result = _tts.speak(text)
    if not result.get("audio_path"):
        return jsonify(error="Offline TTS unavailable — install piper-tts."), 503
    return send_file(result["audio_path"], mimetype="audio/wav")


def main() -> None:
    """Entry point for ``python -m pipeline.run gui``."""
    ssl_context = None
    if config.GUI_SSL_CERT and config.GUI_SSL_KEY:
        ssl_context = (config.GUI_SSL_CERT, config.GUI_SSL_KEY)
    elif config.GUI_SSL_ADHOC:
        ssl_context = "adhoc"
    app.run(host=config.GUI_HOST, port=config.GUI_PORT, debug=config.GUI_DEBUG, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
