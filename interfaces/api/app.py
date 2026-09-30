"""
FastAPI application — the primary HTTP interface for Aetheris.

Every endpoint below is mounted twice: at ``/v1/...`` (canonical) and at the
unprefixed path (kept for backward compatibility with clients written before
versioning existed). New clients should use ``/v1``.

Endpoints:
    POST /v1/query          Submit a query + image upload -> ExecutionResult
    POST /v1/mission/{name} Run a named mission preset (controller/missions.py) -> ExecutionResult
    GET  /v1/missions       List available mission presets (name, label, emoji, question)
    GET  /v1/tools          List registered specialist tools
    GET  /v1/audit/verify   Verify the hash-chain integrity
    GET  /v1/audit/trace/{query_id}   Audit trail for a specific query
    GET  /v1/health         Health check
    GET  /v1/memory/stats   Routing memory statistics
    GET  /metrics           Prometheus metrics (unversioned scrape target)
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from controller import missions
from controller.artifacts import default_store
from controller.audit_log import AuditLog
from controller.memory import RoutingMemory
from controller.ollama_client import OllamaClient
from controller.planner import AgenticPlanner
from controller.schemas import (
    ExecutionResult,
    ImageMetadata,
    Modality,
    Query,
)
from controller.model_registry import all_models
from controller.tool_registry import ToolRegistry
from interfaces.api.deps import (
    get_audit,
    get_jev_planner,
    get_memory,
    get_ollama_client,
    get_planner,
    get_registry,
)
from specialists._imagery import probe_raster_metadata

from interfaces.api.observability import metrics_response, observability_middleware
from interfaces.api.security import (
    get_cors_origins,
    require_api_key,
    safe_error_detail,
    read_upload_limited,
    validate_image_upload,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Aetheris",
    description=(
        "Agentic Remote-Sensing Intelligence System — "
        "dynamic tool-chain orchestration for satellite imagery analysis."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(observability_middleware)


@app.get("/metrics", tags=["system"])
async def metrics() -> Any:
    """Prometheus scrape target — request count and latency histogram."""
    return metrics_response()


# All versioned endpoints are declared on `router` below, then mounted twice:
# once at /v1 (canonical) and once unprefixed (backward compatibility).
router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    tools_loaded: int = 0
    audit_chain_valid: bool = True
    ollama_reachable: bool = False


class AuditVerifyResponse(BaseModel):
    valid: bool
    errors: list[str]


class MemoryStatsResponse(BaseModel):
    total_entries: int
    max_entries: int
    sessions: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(
    registry: ToolRegistry = Depends(get_registry),
    audit: AuditLog = Depends(get_audit),
    ollama: OllamaClient = Depends(get_ollama_client),
) -> HealthResponse:
    """System health check."""
    valid, _ = audit.verify_chain()
    ollama_ok = await ollama.health_check()
    return HealthResponse(
        tools_loaded=len(registry.list_tool_names()),
        audit_chain_valid=valid,
        ollama_reachable=ollama_ok,
    )


@router.get("/tools", tags=["system"])
async def list_tools(
    registry: ToolRegistry = Depends(get_registry),
) -> list[dict[str, Any]]:
    """List all registered specialist tools and their status."""
    return registry.list_tools()


@router.get("/missions", tags=["inference"])
async def list_missions() -> list[dict[str, str]]:
    """List available mission presets for a mission picker (see controller/missions.py)."""
    return missions.mission_catalog()


@router.get("/artifacts/{artifact_id}", tags=["audit"], dependencies=[Depends(require_api_key)])
async def get_artifact(artifact_id: str) -> dict[str, Any]:
    """Return a stored artifact's record and re-hash its bytes.

    ``sha256_matches`` is computed on read, so a mask cited by a report can be
    checked rather than trusted.
    """
    store = default_store()
    record = store.get_record(artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown artifact_id")
    return {**record, "sha256_matches": store.verify(artifact_id)}


@router.get("/models", tags=["system"])
async def list_models() -> list[dict[str, Any]]:
    """List every model backing a task, with its kind, licence and metrics.

    ``kind`` distinguishes a classical baseline from a trained model, so a
    reviewer can see which is which without reading the source.
    """
    return all_models()


def _parse_modality(value: str) -> Modality:
    try:
        return Modality(value)
    except ValueError:
        return Modality.UNKNOWN


async def _save_uploads(
    image: UploadFile | None,
    image2: UploadFile | None,
    modality: str,
    modality2: str | None,
) -> tuple[list[ImageMetadata], bytes | None, list[str]]:
    """Validate and persist up to two uploads, returning ``(images, image_bytes,
    temp_paths)``. Shared by ``/query`` and ``/mission/{name}`` so the upload,
    geo-probe, and temp-file handling lives in exactly one place.

    ``image2`` defaults to ``image``'s modality (bi-temporal pair) but can be
    set independently so a real optical+SAR upload can reach a cross-modal
    tool. Callers are responsible for unlinking ``temp_paths`` when done.
    """
    mods = (_parse_modality(modality), _parse_modality(modality2 or modality))
    images: list[ImageMetadata] = []
    image_bytes: bytes | None = None  # bytes of the first image, used as the cache key
    temp_paths: list[str] = []
    for i, upload in enumerate((image, image2)):
        if upload is None:
            continue
        data = await read_upload_limited(upload)
        validate_image_upload(data, upload.filename or "upload")
        if i == 0:
            image_bytes = data
        suffix = Path(upload.filename or "image.tif").suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="aetheris_")
        tmp.write(data)
        tmp.close()
        temp_paths.append(tmp.name)
        # Read the geospatial header now and carry it on the query. Without
        # this the CRS/transform/resolution die at the upload boundary and every
        # downstream artifact is un-georeferenced.
        geo = probe_raster_metadata(tmp.name)
        images.append(ImageMetadata(
            filename=upload.filename or "upload",
            path=tmp.name,
            modality=mods[i],
            crs=geo.get("crs"),
            bbox=geo.get("bounds"),
            resolution_m=geo.get("resolution_m"),
            bands=geo.get("bands") or [],
        ))
    return images, image_bytes, temp_paths


@router.post(
    "/query",
    response_model=ExecutionResult,
    tags=["inference"],
    dependencies=[Depends(require_api_key)],
)
async def submit_query(
    question: Annotated[str, Form(description="Natural language question")],
    modality: Annotated[str, Form(description="Modality of `image`")] = "optical",
    modality2: Annotated[
        str | None,
        Form(description="Modality of `image2` — required to differ from `modality` for a "
             "cross-modal optical/SAR pair; defaults to `modality` for bi-temporal pairs"),
    ] = None,
    session_id: Annotated[str, Form(description="Session ID")] = "",
    image: UploadFile | None = File(None, description="Satellite image upload"),
    image2: UploadFile | None = File(
        None, description="Second image, e.g. the post-change acquisition for change detection "
        "or the SAR half of a cross-modal pair"
    ),
    planner: AgenticPlanner = Depends(get_planner),
) -> ExecutionResult:
    """Submit a query with an optional image (or image pair) for analysis.

    The agentic controller dynamically composes a tool chain based on the
    query content and image metadata, executes it, and returns the result
    with a full audit trail. Uploaded images are saved to a temp file and
    their real path is threaded through to the specialists — the planner
    never gets to invent one.
    """
    if not session_id:
        session_id = uuid.uuid4().hex[:8]

    images, image_bytes, temp_paths = await _save_uploads(image, image2, modality, modality2)
    try:
        query = Query(text=question, images=images, session_id=session_id)
        result = await planner.process_query(query, image_bytes)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("query_failed", error=str(exc), query_id=session_id)
        raise HTTPException(status_code=500, detail=safe_error_detail(exc))
    finally:
        for p in temp_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


@router.post(
    "/mission/{name}",
    response_model=ExecutionResult,
    tags=["inference"],
    dependencies=[Depends(require_api_key)],
)
async def submit_mission(
    name: str,
    question: Annotated[str, Form(description="Overrides the mission's default question")] = "",
    modality: Annotated[str, Form(description="Modality of `image`")] = "optical",
    modality2: Annotated[str | None, Form(description="Modality of `image2`")] = None,
    session_id: Annotated[str, Form(description="Session ID")] = "",
    image: UploadFile | None = File(None, description="Satellite image upload"),
    image2: UploadFile | None = File(None, description="Second image, e.g. bi-temporal pair"),
    planner: AgenticPlanner = Depends(get_planner),
) -> ExecutionResult:
    """Run a named mission preset (see ``controller/missions.py``) over an upload.

    Thin wrapper: same upload handling as ``/query``, then
    :func:`missions.build_plan` picks the tool chain instead of ``_plan()``.
    """
    if not session_id:
        session_id = uuid.uuid4().hex[:8]

    images, image_bytes, temp_paths = await _save_uploads(image, image2, modality, modality2)
    try:
        query = Query(text=question, images=images, session_id=session_id)
        try:
            query, plan = missions.build_plan(name, query, planner)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown mission: {name!r}")
        except ValueError as exc:
            # e.g. the custom mission generator with no objective text — a
            # client input error, not a server fault, so 422 not 500.
            raise HTTPException(status_code=422, detail=str(exc))
        result = await planner.process_query(query, image_bytes, plan=plan)
        result.mission_summary = missions.mission_summary(result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("mission_failed", error=str(exc), query_id=session_id, mission=name)
        raise HTTPException(status_code=500, detail=safe_error_detail(exc))
    finally:
        for p in temp_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


@router.get(
    "/audit/verify",
    response_model=AuditVerifyResponse,
    tags=["audit"],
    dependencies=[Depends(require_api_key)],
)
async def verify_audit(
    audit: AuditLog = Depends(get_audit),
) -> AuditVerifyResponse:
    """Verify the integrity of the hash-chained execution log."""
    valid, errors = audit.verify_chain()
    return AuditVerifyResponse(valid=valid, errors=errors)


@router.get("/audit/trace/{query_id}", tags=["audit"], dependencies=[Depends(require_api_key)])
async def get_trace(
    query_id: str,
    audit: AuditLog = Depends(get_audit),
) -> list[dict[str, Any]]:
    """Retrieve the full audit trail for a specific query."""
    entries = audit.get_trace(query_id)
    if not entries:
        raise HTTPException(status_code=404, detail=f"No trace found for query {query_id}")
    return [e.model_dump(mode="json") for e in entries]


@router.get("/audit/recent", tags=["audit"], dependencies=[Depends(require_api_key)])
async def recent_audit(
    n: int = 20,
    audit: AuditLog = Depends(get_audit),
) -> list[dict[str, Any]]:
    """Return the most recent audit log entries."""
    entries = audit.get_last_n(n)
    return [e.model_dump(mode="json") for e in entries]


@router.get("/provenance/{query_id}", tags=["audit"], dependencies=[Depends(require_api_key)])
async def get_provenance(
    query_id: str,
    audit: AuditLog = Depends(get_audit),
) -> dict[str, Any]:
    """Return the provenance chain for a query — every artifact verified.

    Reconstructs which artifacts were produced, which models ran, and
    re-hashes every artifact on read so cited evidence can be checked
    rather than trusted.
    """
    entries = audit.get_trace(query_id)
    if not entries:
        raise HTTPException(status_code=404, detail=f"No trace found for query {query_id}")

    store = default_store()
    answer_entry = next((e for e in reversed(entries) if e.event_type == "answer_emitted"), None)
    chain_hash = answer_entry.chain_hash if answer_entry else entries[-1].chain_hash

    # Collect artifact IDs from tool_returned payloads and verify each
    verified_artifacts: list[dict[str, Any]] = []
    for entry in entries:
        if entry.event_type == "tool_returned":
            payload = entry.payload or {}
            for aid in (payload.get("artifact_ids") or []):
                record = store.get_record(aid)
                if record:
                    verified_artifacts.append({
                        **record,
                        "sha256_matches": store.verify(aid),
                    })

    return {
        "query_id": query_id,
        "audit_chain_hash": chain_hash,
        "audit_chain_valid": audit.verify_chain()[0],
        "artifacts": verified_artifacts,
        "trace_length": len(entries),
    }


@router.get(
    "/memory/stats",
    response_model=MemoryStatsResponse,
    tags=["system"],
)
async def memory_stats(
    memory: RoutingMemory = Depends(get_memory),
) -> MemoryStatsResponse:
    """Return routing memory cache statistics."""
    stats = memory.stats()
    return MemoryStatsResponse(**stats)


@router.delete("/memory/{session_id}", tags=["system"], dependencies=[Depends(require_api_key)])
async def clear_session_memory(
    session_id: str,
    memory: RoutingMemory = Depends(get_memory),
) -> dict[str, Any]:
    """Clear routing memory for a specific session."""
    evicted = memory.clear_session(session_id)
    return {"session_id": session_id, "entries_evicted": evicted}


# ---------------------------------------------------------------------------
# JEV Engine endpoints
# ---------------------------------------------------------------------------

@router.get("/jev/status", tags=["jev"])
async def jev_status() -> dict[str, Any]:
    """JEV engine status: plan count, avg latency, availability."""
    jev = get_jev_planner()
    if jev is None:
        return {"enabled": False}
    return {"enabled": True, "routing_engine": "jev", **jev.stats}


# ---------------------------------------------------------------------------
# Ollama endpoints
# ---------------------------------------------------------------------------

@router.get("/ollama/status", tags=["ollama"])
async def ollama_status(
    ollama: OllamaClient = Depends(get_ollama_client),
) -> dict[str, Any]:
    """Check Ollama connectivity, resolved model, and available models."""
    return await ollama.status()


@router.post("/ollama/pull", tags=["ollama"], dependencies=[Depends(require_api_key)])
async def ollama_pull(
    model: str = "qwen2.5:7b",
    ollama: OllamaClient = Depends(get_ollama_client),
) -> dict[str, Any]:
    """Trigger a model download on the Ollama server.

    Useful for first-run setup when no models are available locally.
    """
    try:
        result = await ollama.pull_model(model)
        return {"status": "ok", "model": model, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# Legacy Flask GUI Mount (Dual Architecture Resolution)
# ---------------------------------------------------------------------------
from a2wsgi import WSGIMiddleware
from pipeline.gui.app import app as flask_app

# Mount the Flask app at /gui so both interfaces share the same host/port.
app.mount("/gui", WSGIMiddleware(flask_app))

# SAT-OS routes (implementation.md vertical slice)
from interfaces.api.sat_routes import router as sat_router
app.include_router(sat_router, prefix="/v1")

# Canonical, versioned mount, plus an unprefixed mount kept for clients
# written before /v1 existed. Same handlers, same behaviour, two paths.
app.include_router(router, prefix="/v1")
app.include_router(router)

