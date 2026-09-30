"""
Pydantic schemas for the Aetheris controller layer.

These models define the data contracts between the agentic planner,
tool registry, specialist modules, and the audit log.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Modality(StrEnum):
    """Sensor modality of an input image."""
    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"
    UNKNOWN = "unknown"


class TaskType(StrEnum):
    """High-level task categories the controller can route to."""
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    SAR_BRIDGE = "sar_bridge"
    GEO_RAG = "geo_rag"


# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------

class OllamaConfig(BaseModel):
    """Configuration for the Ollama LLM backend."""
    base_url: str = Field(
        "http://localhost:11434",
        description="Base URL for the Ollama REST API",
    )
    model_name: str | None = Field(
        None,
        description="Model name override (default: auto-resolve from preference list)",
    )
    temperature: float = Field(
        0.1, ge=0.0, le=2.0,
        description="Sampling temperature for planning",
    )
    timeout_s: float = Field(
        120.0, gt=0.0,
        description="HTTP timeout in seconds",
    )


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class ImageMetadata(BaseModel):
    """Metadata attached to an uploaded satellite image."""
    filename: str
    path: str | None = Field(
        None,
        description="Server-side filesystem path to the saved upload, used to "
        "hand real pixel data to specialists (never supplied by a planner).",
    )
    modality: Modality = Modality.UNKNOWN
    sensor: str | None = Field(None, description="Sensor name, e.g. 'Cartosat-2S', 'RISAT-1'")
    resolution_m: float | None = Field(None, description="Ground sample distance in metres")
    bands: list[str] = Field(
        default_factory=list, description="Band names, e.g. ['B04','B03','B02']"
    )
    crs: str | None = Field(None, description="Coordinate reference system EPSG code")
    bbox: list[float] | None = Field(
        None,
        description="Bounding box [west, south, east, north] in CRS units",
    )
    acquisition_date: datetime | None = None


class Query(BaseModel):
    """A user query submitted to the Aetheris system."""
    query_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str = Field(..., description="Natural language question or instruction")
    images: list[ImageMetadata] = Field(
        default_factory=list,
        description="Metadata per uploaded image (1 for single-image, 2 for change/cross-modal)",
    )
    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:8],
        description="Session ID for routing memory grouping",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Tool-call schemas
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """A single invocation of a specialist tool, as planned by the controller."""
    tool_name: str = Field(..., description="Registry key of the specialist tool")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool, matching its JSON schema",
    )
    rationale: str = Field(
        "",
        description="Controller's reasoning for choosing this tool at this step",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="tool_names whose output this call needs as input",
    )


class ExecutionPlan(BaseModel):
    """Ordered sequence of tool calls the controller plans to execute."""
    query_id: str
    steps: list[ToolCall]
    strategy_note: str = Field(
        "",
        description="High-level rationale for the chosen tool chain",
    )
    planning_method: str = Field(
        "rule_based",
        description="'llm' if planned by the Ollama LLM, 'rule_based' if using deterministic fallback",
    )


# ---------------------------------------------------------------------------
# Evidence schemas (Fix 3)
# ---------------------------------------------------------------------------

class EvidenceArtifact(BaseModel):
    """A typed, verifiable analysis artifact with SHA-256 integrity.

    Replaces the previous ``dict[str, str]`` artifact bags — every mask,
    change map, despeckled raster, and fusion map produced by a specialist
    now has a schema, not just a dict with ad-hoc keys.
    """
    artifact_id: str
    sha256: str
    artifact_type: str = Field(
        ..., description="E.g. 'mask', 'change_map', 'despeckled_raster', 'fusion_map'"
    )
    model_version: str
    path: str
    size_bytes: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SpatialEvidence(BaseModel):
    """Geospatially-located evidence linking a specialist's output to a location.

    Connects a tool's result to the geographic region it covers, with an
    optional artifact reference for the backing data (mask, change map, etc.).
    """
    tool_name: str
    artifact: EvidenceArtifact | None = None
    bbox: list[float] | None = Field(
        None, description="Bounding box [west, south, east, north] in CRS units"
    )
    crs: str | None = None
    change_fraction: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Fraction of the AOI where change was detected (change tools only)"
    )


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class SpecialistOutput(BaseModel):
    """Standardised output from any specialist module.

    ``model_version``, ``parameters``, ``timestamp`` and ``provenance_hash``
    close the Result Contract (docs/STATUS.md §Trust) at the one place every
    specialist call passes through (``PlanExecutor._execute_step``), rather
    than each specialist reporting them itself.
    """
    tool_name: str
    result: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific result payload",
    )
    confidence: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Raw confidence before conformal calibration (may be None for deterministic fallbacks)",
    )
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Paths to generated artifacts (masks, maps, etc.)",
    )
    evidence_artifacts: list[EvidenceArtifact] = Field(
        default_factory=list,
        description="Typed, verifiable artifacts produced by this step (Fix 3).",
    )
    spatial_evidence: list[SpatialEvidence] = Field(
        default_factory=list,
        description="Geospatially-located evidence from this step (Fix 3).",
    )
    model_version: str | None = Field(
        None,
        description="name@version (kind) from models/registry.json — 'unavailable' "
        "for an unbound tool, None only for a routing-memory cache hit.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="The resolved arguments this call actually ran with (server-supplied "
        "paths included) — not the planner's proposal, which policy may have rejected.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Human-readable caveats derived from this result: unavailability, "
        "low confidence, unreliable co-registration. Never fabricated — every entry "
        "traces to a field already present in `result`.",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provenance_hash: str | None = Field(
        None,
        description="SHA-256 of the raw result payload (controller.audit_log.AuditLog._hash_obj) "
        "— lets this specific output be checked against the audit trail without a lookup.",
    )


class ConfidenceEstimate(BaseModel):
    """Calibrated confidence from the conformal prediction / self-consistency layer."""
    method: str = Field(
        ..., description="'conformal', 'self_consistency', 'both', or 'uncalibrated' "
        "(no fitted calibration set — raw score passed through)"
    )
    point_estimate: float = Field(..., ge=0.0, le=1.0)
    interval: tuple[float, float] | None = Field(
        None,
        description="(lower, upper) conformal prediction interval",
    )
    num_consistent_paths: int | None = Field(
        None,
        description="Number of reasoning paths that agreed (self-consistency)",
    )
    total_paths: int | None = None
    calibrated: bool = Field(
        False,
        description="True only when a fitted calibration set backs this score. "
        "False means the raw specialist score was passed through unchanged — "
        "the interval is not a coverage guarantee.",
    )
    model_version: str | None = Field(
        None,
        description="name@version (kind) of the model(s) that produced the score, "
        "from models/registry.json",
    )


# ---------------------------------------------------------------------------
# Provenance chain (Fix 5)
# ---------------------------------------------------------------------------


class ClaimEvidenceLink(BaseModel):
    """Traces a factual claim to the tool output and artifact that produced it."""
    claim: str = Field(..., description="A factual statement from the final answer or change narrative")
    tool_name: str = Field(..., description="The specialist that produced the evidence for this claim")
    evidence_artifact_id: str | None = Field(
        None, description="artifact_id of the artifact backing this claim (if any)"
    )
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence of the backing evidence")

class ProvenanceChain(BaseModel):
    """Links a final answer to every piece of evidence that supports it.

    Makes every AI conclusion traceable: which artifacts were produced,
    which model versions ran, where the weakest confidence lies, and the
    audit chain hash that ties this chain to the tamper-evident log.
    """
    query_id: str
    audit_chain_hash: str = Field(
        ..., description="chain_hash from the audit log for this query's final event"
    )
    evidence_artifacts: list[EvidenceArtifact] = Field(
        default_factory=list,
        description="Every verifiable artifact that contributed to the answer"
    )
    spatial_evidence: list[SpatialEvidence] = Field(
        default_factory=list,
        description="Geospatially-located evidence backing the answer"
    )
    weakest_link: str | None = Field(
        None, description="tool_name of the step with the lowest confidence"
    )
    weakest_confidence: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Confidence of the weakest-link step"
    )
    model_versions: list[str] = Field(
        default_factory=list,
        description="All model versions that contributed to this answer"
    )
    claim_evidence_links: list[ClaimEvidenceLink] = Field(
        default_factory=list,
        description="Maps each factual claim in the final answer to the specialist "
        "output and artifact that produced it — Claim → Evidence → Artifact traceability."
    )


class ExecutionResult(BaseModel):
    """Complete result for a query, including all intermediate outputs."""
    query_id: str
    plan: ExecutionPlan
    outputs: list[SpecialistOutput]
    final_answer: str = Field(..., description="Natural language answer to the user's query")
    confidence: ConfidenceEstimate | None = None
    execution_time_s: float = 0.0
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this result was produced (completion time, not query submission).",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Deduplicated union of every output's warnings, in step order. Same "
        "field mission_summary()['warnings'] already reported for /v1/mission — now on "
        "every /v1/query result too, not just mission ones.",
    )
    provenance: ProvenanceChain | None = Field(
        None,
        description="Traces the final answer to every artifact + model that produced it. "
        "None only for policy-refused queries where nothing executed.",
    )
    change_narrative: str | None = Field(
        None,
        description="Natural-language description of *what* changed, synthesised from "
        "change-detection + VQA outputs. Not a trained Change-VQA model — rule-based "
        "composition of structured specialist results into prose. None when no "
        "change-detection tool ran.",
    )
    mission_summary: dict[str, Any] | None = Field(
        None,
        description="Mission-shaped envelope (prediction/spatial_evidence/metrics/"
        "confidence/warnings) — set by controller/missions.py for /v1/mission calls "
        "only; a plain /v1/query result leaves this None.",
    )


# ---------------------------------------------------------------------------
# Audit schemas
# ---------------------------------------------------------------------------

class AuditEntry(BaseModel):
    """A single entry in the hash-chained execution log."""
    entry_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    query_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(
        ...,
        description="One of: 'plan_created', 'tool_called', 'tool_returned', 'answer_emitted'",
    )
    tool_name: str | None = None
    input_hash: str = Field(..., description="SHA-256 of the serialised input to this step")
    output_hash: str = Field(..., description="SHA-256 of the serialised output from this step")
    chain_hash: str = Field(
        ...,
        description="SHA-256(parent_chain_hash || event_json) — tamper-evident chain link",
    )
    parent_chain_hash: str = Field(
        "",
        description="chain_hash of the preceding entry (empty string for the first entry)",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra context for this event",
    )
