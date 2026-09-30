"""
SAT-OS domain schemas.

Every SAT-OS–specific data contract lives here. Reuses controller.schemas
types (EvidenceArtifact, SpatialEvidence, etc.) wherever possible.
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

class GateStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    QUALIFIED = "qualified"
    ABSTAIN = "abstain"


class EvidenceRole(StrEnum):
    ESSENTIAL = "essential"
    SUPPORTIVE = "supportive"
    REDUNDANT = "redundant"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Observation Contract (§1 of implementation.md)
# ---------------------------------------------------------------------------

class ObservationContract(BaseModel):
    """Structured observation contract compiled from a natural-language query."""
    target: str = Field("", description="What is being observed (e.g. 'river basin', 'urban area')")
    phenomenon: str = Field("", description="What phenomenon to detect (e.g. 'flooding', 'deforestation')")
    spatial_scale: str = Field("", description="Expected spatial scale (e.g. 'city-block', 'regional')")
    temporal_requirement: str = Field("", description="Temporal constraint (e.g. 'bi-temporal', 'single-date')")
    sensor_modality: str = Field("optical", description="Required sensor type")
    required_gsd_m: float | None = Field(None, description="Max acceptable GSD in metres")
    expected_output: str = Field("", description="What the answer should contain")
    quality_constraints: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Data Gate (§3)
# ---------------------------------------------------------------------------

class GateResult(BaseModel):
    """Result from a single observation gate check."""
    gate_name: str
    status: GateStatus
    reason: str = ""
    value: Any = None


class DataGateReport(BaseModel):
    """Aggregated gate results — overall status is the worst individual gate."""
    gates: list[GateResult] = Field(default_factory=list)
    overall: GateStatus = GateStatus.PASS
    critical_failures: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Observability (§4)
# ---------------------------------------------------------------------------

class ObservabilityStats(BaseModel):
    """Spatial observability percentages."""
    observable_pct: float = 0.0
    partially_observable_pct: float = 0.0
    unobservable_pct: float = 0.0
    mask_path: str | None = None


# ---------------------------------------------------------------------------
# Evidence Classification (§6)
# ---------------------------------------------------------------------------

class ClassifiedEvidence(BaseModel):
    """An evidence item with its role and leave-one-out sensitivity."""
    tool_name: str
    role: EvidenceRole = EvidenceRole.UNKNOWN
    removal_breaks_claim: bool = False
    confidence_contribution: float = 0.0


class EvidenceSetReport(BaseModel):
    """Evidence graph with minimum-evidence-set calculation."""
    items: list[ClassifiedEvidence] = Field(default_factory=list)
    minimum_set: list[str] = Field(default_factory=list, description="tool_names in the minimum evidence set")


# ---------------------------------------------------------------------------
# Falsification (§7)
# ---------------------------------------------------------------------------

class FalsificationCheck(BaseModel):
    """A single falsification test result."""
    hypothesis: str
    tested: bool = False
    supported: bool | None = None
    detail: str = ""


class FalsificationReport(BaseModel):
    """All falsification checks for a run."""
    checks: list[FalsificationCheck] = Field(default_factory=list)
    any_contradiction: bool = False


# ---------------------------------------------------------------------------
# Verification (§8) + Abstention (§9)
# ---------------------------------------------------------------------------

class VerificationDecision(BaseModel):
    """The final verification verdict for a claim."""
    status: VerificationStatus
    reasons: list[str] = Field(default_factory=list)
    observability_sufficient: bool = True
    data_quality_sufficient: bool = True
    evidence_sufficient: bool = True
    falsification_passed: bool = True
    abstention_detail: str | None = Field(
        None, description="If ABSTAIN: what prevents the claim + what data would help"
    )


# ---------------------------------------------------------------------------
# Answerability Certificate (§10)
# ---------------------------------------------------------------------------

class AnswerabilityCertificate(BaseModel):
    """Structured certificate for every completed SAT-OS run."""
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    query_text: str = ""
    contract: ObservationContract | None = None
    input_files: list[dict[str, Any]] = Field(default_factory=list)
    input_hashes: dict[str, str] = Field(default_factory=dict)
    algorithms_used: list[str] = Field(default_factory=list)
    observability: ObservabilityStats | None = None
    data_gate: DataGateReport | None = None
    evidence_set: EvidenceSetReport | None = None
    falsification: FalsificationReport | None = None
    verification: VerificationDecision | None = None
    assumptions: list[str] = Field(default_factory=list)
    output_hashes: dict[str, str] = Field(default_factory=dict)
    pipeline_version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Run Record (§12) — immutable reproducibility envelope
# ---------------------------------------------------------------------------

class RunRecord(BaseModel):
    """Immutable run record for reproducibility."""
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config: dict[str, Any] = Field(default_factory=dict)
    input_hashes: dict[str, str] = Field(default_factory=dict)
    contract_version: str = "1.0"
    pipeline_version: str = "0.1.0"
    model_versions: list[str] = Field(default_factory=list)
    intermediate_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    output_hashes: dict[str, str] = Field(default_factory=dict)
    certificate: AnswerabilityCertificate | None = None


# ---------------------------------------------------------------------------
# Feedback (§ Engineering Requirement)
# ---------------------------------------------------------------------------

class FeedbackEntry(BaseModel):
    """A single feedback record for the learning loop."""
    feedback_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    category: str = Field("", description="failed_gate | weak_claim | abstention | operator_correction | model_error | data_quality | observability_error")
    detail: str = ""
    correction: dict[str, Any] = Field(default_factory=dict)
