"""
M08 — Standard Result/Error Contracts.

Unified data contracts for every M01–M10 module.  Old AETHERIS results
(``ExecutionResult``, ``SpecialistOutput``, ``AuditEntry``) are wrapped by
adapter helpers at the bottom of this file — old code is imported, never edited.
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

class ModuleID(StrEnum):
    M01 = "m01_baseline"
    M02 = "m02_policy"
    M03 = "m03_dag"
    M04 = "m04_auth"
    M05 = "m05_rate_limit"
    M06 = "m06_upload_security"
    M07 = "m07_session_isolation"
    M08 = "m08_contracts"
    M09 = "m09_geospatial"
    M10 = "m10_radiometric"
    M11 = "m11_change_preprocessing"
    M12 = "m12_geotiff_artifacts"
    M13 = "m13_conformal"
    M14 = "m14_provenance"
    M15 = "m15_csv_analysis"
    M19 = "m19_evidence_graph"
    M20 = "m20_counterfactual"


class ResultStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    PENDING = "pending"
    RUNNING = "running"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Core contracts
# ---------------------------------------------------------------------------

class ModuleError(BaseModel):
    """Standardized error from any M01–M10 module."""
    code: str = Field(..., description="Machine-readable error code, e.g. 'POLICY_VIOLATION'")
    message: str = Field(..., description="Human-readable error description")
    module: ModuleID
    severity: Severity = Severity.ERROR
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModuleResult(BaseModel):
    """Result from a single M01–M10 module execution."""
    module: ModuleID
    status: ResultStatus = ResultStatus.SUCCESS
    data: dict[str, Any] = Field(default_factory=dict)
    errors: list[ModuleError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    execution_time_s: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def success(cls, module: ModuleID, data: dict[str, Any], exec_time: float = 0.0) -> "ModuleResult":
        return cls(module=module, status=ResultStatus.SUCCESS, data=data, execution_time_s=exec_time)

    @classmethod
    def failure(cls, module: ModuleID, code: str, message: str, **details: Any) -> "ModuleResult":
        return cls(
            module=module,
            status=ResultStatus.FAILURE,
            errors=[ModuleError(code=code, message=message, module=module, details=details)],
        )

    @classmethod
    def skipped(cls, module: ModuleID, reason: str) -> "ModuleResult":
        return cls(module=module, status=ResultStatus.SKIPPED, data={"reason": reason})


class PipelineResult(BaseModel):
    """Complete pipeline result encompassing all module outputs."""
    pipeline_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    status: ResultStatus = ResultStatus.SUCCESS
    modules: dict[str, ModuleResult] = Field(default_factory=dict)
    final_answer: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)
    total_execution_time_s: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def add_module_result(self, result: ModuleResult) -> None:
        self.modules[result.module.value] = result
        if result.status == ResultStatus.FAILURE:
            self.status = ResultStatus.PARTIAL

    def compute_status(self) -> None:
        statuses = {r.status for r in self.modules.values()}
        if not statuses:
            self.status = ResultStatus.PENDING
        elif all(s == ResultStatus.SUCCESS for s in statuses):
            self.status = ResultStatus.SUCCESS
        elif all(s == ResultStatus.FAILURE for s in statuses):
            self.status = ResultStatus.FAILURE
        elif ResultStatus.FAILURE in statuses:
            self.status = ResultStatus.PARTIAL
        elif statuses == {ResultStatus.SKIPPED}:
            # Genuinely nothing ran (was previously the catch-all `else:
            # SUCCESS`, indistinguishable from every module succeeding).
            self.status = ResultStatus.PARTIAL
        else:
            self.status = ResultStatus.SUCCESS


class PipelineRequest(BaseModel):
    """Incoming request to the M01–M10 pipeline."""
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    question: str = ""
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    user_id: str = "anonymous"
    password: str | None = Field(
        None,
        description="Credential for user_id. Omit (or use user_id='anonymous') "
        "for anonymous access. Previously the pipeline called authenticate() "
        "with a hardcoded blank password, so named users could never actually "
        "log in through the full pipeline — every request silently became "
        "anonymous regardless of user_id.",
    )
    modality: str = "optical"
    modality2: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Adapters for old AETHERIS results
# ---------------------------------------------------------------------------

def wrap_execution_result(old_result: Any) -> dict[str, Any]:
    """Convert an old ``ExecutionResult`` into the new contract format."""
    try:
        data = old_result.model_dump(mode="json")
    except AttributeError:
        data = old_result if isinstance(old_result, dict) else {"raw": str(old_result)}
    return {
        "query_id": data.get("query_id", ""),
        "final_answer": data.get("final_answer", ""),
        "plan": data.get("plan", {}),
        "outputs": data.get("outputs", []),
        "confidence": data.get("confidence"),
        "execution_time_s": data.get("execution_time_s", 0.0),
    }


def wrap_audit_entry(old_entry: Any) -> dict[str, Any]:
    """Convert an old ``AuditEntry`` into the new contract format."""
    try:
        return old_entry.model_dump(mode="json")
    except AttributeError:
        return old_entry if isinstance(old_entry, dict) else {"raw": str(old_entry)}


def wrap_policy_error(violation_msg: str) -> ModuleError:
    """Wrap an old ``PolicyViolation`` message into the new error contract."""
    return ModuleError(
        code="POLICY_VIOLATION",
        message=violation_msg,
        module=ModuleID.M02,
        severity=Severity.ERROR,
    )


def wrap_specialist_output(old_output: Any) -> dict[str, Any]:
    """Convert an old ``SpecialistOutput`` into the new contract format."""
    try:
        return old_output.model_dump(mode="json")
    except AttributeError:
        return old_output if isinstance(old_output, dict) else {"raw": str(old_output)}
