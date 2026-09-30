"""
M14 — Audit / Provenance Upgrade.

``PipelineResult.artifacts`` and ``.audit_trail`` have existed on the M08
contract since it was written, but ``AetherisPipeline.execute()`` never
populated either — every response reported an empty audit trail and no
artifact list, even though the old hash-chained ``AuditLog`` and every
specialist's evidence record were being produced the whole time. M14 wires
the existing adapters into the final result, plus a chain-integrity flag so
a caller can see the trail hasn't been tampered with, not just that it exists.
"""

from __future__ import annotations

from typing import Any

from pipeline.adapters.audit_adapter import AuditAdapter
from pipeline.schemas.contracts import ModuleID, ModuleResult, PipelineResult


def attach_provenance(
    result: PipelineResult,
    audit: AuditAdapter,
    pipeline_id: str,
    outputs: list[dict[str, Any]],
    extra_artifacts: list[dict[str, Any]] | None = None,
) -> ModuleResult:
    """Fill in ``result.audit_trail``/``.artifacts`` and report chain integrity."""
    result.audit_trail = audit.get_trace(pipeline_id)
    result.artifacts = [
        rec for out in outputs if (rec := out.get("result", {}).get("artifact"))
    ]
    result.artifacts.extend(extra_artifacts or [])

    chain_valid, chain_errors = audit.verify_chain()
    return ModuleResult.success(
        ModuleID.M14,
        {
            "audit_entries": len(result.audit_trail),
            "artifact_count": len(result.artifacts),
            "chain_valid": chain_valid,
            "chain_errors": chain_errors[:5],
        },
    )
