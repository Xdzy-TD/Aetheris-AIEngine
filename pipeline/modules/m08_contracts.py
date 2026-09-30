"""
M08 — Standard Result/Error Contracts.

This module provides the execution-time API for creating and working
with the standard contracts defined in ``schemas.contracts``.
"""

from __future__ import annotations

import time
from typing import Any

from pipeline.schemas.contracts import (
    ModuleID,
    ModuleResult,
    PipelineRequest,
    PipelineResult,
    ResultStatus,
    wrap_execution_result,
)


class ContractManager:
    """Manages creation and validation of standard contracts."""

    @staticmethod
    def create_pipeline_result(request: PipelineRequest) -> PipelineResult:
        """Create a new pipeline result for tracking."""
        return PipelineResult(pipeline_id=request.request_id)

    @staticmethod
    def finalize_result(
        result: PipelineResult,
        final_answer: str | None = None,
    ) -> PipelineResult:
        """Finalize a pipeline result — compute status and timing."""
        result.compute_status()
        if final_answer:
            result.final_answer = final_answer
        result.total_execution_time_s = sum(
            m.execution_time_s for m in result.modules.values()
        )
        return result

    @staticmethod
    def wrap_old_result(old_result: Any) -> dict[str, Any]:
        """Wrap an old AETHERIS result into the new contract."""
        return wrap_execution_result(old_result)

    @staticmethod
    def validate_module_result(result: ModuleResult) -> bool:
        """Validate that a module result is well-formed."""
        if not result.module:
            return False
        if result.status not in ResultStatus:
            return False
        if result.status == ResultStatus.FAILURE and not result.errors:
            return False
        return True

    @staticmethod
    def get_contract_summary() -> dict[str, Any]:
        """Return a summary of the contract system."""
        return {
            "modules": [m.value for m in ModuleID],
            "statuses": [s.value for s in ResultStatus],
            "version": "1.0.0",
        }
