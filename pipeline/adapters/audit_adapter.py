"""
Audit Adapter — wraps ``controller.audit_log.AuditLog`` without editing it.

Extends the existing hash-chained audit log with new M01–M10 event types
while preserving the tamper-evident chain.
"""

from __future__ import annotations

from typing import Any

from controller.audit_log import AuditLog
from interfaces.api.deps import get_audit

from pipeline.schemas.contracts import ModuleID, wrap_audit_entry


class AuditAdapter:
    """Adapter around the existing AuditLog for M01–M10 events."""

    def __init__(self, audit: AuditLog | None = None) -> None:
        self._audit = audit

    @property
    def audit(self) -> AuditLog:
        if self._audit is None:
            # Share the same process-wide AuditLog the base API uses
            # (interfaces/api/deps.py::get_audit) instead of opening a
            # second independent handle on logs/execution_chain.jsonl —
            # two instances each caching their own chain tip in memory is
            # what produced the parent_chain_hash mismatches.
            self._audit = get_audit()
        return self._audit

    def log_module_event(
        self,
        module: ModuleID,
        event_type: str,
        query_id: str,
        input_data: Any = None,
        output_data: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log an M01–M10 event through the existing audit chain."""
        full_event_type = f"m01_m10:{module.value}:{event_type}"
        entry = self.audit.log_event(
            query_id=query_id,
            event_type=full_event_type,
            input_data=input_data or {},
            output_data=output_data or {},
            tool_name=module.value,
            extra=extra,
        )
        return wrap_audit_entry(entry)

    def log_pipeline_start(self, pipeline_id: str, request_data: dict[str, Any]) -> dict[str, Any]:
        return self.log_module_event(
            ModuleID.M03, "pipeline_start", pipeline_id,
            input_data=request_data,
        )

    def log_pipeline_complete(self, pipeline_id: str, result_data: dict[str, Any]) -> dict[str, Any]:
        return self.log_module_event(
            ModuleID.M03, "pipeline_complete", pipeline_id,
            output_data=result_data,
        )

    def log_auth_event(self, user_id: str, action: str, allowed: bool) -> dict[str, Any]:
        return self.log_module_event(
            ModuleID.M04, "auth_decision", user_id,
            input_data={"action": action},
            output_data={"allowed": allowed},
        )

    def log_upload_event(self, query_id: str, filename: str, result: str) -> dict[str, Any]:
        return self.log_module_event(
            ModuleID.M06, "upload_validation", query_id,
            input_data={"filename": filename},
            output_data={"result": result},
        )

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Delegate chain verification to the old audit log."""
        return self.audit.verify_chain()

    def get_recent(self, n: int = 20) -> list[dict[str, Any]]:
        """Get recent audit entries."""
        entries = self.audit.get_last_n(n)
        return [wrap_audit_entry(e) for e in entries]

    def get_trace(self, query_id: str) -> list[dict[str, Any]]:
        """Get audit trail for a specific query."""
        entries = self.audit.get_trace(query_id)
        return [wrap_audit_entry(e) for e in entries]
