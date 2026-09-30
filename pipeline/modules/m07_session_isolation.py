"""
M07 — Session/Cache Isolation.

Provides strict session and job namespaces, cache key isolation,
TTL management, and cleanup — wrapping the old RoutingMemory.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pipeline.adapters.memory_adapter import MemoryAdapter
from pipeline.config import MAX_CACHE_ENTRIES, SESSION_TTL_SECONDS
from pipeline.schemas.contracts import ModuleID, ModuleResult


class SessionManager:
    """Manages session and job isolation for the pipeline."""

    def __init__(self) -> None:
        self._adapter = MemoryAdapter(
            max_entries=MAX_CACHE_ENTRIES,
            default_ttl=SESSION_TTL_SECONDS,
        )
        self._sessions: dict[str, dict[str, Any]] = {}

    @property
    def adapter(self) -> MemoryAdapter:
        return self._adapter

    def create_session(self, user_id: str = "anonymous") -> ModuleResult:
        """Create a new isolated session."""
        session_id = uuid.uuid4().hex[:12]
        namespace = self._adapter.create_namespace(session_id)
        self._sessions[session_id] = {
            "user_id": user_id,
            "created_at": time.time(),
            "namespace": namespace,
            "job_count": 0,
        }
        return ModuleResult.success(
            ModuleID.M07,
            {
                "session_id": session_id,
                "namespace": namespace,
                "user_id": user_id,
                "ttl_s": SESSION_TTL_SECONDS,
                "isolated": True,
            },
        )

    def create_job_namespace(self, session_id: str) -> ModuleResult:
        """Create a job-level namespace within a session."""
        if session_id not in self._sessions:
            return ModuleResult.failure(
                ModuleID.M07,
                "SESSION_NOT_FOUND",
                f"Session '{session_id}' does not exist",
            )

        job_id = uuid.uuid4().hex[:8]
        namespace = self._adapter.create_namespace(session_id, job_id)
        self._sessions[session_id]["job_count"] += 1

        return ModuleResult.success(
            ModuleID.M07,
            {
                "session_id": session_id,
                "job_id": job_id,
                "namespace": namespace,
                "isolated": True,
            },
        )

    def store_cache(
        self,
        session_id: str,
        key: str,
        tool_name: str,
        result: dict[str, Any],
    ) -> None:
        """Store a result in the session-isolated cache."""
        self._adapter.store(session_id, key, tool_name, result)

    def recall_cache(
        self,
        session_id: str,
        key: str,
        tool_name: str,
    ) -> dict[str, Any] | None:
        """Recall a result from the session-isolated cache."""
        return self._adapter.recall(session_id, key, tool_name)

    def validate_isolation(self, session_id: str) -> ModuleResult:
        """Validate that a session is properly isolated."""
        if session_id not in self._sessions:
            return ModuleResult.failure(
                ModuleID.M07,
                "SESSION_NOT_FOUND",
                f"Session '{session_id}' does not exist",
            )
        return self._adapter.validate_isolation(session_id)

    def cleanup_session(self, session_id: str) -> ModuleResult:
        """Clean up a session and all its cached data."""
        if session_id not in self._sessions:
            return ModuleResult.failure(
                ModuleID.M07,
                "SESSION_NOT_FOUND",
                f"Session '{session_id}' does not exist",
            )
        count = self._adapter.clear_namespace(session_id)
        self._sessions.pop(session_id, None)
        return ModuleResult.success(
            ModuleID.M07,
            {"session_id": session_id, "entries_cleared": count, "cleaned": True},
        )

    def cleanup_expired(self) -> ModuleResult:
        """Clean up all expired sessions."""
        count = self._adapter.clear_expired()
        return ModuleResult.success(
            ModuleID.M07,
            {"expired_cleared": count},
        )

    def get_status(self) -> dict[str, Any]:
        """Return session manager status."""
        return {
            "active_sessions": len(self._sessions),
            "cache_stats": self._adapter.stats(),
            "ttl_seconds": SESSION_TTL_SECONDS,
            "max_cache_entries": MAX_CACHE_ENTRIES,
        }
