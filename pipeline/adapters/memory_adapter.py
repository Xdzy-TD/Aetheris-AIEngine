"""
Memory Adapter — wraps ``controller.memory.RoutingMemory`` without editing it.

Adds session/job namespace isolation, TTL tracking, and cleanup
for M07 while delegating actual caching to the old implementation.
"""

from __future__ import annotations

import time
from typing import Any

from controller.memory import RoutingMemory

from pipeline.schemas.contracts import ModuleID, ModuleResult


class MemoryAdapter:
    """Adapter around the existing RoutingMemory with namespace isolation."""

    def __init__(self, max_entries: int = 512, default_ttl: int = 3600) -> None:
        self._memory = RoutingMemory(max_entries=max_entries)
        self._default_ttl = default_ttl
        self._namespaces: dict[str, dict[str, Any]] = {}
        self._ttl_records: dict[str, float] = {}

    @property
    def inner(self) -> RoutingMemory:
        return self._memory

    def create_namespace(self, session_id: str, job_id: str = "") -> str:
        """Create an isolated namespace for a session/job pair."""
        ns = f"{session_id}:{job_id}" if job_id else session_id
        self._namespaces[ns] = {
            "session_id": session_id,
            "job_id": job_id,
            "created_at": time.time(),
            "entry_count": 0,
        }
        self._ttl_records[ns] = time.time() + self._default_ttl
        return ns

    def store(
        self,
        namespace: str,
        image_data: bytes | str,
        tool_name: str,
        result: dict[str, Any],
    ) -> None:
        """Store a result in the namespace-isolated cache."""
        prefixed_session = f"ns_{namespace}"
        self._memory.store(prefixed_session, image_data, tool_name, result)
        if namespace in self._namespaces:
            self._namespaces[namespace]["entry_count"] += 1

    def recall(
        self,
        namespace: str,
        image_data: bytes | str,
        tool_name: str,
    ) -> dict[str, Any] | None:
        """Recall a result from the namespace-isolated cache."""
        # Check TTL
        if namespace in self._ttl_records:
            if time.time() > self._ttl_records[namespace]:
                self.clear_namespace(namespace)
                return None
        prefixed_session = f"ns_{namespace}"
        return self._memory.recall(prefixed_session, image_data, tool_name)

    def clear_namespace(self, namespace: str) -> int:
        """Clear all cache entries for a namespace."""
        prefixed_session = f"ns_{namespace}"
        count = self._memory.clear_session(prefixed_session)
        self._namespaces.pop(namespace, None)
        self._ttl_records.pop(namespace, None)
        return count

    def clear_expired(self) -> int:
        """Clear all expired namespaces."""
        now = time.time()
        expired = [ns for ns, ttl in self._ttl_records.items() if now > ttl]
        total = 0
        for ns in expired:
            total += self.clear_namespace(ns)
        return total

    def validate_isolation(self, namespace: str) -> ModuleResult:
        """Validate that a namespace is properly isolated."""
        if namespace not in self._namespaces:
            return ModuleResult.failure(
                ModuleID.M07,
                "NAMESPACE_NOT_FOUND",
                f"Namespace '{namespace}' does not exist",
            )
        if namespace in self._ttl_records and time.time() > self._ttl_records[namespace]:
            return ModuleResult.failure(
                ModuleID.M07,
                "NAMESPACE_EXPIRED",
                f"Namespace '{namespace}' has expired",
            )
        return ModuleResult.success(
            ModuleID.M07,
            {
                "namespace": namespace,
                "isolated": True,
                "info": self._namespaces[namespace],
                "ttl_remaining_s": max(0, self._ttl_records.get(namespace, 0) - time.time()),
            },
        )

    def stats(self) -> dict[str, Any]:
        """Return enhanced stats with namespace information."""
        base_stats = self._memory.stats()
        return {
            **base_stats,
            "namespaces": len(self._namespaces),
            "expired_namespaces": sum(
                1 for ttl in self._ttl_records.values() if time.time() > ttl
            ),
        }
