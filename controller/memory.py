"""
Routing memory — session-scoped cache for specialist outputs.

The controller retains a lightweight memory of prior tool invocations within
a session, so follow-up queries on the same AOI/image reuse prior embeddings,
masks, and results instead of recomputing them.

Selected differentiator: Agentic Intelligence.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """A single cached specialist result."""
    tool_name: str
    input_hash: str
    result: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class RoutingMemory:
    """Session-scoped routing memory for the agentic controller.

    Keys are ``(session_id, image_hash, tool_name)`` tuples. When the
    controller sees a follow-up query on the same image within the same
    session, it can skip recomputation for tools whose inputs haven't
    changed.

    Week 4 will add:
    - LRU eviction when memory exceeds a configurable cap.
    - Optional Redis backend for multi-process demos.
    """

    def __init__(self, max_entries: int = 512) -> None:
        self._store: dict[tuple[str, str, str], CacheEntry] = {}
        self._max_entries = max_entries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        session_id: str,
        image_data: bytes | str,
        tool_name: str,
        result: dict[str, Any],
    ) -> None:
        """Cache a specialist result keyed by session, image, and tool."""
        img_hash = self._hash_image(image_data)
        key = (session_id, img_hash, tool_name)

        self._store[key] = CacheEntry(
            tool_name=tool_name,
            input_hash=img_hash,
            result=result,
        )
        logger.debug(
            "routing_memory_stored",
            session_id=session_id,
            tool=tool_name,
            image_hash=img_hash[:12],
        )
        # Basic eviction: drop oldest entry if over capacity
        if len(self._store) > self._max_entries:
            self._evict_oldest()

    def recall(
        self,
        session_id: str,
        image_data: bytes | str,
        tool_name: str,
    ) -> dict[str, Any] | None:
        """Look up a cached result. Returns ``None`` on miss."""
        img_hash = self._hash_image(image_data)
        key = (session_id, img_hash, tool_name)
        entry = self._store.get(key)

        if entry is None:
            logger.debug(
                "routing_memory_miss",
                session_id=session_id,
                tool=tool_name,
            )
            return None

        entry.access_count += 1
        entry.last_accessed = time.time()
        logger.debug(
            "routing_memory_hit",
            session_id=session_id,
            tool=tool_name,
            access_count=entry.access_count,
        )
        return entry.result

    def has_cached(
        self,
        session_id: str,
        image_data: bytes | str,
        tool_name: str,
    ) -> bool:
        """Check whether a cached result exists without touching access stats."""
        img_hash = self._hash_image(image_data)
        return (session_id, img_hash, tool_name) in self._store

    def clear_session(self, session_id: str) -> int:
        """Remove all entries for a session. Returns count of evicted entries."""
        to_remove = [k for k in self._store if k[0] == session_id]
        for k in to_remove:
            del self._store[k]
        logger.info("routing_memory_session_cleared", session_id=session_id, evicted=len(to_remove))
        return len(to_remove)

    def stats(self) -> dict[str, Any]:
        """Return basic cache statistics."""
        return {
            "total_entries": len(self._store),
            "max_entries": self._max_entries,
            "sessions": len({k[0] for k in self._store}),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_image(data: bytes | str) -> str:
        """Hash image data (or a string path/identifier) for cache keying."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def _evict_oldest(self) -> None:
        """Drop the least-recently-accessed entry."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].last_accessed)
        evicted = self._store.pop(oldest_key)
        logger.debug(
            "routing_memory_evicted",
            tool=evicted.tool_name,
            input_hash=evicted.input_hash[:12],
        )
