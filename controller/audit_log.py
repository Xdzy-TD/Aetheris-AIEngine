"""
Hash-chained, tamper-evident execution log.

Every controller decision, tool call, and intermediate result is hashed and
chained so the full reasoning trace for an answer can be audited and cannot
be silently edited after the fact.

Selected differentiator: Integrity & Trust.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from controller.schemas import AuditEntry

logger = structlog.get_logger(__name__)

# Default log location (relative to project root)
_DEFAULT_LOG_PATH = Path("logs/execution_chain.jsonl")


class AuditLog:
    """Append-only, hash-chained execution log.

    Each entry's ``chain_hash`` is ``SHA-256(parent_chain_hash || entry_json)``,
    creating a tamper-evident chain: modifying any past entry invalidates every
    subsequent hash.
    """

    def __init__(self, log_path: Path | str | None = None) -> None:
        self._log_path = Path(log_path) if log_path else _DEFAULT_LOG_PATH
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_chain_hash: str = ""
        self._lock = threading.Lock()
        # Recover chain tip if log already exists
        self._recover_chain_tip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_event(
        self,
        query_id: str,
        event_type: str,
        input_data: Any,
        output_data: Any,
        tool_name: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Append a new entry to the execution log and return it.

        Args:
            query_id:   Unique identifier for the originating query.
            event_type: One of ``plan_created``, ``tool_called``,
                        ``tool_returned``, ``answer_emitted``.
            input_data: Serialisable input to this step.
            output_data: Serialisable output from this step.
            tool_name:  Name of the specialist tool (if applicable).
            extra:      Arbitrary extra metadata.

        Returns:
            The ``AuditEntry`` that was persisted.
        """
        input_hash = self._hash_obj(input_data)
        output_hash = self._hash_obj(output_data)

        # Reading the chain tip, computing the new hash, updating the tip,
        # and appending must be one critical section — otherwise two
        # concurrent requests can both read the same parent hash and the
        # file ends up with two entries claiming the same parent.
        with self._lock:
            entry = AuditEntry(
                query_id=query_id,
                timestamp=datetime.now(UTC),
                event_type=event_type,
                tool_name=tool_name,
                input_hash=input_hash,
                output_hash=output_hash,
                parent_chain_hash=self._last_chain_hash,
                chain_hash="",  # computed below
                payload=extra or {},
            )
            entry_json = entry.model_dump_json(exclude={"chain_hash"})
            chain_hash = self._sha256(self._last_chain_hash + entry_json)
            entry.chain_hash = chain_hash
            self._last_chain_hash = chain_hash
            self._append(entry)
        logger.info(
            "audit_entry_logged",
            query_id=query_id,
            event_type=event_type,
            tool_name=tool_name,
            chain_hash=chain_hash[:16] + "…",
        )
        return entry

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the entire hash chain from disk.

        Returns:
            ``(is_valid, list_of_error_messages)``
        """
        entries = self._read_all()
        errors: list[str] = []
        prev_hash = ""

        for idx, entry in enumerate(entries):
            entry_json = entry.model_dump_json(exclude={"chain_hash"})
            expected = self._sha256(prev_hash + entry_json)
            if entry.chain_hash != expected:
                errors.append(
                    f"Entry {idx} ({entry.entry_id}): expected chain_hash "
                    f"{expected[:16]}…, got {entry.chain_hash[:16]}…"
                )
            if entry.parent_chain_hash != prev_hash:
                errors.append(
                    f"Entry {idx} ({entry.entry_id}): parent_chain_hash mismatch"
                )
            # Cascade the *recomputed* hash forward, not the stored one — a
            # tampered entry's on-disk chain_hash is untrustworthy, so
            # trusting it here would let verification "resync" past the
            # tamper point and stop reporting breaks in every entry after it.
            prev_hash = expected

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("audit_chain_verified", entries=len(entries), status="valid")
        else:
            logger.warning("audit_chain_broken", errors=errors)
        return is_valid, errors

    def get_trace(self, query_id: str) -> list[AuditEntry]:
        """Return all audit entries for a specific query, in order."""
        return [e for e in self._read_all() if e.query_id == query_id]

    def get_last_n(self, n: int = 20) -> list[AuditEntry]:
        """Return the last *n* entries from the log."""
        entries = self._read_all()
        return entries[-n:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_obj(obj: Any) -> str:
        """Deterministically hash an arbitrary object via sorted JSON."""
        raw = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _append(self, entry: AuditEntry) -> None:
        """Write one entry. Caller must already hold ``self._lock``."""
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def _read_all(self) -> list[AuditEntry]:
        if not self._log_path.exists():
            return []
        entries: list[AuditEntry] = []
        with self._log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(AuditEntry.model_validate_json(line))
        return entries

    def _recover_chain_tip(self) -> None:
        """Read the last entry's chain_hash to resume appending."""
        if not self._log_path.exists():
            return
        last_line = ""
        with self._log_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if last_line:
            entry = AuditEntry.model_validate_json(last_line)
            self._last_chain_hash = entry.chain_hash
            logger.info(
                "audit_chain_recovered",
                last_hash=entry.chain_hash[:16] + "…",
            )
