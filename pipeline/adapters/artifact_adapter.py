"""
Artifact Adapter — wraps ``controller.artifacts.ArtifactStore`` without editing it.

Reuses the existing artifact persistence, hash verification, and evidence
recording for M06/M08 results.
"""

from __future__ import annotations

from typing import Any

from controller.artifacts import ArtifactStore, default_store

from pipeline.schemas.contracts import ModuleID, ModuleResult


class ArtifactAdapter:
    """Adapter around the existing ArtifactStore."""

    def __init__(self, store: ArtifactStore | None = None) -> None:
        self._store = store or default_store()

    @property
    def store(self) -> ArtifactStore:
        return self._store

    def store_artifact(
        self,
        data: bytes,
        artifact_type: str,
        model_version: str = "m01_m10@1.0",
        query_id: str = "",
        suffix: str = "",
    ) -> dict[str, Any]:
        """Store an artifact using the existing store."""
        record = self._store.put(
            data, artifact_type, model_version,
            query_id=query_id, suffix=suffix,
        )
        return dict(record)

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Get an artifact record."""
        return self._store.get_record(artifact_id)

    def verify_artifact(self, artifact_id: str) -> ModuleResult:
        """Verify an artifact's integrity."""
        is_valid = self._store.verify(artifact_id)
        if is_valid:
            record = self._store.get_record(artifact_id)
            return ModuleResult.success(
                ModuleID.M08,
                {"artifact_id": artifact_id, "verified": True, "record": record},
            )
        return ModuleResult.failure(
            ModuleID.M08,
            "ARTIFACT_INTEGRITY_FAILED",
            f"Artifact {artifact_id} failed integrity verification",
        )

    def list_artifacts(self) -> list[dict[str, Any]]:
        """List all artifacts in the store."""
        artifacts = []
        for meta_path in self._store.root.glob("*.meta.json"):
            import json
            try:
                record = json.loads(meta_path.read_text(encoding="utf-8"))
                artifacts.append(record)
            except Exception:
                continue
        return artifacts
