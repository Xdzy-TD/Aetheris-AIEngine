"""
Result Contract — every SpecialistOutput gets its trust metadata from here.

Extracted from ``controller/planner.py`` so the same contract applies to both
``/v1/query`` (via PlanExecutor) and ``/v1/mission`` (via missions.py), and
no specialist has to report its own model_version/parameters/warnings.
"""

from __future__ import annotations

from typing import Any

from confidence.conformal import ConformalCalibrator
from controller.audit_log import AuditLog
from controller.model_registry import version_for_task
from controller.schemas import (
    ConfidenceEstimate,
    EvidenceArtifact,
    SpatialEvidence,
    SpecialistOutput,
)


class ResultContract:
    """Wraps raw tool results with trust metadata and rolls up confidence.

    Centralises the Result Contract (docs/STATUS.md §Trust) so
    ``_finish_output``, ``_derive_warnings``, ``_dedup_warnings``, and
    ``_estimate_confidence`` live in one place rather than being methods
    on a 700-line planner class.
    """

    def __init__(
        self,
        calibrator: ConformalCalibrator | None = None,
    ) -> None:
        # Unfitted: no labeled calibration set exists for these specialists,
        # so this honestly passes raw specialist confidence through instead
        # of fabricating a calibrated interval (see ConformalCalibrator.predict).
        self._calibrator = calibrator or ConformalCalibrator()

    @staticmethod
    def finish_output(
        tool_name: str,
        result: Any,
        parameters: dict[str, Any],
        query_bbox: list[float] | None = None,
        query_crs: str | None = None,
    ) -> SpecialistOutput:
        """Wrap a raw tool result with the Result Contract fields."""
        result_dict = result if isinstance(result, dict) else {"output": result}
        evidence_artifacts, spatial_evidence = ResultContract.extract_evidence(
            tool_name, result_dict, parameters, query_bbox, query_crs,
        )
        return SpecialistOutput(
            tool_name=tool_name,
            result=result_dict,
            model_version=version_for_task(tool_name),
            parameters=parameters,
            warnings=ResultContract.derive_warnings(tool_name, result_dict),
            provenance_hash=AuditLog._hash_obj(result_dict),
            evidence_artifacts=evidence_artifacts,
            spatial_evidence=spatial_evidence,
        )

    @staticmethod
    def extract_evidence(
        tool_name: str,
        result: dict[str, Any],
        parameters: dict[str, Any],
        query_bbox: list[float] | None = None,
        query_crs: str | None = None,
    ) -> tuple[list[EvidenceArtifact], list[SpatialEvidence]]:
        """Mine artifact records and geospatial metadata from a specialist result.

        This is called at the executor level so every tool gets evidence
        populated automatically — no specialist has to construct these objects
        itself.
        """
        evidence_artifacts: list[EvidenceArtifact] = []
        spatial_evidence: list[SpatialEvidence] = []
        model_version = version_for_task(tool_name)

        if result.get("analysis_unavailable"):
            return evidence_artifacts, spatial_evidence

        # --- Extract artifact records ---
        # Many specialists report artifact info under various keys.
        artifact_records = []
        # Single artifact (grounding mask, despeckled raster, etc.)
        if isinstance(result.get("artifact"), dict):
            artifact_records.append(result["artifact"])
        # Multiple artifacts (fusion_change_detection reports per-channel)
        if isinstance(result.get("artifacts"), list):
            artifact_records.extend(
                a for a in result["artifacts"] if isinstance(a, dict)
            )
        # Artifact store records with artifact_id at top level
        for key in ("artifact_id", "mask_artifact_id", "change_map_artifact_id"):
            aid = result.get(key)
            if aid and not any(a.get("artifact_id") == aid for a in artifact_records):
                artifact_records.append({
                    "artifact_id": aid,
                    "sha256": result.get("sha256", result.get(f"{key}_sha256", "")),
                    "path": result.get("path", result.get("output_path", "")),
                    "type": result.get("type", tool_name),
                })

        for arec in artifact_records:
            aid = arec.get("artifact_id", "")
            if not aid:
                continue
            evidence_artifacts.append(EvidenceArtifact(
                artifact_id=aid,
                sha256=arec.get("sha256", ""),
                artifact_type=arec.get("type", tool_name),
                model_version=model_version,
                path=str(arec.get("path", "")),
                size_bytes=arec.get("size_bytes", 0),
            ))

        # --- Extract spatial evidence ---
        # Use query-level geospatial info (bbox, crs) if available.
        change_fraction = None
        if isinstance(result.get("change_fraction"), (int, float)):
            change_fraction = result["change_fraction"]
        elif isinstance(result.get("change_percent"), (int, float)):
            change_fraction = result["change_percent"] / 100.0

        # Build a SpatialEvidence entry if we have any geo context
        if query_bbox or change_fraction is not None or evidence_artifacts:
            spatial_evidence.append(SpatialEvidence(
                tool_name=tool_name,
                artifact=evidence_artifacts[0] if evidence_artifacts else None,
                bbox=query_bbox,
                crs=query_crs,
                change_fraction=change_fraction,
            ))

        return evidence_artifacts, spatial_evidence

    @staticmethod
    def dedup_warnings(outputs: list[SpecialistOutput]) -> list[str]:
        """Union of every output's warnings, order-preserving, no duplicates."""
        seen: dict[str, None] = {}
        for out in outputs:
            for w in out.warnings:
                seen.setdefault(w, None)
        return list(seen)

    @staticmethod
    def derive_warnings(tool_name: str, result: dict[str, Any]) -> list[str]:
        """Human-readable caveats traced to fields already in ``result``.

        Never invents a caveat: unavailability, low confidence (<0.4) and
        unreliable co-registration are the only signals a specialist result
        already carries. Shared with ``controller.missions.mission_summary``
        so a mission result and a plain query result flag the same things
        the same way.
        """
        if result.get("analysis_unavailable"):
            return [f"{tool_name}: {result.get('reason', 'unavailable')}"]
        warnings: list[str] = []
        conf = result.get("confidence")
        if isinstance(conf, int | float) and conf < 0.4:
            warnings.append(f"{tool_name}: low confidence ({conf})")
        if result.get("coregistration_reliable") is False:
            warnings.append(f"{tool_name}: co-registration unreliable, shift not applied")
        return warnings

    def estimate_confidence(self, outputs: list[SpecialistOutput]) -> ConfidenceEstimate | None:
        """Roll specialist-reported confidences into a single estimate.

        Takes the weakest (minimum) confidence across the chain — the
        answer is only as trustworthy as its least confident step — and
        passes it through the (unfitted) conformal calibrator, which
        honestly reports itself as "uncalibrated" rather than fabricating
        a statistically-valid interval with no calibration data behind it.
        """
        # A step that could not run has no confidence to contribute. Reading its
        # absent score as 0.0 would understate a chain that did work.
        scoring = [
            (out.tool_name, out.result[key])
            for out in outputs
            if not out.result.get("analysis_unavailable")
            for key in ("confidence", "raw_confidence")
            if key in out.result and isinstance(out.result[key], int | float)
        ]
        # Self-consistency agreement is itself a confidence signal (see
        # confidence/self_consistency.py) — folded into the same weakest-
        # link pool as the specialists' own reported confidence.
        votes = [
            (out.tool_name, out.result["self_consistency"])
            for out in outputs
            if isinstance(out.result.get("self_consistency"), dict)
        ]
        scoring += [(tool, v["agreement_ratio"]) for tool, v in votes]
        if not scoring:
            return None

        tasks = sorted({tool for tool, _ in scoring})
        pred = self._calibrator.predict(min(score for _, score in scoring))
        method = pred["method"]
        if votes:
            method = "self_consistency" if method == "uncalibrated" else "both"
        weakest_vote = min(votes, key=lambda tv: tv[1]["agreement_ratio"], default=(None, None))[1]
        return ConfidenceEstimate(
            method=method,
            point_estimate=pred["calibrated_score"],
            interval=(pred["lower"], pred["upper"]),
            num_consistent_paths=weakest_vote["n_consistent"] if weakest_vote else None,
            total_paths=weakest_vote["total_paths"] if weakest_vote else None,
            # Derived from the calibrator's own state, never asserted: an
            # unfitted calibrator passes the raw score through untouched.
            calibrated=self._calibrator.is_fitted(),
            model_version=", ".join(version_for_task(t) for t in tasks) or None,
        )
