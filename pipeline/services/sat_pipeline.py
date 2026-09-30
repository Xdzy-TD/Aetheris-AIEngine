"""
SAT-OS Pipeline — the full SAT-OS orchestrator.

Composes the SAT-OS vertical slice:
  Query → Contract → Data Gate → Observability → Analysis →
  Evidence → Falsification → Verification → Certificate → Report

Reuses AetherisPipeline as the inner execution engine for the analysis step.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import structlog

from pipeline.modules.sat_certificate import build_certificate, export_certificate_json
from pipeline.modules.sat_data_gate import run_data_gate
from pipeline.modules.sat_evidence_classifier import classify_evidence
from pipeline.modules.sat_falsification import run_falsification
from pipeline.modules.sat_feedback import FeedbackStore
from pipeline.modules.sat_observability import compute_observability
from pipeline.modules.sat_query_compiler import compile_query
from pipeline.modules.sat_report import generate_text_report, save_report
from pipeline.modules.sat_verification import decide_verification
from pipeline.schemas.sat_schemas import (
    AnswerabilityCertificate,
    FeedbackEntry,
    RunRecord,
    VerificationStatus,
)
from pipeline.services.pipeline import AetherisPipeline

from specialists._imagery import probe_raster_metadata

logger = structlog.get_logger(__name__)

_REPORT_DIR = Path("logs/sat_reports")


class SATOSPipeline:
    """Full SAT-OS orchestrator — the vertical slice from implementation.md."""

    def __init__(self) -> None:
        self._inner = AetherisPipeline()
        self._feedback = FeedbackStore()
        self._runs: dict[str, RunRecord] = {}

    def execute(
        self,
        question: str,
        file_path: str | None = None,
        file2_path: str | None = None,
        file_name: str = "",
        file2_name: str = "",
        file_data: bytes | None = None,
        modality: str = "optical",
        modality2: str | None = None,
        user_id: str = "anonymous",
    ) -> dict[str, Any]:
        """Execute the full SAT-OS pipeline and return the complete result."""
        t0 = time.perf_counter()

        # 1. Query Compiler → ObservationContract
        images: list[dict[str, Any]] = []
        if file_path:
            images.append({"filename": file_name or "upload", "path": file_path, "modality": modality})
        if file2_path:
            images.append({"filename": file2_name or "upload2", "path": file2_path,
                           "modality": modality2 or modality})

        contract = compile_query(question, images)
        logger.info("sat_contract_compiled", phenomenon=contract.phenomenon, temporal=contract.temporal_requirement)

        # 2. Data Gate — check data sufficiency against contract
        geo_meta = probe_raster_metadata(file_path) if file_path else None
        gate_report = run_data_gate(contract, images, file_path, geo_meta)
        logger.info("sat_data_gate", overall=gate_report.overall)

        # 3. Observability — generate spatial observability mask
        observability = None
        if file_path:
            try:
                _REPORT_DIR.mkdir(parents=True, exist_ok=True)
                observability = compute_observability(file_path, output_dir=str(_REPORT_DIR))
            except Exception as exc:
                logger.warning("sat_observability_failed", error=str(exc))

        # 4. Analysis — run the inner pipeline (reuses AetherisPipeline)
        from pipeline.schemas.contracts import PipelineRequest
        inner_request = PipelineRequest(
            question=question, user_id=user_id,
            modality=modality, modality2=modality2,
        )
        inner_result = self._inner.execute(
            inner_request, file_data=file_data, file_name=file_name,
            file_path=file_path, file2_path=file2_path, file2_name=file2_name,
        )

        # Extract outputs for downstream SAT-OS stages
        planner_data = inner_result.modules.get("m01_baseline", None)
        # Get outputs from the planner module result
        outputs: list[dict[str, Any]] = []
        for mod_key, mod_result in inner_result.modules.items():
            if mod_result.data.get("outputs"):
                outputs = mod_result.data["outputs"]
                break

        # 5. Evidence Classification
        evidence_set = classify_evidence(outputs)
        logger.info("sat_evidence_classified", minimum_set=evidence_set.minimum_set)

        # 6. Falsification
        obs_pct = observability.observable_pct if observability else None
        falsification = run_falsification(outputs, obs_pct)
        logger.info("sat_falsification", any_contradiction=falsification.any_contradiction)

        # 7. Verification Decision
        verification = decide_verification(observability, gate_report, evidence_set, falsification)
        logger.info("sat_verification", status=verification.status)

        # 8. Certificate
        certificate = build_certificate(
            query_text=question, contract=contract, images=images,
            outputs=outputs, observability=observability,
            data_gate=gate_report, evidence_set=evidence_set,
            falsification=falsification, verification=verification,
        )

        # 9. Report
        report_paths = save_report(certificate, str(_REPORT_DIR))
        text_report = generate_text_report(certificate)

        # 9.5. JEV verification (zero-token consistency check)
        jev_verification: dict[str, Any] | None = None
        try:
            from interfaces.api.deps import get_jev_planner
            jev = get_jev_planner()
            if jev is not None:
                jev_verification = jev.verify_outputs(
                    [{"tool_name": o.get("tool_name", ""), "result": o} for o in outputs],
                    inner_result.final_answer or "",
                )
        except Exception:
            pass  # JEV is optional — never block the pipeline

        # 10. Run Record (immutable)
        from controller.model_registry import version_for_task
        run_record = RunRecord(
            run_id=certificate.run_id,
            config={"question": question, "modality": modality},
            input_hashes=certificate.input_hashes,
            model_versions=certificate.algorithms_used,
            output_hashes=certificate.output_hashes,
            certificate=certificate,
        )
        self._runs[run_record.run_id] = run_record

        elapsed = round(time.perf_counter() - t0, 3)

        return {
            "run_id": certificate.run_id,
            "verification_status": verification.status,
            "final_answer": inner_result.final_answer or "",
            "contract": contract.model_dump(),
            "data_gate": gate_report.model_dump(),
            "observability": observability.model_dump() if observability else None,
            "evidence_set": evidence_set.model_dump(),
            "falsification": falsification.model_dump(),
            "verification": verification.model_dump(),
            "jev_verification": jev_verification,
            "certificate": certificate.model_dump(mode="json"),
            "report_paths": report_paths,
            "inner_pipeline_status": inner_result.status,
            "execution_time_s": elapsed,
        }

    # ------------------------------------------------------------------
    # Run history & feedback
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        return [
            {"run_id": r.run_id, "timestamp": r.timestamp.isoformat(),
             "status": r.certificate.verification.status if r.certificate and r.certificate.verification else "unknown"}
            for r in self._runs.values()
        ]

    def submit_feedback(self, entry: FeedbackEntry) -> FeedbackEntry:
        return self._feedback.submit(entry)

    def get_feedback(self, run_id: str) -> list[FeedbackEntry]:
        return self._feedback.get_for_run(run_id)

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        return self._feedback.compare_runs(run_id_a, run_id_b)
