"""
Pipeline — the full M01–M10 pipeline orchestrator.

Composes all modules into the execution flow described in the spec:
  M04 Auth → M05 Rate → M07 Session → M06 Upload → M09 CRS →
  M10 Radiometric → M02 Policy → M03 DAG → Old Planner → M08 Result
"""

from __future__ import annotations

import time
from typing import Any

from pipeline.adapters.audit_adapter import AuditAdapter
from pipeline.adapters.planner_adapter import PlannerAdapter
from pipeline.modules.m01_baseline import run_health_check
from pipeline.modules.m02_policy import PolicyEngine
from pipeline.modules.m03_dag import DAGNode, ExecutionDAG
from pipeline.modules.m04_auth import AuthManager
from pipeline.modules.m05_rate_limit import ConcurrencyManager, RateLimiter
from pipeline.modules.m06_upload_security import validate_upload
from pipeline.modules.m07_session_isolation import SessionManager
from pipeline.modules.m08_contracts import ContractManager
from pipeline.modules.m09_geospatial import validate_crs
from pipeline.modules.m10_radiometric import validate_radiometric
from pipeline.modules.m11_change_preprocessing import validate_change_pair
from pipeline.modules.m12_geotiff_artifacts import preserve_geotiff
from pipeline.modules.m14_provenance import attach_provenance
from pipeline.modules.m19_evidence_graph import build_evidence_graph
from pipeline.modules.m20_counterfactual import run_counterfactual_check
from pipeline.schemas.contracts import (
    ModuleID,
    ModuleResult,
    PipelineRequest,
    PipelineResult,
    ResultStatus,
)


class AetherisPipeline:
    """Full M01–M10 pipeline orchestrator."""

    def __init__(self) -> None:
        self.auth = AuthManager()
        self.rate_limiter = RateLimiter()
        self.concurrency = ConcurrencyManager()
        self.session_mgr = SessionManager()
        self.policy = PolicyEngine()
        self.planner = PlannerAdapter()
        self.audit = AuditAdapter()
        self.contracts = ContractManager()

    def execute(
        self,
        request: PipelineRequest,
        file_data: bytes | None = None,
        file_name: str = "",
        file_path: str | None = None,
        file2_path: str | None = None,
        file2_name: str = "",
    ) -> PipelineResult:
        """Execute the full pipeline for a request."""
        t0 = time.perf_counter()
        result = PipelineResult(pipeline_id=request.request_id)

        # Log pipeline start
        self.audit.log_pipeline_start(
            request.request_id,
            {"question": request.question, "user_id": request.user_id},
        )

        # M04: Authentication & Authorization
        # NOTE: previously this always called authenticate(user_id, "") — an
        # empty password can never match a real user's hash, so every
        # request silently fell through to an anonymous identity no matter
        # what user_id was passed in, and named users/roles were dead code
        # in the actual pipeline. Anonymous access (the common case for this
        # demo) still works the same as before; a caller that wants a real
        # role now has to actually supply the password for it.
        auth_result = self.auth.authenticate(request.user_id, request.password or "")
        result.add_module_result(auth_result)
        if auth_result.status == ResultStatus.FAILURE:
            return self._finalize(result, t0)

        session_token = auth_result.data.get("session_token", "")
        authz_result = self.auth.authorize(session_token, "query")
        if authz_result.status == ResultStatus.FAILURE:
            result.add_module_result(authz_result)
            return self._finalize(result, t0)

        # M05: Rate Limiting
        rate_result = self.rate_limiter.check(request.user_id)
        result.add_module_result(rate_result)
        if rate_result.status == ResultStatus.FAILURE:
            return self._finalize(result, t0)

        # M05: Concurrency
        job_result = self.concurrency.submit_job(request.user_id)
        job_id = job_result.data.get("job_id", "")

        # Everything from here on holds a concurrency slot (job_id is
        # RUNNING/QUEUED). Previously each failure branch had to remember to
        # call complete_job() itself, and nothing covered an unhandled
        # exception (e.g. the planner call below raising) — that job just
        # stayed RUNNING forever, permanently eating one of max_concurrent
        # slots. try/finally guarantees exactly one completion call no
        # matter how this block exits.
        try:
            # M07: Session Isolation
            session_result = self.session_mgr.create_session(request.user_id)
            result.add_module_result(session_result)
            session_id = session_result.data.get("session_id", request.session_id)

            # M06: Upload Security (if file provided)
            if file_data and file_name:
                upload_result = validate_upload(file_data, file_name)
                result.add_module_result(upload_result)
                if upload_result.status == ResultStatus.FAILURE:
                    return self._finalize(result, t0)

            # M09: CRS Validation (if file path provided)
            if file_path:
                crs_result = validate_crs(file_path)
                result.add_module_result(crs_result)
                # Bug: M09/M10 results were recorded but never checked, so a
                # raster that failed to read (CRS_PROBE_ERROR) still went on
                # to the planner. validate_crs() only fails on a genuine read
                # error, not on "no CRS" (a plain optical image is legitimate
                # input), so gating here can't reject valid uploads.
                if crs_result.status == ResultStatus.FAILURE:
                    return self._finalize(result, t0)

                # M10: Radiometric Validation — the actual scientific gate:
                # rejects excessive NoData/NaN/Inf or an image with no valid
                # pixels *before* any specialist runs on it, instead of only
                # reporting the problem after an analysis was already produced
                # from unusable data.
                radio_result = validate_radiometric(file_path)
                result.add_module_result(radio_result)
                if radio_result.status == ResultStatus.FAILURE:
                    return self._finalize(result, t0)
            else:
                result.add_module_result(
                    ModuleResult.skipped(ModuleID.M09, "No file path for CRS validation")
                )
                result.add_module_result(
                    ModuleResult.skipped(ModuleID.M10, "No file path for radiometric validation")
                )

            # M11: Change-detection pair validation (only when a second file
            # is present). Guards the pair before it reaches the old
            # change_detection specialist — see
            # modules/m11_change_preprocessing.py docstring. A failing pair
            # is not passed on to the planner; the query falls back to
            # single-image analysis of file_path.
            image_b_path = file2_path
            if file_path and file2_path:
                m11_result = validate_change_pair(file_path, file2_path)
                result.add_module_result(m11_result)
                if m11_result.status == ResultStatus.FAILURE:
                    image_b_path = None
                elif m11_result.data.get("coregistered_path"):
                    image_b_path = m11_result.data["coregistered_path"]
            else:
                result.add_module_result(
                    ModuleResult.skipped(ModuleID.M11, "Fewer than two files for change-pair validation")
                )

            # Images for both M02 (policy needs to know what was actually
            # uploaded) and M03 below — built once, shared by both. Bug was:
            # M02 was called with no images at all, so it always validated a
            # plan against 0 images and failed every request with "needs 1
            # readable image(s); 0 were provided", regardless of the real
            # upload.
            images = []
            if file_path:
                images.append({
                    "filename": file_name or "upload",
                    "path": file_path,
                    "modality": request.modality,
                })
            if image_b_path:
                images.append({
                    "filename": file2_name or "upload2",
                    "path": image_b_path,
                    "modality": request.modality2 or request.modality,
                })

            # M02: Policy Validation
            policy_result = self.policy.validate_request(
                request.question,
                images=images if images else None,
                modality=request.modality,
                file_sizes=[len(file_data)] if file_data else None,
            )
            result.add_module_result(policy_result)

            planner_result = self.planner.execute_query_sync(
                request.question,
                images=images if images else None,
                session_id=session_id,
                image_bytes=file_data,
            )
            result.add_module_result(planner_result)

            # Extract final answer from planner
            if planner_result.status == ResultStatus.SUCCESS:
                result.final_answer = planner_result.data.get("final_answer", "")

            # M12: GeoTIFF artifact preservation — see
            # modules/m12_geotiff_artifacts.py. Only meaningful when the
            # source image itself carried a CRS.
            outputs = planner_result.data.get("outputs", [])
            geo_artifacts = []
            if file_path:
                for out in outputs:
                    rec = out.get("result", {}).get("artifact")
                    if rec and rec.get("path"):
                        geo_result = preserve_geotiff(rec["path"], file_path, rec.get("type", "artifact"))
                        result.add_module_result(geo_result)
                        if geo_result.data.get("geotiff_artifact"):
                            geo_artifacts.append(geo_result.data["geotiff_artifact"])
            else:
                result.add_module_result(
                    ModuleResult.skipped(ModuleID.M12, "No source file to georeference from")
                )

            # M14: Audit/provenance — see modules/m14_provenance.py.
            result.add_module_result(
                attach_provenance(result, self.audit, request.request_id, outputs, geo_artifacts)
            )

            # M19: Evidence graph — connects the answer to the pixels, models,
            # parameters and artifacts behind it. See modules/m19_evidence_graph.py.
            plan_steps = planner_result.data.get("plan", {}).get("steps", [])
            result.add_module_result(
                build_evidence_graph(
                    result.final_answer, outputs, plan_steps, images, self.planner.registry
                )
            )

            # M20: Counterfactual self-check — attempts to falsify the answer
            # via an independent route or resampling agreement. See
            # modules/m20_counterfactual.py.
            result.add_module_result(
                run_counterfactual_check(outputs, plan_steps, images, self.planner.registry)
            )
        except Exception as exc:  # noqa: BLE001 - last-resort guard, see try comment above
            result.add_module_result(
                ModuleResult.failure(ModuleID.M03, "PIPELINE_EXCEPTION", str(exc))
            )
        finally:
            error = "Pipeline execution failed" if result.status == ResultStatus.FAILURE else None
            self.concurrency.complete_job(job_id, error)

        # Log pipeline completion
        self.audit.log_pipeline_complete(
            request.request_id,
            {"status": result.status.value, "final_answer": result.final_answer},
        )

        return self._finalize(result, t0)

    def health_check(self) -> ModuleResult:
        """Run M01 health check."""
        return run_health_check()

    def get_pipeline_status(self) -> dict[str, Any]:
        """Get the overall pipeline status."""
        return {
            "auth": self.auth.get_status(),
            "rate_limiter": self.rate_limiter.get_status(),
            "concurrency": self.concurrency.get_status(),
            "sessions": self.session_mgr.get_status(),
            "policy": self.policy.get_policy_summary(),
            "planner": self.planner.get_planner_status(),
            "contracts": self.contracts.get_contract_summary(),
        }

    def _finalize(self, result: PipelineResult, t0: float) -> PipelineResult:
        result.compute_status()
        result.total_execution_time_s = round(time.perf_counter() - t0, 4)
        return result
