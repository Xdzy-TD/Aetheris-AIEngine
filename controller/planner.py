"""
Agentic Planner — dynamic tool-chain composition.

The planner inspects a query + input metadata and dynamically composes an
ordered sequence of specialist tool calls. This mirrors the function-calling /
MCP paradigm: the controller genuinely decides which tools to invoke and in
what order, rather than running a fixed pipeline.

Uses LLM-driven planning (:class:`LLMPlanner`) when an Ollama model is
reachable, falling back to the deterministic fallback planner otherwise.

Execution, image-arg resolution, caching, and voting are delegated to
:class:`controller.executor.PlanExecutor`.  Result-contract metadata
(model_version, warnings, confidence) is delegated to
:class:`controller.result_contract.ResultContract`.
"""

from __future__ import annotations

import time
from typing import Any, cast

import structlog

from confidence.conformal import ConformalCalibrator
from confidence.self_consistency import SelfConsistencyVoter
from controller.audit_log import AuditLog
from controller.executor import PlanExecutor
from controller.llm_planner import LLMPlanner
from controller.memory import RoutingMemory
from controller.policy import PolicyViolation, check_plan
from controller.result_contract import ResultContract
from controller.schemas import (
    ClaimEvidenceLink,
    ExecutionPlan,
    ExecutionResult,
    Modality,
    ProvenanceChain,
    Query,
    SpecialistOutput,
    ToolCall,
)
from controller.tool_registry import ToolRegistry

logger = structlog.get_logger(__name__)


class AgenticPlanner:
    """Dynamic tool-chain planner for the Aetheris controller.

    Composes an :class:`ExecutionPlan` for each incoming :class:`Query`,
    delegates execution to :class:`PlanExecutor`, and records every action
    in the :class:`AuditLog`.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        memory: RoutingMemory,
        audit: AuditLog,
        llm_planner: LLMPlanner | None = None,
        use_llm: bool = True,
        n_self_consistency_paths: int = 5,
    ) -> None:
        self.registry = registry
        self.memory = memory
        self.audit = audit
        self.llm_planner = llm_planner
        self.use_llm = use_llm and llm_planner is not None

        self._contract = ResultContract(calibrator=ConformalCalibrator())
        self._executor = PlanExecutor(
            registry=registry,
            memory=memory,
            audit=audit,
            contract=self._contract,
            voter=SelfConsistencyVoter(n_paths=n_self_consistency_paths),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process_query(
        self,
        query: Query,
        image_bytes: bytes | None = None,
        plan: ExecutionPlan | None = None,
    ) -> ExecutionResult:
        """Plan and execute a tool chain for the given query.

        Args:
            query:       Parsed user query with metadata.
            image_bytes: Raw bytes of the uploaded image (used for
                         routing-memory keying and passed to specialists).
            plan:        Pre-built plan to execute as-is instead of calling
                         :meth:`_plan` (e.g. a mission preset from
                         ``controller/missions.py``). Still goes through the
                         same policy gate, execution and audit trail below.

        Returns:
            A complete :class:`ExecutionResult` with all intermediate outputs,
            the final answer, and a reference to the audit trail.
        """
        t0 = time.perf_counter()

        # 1. Plan
        plan = plan or await self._plan(query)
        self.audit.log_event(
            query_id=query.query_id,
            event_type="plan_created",
            input_data=query.model_dump(mode="json"),
            output_data=plan.model_dump(mode="json"),
        )
        logger.info(
            "plan_created",
            query_id=query.query_id,
            steps=[s.tool_name for s in plan.steps],
        )

        # 2. Policy gate: the plan is untrusted input (an LLM wrote it), so it
        # is validated *before* anything executes. A rejected plan produces one
        # clear refusal rather than a chain of specialists improvising.
        try:
            check_plan(plan, query, self.registry)
        except PolicyViolation as violation:
            return self._refuse(query, plan, str(violation), time.perf_counter() - t0)

        # 3. Execute via PlanExecutor
        outputs = await self._executor.execute(query, plan, image_bytes)

        # 4. Synthesize final answer
        final_answer = await self._synthesize(query, outputs)

        # 5. Build provenance chain
        audit_entry = self.audit.log_event(
            query_id=query.query_id,
            event_type="answer_emitted",
            input_data=[o.model_dump(mode="json") for o in outputs],
            output_data={"answer": final_answer},
        )
        provenance = self._build_provenance(query.query_id, outputs, audit_entry.chain_hash)

        elapsed = time.perf_counter() - t0
        result = ExecutionResult(
            query_id=query.query_id,
            plan=plan,
            outputs=outputs,
            final_answer=final_answer,
            confidence=self._contract.estimate_confidence(outputs),
            execution_time_s=round(elapsed, 3),
            warnings=ResultContract.dedup_warnings(outputs),
            provenance=provenance,
            change_narrative=self._synthesize_change_narrative(outputs),
        )

        return result

    def _refuse(
        self, query: Query, plan: ExecutionPlan, reason: str, elapsed: float
    ) -> ExecutionResult:
        """Return a policy refusal — never a fabricated analysis.

        Confidence is ``None``: nothing was measured, so there is no score to
        report. A number here would be indistinguishable from a real one.
        """
        logger.warning("plan_rejected", query_id=query.query_id, reason=reason)
        self.audit.log_event(
            query_id=query.query_id,
            event_type="plan_rejected",
            input_data=plan.model_dump(mode="json"),
            output_data={"reason": reason},
        )
        return ExecutionResult(
            query_id=query.query_id,
            plan=plan,
            outputs=[],
            final_answer=f"Analysis unavailable — {reason}.",
            confidence=None,
            execution_time_s=round(elapsed, 3),
            warnings=[f"plan rejected: {reason}"],
        )

    # ------------------------------------------------------------------
    # Planning (deterministic fallback when LLM is unavailable)
    # ------------------------------------------------------------------

    async def _plan(self, query: Query) -> ExecutionPlan:
        """Compose a tool-chain plan for the query.

        If ``use_llm`` is True and Ollama is available, delegates to the
        :class:`LLMPlanner` for dynamic, LLM-driven planning.  Falls back
        to deterministic fallback routing on any failure.
        """
        # Try LLM-driven planning first
        if self.use_llm and self.llm_planner is not None:
            tool_schemas = self.registry.get_schemas_for_planner()
            llm_plan = await self.llm_planner.plan(query, tool_schemas)
            if llm_plan is not None:
                # Validate that all planned tools actually exist in the registry
                valid_tools = set(self.registry.list_tool_names())
                validated_steps = [
                    s for s in llm_plan.steps if s.tool_name in valid_tools
                ]
                if validated_steps:
                    llm_plan.steps = validated_steps
                    return llm_plan
                else:
                    logger.warning(
                        "llm_plan_no_valid_tools",
                        planned=[s.tool_name for s in llm_plan.steps],
                        available=list(valid_tools),
                    )

        # Fallback: deterministic fallback routing
        return self._plan_rule_based(query)

    def _plan_rule_based(self, query: Query) -> ExecutionPlan:
        """Deterministic rule-based routing (fallback when LLM is unavailable).

        Always routes to ``vqa_caption``.  Two images of *different*
        modalities (one optical/multispectral, one SAR) are a cross-modal
        pair and route to ``optical_sar_fusion``; two images of the *same*
        modality are a bi-temporal pair and route to ``change_detection``.
        A lone SAR image gets ``sar_bridge`` despeckling first.
        """
        steps: list[ToolCall] = []

        has_sar = any(img.modality == Modality.SAR for img in query.images)
        has_optical_like = any(
            img.modality in (Modality.OPTICAL, Modality.MULTISPECTRAL) for img in query.images
        )
        is_cross_modal_pair = (
            len(query.images) == 2
            and has_sar
            and has_optical_like
            and "optical_sar_fusion" in self.registry.list_tool_names()
        )

        if is_cross_modal_pair:
            steps.append(ToolCall(
                tool_name="optical_sar_fusion",
                arguments={},
                rationale=(
                    "One optical/multispectral and one SAR image provided — "
                    "extracting complementary information via cross-modal agreement"
                ),
            ))
        elif has_sar and "sar_bridge" in self.registry.list_tool_names():
            steps.append(ToolCall(
                tool_name="sar_bridge",
                arguments={},
                rationale="SAR input detected — applying speckle despeckling before VQA",
            ))

        # Same-modality multi-image → bi-temporal change detection
        if (
            not is_cross_modal_pair
            and len(query.images) >= 2
            and "change_detection" in self.registry.list_tool_names()
        ):
            steps.append(ToolCall(
                tool_name="change_detection",
                arguments={},
                rationale="Two images provided — running bi-temporal change detection",
                depends_on=["sar_bridge"] if has_sar else [],
            ))

        # Always route to VQA as the final step
        steps.append(ToolCall(
            tool_name="vqa_caption",
            arguments={"question": query.text},
            rationale="Core VQA/caption step — answers the user's question",
            depends_on=[s.tool_name for s in steps],
        ))

        strategy = (
            "Rule-based routing: optical_sar_fusion for cross-modal pairs, "
            "sar_bridge for a lone SAR image, change_detection for same-modality "
            "pairs, then VQA."
        )
        return ExecutionPlan(
            query_id=query.query_id,
            steps=steps,
            strategy_note=strategy,
            planning_method="rule_based",
        )

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(self, query: Query, outputs: list[SpecialistOutput]) -> str:
        """Combine specialist outputs into a final answer.

        If the LLM planner is available, sends all specialist outputs to the
        LLM for natural-language synthesis.  Falls back to extracting the
        VQA specialist's answer directly.
        """
        # Try LLM synthesis for structured report output
        if self.use_llm and self.llm_planner is not None and outputs:
            output_dicts = [o.model_dump(mode="json") for o in outputs]
            llm_answer = await self.llm_planner.synthesize(query, output_dicts)
            if llm_answer:
                return llm_answer

        # Fallback: return VQA specialist's answer directly
        for out in reversed(outputs):
            if out.tool_name == "vqa_caption":
                return cast(str, out.result.get("answer", "No answer generated."))
        return "The controller was unable to produce an answer for this query."

    # ------------------------------------------------------------------
    # Change narrative (semantic change understanding)
    # ------------------------------------------------------------------

    @staticmethod
    def _synthesize_change_narrative(
        outputs: list[SpecialistOutput],
    ) -> str | None:
        """Compose a natural-language description of *what* changed.

        Combines structured change-detection results (change_fraction,
        change_summary, etc.) with VQA/caption context and grounding
        outputs to produce a narrative like "Vegetation in the AOI
        decreased by ~40%, replaced by built-up area".  Returns None if
        no change tool ran.

        This is not a trained Change-VQA model — it synthesises prose from
        structured tool outputs. See STATUS.md for the honest distinction.
        """
        change_outputs = [
            out for out in outputs
            if out.tool_name in (
                "change_detection", "vlm_change_detection", "fusion_change_detection",
            )
            and not out.result.get("analysis_unavailable")
        ]
        if not change_outputs:
            return None

        # Collect grounding labels for semantic cross-referencing
        grounding_labels: list[str] = []
        for out in outputs:
            if out.tool_name in ("grounding", "vlm_grounding") and not out.result.get("analysis_unavailable"):
                label = out.result.get("class_name") or out.result.get("label") or out.result.get("keyword")
                if label:
                    grounding_labels.append(str(label).lower())

        vqa_answer = None
        for out in reversed(outputs):
            if out.tool_name in ("vqa_caption", "vlm_vqa_caption") and not out.result.get("analysis_unavailable"):
                vqa_answer = out.result.get("answer")
                break

        parts: list[str] = []
        for co in change_outputs:
            r = co.result
            tool_label = co.tool_name.replace("_", " ").title()

            # Change fraction with semantic label cross-reference
            frac = r.get("change_fraction") or r.get("change_percent")
            if isinstance(frac, (int, float)):
                pct = frac * 100 if frac <= 1.0 else frac
                if grounding_labels:
                    labels_str = ", ".join(grounding_labels)
                    parts.append(
                        f"{tool_label} detected change in ~{pct:.0f}% of the area "
                        f"(land-cover classes identified: {labels_str})."
                    )
                else:
                    parts.append(f"{tool_label} detected change in ~{pct:.0f}% of the area.")
            elif r.get("changed") is True:
                parts.append(f"{tool_label} detected change.")
            elif r.get("changed") is False:
                parts.append(f"{tool_label} detected no significant change.")

            # Change summary from the specialist
            summary = r.get("change_summary") or r.get("summary")
            if summary:
                parts.append(str(summary))

            # Fusion per-channel evidence scores (optical, SAR, temporal)
            if co.tool_name == "fusion_change_detection":
                scores = []
                for ch in ("optical_score", "sar_score", "temporal_score"):
                    val = r.get(ch)
                    if isinstance(val, (int, float)):
                        scores.append(f"{ch.replace('_', ' ')}: {val:.2f}")
                if scores:
                    parts.append(f"Per-channel evidence: {', '.join(scores)}.")

            # Reliability note
            if r.get("coregistration_reliable") is False:
                parts.append(
                    "Note: co-registration was unreliable; the change estimate "
                    "may be affected by misalignment."
                )
            # Per-channel reliability for fusion
            coreg_detail = r.get("coregistration_reliable")
            if isinstance(coreg_detail, dict):
                unreliable = [k for k, v in coreg_detail.items() if v is False]
                if unreliable:
                    parts.append(
                        f"Note: co-registration was unreliable for {', '.join(unreliable)} channel(s)."
                    )

        # Add VQA context if available
        if vqa_answer and parts:
            parts.append(f"Scene context: {vqa_answer}")

        return " ".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    @staticmethod
    def _build_provenance(
        query_id: str,
        outputs: list[SpecialistOutput],
        audit_chain_hash: str,
    ) -> ProvenanceChain | None:
        """Collect every evidence artifact and identify the weakest link."""
        if not outputs:
            return None

        from controller.model_registry import version_for_task

        all_evidence = [ea for out in outputs for ea in out.evidence_artifacts]
        all_spatial = [se for out in outputs for se in out.spatial_evidence]
        model_versions = sorted({
            version_for_task(out.tool_name) for out in outputs
            if not out.result.get("analysis_unavailable")
        })

        # Weakest link: step with lowest confidence
        scored = [
            (out.tool_name, out.result.get("confidence") or out.result.get("raw_confidence"))
            for out in outputs
            if not out.result.get("analysis_unavailable")
            and isinstance(out.result.get("confidence", out.result.get("raw_confidence")), (int, float))
        ]
        weakest_name, weakest_conf = min(scored, key=lambda t: t[1]) if scored else (None, None)

        # Claim → Evidence → Artifact links
        claim_links: list[ClaimEvidenceLink] = []
        for out in outputs:
            if out.result.get("analysis_unavailable"):
                continue
            # Extract the primary claim from each specialist
            claim = (
                out.result.get("answer")
                or out.result.get("change_summary")
                or out.result.get("summary")
            )
            if not claim:
                continue
            artifact_id = (
                out.evidence_artifacts[0].artifact_id if out.evidence_artifacts else None
            )
            claim_links.append(ClaimEvidenceLink(
                claim=str(claim),
                tool_name=out.tool_name,
                evidence_artifact_id=artifact_id,
                confidence=out.result.get("confidence") or out.result.get("raw_confidence"),
            ))

        return ProvenanceChain(
            query_id=query_id,
            audit_chain_hash=audit_chain_hash,
            evidence_artifacts=all_evidence,
            spatial_evidence=all_spatial,
            weakest_link=weakest_name,
            weakest_confidence=weakest_conf,
            model_versions=model_versions,
            claim_evidence_links=claim_links,
        )
