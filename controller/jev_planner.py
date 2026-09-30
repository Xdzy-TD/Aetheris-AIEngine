"""
JEV Planner — zero-token System One routing.  Drop-in for LLMPlanner.

Planning:  ~1200 tokens → 0 tokens, 200-2000ms → <1ms.
Synthesis: delegates to LLM when available, else returns None for
           deterministic fallback.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from controller.jev_engine import EvidenceState, JEVEngine, QueryState
from controller.schemas import ExecutionPlan, Query, ToolCall

logger = structlog.get_logger(__name__)


class JEVPlanner:
    """Drop-in for ``LLMPlanner`` — same plan()/synthesize() contract."""

    def __init__(self, llm_planner: Any | None = None) -> None:
        self._engine = JEVEngine()
        self._llm = llm_planner
        self._plan_count = 0
        self._total_ms = 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "plan_count": self._plan_count,
            "avg_plan_ms": round(self._total_ms / max(self._plan_count, 1), 2),
            "llm_synthesis_available": self._llm is not None,
        }

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    async def plan(
        self, query: Query, tool_schemas: list[dict[str, Any]],
    ) -> ExecutionPlan | None:
        t0 = time.perf_counter()
        tools = [s["name"] for s in tool_schemas if s.get("name")]
        state = QueryState.from_query(query, tools)

        route = self._engine.decide_route(state)
        guard = self._engine.check_guard(state, route.answer)

        # Build steps with dependency chain
        steps, prior = [], []
        for tool_name in (route.answer if isinstance(route.answer, list) else [route.answer]):
            deps = list(prior) if tool_name in ("vqa_caption", "vlm_vqa_caption") else (
                ["sar_bridge"] if tool_name in ("change_detection", "vlm_change_detection", "fusion_change_detection")
                and "sar_bridge" in prior else []
            )
            args = {"question": query.text} if tool_name in ("vqa_caption", "vlm_vqa_caption") else {}
            steps.append(ToolCall(
                tool_name=tool_name, arguments=args,
                rationale=f"JEV routed (p={route.probability:.2f})",
                depends_on=deps,
            ))
            prior.append(tool_name)

        if not steps:
            steps.append(ToolCall(tool_name="vqa_caption", arguments={"question": query.text},
                                  rationale="JEV fallback: no tools above threshold"))

        plan = ExecutionPlan(
            query_id=query.query_id, steps=steps,
            strategy_note=f"JEV routing (p={route.probability:.2f}), guard={'PASS' if guard.answer else 'FAIL'}",
            planning_method="jev",
        )

        elapsed = (time.perf_counter() - t0) * 1000
        self._plan_count += 1
        self._total_ms += elapsed

        if not guard.answer:
            logger.warning("jev_guard_failed", query_id=query.query_id, reason=guard.reasoning)

        logger.info("jev_plan", query_id=query.query_id, tools=[s.tool_name for s in steps],
                     p=route.probability, ms=round(elapsed, 2))
        return plan

    # ------------------------------------------------------------------
    # Synthesis — delegates to LLM, JEV can't generate prose
    # ------------------------------------------------------------------

    async def synthesize(self, query: Query, specialist_outputs: list[dict[str, Any]]) -> str | None:
        if self._llm is not None:
            try:
                return await self._llm.synthesize(query, specialist_outputs)
            except Exception as exc:
                logger.warning("jev_llm_synthesis_fallback", error=str(exc))
        return None

    async def is_available(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Verification — new capability, still zero tokens
    # ------------------------------------------------------------------

    def verify_outputs(self, outputs: list[dict[str, Any]], final_answer: str) -> dict[str, Any]:
        """Verify consistency + score evidence quality.  Zero tokens."""
        consistency = self._engine.verify_consistency(outputs, final_answer)
        evidence = [
            EvidenceState(
                tool_name=o.get("tool_name", ""),
                confidence=o.get("result", {}).get("confidence"),
                has_artifact=bool(o.get("result", {}).get("artifact") or o.get("artifacts")),
                analysis_unavailable=o.get("result", {}).get("analysis_unavailable", False),
                coregistration_reliable=o.get("result", {}).get("coregistration_reliable"),
                warning_count=len(o.get("warnings", [])),
                result_keys=list(o.get("result", {}).keys()),
            )
            for o in outputs
        ]
        score = self._engine.score_evidence(evidence)
        return {
            "consistent": consistency.answer,
            "consistency_probability": consistency.probability,
            "evidence_quality": score.answer,
            "evidence_probability": score.probability,
            "total_elapsed_ms": consistency.elapsed_ms + score.elapsed_ms,
        }
