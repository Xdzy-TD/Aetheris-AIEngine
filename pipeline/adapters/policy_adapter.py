"""
Policy Adapter — wraps ``controller.policy`` without modifying it.

Converts new pipeline requests into old AETHERIS schemas, invokes the
existing ``check_plan()`` logic, and translates the outcome into M08 contracts.
"""

from __future__ import annotations

import time
from typing import Any

from controller.policy import PolicyViolation, check_plan
from controller.schemas import ExecutionPlan, ImageMetadata, Modality, Query, ToolCall
from controller.tool_registry import ToolRegistry

from pipeline.schemas.contracts import (
    ModuleID,
    ModuleResult,
    wrap_policy_error,
)


class PolicyAdapter:
    """Adapter around the existing AETHERIS policy engine."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        # Bug: this used to build its own bare ToolRegistry() with nothing
        # bound, so tool_binding_check failed on every single request
        # ("no model bound") even though the real planner (PlannerAdapter,
        # via interfaces.api.deps.get_registry()) runs against a properly
        # bound registry and succeeds. Reuse that same singleton instead of
        # a second, permanently-unbound one.
        if self._registry is None:
            from interfaces.api.deps import get_registry
            self._registry = get_registry()
        return self._registry

    def validate_plan(
        self,
        plan: ExecutionPlan,
        query: Query,
    ) -> ModuleResult:
        """Run the existing policy engine and return an M08 result."""
        t0 = time.perf_counter()
        try:
            check_plan(plan, query, self.registry)
            return ModuleResult.success(
                ModuleID.M02,
                {"policy_check": "passed", "steps_validated": len(plan.steps)},
                exec_time=round(time.perf_counter() - t0, 4),
            )
        except PolicyViolation as exc:
            result = ModuleResult.failure(
                ModuleID.M02,
                "POLICY_VIOLATION",
                str(exc),
            )
            result.execution_time_s = round(time.perf_counter() - t0, 4)
            return result

    def validate_request(
        self,
        question: str,
        images: list[dict[str, Any]] | None = None,
        modality: str = "optical",
    ) -> ModuleResult:
        """Higher-level validation that builds old schemas from new request fields."""
        t0 = time.perf_counter()

        # Build old-style image metadata
        image_metas = []
        if images:
            for img in images:
                image_metas.append(ImageMetadata(
                    filename=img.get("filename", "upload"),
                    path=img.get("path"),
                    modality=Modality(img.get("modality", modality)),
                ))

        query = Query(text=question, images=image_metas)

        # Build a minimal plan for policy checking
        steps = [ToolCall(
            tool_name="vqa_caption",
            arguments={"question": question},
            rationale="Default VQA step",
        )]
        plan = ExecutionPlan(query_id=query.query_id, steps=steps)

        try:
            check_plan(plan, query, self.registry)
            return ModuleResult.success(
                ModuleID.M02,
                {
                    "policy_check": "passed",
                    "query_id": query.query_id,
                    "tools_available": self.registry.list_tool_names(),
                },
                exec_time=round(time.perf_counter() - t0, 4),
            )
        except PolicyViolation as exc:
            result = ModuleResult.failure(ModuleID.M02, "POLICY_VIOLATION", str(exc))
            result.execution_time_s = round(time.perf_counter() - t0, 4)
            return result

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas from the existing registry."""
        return self.registry.list_tools()
