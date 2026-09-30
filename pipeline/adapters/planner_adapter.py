"""
Planner Adapter — wraps ``controller.planner.AgenticPlanner`` without editing it.

Provides a DAG-invokable interface to the old planner, translating its
``ExecutionResult`` into M08 contracts.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pipeline.modules.m13_conformal import attach_saved_calibration
from pipeline.schemas.contracts import (
    ModuleID,
    ModuleResult,
    wrap_execution_result,
)


class PlannerAdapter:
    """Adapter around the existing AgenticPlanner."""

    def __init__(self) -> None:
        self._planner = None
        self._registry = None

    def _ensure_planner(self) -> Any:
        if self._planner is None:
            try:
                from interfaces.api.deps import get_planner, get_registry
                self._planner = get_planner()
                self._registry = get_registry()
            except Exception as exc:
                # Bug (same class as the one fixed in policy_adapter.py /
                # m01_baseline.py): this silently swallowed *any* exception
                # from the real singleton and fell back to a bare
                # ToolRegistry() with nothing bound, so every tool showed
                # "bound": false even when the real registry was fine. Now:
                # log the real cause instead of hiding it, and bind the
                # fallback registry the same way the singleton does so it
                # isn't permanently unbound either.
                import structlog

                structlog.get_logger(__name__).warning(
                    "planner_singleton_unavailable_falling_back", error=str(exc)
                )
                from controller.memory import RoutingMemory
                from controller.planner import AgenticPlanner
                from controller.tool_registry import ToolRegistry
                from interfaces.api.deps import _bind_specialists, get_audit

                self._registry = ToolRegistry()
                _bind_specialists(self._registry)
                self._planner = AgenticPlanner(
                    registry=self._registry,
                    memory=RoutingMemory(),
                    audit=get_audit(),
                    use_llm=False,
                )
            # M13: load a previously-fitted calibration artifact, if one
            # exists — see modules/m13_conformal.py.
            attach_saved_calibration(self._planner)
        return self._planner

    @property
    def registry(self) -> Any:
        self._ensure_planner()
        return self._registry

    async def execute_query(
        self,
        question: str,
        images: list[dict[str, Any]] | None = None,
        session_id: str = "",
        image_bytes: bytes | None = None,
    ) -> ModuleResult:
        """Execute a query through the old planner and return an M08 result."""
        from controller.schemas import ImageMetadata, Modality, Query

        t0 = time.perf_counter()
        try:
            planner = self._ensure_planner()

            image_metas = []
            if images:
                for img in images:
                    image_metas.append(ImageMetadata(
                        filename=img.get("filename", "upload"),
                        path=img.get("path"),
                        modality=Modality(img.get("modality", "optical")),
                    ))

            query = Query(
                text=question,
                images=image_metas,
                session_id=session_id or "default",
            )

            result = await planner.process_query(query, image_bytes)
            wrapped = wrap_execution_result(result)

            return ModuleResult.success(
                ModuleID.M03,
                wrapped,
                exec_time=round(time.perf_counter() - t0, 4),
            )
        except Exception as exc:
            return ModuleResult.failure(
                ModuleID.M03,
                "PLANNER_ERROR",
                str(exc),
            )

    def execute_query_sync(
        self,
        question: str,
        images: list[dict[str, Any]] | None = None,
        session_id: str = "",
        image_bytes: bytes | None = None,
    ) -> ModuleResult:
        """Synchronous wrapper around execute_query."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.execute_query(question, images, session_id, image_bytes),
                    )
                    return future.result(timeout=120)
            else:
                return loop.run_until_complete(
                    self.execute_query(question, images, session_id, image_bytes)
                )
        except RuntimeError:
            return asyncio.run(
                self.execute_query(question, images, session_id, image_bytes)
            )

    def get_planner_status(self) -> dict[str, Any]:
        """Return the status of the old planner."""
        try:
            planner = self._ensure_planner()
            # Real active-model name (was missing: the GUI's model badge had
            # no way to show anything but "LLM Active" vs "Demo Mode", not
            # which model). Reuses the same OllamaClient the planner already
            # holds rather than tracking a second copy of the model name.
            if planner.use_llm and planner.llm_planner is not None:
                # Check if the planner is JEV-backed
                from controller.jev_planner import JEVPlanner
                if isinstance(planner.llm_planner, JEVPlanner):
                    model_name = "JEV decision engine (zero-token)"
                    routing_engine = "jev"
                    jev_stats = planner.llm_planner.stats
                elif hasattr(planner.llm_planner, "client"):
                    model_name = planner.llm_planner.client.model
                    routing_engine = "llm"
                    jev_stats = None
                else:
                    model_name = "unknown planner"
                    routing_engine = "unknown"
                    jev_stats = None
            else:
                model_name = "classical baseline (no LLM)"
                routing_engine = "rule_based"
                jev_stats = None
            status = {
                "available": True,
                "use_llm": planner.use_llm,
                "model_name": model_name,
                "routing_engine": routing_engine,
                "tools": self._registry.list_tool_names() if self._registry else [],
                "tool_count": len(self._registry.list_tool_names()) if self._registry else 0,
            }
            if jev_stats is not None:
                status["jev_stats"] = jev_stats
            return status
        except Exception as exc:
            return {"available": False, "error": str(exc)}
