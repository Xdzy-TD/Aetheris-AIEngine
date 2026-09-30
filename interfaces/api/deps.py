"""
FastAPI dependency injection — initialises the controller singletons.

Provides FastAPI ``Depends`` callables that lazily create and cache
the ToolRegistry, RoutingMemory, AuditLog, OllamaClient, LLMPlanner,
JEVPlanner, and AgenticPlanner — all bundled into :class:`ControllerContext`
so every entry point (FastAPI, CLI, TUI) shares exactly one wiring path.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import structlog

from controller.audit_log import AuditLog
from controller.jev_planner import JEVPlanner
from controller.llm_planner import LLMPlanner
from controller.memory import RoutingMemory
from controller.model_registry import version_for_task
from controller.ollama_client import OllamaClient
from controller.planner import AgenticPlanner
from controller.tool_registry import ToolRegistry

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

_USE_LLM = os.environ.get("AETHERIS_NO_LLM", "").lower() not in ("1", "true", "yes")

# JEV (Judgement-Evaluation-Verification) engine — fast, typed, probabilistic
# decision engine for routing, scoring, verification, and guardrails.
# Enabled by default. Handles tool-chain planning with zero token cost in <1ms.
# When both JEV and LLM are enabled, JEV handles routing while LLM handles
# free-form synthesis — this saves ~1200 tokens per query.
_USE_JEV = os.environ.get("AETHERIS_USE_JEV", "true").lower() in ("1", "true", "yes")

# Self-consistency re-invokes vlm_* specialists this many times per call —
# a real latency/cost multiplier (see AgenticPlanner._SELF_CONSISTENCY_TOOLS)
# — so it's tunable rather than a hidden constant. 1 effectively disables it.
_SELF_CONSISTENCY_PATHS = int(os.environ.get("AETHERIS_SC_PATHS", "5"))


# ---------------------------------------------------------------------------
# Unified context — single source of truth for all entry points
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControllerContext:
    """Bundle of every controller-layer singleton.

    Replaces 6 separate ``@lru_cache`` singletons with one object so the CLI
    (``run.py``), the FastAPI app, and the TUI all share exactly one wiring
    path. Individual singletons are still accessible via the ``get_*()``
    delegates below for backward-compatible ``Depends()`` usage.
    """

    registry: ToolRegistry
    memory: RoutingMemory
    audit: AuditLog
    ollama_client: OllamaClient
    llm_planner: LLMPlanner | None
    jev_planner: JEVPlanner | None
    planner: AgenticPlanner


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_context() -> ControllerContext:
    """Create and return the singleton :class:`ControllerContext`.

    All sub-objects are created here in dependency order, so there is
    exactly one place that wires the controller layer together.

    When JEV is enabled (default), it handles tool-chain routing with zero
    token cost. The LLM planner, if also enabled, is used only for free-form
    synthesis — saving ~1200 tokens per query on planning alone.
    """
    registry = _build_registry()
    memory = RoutingMemory(max_entries=512)
    audit = AuditLog()
    ollama_client = OllamaClient()
    llm_planner = _build_llm_planner(ollama_client)
    jev_planner = _build_jev_planner(llm_planner)

    # Determine which planner to use:
    # - JEV enabled → JEV routes, LLM synthesises (if available)
    # - JEV disabled, LLM enabled → LLM routes + synthesises (original behaviour)
    # - Both disabled → deterministic fallback routing
    effective_llm_planner = llm_planner
    effective_use_llm = _USE_LLM and llm_planner is not None

    if jev_planner is not None:
        # JEV replaces LLM for routing; pass JEV as the "llm_planner" since
        # AgenticPlanner only calls plan() and synthesize() on it — JEV
        # implements both (delegating synthesize to the real LLM when available).
        effective_llm_planner = jev_planner  # type: ignore[assignment]
        effective_use_llm = True  # JEV is always "available"

    planner = AgenticPlanner(
        registry=registry,
        memory=memory,
        audit=audit,
        llm_planner=effective_llm_planner,  # type: ignore[arg-type]
        use_llm=effective_use_llm,
        n_self_consistency_paths=_SELF_CONSISTENCY_PATHS,
    )

    logger.info(
        "controller_context_ready",
        tools=registry.list_tool_names(),
        jev_enabled=jev_planner is not None,
        llm_enabled=llm_planner is not None,
        routing_engine="jev" if jev_planner else ("llm" if llm_planner else "rule_based"),
    )

    return ControllerContext(
        registry=registry,
        memory=memory,
        audit=audit,
        ollama_client=ollama_client,
        llm_planner=llm_planner,
        jev_planner=jev_planner,
        planner=planner,
    )


# ---------------------------------------------------------------------------
# Backward-compatible delegates (FastAPI Depends)
# ---------------------------------------------------------------------------

def get_registry() -> ToolRegistry:
    """Delegate for ``Depends(get_registry)``."""
    return get_context().registry


def get_memory() -> RoutingMemory:
    """Delegate for ``Depends(get_memory)``."""
    return get_context().memory


def get_audit() -> AuditLog:
    """Delegate for ``Depends(get_audit)``."""
    return get_context().audit


def get_ollama_client() -> OllamaClient:
    """Delegate for ``Depends(get_ollama_client)``."""
    return get_context().ollama_client


def get_llm_planner() -> LLMPlanner | None:
    """Delegate for ``Depends(get_llm_planner)``."""
    return get_context().llm_planner


def get_jev_planner() -> JEVPlanner | None:
    """Delegate for ``Depends(get_jev_planner)``."""
    return get_context().jev_planner


def get_planner() -> AgenticPlanner:
    """Delegate for ``Depends(get_planner)``."""
    return get_context().planner


# ---------------------------------------------------------------------------
# Internal builders (called only from get_context)
# ---------------------------------------------------------------------------

def _build_registry() -> ToolRegistry:
    """Create the ToolRegistry and bind all specialist implementations."""
    registry = ToolRegistry()
    _bind_specialists(registry)
    logger.info("registry_initialised", tools=registry.list_tool_names())
    return registry


def _build_llm_planner(client: OllamaClient) -> LLMPlanner | None:
    """Create the LLMPlanner if LLM mode is enabled."""
    if not _USE_LLM:
        logger.info("llm_planner_disabled", reason="AETHERIS_NO_LLM is set")
        return None

    planner = LLMPlanner(client)

    # First-run guidance: check if any model is actually available
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't await here; model resolution happens on first query
            logger.info("llm_planner_initialised", base_url=client.base_url, model=client.model)
        else:
            resolved = loop.run_until_complete(client.resolve_model())
            if not resolved:
                logger.warning(
                    "no_ollama_models_found",
                    hint="Run: ollama pull qwen2.5:7b",
                )
    except Exception:
        logger.info("llm_planner_initialised", base_url=client.base_url, model=client.model)

    return planner


def _build_jev_planner(llm_planner: LLMPlanner | None) -> JEVPlanner | None:
    """Create the JEVPlanner if JEV mode is enabled.

    JEV handles routing (zero tokens, <1ms). When an LLM planner is also
    available, JEV delegates synthesis to it — saving ~1200 planning tokens
    per query while keeping the LLM's natural-language answer quality.
    """
    if not _USE_JEV:
        logger.info("jev_planner_disabled", reason="AETHERIS_USE_JEV is not set")
        return None

    jev = JEVPlanner(llm_planner=llm_planner)
    logger.info(
        "jev_planner_initialised",
        llm_synthesis_available=llm_planner is not None,
    )
    return jev


def _bind_specialists(registry: ToolRegistry) -> None:
    """Bind specialist ``run()`` methods to their registry entries.

    ``demo_mode`` is not passed here: every specialist's constructor now
    accepts and ignores it (see e.g. ``specialists/vqa_caption/model.py``),
    so it stopped meaning anything — passing it just produced a "mode=demo"
    log line for tools that were never a demo/fallback. What each tool actually
    is (classical baseline vs. pretrained model, and which checkpoint) is
    reported per-binding below straight from ``models/registry.json`` via
    :func:`version_for_task` — the same source of truth every result's
    ``model_version`` field already cites, so this can't drift from it.
    """
    specialists: list[tuple[str, str, str]] = [
        ("vqa_caption", "specialists.vqa_caption.model", "VQACaptionSpecialist"),
        ("vlm_vqa_caption", "specialists.vlm_caption.model", "VLMCaptionSpecialist"),
        ("sar_bridge", "specialists.sar_bridge.model", "SARBridgeSpecialist"),
        ("grounding", "specialists.grounding.model", "GroundingSpecialist"),
        ("vlm_grounding", "specialists.vlm_grounding.model", "VLMGroundingSpecialist"),
        ("change_detection", "specialists.change_detection.model", "ChangeDetectionSpecialist"),
        (
            "vlm_change_detection",
            "specialists.vlm_change_detection.model",
            "VLMChangeDetectionSpecialist",
        ),
        ("geo_rag", "specialists.geo_rag.model", "GeoRAGSpecialist"),
        (
            "optical_sar_fusion",
            "specialists.optical_sar_fusion.model",
            "OpticalSARFusionSpecialist",
        ),
        (
            "fusion_change_detection",
            "specialists.fusion_change_detection.model",
            "FusionChangeDetectionSpecialist",
        ),
    ]

    for tool_name, module_path, class_name in specialists:
        if tool_name not in registry.list_tool_names():
            logger.debug("specialist_not_in_registry", tool=tool_name)
            continue

        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            instance = cls()
            registry.bind(tool_name, instance.run)
            logger.info("specialist_bound", tool=tool_name, model=version_for_task(tool_name))
        except Exception as exc:
            logger.warning(
                "specialist_bind_failed", tool=tool_name, error=str(exc)
            )
