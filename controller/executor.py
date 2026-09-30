"""
Plan Executor — runs an ExecutionPlan step-by-step.

Extracted from ``controller/planner.py`` to separate *planning* (what tools to
call) from *execution* (calling them, caching, voting, image-arg resolution).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from confidence.self_consistency import SelfConsistencyVoter
from controller.audit_log import AuditLog
from controller.memory import RoutingMemory
from controller.result_contract import ResultContract
from controller.schemas import (
    Modality,
    Query,
    SpecialistOutput,
    ToolCall,
)
from controller.tool_registry import ToolRegistry

logger = structlog.get_logger(__name__)

# Tools that consume real pixel data, and the argument name each expects it
# under. Paths always come from the uploaded images / prior step outputs —
# never from the planner (LLM or deterministic fallback), which has no way to know the real
# server-side temp path and must not be trusted to invent one. Includes the
# vlm_* variants: they take the same image_path/image_t1_path/image_t2_path
# arguments as their classical counterparts (see each specialist's run()
# docstring), so omitting them here left an LLM-supplied path unresolved and
# passed straight to load_pil/load_raster — the exact hallucinated/attacker-
# controlled-path case this override exists to prevent.
_IMAGE_ARG_TOOLS = {
    "vqa_caption", "vlm_vqa_caption",
    "grounding", "vlm_grounding",
    "sar_bridge",
    "change_detection", "vlm_change_detection",
    "optical_sar_fusion", "fusion_change_detection",
}

# Tools re-invoked through SelfConsistencyVoter (see confidence/self_consistency.py).
# Deliberately just the LLM/VLM-backed specialists — the classical baselines
# (vqa_caption, grounding, change_detection, sar_bridge, optical_sar_fusion)
# are deterministic signal processing over the same pixels every time, so
# voting would always report agreement_ratio == 1.0 while still paying for
# n_paths re-invocations. No design note or test pinned this down before;
# this is that policy, made explicit.
_SELF_CONSISTENCY_TOOLS = {"vlm_vqa_caption", "vlm_grounding", "vlm_change_detection"}


class PlanExecutor:
    """Executes an :class:`ExecutionPlan` step-by-step through the ToolRegistry.

    Owns: step execution, image-arg resolution, routing-memory caching,
    self-consistency voting, and the ``analysis_unavailable`` fallback.
    Does NOT own: planning, synthesis, or confidence estimation.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        memory: RoutingMemory,
        audit: AuditLog,
        contract: ResultContract,
        voter: SelfConsistencyVoter,
    ) -> None:
        self.registry = registry
        self.memory = memory
        self.audit = audit
        self.contract = contract
        self._voter = voter

    async def execute(
        self,
        query: Query,
        plan: ExecutionPlan,
        image_bytes: bytes | None = None,
    ) -> list[SpecialistOutput]:
        """Execute every step, threading prior outputs forward."""
        outputs: list[SpecialistOutput] = []
        prior_results: dict[str, dict[str, Any]] = {}
        for step in plan.steps:
            output = await self._execute_step(query, step, image_bytes, prior_results)
            outputs.append(output)
            prior_results[step.tool_name] = output.result
        return outputs

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(step: ToolCall, query: Query, image_bytes: bytes | None) -> bytes:
        """Routing-memory identity: tool arguments + *every* image's bytes.

        `RoutingMemory` sha256-hashes whatever we hand it, so we just need to
        hand it something that changes whenever the effective input does —
        the question/prompt/filter args, and each image (not only the first).
        """
        blob = json.dumps(step.arguments, sort_keys=True, default=str).encode()
        for i, img in enumerate(query.images):
            if i == 0 and image_bytes is not None:
                blob += b"|" + image_bytes
            elif img.path:
                blob += b"|" + Path(img.path).read_bytes()
        return blob

    async def _execute_step(
        self,
        query: Query,
        step: ToolCall,
        image_bytes: bytes | None,
        prior_results: dict[str, dict[str, Any]],
    ) -> SpecialistOutput:
        """Execute a single tool-call step, with routing-memory caching."""
        # Derive geospatial context from the query's images for evidence linkage.
        query_bbox = next((img.bbox for img in query.images if img.bbox), None)
        query_crs = next((img.crs for img in query.images if img.crs), None)

        # Override with real, server-side image paths and the trusted modality —
        # the planner (LLM or deterministic fallback) never has access to these and must not be
        # trusted with them. This happens *before* the cache key is derived so
        # the key describes the call we actually make, not the one the LLM
        # proposed; otherwise two plans that resolve to an identical call get
        # separate cache entries.
        if step.tool_name in _IMAGE_ARG_TOOLS:
            step = step.model_copy(
                update={"arguments": self._resolve_image_args(step, query, prior_results)}
            )

        # Check routing memory. The key must capture the *effective* tool
        # input — not just "some image was involved" — or two different
        # questions (or a second image) on the same session/tool collide.
        cache_key = self._cache_key(step, query, image_bytes)
        cached = self.memory.recall(query.session_id, cache_key, step.tool_name)
        if cached is not None:
            logger.info(
                "step_cache_hit",
                query_id=query.query_id,
                tool=step.tool_name,
            )
            return self.contract.finish_output(
                step.tool_name, cached, step.arguments, query_bbox, query_crs,
            )

        # Log tool invocation
        self.audit.log_event(
            query_id=query.query_id,
            event_type="tool_called",
            input_data=step.model_dump(mode="json"),
            output_data={},
            tool_name=step.tool_name,
        )

        # Call the tool
        try:
            raw_result = self.registry.call_tool(step.tool_name, **step.arguments)
        except RuntimeError as exc:
            # Registered but unbound: no model to run. Say so.
            raw_result = self._unavailable(
                step.tool_name, "no model implementation is bound for this tool", exc
            )
        except TypeError as exc:
            # The planner is an LLM: it can emit argument names no specialist
            # accepts, or omit required ones. The policy gate catches the cases
            # the schema declares; anything left is still a bad plan, not a
            # server fault, so report it as unavailable rather than 500-ing.
            logger.warning(
                "step_bad_arguments",
                query_id=query.query_id,
                tool=step.tool_name,
                arguments=sorted(step.arguments),
                error=str(exc),
            )
            raw_result = self._unavailable(
                step.tool_name, "the planned arguments did not match the tool's signature", exc
            )

        # Self-consistency voting (see _SELF_CONSISTENCY_TOOLS above). The
        # call just above already proved the tool is bound and the arguments
        # are valid, so this only re-runs on that known-good path — it never
        # duplicates the RuntimeError/TypeError classification above.
        if (
            step.tool_name in _SELF_CONSISTENCY_TOOLS
            and isinstance(raw_result, dict)
            and not raw_result.get("analysis_unavailable")
        ):
            raw_result = self._vote(step)

        output = self.contract.finish_output(
            step.tool_name, raw_result, step.arguments, query_bbox, query_crs,
        )

        # Store in routing memory
        self.memory.store(
            query.session_id,
            cache_key,
            step.tool_name,
            output.result,
        )

        # Log tool return
        self.audit.log_event(
            query_id=query.query_id,
            event_type="tool_returned",
            input_data=step.model_dump(mode="json"),
            output_data=output.model_dump(mode="json"),
            tool_name=step.tool_name,
        )

        return output

    # ------------------------------------------------------------------
    # Image-argument resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_image_args(
        step: ToolCall,
        query: Query,
        prior_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Fill in real image-path arguments for a pixel-consuming tool.

        Always overrides (never merely fills in) path arguments, since a
        planner-supplied path would either be hallucinated or attacker-
        controlled — the only trustworthy source is the server-side upload.
        """
        args = dict(step.arguments)
        paths = [img.path for img in query.images if img.path]

        # Modality is declared per-upload and validated server-side, so the
        # planner's guess must not survive: an LLM that omits or misreads it
        # gets a SAR pair labelled "[OPTICAL pair]". Writing the trusted value
        # back into the arguments also lands it in the routing-memory cache
        # key, so the same bytes read as optical and as SAR stop colliding.
        if step.tool_name in (
            "vqa_caption", "vlm_vqa_caption", "change_detection", "vlm_change_detection",
        ) and query.images:
            args["modality"] = query.images[0].modality.value

        if step.tool_name in ("vqa_caption", "vlm_vqa_caption", "grounding", "vlm_grounding") and paths:
            bridged = prior_results.get("sar_bridge", {}).get("bridged_image_path")
            optical_paths = [
                img.path for img in query.images
                if img.modality in (Modality.OPTICAL, Modality.MULTISPECTRAL) and img.path
            ]
            # Bridged > real optical/multispectral > raw first upload. Blindly
            # using paths[0] ran the optical spectral-index baseline on SAR
            # backscatter whenever the SAR upload happened to be listed first.
            image_path = bridged or (optical_paths[0] if optical_paths else paths[0])
            args["image_path"] = image_path
            if step.tool_name in ("vqa_caption", "vlm_vqa_caption") and optical_paths and not bridged:
                args["modality"] = next(
                    img.modality.value for img in query.images if img.path == optical_paths[0]
                )
        elif step.tool_name == "sar_bridge" and paths:
            sar_paths = [
                img.path for img in query.images if img.modality == Modality.SAR and img.path
            ] or paths
            # Despeckle every SAR image, not just the first — a bi-temporal
            # SAR pair headed for change_detection needs both bridged, or
            # the diff ends up comparing despeckled vs. still-noisy imagery.
            args["sar_image_path"] = sar_paths[0]
            args["sar_image_paths"] = sar_paths
        elif step.tool_name in ("change_detection", "vlm_change_detection") and len(paths) >= 2:
            bridged = prior_results.get("sar_bridge", {}).get("bridged_image_paths") or []
            if len(bridged) >= 2:
                args["image_t1_path"], args["image_t2_path"] = bridged[0], bridged[1]
            else:
                args["image_t1_path"], args["image_t2_path"] = paths[0], paths[1]
        elif step.tool_name == "fusion_change_detection":
            optical_imgs = [
                img for img in query.images
                if img.modality in (Modality.OPTICAL, Modality.MULTISPECTRAL) and img.path
            ]
            sar_imgs = [img for img in query.images if img.modality == Modality.SAR and img.path]
            # Prefer an optical/multispectral bi-temporal pair as the primary
            # evidence; fall back to whatever two images exist otherwise.
            primary = optical_imgs if len(optical_imgs) >= 2 else [
                img for img in query.images if img.path
            ]
            if len(primary) >= 2:
                args["image_t1_path"], args["image_t2_path"] = primary[0].path, primary[1].path
                if primary[0].acquisition_date:
                    args["date_t1"] = primary[0].acquisition_date.isoformat()
                if primary[1].acquisition_date:
                    args["date_t2"] = primary[1].acquisition_date.isoformat()
            # A separate SAR pair (not already used as the primary pair) is
            # additional cross-modal evidence, not a replacement for it.
            if len(sar_imgs) >= 2 and sar_imgs != primary:
                args["sar_image_t1_path"] = sar_imgs[0].path
                args["sar_image_t2_path"] = sar_imgs[1].path
        elif step.tool_name == "optical_sar_fusion":
            sar_paths = [
                img.path for img in query.images if img.modality == Modality.SAR and img.path
            ]
            optical_paths = [
                img.path for img in query.images
                if img.modality in (Modality.OPTICAL, Modality.MULTISPECTRAL) and img.path
            ]
            if sar_paths and optical_paths:
                args["sar_image_path"] = sar_paths[0]
                args["optical_image_path"] = optical_paths[0]

        return args

    # ------------------------------------------------------------------
    # Self-consistency voting
    # ------------------------------------------------------------------

    def _vote(self, step: ToolCall) -> dict[str, Any]:
        """Re-run an LLM/VLM-backed specialist through SelfConsistencyVoter.

        ``ToolRegistry.call_tool`` is synchronous, so the sync ``vote()`` API
        is enough here — no need to make this async just to re-invoke it
        n_paths times.
        """
        vote = self._voter.vote(
            lambda **kw: self.registry.call_tool(step.tool_name, **kw), **step.arguments
        )
        if not vote["all_answers"]:
            # Every path failed after the probe call above succeeded — a
            # transient/stochastic failure, not a binding or argument issue.
            return self._unavailable(
                step.tool_name, f"all {self._voter.n_paths} self-consistency paths failed"
            )
        result = dict(vote["best_result"])
        result["self_consistency"] = {
            "agreement_ratio": vote["agreement_ratio"],
            "n_consistent": vote["n_consistent"],
            "total_paths": vote["total_paths"],
        }
        return result

    # ------------------------------------------------------------------
    # Unavailability (never fabrication)
    # ------------------------------------------------------------------

    @staticmethod
    def _unavailable(
        tool_name: str, reason: str, exc: Exception | None = None
    ) -> dict[str, Any]:
        """Report that a tool could not run, with the reason.

        This replaces the former placeholder responses. Those returned invented
        summaries, zeroed statistics and ``/tmp`` paths that never existed —
        shaped exactly like a real result, so nothing downstream (or no one
        reading a report) could tell the difference. An explicit
        ``analysis_unavailable`` flag cannot be mistaken for an analysis.
        """
        from controller.model_registry import version_for_task

        logger.warning("tool_unavailable", tool=tool_name, reason=reason)
        record: dict[str, Any] = {
            "analysis_unavailable": True,
            "tool": tool_name,
            "reason": reason,
            "model_version": version_for_task(tool_name),
        }
        if exc is not None:
            record["detail"] = str(exc)
        return record
