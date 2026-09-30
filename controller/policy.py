"""
Policy Engine — the gate between a *proposed* plan and any execution.

The LLM is a planner, not an executor: everything it emits is untrusted input.
This module rejects a plan *before* a single tool runs, so an invalid plan
produces one clear refusal instead of a chain of specialists each improvising
on arguments that never made sense.

Checks are driven by the tool's own JSON schema in ``controller/tool_schemas/``
rather than a second hand-maintained copy of the rules.
"""

from __future__ import annotations

import structlog

from controller.schemas import ExecutionPlan, Modality, Query
from controller.tool_registry import ToolRegistry

logger = structlog.get_logger(__name__)


class PolicyViolationError(Exception):
    """A planned step is not permitted to execute. Message is user-safe."""


PolicyViolation = PolicyViolationError  # backward-compat alias; do not remove


# Tools whose *inputs* the server supplies from the upload set, keyed by the
# minimum number of images each needs to be answerable at all.
_MIN_IMAGES = {
    "vqa_caption": 1,
    "vlm_vqa_caption": 1,
    "grounding": 1,
    "vlm_grounding": 1,
    "sar_bridge": 1,
    "change_detection": 2,
    "vlm_change_detection": 2,
    "optical_sar_fusion": 2,
    # Base requirement is the optical bi-temporal pair; an optional SAR pair
    # adds evidence but isn't required to run (see FusionChangeDetectionSpecialist).
    "fusion_change_detection": 2,
}

# Argument names the server injects itself (see AgenticPlanner._resolve_image_args).
# A plan need not supply them, and anything it does supply is overwritten.
_SERVER_SUPPLIED = {
    "image_path", "image_t1_path", "image_t2_path",
    "sar_image_path", "optical_image_path", "modality",
}


def check_plan(plan: ExecutionPlan, query: Query, registry: ToolRegistry) -> None:
    """Validate a plan against the registry and the actual inputs.

    Raises:
        PolicyViolationError: on the first rule the plan breaks. The message states
            the reason plainly so it can be surfaced to the user verbatim.
    """
    if not plan.steps:
        raise PolicyViolationError("the planner produced no executable steps")

    known = set(registry.list_tool_names())
    modalities = {img.modality for img in query.images}
    # Metadata without a server-side path is not readable pixel data, so it
    # cannot satisfy a tool's input requirement.
    n_images = sum(1 for img in query.images if img.path)
    seen: list[str] = []

    for step in plan.steps:
        tool = step.tool_name

        if tool not in known:
            raise PolicyViolationError(f"tool {tool!r} is not in the registry")

        # An unbound tool has no implementation. Refusing here is what keeps a
        # missing model from being answered with an invented result.
        if registry.get_tool(tool).implementation is None:
            raise PolicyViolationError(
                f"tool {tool!r} has no model bound in this deployment"
            )

        need = _MIN_IMAGES.get(tool, 0)
        if n_images < need:
            raise PolicyViolationError(
                f"{tool!r} needs {need} readable image(s); {n_images} were provided"
            )

        # Modality preconditions: these tools are defined by the sensor
        # combination they consume, so the wrong pair is a routing error.
        if tool == "sar_bridge" and Modality.SAR not in modalities:
            raise PolicyViolationError("'sar_bridge' requires a SAR image; none was provided")
        if tool == "optical_sar_fusion" and not (
            Modality.SAR in modalities
            and modalities & {Modality.OPTICAL, Modality.MULTISPECTRAL}
        ):
            raise PolicyViolationError(
                "'optical_sar_fusion' requires one SAR and one optical/multispectral image"
            )
        # Bi-temporal means same sensor. A SAR/optical pair differenced as if it
        # were one belongs in optical_sar_fusion — the "change" would be sensor
        # physics, not ground change. An UNKNOWN modality is left alone: it is
        # an undeclared upload, not a declared mismatch.
        if tool in ("change_detection", "vlm_change_detection") and Modality.SAR in modalities and (
            modalities & {Modality.OPTICAL, Modality.MULTISPECTRAL}
        ):
            raise PolicyViolationError(
                f"{tool!r} requires a bi-temporal pair of the same modality; "
                f"got {sorted(m.value for m in modalities)} — use 'optical_sar_fusion'"
            )

        # Dependencies must already have run — a forward or self reference means
        # the step would read an output that does not exist yet.
        for dep in step.depends_on:
            if dep == tool:
                raise PolicyViolationError(f"{tool!r} depends on itself")
            if dep not in seen:
                if dep in known:      # planned later, or not planned at all
                    raise PolicyViolationError(
                        f"{tool!r} depends on {dep!r}, which does not run before it"
                    )
                raise PolicyViolationError(f"{tool!r} depends on unknown tool {dep!r}")

        _check_arguments(step.tool_name, step.arguments, registry)
        seen.append(tool)

    logger.info("plan_policy_passed", query_id=plan.query_id, steps=seen)


def _check_arguments(tool: str, arguments: dict, registry: ToolRegistry) -> None:
    """Reject unknown or missing parameters, per the tool's own JSON schema."""
    params = registry.get_tool(tool).parameters
    properties = params.get("properties") or {}
    if not properties:
        return  # schema declares no parameter contract to enforce

    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        raise PolicyViolationError(
            f"{tool!r} was given parameter(s) it does not accept: {unknown}"
        )

    missing = sorted(
        set(params.get("required", [])) - set(arguments) - _SERVER_SUPPLIED
    )
    if missing:
        raise PolicyViolationError(f"{tool!r} is missing required parameter(s): {missing}")
