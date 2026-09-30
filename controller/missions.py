"""
Mission-oriented workflow presets.

No new specialists: each mission is a named query template over the tool
chain that already exists (see ``controller/planner.py``). ``_plan_rule_based``
already routes multi-image/cross-modal/SAR queries correctly; the only gap
a mission preset fills is the ``grounding`` call the deterministic fallback router never makes
on its own, using the same water/vegetation/urban/bare/coast keyword classes
already in ``specialists/_imagery.py`` (``flood``/``river`` -> water,
``crop`` -> vegetation, ``built`` -> urban, ``burn``/``landslide``/``bare``
-> bare/other, ``coast``/``shoreline`` -> water, etc).

Grounding is skipped for an all-SAR query: its classical thresholds are
spectral-index math defined for optical bands, and running them on SAR
amplitude would be a misapplied baseline, not a real mission capability.

Adding a mission is adding a ``MISSIONS`` entry (name, question,
grounding_prompt, label, emoji) — nothing in the planner, policy, specialist,
confidence, provenance or audit layers changes. The one exception is mission
12, the Custom Mission Generator (``CUSTOM_MISSION``): it has no fixed
question or keyword, so ``build_plan`` routes it to ``_build_custom_plan``
below instead of the generic preset path.
"""

from __future__ import annotations

from typing import Any

from controller.planner import AgenticPlanner
from controller.schemas import ExecutionPlan, ExecutionResult, Modality, Query, ToolCall
from specialists._imagery import class_label, detect_prompt_classes

# Keys already returned by every specialist's result dict (see specialists/*)
# that mean "this isn't a real measurement" rather than a metric. Everything
# else numeric/boolean is generic per-tool signal — no per-mission unpacking.
_NON_METRIC_KEYS = {"analysis_unavailable", "confidence"}

# Mission 12 — handled by _build_custom_plan below instead of the generic
# preset path, since it has no fixed question/grounding_prompt of its own.
CUSTOM_MISSION = "custom_mission_generator"

MISSIONS: dict[str, dict[str, str]] = {
    "flood": {
        "question": "Assess flood extent: identify and quantify inundated water areas.",
        "grounding_prompt": "flood water",
        "label": "Flood & Inundation", "emoji": "🌊",
    },
    "urban_expansion": {
        "question": "Assess urban expansion: identify new or expanded built-up area.",
        "grounding_prompt": "urban built-up",
        "label": "Urban Expansion", "emoji": "🏙️",
    },
    "crop_change": {
        "question": "Assess crop and vegetation change over the observed period.",
        "grounding_prompt": "crop vegetation",
        "label": "Crop Change", "emoji": "🌾",
    },
    "disaster_assessment": {
        "question": "Assess disaster-affected area and the extent of visible damage.",
        "grounding_prompt": "damage",
        "label": "Disaster Assessment", "emoji": "🆘",
    },
    "water_body_monitoring": {
        "question": "Monitor water body extent and boundary.",
        "grounding_prompt": "water",
        "label": "Water Monitoring", "emoji": "💧",
    },
    "wildfire_burn_severity": {
        "question": (
            "Detect burned areas and describe burn severity (low/moderate/high) "
            "based on the measured change extent between the pre- and post-fire images."
        ),
        "grounding_prompt": "burn scars",
        "label": "Wildfire / Burn Severity", "emoji": "🔥",
    },
    "landslide_detection": {
        "question": (
            "Identify surface disturbance consistent with a landslide by comparing "
            "the available optical and/or SAR imagery."
        ),
        "grounding_prompt": "landslide debris",
        "label": "Landslide Detection", "emoji": "🏔️",
    },
    "deforestation": {
        "question": "Detect forest loss and highlight deforestation hotspots over the observed period.",
        "grounding_prompt": "forest cover",
        "label": "Deforestation", "emoji": "🌳",
    },
    "coastal_erosion": {
        "question": "Measure shoreline movement and classify areas of erosion versus accretion.",
        "grounding_prompt": "coastal shoreline",
        "label": "Coastal Erosion", "emoji": "🌊",
    },
    "infrastructure_development": {
        "question": "Detect new, removed, or modified buildings, roads and other built structures.",
        "grounding_prompt": "built-up structures",
        "label": "Infrastructure Development", "emoji": "🏗️",
    },
    "mining_industrial_expansion": {
        "question": (
            "Detect surface disturbance and areal expansion associated with "
            "mining or industrial activity."
        ),
        "grounding_prompt": "bare disturbed ground",
        "label": "Mining / Industrial Expansion", "emoji": "⛏️",
    },
    "road_network_change": {
        "question": "Detect new, removed, or modified road segments.",
        "grounding_prompt": "road network",
        "label": "Road-Network Change", "emoji": "🛣️",
    },
    CUSTOM_MISSION: {
        "question": "",  # supplied by the caller's own objective — no fixed default
        "grounding_prompt": "",  # resolved per-query, see _build_custom_plan
        "label": "Custom Mission Generator", "emoji": "🧠",
    },
}


def mission_catalog() -> list[dict[str, str]]:
    """Public catalog for a mission picker: name, label, emoji, default question.

    Reads straight off ``MISSIONS`` — a new mission dict entry is all that's
    needed for it to show up here too.
    """
    return [
        {
            "name": name,
            "label": preset.get("label", name),
            "emoji": preset.get("emoji", ""),
            "question": preset.get("question", ""),
        }
        for name, preset in MISSIONS.items()
    ]


def build_plan(mission: str, query: Query, planner: AgenticPlanner) -> tuple[Query, ExecutionPlan]:
    """Return ``(query, plan)`` for a named mission preset.

    ``query.text`` is kept if the caller already supplied one; only a blank
    question is replaced by the mission's default. Raises ``KeyError`` for
    an unrecognised mission name.
    """
    preset = MISSIONS[mission]  # KeyError for an unrecognised mission name
    if mission == CUSTOM_MISSION:
        return _build_custom_plan(query, planner)

    q = query.model_copy(update={"text": query.text or preset["question"]})
    plan = planner._plan_rule_based(q)  # same routing rules as every other query

    if any(img.modality != Modality.SAR for img in q.images) and (
        "grounding" in planner.registry.list_tool_names()
    ):
        vqa = plan.steps[-1]
        grounding = ToolCall(
            tool_name="grounding",
            arguments={"text_prompt": preset["grounding_prompt"]},
            rationale=f"Mission {mission!r} — segment {preset['grounding_prompt']!r}",
            depends_on=[s.tool_name for s in plan.steps[:-1]],
        )
        vqa.depends_on = [*vqa.depends_on, "grounding"]
        plan.steps[-1:] = [grounding, vqa]

    plan.strategy_note = f"Mission preset {mission!r}. {plan.strategy_note}"
    return q, plan


def mission_summary(result: ExecutionResult) -> dict[str, Any]:
    """Reshape any mission's :class:`ExecutionResult` into one fixed envelope.

    ``prediction`` + ``spatial_evidence`` + ``metrics`` + ``confidence`` +
    ``warnings`` — read generically off ``result.outputs``, the same shape
    for all 12 missions. No per-tool-name branching: every specialist
    already reports artifacts under an ``artifact``/``*_path`` key, plain
    numbers/flags as metrics, and failures via ``analysis_unavailable`` +
    ``reason`` (see specialists/*), so one pass over each output's result
    dict covers every mission without new backend logic.
    """
    spatial_evidence: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for out in result.outputs:
        r = out.result
        # Already computed by AgenticPlanner._finish_output — same rule set a
        # plain /v1/query result uses, so a mission and a query flag the same
        # things the same way.
        warnings.extend(out.warnings)
        if r.get("analysis_unavailable"):
            continue

        artifact = r.get("artifact")
        if isinstance(artifact, dict) and artifact.get("path"):
            spatial_evidence.append({"tool": out.tool_name, **artifact})
        if "bbox" in r:
            spatial_evidence.append({"tool": out.tool_name, "bbox": r["bbox"]})

        metrics[out.tool_name] = {
            "model_version": out.model_version,
            "parameters": out.parameters,
            **{
                k: v for k, v in r.items()
                if k not in _NON_METRIC_KEYS and isinstance(v, int | float | bool)
            },
        }

    return {
        "prediction": result.final_answer,
        "spatial_evidence": spatial_evidence,
        "metrics": metrics,
        "confidence": result.confidence.model_dump() if result.confidence else None,
        "warnings": warnings,
        "timestamp": result.timestamp.isoformat(),
    }


def _build_custom_plan(query: Query, planner: AgenticPlanner) -> tuple[Query, ExecutionPlan]:
    """Mission 12 — Custom Mission Generator.

    Turns a free-text EO objective into a workflow with no new intent
    classifier and no new specialists: reuses the same routing
    (``AgenticPlanner._plan_rule_based``) and the same land-cover keyword table
    every preset mission already draws on
    (``specialists._imagery.detect_prompt_classes``), just applied for
    *every* distinct concept named in the objective instead of one fixed
    keyword. "Vegetation decreased and built-up increased" becomes one
    ``grounding`` call per concept, ahead of whatever change/fusion step the
    supplied images already earn from ``_plan_rule_based``. The diagram's final
    "spatial intersection" step is left to the existing LLM synthesis pass,
    which sees every grounding result side by side — computing a real
    pixel-wise mask intersection would be new specialist logic, which this
    generator deliberately does not add.

    Raises:
        ValueError: if no objective text was supplied — there is no default
            question to fall back on, unlike a named preset.
    """
    if not query.text.strip():
        raise ValueError("the custom mission requires a natural-language objective")

    plan = planner._plan_rule_based(query)  # same routing rules as every other query
    all_sar = bool(query.images) and all(img.modality == Modality.SAR for img in query.images)
    classes = (
        []
        if all_sar or "grounding" not in planner.registry.list_tool_names()
        else detect_prompt_classes(query.text)
    )

    if classes:
        vqa = plan.steps[-1]
        prior = [s.tool_name for s in plan.steps[:-1]]
        grounding_steps = [
            ToolCall(
                tool_name="grounding",
                arguments={"text_prompt": class_label(c)},
                rationale=f"Custom mission — segment {class_label(c)!r}, detected in objective",
                depends_on=prior,
            )
            for c in classes
        ]
        vqa.depends_on = [*vqa.depends_on, "grounding"]
        plan.steps[-1:] = [*grounding_steps, vqa]
        concepts = ", ".join(class_label(c) for c in classes)
        plan.strategy_note = f"Custom mission — objective decomposed into: {concepts}. {plan.strategy_note}"
    else:
        plan.strategy_note = f"Custom mission — no known land-cover concept detected. {plan.strategy_note}"

    return query, plan
