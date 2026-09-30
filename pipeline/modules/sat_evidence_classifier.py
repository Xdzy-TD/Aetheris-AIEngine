"""
SAT-OS Evidence Classifier — ESSENTIAL/SUPPORTIVE/REDUNDANT + leave-one-out.

Extends the existing m19 evidence graph with role classification and
minimum-evidence-set computation. Reuses the graph structure from m19.
"""

from __future__ import annotations

from typing import Any

from pipeline.schemas.sat_schemas import (
    ClassifiedEvidence,
    EvidenceRole,
    EvidenceSetReport,
)


def _classify_role(
    tool_name: str,
    result: dict[str, Any],
    all_tools: list[str],
) -> EvidenceRole:
    """Classify a tool's evidence role based on what it contributes."""
    if result.get("analysis_unavailable"):
        return EvidenceRole.UNKNOWN

    # Change detection outputs are essential in bi-temporal analyses
    if tool_name in ("change_detection", "vlm_change_detection", "fusion_change_detection"):
        return EvidenceRole.ESSENTIAL

    # VQA/caption is essential when it's the only answer source
    if tool_name in ("vqa_caption", "vlm_vqa_caption"):
        other_answer_sources = [
            t for t in all_tools
            if t != tool_name and t in ("vqa_caption", "vlm_vqa_caption")
        ]
        return EvidenceRole.ESSENTIAL if not other_answer_sources else EvidenceRole.REDUNDANT

    # SAR bridge is supportive (preprocessing, not direct evidence)
    if tool_name == "sar_bridge":
        return EvidenceRole.SUPPORTIVE

    # Grounding is supportive (localization, not the primary claim)
    if tool_name in ("grounding", "vlm_grounding"):
        return EvidenceRole.SUPPORTIVE

    # Fusion tools are supportive when a single-modality tool already ran
    if tool_name in ("optical_sar_fusion",):
        return EvidenceRole.SUPPORTIVE

    return EvidenceRole.UNKNOWN


def _leave_one_out(
    outputs: list[dict[str, Any]],
    tool_name: str,
) -> bool:
    """Would removing this tool's output break the final claim?

    A claim breaks if removing the tool leaves no confidence-bearing output
    or removes the only change-detection result in a change analysis.
    """
    remaining = [o for o in outputs if o.get("tool_name") != tool_name]
    remaining_available = [
        o for o in remaining if not o.get("result", {}).get("analysis_unavailable")
    ]

    # Nothing left → definitely breaks
    if not remaining_available:
        return True

    # Removing the only change-detection tool breaks a change claim
    change_tools = {"change_detection", "vlm_change_detection", "fusion_change_detection"}
    if tool_name in change_tools:
        remaining_change = [o for o in remaining_available if o.get("tool_name") in change_tools]
        if not remaining_change:
            return True

    # Removing the only answer source breaks
    answer_tools = {"vqa_caption", "vlm_vqa_caption"}
    if tool_name in answer_tools:
        remaining_answers = [o for o in remaining_available if o.get("tool_name") in answer_tools]
        if not remaining_answers:
            return True

    return False


def classify_evidence(
    outputs: list[dict[str, Any]],
) -> EvidenceSetReport:
    """Classify all evidence items and compute the minimum evidence set."""
    all_tools = [o.get("tool_name", "") for o in outputs]
    items: list[ClassifiedEvidence] = []

    for out in outputs:
        tool = out.get("tool_name", "")
        result = out.get("result", {})
        if result.get("analysis_unavailable"):
            continue

        role = _classify_role(tool, result, all_tools)
        breaks = _leave_one_out(outputs, tool)
        conf = result.get("confidence", 0.0)
        if not isinstance(conf, (int, float)):
            conf = 0.0

        items.append(ClassifiedEvidence(
            tool_name=tool,
            role=role,
            removal_breaks_claim=breaks,
            confidence_contribution=conf,
        ))

    # Minimum evidence set: everything that's ESSENTIAL or whose removal breaks the claim
    minimum_set = [
        item.tool_name for item in items
        if item.role == EvidenceRole.ESSENTIAL or item.removal_breaks_claim
    ]

    return EvidenceSetReport(items=items, minimum_set=minimum_set)
