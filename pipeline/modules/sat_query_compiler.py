"""
SAT-OS Query Compiler — NL question → ObservationContract.

Uses the existing Ollama planner when available, falls back to keyword
extraction. No new deps — just string matching and the schemas.
"""

from __future__ import annotations

import re
from typing import Any

from pipeline.schemas.sat_schemas import ObservationContract


# Keyword → phenomenon mapping. Covers the MVP flood path; extensible.
_PHENOMENON_KEYWORDS: dict[str, str] = {
    "flood": "flooding", "inundat": "flooding", "water level": "flooding",
    "deforest": "deforestation", "forest loss": "deforestation",
    "urban": "urban_expansion", "construction": "urban_expansion", "built": "urban_expansion",
    "change": "change_detection", "differ": "change_detection",
    "fire": "wildfire", "burn": "wildfire",
    "crop": "agricultural_change", "harvest": "agricultural_change",
    "landslide": "landslide", "erosion": "erosion",
}

_SCALE_KEYWORDS: dict[str, str] = {
    "building": "building-level", "house": "building-level", "structure": "building-level",
    "block": "city-block", "neighborhood": "city-block", "neighbourhood": "city-block",
    "city": "city-level", "town": "city-level", "village": "city-level",
    "region": "regional", "district": "regional", "state": "regional",
    "river": "watershed", "basin": "watershed", "catchment": "watershed",
}

_MODALITY_KEYWORDS: dict[str, str] = {
    "sar": "sar", "radar": "sar", "sentinel-1": "sar",
    "optical": "optical", "rgb": "optical", "sentinel-2": "optical",
    "multispectral": "multispectral", "ndvi": "multispectral",
}

# GSD thresholds implied by scale
_SCALE_GSD: dict[str, float] = {
    "building-level": 1.0, "city-block": 5.0, "city-level": 10.0,
    "regional": 30.0, "watershed": 30.0,
}


def compile_query(text: str, images: list[dict[str, Any]] | None = None) -> ObservationContract:
    """Extract an ObservationContract from a natural-language query.

    Rule-based keyword extraction — honest about being a heuristic, not NLU.
    """
    lower = text.lower()

    # Phenomenon
    phenomenon = ""
    for kw, phenom in _PHENOMENON_KEYWORDS.items():
        if kw in lower:
            phenomenon = phenom
            break

    # Target — take the noun phrase after "in" / "of" / "at" / "near" if present
    target = ""
    m = re.search(r"\b(?:in|of|at|near|around)\s+(?:the\s+)?(.+?)(?:\?|$|,|\.|before|after|between)", lower)
    if m:
        target = m.group(1).strip()

    # Spatial scale
    spatial_scale = ""
    for kw, scale in _SCALE_KEYWORDS.items():
        if kw in lower:
            spatial_scale = scale
            break

    # Temporal
    temporal = "single-date"
    if any(kw in lower for kw in ("before", "after", "change", "between", "compar")):
        temporal = "bi-temporal"
    if images and len(images) >= 2:
        temporal = "bi-temporal"

    # Modality — from query text or image metadata
    modality = "optical"
    for kw, mod in _MODALITY_KEYWORDS.items():
        if kw in lower:
            modality = mod
            break
    if images:
        for img in images:
            m_val = img.get("modality", "")
            if m_val and m_val != "unknown":
                modality = m_val
                break

    # GSD from scale
    required_gsd = _SCALE_GSD.get(spatial_scale)

    # Expected output
    if phenomenon == "flooding":
        expected_output = "flood extent map with change percentage"
    elif phenomenon == "change_detection":
        expected_output = "change map with affected area statistics"
    else:
        expected_output = "analysis result with evidence"

    return ObservationContract(
        target=target,
        phenomenon=phenomenon or "general_observation",
        spatial_scale=spatial_scale,
        temporal_requirement=temporal,
        sensor_modality=modality,
        required_gsd_m=required_gsd,
        expected_output=expected_output,
        quality_constraints=["cloud_cover < 30%"] if modality == "optical" else [],
        evidence_requirements=["change_map", "observability_mask"] if temporal == "bi-temporal" else ["observability_mask"],
    )
