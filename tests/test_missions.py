"""Tests for controller/missions.py."""

from __future__ import annotations

import pytest

from controller.missions import CUSTOM_MISSION, MISSIONS, build_plan, mission_catalog
from controller.schemas import ImageMetadata, Modality, Query
from interfaces.api.deps import get_planner


def _img(modality: Modality, path: str | None = None) -> ImageMetadata:
    return ImageMetadata(filename="x.png", path=path, modality=modality)


def test_unknown_mission_raises() -> None:
    with pytest.raises(KeyError):
        build_plan("not_a_mission", Query(text=""), get_planner())


def test_single_image_inserts_grounding_before_vqa() -> None:
    q = Query(text="", images=[_img(Modality.OPTICAL, "/tmp/a.png")])
    query, plan = build_plan("water_body_monitoring", q, get_planner())
    assert [s.tool_name for s in plan.steps] == ["grounding", "vqa_caption"]
    assert query.text == MISSIONS["water_body_monitoring"]["question"]
    grounding_step = plan.steps[0]
    assert grounding_step.arguments["text_prompt"] == "water"
    assert plan.steps[-1].depends_on[-1] == "grounding"


def test_existing_question_is_not_overridden() -> None:
    q = Query(text="Custom question", images=[_img(Modality.OPTICAL, "/tmp/a.png")])
    query, _plan = build_plan("flood", q, get_planner())
    assert query.text == "Custom question"


def test_bitemporal_pair_routes_change_detection_then_grounding() -> None:
    q = Query(text="", images=[
        _img(Modality.OPTICAL, "/tmp/t1.png"), _img(Modality.OPTICAL, "/tmp/t2.png"),
    ])
    _query, plan = build_plan("crop_change", q, get_planner())
    assert [s.tool_name for s in plan.steps] == ["change_detection", "grounding", "vqa_caption"]


def test_all_sar_query_skips_grounding() -> None:
    q = Query(text="", images=[_img(Modality.SAR, "/tmp/a.png")])
    _query, plan = build_plan("flood", q, get_planner())
    assert "grounding" not in [s.tool_name for s in plan.steps]


def test_no_images_still_produces_a_plan() -> None:
    q = Query(text="")
    _query, plan = build_plan("disaster_assessment", q, get_planner())
    assert [s.tool_name for s in plan.steps] == ["vqa_caption"]


@pytest.mark.asyncio
async def test_mission_runs_end_to_end_via_process_query(tmp_path) -> None:
    import numpy as np
    from PIL import Image

    p = tmp_path / "img.png"
    arr = np.random.default_rng(0).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(p)

    planner = get_planner()
    q = Query(text="", images=[_img(Modality.OPTICAL, str(p))])
    query, plan = build_plan("water_body_monitoring", q, planner)
    result = await planner.process_query(query, plan=plan)

    assert {o.tool_name for o in result.outputs} == {"grounding", "vqa_caption"}
    assert "unavailable" not in result.final_answer.lower()


@pytest.mark.parametrize("mission", [
    "wildfire_burn_severity", "landslide_detection", "deforestation", "coastal_erosion",
    "infrastructure_development", "mining_industrial_expansion", "road_network_change",
])
def test_new_preset_missions_insert_grounding(mission: str) -> None:
    q = Query(text="", images=[_img(Modality.OPTICAL, "/tmp/a.png")])
    query, plan = build_plan(mission, q, get_planner())
    assert [s.tool_name for s in plan.steps] == ["grounding", "vqa_caption"]
    assert query.text == MISSIONS[mission]["question"]


def test_mission_catalog_covers_every_entry() -> None:
    catalog = mission_catalog()
    assert {c["name"] for c in catalog} == set(MISSIONS)
    assert all(c["emoji"] for c in catalog)


def test_custom_mission_decomposes_multiple_concepts() -> None:
    q = Query(
        text="Find areas where vegetation decreased and built-up area increased from 2022-2026.",
        images=[_img(Modality.OPTICAL, "/tmp/t1.png"), _img(Modality.OPTICAL, "/tmp/t2.png")],
    )
    query, plan = build_plan(CUSTOM_MISSION, q, get_planner())
    assert [s.tool_name for s in plan.steps] == [
        "change_detection", "grounding", "grounding", "vqa_caption",
    ]
    prompts = [s.arguments["text_prompt"] for s in plan.steps if s.tool_name == "grounding"]
    assert prompts == ["vegetation", "urban/built-up"]
    # no grounding step depends on another grounding step (same tool_name, would
    # trip the policy engine's "depends on itself" check)
    assert all("grounding" not in s.depends_on for s in plan.steps if s.tool_name == "grounding")


def test_custom_mission_requires_objective_text() -> None:
    with pytest.raises(ValueError):
        build_plan(CUSTOM_MISSION, Query(text="   "), get_planner())


def test_custom_mission_skips_grounding_for_all_sar() -> None:
    q = Query(text="find water", images=[_img(Modality.SAR, "/tmp/a.png")])
    _query, plan = build_plan(CUSTOM_MISSION, q, get_planner())
    assert "grounding" not in [s.tool_name for s in plan.steps]
