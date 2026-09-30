"""
Tests for M20 — Counterfactual / Self-Checking Agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.modules.m20_counterfactual import run_counterfactual_check
from pipeline.schemas.contracts import ResultStatus

IMAGES = [{"path": "t1.tif", "modality": "optical"}, {"path": "t2.tif", "modality": "optical"}]


class _FakeRegistry:
    """Only what M20 touches: call_tool(name, **kwargs)."""
    def __init__(self, responses=None, raises_for=()):
        self._responses = responses or {}
        self._raises_for = set(raises_for)

    def call_tool(self, tool_name, **kwargs):
        if tool_name in self._raises_for:
            raise RuntimeError("alternate route exploded")
        return self._responses.get(tool_name, {"analysis_unavailable": True, "reason": "no model bound"})


class TestM20Counterfactual:
    def test_skips_with_no_outputs(self):
        result = run_counterfactual_check([], [], IMAGES, _FakeRegistry())
        assert result.status == ResultStatus.SKIPPED

    def test_agreeing_alternate_change_route_corroborates(self):
        outputs = [{"tool_name": "change_detection", "result": {"changed_area_pct": 10.0}}]
        registry = _FakeRegistry(responses={"vlm_change_detection": {"changed_area_pct": 8.0}})
        result = run_counterfactual_check(outputs, [], IMAGES, registry)

        check = result.data["checks"][0]
        assert check["method"] == "alternate_route"
        assert check["corroborated"] is True
        assert result.data["falsification_succeeded"] is False

    def test_disagreeing_alternate_change_route_is_contradicted(self):
        outputs = [{"tool_name": "change_detection", "result": {"changed_area_pct": 10.0}}]
        registry = _FakeRegistry(responses={"vlm_change_detection": {"changed_area_pct": 0.5}})
        result = run_counterfactual_check(outputs, [], IMAGES, registry)

        assert result.data["falsification_succeeded"] is True
        assert result.data["contradicted_tools"] == ["change_detection"]

    def test_grounding_bbox_iou_agreement(self):
        outputs = [{"tool_name": "grounding", "result": {"bbox": [0, 0, 10, 10]}}]
        registry = _FakeRegistry(responses={"vlm_grounding": {"bbox": [1, 1, 11, 11]}})
        plan_steps = [{"tool_name": "grounding", "arguments": {"text_prompt": "the building"}}]
        result = run_counterfactual_check(outputs, plan_steps, [IMAGES[0]], registry)

        check = result.data["checks"][0]
        assert check["method"] == "alternate_route"
        assert check["bbox_iou"] > 0.3
        assert check["corroborated"] is True

    def test_self_consistency_agreement_is_surfaced_not_rerun(self):
        outputs = [{
            "tool_name": "vlm_caption",
            "result": {"self_consistency": {"agreement_ratio": 0.9}},
        }]
        result = run_counterfactual_check(outputs, [], IMAGES, _FakeRegistry())

        check = result.data["checks"][0]
        assert check["method"] == "resample_agreement"
        assert check["corroborated"] is True

    def test_low_self_consistency_agreement_is_contradicted(self):
        outputs = [{
            "tool_name": "vlm_caption",
            "result": {"self_consistency": {"agreement_ratio": 0.2}},
        }]
        result = run_counterfactual_check(outputs, [], IMAGES, _FakeRegistry())
        assert result.data["falsification_succeeded"] is True

    def test_tool_with_no_check_available_is_reported_honestly(self):
        outputs = [{"tool_name": "vqa_caption", "result": {"answer": "a road"}}]
        result = run_counterfactual_check(outputs, [], IMAGES, _FakeRegistry())

        check = result.data["checks"][0]
        assert check["method"] == "none"
        assert check["attempted"] is False
        assert result.data["falsification_attempted"] == 0

    def test_failed_alternate_call_is_inconclusive_not_fatal(self):
        outputs = [{"tool_name": "change_detection", "result": {"changed_area_pct": 10.0}}]
        registry = _FakeRegistry(raises_for={"vlm_change_detection"})
        result = run_counterfactual_check(outputs, [], IMAGES, registry)

        check = result.data["checks"][0]
        assert check["corroborated"] is None
        assert "change_detection" in result.data["inconclusive_tools"]
        assert result.data["falsification_succeeded"] is False

    def test_analysis_unavailable_output_is_skipped_entirely(self):
        outputs = [{"tool_name": "change_detection", "result": {"analysis_unavailable": True}}]
        result = run_counterfactual_check(outputs, [], IMAGES, _FakeRegistry())
        assert result.data["checks"] == []

    def test_fewer_than_two_images_not_attempted(self):
        outputs = [{"tool_name": "change_detection", "result": {"changed_area_pct": 10.0}}]
        result = run_counterfactual_check(outputs, [], [IMAGES[0]], _FakeRegistry())
        check = result.data["checks"][0]
        assert check["attempted"] is False
