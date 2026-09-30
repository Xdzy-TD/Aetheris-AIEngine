"""Tests for the geometric evaluation harness (evaluation/)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from evaluation.metrics import f1_mask, iou_bbox, iou_mask, load_binary_mask, mean
from evaluation.runner import _score_item


# ---------------------------------------------------------------------------
# Pure metric math — no planner, no disk beyond what a test builds itself.
# ---------------------------------------------------------------------------

class TestIoUMask:
    def test_identical_masks_score_one(self):
        m = np.array([[True, False], [False, True]])
        assert iou_mask(m, m) == 1.0

    def test_disjoint_masks_score_zero(self):
        a = np.array([[True, False], [False, False]])
        b = np.array([[False, True], [False, False]])
        assert iou_mask(a, b) == 0.0

    def test_partial_overlap(self):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        # intersection=1, union=3
        assert iou_mask(a, b) == pytest.approx(1 / 3)

    def test_both_empty_is_undefined_not_zero(self):
        m = np.zeros((4, 4), dtype=bool)
        assert iou_mask(m, m) is None

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            iou_mask(np.zeros((2, 2), dtype=bool), np.zeros((3, 3), dtype=bool))


class TestIoUBbox:
    def test_identical_boxes_score_one(self):
        box = [0, 0, 10, 10]
        assert iou_bbox(box, box) == 1.0

    def test_disjoint_boxes_score_zero(self):
        assert iou_bbox([0, 0, 5, 5], [10, 10, 15, 15]) == 0.0

    def test_partial_overlap(self):
        # [0,0,10,10] and [5,5,15,15] -> intersection 5x5=25, union 100+100-25=175
        assert iou_bbox([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)

    def test_degenerate_box_is_undefined(self):
        assert iou_bbox([0, 0, 0, 0], [0, 0, 0, 0]) is None


class TestF1Mask:
    def test_perfect_prediction(self):
        m = np.array([True, True, False, False])
        scores = f1_mask(m, m)
        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0
        assert scores["f1"] == 1.0

    def test_no_predicted_positives_precision_undefined(self):
        pred = np.array([False, False])
        gt = np.array([True, False])
        scores = f1_mask(pred, gt)
        assert scores["precision"] is None  # tp+fp == 0, not a fabricated 0.0
        assert scores["recall"] == 0.0

    def test_no_positives_at_all_recall_undefined(self):
        pred = np.array([False, False])
        gt = np.array([False, False])
        scores = f1_mask(pred, gt)
        assert scores["precision"] is None
        assert scores["recall"] is None
        assert scores["f1"] is None


class TestMean:
    def test_empty_is_none_not_zero(self):
        assert mean([]) is None

    def test_averages(self):
        assert mean([1.0, 2.0, 3.0]) == 2.0


class TestLoadBinaryMask:
    def test_thresholds_grayscale(self, tmp_path):
        arr = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        p = tmp_path / "m.png"
        Image.fromarray(arr, mode="L").save(p)
        mask = load_binary_mask(p)
        assert mask.tolist() == [[False, True], [True, False]]


# ---------------------------------------------------------------------------
# _score_item — the plumbing that reads a specialist's mask/bbox key and
# compares it to ground truth, without needing a live planner/registry.
# ---------------------------------------------------------------------------

def _make_output(tool_name: str, result: dict):
    from controller.schemas import SpecialistOutput
    return SpecialistOutput(tool_name=tool_name, result=result)


class TestScoreItem:
    def test_grounding_mask_scored_against_ground_truth(self, tmp_path):
        pred_arr = np.array([[255, 0], [0, 0]], dtype=np.uint8)
        gt_arr = np.array([[255, 0], [255, 0]], dtype=np.uint8)
        pred_path, gt_path = tmp_path / "pred.png", tmp_path / "gt.png"
        Image.fromarray(pred_arr, mode="L").save(pred_path)
        Image.fromarray(gt_arr, mode="L").save(gt_path)

        outputs = [_make_output("grounding", {"mask_path": str(pred_path), "bbox": [0, 0, 1, 1]})]
        expected = {"tool": "grounding", "gt_mask_path": str(gt_path), "gt_bbox": [0, 0, 1, 1]}

        scores = _score_item(outputs, expected)
        assert scores["iou"] == pytest.approx(1 / 2)  # intersection=1, union=2
        assert scores["bbox_iou"] == 1.0

    def test_unavailable_output_scores_nothing(self):
        outputs = [_make_output("grounding", {"analysis_unavailable": True, "reason": "x"})]
        expected = {"tool": "grounding", "gt_mask_path": "/does/not/matter.png"}
        assert _score_item(outputs, expected) == {}

    def test_missing_tool_in_outputs_scores_nothing(self):
        outputs = [_make_output("vqa_caption", {"answer": "hi"})]
        expected = {"tool": "grounding", "gt_mask_path": "/x.png"}
        assert _score_item(outputs, expected) == {}

    def test_no_ground_truth_scores_nothing(self):
        outputs = [_make_output("grounding", {"mask_path": "/x.png", "bbox": [0, 0, 1, 1]})]
        assert _score_item(outputs, {"tool": "grounding"}) == {}


# ---------------------------------------------------------------------------
# End-to-end: run_evaluation / run_mission_evaluation through a real planner,
# same convention as tests/test_benchmark.py's _real_image_dataset.
# ---------------------------------------------------------------------------

def _grounding_dataset(tmp_path) -> str:
    """One real image, scored against its own (deterministic) grounding mask.

    Classical spectral thresholding is deterministic given the same pixels
    and prompt, so a mask produced once and reused as ground truth must
    match a second run exactly — a real plumbing check, not a fabricated
    "IoU is always 1.0" stub. See evaluation/README.md for why this repo
    can't ship a genuinely independent ground-truth dataset.
    """
    from interfaces.api.deps import get_registry

    arr = np.random.default_rng(3).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    img_path = tmp_path / "scene.png"
    Image.fromarray(arr, "RGB").save(img_path)

    gt_result = get_registry().call_tool(
        "grounding", image_path=str(img_path), text_prompt="water"
    )

    rows = [{
        "query": {"text": "segment water", "images": [
            {"filename": "scene.png", "path": str(img_path), "modality": "optical"}
        ]},
        "expected": {"tool": "grounding", "gt_mask_path": gt_result["mask_path"]},
    }]
    out = tmp_path / "eval.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(out)


@pytest.mark.asyncio
async def test_run_evaluation_scores_deterministic_grounding(tmp_path):
    from evaluation.runner import run_evaluation
    from interfaces.api.deps import get_planner

    dataset = _grounding_dataset(tmp_path)
    report = await run_evaluation(dataset, get_planner())

    assert report["total"] == 1
    stats = report["by_tool"]["grounding"]
    assert stats["n"] == 1
    assert stats["mean_iou"] == pytest.approx(1.0)
    assert stats["mean_f1"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_run_mission_evaluation_averages_mission_metrics(tmp_path):
    from evaluation.runner import run_mission_evaluation
    from interfaces.api.deps import get_planner

    arr = np.random.default_rng(4).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    img_path = tmp_path / "flood_scene.png"
    Image.fromarray(arr, "RGB").save(img_path)

    rows = [{
        "mission": "flood",
        "query": {"text": "", "images": [
            {"filename": "flood_scene.png", "path": str(img_path), "modality": "optical"}
        ]},
    }]
    out = tmp_path / "missions.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    report = await run_mission_evaluation(str(out), get_planner())

    assert "flood" in report["missions"]
    assert report["missions"]["flood"]["n_items"] == 1
    assert isinstance(report["missions"]["flood"]["mean_metrics"], dict)
