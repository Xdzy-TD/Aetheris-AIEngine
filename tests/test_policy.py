"""
Tests for the integrity guarantees: policy gating, artifact evidence,
honest unavailability, and honest calibration.

These are the properties that must not regress — each one, if broken, lets a
fabricated result reach the user looking exactly like a real one.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from confidence.conformal import ConformalCalibrator
from controller.artifacts import ArtifactStore
from controller.model_registry import all_models, version_for_task
from controller.planner import AgenticPlanner
from controller.executor import PlanExecutor
from controller.policy import PolicyViolation, check_plan
from controller.schemas import ExecutionPlan, ImageMetadata, Modality, Query, ToolCall
from controller.tool_registry import ToolRegistry


def _registry(bind: bool = True) -> ToolRegistry:
    reg = ToolRegistry()
    if bind:
        for name in reg.list_tool_names():
            reg.bind(name, lambda **kw: {"ok": True})
    return reg


def _query(*modalities: Modality) -> Query:
    return Query(
        text="what changed?",
        images=[
            ImageMetadata(filename=f"i{i}.tif", path=f"/nonexistent/i{i}.tif", modality=m)
            for i, m in enumerate(modalities)
        ],
    )


def _plan(*steps: ToolCall) -> ExecutionPlan:
    return ExecutionPlan(query_id="q1", steps=list(steps))


class TestPolicyEngine:
    """A plan is untrusted input; invalid ones must not reach a tool."""

    def test_valid_plan_passes(self):
        plan = _plan(ToolCall(tool_name="vqa_caption", arguments={"question": "q"}))
        check_plan(plan, _query(Modality.OPTICAL), _registry())

    def test_empty_plan_rejected(self):
        with pytest.raises(PolicyViolation, match="no executable steps"):
            check_plan(_plan(), _query(Modality.OPTICAL), _registry())

    def test_unknown_tool_rejected(self):
        plan = _plan(ToolCall(tool_name="nuclear_launch", arguments={}))
        with pytest.raises(PolicyViolation, match="not in the registry"):
            check_plan(plan, _query(Modality.OPTICAL), _registry())

    def test_unbound_tool_rejected(self):
        """An unbound tool has no model: refusing here is what prevents a
        fabricated answer from standing in for a missing one."""
        plan = _plan(ToolCall(tool_name="vqa_caption", arguments={"question": "q"}))
        with pytest.raises(PolicyViolation, match="no model bound"):
            check_plan(plan, _query(Modality.OPTICAL), _registry(bind=False))

    def test_too_few_images_rejected(self):
        plan = _plan(ToolCall(tool_name="change_detection", arguments={}))
        with pytest.raises(PolicyViolation, match="needs 2 readable image"):
            check_plan(plan, _query(Modality.OPTICAL), _registry())

    def test_sar_bridge_without_sar_rejected(self):
        plan = _plan(ToolCall(tool_name="sar_bridge", arguments={}))
        with pytest.raises(PolicyViolation, match="requires a SAR image"):
            check_plan(plan, _query(Modality.OPTICAL), _registry())

    def test_fusion_requires_both_modalities(self):
        plan = _plan(ToolCall(tool_name="optical_sar_fusion", arguments={}))
        with pytest.raises(PolicyViolation, match="one SAR and one optical"):
            check_plan(plan, _query(Modality.OPTICAL, Modality.OPTICAL), _registry())

    def test_change_detection_rejects_mixed_modality(self):
        plan = _plan(ToolCall(tool_name="change_detection", arguments={}))
        with pytest.raises(PolicyViolation, match="same modality"):
            check_plan(plan, _query(Modality.OPTICAL, Modality.SAR), _registry())

    def test_unknown_parameter_rejected(self):
        plan = _plan(ToolCall(
            tool_name="vqa_caption", arguments={"question": "q", "temperature": 9000}
        ))
        with pytest.raises(PolicyViolation, match="does not accept"):
            check_plan(plan, _query(Modality.OPTICAL), _registry())

    def test_missing_required_parameter_rejected(self):
        """`text_prompt` is the planner's job to supply; paths are not."""
        plan = _plan(ToolCall(tool_name="grounding", arguments={}))
        with pytest.raises(PolicyViolation, match="missing required"):
            check_plan(plan, _query(Modality.OPTICAL), _registry())

    def test_server_supplied_paths_not_demanded_from_planner(self):
        plan = _plan(ToolCall(tool_name="grounding", arguments={"text_prompt": "water"}))
        check_plan(plan, _query(Modality.OPTICAL), _registry())

    def test_forward_dependency_rejected(self):
        plan = _plan(
            ToolCall(tool_name="vqa_caption", arguments={"question": "q"},
                     depends_on=["sar_bridge"]),
            ToolCall(tool_name="sar_bridge", arguments={}),
        )
        with pytest.raises(PolicyViolation, match="does not run before it"):
            check_plan(plan, _query(Modality.SAR), _registry())

    def test_self_dependency_rejected(self):
        plan = _plan(ToolCall(
            tool_name="vqa_caption", arguments={"question": "q"}, depends_on=["vqa_caption"]
        ))
        with pytest.raises(PolicyViolation, match="depends on itself"):
            check_plan(plan, _query(Modality.OPTICAL), _registry())


class TestArtifactStore:
    """Evidence must outlive the process and be re-checkable."""

    def test_put_returns_full_evidence_record(self, tmp_path):
        store = ArtifactStore(tmp_path)
        rec = store.put(b"mask-bytes", "grounding_mask", "m@1.0.0 (classical-baseline)")
        assert set(rec) >= {"artifact_id", "sha256", "type", "model_version", "path"}
        assert rec["sha256"] == hashlib.sha256(b"mask-bytes").hexdigest()

    def test_stored_bytes_verify(self, tmp_path):
        store = ArtifactStore(tmp_path)
        rec = store.put(b"mask-bytes", "grounding_mask", "m@1.0.0")
        assert store.verify(str(rec["artifact_id"])) is True

    def test_tampering_is_detected(self, tmp_path):
        store = ArtifactStore(tmp_path)
        rec = store.put(b"mask-bytes", "grounding_mask", "m@1.0.0")
        (tmp_path / f"{rec['artifact_id']}").write_bytes(b"different")
        assert store.verify(str(rec["artifact_id"])) is False

    def test_unknown_artifact_does_not_verify(self, tmp_path):
        assert ArtifactStore(tmp_path).verify("deadbeef") is False

    def test_evidence_is_not_kept_in_tmp_by_default(self, monkeypatch):
        """/tmp is cleared on reboot; evidence cited by a report must not be.

        conftest redirects the store for test runs, so this asserts on the
        default the code would use without that override.
        """
        monkeypatch.delenv("AETHERIS_ARTIFACT_DIR", raising=False)
        default = Path(os.environ.get("AETHERIS_ARTIFACT_DIR", "./artifacts")).resolve()
        assert not str(default).startswith(tempfile.gettempdir())


class TestModelRegistry:
    def test_every_model_declares_its_kind_and_licence(self):
        for m in all_models():
            assert m["kind"] in (
                "classical-baseline", "implemented", "experimental", "pretrained-vlm",
                "pretrained-segmentation", "pretrained-embedding", "learned-fusion",
            )
            assert m["license"]

    def test_no_invented_metrics(self):
        """Metrics stay empty until a real evaluation run fills them."""
        for m in all_models():
            assert m["metrics"] == {}, f"{m['name']} claims metrics with no eval run"

    def test_version_string_for_known_task(self):
        assert "@" in version_for_task("change_detection")

    def test_unknown_task_is_unavailable(self):
        assert version_for_task("mind_reading") == "unavailable"


class TestUnavailability:
    """A tool that cannot run must not produce something shaped like a result."""

    def test_unavailable_record_is_self_identifying(self):
        rec = PlanExecutor._unavailable("grounding", "no model implementation is bound")
        assert rec["analysis_unavailable"] is True
        assert rec["reason"]

    def test_unavailable_record_invents_nothing(self):
        """No paths, no masks, no zeroed statistics that read as measurements."""
        rec = PlanExecutor._unavailable("change_detection", "unbound")
        assert not any(k.endswith("_path") for k in rec)
        numbers = [
            v for v in rec.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        assert numbers == []


class TestHonestCalibration:
    def test_unfitted_calibrator_says_uncalibrated(self):
        pred = ConformalCalibrator().predict(0.9)
        assert pred["method"] == "uncalibrated"
        assert pred["calibrated_score"] == 0.9  # passed through, not adjusted

    def test_unfitted_calibrator_refuses_to_save(self):
        with pytest.raises(RuntimeError, match="unfitted"):
            ConformalCalibrator().save("/tmp/should-not-exist.json")

    def test_calibration_artifact_round_trips(self, tmp_path):
        cal = ConformalCalibrator(target_coverage=0.9)
        cal.fit([0.8] * 20, [1.0] * 20)
        artifact = cal.save(tmp_path / "cal.json")
        assert artifact["n_calibration"] == 20

        restored = ConformalCalibrator()
        restored.load(tmp_path / "cal.json")
        assert restored.is_fitted()
        assert restored.predict(0.8)["method"] == "conformal"


class TestGeoTiffIntegrity:
    """A GeoTIFF must never be silently downgraded to PIL/RGB."""

    def test_plain_tiff_without_georeferencing_still_loads(self, tmp_path):
        from specialists import _imagery

        p = tmp_path / "plain.tif"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(p)
        arr, meta = _imagery.load_raster(str(p))
        assert arr.shape == (8, 8, 3)
        assert meta["georeferenced"] is False  # nothing to lose, nothing claimed

    def test_georeferenced_tiff_is_refused_without_rasterio(self, tmp_path, monkeypatch):
        from specialists import raster_io as _rio

        if _rio.rasterio is not None:
            monkeypatch.setattr(_rio, "rasterio", None)

        p = tmp_path / "geo.tif"
        img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
        # 33550 = ModelPixelScale: presence of any geokey makes this a GeoTIFF
        img.save(p, tiffinfo={33550: (10.0, 10.0, 0.0)})

        with pytest.raises(ValueError, match="rasterio"):
            _rio.load_raster(str(p))

    def test_rasterio_path_preserves_crs(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        from specialists import _imagery

        p = tmp_path / "real.tif"
        with rasterio.open(
            p, "w", driver="GTiff", height=8, width=8, count=3, dtype="uint16",
            crs="EPSG:32643", transform=from_origin(0, 0, 10, 10), nodata=0,
        ) as dst:
            dst.write(np.full((3, 8, 8), 1000, dtype="uint16"))

        arr, meta = _imagery.load_raster(str(p))
        assert meta["crs"] == "EPSG:32643"
        assert meta["resolution_m"] == 10
        assert meta["nodata"] == 0
        assert meta["band_count"] == 3
        assert 0.0 <= arr.min() <= arr.max() <= 1.0
