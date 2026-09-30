"""
Tests for specialist modules.

Verifies that each specialist's ``run()`` returns the expected schema
shape in demo mode (no model weights required).
"""

from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# A fixed filename here made every "two-image" test (fusion, change
# detection) secretly compare an image against itself: both calls wrote to
# and then read back the same path. Each call now gets a unique path.
_test_img_counter = itertools.count()


def _create_test_image(w: int = 64, h: int = 64) -> str:
    """Create a tiny test image at a unique path and return it."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = Path(tempfile.gettempdir()) / f"aetheris_test_img_{next(_test_img_counter)}.png"
    img.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# VQA / Caption
# ---------------------------------------------------------------------------

class TestVQACaptionSpecialist:
    """Tests for VQACaptionSpecialist in demo mode."""

    def test_demo_returns_answer(self) -> None:
        from specialists.vqa_caption.model import VQACaptionSpecialist

        spec = VQACaptionSpecialist(demo_mode=True)
        img_path = _create_test_image()
        result = spec.run(image_path=img_path, question="What is in this image?")

        assert "answer" in result
        assert "raw_confidence" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert 0 <= result["raw_confidence"] <= 1

    def test_demo_water_query(self) -> None:
        from specialists.vqa_caption.model import VQACaptionSpecialist

        spec = VQACaptionSpecialist(demo_mode=True)
        img_path = _create_test_image()
        result = spec.run(image_path=img_path, question="Are there water bodies?")

        assert "water" in result["answer"].lower()

    def test_demo_caption_mode(self) -> None:
        from specialists.vqa_caption.model import VQACaptionSpecialist

        spec = VQACaptionSpecialist(demo_mode=True)
        img_path = _create_test_image()
        result = spec.run(image_path=img_path, question="caption")

        assert "answer" in result


# ---------------------------------------------------------------------------
# VLM VQA / Caption (real pretrained model, optional dependency)
# ---------------------------------------------------------------------------

class TestVLMCaptionSpecialist:
    """Tests for VLMCaptionSpecialist. No real weights are downloaded here."""

    def test_refuses_when_dependency_missing(self, monkeypatch) -> None:
        """No torch/transformers installed -> analysis_unavailable, not a fake answer."""
        from specialists.vlm_caption.model import VLMCaptionSpecialist

        spec = VLMCaptionSpecialist()

        def _boom() -> None:
            raise ImportError("No module named 'transformers'")

        monkeypatch.setattr(spec, "load_model", _boom)
        result = spec.run(image_path=_create_test_image(), question="What is this?")

        assert result["analysis_unavailable"] is True
        assert result["raw_confidence"] == 0.0

    def test_mocked_model_returns_measured_confidence(self, monkeypatch) -> None:
        """Confidence must come from the model's own scores, not be invented."""
        torch = pytest.importorskip("torch")

        from specialists.vlm_caption.model import VLMCaptionSpecialist

        spec = VLMCaptionSpecialist()
        monkeypatch.setattr(spec, "load_model", lambda: None)
        spec._model_loaded = True  # skip ensure_loaded's real load_model call

        class _FakeProcessor:
            def __call__(self, image, prompt, return_tensors):
                return {}

            def decode(self, ids, skip_special_tokens):
                return "a runway"

        class _FakeOutput:
            sequences = torch.tensor([[0, 1]])
            scores = [torch.tensor([[1.0, 5.0, 0.1]]), torch.tensor([[0.1, 0.1, 4.0]])]

        class _FakeModel:
            def generate(self, **kw):
                return _FakeOutput()

        spec._processor = _FakeProcessor()
        spec._model = _FakeModel()

        result = spec.run(image_path=_create_test_image(), question="What is this?")

        assert result["answer"] == "a runway"
        assert 0 <= result["raw_confidence"] <= 1


# ---------------------------------------------------------------------------
# VLM Grounding (real pretrained model, optional dependency)
# ---------------------------------------------------------------------------

class TestVLMGroundingSpecialist:
    """Tests for VLMGroundingSpecialist. No real weights are downloaded here."""

    def test_refuses_when_dependency_missing(self, monkeypatch) -> None:
        """No torch/transformers installed -> analysis_unavailable, not a fake mask."""
        from specialists.vlm_grounding.model import VLMGroundingSpecialist

        spec = VLMGroundingSpecialist()

        def _boom() -> None:
            raise ImportError("No module named 'transformers'")

        monkeypatch.setattr(spec, "load_model", _boom)
        result = spec.run(image_path=_create_test_image(), text_prompt="water")

        assert result["analysis_unavailable"] is True
        assert result["confidence"] == 0.0
        assert result["mask_area_px"] == 0

    def test_mocked_model_returns_measured_confidence(self, monkeypatch) -> None:
        """Confidence must come from the model's own sigmoid output, not be invented."""
        torch = pytest.importorskip("torch")

        from specialists.vlm_grounding.model import VLMGroundingSpecialist

        spec = VLMGroundingSpecialist()
        spec._model_loaded = True  # skip ensure_loaded's real load_model call

        class _FakeProcessor:
            def __call__(self, text, images, return_tensors):
                return {}

        class _FakeOutput:
            logits = torch.zeros((1, 4, 4))

        class _FakeModel:
            def __call__(self, **kw):
                return _FakeOutput()

        spec._processor = _FakeProcessor()
        spec._model = _FakeModel()

        result = spec.run(image_path=_create_test_image(), text_prompt="water")

        assert "mask_path" in result
        assert 0 <= result["confidence"] <= 1


# ---------------------------------------------------------------------------
# VLM Change Detection (real pretrained model, optional dependency)
# ---------------------------------------------------------------------------

class TestVLMChangeDetectionSpecialist:
    """Tests for VLMChangeDetectionSpecialist. No real weights are downloaded here."""

    def test_refuses_when_dependency_missing(self, monkeypatch) -> None:
        """No torch/transformers installed -> analysis_unavailable, not a fake change map."""
        from specialists.vlm_change_detection.model import VLMChangeDetectionSpecialist

        spec = VLMChangeDetectionSpecialist()

        def _boom() -> None:
            raise ImportError("No module named 'transformers'")

        monkeypatch.setattr(spec, "load_model", _boom)
        result = spec.run(
            image_t1_path=_create_test_image(), image_t2_path=_create_test_image()
        )

        assert result["analysis_unavailable"] is True
        assert result["change_embedding"] == []

    def test_mocked_model_returns_patch_distance(self, monkeypatch) -> None:
        """Change embedding must come from real patch-embedding distances, not be invented."""
        torch = pytest.importorskip("torch")

        from specialists.vlm_change_detection.model import VLMChangeDetectionSpecialist

        spec = VLMChangeDetectionSpecialist()
        spec._model_loaded = True  # skip ensure_loaded's real load_model call

        class _FakeProcessor:
            def __call__(self, images, return_tensors):
                return {}

        class _FakeOutput:
            # 1 CLS token + 4 patch tokens (2x2 grid), tiny hidden dim.
            last_hidden_state = torch.rand(1, 5, 8)

        class _FakeModel:
            def __call__(self, **kw):
                return _FakeOutput()

        spec._processor = _FakeProcessor()
        spec._model = _FakeModel()

        result = spec.run(
            image_t1_path=_create_test_image(), image_t2_path=_create_test_image()
        )

        assert "change_map_path" in result
        assert len(result["change_embedding"]) == 4
        assert 0 <= result["changed_area_pct"] <= 100


# ---------------------------------------------------------------------------
# SAR Bridge
# ---------------------------------------------------------------------------

class TestSARBridgeSpecialist:
    """Tests for SARBridgeSpecialist in demo mode."""

    def test_demo_returns_bridged_path(self) -> None:
        from specialists.sar_bridge.model import SARBridgeSpecialist

        spec = SARBridgeSpecialist(demo_mode=True)
        img_path = _create_test_image()
        result = spec.run(sar_image_path=img_path, mode="both")

        assert "bridged_image_path" in result
        assert "method_used" in result


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

class TestGroundingSpecialist:
    """Tests for GroundingSpecialist in demo mode."""

    def test_demo_returns_mask(self) -> None:
        from specialists.grounding.model import GroundingSpecialist

        spec = GroundingSpecialist(demo_mode=True)
        img_path = _create_test_image()
        result = spec.run(
            image_path=img_path,
            text_prompt="water bodies",
            return_bbox=True,
        )

        assert "mask_path" in result
        assert "mask_area_px" in result
        assert "confidence" in result
        assert "bbox" in result
        assert result["mask_area_px"] > 0

    def test_demo_vegetation_query(self) -> None:
        from specialists.grounding.model import GroundingSpecialist

        spec = GroundingSpecialist(demo_mode=True)
        img_path = _create_test_image()
        result = spec.run(image_path=img_path, text_prompt="vegetation")

        assert result["mask_area_px"] > 0
        assert result["confidence"] > 0


# ---------------------------------------------------------------------------
# Change Detection
# ---------------------------------------------------------------------------

class TestChangeDetectionSpecialist:
    """Tests for ChangeDetectionSpecialist in demo mode."""

    def test_demo_returns_change_map(self) -> None:
        from specialists.change_detection.model import ChangeDetectionSpecialist

        spec = ChangeDetectionSpecialist(demo_mode=True)
        img1 = _create_test_image()
        img2 = _create_test_image()
        result = spec.run(
            image_t1_path=img1,
            image_t2_path=img2,
            modality="optical",
        )

        assert "change_map_path" in result
        assert "change_summary" in result
        assert "changed_area_pct" in result
        assert "change_embedding" in result
        assert isinstance(result["change_embedding"], list)


# ---------------------------------------------------------------------------
# Optical-SAR Fusion
# ---------------------------------------------------------------------------

class TestOpticalSARFusionSpecialist:
    """Tests for OpticalSARFusionSpecialist in demo mode."""

    def test_demo_returns_fused_stats(self) -> None:
        from specialists.optical_sar_fusion.model import OpticalSARFusionSpecialist

        spec = OpticalSARFusionSpecialist(demo_mode=True)
        optical_path = _create_test_image()
        sar_path = _create_test_image()
        result = spec.run(optical_image_path=optical_path, sar_image_path=sar_path)

        assert "optical_landcover_pct" in result
        assert "fused_water_pct" in result
        assert "fused_veg_pct" in result
        assert "fused_urban_pct" in result
        assert "agreement_pct" in result
        assert "fused_map_path" in result
        assert "applied_shift_px" in result
        assert "summary" in result
        assert 0 <= result["agreement_pct"] <= 100
        assert 0 <= result["confidence"] <= 1

    def test_missing_image_fails_gracefully(self) -> None:
        from specialists.optical_sar_fusion.model import OpticalSARFusionSpecialist

        spec = OpticalSARFusionSpecialist(demo_mode=True)
        result = spec.run(optical_image_path="", sar_image_path="")

        assert result["fused_map_path"] == ""
        assert "Could not load" in result["summary"]

    def test_two_calls_do_not_compare_the_same_image(self) -> None:
        """Regression guard for the shared-filename test-fixture bug: two
        independent calls to ``_create_test_image`` must not collide on disk."""
        optical_path = _create_test_image()
        sar_path = _create_test_image()
        assert optical_path != sar_path

    def _landcover_bands(self, h: int = 126, w: int = 126) -> tuple[np.ndarray, np.ndarray]:
        """Optical RGB + matching SAR grayscale, three exact horizontal bands
        (water/vegetation/urban) with known ground truth for coreg testing."""
        third = h // 3
        optical = np.zeros((h, w, 3), dtype=np.uint8)
        optical[:third] = [10, 10, 200]          # water: blue-dominant
        optical[third:2 * third] = [10, 200, 10]  # vegetation: green-dominant
        optical[2 * third:] = [200, 200, 200]     # urban: bright, low-saturation
        sar = np.full((h, w), 128, dtype=np.uint8)  # vegetation backscatter (mid)
        sar[:third] = 10                          # water: dark backscatter
        sar[2 * third:] = 245                     # urban: bright backscatter
        return optical, sar

    def test_coregistration_recovers_agreement_from_known_shift(self) -> None:
        """Dedicated regression test with known synthetic ground truth: a
        SAR image misaligned by a known pixel shift must still be fused with
        high agreement once co-registered, and the applied shift reported."""
        from specialists.optical_sar_fusion.model import OpticalSARFusionSpecialist

        optical, sar = self._landcover_bands()
        misaligned_sar = np.roll(sar, shift=15, axis=0)  # simulate a 15px coreg error

        tmp = Path(tempfile.gettempdir())
        optical_path = tmp / f"aetheris_coreg_opt_{next(_test_img_counter)}.png"
        sar_path = tmp / f"aetheris_coreg_sar_{next(_test_img_counter)}.png"
        Image.fromarray(optical, mode="RGB").save(str(optical_path))
        Image.fromarray(misaligned_sar, mode="L").save(str(sar_path))

        spec = OpticalSARFusionSpecialist(demo_mode=True)
        result = spec.run(optical_image_path=str(optical_path), sar_image_path=str(sar_path))

        assert result["applied_shift_px"] != [0, 0]  # misalignment was detected
        assert result["agreement_pct"] > 90.0  # and corrected, not scored as disagreement


# ---------------------------------------------------------------------------
# Fusion Change Detection
# ---------------------------------------------------------------------------

class TestFusionChangeDetectionSpecialist:
    """Tests for FusionChangeDetectionSpecialist. No fitted checkpoint is
    configured here, so every run exercises the unfitted informative prior
    (AETHERIS_FUSION_MODEL_PATH is left unset)."""

    def test_optical_only_pair_returns_expected_schema(self, monkeypatch) -> None:
        from specialists.fusion_change_detection.model import FusionChangeDetectionSpecialist

        monkeypatch.delenv("AETHERIS_FUSION_MODEL_PATH", raising=False)
        spec = FusionChangeDetectionSpecialist()
        result = spec.run(
            image_t1_path=_create_test_image(), image_t2_path=_create_test_image()
        )

        assert "change_map_path" in result
        assert "change_summary" in result
        assert 0 <= result["changed_area_pct"] <= 100
        assert isinstance(result["change_embedding"], list)
        assert result["evidence_channels_used"] == ["optical"]
        assert result["fusion_model_fitted"] is False
        assert result["applied_shift_px"]["sar"] == [0, 0]

    def test_missing_image_returns_refusal(self) -> None:
        from specialists.fusion_change_detection.model import FusionChangeDetectionSpecialist

        spec = FusionChangeDetectionSpecialist()
        result = spec.run(image_t1_path="", image_t2_path="")

        assert result["analysis_unavailable"] is True
        assert result["change_embedding"] == []
        assert result["evidence_channels_used"] == []

    def test_mismatched_extent_is_refused_not_differenced(self) -> None:
        """Same guard as change_detection: comparing unmatched extents would
        report the clipping mismatch itself as 'change'."""
        from specialists.fusion_change_detection.model import FusionChangeDetectionSpecialist

        spec = FusionChangeDetectionSpecialist()
        result = spec.run(
            image_t1_path=_create_test_image(64, 64),
            image_t2_path=_create_test_image(64, 32),
        )

        assert result["analysis_unavailable"] is True
        assert "unmatched pair" in result["reason"]

    def test_sar_pair_adds_sar_evidence_channel(self) -> None:
        from specialists.fusion_change_detection.model import FusionChangeDetectionSpecialist

        spec = FusionChangeDetectionSpecialist()
        result = spec.run(
            image_t1_path=_create_test_image(),
            image_t2_path=_create_test_image(),
            sar_image_t1_path=_create_test_image(),
            sar_image_t2_path=_create_test_image(),
        )

        assert result["evidence_channels_used"] == ["optical", "sar"]
        assert "sar" in result["change_summary"]

    def test_dates_add_temporal_evidence_channel(self) -> None:
        from specialists.fusion_change_detection.model import FusionChangeDetectionSpecialist

        spec = FusionChangeDetectionSpecialist()
        result = spec.run(
            image_t1_path=_create_test_image(),
            image_t2_path=_create_test_image(),
            date_t1="2023-01-01T00:00:00",
            date_t2="2023-07-01T00:00:00",
        )

        assert result["evidence_channels_used"] == ["optical", "temporal"]
        assert "day gap" in result["change_summary"]

    def test_unparseable_dates_are_dropped_not_fatal(self) -> None:
        """A date that fails to parse should fall back to neutral temporal
        evidence, not blow up the whole request."""
        from specialists.fusion_change_detection.model import FusionChangeDetectionSpecialist

        spec = FusionChangeDetectionSpecialist()
        result = spec.run(
            image_t1_path=_create_test_image(),
            image_t2_path=_create_test_image(),
            date_t1="not-a-date",
            date_t2="also-not-a-date",
        )

        assert result["evidence_channels_used"] == ["optical"]

    def test_unfitted_model_never_biases_missing_channels_toward_no_change(self) -> None:
        """A missing SAR/temporal channel is passed as neutral (0.5), not
        zeroed -- zeroing would pull every fused probability down regardless
        of what the optical evidence actually shows."""
        from specialists._fusion_model import EvidenceFusionModel

        model = EvidenceFusionModel()
        with_evidence = model.predict_proba(np.array([[0.9, 0.5, 0.5]]))[0]
        zeroed = model.predict_proba(np.array([[0.9, 0.0, 0.0]]))[0]

        assert with_evidence > zeroed

    def test_loads_fitted_checkpoint_when_configured(self, monkeypatch) -> None:
        """fusion_model_fitted must reflect a real loaded checkpoint, not
        just be hardcoded True/False."""
        from specialists._fusion_model import EvidenceFusionModel
        from specialists.fusion_change_detection.model import FusionChangeDetectionSpecialist

        fitted = EvidenceFusionModel()
        fitted.fit(
            X=np.array([[0.9, 0.5, 0.5], [0.1, 0.5, 0.5]]), y=np.array([1.0, 0.0]), epochs=200
        )
        ckpt_name = f"aetheris_test_fusion_ckpt_{next(_test_img_counter)}.json"
        path = Path(tempfile.gettempdir()) / ckpt_name
        fitted.save(path)

        spec = FusionChangeDetectionSpecialist(fusion_model_path=str(path))
        result = spec.run(
            image_t1_path=_create_test_image(), image_t2_path=_create_test_image()
        )

        assert result["fusion_model_fitted"] is True


# ---------------------------------------------------------------------------
# Shared imagery helpers
# ---------------------------------------------------------------------------

class TestImageryCoregistration:
    """Tests for the FFT phase-correlation shift estimator used by both
    ``change_detection`` and ``optical_sar_fusion``."""

    def test_estimate_shift_recovers_known_synthetic_shift(self) -> None:
        from specialists._imagery import estimate_shift

        rng = np.random.default_rng(0)
        ref = rng.random((100, 100))
        tgt = np.roll(ref, (15, -8), axis=(0, 1))

        dy, dx = estimate_shift(ref, tgt)
        aligned = np.roll(tgt, (dy, dx), axis=(0, 1))

        assert np.allclose(aligned, ref)

    def test_sar_landcover_masks_partitions_three_classes(self) -> None:
        from specialists._imagery import sar_landcover_masks

        _, sar = TestOpticalSARFusionSpecialist()._landcover_bands()
        gray = sar.astype(np.float64) / 255.0
        water, veg, urban = sar_landcover_masks(gray)

        assert water[: len(gray) // 3].all()
        assert urban[2 * (len(gray) // 3) :].all()
        assert veg[len(gray) // 3 : 2 * (len(gray) // 3)].all()
        # Every pixel belongs to exactly one class.
        assert np.array_equal(water | veg | urban, np.ones_like(water))
        assert not (water & veg).any()
        assert not (veg & urban).any()


# ---------------------------------------------------------------------------
# Geo RAG
# ---------------------------------------------------------------------------

class TestGeoRAGSpecialist:
    """Tests for GeoRAGSpecialist in demo mode."""

    def test_empty_index_returns_honest_empty_result(self) -> None:
        """With no data indexed, geo_rag must return empty results plus an
        explanatory note — never fabricated OSM/DEM/land-cover numbers."""
        from specialists.geo_rag.model import GeoRAGSpecialist

        spec = GeoRAGSpecialist(demo_mode=True)
        result = spec.run(
            bbox=[77.5, 12.9, 77.6, 13.0],
            query_text="land cover",
            data_layers=["osm", "dem", "landcover"],
        )

        assert result["context_snippets"] == []
        assert result["elevation_stats"] == {}
        assert result["landcover_distribution"] == {}
        assert "note" in result
        assert "no" in result["note"].lower() or "connect" in result["note"].lower()

    def test_indexed_data_returns_real_context(self) -> None:
        """Once documents are indexed, geo_rag must retrieve and return them
        instead of falling back to the empty-index path."""
        from specialists.geo_rag.indexer import GeoIndex
        from specialists.geo_rag.model import GeoRAGSpecialist

        spec = GeoRAGSpecialist(demo_mode=True)
        spec.ensure_loaded()  # not load_model() directly — that would be
        # re-invoked (and _index rebuilt) by run()'s own ensure_loaded() call
        assert isinstance(spec._index, GeoIndex)
        bbox = [77.5, 12.9, 77.6, 13.0]
        spec._index.add_elevation_stats({"min": 850, "max": 920, "mean": 885, "std": 12}, bbox)
        spec._index.add_landcover({"cropland": 0.62, "urban": 0.18}, bbox)

        result = spec.run(
            bbox=[77.5, 12.9, 77.6, 13.0],
            query_text="land cover",
            data_layers=["osm", "dem", "landcover"],
        )

        assert len(result["context_snippets"]) > 0
        assert result["elevation_stats"].get("min") == 850
        assert result["landcover_distribution"].get("cropland") == 0.62
        assert "note" not in result


# ---------------------------------------------------------------------------
# Confidence Layer
# ---------------------------------------------------------------------------

class TestConformalCalibrator:
    """Tests for ConformalCalibrator."""

    def test_fit_and_predict(self) -> None:
        from confidence.conformal import ConformalCalibrator

        cal = ConformalCalibrator(target_coverage=0.90)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        labels = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        cal.fit(scores, labels)

        result = cal.predict(0.75)
        assert result["method"] == "conformal"
        assert 0 <= result["lower"] <= result["upper"] <= 1

    def test_uncalibrated_fallback(self) -> None:
        from confidence.conformal import ConformalCalibrator

        cal = ConformalCalibrator()
        result = cal.predict(0.8)
        assert result["method"] == "uncalibrated"


class TestSelfConsistencyVoter:
    """Tests for SelfConsistencyVoter."""

    def test_deterministic_specialist(self) -> None:
        from confidence.self_consistency import SelfConsistencyVoter

        voter = SelfConsistencyVoter(n_paths=5)

        def stub_specialist(**kw):
            return {"answer": "always the same"}

        result = voter.vote(stub_specialist)
        assert result["agreement_ratio"] == 1.0
        assert result["best_answer"] == "always the same"
        assert result["n_consistent"] == 5
