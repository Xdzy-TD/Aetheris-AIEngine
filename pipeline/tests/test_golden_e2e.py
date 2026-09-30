"""
Tests for M18 — Golden E2E test.

Pins the full pipeline's behaviour for the canonical anonymous, no-image
query end to end: which modules run, in what order, with what statuses,
and the exact refusal text (per docs/STATUS.md's "no image -> refused"
contract). A change to routing, module ordering, or contract shape should
break this test, not just the individual module tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.schemas.contracts import PipelineRequest, ResultStatus
from pipeline.services.pipeline import AetherisPipeline

GOLDEN_ANSWER = "Analysis unavailable — 'vqa_caption' needs 1 readable image(s); 0 were provided."
GOLDEN_MODULE_ORDER = [
    "m04_auth", "m05_rate_limit", "m07_session_isolation",
    "m09_geospatial", "m10_radiometric", "m11_change_preprocessing",
    "m02_policy", "m03_dag", "m12_geotiff_artifacts", "m14_provenance",
    "m19_evidence_graph", "m20_counterfactual",
]


class TestGoldenE2E:
    """No image, anonymous user, demo router — the smallest real query."""

    def test_golden_no_image_query(self):
        result = AetherisPipeline().execute(PipelineRequest(question="What is visible?"))

        assert list(result.modules.keys()) == GOLDEN_MODULE_ORDER
        assert result.final_answer == GOLDEN_ANSWER
        assert result.status == ResultStatus.PARTIAL  # m02_policy fails the imageless plan

        skipped = {
            "m09_geospatial", "m10_radiometric", "m11_change_preprocessing",
            "m12_geotiff_artifacts", "m19_evidence_graph", "m20_counterfactual",
        }
        for name in skipped:
            assert result.modules[name].status == ResultStatus.SKIPPED

        assert result.modules["m04_auth"].status == ResultStatus.SUCCESS
        assert result.modules["m03_dag"].status == ResultStatus.SUCCESS  # planner ran, just found no image
        assert result.modules["m02_policy"].status == ResultStatus.FAILURE
