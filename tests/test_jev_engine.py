"""Tests for the JEV decision engine and planner."""

from __future__ import annotations

import pytest

from controller.jev_engine import (
    EvidenceState,
    JEVDecision,
    JEVEngine,
    JEVQuestionType,
    QueryState,
)


@pytest.fixture
def engine() -> JEVEngine:
    return JEVEngine()


def _state(
    text: str = "",
    modalities: list[str] | None = None,
    tools: list[str] | None = None,
) -> QueryState:
    """Build a QueryState without a real Query object."""
    mods = modalities or []
    tools = tools or [
        "vqa_caption", "sar_bridge", "grounding",
        "change_detection", "geo_rag", "optical_sar_fusion",
        "fusion_change_detection",
    ]
    lower = text.lower()
    has_sar = "sar" in mods
    has_opt = any(m in ("optical", "multispectral") for m in mods)
    n = len(mods)
    is_cross = n == 2 and has_sar and has_opt and mods[0] != mods[1]
    _kw = lambda *ws: any(w in lower for w in ws)
    return QueryState(
        text_lower=lower, n_images=n, has_sar=has_sar, has_optical=has_opt,
        is_cross_modal_pair=is_cross, is_bitemporal_pair=(n >= 2 and not is_cross),
        available_tools=tools,
        mentions_change=_kw("change", "differ", "before", "after"),
        mentions_location=_kw("locat", "segment", "mask", "where"),
        mentions_sar=_kw("sar", "radar"),
        mentions_elevation=_kw("elevat", "terrain"),
    )


# --- Routing ---

class TestRouting:
    def test_optical_routes_to_vqa(self, engine):
        d = engine.decide_route(_state("What is here?", ["optical"]))
        assert "vqa_caption" in d.answer
        assert d.elapsed_ms < 100

    def test_sar_routes_through_bridge(self, engine):
        d = engine.decide_route(_state("Analyze radar", ["sar"]))
        assert "sar_bridge" in d.answer and "vqa_caption" in d.answer

    def test_cross_modal_routes_to_fusion(self, engine):
        d = engine.decide_route(_state("Compare", ["optical", "sar"]))
        assert "optical_sar_fusion" in d.answer
        assert "sar_bridge" not in d.answer

    def test_bitemporal_change(self, engine):
        d = engine.decide_route(_state("What changed?", ["optical", "optical"]))
        assert any(t in d.answer for t in ("change_detection", "fusion_change_detection"))

    def test_grounding(self, engine):
        d = engine.decide_route(_state("Segment water bodies", ["optical"]))
        assert "grounding" in d.answer

    def test_geo_rag(self, engine):
        d = engine.decide_route(_state("What is the terrain?", ["optical"]))
        assert "geo_rag" in d.answer

    def test_vqa_always_last(self, engine):
        tools = engine.decide_route(_state("SAR radar analysis", ["sar"])).answer
        if "vqa_caption" in tools:
            assert tools[-1] == "vqa_caption"

    def test_deterministic(self, engine):
        s = _state("Detect flooding", ["optical"])
        assert engine.decide_route(s).answer == engine.decide_route(s).answer

    def test_sub_10ms(self, engine):
        d = engine.decide_route(_state("Complex query", ["optical", "sar"]))
        assert d.elapsed_ms < 10


# --- Evidence scoring ---

class TestScoring:
    def test_high_confidence(self, engine):
        e = [EvidenceState(tool_name="vqa", confidence=0.9, has_artifact=True, result_keys=["a", "b", "c"])]
        assert engine.score_evidence(e).probability > 0.5

    def test_unavailable(self, engine):
        assert engine.score_evidence([EvidenceState(analysis_unavailable=True)]).probability == 0.0

    def test_empty(self, engine):
        assert engine.score_evidence([]).probability == 0.0

    def test_bad_coreg_penalised(self, engine):
        good = engine.score_evidence([EvidenceState(confidence=0.7, coregistration_reliable=True)]).probability
        bad = engine.score_evidence([EvidenceState(confidence=0.7, coregistration_reliable=False)]).probability
        assert good > bad


# --- Verification ---

class TestVerification:
    def test_consistent(self, engine):
        d = engine.verify_consistency(
            [{"tool_name": "vqa_caption", "result": {"answer": "flood", "confidence": 0.8}}],
            "Flooding detected",
        )
        assert d.question_type == JEVQuestionType.VERIFY

    def test_empty_returns_zero(self, engine):
        assert engine.verify_consistency([], "answer").probability == 0.0
        assert engine.verify_consistency([{"tool_name": "x", "result": {}}], "").probability == 0.0


# --- Guards ---

class TestGuards:
    def test_valid_plan(self, engine):
        d = engine.check_guard(_state("test", ["optical"]), ["vqa_caption"])
        assert d.answer is True and d.probability > 0.9

    def test_unknown_tool_fails(self, engine):
        d = engine.check_guard(_state("test"), ["nonexistent_tool"])
        assert d.answer is False

    def test_pixel_tool_no_image_fails(self, engine):
        assert engine.check_guard(_state("test"), ["vqa_caption"]).answer is False

    def test_bitemporal_one_image_fails(self, engine):
        assert engine.check_guard(_state("test", ["optical"]), ["change_detection"]).answer is False

    def test_fusion_wrong_modalities_fails(self, engine):
        assert engine.check_guard(_state("test", ["optical", "optical"]),
                                   ["optical_sar_fusion"]).answer is False

    def test_no_terminal_warns(self, engine):
        d = engine.check_guard(_state("test", ["optical"]), ["grounding"])
        assert d.answer is True and d.probability < 0.95


# --- JEV Planner ---

class TestJEVPlanner:
    @pytest.fixture
    def planner(self):
        from controller.jev_planner import JEVPlanner
        return JEVPlanner()

    @pytest.mark.asyncio
    async def test_plan_returns_plan(self, planner):
        from controller.schemas import Query, ExecutionPlan
        plan = await planner.plan(Query(text="What is visible?"), [{"name": "vqa_caption"}, {"name": "grounding"}])
        assert isinstance(plan, ExecutionPlan)
        assert plan.planning_method == "jev"

    @pytest.mark.asyncio
    async def test_always_available(self, planner):
        assert await planner.is_available() is True

    @pytest.mark.asyncio
    async def test_no_llm_synthesis_returns_none(self, planner):
        from controller.schemas import Query
        assert await planner.synthesize(Query(text="t"), []) is None

    @pytest.mark.asyncio
    async def test_stats(self, planner):
        from controller.schemas import Query
        schemas = [{"name": "vqa_caption"}]
        await planner.plan(Query(text="a"), schemas)
        await planner.plan(Query(text="b"), schemas)
        assert planner.stats["plan_count"] == 2

    def test_verify(self, planner):
        r = planner.verify_outputs([{"tool_name": "vqa", "result": {"confidence": 0.8}}], "answer")
        assert "consistent" in r and "evidence_quality" in r


# --- Performance ---

class TestPerformance:
    def test_1000_routes_under_500ms(self, engine):
        import time
        s = _state("Detect flooding", ["optical", "optical"])
        t0 = time.perf_counter()
        for _ in range(1000):
            engine.decide_route(s)
        assert (time.perf_counter() - t0) * 1000 < 500

    def test_zero_network(self, engine):
        import inspect
        src = inspect.getsource(engine.__class__)
        assert "httpx" not in src and "await" not in src
