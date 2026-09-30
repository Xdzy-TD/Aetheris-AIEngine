"""
JEV Engine — fast, typed, probabilistic decision engine.

Zero-token, fully offline replacement for LLM-based routing.
Ingests typed program state, returns constrained answers with
calibrated probabilities.  Sub-millisecond, deterministic.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class JEVQuestionType(StrEnum):
    ROUTE = "route"
    SCORE = "score"
    VERIFY = "verify"
    GUARD = "guard"


@dataclass(frozen=True)
class JEVDecision:
    """Immutable, typed answer from the JEV engine."""
    question_type: JEVQuestionType
    answer: Any
    probability: float
    reasoning: str
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# State descriptors
# ---------------------------------------------------------------------------

@dataclass
class QueryState:
    """Typed descriptor of a query — the only input the engine reads."""
    text_lower: str = ""
    n_images: int = 0
    has_sar: bool = False
    has_optical: bool = False
    is_cross_modal_pair: bool = False
    is_bitemporal_pair: bool = False
    available_tools: list[str] = field(default_factory=list)
    mentions_change: bool = False
    mentions_location: bool = False
    mentions_sar: bool = False
    mentions_elevation: bool = False
    has_dates: bool = False

    @classmethod
    def from_query(cls, query: Any, available_tools: list[str]) -> QueryState:
        """Build from a controller Query + available tool names."""
        images = getattr(query, "images", [])
        mods = [getattr(img, "modality", "unknown") for img in images]
        lower = getattr(query, "text", "").lower()
        n = len(images)
        has_sar = "sar" in mods
        has_opt = any(m in ("optical", "multispectral") for m in mods)
        is_cross = n == 2 and has_sar and has_opt and mods[0] != mods[1]

        _kw = lambda *words: any(w in lower for w in words)
        return cls(
            text_lower=lower,
            n_images=n,
            has_sar=has_sar,
            has_optical=has_opt,
            is_cross_modal_pair=is_cross,
            is_bitemporal_pair=(n >= 2 and not is_cross),
            available_tools=available_tools,
            mentions_change=_kw("change", "differ", "before", "after", "compar"),
            mentions_location=_kw("locat", "segment", "mask", "where", "find",
                                  "identify", "delineat", "outline", "bound"),
            mentions_sar=_kw("sar", "radar", "sentinel-1", "backscatter"),
            mentions_elevation=_kw("elevat", "terrain", "dem", "slope", "topograph"),
            has_dates=any(getattr(img, "acquisition_date", None) for img in images),
        )


@dataclass
class EvidenceState:
    """Typed descriptor of a specialist output for scoring."""
    tool_name: str = ""
    confidence: float | None = None
    has_artifact: bool = False
    analysis_unavailable: bool = False
    coregistration_reliable: bool | None = None
    warning_count: int = 0
    result_keys: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class JEVEngine:
    """Probabilistic decision engine — sigmoid/softmax over log-odds features.

    All decisions are deterministic given the same input state.
    """

    # Per-tool log-odds weights.  Positive = evidence FOR routing to it.
    _ROUTING_WEIGHTS: dict[str, dict[str, float]] = {
        "sar_bridge": {
            "has_sar": 3.0, "has_optical": -2.0, "is_cross_modal": -3.0,
            "mentions_sar": 1.5,
        },
        "optical_sar_fusion": {
            "is_cross_modal": 4.0, "has_sar": 1.0, "has_optical": 1.0,
        },
        "change_detection": {
            "is_bitemporal": 3.0, "mentions_change": 2.5, "is_cross_modal": -3.0,
            "n_images": 0.5,
        },
        "fusion_change_detection": {
            "is_bitemporal": 2.5, "mentions_change": 2.0, "has_dates": 2.0,
            "has_sar": 1.5,
        },
        "grounding": {"mentions_location": 2.5},
        "geo_rag": {"mentions_elevation": 2.5},
        "vqa_caption": {"_bias": 0.5},
    }
    _ROUTING_THRESHOLD = 0.35

    # Execution ordering (lower = earlier)
    _PRIORITY = {
        "sar_bridge": 0, "geo_rag": 1, "optical_sar_fusion": 2,
        "grounding": 3, "vlm_grounding": 3,
        "change_detection": 4, "fusion_change_detection": 4, "vlm_change_detection": 4,
        "vqa_caption": 10, "vlm_vqa_caption": 10,
    }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def decide_route(self, state: QueryState) -> JEVDecision:
        """Which tools to invoke, in what order.  Returns list of tool names."""
        t0 = time.perf_counter()
        feats = self._features(state)

        # Score every tool
        tool_probs: dict[str, float] = {}
        for tool, weights in self._ROUTING_WEIGHTS.items():
            if tool not in state.available_tools and tool != "vqa_caption":
                continue
            logit = weights.get("_bias", -1.0)
            for feat, w in weights.items():
                if feat != "_bias":
                    logit += w * feats.get(feat, 0.0)
            tool_probs[tool] = _sigmoid(logit)

        # Select above threshold, filter to available
        selected = [
            (t, p) for t, p in tool_probs.items()
            if p >= self._ROUTING_THRESHOLD and t in state.available_tools
        ]

        # Always include vqa_caption terminal step
        if not any(t == "vqa_caption" for t, _ in selected) and "vqa_caption" in state.available_tools:
            selected.append(("vqa_caption", tool_probs.get("vqa_caption", 0.5)))

        selected = self._resolve_exclusions(selected)
        selected.sort(key=lambda tp: self._PRIORITY.get(tp[0], 5))

        # Overall confidence: geometric mean
        if selected:
            overall = math.exp(sum(math.log(max(p, 1e-10)) for _, p in selected) / len(selected))
        else:
            overall = 0.0

        tools = [t for t, _ in selected]
        reasoning = "; ".join(f"{t}(p={p:.3f})" for t, p in selected)
        elapsed = (time.perf_counter() - t0) * 1000

        logger.info("jev_route", tools=tools, p=round(overall, 3), ms=round(elapsed, 2))
        return JEVDecision(JEVQuestionType.ROUTE, tools, round(overall, 4), reasoning, round(elapsed, 2))

    # ------------------------------------------------------------------
    # Evidence scoring
    # ------------------------------------------------------------------

    def score_evidence(self, evidence: list[EvidenceState]) -> JEVDecision:
        """Score overall evidence quality.  Harmonic mean penalises weak links."""
        t0 = time.perf_counter()
        if not evidence:
            return JEVDecision(JEVQuestionType.SCORE, 0.0, 0.0, "No evidence", 0.0)

        scores, parts = [], []
        for e in evidence:
            if e.analysis_unavailable:
                scores.append(0.0)
                parts.append(f"{e.tool_name}:unavail")
                continue
            logit = 0.0
            if e.confidence is not None:
                logit += (e.confidence - 0.5) * 3.0
            if e.has_artifact:
                logit += 0.5
            if e.coregistration_reliable is False:
                logit -= 1.5
            logit -= e.warning_count * 0.3
            if len(e.result_keys) >= 3:
                logit += 0.3
            s = _sigmoid(logit)
            scores.append(s)
            parts.append(f"{e.tool_name}:{s:.3f}")

        overall = len(scores) / sum(1.0 / max(s, 1e-10) for s in scores) if all(s > 0 for s in scores) else 0.0
        elapsed = (time.perf_counter() - t0) * 1000
        return JEVDecision(JEVQuestionType.SCORE, round(overall, 4), round(overall, 4),
                           "; ".join(parts), round(elapsed, 2))

    # ------------------------------------------------------------------
    # Consistency verification
    # ------------------------------------------------------------------

    def verify_consistency(self, outputs: list[dict[str, Any]], final_answer: str) -> JEVDecision:
        """Check whether final_answer is consistent with specialist outputs."""
        t0 = time.perf_counter()
        if not outputs or not final_answer:
            return JEVDecision(JEVQuestionType.VERIFY, False, 0.0, "Nothing to verify", 0.0)

        answer_lower = final_answer.lower()
        checks: list[tuple[str, bool]] = []

        for o in outputs:
            result = o.get("result", {})
            tool = o.get("tool_name", "")

            # Unavailable tools should not be claimed as successful
            if result.get("analysis_unavailable") and tool.replace("_", " ") in answer_lower:
                mentions_fail = any(w in answer_lower for w in ("unavailabl", "unable", "cannot"))
                checks.append((f"unavail_{tool}", mentions_fail))

            # Tool referenced at all (positive signal)
            if tool.replace("_", " ") in answer_lower or tool in answer_lower:
                checks.append((f"ref_{tool}", True))

            # Has real data
            if result.get("confidence") is not None:
                checks.append((f"conf_{tool}", True))

        if checks:
            pass_rate = sum(v for _, v in checks) / len(checks)
        else:
            pass_rate = 0.5

        prob = _sigmoid((pass_rate - 0.5) * 4.0)
        elapsed = (time.perf_counter() - t0) * 1000
        reasoning = "; ".join(f"{n}={'OK' if v else 'FAIL'}" for n, v in checks) or "No checkable claims"
        return JEVDecision(JEVQuestionType.VERIFY, prob >= 0.5, round(prob, 4), reasoning, round(elapsed, 2))

    # ------------------------------------------------------------------
    # Guard (policy gate)
    # ------------------------------------------------------------------

    def check_guard(self, state: QueryState, plan_tools: list[str]) -> JEVDecision:
        """Validate a proposed plan against safety constraints."""
        t0 = time.perf_counter()
        violations, warnings = [], []

        unknown = [t for t in plan_tools if t not in state.available_tools]
        if unknown:
            violations.append(f"Unknown tools: {unknown}")

        _PIXEL = {"vqa_caption", "vlm_vqa_caption", "grounding", "vlm_grounding",
                   "sar_bridge", "change_detection", "vlm_change_detection",
                   "optical_sar_fusion", "fusion_change_detection"}
        pixel_in = [t for t in plan_tools if t in _PIXEL]
        if pixel_in and state.n_images == 0:
            violations.append(f"Pixel tools {pixel_in} need ≥1 image")

        _BITEMP = {"change_detection", "vlm_change_detection", "fusion_change_detection"}
        if any(t in _BITEMP for t in plan_tools) and state.n_images < 2:
            violations.append("Bi-temporal tools need ≥2 images")

        if "optical_sar_fusion" in plan_tools and not (state.has_sar and state.has_optical):
            violations.append("optical_sar_fusion needs SAR + optical")

        if not any(t in ("vqa_caption", "vlm_vqa_caption") for t in plan_tools):
            warnings.append("No terminal synthesis step")

        if violations:
            prob, safe = 0.0, False
        elif warnings:
            prob, safe = 0.7, True
        else:
            prob, safe = 0.98, True

        elapsed = (time.perf_counter() - t0) * 1000
        parts = [f"VIOLATION: {v}" for v in violations] + [f"WARNING: {w}" for w in warnings]
        return JEVDecision(JEVQuestionType.GUARD, safe, round(prob, 4),
                           "; ".join(parts) or "All guards passed", round(elapsed, 2))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _features(state: QueryState) -> dict[str, float]:
        """Extract numeric features from state (no separate dataclass needed)."""
        return {
            "has_sar": float(state.has_sar),
            "has_optical": float(state.has_optical),
            "is_cross_modal": float(state.is_cross_modal_pair),
            "is_bitemporal": float(state.is_bitemporal_pair),
            "mentions_change": float(state.mentions_change),
            "mentions_location": float(state.mentions_location),
            "mentions_sar": float(state.mentions_sar),
            "mentions_elevation": float(state.mentions_elevation),
            "n_images": float(state.n_images),
            "has_dates": float(state.has_dates),
        }

    @staticmethod
    def _resolve_exclusions(selected: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Mutual exclusion: fusion XOR bridge+change, fusion_cd XOR cd."""
        names = {t for t, _ in selected}
        if "optical_sar_fusion" in names:
            selected = [(t, p) for t, p in selected if t not in ("sar_bridge", "change_detection")]
        if "fusion_change_detection" in names and "change_detection" in names:
            fp = next(p for t, p in selected if t == "fusion_change_detection")
            cp = next(p for t, p in selected if t == "change_detection")
            drop = "change_detection" if fp >= cp else "fusion_change_detection"
            selected = [(t, p) for t, p in selected if t != drop]
        return selected
