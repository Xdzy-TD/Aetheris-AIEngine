"""
SAT-OS Verification — VERIFIED / QUALIFIED / ABSTAIN decision.

Combines observability, data gate, evidence sufficiency, and falsification
into a single verdict. A claim is never VERIFIED solely because a model
is confident (implementation.md §8).
"""

from __future__ import annotations

from pipeline.schemas.sat_schemas import (
    DataGateReport,
    EvidenceSetReport,
    FalsificationReport,
    GateStatus,
    ObservabilityStats,
    VerificationDecision,
    VerificationStatus,
)


def decide_verification(
    observability: ObservabilityStats | None,
    gate: DataGateReport | None,
    evidence: EvidenceSetReport | None,
    falsification: FalsificationReport | None,
) -> VerificationDecision:
    """Produce a VERIFIED/QUALIFIED/ABSTAIN verdict from all SAT-OS layers."""
    reasons: list[str] = []
    obs_ok = True
    gate_ok = True
    evidence_ok = True
    falsification_ok = True

    # Observability check
    if observability:
        if observability.observable_pct < 30:
            obs_ok = False
            reasons.append(f"Only {observability.observable_pct:.0f}% observable — insufficient spatial coverage")
        elif observability.observable_pct < 60:
            obs_ok = True  # qualifiable, not a hard fail
            reasons.append(f"Partial observability ({observability.observable_pct:.0f}%) — result is qualified")

    # Data gate check
    if gate:
        if gate.overall == GateStatus.FAIL:
            gate_ok = False
            reasons.append(f"Data gate FAIL: {', '.join(gate.critical_failures)}")
        elif gate.overall == GateStatus.WARNING:
            reasons.append("Data gate warnings present")

    # Evidence sufficiency
    if evidence:
        if not evidence.minimum_set:
            evidence_ok = False
            reasons.append("No evidence items in minimum set — claim unsupported")
        essential_with_breaks = [
            item for item in evidence.items
            if item.removal_breaks_claim and item.confidence_contribution < 0.3
        ]
        if essential_with_breaks:
            reasons.append(
                f"Essential evidence has low confidence: "
                f"{', '.join(i.tool_name for i in essential_with_breaks)}"
            )

    # Falsification check
    if falsification and falsification.any_contradiction:
        falsification_ok = False
        contradicted = [c.hypothesis for c in falsification.checks if c.supported is True]
        reasons.append(f"Falsification found contradictions: {'; '.join(contradicted)}")

    # Decision logic (implementation.md §8):
    # ABSTAIN if fundamental physical/data constraint prevents the claim
    if not obs_ok or not gate_ok or not evidence_ok:
        abstention_parts = []
        if not obs_ok:
            abstention_parts.append("insufficient spatial observability")
        if not gate_ok:
            abstention_parts.append(f"data gate failures: {', '.join(gate.critical_failures) if gate else 'unknown'}")
        if not evidence_ok:
            abstention_parts.append("insufficient evidence")

        # What would help
        remedies = []
        if not obs_ok:
            remedies.append("imagery with <30% cloud cover over the AOI")
        if not gate_ok and gate:
            if "spatial_resolution" in gate.critical_failures:
                remedies.append("higher-resolution imagery matching the target scale")
            if "temporal_coverage" in gate.critical_failures:
                remedies.append("a second image for bi-temporal comparison")
        if not evidence_ok:
            remedies.append("additional corroborating data sources")

        return VerificationDecision(
            status=VerificationStatus.ABSTAIN,
            reasons=reasons,
            observability_sufficient=obs_ok,
            data_quality_sufficient=gate_ok,
            evidence_sufficient=evidence_ok,
            falsification_passed=falsification_ok,
            abstention_detail=(
                f"Claim cannot be supported: {'; '.join(abstention_parts)}. "
                f"Would need: {'; '.join(remedies)}."
            ),
        )

    # QUALIFIED if falsification found contradictions or observability is partial
    if not falsification_ok or (observability and observability.observable_pct < 60):
        return VerificationDecision(
            status=VerificationStatus.QUALIFIED,
            reasons=reasons,
            observability_sufficient=obs_ok,
            data_quality_sufficient=gate_ok,
            evidence_sufficient=evidence_ok,
            falsification_passed=falsification_ok,
        )

    # VERIFIED only when all checks pass
    if not reasons:
        reasons.append("All checks passed")
    return VerificationDecision(
        status=VerificationStatus.VERIFIED,
        reasons=reasons,
        observability_sufficient=obs_ok,
        data_quality_sufficient=gate_ok,
        evidence_sufficient=evidence_ok,
        falsification_passed=falsification_ok,
    )
