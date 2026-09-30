"""
SAT-OS Audit Report — human-readable report from certificate data.

Generates JSON (always) and a plain-text report (always). PDF is deferred
to avoid a new dependency — the text report is the human-readable output.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.schemas.sat_schemas import AnswerabilityCertificate


def _section(title: str, content: str) -> str:
    return f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n{content}\n"


def generate_text_report(cert: AnswerabilityCertificate) -> str:
    """Generate a professional human-readable audit report from a certificate.

    Follows implementation.md §11:
    What was asked → what data was available → what could be observed →
    what evidence was used → what was tested → what could/couldn't be
    verified → why the final decision was reached.
    """
    lines: list[str] = []
    lines.append("SAT-OS AUDIT REPORT")
    lines.append(f"Run ID: {cert.run_id}")
    lines.append(f"Generated: {cert.timestamp.isoformat()}")
    lines.append(f"Pipeline: v{cert.pipeline_version}")

    # 1. What was asked
    section = f"  Query: {cert.query_text}\n"
    if cert.contract:
        c = cert.contract
        section += f"  Target: {c.target or 'unspecified'}\n"
        section += f"  Phenomenon: {c.phenomenon}\n"
        section += f"  Spatial scale: {c.spatial_scale or 'unspecified'}\n"
        section += f"  Temporal: {c.temporal_requirement}\n"
        section += f"  Sensor: {c.sensor_modality}\n"
        if c.required_gsd_m:
            section += f"  Required GSD: ≤{c.required_gsd_m}m\n"
    lines.append(_section("1. QUERY & OBSERVATION CONTRACT", section))

    # 2. What data was available
    section = ""
    for f in cert.input_files:
        h = cert.input_hashes.get(f.get("filename", ""), "n/a")
        section += f"  {f.get('filename')}: modality={f.get('modality')}, sha256={h[:16]}…\n"
    lines.append(_section("2. INPUT DATA", section))

    # 3. Data gate results
    section = ""
    if cert.data_gate:
        section += f"  Overall: {cert.data_gate.overall.upper()}\n"
        for g in cert.data_gate.gates:
            section += f"    [{g.status.upper():7s}] {g.gate_name}: {g.reason}\n"
        if cert.data_gate.critical_failures:
            section += f"  Critical failures: {', '.join(cert.data_gate.critical_failures)}\n"
    else:
        section = "  Not run\n"
    lines.append(_section("3. DATA QUALITY GATE", section))

    # 4. Observability
    section = ""
    if cert.observability:
        o = cert.observability
        section += f"  Observable:           {o.observable_pct:.1f}%\n"
        section += f"  Partially observable: {o.partially_observable_pct:.1f}%\n"
        section += f"  Unobservable:         {o.unobservable_pct:.1f}%\n"
        if o.mask_path:
            section += f"  Mask: {o.mask_path}\n"
    else:
        section = "  Not computed\n"
    lines.append(_section("4. SPATIAL OBSERVABILITY", section))

    # 5. Analysis & evidence
    section = ""
    section += f"  Algorithms: {', '.join(cert.algorithms_used) or 'none'}\n"
    if cert.evidence_set:
        section += f"  Evidence items: {len(cert.evidence_set.items)}\n"
        section += f"  Minimum evidence set: {', '.join(cert.evidence_set.minimum_set) or 'empty'}\n"
        for item in cert.evidence_set.items:
            section += f"    {item.tool_name}: role={item.role}, breaks_if_removed={item.removal_breaks_claim}\n"
    lines.append(_section("5. EVIDENCE", section))

    # 6. Falsification
    section = ""
    if cert.falsification:
        for check in cert.falsification.checks:
            marker = "✓" if check.supported is False else ("✗" if check.supported else "?")
            section += f"  [{marker}] {check.hypothesis}\n"
            if check.detail:
                section += f"      {check.detail}\n"
        if cert.falsification.any_contradiction:
            section += "  ⚠ CONTRADICTIONS FOUND\n"
    else:
        section = "  Not run\n"
    lines.append(_section("6. FALSIFICATION", section))

    # 7. Verification
    section = ""
    if cert.verification:
        v = cert.verification
        section += f"  STATUS: {v.status.upper()}\n"
        for r in v.reasons:
            section += f"    • {r}\n"
        if v.abstention_detail:
            section += f"\n  ABSTENTION: {v.abstention_detail}\n"
    lines.append(_section("7. VERIFICATION DECISION", section))

    # 8. Assumptions
    if cert.assumptions:
        section = "".join(f"  • {a}\n" for a in cert.assumptions)
        lines.append(_section("8. ASSUMPTIONS", section))

    # 9. Output artifacts
    if cert.output_hashes:
        section = "".join(f"  {k}: {v[:16]}…\n" for k, v in cert.output_hashes.items())
        lines.append(_section("9. OUTPUT ARTIFACTS", section))

    lines.append(f"\n{'=' * 60}")
    lines.append("  END OF REPORT")
    lines.append(f"{'=' * 60}\n")
    return "\n".join(lines)


def save_report(
    cert: AnswerabilityCertificate,
    output_dir: str,
) -> dict[str, str]:
    """Save JSON certificate + text audit report. Returns file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = str(out / f"certificate_{cert.run_id}.json")
    text_path = str(out / f"audit_report_{cert.run_id}.txt")

    Path(json_path).write_text(cert.model_dump_json(indent=2), encoding="utf-8")
    Path(text_path).write_text(generate_text_report(cert), encoding="utf-8")

    return {"json": json_path, "text": text_path}
