"""
SAT-OS Certificate — assembles an AnswerabilityCertificate from run data.

Pure data assembly — every field comes from existing pipeline outputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from controller.model_registry import version_for_task
from pipeline.schemas.sat_schemas import (
    AnswerabilityCertificate,
    DataGateReport,
    EvidenceSetReport,
    FalsificationReport,
    ObservabilityStats,
    ObservationContract,
    VerificationDecision,
)


def _hash_file(path: str) -> str:
    """SHA-256 of a file's bytes."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return ""


def build_certificate(
    query_text: str,
    contract: ObservationContract,
    images: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    observability: ObservabilityStats | None = None,
    data_gate: DataGateReport | None = None,
    evidence_set: EvidenceSetReport | None = None,
    falsification: FalsificationReport | None = None,
    verification: VerificationDecision | None = None,
) -> AnswerabilityCertificate:
    """Assemble a certificate from existing run artifacts."""
    # Input hashes
    input_hashes = {}
    input_files = []
    for img in images:
        path = img.get("path", "")
        name = img.get("filename", "")
        if path:
            input_hashes[name] = _hash_file(path)
        input_files.append({"filename": name, "modality": img.get("modality"), "path": path})

    # Algorithms/models used
    algorithms = []
    for out in outputs:
        tool = out.get("tool_name", "")
        if not out.get("result", {}).get("analysis_unavailable"):
            algorithms.append(version_for_task(tool))

    # Output hashes (from artifacts)
    output_hashes = {}
    for out in outputs:
        artifact = out.get("result", {}).get("artifact")
        if isinstance(artifact, dict) and artifact.get("sha256"):
            output_hashes[artifact.get("artifact_id", out.get("tool_name", ""))] = artifact["sha256"]

    # Assumptions
    assumptions = []
    if contract.sensor_modality == "optical":
        assumptions.append("Optical imagery assumed cloud-free where not masked")
    if contract.temporal_requirement == "bi-temporal":
        assumptions.append("Both images assumed to cover the same geographic extent")
    if any(out.get("result", {}).get("coregistration_reliable") is False for out in outputs):
        assumptions.append("Co-registration was unreliable — change estimates may include alignment error")

    return AnswerabilityCertificate(
        query_text=query_text,
        contract=contract,
        input_files=input_files,
        input_hashes=input_hashes,
        algorithms_used=algorithms,
        observability=observability,
        data_gate=data_gate,
        evidence_set=evidence_set,
        falsification=falsification,
        verification=verification,
        assumptions=assumptions,
        output_hashes=output_hashes,
    )


def export_certificate_json(cert: AnswerabilityCertificate) -> str:
    """Export certificate as JSON string."""
    return cert.model_dump_json(indent=2)
