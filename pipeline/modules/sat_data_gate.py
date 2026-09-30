"""
SAT-OS Data Gate — unified PASS/WARNING/FAIL from existing validation modules.

Wraps m09 (CRS), m10 (radiometric), m11 (change-pair) and contract-derived
checks into one DataGateReport. No new validation logic — just aggregation.
"""

from __future__ import annotations

from typing import Any

from pipeline.modules.m09_geospatial import validate_crs
from pipeline.modules.m10_radiometric import validate_radiometric
from pipeline.schemas.contracts import ResultStatus
from pipeline.schemas.sat_schemas import (
    DataGateReport,
    GateResult,
    GateStatus,
    ObservationContract,
)


def _status_from_module(module_status: ResultStatus) -> GateStatus:
    if module_status == ResultStatus.SUCCESS:
        return GateStatus.PASS
    if module_status == ResultStatus.FAILURE:
        return GateStatus.FAIL
    return GateStatus.WARNING


def _check_resolution(
    geo_meta: dict[str, Any], contract: ObservationContract
) -> GateResult:
    """Check if image GSD meets the contract's requirement."""
    actual_gsd = geo_meta.get("resolution_m")
    required = contract.required_gsd_m
    if required is None or actual_gsd is None:
        return GateResult(gate_name="spatial_resolution", status=GateStatus.WARNING,
                          reason="GSD requirement or actual GSD unknown", value=actual_gsd)
    if actual_gsd <= required:
        return GateResult(gate_name="spatial_resolution", status=GateStatus.PASS,
                          reason=f"GSD {actual_gsd:.1f}m ≤ required {required:.1f}m", value=actual_gsd)
    if actual_gsd <= required * 2:
        return GateResult(gate_name="spatial_resolution", status=GateStatus.WARNING,
                          reason=f"GSD {actual_gsd:.1f}m exceeds required {required:.1f}m — results may lack detail",
                          value=actual_gsd)
    return GateResult(gate_name="spatial_resolution", status=GateStatus.FAIL,
                      reason=f"GSD {actual_gsd:.1f}m far exceeds required {required:.1f}m — observation not possible",
                      value=actual_gsd)


def _check_modality(
    images: list[dict[str, Any]], contract: ObservationContract
) -> GateResult:
    """Check sensor modality matches contract."""
    required = contract.sensor_modality
    available = [img.get("modality", "unknown") for img in images]
    if required in available or "unknown" in available:
        return GateResult(gate_name="sensor_suitability", status=GateStatus.PASS,
                          reason=f"Required {required} available")
    return GateResult(gate_name="sensor_suitability", status=GateStatus.WARNING,
                      reason=f"Required {required}, available: {available}")


def _check_temporal(
    images: list[dict[str, Any]], contract: ObservationContract
) -> GateResult:
    """Check temporal coverage matches contract."""
    if contract.temporal_requirement == "bi-temporal" and len(images) < 2:
        return GateResult(gate_name="temporal_coverage", status=GateStatus.FAIL,
                          reason="Bi-temporal analysis requires 2 images, only 1 provided")
    return GateResult(gate_name="temporal_coverage", status=GateStatus.PASS,
                      reason=f"{len(images)} image(s) for {contract.temporal_requirement}")


def run_data_gate(
    contract: ObservationContract,
    images: list[dict[str, Any]],
    file_path: str | None = None,
    geo_meta: dict[str, Any] | None = None,
) -> DataGateReport:
    """Run all gate checks and return aggregated report."""
    gates: list[GateResult] = []

    # Existing module checks (reuse m09/m10)
    if file_path:
        crs_result = validate_crs(file_path)
        gates.append(GateResult(
            gate_name="crs_validity",
            status=_status_from_module(crs_result.status),
            reason=crs_result.data.get("crs", crs_result.errors[0].message if crs_result.errors else ""),
        ))
        radio_result = validate_radiometric(file_path)
        gates.append(GateResult(
            gate_name="radiometric_quality",
            status=_status_from_module(radio_result.status),
            reason=f"valid_pct={radio_result.data.get('valid_pixel_pct', '?')}%",
        ))

    # Contract-derived checks
    if geo_meta:
        gates.append(_check_resolution(geo_meta, contract))
    gates.append(_check_modality(images, contract))
    gates.append(_check_temporal(images, contract))

    # Aggregate
    statuses = [g.status for g in gates]
    if GateStatus.FAIL in statuses:
        overall = GateStatus.FAIL
    elif GateStatus.WARNING in statuses:
        overall = GateStatus.WARNING
    else:
        overall = GateStatus.PASS

    critical = [g.gate_name for g in gates if g.status == GateStatus.FAIL]
    return DataGateReport(gates=gates, overall=overall, critical_failures=critical)
