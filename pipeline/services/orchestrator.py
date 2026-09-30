"""
Orchestrator — builds the M01–M10 DAG *shape* for visualization.

This is a structural diagram, not a live trace: every node below except M01
is a placeholder lambda returning a fixed {"x": "ok"} payload, so /api/dag/execute
shows dependency order and pass/fail propagation, not real module output.
The actual per-request execution is the linear call chain in
services/pipeline.py (AetherisPipeline.execute), which calls the real M02,
M04–M14 functions directly and is what /api/pipeline/execute and
/api/pipeline/upload run. See gui/templates/index.html's "Execution DAG"
card, which is labelled accordingly, and DAG_IS_VISUALIZATION_ONLY below.
"""

from __future__ import annotations

from typing import Any

from pipeline.modules.m01_baseline import run_health_check
from pipeline.modules.m03_dag import DAGNode, ExecutionDAG
from pipeline.schemas.contracts import ModuleID, ModuleResult

DAG_IS_VISUALIZATION_ONLY = True


def build_pipeline_dag() -> ExecutionDAG:
    """Build the standard M01–M10 execution DAG."""
    dag = ExecutionDAG()

    # M01: Baseline (no dependencies)
    dag.add_node(DAGNode(
        node_id="m01",
        module_id=ModuleID.M01,
        execute_fn=lambda ctx: run_health_check(),
        optional=True,
    ))

    # M04: Auth (depends on M01)
    dag.add_node(DAGNode(
        node_id="m04",
        module_id=ModuleID.M04,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M04, {"auth": "ok"}),
        depends_on=["m01"],
    ))

    # M05: Rate Limit (depends on M04)
    dag.add_node(DAGNode(
        node_id="m05",
        module_id=ModuleID.M05,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M05, {"rate": "ok"}),
        depends_on=["m04"],
    ))

    # M07: Session Isolation (depends on M04)
    dag.add_node(DAGNode(
        node_id="m07",
        module_id=ModuleID.M07,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M07, {"session": "ok"}),
        depends_on=["m04"],
    ))

    # M06: Upload Security (depends on M05, M07)
    dag.add_node(DAGNode(
        node_id="m06",
        module_id=ModuleID.M06,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M06, {"upload": "ok"}),
        depends_on=["m05", "m07"],
        optional=True,
    ))

    # M09: CRS Validation (depends on M06)
    dag.add_node(DAGNode(
        node_id="m09",
        module_id=ModuleID.M09,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M09, {"crs": "ok"}),
        depends_on=["m06"],
        optional=True,
    ))

    # M10: Radiometric (depends on M06)
    dag.add_node(DAGNode(
        node_id="m10",
        module_id=ModuleID.M10,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M10, {"radiometric": "ok"}),
        depends_on=["m06"],
        optional=True,
    ))

    # M11: Change-pair validation (depends on M06, only relevant for
    # bi-temporal queries — optional like M09/M10)
    dag.add_node(DAGNode(
        node_id="m11",
        module_id=ModuleID.M11,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M11, {"change_pair": "ok"}),
        depends_on=["m06"],
        optional=True,
    ))

    # M02: Policy (depends on M09, M10)
    dag.add_node(DAGNode(
        node_id="m02",
        module_id=ModuleID.M02,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M02, {"policy": "ok"}),
        depends_on=["m09", "m10"],
    ))

    # M03: DAG/Orchestrator (depends on M02, M11) — this is the old planner call
    dag.add_node(DAGNode(
        node_id="m03",
        module_id=ModuleID.M03,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M03, {"dag": "ok"}),
        depends_on=["m02", "m11"],
    ))

    # M12: GeoTIFF artifact preservation (depends on M03 — operates on the
    # artifact the old planner/specialist just produced)
    dag.add_node(DAGNode(
        node_id="m12",
        module_id=ModuleID.M12,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M12, {"geotiff": "ok"}),
        depends_on=["m03"],
        optional=True,
    ))

    # M13: Conformal calibration (independent of the file/DAG flow — reflects
    # whether a saved calibration artifact was loaded at planner start-up)
    dag.add_node(DAGNode(
        node_id="m13",
        module_id=ModuleID.M13,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M13, {"calibration": "ok"}),
        depends_on=["m01"],
        optional=True,
    ))

    # M14: Audit/provenance (depends on M03 — needs the query's full trail)
    dag.add_node(DAGNode(
        node_id="m14",
        module_id=ModuleID.M14,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M14, {"provenance": "ok"}),
        depends_on=["m03"],
        optional=True,
    ))

    # M08: Contracts (depends on M03, M12, M13, M14) — final result assembly
    dag.add_node(DAGNode(
        node_id="m08",
        module_id=ModuleID.M08,
        execute_fn=lambda ctx: ModuleResult.success(ModuleID.M08, {"contracts": "ok"}),
        depends_on=["m03", "m12", "m13", "m14"],
    ))

    return dag
