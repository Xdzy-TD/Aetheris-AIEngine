"""
Tests for M03 — Execution DAG.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.modules.m03_dag import DAGNode, ExecutionDAG
from pipeline.schemas.contracts import ModuleID, ModuleResult


class TestM03DAG:
    """Test M03 DAG orchestrator."""

    def test_dag_creation(self):
        dag = ExecutionDAG()
        assert dag is not None

    def test_dag_add_nodes(self):
        dag = ExecutionDAG()
        dag.add_node(DAGNode(
            "n1", ModuleID.M01,
            lambda ctx: ModuleResult.success(ModuleID.M01, {"ok": True}),
        ))
        dag.add_node(DAGNode(
            "n2", ModuleID.M02,
            lambda ctx: ModuleResult.success(ModuleID.M02, {"ok": True}),
            depends_on=["n1"],
        ))
        state = dag.get_state()
        assert state["total_nodes"] == 2

    def test_dag_validation_passes(self):
        dag = ExecutionDAG()
        dag.add_node(DAGNode("a", ModuleID.M01, lambda ctx: ModuleResult.success(ModuleID.M01, {})))
        dag.add_node(DAGNode("b", ModuleID.M02, lambda ctx: ModuleResult.success(ModuleID.M02, {}), depends_on=["a"]))
        result = dag.validate()
        assert result.status.value == "success"

    def test_dag_detects_missing_dependency(self):
        dag = ExecutionDAG()
        dag.add_node(DAGNode("a", ModuleID.M01, lambda ctx: ModuleResult.success(ModuleID.M01, {}), depends_on=["missing"]))
        result = dag.validate()
        assert result.status.value == "failure"
        assert "unknown" in result.errors[0].message.lower()

    def test_dag_execution(self):
        results = []
        dag = ExecutionDAG()
        dag.add_node(DAGNode("a", ModuleID.M01, lambda ctx: ModuleResult.success(ModuleID.M01, {"step": "a"})))
        dag.add_node(DAGNode("b", ModuleID.M02, lambda ctx: ModuleResult.success(ModuleID.M02, {"step": "b"}), depends_on=["a"]))
        result = dag.execute()
        assert result.status.value == "success"
        state = dag.get_state()
        assert state["completed"] == 2

    def test_dag_skips_on_dependency_failure(self):
        dag = ExecutionDAG()
        dag.add_node(DAGNode("a", ModuleID.M01, lambda ctx: ModuleResult.failure(ModuleID.M01, "ERR", "fail")))
        dag.add_node(DAGNode("b", ModuleID.M02, lambda ctx: ModuleResult.success(ModuleID.M02, {}), depends_on=["a"]))
        result = dag.execute()
        state = dag.get_state()
        assert state["failed"] == 1
        assert state["skipped"] == 1

    def test_full_pipeline_dag(self):
        from pipeline.services.orchestrator import build_pipeline_dag
        dag = build_pipeline_dag()
        validation = dag.validate()
        assert validation.status.value == "success"
        state = dag.get_state()
        assert state["total_nodes"] == 14  # M01–M14 (M11–M14 wired in since)
