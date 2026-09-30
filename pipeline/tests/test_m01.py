"""
Tests for M01 — Baseline & Health Check.
"""

import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestM01Baseline:
    """Test M01 health check functionality."""

    def test_health_check_runs(self):
        """Health check should return a result without crashing."""
        from pipeline.modules.m01_baseline import run_health_check
        result = run_health_check()
        assert result is not None
        assert result.module.value == "m01_baseline"
        assert result.status is not None

    def test_health_check_has_system_info(self):
        """Health check should include system information."""
        from pipeline.modules.m01_baseline import run_health_check
        result = run_health_check()
        assert "system" in result.data
        assert "python_version" in result.data["system"]

    def test_health_check_has_components(self):
        """Health check should report on old AETHERIS components."""
        from pipeline.modules.m01_baseline import run_health_check
        result = run_health_check()
        assert "old_components" in result.data
        assert len(result.data["old_components"]) > 0

    def test_health_check_has_dependencies(self):
        """Health check should report on dependencies."""
        from pipeline.modules.m01_baseline import run_health_check
        result = run_health_check()
        assert "dependencies" in result.data

    def test_health_check_has_summary(self):
        """Health check should have an overall summary."""
        from pipeline.modules.m01_baseline import run_health_check
        result = run_health_check()
        assert "summary" in result.data
        assert "overall_healthy" in result.data["summary"]


class TestOldComponentImports:
    """Verify that old AETHERIS components can be imported."""

    def test_import_schemas(self):
        from controller.schemas import Query, ExecutionPlan, ExecutionResult
        assert Query is not None

    def test_import_policy(self):
        from controller.policy import check_plan, PolicyViolation
        assert check_plan is not None

    def test_import_tool_registry(self):
        from controller.tool_registry import ToolRegistry
        reg = ToolRegistry()
        assert len(reg.list_tool_names()) > 0

    def test_import_memory(self):
        from controller.memory import RoutingMemory
        mem = RoutingMemory()
        assert mem.stats()["total_entries"] == 0

    def test_import_audit_log(self):
        from controller.audit_log import AuditLog
        assert AuditLog is not None

    def test_import_artifacts(self):
        from controller.artifacts import ArtifactStore
        assert ArtifactStore is not None

    def test_import_imagery(self):
        from specialists._imagery import probe_raster_metadata, load_raster
        assert probe_raster_metadata is not None
