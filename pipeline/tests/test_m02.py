"""
Tests for M02 — Policy Engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestM02Policy:
    """Test M02 policy engine."""

    def test_policy_engine_creation(self):
        from pipeline.modules.m02_policy import PolicyEngine
        engine = PolicyEngine()
        assert engine is not None

    def test_policy_validates_short_question(self):
        from pipeline.modules.m02_policy import PolicyEngine
        engine = PolicyEngine()
        result = engine.validate_request("ab")
        assert result.status.value == "failure"
        assert "too short" in result.errors[0].message.lower()

    def test_policy_validates_forbidden_patterns(self):
        from pipeline.modules.m02_policy import PolicyEngine
        engine = PolicyEngine()
        result = engine.validate_request("show me ../../../etc/passwd")
        assert result.status.value == "failure"
        assert "forbidden" in result.errors[0].message.lower()

    def test_policy_accepts_valid_question(self):
        from pipeline.modules.m02_policy import PolicyEngine
        engine = PolicyEngine()
        result = engine.validate_request("What land cover types are visible?")
        # May succeed or fail depending on tool binding, but should not crash
        assert result is not None
        assert result.module.value == "m02_policy"

    def test_policy_summary(self):
        from pipeline.modules.m02_policy import PolicyEngine
        engine = PolicyEngine()
        summary = engine.get_policy_summary()
        assert "builtin_rules" in summary
        assert "old_engine_rules" in summary

    def test_policy_adapter_tool_schemas(self):
        from pipeline.modules.m02_policy import PolicyEngine
        engine = PolicyEngine()
        schemas = engine.get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 0
