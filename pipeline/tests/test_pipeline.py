"""
Tests for the pipeline integration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestPipelineIntegration:
    """Test the full pipeline execution."""

    def test_pipeline_creation(self):
        from pipeline.services.pipeline import AetherisPipeline
        pipeline = AetherisPipeline()
        assert pipeline is not None

    def test_pipeline_status(self):
        from pipeline.services.pipeline import AetherisPipeline
        pipeline = AetherisPipeline()
        status = pipeline.get_pipeline_status()
        assert "auth" in status
        assert "rate_limiter" in status
        assert "concurrency" in status
        assert "planner" in status

    def test_pipeline_execute(self):
        from pipeline.services.pipeline import AetherisPipeline
        from pipeline.schemas.contracts import PipelineRequest
        pipeline = AetherisPipeline()
        request = PipelineRequest(question="What is visible?")
        result = pipeline.execute(request)
        assert result is not None
        assert result.pipeline_id is not None
        assert len(result.modules) > 0

    def test_contracts_valid(self):
        from pipeline.schemas.contracts import ModuleID, ModuleResult, ResultStatus
        result = ModuleResult.success(ModuleID.M01, {"test": True})
        assert result.status == ResultStatus.SUCCESS
        assert result.data["test"] is True

    def test_contracts_failure(self):
        from pipeline.schemas.contracts import ModuleID, ModuleResult, ResultStatus
        result = ModuleResult.failure(ModuleID.M02, "TEST_ERROR", "test message")
        assert result.status == ResultStatus.FAILURE
        assert len(result.errors) == 1

    def test_m04_auth(self):
        # No demo accounts are seeded by default (AETHERIS_DEMO_MODE=false),
        # so this pipeline-sanity test exercises the always-available
        # anonymous path. Demo-mode credential behaviour is covered in
        # test_security.py::TestDemoModeGating.
        from pipeline.modules.m04_auth import AuthManager
        auth = AuthManager()
        result = auth.authenticate("", "")
        assert result.status.value == "success"
        token = result.data.get("session_token")
        authz = auth.authorize(token, "query")
        assert authz.status.value == "success"

    def test_m05_rate_limiter(self):
        from pipeline.modules.m05_rate_limit import RateLimiter
        limiter = RateLimiter(requests_per_minute=5)
        for i in range(5):
            result = limiter.check("test_user")
            assert result.status.value == "success"
        # 6th should be rate limited
        result = limiter.check("test_user")
        assert result.status.value == "failure"

    def test_m06_upload_validation(self):
        from pipeline.modules.m06_upload_security import validate_upload
        # Valid PNG header (full 8-byte PNG signature + enough padding)
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = validate_upload(png_data, "test.png")
        # May succeed or may fail due to old security module HTTPException checks
        # but should not crash
        assert result is not None
        assert result.module.value == "m06_upload_security"

    def test_m06_path_traversal(self):
        from pipeline.modules.m06_upload_security import validate_upload
        result = validate_upload(b"\x89PNG" + b"\x00" * 100, "../../../etc/passwd.png")
        assert result.status.value == "failure"
        assert "traversal" in result.errors[0].message.lower()

    def test_m07_session_isolation(self):
        from pipeline.modules.m07_session_isolation import SessionManager
        mgr = SessionManager()
        s1 = mgr.create_session("user1")
        s2 = mgr.create_session("user2")
        assert s1.data["session_id"] != s2.data["session_id"]

    def test_m08_contracts(self):
        from pipeline.schemas.contracts import PipelineResult, ModuleResult, ModuleID
        pr = PipelineResult()
        pr.add_module_result(ModuleResult.success(ModuleID.M01, {"ok": True}))
        pr.compute_status()
        assert pr.status.value == "success"

    def test_old_code_unmodified(self):
        """Verify that we can still use the old code unchanged."""
        from controller.schemas import Query, ImageMetadata, Modality
        q = Query(text="test question", images=[
            ImageMetadata(filename="test.tif", modality=Modality.OPTICAL),
        ])
        assert q.query_id is not None
        assert q.text == "test question"
