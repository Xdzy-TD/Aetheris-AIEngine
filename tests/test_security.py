"""
Tests for M16 — Security test suite.

Covers the M15 findings directly: demo-mode gating of the seeded accounts,
wrong-password rejection, rate-limit persistence across repeated calls (not
just the first), upload validation edge cases, audit hash-chain tamper
detection, and the new GUI API-key enforcement (including the health-check
exemption).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Demo-mode gating (m04_auth seeded accounts)
# ---------------------------------------------------------------------------

class TestDemoModeGating:
    """AETHERIS_DEMO_MODE must actually control the seeded accounts."""

    def _reload_with_demo_mode(self, monkeypatch, value: str | None):
        if value is None:
            monkeypatch.delenv("AETHERIS_DEMO_MODE", raising=False)
        else:
            monkeypatch.setenv("AETHERIS_DEMO_MODE", value)
        import pipeline.config as config
        import pipeline.modules.m04_auth as m04_auth
        importlib.reload(config)
        importlib.reload(m04_auth)
        return m04_auth

    def test_demo_mode_off_by_default_seeds_no_accounts(self, monkeypatch):
        m04_auth = self._reload_with_demo_mode(monkeypatch, None)
        mgr = m04_auth.AuthManager()
        assert mgr._users == {}

    def test_demo_mode_false_seeds_no_accounts(self, monkeypatch):
        m04_auth = self._reload_with_demo_mode(monkeypatch, "false")
        mgr = m04_auth.AuthManager()
        assert mgr._users == {}
        result = mgr.authenticate("admin", "admin")
        assert result.status.value == "failure"

    def test_demo_mode_true_seeds_known_accounts(self, monkeypatch):
        m04_auth = self._reload_with_demo_mode(monkeypatch, "true")
        mgr = m04_auth.AuthManager()
        assert set(mgr._users) == {"admin", "analyst", "viewer"}
        result = mgr.authenticate("admin", "admin")
        assert result.status.value == "success"
        assert result.data["identity"]["role"] == m04_auth.Role.ADMIN

    @pytest.fixture(autouse=True)
    def _restore_module_state(self, monkeypatch):
        """Leave config/m04_auth reloaded back to the (unset) default,
        deterministically — done here rather than relying on monkeypatch's
        own teardown ordering relative to this fixture's.
        """
        yield
        monkeypatch.delenv("AETHERIS_DEMO_MODE", raising=False)
        import pipeline.config as config
        import pipeline.modules.m04_auth as m04_auth
        importlib.reload(config)
        importlib.reload(m04_auth)


# ---------------------------------------------------------------------------
# Wrong-password rejection
# ---------------------------------------------------------------------------

class TestWrongPasswordRejection:
    """A known user with a bad password must be rejected, not silently
    downgraded to anonymous."""

    @pytest.fixture
    def demo_auth_manager(self, monkeypatch):
        monkeypatch.setenv("AETHERIS_DEMO_MODE", "true")
        import pipeline.config as config
        import pipeline.modules.m04_auth as m04_auth
        importlib.reload(config)
        importlib.reload(m04_auth)
        yield m04_auth.AuthManager()
        monkeypatch.delenv("AETHERIS_DEMO_MODE", raising=False)
        importlib.reload(config)
        importlib.reload(m04_auth)

    def test_correct_password_succeeds(self, demo_auth_manager):
        result = demo_auth_manager.authenticate("viewer", "viewer")
        assert result.status.value == "success"

    def test_wrong_password_is_rejected(self, demo_auth_manager):
        result = demo_auth_manager.authenticate("admin", "not-the-password")
        assert result.status.value == "failure"
        assert result.errors[0].code == "AUTH_FAILED"

    def test_wrong_password_does_not_grant_anonymous(self, demo_auth_manager):
        result = demo_auth_manager.authenticate("admin", "wrong")
        assert result.status.value == "failure"
        assert "identity" not in result.data

    def test_unknown_user_is_rejected(self, demo_auth_manager):
        result = demo_auth_manager.authenticate("nobody", "whatever")
        assert result.status.value == "failure"

    def test_blank_credentials_grant_anonymous(self, demo_auth_manager):
        result = demo_auth_manager.authenticate("", "")
        assert result.status.value == "success"
        assert result.data["identity"]["user_id"] == "anonymous"


# ---------------------------------------------------------------------------
# Rate-limit persistence (not just the first call)
# ---------------------------------------------------------------------------

class TestRateLimitPersistence:
    """The limiter must keep rejecting once the window is full, and only
    recover as the window slides — not just reject the very next call."""

    def test_allows_up_to_the_limit(self):
        from pipeline.modules.m05_rate_limit import RateLimiter
        limiter = RateLimiter(requests_per_minute=3)
        for _ in range(3):
            result = limiter.check("user-a")
            assert result.status.value == "success"

    def test_rejects_after_limit_and_keeps_rejecting(self):
        from pipeline.modules.m05_rate_limit import RateLimiter
        limiter = RateLimiter(requests_per_minute=3)
        for _ in range(3):
            assert limiter.check("user-a").status.value == "success"

        # 4th, 5th, 6th calls must ALL be rejected — this used to only be
        # verified for the first over-limit call.
        for _ in range(3):
            result = limiter.check("user-a")
            assert result.status.value == "failure"
            assert result.errors[0].code == "RATE_LIMITED"

    def test_limit_is_per_user(self):
        from pipeline.modules.m05_rate_limit import RateLimiter
        limiter = RateLimiter(requests_per_minute=1)
        assert limiter.check("user-a").status.value == "success"
        assert limiter.check("user-a").status.value == "failure"
        # A different user's bucket is untouched by user-a's usage.
        assert limiter.check("user-b").status.value == "success"


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

class TestUploadValidation:
    _PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    def test_valid_png_passes(self):
        from pipeline.modules.m06_upload_security import validate_upload
        result = validate_upload(self._PNG_MAGIC, "scene.png")
        assert result.status.value == "success"

    def test_oversized_upload_rejected(self):
        from pipeline.modules.m06_upload_security import validate_upload
        oversized = self._PNG_MAGIC + b"\x00" * (10 * 1024 * 1024)
        result = validate_upload(oversized, "scene.png", max_size=1024)
        assert result.status.value == "failure"
        assert "exceeds" in result.errors[0].message

    def test_bad_magic_bytes_rejected(self):
        from pipeline.modules.m06_upload_security import validate_upload
        fake_image = b"NOT_AN_IMAGE" + b"\x00" * 32
        result = validate_upload(fake_image, "scene.png")
        assert result.status.value == "failure"
        assert "does not match" in result.errors[0].message

    def test_extension_mismatch_rejected(self):
        from pipeline.modules.m06_upload_security import validate_upload
        result = validate_upload(self._PNG_MAGIC, "scene.exe")
        assert result.status.value == "failure"

    @pytest.mark.parametrize(
        "malicious_name",
        [
            "../../etc/passwd.png",
            "..\\..\\windows\\system32\\evil.png",
            "~/.ssh/id_rsa.png",
            "a/../../b.png",
        ],
    )
    def test_path_traversal_rejected(self, malicious_name):
        from pipeline.modules.m06_upload_security import validate_upload
        result = validate_upload(self._PNG_MAGIC, malicious_name)
        assert result.status.value == "failure"
        assert result.errors[0].code == "PATH_TRAVERSAL_DETECTED"

    def test_filename_is_sanitized_not_just_flagged(self):
        from pipeline.modules.m06_upload_security import validate_upload
        result = validate_upload(self._PNG_MAGIC, "sc<e>n:e?.png")
        assert result.status.value == "success"
        assert "<" not in result.data["sanitized_filename"]


# ---------------------------------------------------------------------------
# Audit hash-chain tamper detection
# ---------------------------------------------------------------------------

class TestAuditHashChainTamper:
    def _make_log(self, tmp_path: Path):
        from controller.audit_log import AuditLog
        return AuditLog(log_path=tmp_path / "chain.jsonl")

    def test_untampered_chain_verifies(self, tmp_path):
        audit = self._make_log(tmp_path)
        audit.log_event("q1", "plan_created", {"q": "a"}, {"plan": "x"})
        audit.log_event("q1", "tool_returned", {"in": 1}, {"out": 2})
        valid, errors = audit.verify_chain()
        assert valid is True
        assert errors == []

    def test_editing_a_past_entry_breaks_the_chain(self, tmp_path):
        log_path = tmp_path / "chain.jsonl"
        audit = self._make_log(tmp_path)
        audit.log_event("q1", "plan_created", {"q": "a"}, {"plan": "x"})
        audit.log_event("q1", "tool_called", {"tool": "vqa"}, {})
        audit.log_event("q1", "answer_emitted", {}, {"answer": "y"})

        lines = log_path.read_text().splitlines()
        first = json.loads(lines[0])
        first["output_hash"] = "0" * 64  # tamper, chain_hash left as-is
        lines[0] = json.dumps(first)
        log_path.write_text("\n".join(lines) + "\n")

        # Fresh reader — verification is over what's on disk, not in-memory.
        fresh = self._make_log(tmp_path)
        valid, errors = fresh.verify_chain()
        assert valid is False
        assert len(errors) > 0

    def test_tamper_invalidates_every_later_entry(self, tmp_path):
        log_path = tmp_path / "chain.jsonl"
        audit = self._make_log(tmp_path)
        for i in range(5):
            audit.log_event(f"q{i}", "tool_called", {"i": i}, {"i": i})

        lines = log_path.read_text().splitlines()
        entry = json.loads(lines[0])
        entry["input_hash"] = "f" * 64
        lines[0] = json.dumps(entry)
        log_path.write_text("\n".join(lines) + "\n")

        fresh = self._make_log(tmp_path)
        valid, errors = fresh.verify_chain()
        assert valid is False
        # Every entry from the tamper point on should report a break
        # (mismatched chain_hash and/or parent_chain_hash), not just entry 0.
        assert len(errors) >= 5

    def test_unmodified_log_survives_process_restart(self, tmp_path):
        audit = self._make_log(tmp_path)
        audit.log_event("q1", "plan_created", {}, {})
        reopened = self._make_log(tmp_path)
        reopened.log_event("q1", "answer_emitted", {}, {})
        valid, errors = reopened.verify_chain()
        assert valid is True
        assert errors == []


# ---------------------------------------------------------------------------
# GUI API-key enforcement
# ---------------------------------------------------------------------------

class TestGuiApiKeyEnforcement:
    """The Flask GUI had zero API-key enforcement despite config.API_KEY /
    AUTH_ENABLED existing for exactly that. Verify the before_request guard
    mirrors the FastAPI side, including that health checks stay open."""

    @pytest.fixture
    def client(self):
        from pipeline.gui import app as gui_app
        gui_app.app.config["TESTING"] = True
        return gui_app.app.test_client(), gui_app

    def test_no_api_key_configured_allows_requests(self, client, monkeypatch):
        test_client, gui_app = client
        monkeypatch.setattr(gui_app, "API_KEY", "")
        monkeypatch.setattr(gui_app, "AUTH_ENABLED", True)
        resp = test_client.get("/api/pipeline/status")
        assert resp.status_code != 401

    def test_auth_disabled_allows_requests_even_with_key_set(self, client, monkeypatch):
        test_client, gui_app = client
        monkeypatch.setattr(gui_app, "API_KEY", "secret123")
        monkeypatch.setattr(gui_app, "AUTH_ENABLED", False)
        resp = test_client.get("/api/pipeline/status")
        assert resp.status_code != 401

    def test_missing_key_rejected_when_enforced(self, client, monkeypatch):
        test_client, gui_app = client
        monkeypatch.setattr(gui_app, "API_KEY", "secret123")
        monkeypatch.setattr(gui_app, "AUTH_ENABLED", True)
        resp = test_client.get("/api/pipeline/status")
        assert resp.status_code == 401

    def test_wrong_key_rejected_when_enforced(self, client, monkeypatch):
        test_client, gui_app = client
        monkeypatch.setattr(gui_app, "API_KEY", "secret123")
        monkeypatch.setattr(gui_app, "AUTH_ENABLED", True)
        resp = test_client.get(
            "/api/pipeline/status", headers={"X-API-Key": "wrong-key"}
        )
        assert resp.status_code == 401

    def test_correct_key_accepted_when_enforced(self, client, monkeypatch):
        test_client, gui_app = client
        monkeypatch.setattr(gui_app, "API_KEY", "secret123")
        monkeypatch.setattr(gui_app, "AUTH_ENABLED", True)
        resp = test_client.get(
            "/api/pipeline/status", headers={"X-API-Key": "secret123"}
        )
        assert resp.status_code != 401

    def test_health_check_stays_open_even_when_enforced(self, client, monkeypatch):
        test_client, gui_app = client
        monkeypatch.setattr(gui_app, "API_KEY", "secret123")
        monkeypatch.setattr(gui_app, "AUTH_ENABLED", True)
        resp = test_client.get("/api/health")
        assert resp.status_code != 401

    def test_non_api_pages_are_not_gated(self, client, monkeypatch):
        """The guard only protects /api/*, matching the FastAPI side's
        per-route Depends() — it never blocks the SPA shell itself."""
        test_client, gui_app = client
        monkeypatch.setattr(gui_app, "API_KEY", "secret123")
        monkeypatch.setattr(gui_app, "AUTH_ENABLED", True)
        resp = test_client.get("/")
        assert resp.status_code != 401
