"""
M04 — Authentication & Authorization.

Wraps the old API security with identity context, roles, permissions,
and authorization decisions.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

try:
    import argon2
    from argon2 import PasswordHasher as _Argon2Hasher
    from argon2.exceptions import VerifyMismatchError as _VerifyMismatch

    # Explicitly require Argon2id — the OWASP-recommended variant.
    _ph = _Argon2Hasher(type=argon2.Type.ID)
    _ARGON2_AVAILABLE = True
except ImportError:
    # Degraded fallback: argon2-cffi is not installed. Use stdlib scrypt
    # with a fixed salt (the old behaviour). This keeps the system
    # functional on minimal installs at the cost of weaker hashing.
    import hashlib
    import warnings

    warnings.warn(
        "argon2-cffi is not installed — password hashing falls back to "
        "stdlib scrypt. Install argon2-cffi>=23.1 for Argon2id (recommended).",
        stacklevel=1,
    )
    _ARGON2_AVAILABLE = False

from pipeline.config import DEMO_MODE
from pipeline.schemas.contracts import ModuleID, ModuleResult


class Role:
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    ANONYMOUS = "anonymous"


ROLE_PERMISSIONS: dict[str, set[str]] = {
    Role.ADMIN: {"query", "upload", "audit", "manage", "configure", "delete"},
    Role.ANALYST: {"query", "upload", "audit"},
    Role.VIEWER: {"query", "audit"},
    Role.ANONYMOUS: {"query"},
}


class Identity:
    """Represents an authenticated user identity."""

    def __init__(
        self,
        user_id: str = "anonymous",
        role: str = Role.ANONYMOUS,
        session_token: str = "",
    ) -> None:
        self.user_id = user_id
        self.role = role
        self.session_token = session_token or uuid.uuid4().hex[:16]
        self.authenticated_at = time.time()
        self.permissions = ROLE_PERMISSIONS.get(role, set())

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "role": self.role,
            "session_token": self.session_token,
            "authenticated_at": self.authenticated_at,
            "permissions": sorted(self.permissions),
        }


class AuthManager:
    """Authentication and authorization manager for M04."""

    def __init__(self) -> None:
        self._sessions: dict[str, Identity] = {}
        # Built-in users for hackathon demo — seeded only when AETHERIS_DEMO_MODE
        # is on. Previously unconditional: docker-compose.yml set the flag but
        # nothing read it, so admin/admin, analyst/analyst, viewer/viewer shipped
        # in every deployment regardless of the setting.
        self._users: dict[str, dict[str, str]] = (
            {
                "admin": {"password_hash": self._hash("admin"), "role": Role.ADMIN},
                "analyst": {"password_hash": self._hash("analyst"), "role": Role.ANALYST},
                "viewer": {"password_hash": self._hash("viewer"), "role": Role.VIEWER},
            }
            if DEMO_MODE
            else {}
        )

    @staticmethod
    def _hash(password: str) -> str:
        """Hash a password using Argon2id (preferred) or scrypt (fallback).

        argon2-cffi generates a random per-user salt internally and encodes
        it into the output string, so two users with the same password get
        different hashes — unlike the old fixed-salt scrypt approach.
        Argon2id is the OWASP-recommended variant (memory-hard + data-independent).
        """
        if _ARGON2_AVAILABLE:
            return _ph.hash(password)
        # Fallback: stdlib scrypt with a fixed salt. Acceptable only for
        # demo accounts gated by DEMO_MODE.
        salt = b"aetheris-demo-salt"
        return hashlib.scrypt(
            password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32
        ).hex()

    @staticmethod
    def _verify(stored_hash: str, password: str) -> bool:
        """Verify a password against a stored hash (timing-safe)."""
        if _ARGON2_AVAILABLE:
            try:
                return _ph.verify(stored_hash, password)
            except _VerifyMismatch:
                return False
        # Fallback: recompute and constant-time compare.
        import hmac

        salt = b"aetheris-demo-salt"
        candidate = hashlib.scrypt(
            password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32
        ).hex()
        return hmac.compare_digest(stored_hash, candidate)

    def authenticate(self, user_id: str, password: str) -> ModuleResult:
        """Authenticate a user and create a session.

        Three distinct outcomes, not two:
        - No credentials supplied at all (blank user_id/password, or the
          literal "anonymous" user) -> anonymous session, granted.
        - A known user_id with the correct password -> session for that
          user's real role.
        - A known user_id with an INCORRECT password -> authentication
          FAILURE. This used to fall through to a silent anonymous grant,
          which meant a wrong password was never actually rejected and
          M04 could never distinguish "bad credentials" from "no
          credentials" — a real bug, not just a design choice.
        """
        t0 = time.perf_counter()

        # Explicit anonymous request: no credentials supplied.
        if not user_id or user_id == "anonymous":
            identity = Identity(user_id="anonymous", role=Role.ANONYMOUS)
            self._sessions[identity.session_token] = identity
            return ModuleResult.success(
                ModuleID.M04,
                {
                    "authenticated": True,
                    "identity": identity.to_dict(),
                    "session_token": identity.session_token,
                    "note": "Anonymous access granted",
                },
                exec_time=round(time.perf_counter() - t0, 4),
            )

        user = self._users.get(user_id)
        if user and self._verify(user["password_hash"], password):
            identity = Identity(
                user_id=user_id,
                role=user["role"],
            )
            self._sessions[identity.session_token] = identity
            return ModuleResult.success(
                ModuleID.M04,
                {
                    "authenticated": True,
                    "identity": identity.to_dict(),
                    "session_token": identity.session_token,
                },
                exec_time=round(time.perf_counter() - t0, 4),
            )

        # Known or unknown user_id with the wrong (or missing) password:
        # a real authentication failure, not a silent downgrade.
        result = ModuleResult.failure(
            ModuleID.M04,
            "AUTH_FAILED",
            f"Invalid credentials for user '{user_id}'",
        )
        result.execution_time_s = round(time.perf_counter() - t0, 4)
        return result

    def authorize(self, session_token: str, action: str) -> ModuleResult:
        """Check if a session has permission to perform an action."""
        t0 = time.perf_counter()
        identity = self._sessions.get(session_token)
        if identity is None:
            # Unknown token -> transient anonymous identity, NOT persisted.
            # Persisting one entry per arbitrary caller-supplied token let an
            # attacker grow self._sessions without bound (memory-exhaustion
            # DoS); real sessions are only ever added by authenticate().
            identity = Identity(user_id="anonymous", role=Role.ANONYMOUS)

        allowed = identity.has_permission(action)
        data = {
            "authorized": allowed,
            "user_id": identity.user_id,
            "role": identity.role,
            "action": action,
            "permissions": sorted(identity.permissions),
        }

        if allowed:
            return ModuleResult.success(ModuleID.M04, data,
                                       exec_time=round(time.perf_counter() - t0, 4))
        return ModuleResult.failure(
            ModuleID.M04,
            "UNAUTHORIZED",
            f"User '{identity.user_id}' ({identity.role}) lacks permission '{action}'",
        )

    def get_session(self, session_token: str) -> Identity | None:
        return self._sessions.get(session_token)

    def get_status(self) -> dict[str, Any]:
        return {
            "active_sessions": len(self._sessions),
            "registered_users": len(self._users),
            "roles": list(ROLE_PERMISSIONS.keys()),
        }
