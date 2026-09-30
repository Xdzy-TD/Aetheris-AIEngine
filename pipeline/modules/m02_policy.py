"""
M02 — JSON Schema Policy Engine.

Wraps the existing ``controller.policy`` engine and adds new validation
rules (file constraints, CRS requirements, upload policies) on top.
"""

from __future__ import annotations

import time
from typing import Any

from pipeline.adapters.policy_adapter import PolicyAdapter
from pipeline.schemas.contracts import ModuleID, ModuleResult


# Additional policy rules not covered by the old engine
_MAX_QUESTION_LENGTH = 2000
_MIN_QUESTION_LENGTH = 3
_FORBIDDEN_PATTERNS = ["../", "..\\", "<script", "javascript:", "data:text"]


class PolicyEngine:
    """Extended policy engine wrapping the old AETHERIS policy."""

    def __init__(self) -> None:
        self._adapter = PolicyAdapter()
        self._custom_rules: list[dict[str, Any]] = []

    @property
    def adapter(self) -> PolicyAdapter:
        return self._adapter

    def add_rule(self, name: str, check_fn: Any, description: str = "") -> None:
        """Add a custom policy rule."""
        self._custom_rules.append({
            "name": name,
            "check": check_fn,
            "description": description,
        })

    def validate_request(
        self,
        question: str,
        images: list[dict[str, Any]] | None = None,
        modality: str = "optical",
        file_sizes: list[int] | None = None,
    ) -> ModuleResult:
        """Full policy validation: custom rules + old engine."""
        t0 = time.perf_counter()
        violations: list[str] = []
        warnings: list[str] = []

        # Custom rule: question length
        if len(question) < _MIN_QUESTION_LENGTH:
            violations.append(f"Question too short (min {_MIN_QUESTION_LENGTH} characters)")
        if len(question) > _MAX_QUESTION_LENGTH:
            violations.append(f"Question too long (max {_MAX_QUESTION_LENGTH} characters)")

        # Custom rule: forbidden patterns in question
        q_lower = question.lower()
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern in q_lower:
                violations.append(f"Forbidden pattern detected in question: {pattern!r}")

        # Custom rule: file size limits
        if file_sizes:
            from pipeline.config import MAX_UPLOAD_BYTES
            for i, size in enumerate(file_sizes):
                if size > MAX_UPLOAD_BYTES:
                    violations.append(
                        f"Image {i+1} exceeds max upload size "
                        f"({size // (1024*1024)} MB > {MAX_UPLOAD_BYTES // (1024*1024)} MB)"
                    )

        # Custom rule: modality validation
        valid_modalities = {"optical", "sar", "multispectral", "unknown"}
        if modality not in valid_modalities:
            warnings.append(f"Unknown modality '{modality}', defaulting to 'unknown'")

        # Run custom rules
        for rule in self._custom_rules:
            try:
                result = rule["check"](question=question, images=images, modality=modality)
                if result is not True:
                    violations.append(f"Custom rule '{rule['name']}': {result}")
            except Exception as exc:
                warnings.append(f"Custom rule '{rule['name']}' raised: {exc}")

        # If custom rules already found violations, return early
        if violations:
            result = ModuleResult.failure(
                ModuleID.M02,
                "POLICY_VIOLATION",
                "; ".join(violations),
            )
            result.warnings = warnings
            result.execution_time_s = round(time.perf_counter() - t0, 4)
            return result

        # Delegate to old policy engine through adapter
        old_result = self._adapter.validate_request(question, images, modality)
        old_result.warnings.extend(warnings)
        old_result.execution_time_s = round(time.perf_counter() - t0, 4)
        return old_result

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas."""
        return self._adapter.get_tool_schemas()

    def get_policy_summary(self) -> dict[str, Any]:
        """Return a summary of all policy rules in effect."""
        return {
            "builtin_rules": [
                "question_length_check",
                "forbidden_pattern_check",
                "file_size_check",
                "modality_validation",
            ],
            "old_engine_rules": [
                "tool_existence_check",
                "tool_binding_check",
                "image_count_check",
                "modality_precondition_check",
                "dependency_order_check",
                "argument_schema_check",
            ],
            "custom_rules": [r["name"] for r in self._custom_rules],
            "max_question_length": _MAX_QUESTION_LENGTH,
            "min_question_length": _MIN_QUESTION_LENGTH,
            "forbidden_patterns": _FORBIDDEN_PATTERNS,
        }
