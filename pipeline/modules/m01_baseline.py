"""
M01 — Baseline & Dependency Health Check.

Inspects the runtime environment and validates that all old AETHERIS
components can be imported and used through adapters.  Returns a
structured health report via the M08 contract.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
import time
from typing import Any

from pipeline.schemas.contracts import ModuleID, ModuleResult


def _check_import(module_name: str) -> dict[str, Any]:
    """Try to import a module and return status."""
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
        return {"module": module_name, "available": True, "version": str(version)}
    except ImportError as exc:
        return {"module": module_name, "available": False, "error": str(exc)}
    except Exception as exc:
        return {"module": module_name, "available": False, "error": str(exc)}


def _check_old_components() -> list[dict[str, Any]]:
    """Validate that old AETHERIS components can be imported."""
    components = [
        "controller.schemas",
        "controller.policy",
        "controller.planner",
        "controller.tool_registry",
        "controller.memory",
        "controller.audit_log",
        "controller.artifacts",
        "controller.model_registry",
        "controller.ollama_client",
        "controller.llm_planner",
        "specialists._imagery",
        "specialists.base",
        "confidence.conformal",
        "confidence.self_consistency",
        "interfaces.api.security",
        "interfaces.api.deps",
        "preprocessing",
    ]
    return [_check_import(c) for c in components]


def _check_dependencies() -> list[dict[str, Any]]:
    """Check critical Python dependencies."""
    deps = [
        "fastapi", "uvicorn", "pydantic", "structlog", "httpx",
        "numpy", "PIL", "scipy", "slowapi", "ollama",
    ]
    results = []
    for dep in deps:
        results.append(_check_import(dep))

    # Optional geo dependencies
    for dep in ["rasterio", "qdrant_client", "sentence_transformers"]:
        results.append(_check_import(dep))

    return results


def _check_directories() -> list[dict[str, Any]]:
    """Check required directories exist or can be created."""
    from pipeline.config import UPLOAD_DIR, ARTIFACTS_DIR, LOGS_DIR

    dirs = [
        ("upload_dir", UPLOAD_DIR),
        ("artifacts_dir", ARTIFACTS_DIR),
        ("logs_dir", LOGS_DIR),
    ]
    results = []
    for name, path in dirs:
        exists = path.exists()
        if not exists:
            try:
                path.mkdir(parents=True, exist_ok=True)
                results.append({"name": name, "path": str(path), "status": "created"})
            except Exception as exc:
                results.append({"name": name, "path": str(path), "status": "error", "error": str(exc)})
        else:
            results.append({"name": name, "path": str(path), "status": "exists"})
    return results


def _check_ollama() -> dict[str, Any]:
    """Check Ollama availability."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return {"available": True, "models": models, "model_count": len(models)}
        return {"available": False, "status_code": resp.status_code}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _check_tool_registry() -> dict[str, Any]:
    """Check the old tool registry.

    Bug (same class as the one fixed in policy_adapter.py): this used to
    build its own bare ToolRegistry() with nothing bound, so the dashboard
    always reported "bound": 0 even though the real singleton registry
    (interfaces.api.deps.get_registry()) has all 9 specialists bound.
    Reuse that singleton instead of a second, permanently-unbound one.
    """
    try:
        from interfaces.api.deps import get_registry
        registry = get_registry()
        tools = registry.list_tools()
        return {
            "available": True,
            "tool_count": len(tools),
            "tools": [t["name"] for t in tools],
            "tools_bound": {t["name"]: bool(t.get("bound")) for t in tools},
            "bound": sum(1 for t in tools if t.get("bound")),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _check_audit_log() -> dict[str, Any]:
    """Check the old audit log."""
    try:
        from controller.audit_log import AuditLog
        audit = AuditLog()
        valid, errors = audit.verify_chain()
        entries = audit.get_last_n(5)
        return {
            "available": True,
            "chain_valid": valid,
            "chain_errors": errors,
            "recent_entries": len(entries),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _check_model_registry() -> dict[str, Any]:
    """Check the model registry."""
    try:
        from controller.model_registry import all_models
        models = all_models()
        return {
            "available": True,
            "model_count": len(models),
            "models": [
                {"task": m.get("task"), "name": m.get("name"), "kind": m.get("kind")}
                for m in models
            ],
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def run_health_check() -> ModuleResult:
    """Run the complete M01 baseline health check."""
    t0 = time.perf_counter()

    report: dict[str, Any] = {
        "system": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "os": platform.system(),
            "architecture": platform.architecture()[0],
            "cwd": os.getcwd(),
        },
        "old_components": _check_old_components(),
        "dependencies": _check_dependencies(),
        "directories": _check_directories(),
        "ollama": _check_ollama(),
        "tool_registry": _check_tool_registry(),
        "audit_log": _check_audit_log(),
        "model_registry": _check_model_registry(),
    }

    # Compute overall health
    component_ok = sum(1 for c in report["old_components"] if c["available"])
    component_total = len(report["old_components"])
    dep_ok = sum(1 for d in report["dependencies"] if d["available"])
    dep_total = len(report["dependencies"])

    report["summary"] = {
        "components_available": f"{component_ok}/{component_total}",
        "dependencies_available": f"{dep_ok}/{dep_total}",
        "ollama_available": report["ollama"].get("available", False),
        "tool_registry_available": report["tool_registry"].get("available", False),
        "audit_log_valid": report["audit_log"].get("chain_valid", False),
        "overall_healthy": component_ok >= component_total - 2 and dep_ok >= 8,
    }

    elapsed = round(time.perf_counter() - t0, 4)

    if report["summary"]["overall_healthy"]:
        return ModuleResult.success(ModuleID.M01, report, exec_time=elapsed)
    else:
        result = ModuleResult.failure(
            ModuleID.M01,
            "HEALTH_CHECK_DEGRADED",
            "Some components are unavailable — see report for details",
        )
        result.data = report
        result.execution_time_s = elapsed
        return result
