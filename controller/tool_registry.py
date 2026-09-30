"""
Tool Registry — JSON-schema-driven catalog of specialist tools.

Each tool is described by a JSON schema file that specifies its name,
description, input parameters, and output format — mirroring the
function-calling / MCP paradigm described in §2.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default location for tool schema JSON files
_SCHEMAS_DIR = Path(__file__).parent / "tool_schemas"


class ToolSpec:
    """A registered tool: its JSON schema plus a callable implementation."""

    def __init__(
        self,
        schema: dict[str, Any],
        implementation: Callable[..., Any] | None = None,
    ) -> None:
        self.name: str = schema["name"]
        self.description: str = schema.get("description", "")
        self.parameters: dict[str, Any] = schema.get("parameters", {})
        self.returns: dict[str, Any] = schema.get("returns", {})
        self.schema = schema
        self.implementation = implementation

    def __repr__(self) -> str:
        bound = "bound" if self.implementation else "unbound"
        return f"<ToolSpec {self.name!r} ({bound})>"


class ToolRegistry:
    """Registry of specialist tools available to the agentic controller.

    Tools are loaded from JSON schema files in ``controller/tool_schemas/``
    at startup.  Specialist modules register their callable implementations
    via :meth:`bind`.
    """

    def __init__(self, schemas_dir: Path | str | None = None) -> None:
        self._schemas_dir = Path(schemas_dir) if schemas_dir else _SCHEMAS_DIR
        self._tools: dict[str, ToolSpec] = {}
        self._load_schemas()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, schema: dict[str, Any], impl: Callable[..., Any] | None = None) -> None:
        """Register a tool from a raw schema dict (and optional callable)."""
        spec = ToolSpec(schema, implementation=impl)
        self._tools[spec.name] = spec
        logger.info("tool_registered", tool=spec.name, bound=impl is not None)

    def bind(self, tool_name: str, impl: Callable[..., Any]) -> None:
        """Bind a callable implementation to an already-registered tool."""
        if tool_name not in self._tools:
            raise KeyError(f"Tool {tool_name!r} not found in registry")
        self._tools[tool_name].implementation = impl
        logger.info("tool_bound", tool=tool_name)

    def get_tool(self, tool_name: str) -> ToolSpec:
        """Retrieve a tool spec by name."""
        if tool_name not in self._tools:
            raise KeyError(f"Tool {tool_name!r} not found in registry")
        return self._tools[tool_name]

    def list_tools(self) -> list[dict[str, Any]]:
        """Return a summary of all registered tools (for the /tools endpoint)."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "bound": spec.implementation is not None,
                "parameters": spec.parameters,
            }
            for spec in self._tools.values()
        ]

    def list_tool_names(self) -> list[str]:
        """Return just the tool names."""
        return list(self._tools.keys())

    def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke a tool by name. Raises if the tool is unbound."""
        spec = self.get_tool(tool_name)
        if spec.implementation is None:
            raise RuntimeError(
                f"Tool {tool_name!r} is registered but has no bound implementation. "
                f"Bind it via registry.bind('{tool_name}', callable)."
            )
        return spec.implementation(**kwargs)

    def get_schemas_for_planner(self) -> list[dict[str, Any]]:
        """Return tool schemas in the format expected by an LLM function-calling prompt."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._tools.values()
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_schemas(self) -> None:
        """Load all .json files from the schemas directory."""
        if not self._schemas_dir.exists():
            logger.warning("tool_schemas_dir_missing", path=str(self._schemas_dir))
            return
        for schema_path in sorted(self._schemas_dir.glob("*.json")):
            try:
                with schema_path.open("r", encoding="utf-8") as f:
                    schema = json.load(f)
                self.register(schema)
            except Exception as exc:
                logger.error("tool_schema_load_failed", path=str(schema_path), error=str(exc))
