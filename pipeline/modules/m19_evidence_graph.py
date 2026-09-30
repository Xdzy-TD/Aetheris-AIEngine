"""
M19 — Evidence Graph.

Connects the final answer to the pixels, models, parameters and artifacts
behind it. Nothing here is re-derived or re-scored: every field already
exists somewhere in the pipeline's own output — the executed plan (tool,
arguments), each specialist's own result and artifact record, and
models/registry.json. M19 only assembles what already ran into a graph a
reviewer can walk: answer -> tool/model -> {parameters, images, artifacts}.
Artifact integrity is re-checked via the existing ArtifactStore.verify()
(controller/artifacts.py), the same re-hash `GET /artifacts/{id}` uses, so a
cited artifact can be checked rather than trusted.
"""

from __future__ import annotations

from typing import Any

from controller.model_registry import entry_for_task

from pipeline.adapters.artifact_adapter import ArtifactAdapter
from pipeline.schemas.contracts import ModuleID, ModuleResult


def _consumes_pixels(tool_name: str, registry: Any) -> bool:
    """Whether a tool's own JSON schema declares a ``*_path`` argument.

    Reuses the schema already on file in ``controller/tool_schemas/`` instead
    of hardcoding a second "which tools take images" list that could drift
    from the real one.
    """
    if registry is None:
        return True
    try:
        props = registry.get_tool(tool_name).parameters.get("properties", {})
    except KeyError:
        return True
    return any("path" in key.lower() for key in props)


def build_evidence_graph(
    final_answer: str,
    outputs: list[dict[str, Any]],
    plan_steps: list[dict[str, Any]],
    images: list[dict[str, Any]],
    registry: Any = None,
    artifact_adapter: ArtifactAdapter | None = None,
) -> ModuleResult:
    """Assemble the evidence graph for one query's answer."""
    if not outputs:
        return ModuleResult.skipped(ModuleID.M19, "No specialist outputs to graph")

    store = (artifact_adapter or ArtifactAdapter()).store
    args_by_tool = {s.get("tool_name"): s.get("arguments", {}) for s in plan_steps}

    nodes: list[dict[str, Any]] = [{"id": "answer", "kind": "answer", "text": final_answer}]
    edges: list[dict[str, str]] = []

    image_node_ids: list[str] = []
    for i, img in enumerate(images):
        node_id = f"image:{i}"
        image_node_ids.append(node_id)
        nodes.append({"id": node_id, "kind": "image", **img})

    for out in outputs:
        tool = out.get("tool_name", "")
        result = out.get("result") or {}
        tool_node = f"tool:{tool}"
        entry = entry_for_task(tool)
        nodes.append({
            "id": tool_node,
            "kind": "model",
            "tool_name": tool,
            "model_version": f"{entry['name']}@{entry['version']}" if entry else "unavailable",
            "model_kind": entry.get("kind") if entry else None,
            "ran": not bool(result.get("analysis_unavailable")),
        })
        edges.append({"from": "answer", "to": tool_node, "relation": "used"})

        params = args_by_tool.get(tool)
        if params:
            param_node = f"params:{tool}"
            nodes.append({"id": param_node, "kind": "parameters", "tool_name": tool, "values": params})
            edges.append({"from": tool_node, "to": param_node, "relation": "configured_with"})

        if _consumes_pixels(tool, registry):
            for image_node in image_node_ids:
                edges.append({"from": tool_node, "to": image_node, "relation": "reads_pixels_from"})

        artifact = result.get("artifact")
        if artifact:
            art_id = artifact.get("artifact_id")
            art_node = f"artifact:{art_id or tool}"
            nodes.append({
                "id": art_node,
                "kind": "artifact",
                "artifact_id": art_id,
                "type": artifact.get("type"),
                "sha256": artifact.get("sha256"),
                "model_version": artifact.get("model_version"),
                # Re-hashed now, not trusted from the record — a tampered
                # artifact on disk is caught here, same as GET /artifacts/{id}.
                "sha256_verified": store.verify(art_id) if art_id else None,
            })
            edges.append({"from": tool_node, "to": art_node, "relation": "produced"})

    tampered = [n["id"] for n in nodes if n.get("kind") == "artifact" and n.get("sha256_verified") is False]
    return ModuleResult.success(
        ModuleID.M19,
        {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "tampered_artifacts": tampered,
            "fully_verified": not tampered,
        },
    )
