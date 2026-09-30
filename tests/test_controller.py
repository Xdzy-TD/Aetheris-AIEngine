"""
Tests for the Aetheris controller layer.

Covers:
- Tool registry loading and binding
- Planner routing logic (deterministic fallback planner)
- Routing memory cache hits/misses/eviction
- Audit log hash chain integrity
- Artifact store artifact_id validation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from controller.audit_log import AuditLog
from controller.memory import RoutingMemory
from controller.schemas import (
    ImageMetadata,
    Modality,
    Query,
    ToolCall,
)
from controller.tool_registry import ToolRegistry

# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_load_schemas_from_default_dir(self) -> None:
        """Should load all 5 tool schemas from controller/tool_schemas/."""
        registry = ToolRegistry()
        names = registry.list_tool_names()
        assert "vqa_caption" in names
        assert "sar_bridge" in names
        assert "grounding" in names
        assert "change_detection" in names
        assert "geo_rag" in names
        assert "optical_sar_fusion" in names

    def test_list_tools_shows_unbound(self) -> None:
        """Tools should be listed as unbound before specialist binding."""
        registry = ToolRegistry()
        tools = registry.list_tools()
        for tool in tools:
            assert "name" in tool
            assert "bound" in tool

    def test_bind_and_call(self) -> None:
        """After binding a callable, call_tool should invoke it."""
        registry = ToolRegistry()
        registry.bind("vqa_caption", lambda **kw: {"answer": "test", "raw_confidence": 1.0})
        result = registry.call_tool("vqa_caption", question="What is this?")
        assert result["answer"] == "test"

    def test_call_unbound_raises(self) -> None:
        """Calling an unbound tool should raise RuntimeError."""
        registry = ToolRegistry()
        with pytest.raises(RuntimeError, match="no bound implementation"):
            registry.call_tool("vqa_caption")

    def test_get_schemas_for_planner(self) -> None:
        """Should return schemas in function-calling format."""
        registry = ToolRegistry()
        schemas = registry.get_schemas_for_planner()
        assert len(schemas) >= 5
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]

    def test_register_custom_tool(self) -> None:
        """Should accept a custom tool schema."""
        registry = ToolRegistry()
        registry.register(
            {"name": "custom_tool", "description": "Test tool", "parameters": {}},
            impl=lambda: {"status": "ok"},
        )
        assert "custom_tool" in registry.list_tool_names()
        assert registry.call_tool("custom_tool")["status"] == "ok"


# ---------------------------------------------------------------------------
# Routing Memory
# ---------------------------------------------------------------------------

class TestRoutingMemory:
    """Tests for RoutingMemory."""

    def test_store_and_recall(self) -> None:
        """Should cache and retrieve a result."""
        mem = RoutingMemory()
        mem.store("s1", b"imagedata", "vqa_caption", {"answer": "test"})
        result = mem.recall("s1", b"imagedata", "vqa_caption")
        assert result is not None
        assert result["answer"] == "test"

    def test_miss_on_different_tool(self) -> None:
        """Different tool name should miss."""
        mem = RoutingMemory()
        mem.store("s1", b"imagedata", "vqa_caption", {"answer": "test"})
        assert mem.recall("s1", b"imagedata", "grounding") is None

    def test_miss_on_different_session(self) -> None:
        """Different session should miss."""
        mem = RoutingMemory()
        mem.store("s1", b"imagedata", "vqa_caption", {"answer": "test"})
        assert mem.recall("s2", b"imagedata", "vqa_caption") is None

    def test_eviction_on_capacity(self) -> None:
        """Should evict oldest entry when over max_entries."""
        mem = RoutingMemory(max_entries=3)
        for i in range(5):
            mem.store("s1", f"image{i}".encode(), "vqa_caption", {"i": i})
        stats = mem.stats()
        assert stats["total_entries"] <= 3

    def test_clear_session(self) -> None:
        """Should remove all entries for a session."""
        mem = RoutingMemory()
        mem.store("s1", b"img1", "vqa_caption", {"a": 1})
        mem.store("s1", b"img2", "grounding", {"b": 2})
        mem.store("s2", b"img1", "vqa_caption", {"c": 3})
        evicted = mem.clear_session("s1")
        assert evicted == 2
        assert mem.stats()["total_entries"] == 1

    def test_has_cached(self) -> None:
        """has_cached should not update access stats."""
        mem = RoutingMemory()
        mem.store("s1", b"img", "vqa", {"x": 1})
        assert mem.has_cached("s1", b"img", "vqa") is True
        assert mem.has_cached("s1", b"img", "other") is False


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

class TestAuditLog:
    """Tests for AuditLog hash chain."""

    def _make_audit(self) -> AuditLog:
        """Create an audit log in a temp directory."""
        tmp = Path(tempfile.mkdtemp()) / "test_chain.jsonl"
        return AuditLog(log_path=tmp)

    def test_log_and_verify(self) -> None:
        """A fresh chain with entries should verify as valid."""
        audit = self._make_audit()
        audit.log_event("q1", "plan_created", {"query": "test"}, {"steps": []})
        audit.log_event("q1", "tool_called", {"tool": "vqa"}, {}, tool_name="vqa_caption")
        audit.log_event("q1", "answer_emitted", {}, {"answer": "test"})

        valid, errors = audit.verify_chain()
        assert valid is True
        assert errors == []

    def test_tamper_detection(self) -> None:
        """Modifying an entry should break the chain."""
        audit = self._make_audit()
        audit.log_event("q1", "plan_created", {"q": "a"}, {"s": []})
        audit.log_event("q1", "answer_emitted", {}, {"answer": "b"})

        # Tamper with the log file
        lines = audit._log_path.read_text().splitlines()
        import json
        entry = json.loads(lines[0])
        entry["event_type"] = "TAMPERED"
        lines[0] = json.dumps(entry)
        audit._log_path.write_text("\n".join(lines) + "\n")

        valid, errors = audit.verify_chain()
        assert valid is False
        assert len(errors) > 0

    def test_get_trace(self) -> None:
        """Should return entries filtered by query_id."""
        audit = self._make_audit()
        audit.log_event("q1", "plan_created", {}, {})
        audit.log_event("q2", "plan_created", {}, {})
        audit.log_event("q1", "answer_emitted", {}, {})

        trace = audit.get_trace("q1")
        assert len(trace) == 2
        assert all(e.query_id == "q1" for e in trace)

    def test_chain_recovery(self) -> None:
        """A new AuditLog instance should recover the chain tip."""
        tmp = Path(tempfile.mkdtemp()) / "recovery_test.jsonl"
        audit1 = AuditLog(log_path=tmp)
        audit1.log_event("q1", "plan_created", {}, {})
        chain_tip = audit1._last_chain_hash

        # Create a new instance pointing to the same file
        audit2 = AuditLog(log_path=tmp)
        assert audit2._last_chain_hash == chain_tip

        # Append and verify the combined chain
        audit2.log_event("q2", "answer_emitted", {}, {})
        valid, errors = audit2.verify_chain()
        assert valid is True


# ---------------------------------------------------------------------------
# Planner (deterministic fallback routing)
# ---------------------------------------------------------------------------

class TestPlanner:
    """Tests for the AgenticPlanner deterministic fallback routing logic."""

    def _make_planner(self):
        from controller.planner import AgenticPlanner
        registry = ToolRegistry()
        # Bind all tools to stubs
        for name in registry.list_tool_names():
            registry.bind(name, lambda **kw: {"status": "stub"})
        memory = RoutingMemory()
        audit = AuditLog(log_path=Path(tempfile.mkdtemp()) / "planner_test.jsonl")
        return AgenticPlanner(registry, memory, audit, use_llm=False)

    def test_optical_vqa_routes_to_vqa_only(self) -> None:
        """Single optical image should route to vqa_caption only."""
        planner = self._make_planner()
        query = Query(
            text="What is in this image?",
            images=[ImageMetadata(filename="img.tif", modality=Modality.OPTICAL)],
        )
        plan = planner._plan_rule_based(query)
        tool_names = [s.tool_name for s in plan.steps]
        assert "vqa_caption" in tool_names
        assert "sar_bridge" not in tool_names

    def test_sar_input_adds_sar_bridge(self) -> None:
        """SAR input should prepend sar_bridge before VQA."""
        planner = self._make_planner()
        query = Query(
            text="What is in this image?",
            images=[ImageMetadata(filename="img.tif", modality=Modality.SAR)],
        )
        plan = planner._plan_rule_based(query)
        tool_names = [s.tool_name for s in plan.steps]
        assert tool_names[0] == "sar_bridge"
        assert "vqa_caption" in tool_names

    def test_multi_image_adds_change_detection(self) -> None:
        """Two images should add change_detection."""
        planner = self._make_planner()
        query = Query(
            text="What changed?",
            images=[
                ImageMetadata(filename="t1.tif", modality=Modality.OPTICAL),
                ImageMetadata(filename="t2.tif", modality=Modality.OPTICAL),
            ],
        )
        plan = planner._plan_rule_based(query)
        tool_names = [s.tool_name for s in plan.steps]
        assert "change_detection" in tool_names
        assert "vqa_caption" in tool_names

    def test_cross_modal_pair_routes_to_fusion(self) -> None:
        """One optical + one SAR image should route to optical_sar_fusion,
        not sar_bridge or change_detection."""
        planner = self._make_planner()
        query = Query(
            text="Use the optical and SAR images together to identify built-up and water.",
            images=[
                ImageMetadata(filename="opt.tif", modality=Modality.OPTICAL),
                ImageMetadata(filename="sar.tif", modality=Modality.SAR),
            ],
        )
        plan = planner._plan_rule_based(query)
        tool_names = [s.tool_name for s in plan.steps]
        assert "optical_sar_fusion" in tool_names
        assert "sar_bridge" not in tool_names
        assert "change_detection" not in tool_names
        assert "vqa_caption" in tool_names


# ---------------------------------------------------------------------------
# Image-argument resolution (cross-modal wiring)
# ---------------------------------------------------------------------------

class TestResolveImageArgs:
    """Regression tests: vqa_caption/grounding must not resolve to a blind
    ``images[0]`` in a mixed SAR+optical pair.

    Previously, a cross-modal plan (``optical_sar_fusion`` + ``vqa_caption``,
    with no ``sar_bridge`` step) always handed ``vqa_caption``/``grounding``
    ``images[0]``. If the SAR upload happened to be listed first, the optical
    spectral-index baseline silently ran on SAR backscatter data.
    """

    def _query(self, first_modality: Modality, second_modality: Modality) -> Query:
        return Query(
            text="What is the built-up fraction?",
            images=[
                ImageMetadata(filename="a.tif", modality=first_modality, path="/tmp/a.tif"),
                ImageMetadata(filename="b.tif", modality=second_modality, path="/tmp/b.tif"),
            ],
        )

    def test_vqa_prefers_optical_when_sar_listed_first(self) -> None:
        """SAR-first upload order must not route SAR pixels to the optical baseline."""
        from controller.executor import PlanExecutor

        query = self._query(Modality.SAR, Modality.OPTICAL)
        step = ToolCall(tool_name="vqa_caption", arguments={"question": query.text})

        args = PlanExecutor._resolve_image_args(step, query, prior_results={})

        assert args["image_path"] == "/tmp/b.tif"  # the optical image, not images[0]
        assert args["modality"] == "optical"  # label matches the image actually analyzed

    def test_grounding_prefers_optical_when_sar_listed_first(self) -> None:
        """Same fix applies to grounding, which has no modality label to correct."""
        from controller.executor import PlanExecutor

        query = self._query(Modality.SAR, Modality.OPTICAL)
        step = ToolCall(tool_name="grounding", arguments={"text_prompt": "water"})

        args = PlanExecutor._resolve_image_args(step, query, prior_results={})

        assert args["image_path"] == "/tmp/b.tif"

    def test_vqa_uses_bridged_image_over_raw_optical(self) -> None:
        """A completed sar_bridge step still takes priority over the raw upload."""
        from controller.executor import PlanExecutor

        query = self._query(Modality.SAR, Modality.OPTICAL)
        step = ToolCall(tool_name="vqa_caption", arguments={"question": query.text})
        prior_results = {"sar_bridge": {"bridged_image_path": "/tmp/bridged.tif"}}

        args = PlanExecutor._resolve_image_args(step, query, prior_results)

        assert args["image_path"] == "/tmp/bridged.tif"

    def test_vqa_falls_back_to_first_path_for_sar_only(self) -> None:
        """With no optical image at all, the original images[0] behaviour is preserved."""
        from controller.executor import PlanExecutor

        query = Query(
            text="Describe this image.",
            images=[ImageMetadata(filename="a.tif", modality=Modality.SAR, path="/tmp/a.tif")],
        )
        step = ToolCall(tool_name="vqa_caption", arguments={"question": query.text})

        args = PlanExecutor._resolve_image_args(step, query, prior_results={})

        assert args["image_path"] == "/tmp/a.tif"
        assert args["modality"] == "sar"

    def test_vlm_variants_get_trusted_paths_too(self) -> None:
        """vlm_vqa_caption/vlm_grounding/vlm_change_detection must resolve to the
        real server-side path, exactly like their classical counterparts.

        Regression guard: these three were missing from
        ``AgenticPlanner._IMAGE_ARG_TOOLS`` / ``_resolve_image_args``, so an
        LLM-planned step for one of them skipped the trusted-path override
        entirely and reached the specialist with whatever ``image_path`` the
        plan itself supplied — untrusted, LLM-controlled input handed straight
        to a local file read.
        """
        from controller.executor import _IMAGE_ARG_TOOLS, PlanExecutor

        for tool in ("vlm_vqa_caption", "vlm_grounding"):
            assert tool in _IMAGE_ARG_TOOLS
            query = self._query(Modality.SAR, Modality.OPTICAL)
            step = ToolCall(
                tool_name=tool,
                arguments={"image_path": "/etc/passwd"},  # planner-supplied; must be overridden
            )
            args = PlanExecutor._resolve_image_args(step, query, prior_results={})
            assert args["image_path"] == "/tmp/b.tif"

        assert "vlm_change_detection" in _IMAGE_ARG_TOOLS
        query = self._query(Modality.OPTICAL, Modality.OPTICAL)
        step = ToolCall(
            tool_name="vlm_change_detection",
            arguments={"image_t1_path": "/etc/passwd", "image_t2_path": "/etc/shadow"},
        )
        args = PlanExecutor._resolve_image_args(step, query, prior_results={})
        assert args["image_t1_path"] == "/tmp/a.tif"
        assert args["image_t2_path"] == "/tmp/b.tif"


# ---------------------------------------------------------------------------
# Artifact Store — artifact_id path-traversal guard
# ---------------------------------------------------------------------------

class TestArtifactStoreIdValidation:
    """``GET /artifacts/{artifact_id}`` passes the URL segment straight into
    ``get_record``/``verify``. Regression guard for a path-traversal id
    (e.g. ``../../etc/passwd``) escaping the store's root directory instead
    of being rejected as unknown.
    """

    def test_traversal_id_rejected(self, tmp_path: Path) -> None:
        from controller.artifacts import ArtifactStore

        store = ArtifactStore(root=tmp_path)
        outside = tmp_path.parent / "secret.meta.json"
        outside.write_text('{"leaked": true}', encoding="utf-8")

        assert store.get_record("../secret") is None
        assert store.verify("../secret") is False

    def test_real_id_still_resolves(self, tmp_path: Path) -> None:
        from controller.artifacts import ArtifactStore

        store = ArtifactStore(root=tmp_path)
        record = store.put(b"hello", "mask", "v1")
        assert store.get_record(record["artifact_id"]) == record
        assert store.verify(record["artifact_id"]) is True
