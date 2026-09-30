"""
Tests for M19 — Evidence Graph.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.modules.m19_evidence_graph import build_evidence_graph
from pipeline.schemas.contracts import ResultStatus


class _FakeSpec:
    def __init__(self, properties):
        self.parameters = {"properties": properties}


class _FakeRegistry:
    """Only what M19 touches: get_tool(name).parameters."""
    def __init__(self, specs):
        self._specs = specs

    def get_tool(self, name):
        return self._specs[name]


class _FakeStore:
    def __init__(self, verified=True):
        self._verified = verified

    def verify(self, artifact_id):
        return self._verified


class _FakeArtifactAdapter:
    def __init__(self, verified=True):
        self.store = _FakeStore(verified)


IMAGES = [{"path": "t1.tif", "modality": "optical"}, {"path": "t2.tif", "modality": "optical"}]
PLAN_STEPS = [{"tool_name": "change_detection", "arguments": {"image_t1_path": "t1.tif", "image_t2_path": "t2.tif"}}]
REGISTRY = _FakeRegistry({"change_detection": _FakeSpec({"image_t1_path": {}, "image_t2_path": {}})})


class TestM19EvidenceGraph:
    def test_skips_with_no_outputs(self):
        result = build_evidence_graph("answer", [], PLAN_STEPS, IMAGES, REGISTRY)
        assert result.status == ResultStatus.SKIPPED

    def test_builds_nodes_and_edges_without_artifact(self):
        outputs = [{"tool_name": "change_detection", "result": {"changed_area_pct": 5.0}}]
        result = build_evidence_graph("Area changed", outputs, PLAN_STEPS, IMAGES, REGISTRY)

        assert result.status == ResultStatus.SUCCESS
        kinds = {n["kind"] for n in result.data["nodes"]}
        assert kinds == {"answer", "image", "model", "parameters"}
        relations = {e["relation"] for e in result.data["edges"]}
        assert relations == {"used", "configured_with", "reads_pixels_from"}
        # a pixel-consuming tool reads from every image supplied
        pixel_edges = [e for e in result.data["edges"] if e["relation"] == "reads_pixels_from"]
        assert len(pixel_edges) == len(IMAGES)

    def test_artifact_node_re_verifies_hash(self):
        outputs = [{
            "tool_name": "change_detection",
            "result": {
                "changed_area_pct": 5.0,
                "artifact": {"artifact_id": "a1", "type": "geotiff", "sha256": "deadbeef"},
            },
        }]
        result = build_evidence_graph(
            "Area changed", outputs, PLAN_STEPS, IMAGES, REGISTRY,
            artifact_adapter=_FakeArtifactAdapter(verified=True),
        )
        art_node = next(n for n in result.data["nodes"] if n["kind"] == "artifact")
        assert art_node["sha256_verified"] is True
        assert result.data["fully_verified"] is True
        assert result.data["tampered_artifacts"] == []

    def test_tampered_artifact_is_flagged_and_not_trusted(self):
        outputs = [{
            "tool_name": "change_detection",
            "result": {"artifact": {"artifact_id": "a1", "type": "geotiff", "sha256": "deadbeef"}},
        }]
        result = build_evidence_graph(
            "Area changed", outputs, PLAN_STEPS, IMAGES, REGISTRY,
            artifact_adapter=_FakeArtifactAdapter(verified=False),
        )
        assert result.data["fully_verified"] is False
        assert result.data["tampered_artifacts"] == ["artifact:a1"]

    def test_tool_missing_from_registry_conservatively_assumes_pixels(self):
        outputs = [{"tool_name": "unknown_tool", "result": {}}]
        result = build_evidence_graph("x", outputs, [], IMAGES, REGISTRY)
        pixel_edges = [e for e in result.data["edges"] if e["relation"] == "reads_pixels_from"]
        assert len(pixel_edges) == len(IMAGES)

    def test_no_registry_conservatively_assumes_pixels(self):
        outputs = [{"tool_name": "change_detection", "result": {}}]
        result = build_evidence_graph("x", outputs, PLAN_STEPS, IMAGES, registry=None)
        pixel_edges = [e for e in result.data["edges"] if e["relation"] == "reads_pixels_from"]
        assert len(pixel_edges) == len(IMAGES)
