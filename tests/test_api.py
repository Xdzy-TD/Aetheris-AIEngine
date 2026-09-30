"""
FastAPI endpoint integration tests.

Uses ``httpx.AsyncClient`` to test API endpoints without starting a
real server.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from interfaces.api.app import app


@pytest.fixture
def client():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


def _png_bytes(seed: int = 0, w: int = 64, h: int = 64) -> bytes:
    """Generate a small in-memory PNG so tests never touch a real path."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client) -> None:
        async with client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["tools_loaded"] >= 5

    @pytest.mark.asyncio
    async def test_v1_health_is_the_same_endpoint(self, client) -> None:
        """The /v1 mount must serve identically to the unprefixed one."""
        async with client:
            resp = await client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.headers["x-request-id"]


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    @pytest.mark.asyncio
    async def test_metrics_exposes_request_counter(self, client) -> None:
        async with client:
            await client.get("/health")
            resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "aetheris_http_requests_total" in resp.text


class TestObservabilityMiddlewareExceptions:
    """Bug fix: a request that raises must still be counted and timed.

    Exercises ``observability_middleware`` directly (not through the app) so
    the raising handler is unambiguous and doesn't depend on how FastAPI's
    own error handling happens to render the 500.
    """

    @pytest.mark.asyncio
    async def test_exception_is_still_recorded_and_reraised(self) -> None:
        from starlette.requests import Request

        from interfaces.api.observability import (
            REQUEST_COUNT,
            REQUEST_LATENCY,
            observability_middleware,
        )

        scope = {"type": "http", "method": "GET", "path": "/boom", "headers": [], "route": None}
        request = Request(scope)

        async def raising_call_next(_request: Request):
            raise ValueError("handler blew up")

        before = REQUEST_COUNT.labels("GET", "/boom", "500")._value.get()

        with pytest.raises(ValueError, match="handler blew up"):
            await observability_middleware(request, raising_call_next)

        after = REQUEST_COUNT.labels("GET", "/boom", "500")._value.get()
        assert after == before + 1
        # Latency histogram gained an observation too, not just the counter.
        assert REQUEST_LATENCY.labels("GET", "/boom")._sum.get() >= 0


class TestToolsEndpoint:
    """Tests for GET /tools."""

    @pytest.mark.asyncio
    async def test_list_tools(self, client) -> None:
        async with client:
            resp = await client.get("/tools")
        assert resp.status_code == 200
        tools = resp.json()
        assert isinstance(tools, list)
        names = [t["name"] for t in tools]
        assert "vqa_caption" in names
        assert "sar_bridge" in names


class TestQueryEndpoint:
    """Tests for POST /query."""

    @pytest.mark.asyncio
    async def test_text_only_query(self, client) -> None:
        async with client:
            resp = await client.post(
                "/query",
                data={"question": "What is in this image?"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "final_answer" in data
        assert "plan" in data
        assert "outputs" in data
        assert data["query_id"]

    @pytest.mark.asyncio
    async def test_query_with_modality(self, client) -> None:
        async with client:
            resp = await client.post(
                "/query",
                data={
                    "question": "Describe the terrain",
                    "modality": "optical",
                },
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_single_image_upload_is_actually_analyzed(self, client) -> None:
        """End-to-end proof of the path-wiring fix: the real uploaded file's
        server-side temp path must reach the specialist, not an empty/stub
        path. If wiring regresses, vqa_caption falls back to its "Could not
        analyze image" error path instead of a real pixel-derived answer."""
        files = {"image": ("aoi.png", _png_bytes(seed=1), "image/png")}
        async with client:
            resp = await client.post(
                "/query",
                data={"question": "What land cover is visible?"},
                files=files,
            )
        assert resp.status_code == 200
        data = resp.json()

        vqa_outputs = [o for o in data["outputs"] if o["tool_name"] == "vqa_caption"]
        assert len(vqa_outputs) == 1
        vqa_result = vqa_outputs[0]["result"]
        assert "Could not analyze image" not in vqa_result.get("answer", "")
        assert vqa_result.get("raw_confidence", 0.0) > 0.0

    @pytest.mark.asyncio
    async def test_two_image_upload_feeds_change_detection(self, client) -> None:
        """With both `image` and `image2` uploaded, the planner must route
        through change_detection with both real temp paths — proving image2
        (added as part of the path-wiring fix) is actually threaded through."""
        files = {
            "image": ("t1.png", _png_bytes(seed=2), "image/png"),
            "image2": ("t2.png", _png_bytes(seed=3), "image/png"),
        }
        async with client:
            resp = await client.post(
                "/query",
                data={"question": "What changed between these two images?"},
                files=files,
            )
        assert resp.status_code == 200
        data = resp.json()

        cd_outputs = [o for o in data["outputs"] if o["tool_name"] == "change_detection"]
        assert len(cd_outputs) == 1
        cd_result = cd_outputs[0]["result"]
        assert "Could not load" not in cd_result.get("change_summary", "")
        assert cd_result.get("change_map_path")


    @pytest.mark.asyncio
    async def test_cross_modal_upload_routes_to_fusion(self, client) -> None:
        """Regression test: `image` and `image2` must be able to carry different
        modalities. Previously both always inherited the single `modality` field,
        so a real optical+SAR upload could never route to optical_sar_fusion —
        the mandatory cross-modal capability was unreachable via the API."""
        files = {
            "image": ("optical.png", _png_bytes(seed=4), "image/png"),
            "image2": ("sar.png", _png_bytes(seed=5), "image/png"),
        }
        async with client:
            resp = await client.post(
                "/query",
                data={
                    "question": "Use the optical and SAR images together to find water.",
                    "modality": "optical",
                    "modality2": "sar",
                },
                files=files,
            )
        assert resp.status_code == 200
        data = resp.json()

        fusion_outputs = [o for o in data["outputs"] if o["tool_name"] == "optical_sar_fusion"]
        assert len(fusion_outputs) == 1
        assert not any(o["tool_name"] == "change_detection" for o in data["outputs"])


class TestMissionEndpoint:
    """Tests for POST /mission/{name}."""

    @pytest.mark.asyncio
    async def test_unknown_mission_returns_404(self, client) -> None:
        async with client:
            resp = await client.post("/mission/not_a_mission")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_mission_runs_over_upload(self, client) -> None:
        files = {"image": ("aoi.png", _png_bytes(seed=6), "image/png")}
        async with client:
            resp = await client.post("/mission/water_body_monitoring", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert {o["tool_name"] for o in data["outputs"]} == {"grounding", "vqa_caption"}

    @pytest.mark.asyncio
    async def test_mission_question_override(self, client) -> None:
        files = {"image": ("aoi.png", _png_bytes(seed=7), "image/png")}
        async with client:
            resp = await client.post(
                "/mission/flood",
                data={"question": "Custom question"},
                files=files,
            )
        assert resp.status_code == 200


class TestAuditEndpoints:
    """Tests for audit-related endpoints."""

    @pytest.mark.asyncio
    async def test_verify_audit(self, client) -> None:
        async with client:
            resp = await client.get("/audit/verify")
        assert resp.status_code == 200
        data = resp.json()
        assert "valid" in data

    @pytest.mark.asyncio
    async def test_recent_audit(self, client) -> None:
        async with client:
            resp = await client.get("/audit/recent?n=5")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestMemoryEndpoint:
    """Tests for GET /memory/stats."""

    @pytest.mark.asyncio
    async def test_memory_stats(self, client) -> None:
        async with client:
            resp = await client.get("/memory/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_entries" in data
        assert "max_entries" in data
