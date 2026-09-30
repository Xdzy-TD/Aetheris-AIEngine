"""Observability: Prometheus metrics + a request ID on every log line.

One counter, one histogram, one middleware — this is scrape-and-correlate,
not a tracing stack. Nothing here claims to be more than that.
"""

from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

import structlog
from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "aetheris_http_requests_total",
    "HTTP requests handled",
    ["method", "route", "status"],
)
REQUEST_LATENCY = Histogram(
    "aetheris_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
)


async def observability_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Bind a request ID to every log line for this request and record metrics.

    The route template (``/v1/audit/trace/{query_id}``), not the raw path, is
    the metric label — a raw path would give every distinct query_id its own
    time series.
    """
    request_id = uuid.uuid4().hex[:12]
    structlog.contextvars.bind_contextvars(request_id=request_id)
    t0 = time.perf_counter()
    status_code = 500  # stays 500 only if call_next raises before returning
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        # Runs on every exit path, including an unhandled exception from
        # call_next — previously that path skipped straight past the metrics
        # calls below it, so a request that errored was never counted or
        # timed. The exception itself is not caught here, so it still
        # propagates to FastAPI's normal error handling.
        route = request.scope.get("route")
        route_path = route.path if route is not None else request.url.path
        REQUEST_LATENCY.labels(request.method, route_path).observe(time.perf_counter() - t0)
        REQUEST_COUNT.labels(request.method, route_path, status_code).inc()
        structlog.contextvars.unbind_contextvars("request_id")


def metrics_response() -> Response:
    """Render current metrics in the Prometheus text exposition format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
