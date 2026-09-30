"""
Job Manager — manages pipeline job lifecycle.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from pipeline.modules.m05_rate_limit import ConcurrencyManager, JobState


class JobManager:
    """Manages pipeline job submission and tracking."""

    def __init__(self, max_concurrent: int = 4) -> None:
        self._concurrency = ConcurrencyManager(max_concurrent)
        self._results: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, user_id: str = "anonymous") -> dict[str, Any]:
        """Submit a new job."""
        result = self._concurrency.submit_job(user_id)
        job_id = result.data.get("job_id", uuid.uuid4().hex[:12])
        with self._lock:
            self._results[job_id] = {"status": "submitted", "submitted_at": time.time()}
        return {"job_id": job_id, "result": result.data}

    def complete(self, job_id: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        """Mark a job as complete."""
        self._concurrency.complete_job(job_id, error)
        with self._lock:
            self._results[job_id] = {
                "status": "failed" if error else "complete",
                "completed_at": time.time(),
                "result": result,
                "error": error,
            }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._concurrency.get_job(job_id)

    def get_status(self) -> dict[str, Any]:
        return self._concurrency.get_status()

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._concurrency.list_jobs(limit)
