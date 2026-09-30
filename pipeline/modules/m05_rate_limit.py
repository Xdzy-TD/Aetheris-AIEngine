"""
M05 — Rate Limiting & Concurrency Control.

Implements actual request rate limiting and job concurrency management.
Integrates with the existing SlowAPI dependency where possible.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from enum import StrEnum
from typing import Any

from pipeline.schemas.contracts import ModuleID, ModuleResult


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"


class JobInfo:
    def __init__(self, job_id: str, user_id: str = "anonymous") -> None:
        self.job_id = job_id
        self.user_id = user_id
        self.state = JobState.QUEUED
        self.created_at = time.time()
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": (
                round((self.completed_at or time.time()) - (self.started_at or self.created_at), 3)
            ),
            "error": self.error,
        }


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self._rpm = requests_per_minute
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, user_id: str = "global") -> ModuleResult:
        """Check if a request is allowed under the rate limit."""
        with self._lock:
            now = time.time()
            window_start = now - 60.0

            # Clean old entries
            self._requests[user_id] = [
                t for t in self._requests[user_id] if t > window_start
            ]

            current_count = len(self._requests[user_id])
            if current_count >= self._rpm:
                return ModuleResult.failure(
                    ModuleID.M05,
                    "RATE_LIMITED",
                    f"Rate limit exceeded: {current_count}/{self._rpm} requests per minute",
                    retry_after_s=round(self._requests[user_id][0] - window_start + 1, 1),
                )

            # Record this request
            self._requests[user_id].append(now)
            return ModuleResult.success(
                ModuleID.M05,
                {
                    "allowed": True,
                    "requests_in_window": current_count + 1,
                    "limit": self._rpm,
                    "remaining": self._rpm - current_count - 1,
                },
            )

    def get_status(self, user_id: str = "global") -> dict[str, Any]:
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            recent = [t for t in self._requests.get(user_id, []) if t > window_start]
        return {
            "user_id": user_id,
            "requests_in_window": len(recent),
            "limit": self._rpm,
            "remaining": max(0, self._rpm - len(recent)),
        }


class ConcurrencyManager:
    """Manages concurrent job execution.

    Bugs fixed here:
    - ``self._semaphore`` was created but never acquired or released
      anywhere — it was dead code that gave the false impression
      concurrency was being enforced by it, when in fact only the manual
      ``running`` count did that.
    - ``complete_job`` never promoted a queued job to running when a slot
      freed up, so once the concurrency limit was hit once, every
      subsequent job stayed in ``queued`` state forever (there was no
      code path that ever moved a job out of QUEUED).
    """

    def __init__(self, max_concurrent: int = 4) -> None:
        self._max_concurrent = max_concurrent
        self._jobs: dict[str, JobInfo] = {}
        self._queue: list[str] = []
        self._lock = threading.Lock()

    def _running_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.state == JobState.RUNNING)

    def submit_job(self, user_id: str = "anonymous") -> ModuleResult:
        """Try to submit a new job."""
        job_id = uuid.uuid4().hex[:12]
        job = JobInfo(job_id, user_id)

        with self._lock:
            running = self._running_count()
            if running >= self._max_concurrent:
                job.state = JobState.QUEUED
                self._jobs[job_id] = job
                self._queue.append(job_id)
                return ModuleResult.success(
                    ModuleID.M05,
                    {
                        "job_id": job_id,
                        "state": "queued",
                        "position": len(self._queue),
                        "running_jobs": running,
                        "max_concurrent": self._max_concurrent,
                    },
                )

            job.state = JobState.RUNNING
            job.started_at = time.time()
            self._jobs[job_id] = job
            running_after = running + 1

        return ModuleResult.success(
            ModuleID.M05,
            {
                "job_id": job_id,
                "state": "running",
                "running_jobs": running_after,
                "max_concurrent": self._max_concurrent,
            },
        )

    def complete_job(self, job_id: str, error: str | None = None) -> None:
        """Mark a job finished and promote the next queued job, if any."""
        promoted: str | None = None
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.completed_at = time.time()
                job.state = JobState.FAILED if error else JobState.COMPLETE
                job.error = error

            # Promote the next queued job now that a slot may be free.
            if self._running_count() < self._max_concurrent and self._queue:
                promoted = self._queue.pop(0)
                next_job = self._jobs.get(promoted)
                if next_job is not None and next_job.state == JobState.QUEUED:
                    next_job.state = JobState.RUNNING
                    next_job.started_at = time.time()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            states = defaultdict(int)
            for job in self._jobs.values():
                states[job.state.value] += 1
        return {
            "max_concurrent": self._max_concurrent,
            "total_jobs": len(self._jobs),
            "by_state": dict(states),
            "running": states.get("running", 0),
            "queued": states.get("queued", 0),
        }

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]
