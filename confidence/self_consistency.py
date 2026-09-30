"""
Self-consistency voter — multi-path sampling + majority vote.

Instead of trusting a single forward pass, the voter runs the specialist
``n_paths`` times (with temperature sampling / prompt perturbation) and
takes the majority vote.  The agreement ratio is a meaningful confidence
signal: if 9/10 paths agree, the system is much more trustworthy than
if only 5/10 agree.

This is complementary to conformal prediction: conformal calibrates a
single raw score, self-consistency generates a population of answers.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class SelfConsistencyVoter:
    """Multi-path sampling and majority-vote aggregation.

    Works with any specialist that returns a dict containing an
    ``"answer"`` key (or a configurable key).

    For deterministic / non-LLM specialists, the voter still runs multiple
    passes — the agreement ratio is simply 1.0 (deterministic tool).
    """

    def __init__(
        self,
        n_paths: int = 5,
        answer_key: str = "answer",
        temperature_range: tuple[float, float] = (0.5, 1.0),
    ) -> None:
        """
        Args:
            n_paths:           Number of independent reasoning paths.
            answer_key:        Key in the specialist output dict that
                               contains the answer to vote on.
            temperature_range: (min, max) temperature for sampling diversity.
                               Specialist wrappers can use this to vary
                               inference parameters.
        """
        self.n_paths = max(n_paths, 1)
        self.answer_key = answer_key
        self.temperature_range = temperature_range

    # ------------------------------------------------------------------
    # Synchronous API (for non-async specialists)
    # ------------------------------------------------------------------

    def vote(
        self,
        specialist_fn: Callable[..., dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the specialist ``n_paths`` times and return majority vote.

        Args:
            specialist_fn: Callable that returns a dict with the answer.
            **kwargs:      Arguments forwarded to the specialist.

        Returns:
            Dict with: ``best_answer``, ``agreement_ratio``,
            ``n_consistent``, ``total_paths``, ``all_answers``.
        """
        answers: list[str] = []
        all_results: list[dict[str, Any]] = []

        for i in range(self.n_paths):
            try:
                result = specialist_fn(**kwargs)
                all_results.append(result)
                ans = str(result.get(self.answer_key, ""))
                answers.append(ans)
            except Exception as exc:
                logger.warning("vote_path_failed", path=i, error=str(exc))

        return self._aggregate(answers, all_results)

    # ------------------------------------------------------------------
    # Async API (for async specialists)
    # ------------------------------------------------------------------

    async def vote_async(
        self,
        specialist_fn: Callable[..., Awaitable[dict[str, Any]]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async variant — runs paths concurrently.

        Args:
            specialist_fn: Async callable returning a dict.
            **kwargs:      Arguments forwarded to the specialist.

        Returns:
            Same structure as :meth:`vote`.
        """
        tasks = [
            self._safe_call_async(specialist_fn, i, **kwargs)
            for i in range(self.n_paths)
        ]
        results = await asyncio.gather(*tasks)

        answers: list[str] = []
        all_results: list[dict[str, Any]] = []
        for result in results:
            if result is not None:
                all_results.append(result)
                answers.append(str(result.get(self.answer_key, "")))

        return self._aggregate(answers, all_results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _safe_call_async(
        self,
        fn: Callable[..., Awaitable[dict[str, Any]]],
        path_idx: int,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Call the async specialist, catching errors."""
        try:
            return await fn(**kwargs)
        except Exception as exc:
            logger.warning("vote_path_failed", path=path_idx, error=str(exc))
            return None

    def _aggregate(
        self,
        answers: list[str],
        all_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate answers via majority vote."""
        if not answers:
            logger.error("self_consistency_no_valid_paths")
            return {
                "best_answer": "",
                "agreement_ratio": 0.0,
                "n_consistent": 0,
                "total_paths": self.n_paths,
                "all_answers": [],
                "best_result": {},
            }

        counter = Counter(answers)
        best_answer, n_consistent = counter.most_common(1)[0]
        agreement_ratio = n_consistent / len(answers)
        # The first path whose answer matches the winning vote — callers need
        # the specialist's full result dict (paths, masks, etc.), not just
        # the answer string used to vote.
        best_result = next(
            (r for ans, r in zip(answers, all_results) if ans == best_answer), {}
        )

        logger.info(
            "self_consistency_vote",
            best_answer=best_answer[:80],
            agreement=f"{n_consistent}/{len(answers)}",
            ratio=round(agreement_ratio, 3),
        )

        return {
            "best_answer": best_answer,
            "agreement_ratio": round(agreement_ratio, 4),
            "n_consistent": n_consistent,
            "total_paths": len(answers),
            "all_answers": answers,
            "best_result": best_result,
        }
