"""
Conformal prediction calibrator.

Uses MAPIE to wrap specialist outputs in statistically valid prediction
intervals.  Unlike raw softmax confidence, conformal prediction guarantees
that the true answer falls within the interval at a user-specified
coverage rate (e.g., 90%).

When MAPIE is not installed, falls back to a simple quantile-based
calibration using only NumPy.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


def _is_mapie_available() -> bool:
    try:
        import mapie  # noqa: F401
        return True
    except ImportError:
        return False


def _mapie_nonconformity_scores(
    scores: NDArray[np.floating],
    labels: NDArray[np.integer],
) -> NDArray[np.floating]:
    """Compute LAC nonconformity scores via MAPIE's split-conformal classifier.

    Wraps the raw scores in a passthrough classifier (``P(correct) = score``)
    so MAPIE's ``SplitConformalClassifier`` can conformalize them; its "lac"
    conformity score reduces to the same ``1 - score`` / ``score`` formula
    used by the NumPy fallback, so results agree when both are available.
    """
    from mapie.classification import SplitConformalClassifier
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _IdentityScoreClassifier(ClassifierMixin, BaseEstimator):
        classes_ = np.array([0, 1])

        def fit(self, X: Any, y: Any = None) -> "_IdentityScoreClassifier":
            return self

        def predict_proba(self, X: Any) -> NDArray[np.floating]:
            p1 = np.clip(np.asarray(X, dtype=np.float64).ravel(), 0.0, 1.0)
            return np.column_stack([1.0 - p1, p1])

        def predict(self, X: Any) -> NDArray[np.integer]:
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        def __sklearn_is_fitted__(self) -> bool:
            return True

    clf = SplitConformalClassifier(
        estimator=_IdentityScoreClassifier(), conformity_score="lac", prefit=True
    )
    clf.conformalize(scores.reshape(-1, 1), labels)
    return np.asarray(clf.conformity_scores, dtype=np.float64).ravel()


class ConformalCalibrator:
    """Calibrates raw specialist scores into conformal prediction intervals.

    Workflow:
        1. ``fit(calibration_scores, calibration_labels)`` on a held-out set.
        2. ``predict(raw_score)`` at inference to get ``(calibrated, lower, upper)``.

    The calibrator stores non-conformity scores from the calibration set and
    uses them to construct intervals at a target coverage level.
    """

    def __init__(self, target_coverage: float = 0.90) -> None:
        """
        Args:
            target_coverage: Desired marginal coverage (e.g. 0.90 = 90%).
        """
        if not 0.0 < target_coverage < 1.0:
            raise ValueError(f"target_coverage must be in (0, 1), got {target_coverage}")

        self.target_coverage = target_coverage
        self._calibration_scores: NDArray[np.floating] | None = None
        self._nonconformity_scores: NDArray[np.floating] | None = None
        self._quantile: float | None = None
        self._fitted = False
        self._backend = "numpy"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        calibration_scores: NDArray[np.floating],
        calibration_labels: NDArray[np.integer],
    ) -> None:
        """Fit the calibrator on a held-out calibration set.

        Args:
            calibration_scores: Raw model confidence scores, shape (N,).
            calibration_labels: Binary correctness labels (1 = correct), shape (N,).
        """
        scores = np.asarray(calibration_scores, dtype=np.float64)
        labels = np.asarray(calibration_labels, dtype=np.int64)

        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} vs labels {labels.shape}"
            )

        n = len(scores)
        if n == 0:
            raise ValueError(
                "ConformalCalibrator.fit() requires a non-empty calibration "
                "set (got 0 samples)."
            )

        # Non-conformity score: 1 - raw_score for correct predictions,
        # raw_score for incorrect (higher = more non-conforming). Prefer the
        # MAPIE-backed computation when the library is installed; fall back
        # to the manual formula otherwise (or if MAPIE errors at runtime).
        self._backend = "numpy"
        if _is_mapie_available():
            try:
                self._nonconformity_scores = _mapie_nonconformity_scores(scores, labels)
                self._backend = "mapie"
            except Exception as exc:
                logger.warning("mapie_calibration_failed_falling_back_to_numpy", error=str(exc))

        if self._backend == "numpy":
            self._nonconformity_scores = np.where(
                labels == 1,
                1.0 - scores,
                scores,
            )

        # Compute quantile for the target coverage
        quantile_level = np.ceil((n + 1) * self.target_coverage) / n
        quantile_level = min(quantile_level, 1.0)
        self._quantile = float(
            np.quantile(self._nonconformity_scores, quantile_level)
        )

        self._calibration_scores = scores
        self._fitted = True

        logger.info(
            "conformal_calibrator_fitted",
            n_calibration=n,
            target_coverage=self.target_coverage,
            quantile=round(self._quantile, 4),
            backend=self._backend,
        )

    def predict(
        self,
        raw_score: float,
    ) -> dict[str, Any]:
        """Produce a calibrated prediction interval for a raw score.

        Args:
            raw_score: Raw specialist confidence in [0, 1].

        Returns:
            Dict with keys: ``calibrated_score``, ``lower``, ``upper``,
            ``method``, ``target_coverage``.
        """
        if not self._fitted:
            logger.warning("conformal_calibrator_not_fitted_returning_raw")
            return {
                "calibrated_score": raw_score,
                "lower": 0.0,
                "upper": 1.0,
                "method": "uncalibrated",
                "target_coverage": self.target_coverage,
            }

        assert self._quantile is not None

        # Interval: [score - quantile, score + quantile], clipped to [0, 1]
        lower = max(0.0, raw_score - self._quantile)
        upper = min(1.0, raw_score + self._quantile)

        # Calibrated score = midpoint of interval
        calibrated = (lower + upper) / 2.0

        return {
            "calibrated_score": round(calibrated, 4),
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "method": "conformal",
            "target_coverage": self.target_coverage,
        }

    def save(self, path: Path | str) -> dict[str, Any]:
        """Write the calibration artifact so a claimed calibration is checkable.

        A calibrated confidence is only meaningful if the calibration set that
        produced it can be inspected, so the quantile and the nonconformity
        scores are stored, not just the fact that fitting happened.
        """
        if not self._fitted:
            raise RuntimeError("refusing to save an unfitted calibrator")
        assert self._nonconformity_scores is not None
        artifact = {
            "method": "split_conformal",
            "backend": self._backend,
            "target_coverage": self.target_coverage,
            "quantile": self._quantile,
            "n_calibration": len(self._nonconformity_scores),
            "nonconformity_scores": [float(v) for v in self._nonconformity_scores],
            "saved_at": datetime.now(UTC).isoformat(),
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        logger.info("conformal_artifact_saved", path=str(p))
        return artifact

    def load(self, path: Path | str) -> None:
        """Restore a calibrator from a saved artifact."""
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        self.target_coverage = float(artifact["target_coverage"])
        self._quantile = float(artifact["quantile"])
        self._nonconformity_scores = np.asarray(artifact["nonconformity_scores"], dtype=float)
        self._backend = artifact.get("backend", "numpy")
        self._fitted = True
        logger.info("conformal_artifact_loaded", path=str(path))

    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    def stats(self) -> dict[str, Any]:
        """Return calibrator statistics."""
        if not self._fitted:
            return {"fitted": False}
        assert self._nonconformity_scores is not None
        return {
            "fitted": True,
            "n_calibration": len(self._nonconformity_scores),
            "target_coverage": self.target_coverage,
            "quantile": self._quantile,
            "mean_nonconformity": float(np.mean(self._nonconformity_scores)),
            "backend": self._backend,
        }
