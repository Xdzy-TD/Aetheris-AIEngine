"""
Tests for ConformalCalibrator edge cases.

fit() is never called on the runtime path today (calibrator ships unfitted
per STATUS.md), but it's public API and should fail loudly rather than
divide-by-zero if ever called with an empty calibration set.
"""

from __future__ import annotations

import numpy as np
import pytest

from confidence.conformal import ConformalCalibrator


def test_fit_empty_calibration_set_raises_value_error() -> None:
    calibrator = ConformalCalibrator()

    with pytest.raises(ValueError, match="non-empty calibration set"):
        calibrator.fit(
            calibration_scores=np.array([], dtype=np.float64),
            calibration_labels=np.array([], dtype=np.int64),
        )

    assert not calibrator.is_fitted()


def test_fit_nonempty_calibration_set_still_works() -> None:
    calibrator = ConformalCalibrator(target_coverage=0.9)

    calibrator.fit(
        calibration_scores=np.array([0.6, 0.7, 0.8, 0.9, 0.95]),
        calibration_labels=np.array([1, 1, 0, 1, 1]),
    )

    assert calibrator.is_fitted()


def test_fit_uses_mapie_backend_when_installed_and_agrees_with_numpy() -> None:
    pytest.importorskip("mapie")

    scores = np.array([0.6, 0.7, 0.8, 0.9, 0.95])
    labels = np.array([1, 1, 0, 1, 1])

    calibrator = ConformalCalibrator(target_coverage=0.9)
    calibrator.fit(calibration_scores=scores, calibration_labels=labels)

    # fit() must actually take the MAPIE path when MAPIE is installed, not
    # silently use the manual formula regardless (the bug this guards against).
    assert calibrator.stats()["backend"] == "mapie"

    # MAPIE's "lac" conformity score reduces to the same formula the NumPy
    # fallback uses, so the two backends should agree numerically.
    expected = np.where(labels == 1, 1.0 - scores, scores)
    assert np.allclose(sorted(calibrator._nonconformity_scores), sorted(expected))
