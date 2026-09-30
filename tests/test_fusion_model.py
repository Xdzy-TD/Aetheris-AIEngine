"""
Tests for EvidenceFusionModel edge cases.

fit() is never called on the runtime path today (the fusion model ships
unfitted per STATUS.md), but it's public API and should fail loudly rather
than divide-by-zero if ever called with an empty training set, and save()
must refuse to persist an unfitted model as if it were a trained result.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from specialists._fusion_model import EvidenceFusionModel


def test_unfitted_model_uses_informative_prior_not_random() -> None:
    model = EvidenceFusionModel()

    assert not model.is_fitted()
    # All-neutral evidence must be near indecisive (not confidently "changed"
    # or "unchanged") -- a sane prior, not an arbitrary one.
    p = model.predict_proba(np.array([[0.5, 0.5, 0.5]]))[0]
    assert 0.3 < p < 0.7
    # Strong evidence on every channel should still push the prior toward
    # "changed" -- weights are positive, not zero or negative.
    p_strong = model.predict_proba(np.array([[1.0, 1.0, 1.0]]))[0]
    assert p_strong > p


def test_fit_empty_training_set_raises_value_error() -> None:
    model = EvidenceFusionModel()

    with pytest.raises(ValueError, match="non-empty training set"):
        model.fit(
            X=np.empty((0, 3), dtype=np.float64),
            y=np.array([], dtype=np.float64),
        )

    assert not model.is_fitted()


def test_fit_wrong_feature_count_raises_value_error() -> None:
    model = EvidenceFusionModel()

    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        model.fit(X=np.zeros((4, 2)), y=np.array([0, 1, 0, 1]))


def test_fit_recovers_a_clearly_separable_relationship() -> None:
    """Gradient descent must actually learn something, not just run:
    optical-evidence-heavy examples labeled 'changed' vs. all-zero examples
    labeled 'unchanged' should yield a fitted model that separates new,
    unseen examples the same way."""
    rng = np.random.default_rng(0)
    n = 200
    X_pos = rng.uniform(0.7, 1.0, size=(n, 3))
    X_neg = rng.uniform(0.0, 0.3, size=(n, 3))
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n), np.zeros(n)])

    model = EvidenceFusionModel()
    model.fit(X, y, epochs=500)

    assert model.is_fitted()
    assert model.predict_proba(np.array([[0.9, 0.9, 0.9]]))[0] > 0.8
    assert model.predict_proba(np.array([[0.1, 0.1, 0.1]]))[0] < 0.2


def test_save_refuses_to_persist_an_unfitted_model() -> None:
    """An unfitted (prior-only) model saved to disk would look, on load,
    exactly like a real trained checkpoint -- save() must refuse."""
    model = EvidenceFusionModel()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fusion.json"
        with pytest.raises(RuntimeError, match="unfitted"):
            model.save(path)
        assert not path.exists()


def test_save_then_load_round_trips_fitted_weights() -> None:
    model = EvidenceFusionModel()
    model.fit(
        X=np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]]),
        y=np.array([1.0, 0.0]),
        epochs=200,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fusion.json"
        artifact = model.save(path)
        assert path.exists()
        assert artifact["features"] == ["optical_evidence", "sar_evidence", "temporal_evidence"]

        reloaded = EvidenceFusionModel()
        reloaded.load(path)

    assert reloaded.is_fitted()
    np.testing.assert_allclose(reloaded.weights, model.weights)
    assert reloaded.bias == model.bias
    np.testing.assert_allclose(
        reloaded.predict_proba(np.array([[0.5, 0.5, 0.5]])),
        model.predict_proba(np.array([[0.5, 0.5, 0.5]])),
    )
