"""
Learned multimodal evidence-fusion model — logistic regression over
engineered bi-temporal evidence channels (optical difference, SAR
backscatter difference, temporal gap).

A genuinely *learned* combiner: weights come from gradient-descent
``fit()`` on labeled examples, not a hand-picked threshold, closing the
"not only CLIP embedding distance/pixel difference" gap by making the
decision itself a trained function of multiple evidence sources instead
of a fixed threshold on one signal.

Ships **unfitted** — the same honesty pattern as
``confidence.conformal.ConformalCalibrator``. An unfitted instance uses an
informative prior (equal weight on optical/SAR evidence, half-weight on
the temporal-gap channel) rather than random weights: this is a 3-feature
linear model, not a deep network, so a sane hand-set starting point is a
reasonable estimator on its own, unlike the untrained-Siamese-transformer
case ``change_detection`` refuses to ship (see that module's docstring).
``fit()``/``save()``/``load()`` mirror ``ConformalCalibrator``'s API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Order matters — must match the column order every caller builds X in.
FEATURE_NAMES = ("optical_evidence", "sar_evidence", "temporal_evidence")


class EvidenceFusionModel:
    """Logistic-regression fusion over multimodal bi-temporal evidence channels."""

    def __init__(self) -> None:
        self.weights = np.array([2.0, 2.0, 1.0])  # informative prior, not random
        self.bias = -2.0
        self._fitted = False

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """``X``: (..., 3) evidence channels in [0, 1]; unavailable channels
        should be passed as 0.5 (neutral — neither supports nor refutes change)."""
        z = np.asarray(X, dtype=np.float64) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-z))

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        lr: float = 0.5,
        epochs: int = 1000,
        l2: float = 1e-3,
    ) -> None:
        """Batch gradient descent on labeled (evidence, changed) examples."""
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if Xa.ndim != 2 or Xa.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"X must be (n, {len(FEATURE_NAMES)}), got {Xa.shape}")
        if len(Xa) == 0:
            raise ValueError("fit() requires a non-empty training set")

        w = np.zeros(Xa.shape[1])
        b = 0.0
        n = len(ya)
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-(Xa @ w + b)))
            w -= lr * (Xa.T @ (p - ya) / n + l2 * w)
            b -= lr * float(np.mean(p - ya))

        self.weights, self.bias, self._fitted = w, float(b), True

    def is_fitted(self) -> bool:
        return self._fitted

    def save(self, path: Path | str) -> dict[str, Any]:
        """Write weights so a claimed fit can be inspected, not just trusted."""
        if not self._fitted:
            raise RuntimeError("refusing to save an unfitted fusion model")
        artifact = {
            "features": list(FEATURE_NAMES),
            "weights": self.weights.tolist(),
            "bias": self.bias,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return artifact

    def load(self, path: Path | str) -> None:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        self.weights = np.asarray(artifact["weights"], dtype=np.float64)
        self.bias = float(artifact["bias"])
        self._fitted = True
