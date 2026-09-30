"""Tests for the offline synthetic fusion-model trainer.

Unlike ``fit_fusion_model.py`` (needs huggingface.co), everything here is
local and deterministic, so it runs in CI/sandboxes with no network.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from training.fit_fusion_model_synthetic import _synthetic_pair, main


class TestSyntheticPair:
    def test_shapes_and_mask_dtype(self) -> None:
        rng = np.random.default_rng(0)
        t1_rgb, t2_rgb, t1_sar, t2_sar, mask = _synthetic_pair(rng, elapsed_days=365, size=64)

        assert t1_rgb.size == t2_rgb.size == (64, 64)
        assert t1_sar.size == t2_sar.size == (64, 64)
        assert mask.shape == (64, 64)
        assert mask.dtype == bool

    def test_more_elapsed_days_tends_to_more_change(self) -> None:
        """Not deterministic (Poisson-sampled patch count), but the mean
        changed fraction over many draws should be higher for a longer gap
        -- otherwise the temporal channel would have nothing real to fit."""
        short_frac = np.mean([
            _synthetic_pair(np.random.default_rng(i), elapsed_days=20, size=64)[4].mean()
            for i in range(30)
        ])
        long_frac = np.mean([
            _synthetic_pair(np.random.default_rng(i), elapsed_days=700, size=64)[4].mean()
            for i in range(30)
        ])
        assert long_frac > short_frac


class TestFitFusionModelSynthetic:
    def test_end_to_end_produces_a_real_fit(self, monkeypatch) -> None:
        """Tiny, fast, fully-offline run: the fitted checkpoint should move
        away from the informative-prior starting weights and load back into
        a usable, fitted EvidenceFusionModel."""
        out = Path(tempfile.gettempdir()) / "aetheris_test_synthetic_fusion_ckpt.json"
        argv = [
            "fit_fusion_model_synthetic.py",
            "--pairs", "4",
            "--pixels-per-pair", "200",
            "--epochs", "200",
            "--output", str(out),
            "--seed", "0",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        main()

        from specialists._fusion_model import EvidenceFusionModel

        model = EvidenceFusionModel()
        model.load(out)
        assert model.is_fitted()
        assert model.weights.shape == (3,)
        assert not np.allclose(model.weights, [2.0, 2.0, 1.0])  # moved from the prior
