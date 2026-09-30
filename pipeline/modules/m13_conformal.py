"""
M13 — Conformal Calibration Wiring.

``confidence.conformal.ConformalCalibrator`` already has ``fit()``/``save()``/
``load()`` (see confidence/conformal.py), but nothing in the running system
ever calls ``load()`` — the planner always builds a fresh, unfitted
calibrator (models/registry.json documents this: "Ships UNFITTED... rather
than inventing coverage", which is the right call absent real labelled
data). M13 does not fabricate a calibration set to force that flag on; it
just makes a *real* one usable once it exists: if a calibration artifact was
fitted offline and saved to disk, load it at planner start-up so predictions
stop being uncalibrated by construction rather than by lack of evidence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.schemas.contracts import ModuleID, ModuleResult

DEFAULT_CALIBRATION_PATH = Path(
    os.environ.get("AETHERIS_CALIBRATION_PATH", "./calibration/conformal.json")
)


def attach_saved_calibration(
    planner: Any, path: Path | str = DEFAULT_CALIBRATION_PATH
) -> ModuleResult:
    """Load a previously-fitted calibrator artifact into ``planner._calibrator``, if any."""
    calibrator = getattr(planner, "_calibrator", None)
    if calibrator is None:
        return ModuleResult.skipped(ModuleID.M13, "Planner has no _calibrator attribute")
    if calibrator.is_fitted():
        return ModuleResult.success(ModuleID.M13, {"loaded": False, "already_fitted": True})
    if not Path(path).exists():
        return ModuleResult.success(
            ModuleID.M13, {"loaded": False, "note": f"no calibration artifact at {path}"}
        )
    try:
        calibrator.load(path)
        return ModuleResult.success(
            ModuleID.M13, {"loaded": True, "path": str(path), **calibrator.stats()}
        )
    except Exception as exc:
        return ModuleResult.failure(ModuleID.M13, "CALIBRATION_LOAD_FAILED", str(exc))


def fit_and_save(
    calibrator: Any,
    calibration_scores: list[float],
    calibration_labels: list[int],
    path: Path | str = DEFAULT_CALIBRATION_PATH,
) -> ModuleResult:
    """Fit on a real, held-out (score, correctness) set and persist it.

    For whoever runs an offline evaluation and has real labels — this
    module never invents scores or labels on its own.
    """
    try:
        calibrator.fit(np.asarray(calibration_scores), np.asarray(calibration_labels))
        artifact = calibrator.save(path)
        return ModuleResult.success(ModuleID.M13, {"fitted": True, "path": str(path), **artifact})
    except Exception as exc:
        return ModuleResult.failure(ModuleID.M13, "CALIBRATION_FIT_FAILED", str(exc))
