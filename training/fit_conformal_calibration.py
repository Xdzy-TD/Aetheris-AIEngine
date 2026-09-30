"""Fit confidence.conformal.ConformalCalibrator on a real, held-out calibration set.

There is no labelled real satellite-imagery dataset in this repo (see
docs/STATUS.md), so this does not touch remote-sensing imagery and makes no
claim about real-world coverage. What it does do honestly: procedurally
generate synthetic images with a KNOWN planted land-cover majority (the
label is fixed by the pixel-generating distribution, not by re-running the
classifier under test), run the real vqa_caption specialist
(specialists/vqa_caption/model.py) on each one exactly as the app does, and
record its actual (raw_confidence, correct) pairs. Purity is swept from
near-ambiguous to unambiguous so the set spans easy and hard cases instead
of only trivial ones. Those real pairs — not invented numbers — are what
gets fit and saved.

This calibrates the classifier's confidence-vs-correctness relationship on
*this* synthetic distribution only; a real remote-sensing calibration set
would still be needed before trusting coverage on actual satellite imagery
(see the caveat this script prints and writes into the artifact directory).

    python training/fit_conformal_calibration.py
    export AETHERIS_CALIBRATION_PATH=calibration/conformal.json  # already the default
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from confidence.conformal import ConformalCalibrator
from pipeline.modules.m13_conformal import fit_and_save
from specialists.vqa_caption.model import VQACaptionSpecialist

_CLASSES = ["water", "vegetation", "urban/built-up", "bare/other"]


def _canon_pixels(cls: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """n pixels drawn from a class's canonical, human-recognizable color range."""
    if cls == "water":
        lo, hi = (0.03, 0.10, 0.45), (0.20, 0.30, 0.80)
    elif cls == "vegetation":
        lo, hi = (0.03, 0.35, 0.03), (0.20, 0.65, 0.20)
    elif cls == "urban/built-up":
        v = rng.uniform(0.55, 0.85, n)
        noise = rng.uniform(-0.03, 0.03, (n, 3))
        return np.clip(np.stack([v, v, v], axis=1) + noise, 0, 1)
    else:  # bare/other: brownish soil, deliberately outside the other 3 thresholds
        lo, hi = (0.35, 0.20, 0.15), (0.55, 0.35, 0.30)
    return rng.uniform(lo, hi, (n, 3))


def make_synthetic_image(true_class: str, purity: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """(size, size, 3) uint8 image: `purity` fraction canonical pixels, rest uniform noise."""
    n = size * size
    is_canon = rng.random(n) < purity
    px = rng.uniform(0, 1, (n, 3))
    px[is_canon] = _canon_pixels(true_class, int(is_canon.sum()), rng)
    return (px.reshape(size, size, 3) * 255).round().clip(0, 255).astype(np.uint8)


def build_calibration_set(
    n_per_cell: int, size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    specialist = VQACaptionSpecialist(demo_mode=True)
    scores: list[float] = []
    labels: list[int] = []
    with tempfile.TemporaryDirectory() as tmp:
        for true_class in _CLASSES:
            for purity in np.linspace(0.35, 1.0, 14):
                for _ in range(n_per_cell):
                    img = make_synthetic_image(true_class, float(purity), size, rng)
                    path = str(Path(tmp) / "s.png")
                    Image.fromarray(img, mode="RGB").save(path)
                    result = specialist.run(
                        image_path=path,
                        question="Describe this image.",  # no keyword -> generic branch
                        modality="optical",
                    )
                    answer = result.get("answer", "")
                    predicted = answer.rsplit("dominant class: ", 1)[-1].rstrip(".")
                    scores.append(float(result["raw_confidence"]))
                    labels.append(int(predicted == true_class))
    return np.asarray(scores), np.asarray(labels, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-per-cell", type=int, default=6, help="replicates per (class, purity) cell")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--target-coverage", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--output", default="calibration/conformal.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scores, labels = build_calibration_set(args.n_per_cell, args.image_size, args.seed)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(scores))
    n_hold = int(len(idx) * args.holdout_frac)
    hold_idx, fit_idx = idx[:n_hold], idx[n_hold:]

    calibrator = ConformalCalibrator(target_coverage=args.target_coverage)
    result = fit_and_save(calibrator, scores[fit_idx].tolist(), labels[fit_idx].tolist(), args.output)
    if not result.data.get("fitted"):
        raise SystemExit(f"Calibration fit failed: {result.error}")

    # Honest empirical check on synthetic held-out data (same distribution as
    # the fit set, NOT real imagery) — reports whether the fitted quantile
    # actually behaves as intended before anyone trusts the artifact.
    nonconf = np.where(labels[hold_idx] == 1, 1.0 - scores[hold_idx], scores[hold_idx])
    empirical_coverage = float(np.mean(nonconf <= calibrator.stats()["quantile"]))

    print(f"Fit on {len(fit_idx)} samples, held out {len(hold_idx)}.")
    print(f"Raw accuracy of vqa_caption on this synthetic set: {labels.mean():.3f}")
    print(f"Target coverage: {args.target_coverage}  Empirical (synthetic holdout): {empirical_coverage:.3f}")
    print(f"Saved calibration artifact to {args.output}")
    print("Caveat: fitted on procedurally generated synthetic imagery, not real")
    print("satellite data — see docs/STATUS.md before quoting this as a real-world guarantee.")


if __name__ == "__main__":
    main()
