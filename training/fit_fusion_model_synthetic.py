"""Fit :class:`specialists._fusion_model.EvidenceFusionModel` on locally
generated synthetic bi-temporal data — the offline counterpart to
``fit_fusion_model.py`` for sandboxes where ``huggingface.co`` is
unreachable (see that module's docstring, and ``docs/STATUS.md``).

This is NOT a substitute for training on real change-detection data: the
"changed" regions are random rectangles with resampled pixels, not real
land-cover transitions, so the fitted weights encode "logistic regression
over these three evidence channels can, in principle, separate them" —
useful to exercise ``AETHERIS_FUSION_MODEL_PATH`` end-to-end and ship a
better-than-informative-prior starting point offline, not a substitute for
fitting on OSCD once network access is available.

Reuses ``FusionChangeDetectionSpecialist._bitemporal_evidence`` for the
optical and SAR evidence channels, exactly like the real trainer, so the
fit stays on the same feature pipeline inference actually serves. Unlike
OSCD, synthetic data lets every pair carry a SAR channel and an elapsed-day
gap too, so — unlike the real trainer run without --sar*/--date* columns —
this exercises and fits all three evidence channels, not just optical.

    python training/fit_fusion_model_synthetic.py \
        --output models/fusion/evidence_fusion_synthetic.json
    export AETHERIS_FUSION_MODEL_PATH=models/fusion/evidence_fusion_synthetic.json
"""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from specialists._fusion_model import EvidenceFusionModel
from specialists.fusion_change_detection.model import FusionChangeDetectionSpecialist

_SIZE = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pairs", type=int, default=60, help="Number of synthetic bi-temporal pairs")
    p.add_argument(
        "--pixels-per-pair", type=int, default=2000,
        help="Random valid-pixel subsample per pair (keeps the fit fast/bounded)",
    )
    p.add_argument("--output", default=None,
                   help="Output path (default: <output-dir>/evidence_fusion_synthetic.json)")
    p.add_argument("--output-dir", default="models/checkpoints/fusion",
                   help="Directory for checkpoint output (used when --output is not set)")
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", action="store_true",
                   help="After fitting, load the checkpoint and verify a single inference")
    return p.parse_args()


def _synthetic_pair(
    rng: np.random.Generator, elapsed_days: float, size: int = _SIZE
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, NDArray[np.bool_]]:
    """Random RGB + SAR-like grayscale base, each with a few rectangular
    "changed" patches (resampled pixels) plus per-pixel sensor noise
    everywhere so unchanged pixels still carry real (small) diff values,
    the way a noisy real optical/SAR pair would. More elapsed days -> more
    (Poisson-distributed, not guaranteed) patches, so the temporal channel
    gets a real but noisy signal to fit — a longer gap makes change more
    plausible, not certain.
    """
    base_rgb = rng.integers(0, 200, (size, size, 3), dtype=np.uint8)
    base_sar = rng.integers(30, 220, (size, size), dtype=np.uint8)
    t1_rgb, t2_rgb = base_rgb.copy(), np.clip(
        base_rgb.astype(np.int16) + rng.normal(0, 6, base_rgb.shape), 0, 255
    ).astype(np.uint8)
    t1_sar, t2_sar = base_sar.copy(), np.clip(
        base_sar.astype(np.int16) + rng.normal(0, 8, base_sar.shape), 0, 255
    ).astype(np.uint8)

    mask = np.zeros((size, size), dtype=bool)
    n_changes = int(rng.poisson(1 + 3 * min(elapsed_days / 730.0, 1.0)))
    for _ in range(n_changes):
        rh, rw = int(rng.integers(20, size // 3)), int(rng.integers(20, size // 3))
        y0, x0 = int(rng.integers(0, size - rh)), int(rng.integers(0, size - rw))
        t2_rgb[y0:y0 + rh, x0:x0 + rw] = rng.integers(0, 255, (rh, rw, 3), dtype=np.uint8)
        t2_sar[y0:y0 + rh, x0:x0 + rw] = rng.integers(0, 255, (rh, rw), dtype=np.uint8)
        mask[y0:y0 + rh, x0:x0 + rw] = True

    return (
        Image.fromarray(t1_rgb), Image.fromarray(t2_rgb),
        Image.fromarray(t1_sar, mode="L"), Image.fromarray(t2_sar, mode="L"),
        mask,
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        from pathlib import Path
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "evidence_fusion_synthetic.json")

    X_rows: list[list[float]] = []
    y_rows: list[float] = []

    for _ in range(args.pairs):
        elapsed_days = float(rng.uniform(10, 730))
        t1_rgb, t2_rgb, t1_sar, t2_sar, mask = _synthetic_pair(rng, elapsed_days)

        optical, _, valid, _ = FusionChangeDetectionSpecialist._bitemporal_evidence(t1_rgb, t2_rgb)
        sar, _, _, _ = FusionChangeDetectionSpecialist._bitemporal_evidence(
            t1_sar, t2_sar, despeckle=True
        )
        temporal = min(elapsed_days / 730.0, 1.0)

        valid_idx = np.argwhere(valid)
        if len(valid_idx) == 0:
            continue
        n = min(args.pixels_per_pair, len(valid_idx))
        for yy, xx in valid_idx[rng.choice(len(valid_idx), size=n, replace=False)]:
            X_rows.append([optical[yy, xx], sar[yy, xx], temporal])
            y_rows.append(float(mask[yy, xx]))

    if not X_rows:
        raise SystemExit("No valid training pixels generated — try increasing --pairs.")

    X, y = np.array(X_rows), np.array(y_rows)
    print(f"Fitting on {len(y)} synthetic pixels from {args.pairs} pairs ({y.mean():.1%} positive).")

    model = EvidenceFusionModel()
    model.fit(X, y, lr=args.lr, epochs=args.epochs, l2=args.l2)

    train_acc = float(np.mean((model.predict_proba(X) > 0.5) == y.astype(bool)))
    print(f"Train-set accuracy on synthetic data: {train_acc:.1%} (sanity check on synthetic "
          f"rectangles, not a claim about real-world change-detection accuracy).")

    model.save(output_path)
    print(
        f"Synthetic fusion checkpoint saved to {output_path} — set AETHERIS_FUSION_MODEL_PATH "
        f"to use it, or re-fit on real data with fit_fusion_model.py once network access to "
        f"huggingface.co is available."
    )

    # --verify: load the checkpoint and run a single inference round-trip
    if args.verify:
        loaded = EvidenceFusionModel()
        loaded.load(output_path)
        test_input = np.array([[0.5, 0.5, 0.5]])
        prob = loaded.predict_proba(test_input)
        assert prob.shape == (1,), f"Expected shape (1,), got {prob.shape}"
        assert 0.0 <= float(prob[0]) <= 1.0, f"Probability out of range: {prob[0]}"
        print(f"Verify OK: loaded checkpoint, inference returned p={prob[0]:.4f}")


if __name__ == "__main__":
    main()

