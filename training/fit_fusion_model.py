"""Fit :class:`specialists._fusion_model.EvidenceFusionModel` on a labeled
bi-temporal change-detection dataset.

Default dataset is OSCD (Daudt et al., IGARSS 2018) via the
``blanchon/OSCD_MSI`` Hugging Face dataset — 24 real Sentinel-2 pairs with
pixel-level change ground truth. Point the output at
``AETHERIS_FUSION_MODEL_PATH`` to make ``fusion_change_detection`` use it.

Reuses ``FusionChangeDetectionSpecialist._bitemporal_evidence`` to build the
optical (and, if given, SAR) evidence channels exactly as inference does —
training on a different feature pipeline than the one served would silently
invalidate the fit. OSCD has no SAR pairs, so the SAR channel trains on its
neutral (0.5) value unless --sar1-col/--sar2-col point at a dataset that has
one too.

Needs internet access to pull the dataset — not a GPU, since this is a
3-feature logistic regression, but AETHERIS's offline sandbox blocks the
download all the same (see docs/STATUS.md), so nobody has run this here
either. Column names vary by dataset — check the dataset card and pass
--image1-col/--image2-col/--mask-col (and optionally --date1-col/--date2-col)
if they differ from the guessed defaults below; this repo has not verified
blanchon/OSCD_MSI's exact schema against a live download.

    pip install -e ".[vlm-train]"
    python training/fit_fusion_model.py --output models/fusion/evidence_fusion.json
    export AETHERIS_FUSION_MODEL_PATH=models/fusion/evidence_fusion.json
"""

from __future__ import annotations

import argparse
import io
from datetime import datetime

import numpy as np
from datasets import load_dataset
from PIL import Image

from specialists._fusion_model import EvidenceFusionModel
from specialists.fusion_change_detection.model import FusionChangeDetectionSpecialist

_NEUTRAL = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="blanchon/OSCD_MSI", help="HF dataset id")
    p.add_argument("--split", default="train")
    p.add_argument("--image1-col", default="image_1")
    p.add_argument("--image2-col", default="image_2")
    p.add_argument("--mask-col", default="mask")
    p.add_argument("--date1-col", default=None, help="Optional ISO-date column for T1")
    p.add_argument("--date2-col", default=None, help="Optional ISO-date column for T2")
    p.add_argument("--sar1-col", default=None, help="Optional SAR-T1 column, if dataset has one")
    p.add_argument("--sar2-col", default=None, help="Optional SAR-T2 column")
    p.add_argument(
        "--pixels-per-pair", type=int, default=2000,
        help="Random valid-pixel subsample per image pair (keeps the fit fast/bounded)",
    )
    p.add_argument("--output", default="models/fusion/evidence_fusion.json")
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _to_pil(value: object, mode: str = "RGB") -> Image.Image:
    """Dataset image cells arrive as PIL Images, {'bytes': ...} dicts, or arrays."""
    if isinstance(value, Image.Image):
        img = value
    elif isinstance(value, dict) and "bytes" in value:
        img = Image.open(io.BytesIO(value["bytes"]))
    else:
        img = Image.fromarray(np.asarray(value))
    return img.convert(mode)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ds = load_dataset(args.dataset, split=args.split)

    X_rows: list[list[float]] = []
    y_rows: list[float] = []

    for ex in ds:
        src1, src2 = _to_pil(ex[args.image1_col]), _to_pil(ex[args.image2_col])
        mask = np.asarray(_to_pil(ex[args.mask_col], mode="L").resize((256, 256))) > 127

        optical, _, valid, _ = FusionChangeDetectionSpecialist._bitemporal_evidence(src1, src2)

        sar = np.full((256, 256), _NEUTRAL)
        if args.sar1_col and args.sar2_col:
            sar, _, _, _ = FusionChangeDetectionSpecialist._bitemporal_evidence(
                _to_pil(ex[args.sar1_col], mode="L"),
                _to_pil(ex[args.sar2_col], mode="L"),
                despeckle=True,
            )

        temporal = _NEUTRAL
        if args.date1_col and args.date2_col:
            d1 = datetime.fromisoformat(str(ex[args.date1_col]))
            d2 = datetime.fromisoformat(str(ex[args.date2_col]))
            temporal = min(abs((d2 - d1).days) / 730.0, 1.0)

        valid_idx = np.argwhere(valid)
        if len(valid_idx) == 0:
            continue
        n = min(args.pixels_per_pair, len(valid_idx))
        for yy, xx in valid_idx[rng.choice(len(valid_idx), size=n, replace=False)]:
            X_rows.append([optical[yy, xx], sar[yy, xx], temporal])
            y_rows.append(float(mask[yy, xx]))

    if not X_rows:
        raise SystemExit(
            "No valid training pixels found — check --image1-col/--image2-col/--mask-col "
            "against the dataset card; the guessed defaults may not match this dataset."
        )

    X, y = np.array(X_rows), np.array(y_rows)
    print(f"Fitting on {len(y)} pixels from {len(ds)} image pairs ({y.mean():.1%} positive).")

    model = EvidenceFusionModel()
    model.fit(X, y, lr=args.lr, epochs=args.epochs, l2=args.l2)
    model.save(args.output)
    print(f"Fusion model saved to {args.output} — set AETHERIS_FUSION_MODEL_PATH to use it.")


if __name__ == "__main__":
    main()
