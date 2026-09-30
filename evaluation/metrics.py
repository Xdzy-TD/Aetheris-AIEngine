"""
Geometric metrics — IoU and pixel-wise F1 for masks and bboxes.

Pure math over arrays the caller supplies; no model, no I/O beyond loading a
mask image. Every function returns ``None`` (never ``0.0``) when the inputs
make the metric undefined (e.g. both mask and ground truth are empty), so an
undefined score can't be misread as "zero overlap" — same convention
``benchmark/runner.py`` already uses for an unscorable item.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_binary_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    """Load a mask image (any mode) as a boolean array via a fixed threshold."""
    from PIL import Image

    arr = np.array(Image.open(path).convert("L"))
    return arr > threshold


def iou_mask(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Intersection-over-union between two boolean masks of matching shape."""
    if pred.shape != gt.shape:
        raise ValueError(f"mask shape mismatch: {pred.shape} vs {gt.shape}")
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return None  # both empty — undefined, not perfect agreement
    inter = int(np.logical_and(pred, gt).sum())
    return inter / union


def iou_bbox(pred: list[float], gt: list[float]) -> float | None:
    """IoU between two ``[x0, y0, x1, y1]`` boxes."""
    px0, py0, px1, py1 = pred
    gx0, gy0, gx1, gy1 = gt
    iw = max(0.0, min(px1, gx1) - max(px0, gx0))
    ih = max(0.0, min(py1, gy1) - max(py0, gy0))
    inter = iw * ih
    p_area = max(0.0, px1 - px0) * max(0.0, py1 - py0)
    g_area = max(0.0, gx1 - gx0) * max(0.0, gy1 - gy0)
    union = p_area + g_area - inter
    if union <= 0:
        return None
    return inter / union


def f1_mask(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | None]:
    """Pixel-wise precision/recall/F1 between two boolean masks."""
    if pred.shape != gt.shape:
        raise ValueError(f"mask shape mismatch: {pred.shape} vs {gt.shape}")
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def mean(values: list[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty list — never a fabricated 0.0."""
    return (sum(values) / len(values)) if values else None
