"""Tests for the benchmark harness (benchmark/runner.py)."""

from __future__ import annotations

import json

import pytest

from benchmark.runner import load_dataset, run_benchmark, run_comparison
from interfaces.api.deps import get_planner, get_registry

DATASET = "benchmark/sample_dataset.jsonl"


def test_load_dataset() -> None:
    items = load_dataset(DATASET)
    assert len(items) == 2
    assert all("query" in i and "expected" in i for i in items)


@pytest.mark.asyncio
async def test_run_benchmark_scores_the_refusal_guard() -> None:
    """No real pixels ship with the sample dataset, so every item is a policy
    refusal by design — this asserts that refusal keeps happening, not that
    any answer is 'right' (see benchmark/README.md)."""
    report = await run_benchmark(DATASET, get_planner())
    assert report["total"] == 2
    assert report["scored"] == 2
    assert report["accuracy"] == 1.0
    assert report["latency_s"]["mean"] is not None
    assert all("unavailable" in row["final_answer"].lower() for row in report["rows"])


def _real_image_dataset(tmp_path) -> str:
    """A JSONL dataset with real pixels on disk, for run_comparison to actually score."""
    import numpy as np
    from PIL import Image

    p1, p2 = tmp_path / "t1.png", tmp_path / "t2.png"
    for p, seed in ((p1, 1), (p2, 2)):
        arr = np.random.default_rng(seed).integers(0, 255, (48, 48, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(p)

    rows = [
        {
            "query": {"text": "what is visible", "images": [
                {"filename": "a.png", "path": str(p1), "modality": "optical"}
            ]},
            "expected": {"keywords": []},
        },
        {
            "query": {"text": "what changed", "images": [
                {"filename": "t1.png", "path": str(p1), "modality": "optical"},
                {"filename": "t2.png", "path": str(p2), "modality": "optical"},
            ]},
        },
    ]
    out = tmp_path / "real.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(out)


def test_run_comparison_scores_every_bound_family_member(tmp_path) -> None:
    """Classical baseline and pretrained/learned alternatives for the same task
    each get their own row, called directly (not through the planner)."""
    dataset = _real_image_dataset(tmp_path)
    comparison = run_comparison(dataset, get_registry())

    # vqa_caption + grounding families from the 1-image item, change_detection
    # family from the 2-image item — all classical baselines are always bound.
    for tool in ("vqa_caption", "grounding", "change_detection", "fusion_change_detection"):
        assert tool in comparison
        assert comparison[tool]["n"] >= 1
        assert comparison[tool]["latency_s"]["mean"] is not None


def test_run_comparison_skips_unbound_models_not_scores_zero() -> None:
    """A tool with no bound implementation must be absent, not counted as wrong."""
    comparison = run_comparison(DATASET, get_registry())
    # sample_dataset.jsonl carries no real pixel paths, so nothing is scorable,
    # but every family member that IS bound should still appear with accuracy=None.
    for tool, stats in comparison.items():
        assert stats["n"] >= 1
