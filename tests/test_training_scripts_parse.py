"""Smoke-test: verify each training script can be imported without error.

This catches syntax errors, missing constants, and broken imports at CI
time rather than at training time. Does NOT run training — only imports.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Add project root to sys.path so training/ is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


TRAINING_SCRIPTS = [
    "training.download_adapters",
    "training.fit_fusion_model_synthetic",
]

# These require GPU/Hub deps that may not be installed — only test if deps exist
OPTIONAL_SCRIPTS = [
    "training.finetune_vqa_lora",
    "training.finetune_grounding_lora",
    "training.fit_fusion_model",
]


@pytest.mark.parametrize("module_name", TRAINING_SCRIPTS)
def test_training_script_imports(module_name: str) -> None:
    """Each training script must parse and import without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None, f"Failed to import {module_name}"


@pytest.mark.parametrize("module_name", OPTIONAL_SCRIPTS)
def test_optional_training_script_imports(module_name: str) -> None:
    """Optional scripts (GPU deps) import if their dependencies are met."""
    try:
        mod = importlib.import_module(module_name)
        assert mod is not None
    except ImportError as e:
        pytest.skip(f"Optional dependency missing for {module_name}: {e}")
