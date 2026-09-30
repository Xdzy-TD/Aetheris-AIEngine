"""
RS Model Adaptation surface — discovers runtime adapter state.

Reads LoRA/adapter environment variables, probes whether each path actually
exists, and reports what's loaded vs. what's configured. Called once at
startup; result carried on ControllerContext and exposed via ``GET /v1/models``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AdaptationConfig:
    """Runtime state of a single task's domain adapter."""
    task: str
    adapter_path: str | None = None
    adapter_loaded: bool = False
    base_model_id: str | None = None
    domain: str = "general"  # "general" | "remote-sensing"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "adapter_path": self.adapter_path,
            "adapter_loaded": self.adapter_loaded,
            "base_model_id": self.base_model_id,
            "domain": self.domain,
        }


# env-var → task mapping (the only three adapter paths AETHERIS supports)
_ADAPTER_ENV = {
    "vlm_vqa_caption": "AETHERIS_VLM_LORA_PATH",
    "vlm_grounding": "AETHERIS_GROUNDING_LORA_PATH",
    "fusion_change_detection": "AETHERIS_FUSION_MODEL_PATH",
}


def discover_adaptations() -> dict[str, AdaptationConfig]:
    """Probe every adapter env var and report what actually loaded.

    A configured path that doesn't resolve to a real directory/file is
    reported as ``adapter_loaded=False`` — honest, not silent.
    """
    configs: dict[str, AdaptationConfig] = {}
    for task, env_var in _ADAPTER_ENV.items():
        path = os.environ.get(env_var)
        exists = bool(path and Path(path).exists())
        configs[task] = AdaptationConfig(
            task=task,
            adapter_path=path,
            adapter_loaded=exists,
            domain="remote-sensing" if exists else "general",
        )
    return configs
