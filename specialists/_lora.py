"""Optional LoRA/QLoRA adapter loading, shared by vlm_caption and vlm_grounding.

Training an adapter is a separate, offline step (see ``training/``) — this
module only *loads* one if a directory is actually configured and present.
No adapter ships in this repo (no GPU/dataset access in the default sandbox),
so by default this is a no-op and the specialists behave exactly as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_lora(model: Any, adapter_path: str | None) -> tuple[Any, bool]:
    """Wrap ``model`` with a PEFT adapter from ``adapter_path`` if it exists.

    Returns ``(model, False)`` unchanged when ``adapter_path`` is falsy, the
    directory doesn't exist, or ``peft`` isn't installed (part of the
    ``vlm-train`` extra) — adaptation is opt-in, never assumed. The bool lets
    a caller report ``lora_applied`` truthfully instead of inferring it from
    a configured path that may not have actually resolved to a real adapter.
    """
    if not adapter_path or not Path(adapter_path).is_dir():
        return model, False
    from peft import PeftModel  # optional dep; only imported once a path is real

    return PeftModel.from_pretrained(model, adapter_path), True
