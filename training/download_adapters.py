"""Download pre-trained LoRA adapter checkpoints for AETHERIS VLM specialists.

This script downloads RS-adapted LoRA adapters from a configurable source
(Hugging Face Hub or direct URL) and verifies their SHA-256 integrity.

Usage:
    python training/download_adapters.py --all
    python training/download_adapters.py --adapter vlm_vqa
    python training/download_adapters.py --adapter vlm_grounding
    python training/download_adapters.py --adapter vlm_vqa --source /local/path/adapter.safetensors

After download, set the corresponding env vars:
    AETHERIS_VLM_LORA_PATH=models/adapters/vlm_vqa/adapter_model.safetensors
    AETHERIS_GROUNDING_LORA_PATH=models/adapters/vlm_grounding/adapter_model.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

# Registry of known adapters. In a real deployment these would point to
# Hugging Face Hub model IDs or direct download URLs. For now they serve
# as a machine-readable manifest of what adapters exist and where they go.
ADAPTER_REGISTRY: dict[str, dict[str, str]] = {
    "vlm_vqa": {
        "description": "RS-adapted BLIP-2 LoRA for VQA/caption (RSVQA-LR fine-tune)",
        "target_dir": "models/adapters/vlm_vqa",
        "env_var": "AETHERIS_VLM_LORA_PATH",
        "hub_id": "",  # placeholder: set when adapter is published
        "sha256": "",  # placeholder: set when adapter is published
    },
    "vlm_grounding": {
        "description": "RS-adapted CLIPSeg LoRA for pixel-level grounding (FLAIR recipe)",
        "target_dir": "models/adapters/vlm_grounding",
        "env_var": "AETHERIS_GROUNDING_LORA_PATH",
        "hub_id": "",  # placeholder: set when adapter is published
        "sha256": "",  # placeholder: set when adapter is published
    },
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_from_hub(hub_id: str, target_dir: Path) -> Path:
    """Download adapter files from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed. Install it with:\n"
            "  pip install huggingface-hub\n"
            "Or use --source to provide a local path.",
            file=sys.stderr,
        )
        sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from Hugging Face Hub: {hub_id} -> {target_dir}")
    snapshot_download(repo_id=hub_id, local_dir=str(target_dir))
    return target_dir


def copy_from_local(source_path: str, target_dir: Path) -> Path:
    """Copy adapter files from a local path."""
    src = Path(source_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, target_dir / f.name)
                print(f"  Copied {f.name}")
    elif src.is_file():
        shutil.copy2(src, target_dir / src.name)
        print(f"  Copied {src.name}")
    else:
        print(f"ERROR: Source path does not exist: {source_path}", file=sys.stderr)
        sys.exit(1)
    return target_dir


def download_adapter(name: str, source: str | None = None) -> None:
    """Download or copy a single adapter."""
    if name not in ADAPTER_REGISTRY:
        print(f"ERROR: Unknown adapter '{name}'. Known: {list(ADAPTER_REGISTRY)}", file=sys.stderr)
        sys.exit(1)

    info = ADAPTER_REGISTRY[name]
    target_dir = Path(info["target_dir"])
    expected_sha = info["sha256"]

    print(f"\n{'='*60}")
    print(f"Adapter: {name}")
    print(f"Description: {info['description']}")
    print(f"Target: {target_dir}")
    print(f"Env var: {info['env_var']}")
    print(f"{'='*60}")

    if source:
        copy_from_local(source, target_dir)
    elif info["hub_id"]:
        download_from_hub(info["hub_id"], target_dir)
    else:
        print(
            f"WARNING: No hub_id configured for '{name}' and no --source provided.\n"
            f"This adapter is not yet published. Use --source to provide a local path,\n"
            f"or train one with the corresponding training script:\n"
            f"  python training/finetune_vqa_lora.py    (for vlm_vqa)\n"
            f"  python training/finetune_grounding_lora.py (for vlm_grounding)",
            file=sys.stderr,
        )
        return

    # Verify SHA-256 if expected hash is configured
    if expected_sha:
        for fpath in sorted(target_dir.rglob("*.safetensors")):
            actual = sha256_file(fpath)
            if actual != expected_sha:
                print(
                    f"WARNING: SHA-256 mismatch for {fpath.name}!\n"
                    f"  Expected: {expected_sha}\n"
                    f"  Actual:   {actual}",
                    file=sys.stderr,
                )
            else:
                print(f"SHA-256 verified: {fpath.name}")
    else:
        print("(SHA-256 not configured — skipping integrity check)")

    # Report env var to set
    adapter_files = list(target_dir.rglob("*.safetensors")) or list(target_dir.rglob("*.bin"))
    if adapter_files:
        print(f"\nSet environment variable to use this adapter:")
        print(f"  export {info['env_var']}={adapter_files[0]}")
    else:
        print(f"\nNo .safetensors or .bin files found in {target_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--adapter", choices=list(ADAPTER_REGISTRY), help="Which adapter to download")
    p.add_argument("--all", action="store_true", help="Download all adapters")
    p.add_argument("--source", help="Local path to copy from (instead of Hub download)")
    p.add_argument("--list", action="store_true", dest="list_adapters", help="List available adapters")
    args = p.parse_args()

    if args.list_adapters:
        for name, info in ADAPTER_REGISTRY.items():
            status = "published" if info["hub_id"] else "not yet published"
            print(f"  {name}: {info['description']} ({status})")
        return

    if args.all:
        for name in ADAPTER_REGISTRY:
            download_adapter(name, args.source)
    elif args.adapter:
        download_adapter(args.adapter, args.source)
    else:
        p.print_help()
        print("\nUse --list to see available adapters, --adapter NAME to download one, or --all.")


if __name__ == "__main__":
    main()
