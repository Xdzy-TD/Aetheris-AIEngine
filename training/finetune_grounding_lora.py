"""LoRA fine-tuning for the grounding specialist's CLIPSeg backbone.

Trains a LoRA adapter for ``CIDAS/clipseg-rd64-refined`` (the checkpoint
``specialists/vlm_grounding/model.py`` loads) on a real remote-sensing
open-vocabulary segmentation dataset of (image, text prompt, binary mask)
triples. Point the output at ``AETHERIS_GROUNDING_LORA_PATH`` to make
``vlm_grounding`` use it.

This is the same recipe published for CLIPSeg-on-FLAIR land cover (Garioud
et al., "Learning transferable land cover semantics for open vocabulary
interactions with remote sensing images", ISPRS J. Photogramm. 2025), which
lifted CLIPSeg from ~11 to ~49 mIoU on aerial imagery by fully fine-tuning
the decoder and the CLIP attention projections; LoRA here adapts the
projections instead of fully fine-tuning them, for far fewer trainable
params. No real-mask remote-sensing dataset is prepackaged on the Hub in a
ready (image, prompt, mask) layout, so --dataset is required: point it at
one you've prepared, e.g. by turning a labelled RS segmentation dataset
(FLAIR, LoveDA, iSAID) into per-class binary masks with the class name as
the prompt.

Needs a GPU and Hugging Face Hub access — neither is available in this
project's default offline/CPU-only sandbox, so no adapter ships in this
repo. Run this on a real training machine:

    pip install -e ".[vlm,vlm-train]"
    python training/finetune_grounding_lora.py --dataset <your-hf-dataset-id> \
        --output models/lora/grounding
    export AETHERIS_GROUNDING_LORA_PATH=models/lora/grounding
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

_BASE_MODEL = "CIDAS/clipseg-rd64-refined"
_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]  # CLIP attention projections


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="HF dataset id with image/prompt/mask columns")
    p.add_argument("--image-col", default="image")
    p.add_argument("--prompt-col", default="prompt")
    p.add_argument("--mask-col", default="mask")
    p.add_argument("--output", default="models/lora/grounding")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    processor = CLIPSegProcessor.from_pretrained(_BASE_MODEL)
    model = CLIPSegForImageSegmentation.from_pretrained(_BASE_MODEL)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            target_modules=_TARGET_MODULES,
            modules_to_save=["decoder"],  # small head; fully fine-tuned, per the FLAIR recipe
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    train_ds = load_dataset(args.dataset)
    train_ds = train_ds["train"] if "train" in train_ds else train_ds[next(iter(train_ds))]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def _collate(batch: list[dict]) -> tuple[dict, torch.Tensor]:
        images = [b[args.image_col].convert("RGB") for b in batch]
        prompts = [b[args.prompt_col] for b in batch]
        masks = [b[args.mask_col].convert("L") for b in batch]
        enc = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        target = torch.stack(
            [torch.from_numpy((np.array(m) > 127).astype("float32")) for m in masks]
        )
        return enc, target

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)

    model.train()
    for epoch in range(args.epochs):
        for step, (enc, target) in enumerate(loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            target = target.to(device)

            logits = model(**enc).logits  # [B, H', W'] — CLIPSeg's own internal resolution
            # Ground-truth masks are at source resolution; resize to match,
            # rather than assuming a fixed size the checkpoint might not use.
            resized_target = F.interpolate(
                target.unsqueeze(1), size=logits.shape[-2:], mode="nearest"
            ).squeeze(1)

            loss = F.binary_cross_entropy_with_logits(logits, resized_target)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if step % 10 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print(f"LoRA adapter saved to {args.output} — set AETHERIS_GROUNDING_LORA_PATH to use it.")


if __name__ == "__main__":
    main()
