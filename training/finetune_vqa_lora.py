"""LoRA/QLoRA fine-tuning for the VQA/caption specialist's BLIP backbone.

Trains a LoRA adapter for ``Salesforce/blip-vqa-base`` (the checkpoint
``specialists/vlm_caption/model.py`` loads) on a real remote-sensing
VQA/caption dataset — default: RSVQA-LR (Lobry et al., IEEE TGRS 2020) via
the ``exibings/rsvqa-lr`` Hugging Face dataset. Point the output at
``AETHERIS_VLM_LORA_PATH`` to make ``vlm_vqa_caption`` use it.

Needs a GPU and Hugging Face Hub access for the base weights/dataset
download — neither is available in this project's default offline/CPU-only
sandbox, so no adapter ships in this repo. Run this on a real training
machine:

    pip install -e ".[vlm,vlm-train]"
    python training/finetune_vqa_lora.py --output models/lora/vqa_caption
    export AETHERIS_VLM_LORA_PATH=models/lora/vqa_caption

Add --qlora for 4-bit base weights (needs bitsandbytes + a CUDA GPU).
Datasets vary in column naming — check the dataset card and pass
--image-col/--question-col/--answer-col if they differ from the defaults.
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import BlipForQuestionAnswering, BlipProcessor, Trainer, TrainingArguments

_BASE_MODEL = "Salesforce/blip-vqa-base"
# BLIP's attention projections: fused qkv + output proj in the vision
# encoder, separate query/key/value in the BERT-style text decoder.
_TARGET_MODULES = ["query", "key", "value", "qkv", "projection"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="exibings/rsvqa-lr", help="HF dataset id")
    p.add_argument("--image-col", default="image")
    p.add_argument("--question-col", default="question")
    p.add_argument("--answer-col", default="answer")
    p.add_argument("--output", default="models/lora/vqa_caption")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--qlora", action="store_true", help="4-bit base weights (bitsandbytes + CUDA)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    processor = BlipProcessor.from_pretrained(_BASE_MODEL)

    load_kwargs = {}
    if args.qlora:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        load_kwargs["device_map"] = "auto"
    model = BlipForQuestionAnswering.from_pretrained(_BASE_MODEL, **load_kwargs)
    if args.qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            target_modules=_TARGET_MODULES, bias="none",
        ),
    )
    model.print_trainable_parameters()

    ds = load_dataset(args.dataset)
    train_ds = ds["train"] if "train" in ds else ds[next(iter(ds))]

    def _prep(example: dict) -> dict:
        enc = processor(
            images=example[args.image_col], text=example[args.question_col],
            return_tensors="pt", padding="max_length", truncation=True, max_length=32,
        )
        labels = processor(
            text=str(example[args.answer_col]), return_tensors="pt",
            padding="max_length", truncation=True, max_length=8,
        ).input_ids
        return {
            "pixel_values": enc["pixel_values"][0],
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": labels[0],
        }

    train_ds = train_ds.map(_prep, remove_columns=train_ds.column_names)
    train_ds.set_format(type="torch")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            logging_steps=10,
            save_strategy="epoch",
            report_to=[],
        ),
        train_dataset=train_ds,
    )
    trainer.train()

    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print(f"LoRA adapter saved to {args.output} — set AETHERIS_VLM_LORA_PATH to use it.")


if __name__ == "__main__":
    main()
