"""LoRA fine-tuning for evacuation decision LLM.

QLoRA (4-bit) fine-tuning of Qwen2.5-3B-Instruct on simulation oracle data.

Fits within 8-10GB VRAM — single RTX 4090 compatible.

Usage:
    # Step 1: Generate training data
    python -m training.generate_data --num-scenarios 50 --output data/train_lora.jsonl

    # Step 2: Train
    python -m training.train_lora --data data/train_lora.jsonl --epochs 3

    # Step 3: Compare
    python -m training.compare_models --lora output/lora_evac/checkpoint-xxx
"""

import os
import sys
import json
import argparse
import torch
from typing import Dict, List
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================================================================
# Prompt template for instruction-tuning
# ================================================================
INSTRUCTION_TEMPLATE = (
    "<|im_start|>system\n"
    "你是一个灾害疏散决策助手。根据场景信息，输出安全合理的疏散决策JSON。\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "{instruction}\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{output}\n"
    "<|im_end|>"
)


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_example(sample: dict) -> str:
    """Format a single training example into instruction template."""
    return INSTRUCTION_TEMPLATE.format(
        instruction=sample["instruction"],
        output=sample["output"],
    )


def load_and_prepare_data(train_path: str, val_path: str = None):
    """Load JSONL and convert to HuggingFace Dataset."""
    train_data = load_jsonl(train_path)

    def _add_text(examples):
        return {"text": [format_example(s) for s in examples]}

    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(_add_text, batched=True,
                                       remove_columns=train_dataset.column_names)

    if val_path and os.path.exists(val_path):
        val_data = load_jsonl(val_path)
        val_dataset = Dataset.from_list(val_data)
        val_dataset = val_dataset.map(_add_text, batched=True,
                                       remove_columns=val_dataset.column_names)
    else:
        split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]

    return train_dataset, val_dataset


def train(args):
    print("=" * 60)
    print("  LoRA Fine-tuning — Evacuation Decision LLM")
    print(f"  Base: {args.base_model}")
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}")
    print(f"  LR: {args.lr}  LoRA rank: {args.lora_r}")
    print("=" * 60)

    # ---- 1. Load data ----
    train_ds, val_ds = load_and_prepare_data(args.data, args.val_data)
    print(f"\n[Data] {len(train_ds)} train / {len(val_ds)} val examples")

    # ---- 2. Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        pad_token="<|endoftext|>",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_ds = train_ds.map(tokenize, batched=True,
                             remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True,
                         remove_columns=["text"])

    # ---- 3. Load model with QLoRA ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = prepare_model_for_kbit_training(model)

    # ---- 4. LoRA config ----
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 5. Training ----
    output_dir = args.output_dir or f"output/lora_evac_{args.lora_r}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        dataloader_num_workers=0,
        report_to="none",
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=args.seed,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # ---- 6. Train ----
    print(f"\n[Training] Starting... output_dir={output_dir}")
    trainer.train()

    # ---- 7. Save final adapter ----
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[Done] LoRA adapter saved to {final_dir}")
    print(f"  To use: --lora-path {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune evacuation decision LLM")
    parser.add_argument("--data", type=str, required=True,
                        help="Training data JSONL path")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Validation data JSONL path")
    parser.add_argument("--base-model", type=str,
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name (non-AWQ for training)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
