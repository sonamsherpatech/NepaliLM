"""
NepaliLM — Task 2: QLoRA Fine-tuning Script
Trains Qwen 2.5 7B with QLoRA on Nepali instruction pairs.

Modes:
  --mode test    → 1k samples, your RTX 3050 (validates pipeline, free)
  --mode local   → full dataset, your RTX 3050 (slow but works for 3B)
  --mode runpod  → full dataset, A100 80GB (recommended for 7B)

Usage:
  python training/train.py --mode test
  python training/train.py --mode runpod
"""

import os
import sys
import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    base_model:     str  = "Qwen/Qwen2.5-7B"

    # Paths
    train_path:     str  = "data/instructions/train.jsonl"
    val_path:       str  = "data/instructions/val.jsonl"
    output_dir:     str  = "training/checkpoints"

    # LoRA
    lora_r:         int  = 16      # rank — higher = more capacity, more VRAM
    lora_alpha:     int  = 32      # scaling factor (usually 2x rank)
    lora_dropout:   float = 0.05

    # Training — conservative defaults for RTX 3050
    max_seq_length: int  = 512     # reduce to 256 if OOM on 3050
    batch_size:     int  = 1       # always 1 on 6GB
    grad_accum:     int  = 16      # effective batch = batch_size * grad_accum
    num_epochs:     int  = 3
    learning_rate:  float = 2e-4
    warmup_ratio:   float = 0.05
    lr_scheduler:   str  = "cosine"

    # Logging
    log_steps:      int  = 10
    save_steps:     int  = 100
    eval_steps:     int  = 100


# ── Mode presets ──────────────────────────────────────────────────────────────

MODE_PRESETS = {
    "test": {
        # RTX 3050 — pipeline validation only
        "base_model":     "Qwen/Qwen2.5-1.5B",   # tiny model, fits easily
        "max_samples":    1_000,
        "max_seq_length": 256,
        "batch_size":     1,
        "grad_accum":     4,
        "num_epochs":     1,
        "lora_r":         8,
        "description":    "Pipeline test — Qwen 1.5B, 1k samples, 1 epoch (~10 min on RTX 3050)"
    },
    "local": {
        # RTX 3050 — full dataset on smaller model
        "base_model":     "Qwen/Qwen2.5-3B",
        "max_samples":    None,
        "max_seq_length": 512,
        "batch_size":     1,
        "grad_accum":     16,
        "num_epochs":     3,
        "lora_r":         16,
        "description":    "Local full run — Qwen 3B, full dataset, 3 epochs (~8 hrs on RTX 3050)"
    },
    "runpod": {
        # RunPod A100 — full 7B model
        "base_model":     "Qwen/Qwen2.5-7B",
        "max_samples":    None,
        "max_seq_length": 1024,
        "batch_size":     4,
        "grad_accum":     8,
        "num_epochs":     3,
        "lora_r":         16,
        "description":    "RunPod A100 — Qwen 7B, full dataset, 3 epochs (~10 hrs, ~$15)"
    },
}


# ── Quantization config ───────────────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit quantization — critical for fitting 7B on small GPU."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — best quality
        bnb_4bit_use_double_quant=True,      # extra compression, saves ~0.4GB
    )


# ── LoRA config ───────────────────────────────────────────────────────────────

def get_lora_config(cfg: TrainConfig) -> LoraConfig:
    """
    target_modules: which layers to apply LoRA to.
    Qwen 2.5 uses q_proj, k_proj, v_proj, o_proj for attention.
    Adding gate_proj, up_proj, down_proj covers the MLP too — better for language tasks.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",       # MLP
        ],
        bias="none",
        inference_mode=False,
    )


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_jsonl(path: str, max_samples: int = None) -> Dataset:
    data = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)


# ── VRAM monitor ──────────────────────────────────────────────────────────────

def print_vram():
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {used:.1f}GB / {total:.1f}GB used")


def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    print("(LoRA keeps this small — usually 0.1–1% of total params)")


# ── Main training function ────────────────────────────────────────────────────

def train(mode: str = "test"):
    preset = MODE_PRESETS[mode]
    print(f"\nNepaliLM — Training ({mode} mode)")
    print(f"{'='*60}")
    print(f"{preset['description']}")
    print(f"{'='*60}\n")

    # Apply preset to config
    cfg = TrainConfig()
    cfg.base_model     = preset["base_model"]
    cfg.max_seq_length = preset["max_seq_length"]
    cfg.batch_size     = preset["batch_size"]
    cfg.grad_accum     = preset["grad_accum"]
    cfg.num_epochs     = preset["num_epochs"]
    cfg.lora_r         = preset["lora_r"]
    cfg.lora_alpha     = cfg.lora_r * 2
    max_samples        = preset["max_samples"]

    # ── Check GPU ──
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training on CPU will be extremely slow.")
        print("For test mode this is okay. For real training use a GPU.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print_vram()

    # ── Load tokenizer ──
    print(f"\nLoading tokenizer: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
    )
    # Qwen doesn't have a pad token by default — use eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # important for causal LM training

    # ── Load model in 4-bit ──
    print(f"Loading model in 4-bit: {cfg.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False              # required for gradient checkpointing
    model.config.pretraining_tp = 1

    print_vram()

    # ── Apply LoRA ──
    print("\nApplying LoRA adapters...")
    lora_cfg = get_lora_config(cfg)
    model    = get_peft_model(model, lora_cfg)
    print_trainable_params(model)
    print_vram()

    # ── Load dataset ──
    print(f"\nLoading dataset...")
    train_ds = load_jsonl(cfg.train_path, max_samples=max_samples)
    val_ds   = load_jsonl(cfg.val_path,   max_samples=500)  # small val set always
    print(f"Train: {len(train_ds):,} pairs")
    print(f"Val:   {len(val_ds):,} pairs")

    # ── Training arguments ──
    output_dir = Path(cfg.output_dir) / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if W&B is available
    use_wandb = False
    try:
        import wandb
        use_wandb = True
        wandb.init(project="NepaliLM", name=f"nepalilm-{mode}")
    except ImportError:
        print("W&B not installed — logging to console only")

    sft_config = SFTConfig(
        output_dir=str(output_dir),

        # Batching
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.grad_accum,

        # Memory optimization — critical for 6GB
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        optim="paged_adamw_8bit",       # 8-bit optimizer saves ~1GB VRAM
        dataloader_pin_memory=False,    # saves CPU RAM

        # Learning rate
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=0.3,

        # Sequence length
        max_seq_length=cfg.max_seq_length,

        # Logging
        logging_steps=cfg.log_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,             # keep only 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Dataset
        dataset_text_field="text",      # use the combined prompt+response field
        remove_unused_columns=True,

        # Reporting
        report_to="wandb" if use_wandb else "none",
        run_name=f"nepalilm-{mode}",
    )

    # ── Trainer ──
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        peft_config=lora_cfg,
    )

    # ── Train ──
    print(f"\nStarting training...")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum}")
    print(f"Output dir: {output_dir}\n")

    trainer.train()

    # ── Save final adapter ──
    adapter_path = Path("training/adapter") / mode
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nLoRA adapter saved → {adapter_path}")
    print("Next step: run training/merge_and_export.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NepaliLM QLoRA training")
    parser.add_argument(
        "--mode",
        choices=["test", "local", "runpod"],
        default="test",
        help="test=pipeline check, local=RTX3050 full, runpod=A100 full"
    )
    args = parser.parse_args()
    train(mode=args.mode)