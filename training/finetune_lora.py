#!/usr/bin/env python3
"""LoRA fine-tune Aura's base model on personality data via MLX.

This creates a LoRA adapter that makes "Aura" the model's baseline
personality, not "helpful assistant." After fine-tuning, personality
drift in long conversations is eliminated because there's nothing
to drift BACK to.

Prerequisites:
    pip install mlx-lm

Usage:
    # 1. Build training data
    python training/build_dataset.py

    # 2. Fine-tune (takes ~10-30 min on M1 Pro)
    python training/finetune_lora.py

    # 3. The adapter is saved to training/adapters/aura-personality/
    #    Aura's MLX client auto-loads it on next boot.

Config:
    MODEL: The base model to fine-tune (default: whatever's loaded)
    EPOCHS: Training epochs (2-3 is usually enough for personality)
    LORA_RANK: LoRA rank (8-16 for personality, 32+ for knowledge)
"""
import json
import os
import sys
from pathlib import Path

# Paths
TRAINING_DIR = Path(__file__).parent
DATA_DIR = TRAINING_DIR / "data"
ADAPTER_DIR = TRAINING_DIR / "adapters" / "aura-personality"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"

# Hyperparameters
LORA_RANK = 16
LORA_LAYERS = 16       # Number of layers to apply LoRA to
EPOCHS = 3
BATCH_SIZE = 1          # Keep small for M-series memory
LEARNING_RATE = 1e-5    # Conservative for personality (don't break knowledge)
WARMUP_STEPS = 50


def find_base_model() -> str:
    """Find the currently loaded base model path."""
    # Check common MLX model locations
    candidates = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/models"),
        os.path.expanduser("~/.aura/models"),
    ]

    for base in candidates:
        if os.path.isdir(base):
            for d in Path(base).rglob("config.json"):
                model_dir = d.parent
                if (model_dir / "model.safetensors").exists() or list(model_dir.glob("model-*.safetensors")):
                    return str(model_dir)

    # Fallback: use mlx_lm's model resolution
    return "mlx-community/Qwen2.5-32B-Instruct-4bit"


def main():
    try:
        from mlx_lm import lora as mlx_lora
    except ImportError:
        print("ERROR: mlx-lm not installed. Run: pip install mlx-lm")
        sys.exit(1)

    if not TRAIN_FILE.exists():
        print(f"Training data not found at {TRAIN_FILE}")
        print("Run: python training/build_dataset.py")
        sys.exit(1)

    # Count examples
    with open(TRAIN_FILE) as f:
        n_train = sum(1 for _ in f)
    with open(VAL_FILE) as f:
        n_val = sum(1 for _ in f)

    model_path = find_base_model()

    print("=" * 60)
    print("  AURA PERSONALITY LoRA FINE-TUNE")
    print("=" * 60)
    print(f"  Base model:     {model_path}")
    print(f"  Training data:  {n_train} examples")
    print(f"  Validation:     {n_val} examples")
    print(f"  LoRA rank:      {LORA_RANK}")
    print(f"  LoRA layers:    {LORA_LAYERS}")
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Learning rate:  {LEARNING_RATE}")
    print(f"  Adapter output: {ADAPTER_DIR}")
    print("=" * 60)
    print()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # Write LoRA config
    lora_config = {
        "lora_rank": LORA_RANK,
        "lora_layers": LORA_LAYERS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "warmup_steps": WARMUP_STEPS,
        "model": model_path,
        "train_data": str(TRAIN_FILE),
        "val_data": str(VAL_FILE),
        "adapter_path": str(ADAPTER_DIR),
    }

    config_path = ADAPTER_DIR / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(lora_config, f, indent=2)

    print(f"Config saved to {config_path}")
    print("Starting fine-tune...")
    print()

    # Run MLX LoRA fine-tune
    try:
        os.system(
            f"python -m mlx_lm.lora "
            f"--model {model_path} "
            f"--train "
            f"--data {DATA_DIR} "
            f"--adapter-path {ADAPTER_DIR} "
            f"--lora-layers {LORA_LAYERS} "
            f"--batch-size {BATCH_SIZE} "
            f"--num-layers {LORA_RANK} "
            f"--iters {EPOCHS * n_train // BATCH_SIZE} "
            f"--learning-rate {LEARNING_RATE} "
        )
        print()
        print(f"LoRA adapter saved to: {ADAPTER_DIR}")
        print("To use: set AURA_LORA_PATH={} in your environment or config.".format(ADAPTER_DIR))
    except Exception as e:
        print(f"Fine-tune failed: {e}")
        print("You can run it manually:")
        print(f"  python -m mlx_lm.lora --model {model_path} --train --data {DATA_DIR} --adapter-path {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
