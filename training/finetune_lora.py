#!/usr/bin/env python3
"""LoRA fine-tune Aura's base model — Project Zenith configuration.

Upgraded from v2:
  - Rank 32 (up from 8) — needed for architecture knowledge density
  - All 64 layers (up from 16) — personality permeates the full model
  - 4096 max sequence length (up from 2048) — longer explanations
  - Gradient checkpointing enabled — essential for rank-32 on 32B
  - Cosine LR schedule — better convergence for larger datasets
  - Lower learning rate (5e-6) — larger dataset, higher rank

Prerequisites:
    pip install mlx-lm

Usage:
    # 1. Build training data
    python training/build_dataset_v3.py

    # 2. Fine-tune (takes ~6-12 hours on M5)
    python training/finetune_lora.py

    # 3. The adapter is saved to training/adapters/aura-personality/
    #    Previous adapter backed up to aura-personality-v4-backup/
"""
import json
import os
import shutil
import sys
from pathlib import Path

# Paths
TRAINING_DIR = Path(__file__).parent
DATA_DIR = TRAINING_DIR / "data"
ADAPTER_DIR = TRAINING_DIR / "adapters" / "aura-personality"
BACKUP_DIR = TRAINING_DIR / "adapters" / "aura-personality-v4-backup"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "valid.jsonl"

# ── Hyperparameters — Project Zenith ──────────────────────────────────────
LORA_RANK = 32          # Up from 8 — architecture knowledge needs density
LORA_LAYERS = 64        # All layers — personality permeates the full model
EPOCHS = 3
BATCH_SIZE = 1          # Keep small for M-series memory
LEARNING_RATE = 5e-6    # Lower — larger dataset + higher rank = more careful
WARMUP_STEPS = 200      # Up from 50 — larger dataset needs longer warmup
MAX_SEQ_LENGTH = 4096   # Up from 2048 — architecture explanations are longer
GRAD_CHECKPOINT = True  # Essential for rank-32 on 32B model
SAVE_EVERY = 500        # Checkpoint frequency


def find_base_model() -> str:
    """Find the base model path. Honors AURA_LORA_BASE_MODEL env first."""
    explicit = os.environ.get("AURA_LORA_BASE_MODEL", "").strip()
    if explicit and Path(explicit).is_dir():
        return explicit

    repo_root = Path(__file__).resolve().parent.parent
    # Prefer the canonical 8-bit base inside the repo before falling back to
    # caches — this is what train_and_fuse.py expects to fuse against.
    candidates = [
        repo_root / "models" / "Qwen2.5-32B-Instruct-8bit",
        repo_root / "models",
        Path(os.path.expanduser("~/.aura/models")),
        Path(os.path.expanduser("~/models")),
        Path(os.path.expanduser("~/.cache/huggingface/hub")),
    ]

    for base in candidates:
        if not base.exists():
            continue
        if (base / "config.json").exists():
            if (base / "model.safetensors").exists() or list(base.glob("model-*.safetensors")):
                return str(base)
        if base.is_dir():
            for d in base.rglob("config.json"):
                model_dir = d.parent
                if (model_dir / "model.safetensors").exists() or list(model_dir.glob("model-*.safetensors")):
                    return str(model_dir)

    # Fallback: use mlx_lm's model resolution (will download).
    return "mlx-community/Qwen2.5-32B-Instruct-4bit"


def backup_existing_adapter():
    """Backup existing adapter before overwriting."""
    if ADAPTER_DIR.exists() and any(ADAPTER_DIR.glob("*.safetensors")):
        print(f"  Backing up existing adapter to {BACKUP_DIR}...")
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(ADAPTER_DIR, BACKUP_DIR)
        print(f"  Backup complete.")


def _latest_checkpoint() -> Path | None:
    """Find the highest-numbered ``NNNNNNN_adapters.safetensors`` checkpoint."""
    if not ADAPTER_DIR.exists():
        return None
    candidates = sorted(ADAPTER_DIR.glob("[0-9]*_adapters.safetensors"))
    return candidates[-1] if candidates else None


def main():
    try:
        from mlx_lm import lora as mlx_lora
    except ImportError:
        print("ERROR: mlx-lm not installed. Run: pip install mlx-lm")
        sys.exit(1)

    resume = "--resume" in sys.argv

    if not TRAIN_FILE.exists():
        print(f"Training data not found at {TRAIN_FILE}")
        print("Run: python training/build_dataset_v3.py")
        sys.exit(1)

    # Count examples
    with open(TRAIN_FILE) as f:
        n_train = sum(1 for _ in f)

    val_file = VAL_FILE if VAL_FILE.exists() else DATA_DIR / "val.jsonl"
    with open(val_file) as f:
        n_val = sum(1 for _ in f)

    model_path = find_base_model()

    # Calculate total iterations
    total_iters = EPOCHS * n_train // BATCH_SIZE

    print("=" * 60)
    print("  PROJECT ZENITH — AURA PERSONALITY LoRA FINE-TUNE")
    print("=" * 60)
    print(f"  Base model:        {model_path}")
    print(f"  Training data:     {n_train} examples")
    print(f"  Validation:        {n_val} examples")
    print(f"  LoRA rank:         {LORA_RANK}")
    print(f"  LoRA layers:       {LORA_LAYERS} (all)")
    print(f"  Epochs:            {EPOCHS}")
    print(f"  Learning rate:     {LEARNING_RATE}")
    print(f"  Max seq length:    {MAX_SEQ_LENGTH}")
    print(f"  Grad checkpoint:   {GRAD_CHECKPOINT}")
    print(f"  Total iterations:  {total_iters}")
    print(f"  Adapter output:    {ADAPTER_DIR}")
    print("=" * 60)
    print()

    # Backup existing adapter — skipped on resume so the partial
    # checkpoints survive.
    if not resume:
        backup_existing_adapter()
    else:
        print("  --resume: skipping adapter backup; reusing existing checkpoints.")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # Write training config for reference
    training_config = {
        "project": "zenith",
        "lora_rank": LORA_RANK,
        "lora_layers": LORA_LAYERS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "warmup_steps": WARMUP_STEPS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "grad_checkpoint": GRAD_CHECKPOINT,
        "model": model_path,
        "train_data": str(TRAIN_FILE),
        "val_data": str(val_file),
        "adapter_path": str(ADAPTER_DIR),
        "total_train_examples": n_train,
        "total_val_examples": n_val,
        "total_iterations": total_iters,
    }

    config_path = ADAPTER_DIR / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)

    print(f"Config saved to {config_path}")
    print("Starting fine-tune...")
    print()

    # ── Write LoRA config YAML (rank/scale set here, not on CLI) ─────────
    lora_config = {
        "lora_parameters": {
            "rank": LORA_RANK,
            "dropout": 0.0,
            "scale": 20.0,
        }
    }
    lora_config_path = ADAPTER_DIR / "lora_config.yaml"
    import yaml
    try:
        with open(lora_config_path, "w") as f:
            yaml.dump(lora_config, f)
    except ImportError:
        # No PyYAML — write as JSON config instead
        lora_config_path = ADAPTER_DIR / "lora_config.json"
        with open(lora_config_path, "w") as f:
            json.dump(lora_config, f, indent=2)

    # ── Build MLX LoRA command ───────────────────────────────────────────
    cmd_parts = [
        "python", "-m", "mlx_lm", "lora",
        "--model", str(model_path),
        "--train",
        "--data", str(DATA_DIR),
        "--adapter-path", str(ADAPTER_DIR),
        "--num-layers", "-1",   # All layers (was --lora-layers, which doesn't exist)
        "--batch-size", str(BATCH_SIZE),
        "--iters", str(total_iters),
        "--learning-rate", str(LEARNING_RATE),
        "--save-every", str(SAVE_EVERY),
        "--steps-per-eval", "500",
        "--steps-per-report", "100",
        "--max-seq-length", str(MAX_SEQ_LENGTH),
        "-c", str(lora_config_path),
    ]

    if GRAD_CHECKPOINT:
        cmd_parts.append("--grad-checkpoint")

    if resume:
        latest = _latest_checkpoint()
        if latest is None:
            print("  --resume requested but no checkpoint found; starting fresh.")
        else:
            print(f"  --resume: continuing from {latest.name}")
            cmd_parts.extend(["--resume-adapter-file", str(latest)])

    cmd_display = " ".join(cmd_parts)
    print(f"Command: {cmd_display}")
    print()

    try:
        import subprocess
        result = subprocess.run(cmd_parts, cwd=str(TRAINING_DIR.parent))
        print()
        if result.returncode == 0:
            print("=" * 60)
            print(f"  LoRA adapter saved to: {ADAPTER_DIR}")
            print(f"  To use: Aura auto-loads from this path on next boot.")
            print(f"  Backup of previous adapter: {BACKUP_DIR}")
            print("=" * 60)
        else:
            print(f"Training exited with code {result.returncode}")
    except Exception as e:
        print(f"Fine-tune failed: {e}")
        print("You can run it manually:")
        print(f"  {cmd_display}")


if __name__ == "__main__":
    main()
