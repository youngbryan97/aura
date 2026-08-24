#!/usr/bin/env python3
"""LoRA fine-tune Aura's promoted Cortex artifact.

Upgraded from v2:
  - Rank 32 (up from 8) — needed for architecture knowledge density
  - All transformer layers — personality is not tied to one model generation
  - 4096 max sequence length (up from 2048) — longer explanations
  - Gradient checkpointing enabled for bounded unified-memory use
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
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

# Paths
TRAINING_DIR = Path(__file__).parent
REPO_DIR = TRAINING_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402
from training.model_basis import (  # noqa: E402
    TrainingModelBasisError,
    load_recorded_training_model_basis,
    resolve_training_model_basis,
)

DATA_DIR = TRAINING_DIR / "data"
ADAPTER_DIR = TRAINING_DIR / "adapters" / "aura-personality"
BACKUP_DIR = TRAINING_DIR / "adapters" / "aura-personality-v4-backup"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "valid.jsonl"
TRAINING_COMMAND_TIMEOUT_S = float(os.environ.get("AURA_TRAINING_COMMAND_TIMEOUT_S", "86400"))

# ── Hyperparameters — Project Zenith ──────────────────────────────────────
LORA_RANK = 32          # Up from 8 — architecture knowledge needs density
LORA_LAYERS = -1        # All layers, independent of the current Cortex depth
EPOCHS = 1.5            # Balanced: 1.5 epochs (90k steps) for depth without overfitting
BATCH_SIZE = 1          # Keep small for M-series memory
LEARNING_RATE = 5e-6    # Lower — larger dataset + higher rank = more careful
WARMUP_STEPS = 200      # Up from 50 — larger dataset needs longer warmup
MAX_SEQ_LENGTH = 4096   # Up from 2048 — architecture explanations are longer
GRAD_CHECKPOINT = True  # Essential for rank-32 on a large resident Cortex
SAVE_EVERY = 500        # Checkpoint frequency


def find_base_model() -> str:
    """Return the exact local artifact selected by the model registry."""
    return str(resolve_training_model_basis().path)


def _assert_expected_basis(descriptor_sha256: str) -> None:
    expected = str(
        os.environ.get("AURA_LORA_EXPECTED_BASE_DESCRIPTOR_SHA256", "")
    ).strip()
    if expected and descriptor_sha256 != expected:
        raise TrainingModelBasisError("training_parent_model_basis_mismatch")


def backup_existing_adapter():
    """Backup existing adapter before overwriting."""
    if ADAPTER_DIR.exists() and any(ADAPTER_DIR.glob("*.safetensors")):
        print(f"  Backing up existing adapter to {BACKUP_DIR}...")
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(ADAPTER_DIR, BACKUP_DIR)
        print("  Backup complete.")


def _latest_checkpoint() -> Path | None:
    """Find the highest-numbered ``NNNNNNN_adapters.safetensors`` checkpoint."""
    if not ADAPTER_DIR.exists():
        return None
    candidates = sorted(ADAPTER_DIR.glob("[0-9]*_adapters.safetensors"))
    return candidates[-1] if candidates else None


def main():
    if importlib.util.find_spec("mlx_lm.lora") is None:
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

    config_path = ADAPTER_DIR / "training_config.json"
    if resume:
        model_basis = load_recorded_training_model_basis(
            config_path,
            model_override=str(os.environ.get("AURA_LORA_BASE_MODEL", "")).strip() or None,
            verify_full_hash=True,
        )
    else:
        model_basis = resolve_training_model_basis()
    _assert_expected_basis(model_basis.descriptor_sha256)
    model_path = str(model_basis.path)

    # Calculate total iterations
    total_iters = int(EPOCHS * n_train // BATCH_SIZE)

    print("=" * 60)
    print("  PROJECT ZENITH — AURA PERSONALITY LoRA FINE-TUNE")
    print("=" * 60)
    print(f"  Base model:        {model_path}")
    print(f"  Training data:     {n_train} examples")
    print(f"  Validation:        {n_val} examples")
    print(f"  LoRA rank:         {LORA_RANK}")
    print("  LoRA layers:       all")
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
    if not resume:
        training_config = {
            "schema": "aura.personality_lora.training.v2",
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
            "training_basis": model_basis.to_record(),
            "train_data": str(TRAIN_FILE),
            "val_data": str(val_file),
            "adapter_path": str(ADAPTER_DIR),
            "total_train_examples": n_train,
            "total_val_examples": n_val,
            "total_iterations": total_iters,
        }
        atomic_write_text(
            config_path,
            json.dumps(training_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Config saved to {config_path}")
    else:
        recorded = json.loads(config_path.read_text(encoding="utf-8"))
        recorded_total = recorded.get("total_iterations")
        if not isinstance(recorded_total, int) or recorded_total <= 0:
            raise RuntimeError("resume_training_total_iterations_missing")
        total_iters = recorded_total
        print(f"Using immutable resume config from {config_path}")
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
    try:
        import yaml

        with open(lora_config_path, "w") as f:
            yaml.dump(lora_config, f)
    except ImportError:
        # No PyYAML — write as JSON config instead
        lora_config_path = ADAPTER_DIR / "lora_config.json"
        with open(lora_config_path, "w") as f:
            json.dump(lora_config, f, indent=2)

    # ── Build MLX LoRA command ───────────────────────────────────────────
    cmd_parts = [
        sys.executable, "-m", "mlx_lm", "lora",
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
            raise RuntimeError("resume_training_checkpoint_missing")
        else:
            completed = int(latest.stem.split("_", 1)[0])
            remaining = total_iters - completed
            if remaining <= 0:
                raise RuntimeError("resume_training_already_complete")
            print(f"  --resume: continuing from {latest.name}")
            cmd_parts[cmd_parts.index("--iters") + 1] = str(remaining)
            cmd_parts.extend(["--resume-adapter-file", str(latest)])

    cmd_display = " ".join(cmd_parts)
    print(f"Command: {cmd_display}")
    print()

    try:
        result = get_subprocess_gateway().run_model_blocking(
            cmd_parts,
            cwd=REPO_DIR,
            timeout=TRAINING_COMMAND_TIMEOUT_S,
            capture_output=False,
            offline_tooling=True,
            source="training_tooling:finetune_lora",
        )
        print()
        if result.returncode == 0:
            print("=" * 60)
            print(f"  LoRA adapter saved to: {ADAPTER_DIR}")
            print("  To use: Aura auto-loads from this path on next boot.")
            print(f"  Backup of previous adapter: {BACKUP_DIR}")
            print("=" * 60)
        else:
            print(f"Training exited with code {result.returncode}")
            sys.exit(result.returncode)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Fine-tune failed: {type(exc).__name__}: {exc}")
        print("You can run it manually:")
        print(f"  {cmd_display}")
        sys.exit(1)


if __name__ == "__main__":
    main()
