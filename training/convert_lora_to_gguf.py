#!/usr/bin/env python3
"""Convert MLX LoRA adapter to GGUF format for llama-server.

After fine-tuning with mlx-lm, run this to create a GGUF adapter
that llama-server can load with --lora.

Usage:
    python training/convert_lora_to_gguf.py

Requires:
    pip install mlx-lm
    llama.cpp's convert_lora_to_gguf.py must be available
"""
import os
import subprocess
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
ADAPTER_DIR = TRAINING_DIR / "adapters" / "aura-personality"
MODEL_DIR = TRAINING_DIR.parent / "models" / "Qwen2.5-32B-Instruct-8bit"
OUTPUT_GGUF = ADAPTER_DIR / "aura-personality-lora.gguf"


def fuse_and_export():
    """Fuse LoRA into base model, then export as GGUF."""
    print("=" * 60)
    print("  AURA LoRA → GGUF CONVERSION")
    print("=" * 60)

    if not (ADAPTER_DIR / "adapters.safetensors").exists():
        print(f"ERROR: No adapter found at {ADAPTER_DIR}/adapters.safetensors")
        print("Run the fine-tune first: python training/finetune_lora.py")
        sys.exit(1)

    # Method 1: Try mlx_lm fuse to create a merged model, then convert
    fused_dir = TRAINING_DIR / "fused-model" / "Aura-32B-v2"
    print(f"\nStep 1: Fusing LoRA adapter into base model...")
    print(f"  Base model: {MODEL_DIR}")
    print(f"  Adapter: {ADAPTER_DIR}")
    print(f"  Output: {fused_dir}")

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "mlx_lm", "fuse",
                "--model", str(MODEL_DIR),
                "--adapter-path", str(ADAPTER_DIR),
                "--save-path", str(fused_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"  Fused model saved to {fused_dir}")
        else:
            print(f"  Fuse failed: {result.stderr[:500]}")
            print("  Trying alternative method...")
    except Exception as e:
        print(f"  Fuse failed: {e}")

    # Method 2: Try direct LoRA GGUF conversion if llama.cpp tools available
    llama_convert = None
    for candidate in [
        "/opt/homebrew/bin/convert_lora_to_gguf",
        os.path.expanduser("~/llama.cpp/convert_lora_to_gguf.py"),
        "convert_lora_to_gguf",
    ]:
        if os.path.exists(candidate) or os.system(f"which {candidate} > /dev/null 2>&1") == 0:
            llama_convert = candidate
            break

    if llama_convert:
        print(f"\nStep 2: Converting LoRA to GGUF via {llama_convert}...")
        try:
            result = subprocess.run(
                [sys.executable, llama_convert, "--base", str(MODEL_DIR), str(ADAPTER_DIR), "--outfile", str(OUTPUT_GGUF)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                print(f"  GGUF LoRA adapter saved to {OUTPUT_GGUF}")
                return True
            else:
                print(f"  Conversion failed: {result.stderr[:500]}")
        except Exception as e:
            print(f"  Conversion failed: {e}")

    print("\n" + "=" * 60)
    print("GGUF conversion not available. The MLX adapter is still usable")
    print("by setting AURA_LOCAL_BACKEND=mlx in your environment.")
    print(f"Adapter location: {ADAPTER_DIR}")
    print("=" * 60)
    return False


if __name__ == "__main__":
    fuse_and_export()
