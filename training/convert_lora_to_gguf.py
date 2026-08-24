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
import shutil
import sys
import time
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
REPO_DIR = TRAINING_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from core.runtime.model_lane_control import (  # noqa: E402
    LaneClaim,
    estimate_model_job_footprint_gb,
)
from core.runtime.model_runtime_assignment import (  # noqa: E402
    issue_unqualified_model_runtime_assignment,
)
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

ADAPTER_DIR = TRAINING_DIR / "adapters" / "aura-personality"
MODEL_DIR = TRAINING_DIR.parent / "models" / "Qwen2.5-32B-Instruct-8bit"
OUTPUT_GGUF = ADAPTER_DIR / "aura-personality-lora.gguf"


def _conversion_claim(*, source: str) -> LaneClaim:
    timeout = 300.0
    return LaneClaim(
        owner_id=f"training:gguf-conversion:{os.getpid()}:{time.time_ns()}",
        model_path=str(MODEL_DIR),
        request_gb=estimate_model_job_footprint_gb(
            str(MODEL_DIR),
            purpose="fuse",
        ),
        purpose="fuse",
        priority=80,
        preemptible=True,
        reservation_ttl_s=timeout + 30.0,
        owner_lease_ttl_s=timeout + 30.0,
        runtime_assignment=issue_unqualified_model_runtime_assignment(
            model_path=str(MODEL_DIR),
            purpose="fuse",
            authority_source=source,
        ),
        metadata={"source": source},
    )


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
    print("\nStep 1: Fusing LoRA adapter into base model...")
    print(f"  Base model: {MODEL_DIR}")
    print(f"  Adapter: {ADAPTER_DIR}")
    print(f"  Output: {fused_dir}")

    try:
        result = get_subprocess_gateway().run_model_blocking(
            [
                sys.executable, "-m", "mlx_lm", "fuse",
                "--model", str(MODEL_DIR),
                "--adapter-path", str(ADAPTER_DIR),
                "--save-path", str(fused_dir),
            ],
            cwd=REPO_DIR,
            timeout=300,
            offline_tooling=True,
            source="training_tooling:convert_lora_fuse",
        )
        if result.returncode == 0:
            print(f"  Fused model saved to {fused_dir}")
        else:
            print(f"  Fuse failed: {result.stderr[:500]}")
            print("  Trying alternative method...")
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"  Fuse failed: {type(exc).__name__}: {exc}")

    # Method 2: Try direct LoRA GGUF conversion if llama.cpp tools available
    llama_convert = None
    for candidate in [
        "/opt/homebrew/bin/convert_lora_to_gguf",
        os.path.expanduser("~/llama.cpp/convert_lora_to_gguf.py"),
        "convert_lora_to_gguf",
    ]:
        resolved = candidate if os.path.exists(candidate) else shutil.which(candidate)
        if resolved:
            llama_convert = resolved
            break

    if llama_convert:
        print(f"\nStep 2: Converting LoRA to GGUF via {llama_convert}...")
        try:
            convert_cmd = (
                [sys.executable, llama_convert]
                if str(llama_convert).endswith(".py")
                else [llama_convert]
            )
            source = "training_tooling:convert_lora_to_gguf"
            result = get_subprocess_gateway().run_model_blocking(
                [
                    *convert_cmd,
                    "--base",
                    str(MODEL_DIR),
                    str(ADAPTER_DIR),
                    "--outfile",
                    str(OUTPUT_GGUF),
                ],
                cwd=REPO_DIR,
                timeout=300,
                offline_tooling=True,
                source=source,
                model_lane_claim=_conversion_claim(source=source),
            )
            if result.returncode == 0:
                print(f"  GGUF LoRA adapter saved to {OUTPUT_GGUF}")
                return True
            else:
                print(f"  Conversion failed: {result.stderr[:500]}")
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            print(f"  Conversion failed: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 60)
    print("GGUF conversion not available. The MLX adapter is still usable")
    print("by setting AURA_LOCAL_BACKEND=mlx in your environment.")
    print(f"Adapter location: {ADAPTER_DIR}")
    print("=" * 60)
    return False


if __name__ == "__main__":
    fuse_and_export()
