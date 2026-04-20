"""Resume the Aura personality LoRA from the most recent checkpoint.

Keeps memory low so the Metal backend doesn't OOM on a 64GB machine while
other processes are running.
"""
import os
import subprocess
import sys
from pathlib import Path

BASE_MODEL = "/Users/bryan/.cache/huggingface/hub/models--mlx-community--Qwen2.5-32B-Instruct-4bit/snapshots/2938092373e5f97b95538884112085364c2da315"
ADAPTER_PATH = Path("/Users/bryan/.aura/live-source/training/adapters/aura-personality")
DATA_DIR = "/Users/bryan/.aura/live-source/training/data"

# The prior run reached iter 1000 out of a 3004-iter resume that started from
# checkpoint 0006500. Pick up from that saved state instead of re-doing 1000
# iters. Effective total progress: 6500 + 1000 = 7500 of 9504.
RESUME_FILE = ADAPTER_PATH / "0001000_adapters.safetensors"
REMAINING_ITERS = 2004

if not RESUME_FILE.exists():
    print(f"Resume file missing: {RESUME_FILE}", file=sys.stderr)
    sys.exit(1)

cmd = [
    "/opt/homebrew/bin/python3.14", "-m", "mlx_lm", "lora",
    "--model", BASE_MODEL,
    "--train",
    "--data", DATA_DIR,
    "--adapter-path", str(ADAPTER_PATH),
    "--resume-adapter-file", str(RESUME_FILE),
    "--iters", str(REMAINING_ITERS),
    "--num-layers", "-1",
    "--batch-size", "1",
    "--learning-rate", "2e-6",
    "--save-every", "250",
    "--val-batches", "1",
    "--max-seq-length", "2048",  # halved from 4096 to stay under Metal cap
    "--grad-checkpoint",
    "-c", str(ADAPTER_PATH / "lora_config.yaml"),
]

log_path = Path("/Users/bryan/.aura/live-source/training/train_log.txt")

print(f"Resuming from {RESUME_FILE.name}, {REMAINING_ITERS} iters remaining.")
with log_path.open("a") as log:
    log.write(f"\n--- Resume from {RESUME_FILE.name}, {REMAINING_ITERS} iters, seq=2048 ---\n")
    log.flush()
    process = subprocess.Popen(
        cmd,
        cwd="/Users/bryan/.aura/live-source",
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    process.wait()
    sys.exit(process.returncode)
