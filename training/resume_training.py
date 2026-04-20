"""Resume the Aura personality LoRA from the latest saved checkpoint.

This script inspects the training log to avoid redoing already-saved work
after an interrupted resume attempt. It keeps the reduced sequence length and
other low-memory settings that were added to stay under the Metal cap.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import re

BASE_MODEL = (
    "/Users/bryan/.cache/huggingface/hub/"
    "models--mlx-community--Qwen2.5-32B-Instruct-4bit/"
    "snapshots/2938092373e5f97b95538884112085364c2da315"
)
ADAPTER_PATH = Path("/Users/bryan/.aura/live-source/training/adapters/aura-personality")
DATA_DIR = "/Users/bryan/.aura/live-source/training/data"
LOG_PATH = Path("/Users/bryan/.aura/live-source/training/train_log.txt")
TRAINING_CONFIG_PATH = ADAPTER_PATH / "training_config.json"

TOTAL_ITERS_FALLBACK = 9504
SAVE_RE = re.compile(r"Iter (\d+): Saved adapter weights .*?/([0-9]+_adapters\.safetensors)")
RESUME_RE = re.compile(r"--- Resume from ([^,]+), (\d+) iters")


def _load_total_iterations() -> int:
    if not TRAINING_CONFIG_PATH.exists():
        return TOTAL_ITERS_FALLBACK

    try:
        config = json.loads(TRAINING_CONFIG_PATH.read_text())
    except Exception:
        return TOTAL_ITERS_FALLBACK

    total = config.get("total_iterations")
    return int(total) if isinstance(total, int) and total > 0 else TOTAL_ITERS_FALLBACK


def _latest_base_checkpoint(total_iterations: int) -> tuple[Path, int]:
    checkpoints = sorted(
        ADAPTER_PATH.glob("*_adapters.safetensors"),
        key=lambda path: int(path.stem.split("_", 1)[0]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {ADAPTER_PATH}")

    checkpoint = checkpoints[-1]
    completed = int(checkpoint.stem.split("_", 1)[0])
    remaining = total_iterations - completed
    if remaining <= 0:
        raise RuntimeError("Training already appears complete; no iterations remaining.")
    return checkpoint, remaining


def _resume_state_from_log() -> tuple[Path, int] | None:
    if not LOG_PATH.exists():
        return None

    last_resume_file: str | None = None
    remaining_at_resume: int | None = None
    last_saved_file: str | None = None
    last_saved_iter: int | None = None

    for line in LOG_PATH.read_text(errors="ignore").splitlines():
        resume_match = RESUME_RE.search(line)
        if resume_match:
            last_resume_file = resume_match.group(1).strip()
            remaining_at_resume = int(resume_match.group(2))
            last_saved_file = None
            last_saved_iter = None
            continue

        if remaining_at_resume is None:
            continue

        save_match = SAVE_RE.search(line)
        if save_match:
            last_saved_iter = int(save_match.group(1))
            last_saved_file = save_match.group(2)

    if remaining_at_resume is None or last_resume_file is None:
        return None

    if last_saved_file and last_saved_iter is not None:
        checkpoint = ADAPTER_PATH / last_saved_file
        remaining = remaining_at_resume - last_saved_iter
    else:
        checkpoint = ADAPTER_PATH / last_resume_file
        remaining = remaining_at_resume

    if remaining <= 0:
        raise RuntimeError("Training already appears complete; no iterations remaining.")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint missing: {checkpoint}")
    return checkpoint, remaining


def _resolve_resume_state() -> tuple[Path, int]:
    total_iterations = _load_total_iterations()
    log_state = _resume_state_from_log()
    if log_state is not None:
        return log_state
    return _latest_base_checkpoint(total_iterations)


def main() -> int:
    resume_file, remaining_iters = _resolve_resume_state()

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        BASE_MODEL,
        "--train",
        "--data",
        DATA_DIR,
        "--adapter-path",
        str(ADAPTER_PATH),
        "--resume-adapter-file",
        str(resume_file),
        "--iters",
        str(remaining_iters),
        "--num-layers",
        "-1",
        "--batch-size",
        "1",
        "--learning-rate",
        "2e-6",
        "--save-every",
        "250",
        "--val-batches",
        "1",
        "--max-seq-length",
        "2048",
        "--grad-checkpoint",
        "-c",
        str(ADAPTER_PATH / "lora_config.yaml"),
    ]

    print(f"Resuming from {resume_file.name}, {remaining_iters} iters remaining.")
    with LOG_PATH.open("a") as log:
        log.write(
            f"\n--- Resume from {resume_file.name}, {remaining_iters} iters, seq=2048 ---\n"
        )
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd="/Users/bryan/.aura/live-source",
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        process.wait()
        return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
