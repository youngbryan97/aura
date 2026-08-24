"""Resume the Aura personality LoRA from the latest saved checkpoint.

This script inspects the training log to avoid redoing already-saved work
after an interrupted resume attempt. It keeps the reduced sequence length and
other low-memory settings that were added to stay under the Metal cap.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from subprocess import STDOUT

PROJECT_ROOT = Path(os.environ.get("AURA_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402
from training.model_basis import load_recorded_training_model_basis  # noqa: E402

ADAPTER_PATH = Path(
    os.environ.get(
        "AURA_TRAINING_ADAPTER_PATH",
        PROJECT_ROOT / "training" / "adapters" / "aura-personality",
    )
).expanduser()
DATA_DIR = os.environ.get("AURA_TRAINING_DATA_DIR", str(PROJECT_ROOT / "training" / "data"))
LOG_PATH = Path(
    os.environ.get(
        "AURA_TRAINING_LOG_PATH",
        PROJECT_ROOT / "training" / "logs" / "train_and_fuse.log",
    )
).expanduser()
TRAINING_CONFIG_PATH = ADAPTER_PATH / "training_config.json"

TOTAL_ITERS_FALLBACK = 90153
SAVE_RE = re.compile(r"Iter (\d+): Saved adapter weights .*?/([0-9]+_adapters\.safetensors)")
RESUME_RE = re.compile(
    r"--- Resume(?: Zenith)? from ([^,]+), "
    r"(?:(\d+) iters remaining|(\d+) iters|targeting (\d+) total iters)"
)
_TRAINING_CONFIG_RECOVERABLE_ERRORS = (
    OSError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    TypeError,
    ValueError,
)


def _load_total_iterations() -> int:
    if not TRAINING_CONFIG_PATH.exists():
        return TOTAL_ITERS_FALLBACK

    try:
        config = json.loads(TRAINING_CONFIG_PATH.read_text())
    except _TRAINING_CONFIG_RECOVERABLE_ERRORS as exc:
        print(
            f"warning: failed to read training config {TRAINING_CONFIG_PATH}: "
            f"{type(exc).__name__}: {exc}; using fallback total iterations",
            file=sys.stderr,
        )
        return TOTAL_ITERS_FALLBACK

    total = config.get("total_iterations")
    return int(total) if isinstance(total, int) and total > 0 else TOTAL_ITERS_FALLBACK


def _load_base_model() -> Path:
    override = str(os.environ.get("AURA_TRAINING_BASE_MODEL", "")).strip() or None
    return load_recorded_training_model_basis(
        TRAINING_CONFIG_PATH,
        model_override=override,
        verify_full_hash=True,
    ).path


def _latest_base_checkpoint(total_iterations: int) -> tuple[Path, int]:
    checkpoints = sorted(
        ADAPTER_PATH.glob("*_adapters.safetensors"),
        key=lambda path: int(path.stem.split("_", 1)[0]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {ADAPTER_PATH}")

    # Identify the highest checkpoint number
    checkpoint = checkpoints[-1]
    completed = int(checkpoint.stem.split("_", 1)[0])
    
    # Check if we accidentally grabbed a small number (from the 8-bit run)
    # The 4-bit run was already at 66,000.
    if completed < 66000:
        # Scan for the actual 4-bit maximum
        for cp in reversed(checkpoints):
            val = int(cp.stem.split("_", 1)[0])
            if val >= 66000:
                checkpoint = cp
                completed = val
                break

    remaining = total_iterations - completed
    if remaining <= 0:
        raise RuntimeError("Training already appears complete; no iterations remaining.")
    return checkpoint, remaining


def _resume_state_from_log() -> tuple[Path, int] | None:
    if not LOG_PATH.exists():
        return None

    # Collect all saved checkpoints from the log in order
    saves: list[tuple[int, str]] = []
    last_resume_file: str | None = None
    remaining_at_resume: int | None = None

    for line in LOG_PATH.read_text(errors="ignore").splitlines():
        resume_match = RESUME_RE.search(line)
        if resume_match:
            last_resume_file = resume_match.group(1).strip()
            if resume_match.group(4):
                target_total = int(resume_match.group(4))
                try:
                    completed_at_resume = int(Path(last_resume_file).stem.split("_", 1)[0])
                except (TypeError, ValueError, IndexError):
                    completed_at_resume = 0
                remaining_at_resume = max(0, target_total - completed_at_resume)
            else:
                remaining_at_resume = int(resume_match.group(2) or resume_match.group(3))
            saves = [] # Reset saves for this resume session
            continue

        save_match = SAVE_RE.search(line)
        if save_match and remaining_at_resume is not None:
            saves.append((int(save_match.group(1)), save_match.group(2)))

    if remaining_at_resume is None or last_resume_file is None:
        return None

    # Try checkpoints in reverse order (newest first)
    for iter_count, filename in reversed(saves):
        checkpoint = ADAPTER_PATH / filename
        if checkpoint.exists():
            remaining = remaining_at_resume - iter_count
            if remaining > 0:
                return checkpoint, remaining

    # Fallback to the original resume file
    checkpoint = ADAPTER_PATH / last_resume_file
    if checkpoint.exists():
        if remaining_at_resume > 0:
            return checkpoint, remaining_at_resume

    return None


def _resolve_resume_state() -> tuple[Path, int]:
    total_iterations = _load_total_iterations()
    # Prioritize log-based resume for accuracy
    log_state = _resume_state_from_log()
    if log_state is not None:
        return log_state
    # Fallback to filesystem glob
    return _latest_base_checkpoint(total_iterations)


def main() -> int:
    base_model = _load_base_model()
    resume_file, remaining_iters = _resolve_resume_state()

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        str(base_model),
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
        "5e-6",
        "--save-every",
        "500",
        "--steps-per-eval",
        "500",
        "--max-seq-length",
        "4096",
        "--grad-checkpoint",
        "-c",
        str(ADAPTER_PATH / "lora_config.yaml"),
    ]

    print(
        f"Resuming Zenith v3.3 from {resume_file.name}, {remaining_iters} iters remaining "
        f"on {base_model}."
    )
    with LOG_PATH.open("a") as log:
        log.write(
            f"\n--- Resume Zenith from {resume_file.name}, {remaining_iters} iters remaining, "
            f"target_total={remaining_iters + int(resume_file.stem.split('_', 1)[0])}, seq=4096 ---\n"
        )
        log.flush()
        async def _run_resume() -> int:
            process = await get_subprocess_gateway().spawn_async(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=STDOUT,
                offline_tooling=True,
                source="training_tooling:resume_training",
                accelerator_capability="model",
            )
            return int(await process.wait())

        return asyncio.run(_run_resume())


if __name__ == "__main__":
    raise SystemExit(main())
