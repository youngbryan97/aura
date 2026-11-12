"""Local DPO/ORPO training bridge for verifier-derived preference pairs.

The verifiable preference harness records only checked win/loss pairs. This
module turns those pairs into trainer-ready splits and invokes the local
``mlx-lm-lora`` trainer. It deliberately refuses to train when there are not
enough real pairs, so Aura cannot mistake an installed package for learning.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.learning.verifiable_preference_harness import VerifiablePreferenceHarness
from core.runtime.atomic_writer import atomic_write_text
from core.tasks.managed_command import ManagedCommandResult, run_project_command

PAIR_TRAIN_MODES = {"dpo", "orpo", "online_dpo"}
REQUIRED_TRAINER_MODULES = (
    "mlx_lm_lora",
    "mlx_lm_lora.train",
    "mlx_lm_lora.trainer.dpo_trainer",
    "mlx_lm_lora.trainer.orpo_trainer",
    "mlx_lm_lora.trainer.online_dpo_trainer",
)
OPTIONAL_TRAINER_MODULES = (
    "mlx_lm_lora.trainer.grpo_trainer",
)


@dataclass(frozen=True)
class PreferenceTrainingRequest:
    model_path: Path
    store_path: Path
    adapter_path: Path
    data_dir: Path
    train_mode: str = "dpo"
    train_type: str = "lora"
    min_rows: int = 8
    limit: int = 5000
    num_layers: int = 16
    iters: int = 80
    batch_size: int = 1
    learning_rate: float = 5e-6
    save_every: int = 80
    val_batches: int = 1
    max_seq_length: int = 2048
    timeout_seconds: int = 3600
    grad_checkpoint: bool = True
    efficient_long_context: bool = True


def check_preference_trainer_available() -> dict[str, Any]:
    """Return an auditable availability report for the local preference trainer."""
    missing_required: list[str] = []
    missing_optional: list[str] = []
    modules: dict[str, str] = {}
    for name in REQUIRED_TRAINER_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None:
            missing_required.append(name)
        else:
            modules[name] = str(spec.origin or "")
    for name in OPTIONAL_TRAINER_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None:
            missing_optional.append(name)
        else:
            modules[name] = str(spec.origin or "")

    version = ""
    try:
        version = importlib.metadata.version("mlx-lm-lora")
    except importlib.metadata.PackageNotFoundError:
        missing_required.append("distribution:mlx-lm-lora")

    return {
        "ok": not missing_required,
        "package": "mlx-lm-lora",
        "version": version,
        "required_modules": list(REQUIRED_TRAINER_MODULES),
        "optional_modules": list(OPTIONAL_TRAINER_MODULES),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "modules": modules,
        "pair_train_modes": sorted(PAIR_TRAIN_MODES),
        "note": "GRPO support is detected separately; verifier-pair rows feed DPO/ORPO/online_DPO.",
    }


def export_preference_splits(rows: list[dict[str, str]], data_dir: Path) -> dict[str, int]:
    """Write ``train/valid/test.jsonl`` preference splits without duplicating rows."""
    if not rows:
        raise ValueError("cannot export empty preference dataset")

    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        key = json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if not cleaned:
        raise ValueError("no valid preference rows after cleaning")

    valid_count = max(1, int(len(cleaned) * 0.1)) if len(cleaned) >= 10 else 0
    test_count = max(1, int(len(cleaned) * 0.05)) if len(cleaned) >= 20 else 0
    train_count = max(1, len(cleaned) - valid_count - test_count)
    splits = {
        "train": cleaned[:train_count],
        "valid": cleaned[train_count:train_count + valid_count],
        "test": cleaned[train_count + valid_count:train_count + valid_count + test_count],
    }
    counts: dict[str, int] = {}
    for split, split_rows in splits.items():
        if not split_rows and split != "train":
            continue
        atomic_write_text(
            data_dir / f"{split}.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows),
            encoding="utf-8",
        )
        counts[split] = len(split_rows)
    return counts


def build_preference_training_command(request: PreferenceTrainingRequest) -> tuple[str, ...]:
    mode = request.train_mode.strip().lower()
    if mode not in PAIR_TRAIN_MODES:
        raise ValueError(f"unsupported verifier-pair training mode: {request.train_mode}")
    train_type = request.train_type.strip().lower()
    if train_type not in {"lora", "dora", "full"}:
        raise ValueError(f"unsupported train_type: {request.train_type}")

    cmd: list[str] = [
        sys.executable,
        "-m",
        "mlx_lm_lora.train",
        "--model",
        str(request.model_path),
        "--train",
        "--data",
        str(request.data_dir),
        "--train-type",
        train_type,
        "--train-mode",
        mode,
        "--adapter-path",
        str(request.adapter_path),
        "--num-layers",
        str(max(1, int(request.num_layers))),
        "--iters",
        str(max(1, int(request.iters))),
        "--batch-size",
        str(max(1, int(request.batch_size))),
        "--learning-rate",
        str(float(request.learning_rate)),
        "--save-every",
        str(max(1, int(request.save_every))),
        "--val-batches",
        str(max(0, int(request.val_batches))),
        "--max-seq-length",
        str(max(128, int(request.max_seq_length))),
    ]
    if request.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if request.efficient_long_context:
        cmd.append("--efficient-long-context")
    return tuple(cmd)


def run_verifiable_preference_training(
    request: PreferenceTrainingRequest,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Export verified preference pairs and optionally launch local DPO/ORPO training."""
    availability = check_preference_trainer_available()
    harness = VerifiablePreferenceHarness(store_path=request.store_path)
    rows = harness.export_dpo_rows(limit=max(1, int(request.limit)))
    stats = {**harness.stats(), "exportable_rows": len(rows)}
    if not availability["ok"]:
        return {"ok": False, "status": "blocked", "reason": "preference_trainer_unavailable", "availability": availability, "stats": stats}
    if len(rows) < max(1, int(request.min_rows)):
        return {
            "ok": False,
            "status": "blocked",
            "reason": "insufficient_verifiable_preference_rows",
            "rows": len(rows),
            "min_rows": int(request.min_rows),
            "stats": stats,
            "availability": availability,
        }
    split_counts = export_preference_splits(rows, request.data_dir)
    command = build_preference_training_command(request)
    payload: dict[str, Any] = {
        "ok": True,
        "status": "dry_run" if dry_run else "started",
        "train_mode": request.train_mode,
        "train_type": request.train_type,
        "rows": len(rows),
        "split_counts": split_counts,
        "data_dir": str(request.data_dir),
        "adapter_path": str(request.adapter_path),
        "command": list(command),
        "availability": availability,
        "stats": stats,
    }
    if dry_run:
        return payload

    started = time.time()
    result: ManagedCommandResult = run_project_command(
        command,
        timeout_s=float(max(60, int(request.timeout_seconds))),
    )
    payload.update(
        {
            "ok": result.ok,
            "status": "success" if result.ok else "failed",
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "elapsed_s": result.elapsed_s,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
            "started_at": started,
        }
    )
    return payload
