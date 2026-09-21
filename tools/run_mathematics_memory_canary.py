#!/usr/bin/env python3
"""Run the frozen bounded canary for teacher-removed mathematics memory tissue."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.learning.recurrent_work_memory_tissue import (  # noqa: E402
    MathematicsMemoryTissue,
)
from core.learning.recurrent_work_memory_training import (  # noqa: E402
    autonomous_execution_metrics,
    build_mathematics_memory_registry,
    train_mathematics_memory_tissue,
)
from core.runtime.atomic_writer import atomic_write_bytes_if_absent  # noqa: E402

CANARY_SCHEMA: Final = "aura.mathematics_memory_canary.v1"
SOURCE_FILES: Final = (
    "core/learning/recurrent_work_memory.py",
    "core/learning/recurrent_work_memory_tissue.py",
    "core/learning/recurrent_work_memory_training.py",
    "core/learning/frontier_process_supervision.py",
    "tools/run_mathematics_memory_canary.py",
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _source_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain", "--", *SOURCE_FILES),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    hashes = {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in SOURCE_FILES
    }
    body = {
        "commit": commit,
        "measured_source_clean": not bool(status),
        "source_sha256s": hashes,
    }
    return {**body, "identity_sha256": _canonical_sha256(body)}


def run_canary() -> dict[str, Any]:
    started = time.time()
    training, _training_tasks = build_mathematics_memory_registry(
        seeds=range(0, 40),
        difficulties=(1, 2, 3),
    )
    _heldout, heldout_tasks = build_mathematics_memory_registry(
        seeds=range(1_000, 1_100),
        difficulties=(1, 2, 3),
    )
    tissue, training_receipt = train_mathematics_memory_tissue(
        training,
        steps=400,
        learning_rate=0.01,
        hidden_size=32,
        seed=2026081507,
    )
    matched_control = MathematicsMemoryTissue(
        hidden_size=tissue.hidden_size,
        seed=tissue.seed,
    )
    arms = {
        "treatment": autonomous_execution_metrics(tissue, heldout_tasks),
        "matched_initialization_control": autonomous_execution_metrics(
            matched_control,
            heldout_tasks,
        ),
        "no_write": autonomous_execution_metrics(
            tissue,
            heldout_tasks,
            write_mode="never",
        ),
        "always_write": autonomous_execution_metrics(
            tissue,
            heldout_tasks,
            write_mode="always",
        ),
        "no_read": autonomous_execution_metrics(
            tissue,
            heldout_tasks,
            read_mode="never",
        ),
        "rotated_routing": autonomous_execution_metrics(
            tissue,
            heldout_tasks,
            routing_mode="rotated",
        ),
        "reset_memory": autonomous_execution_metrics(
            tissue,
            heldout_tasks,
            memory_mode="reset_each_step",
        ),
    }
    treatment = arms["treatment"]["exact_accuracy"]
    controls = {
        name: row["exact_accuracy"]
        for name, row in arms.items()
        if name != "treatment"
    }
    admitted = (
        treatment == 1.0
        and arms["matched_initialization_control"]["exact_accuracy"] == 0.0
        and all(value < treatment for value in controls.values())
    )
    body = {
        "schema": CANARY_SCHEMA,
        "source_identity": _source_identity(),
        "training": training_receipt,
        "train_task_count": len(training.task_ids),
        "heldout_task_count": len(heldout_tasks),
        "heldout_seed_range": [1_000, 1_100],
        "heldout_difficulties": [1, 2, 3],
        "arms": arms,
        "admitted": admitted,
        "claim_boundary": (
            "bounded mathematics memory predicate acquisition and autonomous "
            "execution; not free-decoded, resident-32B, broad reasoning, or WOW"
        ),
        "elapsed_seconds": round(time.time() - started, 6),
    }
    return {**body, "receipt_sha256": _canonical_sha256(body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = run_canary()
    payload = json.dumps(receipt, indent=2, sort_keys=True).encode("ascii") + b"\n"
    destination = args.out.expanduser().resolve()
    if not atomic_write_bytes_if_absent(destination, payload, mode=0o400):
        if destination.read_bytes() != payload:
            raise RuntimeError("mathematics memory canary artifact differs")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
