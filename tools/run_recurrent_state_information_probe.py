#!/usr/bin/env python3
"""Localize exact program state across Aura's current live recurrent graph."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
)
from core.learning.recurrence_curriculum import (  # noqa: E402
    RecurrenceTrainingTask,
    disjoint_task_split,
)
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    prepare_recurrent_state_trail,
    validate_recurrent_state_trail_receipt,
)
from core.learning.recurrent_behavioral_probe import tokenize_task  # noqa: E402
from core.learning.recurrent_state_probe import (  # noqa: E402
    StateProbeObservation,
    evaluate_recurrent_state_information,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.recurrent_state_information_canary.v1"
PRIVATE_ARTIFACT_SCHEMA: Final = "aura.recurrent_state_information_private.v1"
SOURCE_PATHS: Final = (
    "core/brain/llm/latent_cortex/execution_spec.py",
    "core/learning/recurrence_curriculum.py",
    "core/learning/recurrence_native_objective_v2.py",
    "core/learning/recurrent_state_probe.py",
    "tools/run_recurrent_state_information_probe.py",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_state() -> tuple[str, dict[str, Any]]:
    if subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "diff", "--quiet"], cwd=REPO_ROOT, check=False
    ).returncode != 0 or subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "diff", "--cached", "--quiet"], cwd=REPO_ROOT, check=False
    ).returncode != 0:
        raise RuntimeError("state probe requires a clean source checkout")
    commit = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    origin = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "origin/main"], cwd=REPO_ROOT, text=True
    ).strip()
    if commit != origin:
        raise RuntimeError("state probe source must equal published origin/main")
    bindings: dict[str, Any] = {}
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        payload = path.read_bytes()
        bindings[relative] = {"sha256": _sha256(payload), "size_bytes": len(payload)}
    return commit, bindings


def _write_receipt(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(_canonical_bytes(dict(body)))
    receipt = {**normalized, "receipt_sha256": _sha256(_canonical_bytes(normalized))}
    payload = _canonical_bytes(receipt)
    atomic_write_bytes(path, payload, mode=0o600)
    if path.read_bytes() != payload:
        raise RuntimeError("state probe receipt changed during atomic write")
    return receipt


def _task_commitment(task: RecurrenceTrainingTask) -> dict[str, Any]:
    trace = task.transition_trace
    if trace is None:
        raise ValueError("state probe task has no structured transition trace")
    return {
        "task_id": task.task_id,
        "family": task.family,
        "depth": task.depth,
        "prompt_sha256": _sha256(task.prompt.encode("utf-8")),
        "transition_trace": trace.public_commitment(),
    }


def _capture_split(
    model: Any,
    tokenizer: Any,
    tasks: Sequence[RecurrenceTrainingTask],
    *,
    split: str,
) -> tuple[list[StateProbeObservation], list[dict[str, Any]]]:
    import mlx.core as mx

    observations: list[StateProbeObservation] = []
    trail_receipts: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        trace = task.transition_trace
        if trace is None:
            raise RuntimeError("structured transition trace disappeared")
        prompt_tokens, _answer_tokens = tokenize_task(tokenizer, task.prompt, task.answer)
        spec = RLCExecutionSpec(
            n_slots=4,
            branch_roles=("constructive_solution",),
            recurrent_steps=task.depth,
            exchange_interval=max(1, task.depth),
        )
        trail = prepare_recurrent_state_trail(model, prompt_tokens, spec=spec)
        receipt = validate_recurrent_state_trail_receipt(trail.receipt())
        if len(trail.states_by_depth) != len(trace.states):
            raise RuntimeError("neural and exact state trails disagree on depth")
        for step, (neural_branches, exact_state) in enumerate(
            zip(trail.states_by_depth, trace.states, strict=True)
        ):
            if len(neural_branches) != 1:
                raise RuntimeError("state probe expected one execution branch")
            state = neural_branches[0].astype(mx.float32)
            mx.eval(state)
            observations.append(
                StateProbeObservation(
                    task_id=task.task_id,
                    family=task.family,
                    program_depth=task.depth,
                    recurrence_step=step,
                    field_names=trace.field_names,
                    labels=exact_state,
                    features=np.asarray(state),
                )
            )
        trail_receipts.append(
            {
                "split": split,
                "task_index": task_index,
                "task_id": task.task_id,
                "receipt_sha256": receipt["receipt_sha256"],
                "execution_spec_sha256": receipt["execution_spec_sha256"],
                "depth_branch_sha256s": receipt["depth_branch_sha256s"],
            }
        )
        mx.clear_cache()
        print(
            f"[state-probe] captured split={split} task={task_index + 1}/{len(tasks)} "
            f"family={task.family} depth={task.depth}",
            flush=True,
        )
    return observations, trail_receipts


def _write_private_artifact(
    path: Path,
    *,
    training: Sequence[StateProbeObservation],
    validation: Sequence[StateProbeObservation],
) -> dict[str, Any]:
    rows = [*training, *validation]
    features = np.stack(
        [np.asarray(row.features, dtype=np.float32).reshape(-1) for row in rows]
    )
    labels = np.asarray([row.labels for row in rows], dtype=np.int64)
    metadata = {
        "schema": PRIVATE_ARTIFACT_SCHEMA,
        "training_rows": len(training),
        "validation_rows": len(validation),
        "task_ids": [row.task_id for row in rows],
        "families": [row.family for row in rows],
        "program_depths": [row.program_depth for row in rows],
        "recurrence_steps": [row.recurrence_step for row in rows],
        "field_names": [list(row.field_names) for row in rows],
    }
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        features=features,
        labels=labels,
        metadata=np.frombuffer(_canonical_bytes(metadata), dtype=np.uint8),
    )
    payload = buffer.getvalue()
    atomic_write_bytes(path, payload, mode=0o600)
    if path.read_bytes() != payload or path.stat().st_mode & 0o777 != 0o600:
        raise RuntimeError("private state probe artifact custody failed")
    return {
        "schema": PRIVATE_ARTIFACT_SCHEMA,
        "path": str(path),
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
        "feature_shape": list(features.shape),
        "label_shape": list(labels.shape),
        "mode": "0600",
    }


def run_probe(
    *,
    model_path: Path,
    out_dir: Path,
    seed: int,
    memory_fraction: float,
    depths: Sequence[int],
    training_per_cell: int,
    validation_per_cell: int,
    regularization: float,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load

    source_commit, source_bindings = _source_state()
    out_dir.mkdir(parents=True, exist_ok=False)
    os.chmod(out_dir, 0o700)
    families = ("boolean", "modular")
    training_tasks, validation_tasks = disjoint_task_split(
        families=families,
        depths=tuple(depths),
        train_per_cell=training_per_cell,
        holdout_per_cell=validation_per_cell,
        seed=seed,
    )
    model_before = full_weight_checkpoint_identity(model_path)
    with (
        standalone_model_lane(
            owner_id=f"recurrent-state-probe:{out_dir.name}",
            model_path=str(model_path),
            purpose="evaluation",
            preemptible=False,
            metadata={"tool": Path(__file__).name, "source_commit": source_commit},
        ),
        mlx_memory_envelope(fraction=memory_fraction, restore_limits_on_exit=True),
    ):
        print("[state-probe] model_load", flush=True)
        model, tokenizer = load(str(model_path))
        training, training_trails = _capture_split(
            model, tokenizer, training_tasks, split="training"
        )
        validation, validation_trails = _capture_split(
            model, tokenizer, validation_tasks, split="validation"
        )
        report = evaluate_recurrent_state_information(
            training,
            validation,
            regularization=regularization,
            null_seed=seed ^ 0x5A17E,
        )
        private_artifact = _write_private_artifact(
            out_dir / "private_state_observations.npz",
            training=training,
            validation=validation,
        )
        mx.clear_cache()
    model_after = full_weight_checkpoint_identity(model_path)
    task_ids_disjoint = {task.task_id for task in training_tasks}.isdisjoint(
        {task.task_id for task in validation_tasks}
    )
    gates = {
        "base_checkpoint_immutable": model_before == model_after,
        "source_published": source_commit
        == subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "rev-parse", "origin/main"], cwd=REPO_ROOT, text=True
        ).strip(),
        "task_ids_disjoint": task_ids_disjoint,
        "probe_task_ids_disjoint": report["task_ids_disjoint"] is True,
        "private_artifact_bound": bool(private_artifact["sha256"]),
        "all_state_trails_replayed": len(training_trails) == len(training_tasks)
        and len(validation_trails) == len(validation_tasks),
    }
    body = {
        "schema": CANARY_SCHEMA,
        "source_commit": source_commit,
        "source_bindings": source_bindings,
        "model_path": str(model_path),
        "model_identity": model_before,
        "configuration": {
            "seed": seed,
            "families": list(families),
            "depths": list(depths),
            "training_per_cell": training_per_cell,
            "validation_per_cell": validation_per_cell,
            "regularization": regularization,
        },
        "training_manifest": [_task_commitment(task) for task in training_tasks],
        "validation_manifest": [_task_commitment(task) for task in validation_tasks],
        "state_trails": [*training_trails, *validation_trails],
        "private_artifact": private_artifact,
        "probe": report,
        "gates": gates,
        "admitted": all(gates.values()),
        "claim_boundary": "recurrent_state_localization_only_not_behavioral_or_reasoning_gain",
    }
    return _write_receipt(out_dir / "receipt.json", body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810177)
    parser.add_argument("--memory-fraction", type=float, default=0.30)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--training-per-cell", type=int, default=8)
    parser.add_argument("--validation-per-cell", type=int, default=4)
    parser.add_argument("--regularization", type=float, default=1.0)
    args = parser.parse_args()
    try:
        depths = tuple(int(value.strip()) for value in args.depths.split(",") if value.strip())
        receipt = run_probe(
            model_path=args.model.expanduser().resolve(strict=True),
            out_dir=args.out_dir.expanduser().resolve(strict=False),
            seed=args.seed,
            memory_fraction=args.memory_fraction,
            depths=depths,
            training_per_cell=args.training_per_cell,
            validation_per_cell=args.validation_per_cell,
            regularization=args.regularization,
        )
    except Exception as exc:  # noqa: BLE001 - printed to stderr and returned non-zero
        print(f"state_probe_failed:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0 if receipt["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
