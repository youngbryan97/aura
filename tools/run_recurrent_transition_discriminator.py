#!/usr/bin/env python3
"""Test whether the existing Qwen recurrent window can learn one exact step."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
)
from core.learning.recurrence_curriculum import (  # noqa: E402
    RecurrenceTrainingTask,
    task_battery,
)
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    prepare_recurrent_transition_input,
    validate_recurrent_transition_input_receipt,
)
from core.learning.recurrent_behavioral_probe import tokenize_task  # noqa: E402
from core.learning.recurrent_sft_execution import (  # noqa: E402
    adapter_tensor_dict,
    adapter_tensor_fingerprint,
    wrap_recurrent_window,
)
from core.learning.recurrent_transition_supervision import (  # noqa: E402
    StateCodebookSpec,
    evaluate_state_supervised_transition,
    state_supervised_transition_value_and_grad,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.recurrent_transition_discriminator.v1"
SOURCE_PATHS: Final = (
    "core/brain/llm/latent_cortex/execution_spec.py",
    "core/learning/recurrence_curriculum.py",
    "core/learning/recurrence_native_objective_v2.py",
    "core/learning/recurrent_sft_execution.py",
    "core/learning/recurrent_transition_supervision.py",
    "tools/run_recurrent_transition_discriminator.py",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_state() -> tuple[str, dict[str, Any]]:
    if subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "diff", "--quiet"], cwd=REPO_ROOT, check=False
    ).returncode != 0 or subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "diff", "--cached", "--quiet"], cwd=REPO_ROOT, check=False
    ).returncode != 0:
        raise RuntimeError("transition discriminator requires a clean source checkout")
    commit = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    origin = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "origin/main"], cwd=REPO_ROOT, text=True
    ).strip()
    if commit != origin:
        raise RuntimeError("transition discriminator source must equal published origin/main")
    bindings: dict[str, Any] = {}
    for relative in SOURCE_PATHS:
        payload = (REPO_ROOT / relative).read_bytes()
        bindings[relative] = {"sha256": _sha256(payload), "size_bytes": len(payload)}
    return commit, bindings


def _write_receipt(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(_canonical_bytes(dict(body)))
    receipt = {**normalized, "receipt_sha256": _sha256(_canonical_bytes(normalized))}
    payload = _canonical_bytes(receipt)
    atomic_write_bytes(path, payload, mode=0o600)
    if path.read_bytes() != payload or path.stat().st_mode & 0o777 != 0o600:
        raise RuntimeError("transition discriminator receipt custody failed")
    return receipt


def _task_commitment(task: RecurrenceTrainingTask, *, split: str) -> dict[str, Any]:
    trace = task.transition_trace
    if trace is None:
        raise ValueError("transition discriminator task has no exact state trace")
    return {
        "split": split,
        "task_id": task.task_id,
        "family": task.family,
        "depth": task.depth,
        "prompt_sha256": _sha256(task.prompt.encode("utf-8")),
        "trace": trace.public_commitment(),
    }


def _mint_splits(
    *,
    families: tuple[str, ...],
    depths: tuple[int, ...],
    train_per_cell: int,
    development_per_cell: int,
    holdout_per_cell: int,
    seed: int,
) -> tuple[list[RecurrenceTrainingTask], ...]:
    train = task_battery(families, depths, train_per_cell, seed=seed)
    development = task_battery(
        families,
        depths,
        development_per_cell,
        seed=seed + 7_919,
        excluded_prompts=tuple(task.prompt for task in train),
        excluded_task_ids=tuple(task.task_id for task in train),
    )
    prior = [*train, *development]
    holdout = task_battery(
        families,
        depths,
        holdout_per_cell,
        seed=seed + 15_838,
        excluded_prompts=tuple(task.prompt for task in prior),
        excluded_task_ids=tuple(task.task_id for task in prior),
    )
    split_sets = [
        {task.task_id for task in split} for split in (train, development, holdout)
    ]
    prompt_sets = [
        {task.prompt for task in split} for split in (train, development, holdout)
    ]
    for left in range(3):
        for right in range(left + 1, 3):
            if split_sets[left] & split_sets[right] or prompt_sets[left] & prompt_sets[right]:
                raise RuntimeError("transition discriminator splits overlap")
    return train, development, holdout


def _training_coordinates(
    tasks: Sequence[RecurrenceTrainingTask],
    *,
    seed: int,
) -> list[tuple[int, int]]:
    coordinates = [
        (task_index, transition_index)
        for task_index, task in enumerate(tasks)
        for transition_index in range(task.depth)
    ]
    return sorted(
        coordinates,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[0]}:{item[1]}".encode("ascii")
        ).digest(),
    )


def _aggregate_evaluations(rows: Sequence[Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("transition evaluation is empty")
    grouped: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for task, evaluation in rows:
        grouped[(task.family, task.depth)].append(evaluation)

    def summarize(selected: Sequence[Any]) -> dict[str, Any]:
        field_total = sum(item.field_count for item in selected)
        return {
            "transitions": len(selected),
            "mean_loss": round(sum(item.loss for item in selected) / len(selected), 8),
            "exact_transitions": sum(item.exact for item in selected),
            "exact_accuracy": sum(item.exact for item in selected) / len(selected),
            "exact_fields": sum(item.exact_fields for item in selected),
            "field_count": field_total,
            "field_accuracy": sum(item.exact_fields for item in selected) / field_total,
            "target_field_accuracy": sum(
                item.predicted[1] == item.expected[1] for item in selected
            )
            / len(selected),
        }

    return {
        "aggregate": summarize([evaluation for _task, evaluation in rows]),
        "cells": {
            f"{family}:d{depth}": summarize(evaluations)
            for (family, depth), evaluations in sorted(grouped.items())
        },
    }


def _admission_gates(
    *,
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    development_improved: bool,
    base_checkpoint_immutable: bool,
    source_published: bool,
    splits_disjoint: bool,
    adapter_changed: bool,
) -> dict[str, bool]:
    before = baseline["aggregate"]
    after = treatment["aggregate"]
    return {
        "base_checkpoint_immutable": base_checkpoint_immutable,
        "source_published": source_published,
        "splits_disjoint": splits_disjoint,
        "adapter_changed": adapter_changed,
        "development_improved": development_improved,
        "holdout_loss_lower": after["mean_loss"] < before["mean_loss"],
        "holdout_exact_gain_at_least_0_20": (
            after["exact_accuracy"] - before["exact_accuracy"] >= 0.20
        ),
        "holdout_exact_accuracy_at_least_0_75": after["exact_accuracy"] >= 0.75,
        "holdout_field_accuracy_at_least_0_85": after["field_accuracy"] >= 0.85,
        "holdout_target_accuracy_at_least_0_85": after["target_field_accuracy"] >= 0.85,
        "every_cell_exact_nonregression": all(
            treatment["cells"][cell]["exact_accuracy"]
            >= baseline["cells"][cell]["exact_accuracy"]
            for cell in baseline["cells"]
        ),
    }


def _atomic_save_adapter(path: Path, tensors: Mapping[str, Any]) -> dict[str, Any]:
    import mlx.core as mx

    scratch = path.with_name(f".{path.stem}.{os.getpid()}.tmp.safetensors")
    mx.save_safetensors(str(scratch), dict(tensors))
    os.chmod(scratch, 0o600)
    os.replace(scratch, path)
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
        "mode": "0600",
    }


def run_discriminator(
    *,
    model_path: Path,
    out_dir: Path,
    seed: int,
    memory_fraction: float,
    depths: tuple[int, ...],
    train_per_cell: int,
    development_per_cell: int,
    holdout_per_cell: int,
    max_steps: int,
    evaluate_every: int,
    learning_rate: float,
    lora_rank: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load

    source_commit, source_bindings = _source_state()
    out_dir.mkdir(parents=True, exist_ok=False)
    os.chmod(out_dir, 0o700)
    families = ("boolean", "modular")
    train, development, holdout = _mint_splits(
        families=families,
        depths=depths,
        train_per_cell=train_per_cell,
        development_per_cell=development_per_cell,
        holdout_per_cell=holdout_per_cell,
        seed=seed,
    )
    all_tasks = [*train, *development, *holdout]
    if any(task.transition_trace is None for task in all_tasks):
        raise RuntimeError("structured transition trace disappeared")
    split_ids = [
        {task.task_id for task in split} for split in (train, development, holdout)
    ]
    splits_disjoint = all(
        split_ids[left].isdisjoint(split_ids[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution",),
        recurrent_steps=max(depths),
        exchange_interval=max(depths),
    )
    codebook = StateCodebookSpec(max_program_depth=max(depths), seed=seed ^ 0x51A7E)
    model_before = full_weight_checkpoint_identity(model_path)
    with (
        standalone_model_lane(
            owner_id=f"recurrent-transition-discriminator:{out_dir.name}",
            model_path=str(model_path),
            purpose="training",
            preemptible=False,
            metadata={"tool": Path(__file__).name, "source_commit": source_commit},
        ),
        mlx_memory_envelope(fraction=memory_fraction, restore_limits_on_exit=True),
    ):
        print("[transition-discriminator] model_load", flush=True)
        model, tokenizer = load(str(model_path))
        wrapped_sites = wrap_recurrent_window(
            model,
            spec=spec,
            lora_rank=lora_rank,
            lora_dropout=0.0,
            lora_scale=20.0,
            lora_targets=("o_proj", "v_proj"),
        )
        initial_adapter = adapter_tensor_dict(model)
        initial_fingerprint = adapter_tensor_fingerprint(initial_adapter)
        tokenized = {
            task.task_id: tokenize_task(tokenizer, task.prompt, task.answer)[0]
            for task in all_tasks
        }

        def evaluate(tasks: Sequence[RecurrenceTrainingTask], *, label: str) -> dict[str, Any]:
            rows = []
            for task_index, task in enumerate(tasks):
                trace = task.transition_trace
                assert trace is not None
                for transition_index in range(task.depth):
                    prepared = prepare_recurrent_transition_input(
                        model,
                        tokenized[task.task_id],
                        spec=spec,
                        transition_index=transition_index,
                    )
                    validate_recurrent_transition_input_receipt(prepared.receipt())
                    rows.append(
                        (
                            task,
                            evaluate_state_supervised_transition(
                                model,
                                prepared,
                                trace,
                                spec=spec,
                                codebook=codebook,
                            ),
                        )
                    )
                    mx.clear_cache()
                print(
                    f"[transition-discriminator] eval={label} "
                    f"task={task_index + 1}/{len(tasks)}",
                    flush=True,
                )
            return _aggregate_evaluations(rows)

        development_baseline = evaluate(development, label="development-baseline")
        baseline = evaluate(holdout, label="holdout-baseline")
        print(
            "[transition-discriminator] baselines "
            f"development_exact={development_baseline['aggregate']['exact_accuracy']:.4f} "
            f"development_target="
            f"{development_baseline['aggregate']['target_field_accuracy']:.4f} "
            f"holdout_exact={baseline['aggregate']['exact_accuracy']:.4f} "
            f"holdout_target={baseline['aggregate']['target_field_accuracy']:.4f}",
            flush=True,
        )
        optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
        optimizer.init(model.trainable_parameters())
        coordinates = _training_coordinates(train, seed=seed)
        best_development: dict[str, Any] | None = None
        best_score: tuple[float, float, float] | None = None
        best_step = 0
        best_artifact: dict[str, Any] | None = None
        loss_window: list[float] = []
        training_trail: list[dict[str, Any]] = []
        for step in range(1, max_steps + 1):
            task_index, transition_index = coordinates[(step - 1) % len(coordinates)]
            task = train[task_index]
            trace = task.transition_trace
            assert trace is not None
            prepared = prepare_recurrent_transition_input(
                model,
                tokenized[task.task_id],
                spec=spec,
                transition_index=transition_index,
            )
            gradient = state_supervised_transition_value_and_grad(
                model,
                prepared,
                trace,
                spec=spec,
                codebook=codebook,
            )
            optimizer.update(model, gradient.gradients)
            mx.eval(model.trainable_parameters(), optimizer.state)
            loss_window.append(gradient.value)
            del gradient
            mx.clear_cache()
            if step % evaluate_every == 0 or step == max_steps:
                development_report = evaluate(
                    development,
                    label=f"development-step-{step}",
                )
                aggregate = development_report["aggregate"]
                score = (
                    float(aggregate["exact_accuracy"]),
                    float(aggregate["target_field_accuracy"]),
                    -float(aggregate["mean_loss"]),
                )
                entry = {
                    "step": step,
                    "training_mean_loss": round(sum(loss_window) / len(loss_window), 8),
                    "development": aggregate,
                }
                training_trail.append(entry)
                loss_window.clear()
                print(
                    f"[transition-discriminator] step={step}/{max_steps} "
                    f"dev_exact={aggregate['exact_accuracy']:.4f} "
                    f"dev_target={aggregate['target_field_accuracy']:.4f} "
                    f"dev_loss={aggregate['mean_loss']:.6f}",
                    flush=True,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_step = step
                    best_development = development_report
                    best_artifact = _atomic_save_adapter(
                        out_dir / "best_adapter.safetensors",
                        adapter_tensor_dict(model),
                    )
        if best_artifact is None or best_development is None:
            raise RuntimeError("transition discriminator never selected a checkpoint")
        selected = mx.load(best_artifact["path"])
        model.load_weights(list(selected.items()), strict=False)
        mx.eval(model.trainable_parameters())
        treatment = evaluate(holdout, label="holdout-treatment")
        final_fingerprint = adapter_tensor_fingerprint(adapter_tensor_dict(model))
        mx.clear_cache()
    model_after = full_weight_checkpoint_identity(model_path)
    source_published = source_commit == subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "origin/main"], cwd=REPO_ROOT, text=True
    ).strip()
    development_before = development_baseline["aggregate"]
    development_after = best_development["aggregate"]
    development_improved = bool(
        development_after["exact_accuracy"]
        - development_before["exact_accuracy"]
        >= 0.10
        and development_after["target_field_accuracy"]
        >= development_before["target_field_accuracy"]
        and development_after["mean_loss"] < development_before["mean_loss"]
    )
    gates = _admission_gates(
        baseline=baseline,
        treatment=treatment,
        development_improved=development_improved,
        base_checkpoint_immutable=model_before == model_after,
        source_published=source_published,
        splits_disjoint=splits_disjoint,
        adapter_changed=initial_fingerprint != final_fingerprint,
    )
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
            "train_per_cell": train_per_cell,
            "development_per_cell": development_per_cell,
            "holdout_per_cell": holdout_per_cell,
            "max_steps": max_steps,
            "evaluate_every": evaluate_every,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "branches": 1,
            "adapter_scope": "shared_recurrent_window_only",
            "codebook": codebook.to_dict(),
            "execution_spec": spec.to_dict(),
        },
        "manifests": {
            "training": [_task_commitment(task, split="training") for task in train],
            "development": [
                _task_commitment(task, split="development") for task in development
            ],
            "holdout": [_task_commitment(task, split="holdout") for task in holdout],
        },
        "wrapped_sites": wrapped_sites,
        "initial_adapter_fingerprint": initial_fingerprint,
        "selected_adapter_fingerprint": final_fingerprint,
        "selected_adapter": best_artifact,
        "best_step": best_step,
        "training_trail": training_trail,
        "development_baseline": development_baseline,
        "best_development": best_development,
        "holdout_baseline": baseline,
        "holdout_treatment": treatment,
        "gates": gates,
        "admitted": all(gates.values()),
        "claim_boundary": (
            "one_step_state_transition_transfer_only_not_behavioral_or_reasoning_gain"
        ),
    }
    return _write_receipt(out_dir / "receipt.json", body)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810180)
    parser.add_argument("--memory-fraction", type=float, default=0.34)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--train-per-cell", type=_positive_int, default=8)
    parser.add_argument("--development-per-cell", type=_positive_int, default=2)
    parser.add_argument("--holdout-per-cell", type=_positive_int, default=4)
    parser.add_argument("--max-steps", type=_positive_int, default=256)
    parser.add_argument("--evaluate-every", type=_positive_int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lora-rank", type=_positive_int, default=8)
    args = parser.parse_args()
    try:
        depths = tuple(int(value.strip()) for value in args.depths.split(",") if value.strip())
        if (
            tuple(sorted(set(depths))) != depths
            or max(depths, default=0) > 4
            or args.max_steps % args.evaluate_every != 0
            or not math.isfinite(args.learning_rate)
            or not 0.0 < args.learning_rate <= 0.01
        ):
            raise ValueError("transition discriminator configuration is invalid")
        receipt = run_discriminator(
            model_path=args.model.expanduser().resolve(strict=True),
            out_dir=args.out_dir.expanduser().resolve(strict=False),
            seed=args.seed,
            memory_fraction=args.memory_fraction,
            depths=depths,
            train_per_cell=args.train_per_cell,
            development_per_cell=args.development_per_cell,
            holdout_per_cell=args.holdout_per_cell,
            max_steps=args.max_steps,
            evaluate_every=args.evaluate_every,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
        )
    except Exception as exc:  # noqa: BLE001 - printed to stderr and returned non-zero
        print(
            f"transition_discriminator_failed:{type(exc).__name__}:{exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0 if receipt["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
