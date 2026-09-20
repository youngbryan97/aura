#!/usr/bin/env python3
"""Test exact one-step transfer through Aura's native recurrent core."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
)
from core.brain.llm.latent_cortex.recurrent_transition_core import (  # noqa: E402
    RecurrentTransitionCore,
    RecurrentTransitionCoreConfig,
)
from core.learning.native_recurrent_transition import (  # noqa: E402
    ActionCodebookSpec,
    evaluate_native_transition,
    native_transition_value_and_grad,
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
from core.learning.recurrent_transition_supervision import (  # noqa: E402
    StateCodebookSpec,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

DISCRIMINATOR_SCHEMA: Final = "aura.native_transition_discriminator.v1"
SOURCE_PATHS: Final = (
    "core/brain/llm/latent_cortex/execution_spec.py",
    "core/brain/llm/latent_cortex/recurrent_transition_core.py",
    "core/learning/native_recurrent_transition.py",
    "core/learning/recurrence_curriculum.py",
    "core/learning/recurrence_native_objective_v2.py",
    "core/learning/recurrent_transition_supervision.py",
    "tools/run_native_transition_discriminator.py",
)


@dataclass(frozen=True, slots=True)
class CachedNativeTask:
    task: RecurrenceTrainingTask
    base_state: Any
    context: Any
    input_receipt: Mapping[str, Any]


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
    dirty = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("native discriminator requires a clean source checkout")
    commit = subprocess.check_output(["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    origin = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "origin/main"], cwd=REPO_ROOT, text=True
    ).strip()
    if commit != origin:
        raise RuntimeError("native discriminator source must equal published origin/main")
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
        raise RuntimeError("native discriminator receipt custody failed")
    return receipt


def _task_commitment(task: RecurrenceTrainingTask, *, split: str) -> dict[str, Any]:
    program = task.transition_program
    if program is None:
        raise ValueError("native discriminator task has no transition program")
    return {
        "split": split,
        "task_id": task.task_id,
        "family": task.family,
        "depth": task.depth,
        "prompt_sha256": _sha256(task.prompt.encode("utf-8")),
        "program": program.public_commitment(),
    }


def _mint_splits(
    *,
    depths: tuple[int, ...],
    train_per_cell: int,
    development_per_cell: int,
    holdout_per_cell: int,
    seed: int,
) -> tuple[list[RecurrenceTrainingTask], ...]:
    families = ("boolean", "modular")
    # At depth one the Boolean generator has exactly 2 initial values times
    # (one unary action plus three binary actions with two operands).
    if 1 in depths and (
        train_per_cell + development_per_cell + holdout_per_cell > 14
    ):
        raise ValueError("depth-one Boolean split exceeds its 14-program support")
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
    splits = (train, development, holdout)
    for left in range(3):
        for right in range(left + 1, 3):
            if {task.task_id for task in splits[left]} & {
                task.task_id for task in splits[right]
            } or {task.prompt for task in splits[left]} & {task.prompt for task in splits[right]}:
                raise RuntimeError("native discriminator splits overlap")
    return splits


def _training_coordinates(tasks: Sequence[CachedNativeTask], *, seed: int) -> list[tuple[int, int]]:
    coordinates = [
        (task_index, transition_index)
        for task_index, cached in enumerate(tasks)
        for transition_index in range(cached.task.depth)
    ]
    return sorted(
        coordinates,
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}:{item[1]}".encode("ascii")).digest(),
    )


def _aggregate(rows: Sequence[tuple[RecurrenceTrainingTask, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("native transition evaluation is empty")
    grouped: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for task, evaluation in rows:
        grouped[(task.family, task.depth)].append(evaluation)

    def summarize(selected: Sequence[Any]) -> dict[str, Any]:
        state_fields = sum(item.state_field_count for item in selected)
        action_fields = sum(item.action_field_count for item in selected)
        return {
            "transitions": len(selected),
            "mean_loss": round(sum(item.loss for item in selected) / len(selected), 8),
            "state_exact": sum(item.state_exact for item in selected),
            "state_exact_accuracy": sum(item.state_exact for item in selected) / len(selected),
            "state_field_accuracy": sum(item.state_exact_fields for item in selected)
            / state_fields,
            "target_field_accuracy": sum(
                item.predicted_state[1] == item.expected_state[1] for item in selected
            )
            / len(selected),
            "action_exact": sum(item.action_exact for item in selected),
            "action_exact_accuracy": sum(item.action_exact for item in selected) / len(selected),
            "action_field_accuracy": sum(item.action_exact_fields for item in selected)
            / action_fields,
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
    core_changed: bool,
) -> dict[str, bool]:
    before = baseline["aggregate"]
    after = treatment["aggregate"]
    return {
        "base_checkpoint_immutable": base_checkpoint_immutable,
        "source_published": source_published,
        "splits_disjoint": splits_disjoint,
        "core_changed": core_changed,
        "development_improved": development_improved,
        "holdout_loss_lower": after["mean_loss"] < before["mean_loss"],
        "holdout_state_exact_gain_at_least_0_50": (
            after["state_exact_accuracy"] - before["state_exact_accuracy"] >= 0.50
        ),
        "holdout_state_exact_accuracy_at_least_0_75": (after["state_exact_accuracy"] >= 0.75),
        "holdout_state_field_accuracy_at_least_0_90": (after["state_field_accuracy"] >= 0.90),
        "holdout_target_accuracy_at_least_0_85": (after["target_field_accuracy"] >= 0.85),
        "holdout_action_field_accuracy_at_least_0_85": (after["action_field_accuracy"] >= 0.85),
        "holdout_action_exact_accuracy_at_least_0_60": (after["action_exact_accuracy"] >= 0.60),
        "every_cell_state_exact_accuracy_at_least_0_50": all(
            cell["state_exact_accuracy"] >= 0.50 for cell in treatment["cells"].values()
        ),
        "every_cell_state_exact_nonregression": all(
            treatment["cells"][cell]["state_exact_accuracy"]
            >= baseline["cells"][cell]["state_exact_accuracy"]
            for cell in baseline["cells"]
        ),
    }


def _parameter_fingerprint(core: RecurrentTransitionCore) -> str:
    import numpy as np
    from mlx.utils import tree_flatten

    digest = hashlib.sha256()
    for name, value in sorted(tree_flatten(core.trainable_parameters())):
        array = np.asarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_save_core(path: Path, core: RecurrentTransitionCore) -> dict[str, Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    scratch = path.with_name(f".{path.stem}.{os.getpid()}.tmp.safetensors")
    tensors = dict(tree_flatten(core.trainable_parameters()))
    mx.save_safetensors(str(scratch), tensors)
    os.chmod(scratch, 0o600)
    os.replace(scratch, path)
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
        "mode": "0600",
    }


def _cache_tasks(
    model: Any,
    tokenizer: Any,
    tasks: Sequence[RecurrenceTrainingTask],
    *,
    spec: RLCExecutionSpec,
    label: str,
) -> list[CachedNativeTask]:
    import mlx.core as mx

    cached: list[CachedNativeTask] = []
    for index, task in enumerate(tasks):
        prompt_tokens = tokenize_task(tokenizer, task.prompt, task.answer)[0]
        prepared = prepare_recurrent_transition_input(
            model,
            prompt_tokens,
            spec=spec,
            transition_index=0,
        )
        receipt = validate_recurrent_transition_input_receipt(prepared.receipt())
        if len(prepared.parent_states) != 1 or len(prepared.prompts_at_window) != 1:
            raise RuntimeError("native discriminator expected exactly one frozen branch")
        base_state = mx.stop_gradient(prepared.parent_states[0])
        context = mx.stop_gradient(prepared.prompts_at_window[0])
        mx.eval(base_state, context)
        cached.append(
            CachedNativeTask(
                task=task,
                base_state=base_state,
                context=context,
                input_receipt=receipt,
            )
        )
        if (index + 1) % 8 == 0 or index + 1 == len(tasks):
            print(
                f"[native-transition] cache={label} task={index + 1}/{len(tasks)}",
                flush=True,
            )
    return cached


def _evaluate(
    core: RecurrentTransitionCore,
    tasks: Sequence[CachedNativeTask],
    *,
    state_codebook: StateCodebookSpec,
    action_codebook: ActionCodebookSpec,
    label: str,
) -> dict[str, Any]:
    import mlx.core as mx

    rows = []
    for task_index, cached in enumerate(tasks):
        program = cached.task.transition_program
        assert program is not None
        for transition_index in range(cached.task.depth):
            rows.append(
                (
                    cached.task,
                    evaluate_native_transition(
                        core,
                        cached.base_state,
                        cached.context,
                        program,
                        transition_index=transition_index,
                        state_codebook=state_codebook,
                        action_codebook=action_codebook,
                    ),
                )
            )
        if (task_index + 1) % 8 == 0 or task_index + 1 == len(tasks):
            print(
                f"[native-transition] eval={label} task={task_index + 1}/{len(tasks)}",
                flush=True,
            )
        mx.clear_cache()
    return _aggregate(rows)


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
    bottleneck_size: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load

    source_commit, source_bindings = _source_state()
    out_dir.mkdir(parents=True, exist_ok=False)
    os.chmod(out_dir, 0o700)
    train, development, holdout = _mint_splits(
        depths=depths,
        train_per_cell=train_per_cell,
        development_per_cell=development_per_cell,
        holdout_per_cell=holdout_per_cell,
        seed=seed,
    )
    split_ids = [{task.task_id for task in split} for split in (train, development, holdout)]
    splits_disjoint = all(
        split_ids[left].isdisjoint(split_ids[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    if any(task.transition_program is None for task in (*train, *development, *holdout)):
        raise RuntimeError("native transition program disappeared")
    spec = RLCExecutionSpec(
        n_slots=8,
        branch_roles=("constructive_solution",),
        recurrent_steps=max(depths),
        exchange_interval=max(depths),
    )
    state_codebook = StateCodebookSpec(
        max_program_depth=max(depths),
        seed=seed ^ 0x51A7E,
    )
    action_codebook = ActionCodebookSpec(seed=seed ^ 0xAC710)
    model_before = full_weight_checkpoint_identity(model_path)
    with (
        standalone_model_lane(
            owner_id=f"native-transition-discriminator:{out_dir.name}",
            model_path=str(model_path),
            purpose="training",
            preemptible=False,
            metadata={"tool": Path(__file__).name, "source_commit": source_commit},
        ),
        mlx_memory_envelope(fraction=memory_fraction, restore_limits_on_exit=True),
    ):
        print("[native-transition] model_load", flush=True)
        model, tokenizer = load(str(model_path))
        model.freeze()
        cached_train = _cache_tasks(model, tokenizer, train, spec=spec, label="train")
        cached_development = _cache_tasks(
            model, tokenizer, development, spec=spec, label="development"
        )
        cached_holdout = _cache_tasks(model, tokenizer, holdout, spec=spec, label="holdout")
        hidden_size = int(cached_train[0].base_state.shape[-1])
        if hidden_size != 1536:
            raise RuntimeError(
                f"native discriminator expected Qwen 1.5B hidden size 1536, got {hidden_size}"
            )
        del model, tokenizer
        mx.clear_cache()

        config = RecurrentTransitionCoreConfig(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            attention_heads=4,
        )
        mx.random.seed(seed ^ 0xC04E)
        core = RecurrentTransitionCore(config)
        baseline_core = RecurrentTransitionCore(config)
        from mlx.utils import tree_flatten

        baseline_core.load_weights(
            list(tree_flatten(core.trainable_parameters())),
            strict=True,
        )
        mx.eval(baseline_core.parameters(), core.parameters())
        initial_fingerprint = _parameter_fingerprint(core)
        if _parameter_fingerprint(baseline_core) != initial_fingerprint:
            raise RuntimeError("identity and treatment core initialization differs")
        development_baseline = _evaluate(
            baseline_core,
            cached_development,
            state_codebook=state_codebook,
            action_codebook=action_codebook,
            label="development-baseline",
        )
        optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
        optimizer.init(core.trainable_parameters())
        coordinates = _training_coordinates(cached_train, seed=seed)
        best_development: dict[str, Any] | None = None
        best_score: tuple[float, float, float, float] | None = None
        best_step = 0
        best_artifact: dict[str, Any] | None = None
        loss_window: list[float] = []
        training_trail: list[dict[str, Any]] = []
        for step in range(1, max_steps + 1):
            task_index, transition_index = coordinates[(step - 1) % len(coordinates)]
            cached = cached_train[task_index]
            program = cached.task.transition_program
            assert program is not None
            gradient = native_transition_value_and_grad(
                core,
                cached.base_state,
                cached.context,
                program,
                transition_index=transition_index,
                state_codebook=state_codebook,
                action_codebook=action_codebook,
            )
            optimizer.update(core, gradient.gradients)
            mx.eval(core.trainable_parameters(), optimizer.state)
            loss_window.append(gradient.value)
            del gradient
            if step % evaluate_every == 0 or step == max_steps:
                development_report = _evaluate(
                    core,
                    cached_development,
                    state_codebook=state_codebook,
                    action_codebook=action_codebook,
                    label=f"development-step-{step}",
                )
                aggregate = development_report["aggregate"]
                score = (
                    float(aggregate["state_exact_accuracy"]),
                    float(aggregate["target_field_accuracy"]),
                    float(aggregate["action_exact_accuracy"]),
                    -float(aggregate["mean_loss"]),
                )
                training_trail.append(
                    {
                        "step": step,
                        "training_mean_loss": round(sum(loss_window) / len(loss_window), 8),
                        "development": aggregate,
                    }
                )
                loss_window.clear()
                print(
                    f"[native-transition] step={step}/{max_steps} "
                    f"dev_state={aggregate['state_exact_accuracy']:.4f} "
                    f"dev_target={aggregate['target_field_accuracy']:.4f} "
                    f"dev_action={aggregate['action_exact_accuracy']:.4f} "
                    f"dev_loss={aggregate['mean_loss']:.6f}",
                    flush=True,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_step = step
                    best_development = development_report
                    best_artifact = _atomic_save_core(out_dir / "best_core.safetensors", core)
        if best_development is None or best_artifact is None:
            raise RuntimeError("native discriminator never selected a checkpoint")
        selected = mx.load(best_artifact["path"])
        core.load_weights(list(selected.items()), strict=True)
        mx.eval(core.parameters())

        # The holdout is opened exactly once, after checkpoint selection, for a
        # paired identity-vs-treatment comparison on identical frozen inputs.
        baseline = _evaluate(
            baseline_core,
            cached_holdout,
            state_codebook=state_codebook,
            action_codebook=action_codebook,
            label="holdout-identity",
        )
        treatment = _evaluate(
            core,
            cached_holdout,
            state_codebook=state_codebook,
            action_codebook=action_codebook,
            label="holdout-treatment",
        )
        final_fingerprint = _parameter_fingerprint(core)
        mx.clear_cache()

    model_after = full_weight_checkpoint_identity(model_path)
    source_published = (
        source_commit
        == subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "rev-parse", "origin/main"], cwd=REPO_ROOT, text=True
        ).strip()
    )
    development_before = development_baseline["aggregate"]
    development_after = best_development["aggregate"]
    development_improved = bool(
        development_after["state_exact_accuracy"] - development_before["state_exact_accuracy"]
        >= 0.10
        and development_after["target_field_accuracy"]
        >= development_before["target_field_accuracy"]
        and development_after["action_field_accuracy"]
        >= development_before["action_field_accuracy"]
        and development_after["mean_loss"] < development_before["mean_loss"]
    )
    gates = _admission_gates(
        baseline=baseline,
        treatment=treatment,
        development_improved=development_improved,
        base_checkpoint_immutable=model_before == model_after,
        source_published=source_published,
        splits_disjoint=splits_disjoint,
        core_changed=initial_fingerprint != final_fingerprint,
    )
    body = {
        "schema": DISCRIMINATOR_SCHEMA,
        "source_commit": source_commit,
        "source_bindings": source_bindings,
        "model_path": str(model_path),
        "model_identity": model_before,
        "configuration": {
            "seed": seed,
            "families": ["boolean", "modular"],
            "depths": list(depths),
            "train_per_cell": train_per_cell,
            "development_per_cell": development_per_cell,
            "holdout_per_cell": holdout_per_cell,
            "max_steps": max_steps,
            "evaluate_every": evaluate_every,
            "learning_rate": learning_rate,
            "core": config.to_dict(),
            "state_codebook": state_codebook.to_dict(),
            "action_codebook": action_codebook.to_dict(),
            "execution_spec": spec.to_dict(),
            "base_model_role": "frozen_semantic_encoder_only",
            "transition_signature": "state_typed_action_semantic_context_to_state",
            "holdout_policy": "opened_once_after_development_selection",
        },
        "manifests": {
            "training": [_task_commitment(task, split="training") for task in train],
            "development": [_task_commitment(task, split="development") for task in development],
            "holdout": [_task_commitment(task, split="holdout") for task in holdout],
        },
        "initial_core_fingerprint": initial_fingerprint,
        "selected_core_fingerprint": final_fingerprint,
        "selected_core": best_artifact,
        "best_step": best_step,
        "training_trail": training_trail,
        "development_baseline": development_baseline,
        "best_development": best_development,
        "holdout_baseline": baseline,
        "holdout_treatment": treatment,
        "gates": gates,
        "admitted": all(gates.values()),
        "claim_boundary": (
            "typed_action_given_one_step_native_transition_transfer_only_not_action_"
            "parsing_composition_behavioral_or_reasoning_gain"
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
    parser.add_argument("--seed", type=int, default=20260810185)
    parser.add_argument("--memory-fraction", type=float, default=0.34)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--train-per-cell", type=_positive_int, default=8)
    parser.add_argument("--development-per-cell", type=_positive_int, default=2)
    parser.add_argument("--holdout-per-cell", type=_positive_int, default=4)
    parser.add_argument("--max-steps", type=_positive_int, default=512)
    parser.add_argument("--evaluate-every", type=_positive_int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bottleneck-size", type=_positive_int, default=128)
    args = parser.parse_args()
    try:
        depths = tuple(int(value.strip()) for value in args.depths.split(",") if value.strip())
        if (
            tuple(sorted(set(depths))) != depths
            or max(depths, default=0) > 4
            or args.max_steps % args.evaluate_every != 0
            or not math.isfinite(args.memory_fraction)
            or not 0.05 <= args.memory_fraction <= 0.75
            or not math.isfinite(args.learning_rate)
            or not 0.0 < args.learning_rate <= 0.01
            or args.bottleneck_size % 4 != 0
        ):
            raise ValueError("native discriminator configuration is invalid")
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
            bottleneck_size=args.bottleneck_size,
        )
    except Exception as exc:  # noqa: BLE001 - printed to stderr and returned non-zero
        print(
            f"native_transition_discriminator_failed:{type(exc).__name__}:{exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0 if receipt["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
