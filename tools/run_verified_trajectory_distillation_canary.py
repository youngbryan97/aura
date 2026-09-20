#!/usr/bin/env python3
"""Distill verified RLC state corrections into persistent recurrent tissue.

This is the first persistent-transfer discriminator for the causal CP144
mechanism.  Private exact solutions supervise only internal activation
differences on training tasks.  Held-out generation runs with both the exact
objective producer and the per-query verified teacher disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
)
from core.brain.llm.latent_cortex.types import FastWeightsConfig  # noqa: E402
from core.learning.recurrence_curriculum import task_battery  # noqa: E402
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    generate_cached_live_path_rollin,
)
from core.learning.recurrent_behavioral_probe import (  # noqa: E402
    build_paired_full_engine_probe_reports,
    paired_generation_seed,
    tokenize_task,
)
from core.learning.recurrent_checkpoint_admission import (  # noqa: E402
    build_recurrence_task_manifest,
    validate_free_generation_report,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    attach_coda_policy_adapters_at_sites,
    attach_recurrent_policy_adapters,
)
from core.learning.recurrent_sft_execution import (  # noqa: E402
    adapter_tensor_dict,
    adapter_tensor_fingerprint,
)
from core.learning.verified_trajectory_distillation import (  # noqa: E402
    evaluate_verified_trajectory_transfer,
    fit_verified_trajectory_inventory,
    fit_verified_trajectory_sample_complexity,
    install_verified_trajectory_inventory,
    publish_verified_trajectory_artifact,
)
from core.runtime.atomic_writer import atomic_append_text, atomic_write_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.verified_trajectory_distillation_canary.v1"
SAMPLE_COMPLEXITY_CANARY_SCHEMA: Final = (
    "aura.verified_trajectory_sample_complexity_canary.v1"
)
PROGRESS_SCHEMA: Final = "aura.verified_trajectory_distillation.progress.v1"
SOURCE_PATHS: Final = (
    "core/brain/llm/latent_cortex/engine.py",
    "core/brain/llm/latent_cortex/fast_weights.py",
    "core/brain/llm/latent_cortex/recurrence_adapter.py",
    "core/brain/llm/latent_cortex/types.py",
    "core/learning/recurrence_curriculum.py",
    "core/learning/recurrence_native_objective_v2.py",
    "core/learning/recurrent_behavioral_probe.py",
    "core/learning/recurrent_checkpoint_admission.py",
    "core/learning/recurrent_grpo.py",
    "core/learning/verified_trajectory_distillation.py",
    "tools/run_verified_trajectory_distillation_canary.py",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_receipt_payload(payload: bytes) -> dict[str, Any]:
    """Replay a receipt hash against the exact canonical bytes on disk."""

    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("trajectory receipt is not canonical JSON") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != payload:
        raise RuntimeError("trajectory receipt is not canonical JSON")
    receipt_sha256 = value.pop("receipt_sha256", None)
    if (
        not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or _sha256_bytes(_canonical_bytes(value)) != receipt_sha256
    ):
        raise RuntimeError("trajectory receipt hash does not bind persisted body")
    return {**value, "receipt_sha256": receipt_sha256}


def _write_receipt(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize once, hash what will persist, and verify the atomic write."""

    normalized = json.loads(_canonical_bytes(dict(body)))
    receipt = {
        **normalized,
        "receipt_sha256": _sha256_bytes(_canonical_bytes(normalized)),
    }
    payload = _canonical_bytes(receipt)
    atomic_write_bytes(path, payload, mode=0o600)
    persisted = path.read_bytes()
    if persisted != payload:
        raise RuntimeError("trajectory receipt atomic write changed payload")
    return _validate_receipt_payload(persisted)


class ProgressLedger:
    def __init__(self, out_dir: Path, *, source_commit: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=False)
        self.out_dir = out_dir
        self.source_commit = source_commit
        self.started = time.time()
        self.sequence = 0

    def emit(self, phase: str, **details: Any) -> dict[str, Any]:
        self.sequence += 1
        body = {
            "schema": PROGRESS_SCHEMA,
            "sequence": self.sequence,
            "phase": phase,
            "pid": os.getpid(),
            "source_commit": self.source_commit,
            "elapsed_s": time.time() - self.started,
            "details": details,
        }
        event = {**body, "event_sha256": _sha256_bytes(_canonical_bytes(body))}
        payload = _canonical_bytes(event)
        atomic_append_text(self.out_dir / "progress.jsonl", payload.decode() + "\n")
        atomic_write_bytes(self.out_dir / "progress.json", payload, mode=0o600)
        detail = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
        print(f"[trajectory] {phase} {detail}".rstrip(), file=sys.stderr, flush=True)
        return event


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_state() -> tuple[str, dict[str, dict[str, Any]]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("trajectory canary requires a clean source worktree")
    head = _git("rev-parse", "HEAD")
    if head != _git("rev-parse", "origin/main"):
        raise RuntimeError("trajectory canary source is not published on origin/main")
    bindings = {}
    for relative in SOURCE_PATHS:
        payload = (REPO_ROOT / relative).read_bytes()
        committed = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "show", f"{head}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if payload != committed:
            raise RuntimeError(f"trajectory source differs from commit: {relative}")
        bindings[relative] = {
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        }
    return head, bindings


def _trajectory_io_features(
    model: Any,
    fast_weights: EpisodicFastWeights,
    context_tokens: Sequence[int],
    *,
    token_start: int,
) -> tuple[dict[int, Any], dict[int, Any]]:
    import mlx.core as mx

    tokens = [int(token) for token in context_tokens]
    if not tokens or not 0 <= token_start < len(tokens):
        raise ValueError("trajectory context boundary is invalid")

    def forward() -> Any:
        hidden = model.model.embed_tokens(mx.array([tokens]))
        for layer in model.model.layers:
            hidden = layer(hidden, None, None)
        mx.eval(hidden)
        return hidden

    return fast_weights.capture_io_features(forward, token_start=token_start)


def _capture_training_inventory(
    model: Any,
    tokenizer: Any,
    tasks: Sequence[Any],
    *,
    spec: RLCExecutionSpec,
    rank: int,
    layers: int,
    seed: int,
    progress: ProgressLedger,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    import mlx.core as mx

    prelude_end = max(1, int(len(model.model.layers) * spec.prelude_frac))
    coda_start = min(
        len(model.model.layers) - 1,
        len(model.model.layers) - max(1, int(len(model.model.layers) * spec.coda_frac)),
    )
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(
            enabled=True,
            rank=rank,
            scale=1.0,
            target="o_proj",
            layer_placement="early",
            max_wrapped_layers=layers,
            opt_steps=0,
            canary_enabled=False,
            canary_generated_enabled=False,
        )
    )
    fast_weights.attach(
        model.model,
        (prelude_end, coda_start),
        seed_stat=1.0,
        episode_id=f"trajectory-distillation-{seed}",
    )
    inputs: dict[str, list[np.ndarray]] = defaultdict(list)
    corrections: dict[str, list[np.ndarray]] = defaultdict(list)
    manifest: list[dict[str, Any]] = []
    try:
        for task_ordinal, task in enumerate(tasks):
            prompt_tokens, target_tokens = tokenize_task(
                tokenizer,
                task.prompt,
                task.training_target,
            )
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None and target_tokens and target_tokens[-1] == int(eos):
                target_tokens = target_tokens[:-1]
            for branch_index in range(len(spec.branch_roles)):
                sample_seed = paired_generation_seed(
                    seed,
                    task_ordinal,
                    task.task_id,
                    branch_index + 1,
                )
                incumbent = generate_cached_live_path_rollin(
                    model,
                    prompt_tokens,
                    spec=spec,
                    branch_index=branch_index,
                    token_count=len(target_tokens),
                    seed=sample_seed,
                    temperature=0.0,
                )
                _teacher_inputs, teacher_outputs = _trajectory_io_features(
                    model,
                    fast_weights,
                    [*prompt_tokens, *target_tokens],
                    token_start=len(prompt_tokens),
                )
                incumbent_inputs, incumbent_outputs = _trajectory_io_features(
                    model,
                    fast_weights,
                    [*prompt_tokens, *incumbent.tokens],
                    token_start=len(prompt_tokens),
                )
                if set(teacher_outputs) != set(incumbent_outputs) or set(
                    incumbent_inputs
                ) != set(incumbent_outputs):
                    raise RuntimeError("trajectory decode I/O inventories differ")
                for layer_index in sorted(incumbent_outputs):
                    site = f"model.layers.{layer_index}.self_attn.o_proj"
                    layer_inputs = incumbent_inputs[layer_index]
                    layer_corrections = (
                        teacher_outputs[layer_index] - incumbent_outputs[layer_index]
                    )
                    mx.eval(layer_inputs, layer_corrections)
                    if int(layer_inputs.shape[0]) != int(layer_corrections.shape[0]):
                        raise RuntimeError("trajectory decode I/O row counts differ")
                    for position in range(int(layer_inputs.shape[0])):
                        inputs[site].append(
                            np.asarray(layer_inputs[position]).astype(np.float64)
                        )
                        corrections[site].append(
                            np.asarray(layer_corrections[position]).astype(np.float64)
                        )
                row_counts = {len(values) for values in inputs.values()}
                if len(row_counts) != 1:
                    raise RuntimeError("trajectory decode row offsets differ by site")
                row_stop = row_counts.pop()
                row_start = int(manifest[-1]["row_stop"]) if manifest else 0
                feature_row_count = row_stop - row_start
                if feature_row_count <= 0:
                    raise RuntimeError("trajectory decode example boundary is invalid")
                manifest.append(
                    {
                        "task_id": task.task_id,
                        "branch_index": branch_index,
                        "prompt_sha256": _sha256_bytes(task.prompt.encode()),
                        "private_target_sha256": _sha256_bytes(
                            task.training_target.encode()
                        ),
                        "incumbent_tokens_sha256": _sha256_bytes(
                            _canonical_bytes(list(incumbent.tokens))
                        ),
                        "row_start": row_start,
                        "row_stop": row_stop,
                        "feature_row_count": feature_row_count,
                        "target_token_count": int(len(target_tokens)),
                    }
                )
                progress.emit(
                    "trajectory_pair_captured",
                    task=task_ordinal + 1,
                    total_tasks=len(tasks),
                    branch=branch_index,
                    sites=len(incumbent_outputs),
                )
                mx.clear_cache()
    finally:
        fast_weights.detach()
    inventory = {
        site: (np.stack(inputs[site]), np.stack(corrections[site]))
        for site in sorted(inputs)
    }
    if len(inventory) != layers:
        raise RuntimeError("captured trajectory site inventory differs from topology")
    return inventory, manifest


def _report_score(report: Mapping[str, Any]) -> int:
    return int(report["total_correct"])


def _write_private_pair_artifact(
    out_dir: Path,
    *,
    training_pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    validation_cohorts: Mapping[
        str, Mapping[str, tuple[np.ndarray, np.ndarray]]
    ],
) -> dict[str, Any]:
    """Persist private activation pairs atomically with an exact inventory."""

    arrays: dict[str, np.ndarray] = {}
    inventory: dict[str, Any] = {}
    pair_sets: dict[str, Mapping[str, tuple[np.ndarray, np.ndarray]]] = {
        "training": training_pairs,
        **{
            f"validation_{index:04d}": validation_cohorts[name]
            for index, name in enumerate(sorted(validation_cohorts))
        },
    }
    cohort_names = {
        f"validation_{index:04d}": name
        for index, name in enumerate(sorted(validation_cohorts))
    }
    for set_name, pairs in pair_sets.items():
        inventory[set_name] = {}
        for site_index, site in enumerate(sorted(pairs)):
            pair = pairs[site]
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError(f"private trajectory pair is invalid at {set_name}:{site}")
            inputs = np.ascontiguousarray(pair[0], dtype=np.float64)
            corrections = np.ascontiguousarray(pair[1], dtype=np.float64)
            if (
                inputs.ndim != 2
                or corrections.ndim != 2
                or inputs.shape[0] != corrections.shape[0]
                or not inputs.size
                or not corrections.size
                or not np.all(np.isfinite(inputs))
                or not np.all(np.isfinite(corrections))
            ):
                raise ValueError(
                    f"private trajectory pair geometry is invalid at {set_name}:{site}"
                )
            input_key = f"{set_name}__site_{site_index:04d}__inputs"
            correction_key = f"{set_name}__site_{site_index:04d}__corrections"
            arrays[input_key] = inputs
            arrays[correction_key] = corrections
            inventory[set_name][site] = {
                "input_key": input_key,
                "input_shape": list(inputs.shape),
                "input_sha256": _sha256_bytes(inputs.tobytes(order="C")),
                "correction_key": correction_key,
                "correction_shape": list(corrections.shape),
                "correction_sha256": _sha256_bytes(
                    corrections.tobytes(order="C")
                ),
            }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    path = out_dir / "private_teaching_pairs.npz"
    atomic_write_bytes(path, payload, mode=0o600)
    persisted = path.read_bytes()
    if persisted != payload:
        raise RuntimeError("private trajectory pair artifact changed during write")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "mode": "0600",
        "cohort_names": cohort_names,
        "inventory": inventory,
    }


def _sample_rows_from_complete_examples(
    manifest: Sequence[Mapping[str, Any]],
    *,
    per_cell_levels: Sequence[int],
    stratum_count: int,
    branch_count: int,
) -> tuple[int, ...]:
    """Translate nested per-cell levels into complete trajectory row bounds."""

    if (
        isinstance(per_cell_levels, (str, bytes))
        or len(per_cell_levels) < 2
        or any(type(level) is not int or level < 1 for level in per_cell_levels)
        or tuple(sorted(set(per_cell_levels))) != tuple(per_cell_levels)
    ):
        raise ValueError("sample-complexity per-cell levels must increase")
    if type(stratum_count) is not int or stratum_count < 1:
        raise ValueError("sample-complexity stratum count is invalid")
    if type(branch_count) is not int or branch_count < 1:
        raise ValueError("sample-complexity branch count is invalid")
    expected_examples = per_cell_levels[-1] * stratum_count * branch_count
    if len(manifest) != expected_examples:
        raise ValueError("sample-complexity teaching manifest coverage differs")
    boundaries = []
    previous_stop = 0
    for row in manifest:
        start = row.get("row_start")
        stop = row.get("row_stop")
        feature_rows = row.get("feature_row_count")
        target_tokens = row.get("target_token_count")
        if (
            type(start) is not int
            or type(stop) is not int
            or start != previous_stop
            or stop <= start
            or type(feature_rows) is not int
            or feature_rows != stop - start
            or type(target_tokens) is not int
            or target_tokens < 1
        ):
            raise ValueError("sample-complexity teaching row boundaries are invalid")
        previous_stop = stop
    for level in per_cell_levels:
        examples = level * stratum_count * branch_count
        boundaries.append(int(manifest[examples - 1]["row_stop"]))
    if tuple(sorted(set(boundaries))) != tuple(boundaries):
        raise ValueError("sample-complexity row boundaries do not increase")
    return tuple(boundaries)


@contextmanager
def _zeroed_recurrence_adapter(model: Any) -> Iterator[None]:
    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        ScopedCodaLoRALinear,
        ScopedLoRALinear,
    )

    snapshots = []
    for layer in model.model.layers:
        projection = getattr(getattr(layer, "self_attn", None), "o_proj", None)
        if isinstance(projection, (ScopedLoRALinear, ScopedCodaLoRALinear)):
            snapshots.append((projection, projection.lora_b))
            projection.lora_b = mx.zeros_like(projection.lora_b)
    if not snapshots:
        raise RuntimeError("trajectory lesion found no scoped adapter")
    mx.eval(*(projection.lora_b for projection, _ in snapshots))
    try:
        yield
    finally:
        for projection, value in snapshots:
            projection.lora_b = value
        mx.eval(*(projection.lora_b for projection, _ in snapshots))


@contextmanager
def _permuted_recurrence_adapter(model: Any) -> Iterator[None]:
    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        ScopedCodaLoRALinear,
        ScopedLoRALinear,
    )

    snapshots = []
    for layer in model.model.layers:
        projection = getattr(getattr(layer, "self_attn", None), "o_proj", None)
        if isinstance(projection, (ScopedLoRALinear, ScopedCodaLoRALinear)):
            snapshots.append((projection, projection.lora_b))
            width = int(projection.lora_b.shape[1])
            permutation = mx.array(list(range(1, width)) + [0])
            projection.lora_b = projection.lora_b[:, permutation]
    if not snapshots:
        raise RuntimeError("trajectory sham found no scoped adapter")
    mx.eval(*(projection.lora_b for projection, _ in snapshots))
    try:
        yield
    finally:
        for projection, value in snapshots:
            projection.lora_b = value
        mx.eval(*(projection.lora_b for projection, _ in snapshots))


def run_sample_complexity_canary(
    *,
    model_path: Path,
    out_dir: Path,
    seed: int,
    memory_fraction: float,
    training_families: Sequence[str],
    training_depths: Sequence[int],
    training_per_cell_levels: Sequence[int],
    validation_per_cell: int,
    validation_cohort_count: int,
    lora_rank: int,
    lora_layers: int,
    regularization: float,
    gain: float,
) -> dict[str, Any]:
    """Measure one fixed trajectory learner across nested evidence volumes."""

    import mlx.core as mx
    from mlx_lm import load

    levels = tuple(training_per_cell_levels)
    if (
        len(levels) < 2
        or any(type(level) is not int or level < 1 for level in levels)
        or tuple(sorted(set(levels))) != levels
    ):
        raise ValueError("sample-complexity training levels must increase")
    if type(validation_cohort_count) is not int or validation_cohort_count < 3:
        raise ValueError("sample complexity requires at least three fresh cohorts")
    source_commit, source_bindings = _source_state()
    progress = ProgressLedger(out_dir, source_commit=source_commit)
    progress.emit(
        "source_bound",
        model_path=str(model_path),
        mode="sample_complexity",
    )
    base_before = full_weight_checkpoint_identity(model_path)
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=2,
        exchange_interval=1,
    )
    with (
        standalone_model_lane(
            owner_id=f"verified-trajectory-scaling:{out_dir.name}",
            model_path=str(model_path),
            purpose="training",
            preemptible=False,
            metadata={"tool": Path(__file__).name, "source_commit": source_commit},
        ),
        mlx_memory_envelope(fraction=memory_fraction, restore_limits_on_exit=True),
    ):
        progress.emit("model_load")
        model, tokenizer = load(str(model_path))
        attached_sites = attach_recurrent_policy_adapters(
            model,
            spec,
            lora_rank=lora_rank,
            lora_layers=lora_layers,
            lora_targets=("o_proj",),
            initialization_seed=(seed ^ 0x51F7A11) & 0xFFFFFFFF,
            lora_scale=1.0,
            lora_layer_placement="early",
        )
        training_tasks = task_battery(
            list(training_families),
            list(training_depths),
            levels[-1],
            seed=seed,
        )
        excluded_prompts = [task.prompt for task in training_tasks]
        excluded_task_ids = [task.task_id for task in training_tasks]
        validation_task_cohorts: dict[str, list[Any]] = {}
        for cohort_index in range(validation_cohort_count):
            cohort_name = f"fresh_seed_{cohort_index:04d}"
            cohort_seed = seed + 3_959 + cohort_index * 104_729
            cohort_tasks = task_battery(
                list(training_families),
                list(training_depths),
                validation_per_cell,
                seed=cohort_seed,
                excluded_prompts=tuple(excluded_prompts),
                excluded_task_ids=tuple(excluded_task_ids),
            )
            validation_task_cohorts[cohort_name] = cohort_tasks
            excluded_prompts.extend(task.prompt for task in cohort_tasks)
            excluded_task_ids.extend(task.task_id for task in cohort_tasks)

        teaching_pairs, teaching_manifest = _capture_training_inventory(
            model,
            tokenizer,
            training_tasks,
            spec=spec,
            rank=min(2, lora_rank),
            layers=lora_layers,
            seed=seed,
            progress=progress,
        )
        validation_pairs: dict[
            str, dict[str, tuple[np.ndarray, np.ndarray]]
        ] = {}
        validation_manifests: dict[str, list[dict[str, Any]]] = {}
        for cohort_index, cohort_name in enumerate(sorted(validation_task_cohorts)):
            pairs, manifest = _capture_training_inventory(
                model,
                tokenizer,
                validation_task_cohorts[cohort_name],
                spec=spec,
                rank=min(2, lora_rank),
                layers=lora_layers,
                seed=seed + 3_959 + cohort_index * 104_729,
                progress=progress,
            )
            validation_pairs[cohort_name] = pairs
            validation_manifests[cohort_name] = manifest
        sample_rows = _sample_rows_from_complete_examples(
            teaching_manifest,
            per_cell_levels=levels,
            stratum_count=len(training_families) * len(training_depths),
            branch_count=len(spec.branch_roles),
        )
        decode_phases = {site: "decode" for site in teaching_pairs}
        scaling_report, final_inventory = fit_verified_trajectory_sample_complexity(
            teaching_pairs,
            validation_pairs,
            sample_rows=sample_rows,
            rank=lora_rank,
            regularization=regularization,
            gain=gain,
            adapter_scale=1.0,
            site_phases=decode_phases,
            normalize_corrections=False,
        )
        for stage in scaling_report["stages"]:
            progress.emit(
                "sample_complexity_stage",
                training_rows=stage["training_rows"],
                **stage["summary"],
            )
        pair_artifact = _write_private_pair_artifact(
            out_dir,
            training_pairs=teaching_pairs,
            validation_cohorts=validation_pairs,
        )
        fitted_artifact = publish_verified_trajectory_artifact(
            out_dir / "fitted_adapter",
            final_inventory,
            checkpoint_fingerprint=str(base_before["fingerprint"]),
            source_evidence_sha256=str(scaling_report["report_sha256"]),
        )
        mx.synchronize()
        mx.clear_cache()

    all_manifests = [teaching_manifest, *validation_manifests.values()]
    task_id_sets = [
        {str(row["task_id"]) for row in manifest} for manifest in all_manifests
    ]
    prompt_hash_sets = [
        {str(row["prompt_sha256"]) for row in manifest}
        for manifest in all_manifests
    ]
    gates = {
        "base_checkpoint_immutable": (
            base_before == full_weight_checkpoint_identity(model_path)
        ),
        "task_ids_globally_disjoint": sum(map(len, task_id_sets))
        == len(set().union(*task_id_sets)),
        "prompts_globally_disjoint": (
            sum(map(len, prompt_hash_sets))
            == len(set().union(*prompt_hash_sets))
        ),
        "fixed_hyperparameter_scaling_admitted": bool(scaling_report["admitted"]),
        "fitted_site_topology_complete": set(final_inventory)
        == set(attached_sites),
        "fitted_artifact_published": bool(
            fitted_artifact["manifest"].get("receipt_sha256")
        ),
    }
    body = {
        "schema": SAMPLE_COMPLEXITY_CANARY_SCHEMA,
        "mode": "sample_complexity",
        "source_commit": source_commit,
        "source_bindings": source_bindings,
        "model_path": str(model_path),
        "model_identity": base_before,
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "configuration": {
            "seed": seed,
            "training_families": list(training_families),
            "training_depths": list(training_depths),
            "training_per_cell_levels": list(levels),
            "validation_per_cell": validation_per_cell,
            "validation_cohort_count": validation_cohort_count,
            "lora_rank": lora_rank,
            "lora_layers": lora_layers,
            "regularization": regularization,
            "gain": gain,
        },
        "private_teaching_manifest": teaching_manifest,
        "private_validation_manifests": validation_manifests,
        "private_pair_artifact": pair_artifact,
        "sample_complexity": scaling_report,
        "fitted_adapter_artifact": fitted_artifact,
        "gates": gates,
        "admitted": all(gates.values()),
        "claim_boundary": (
            "fresh_seed_internal_operator_scaling_only_not_behavioral_or_reasoning_gain"
        ),
    }
    receipt = _write_receipt(out_dir / "receipt.json", body)
    progress.emit("complete", admitted=receipt["admitted"], **gates)
    return receipt


def run_canary(
    *,
    model_path: Path,
    out_dir: Path,
    seed: int,
    memory_fraction: float,
    training_families: Sequence[str],
    training_depths: Sequence[int],
    training_per_cell: int,
    validation_per_cell: int,
    proxy_per_cell: int,
    lora_rank: int,
    lora_layers: int,
    regularization: float,
    gain: float,
    stop_after_transfer_diagnostic: bool = False,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load

    source_commit, source_bindings = _source_state()
    progress = ProgressLedger(out_dir, source_commit=source_commit)
    progress.emit("source_bound", model_path=str(model_path))
    base_before = full_weight_checkpoint_identity(model_path)
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=2,
        exchange_interval=1,
    )
    with (
        standalone_model_lane(
            owner_id=f"verified-trajectory:{out_dir.name}",
            model_path=str(model_path),
            purpose="training",
            preemptible=False,
            metadata={"tool": Path(__file__).name, "source_commit": source_commit},
        ),
        mlx_memory_envelope(fraction=memory_fraction, restore_limits_on_exit=True),
    ):
        progress.emit("model_load")
        model, tokenizer = load(str(model_path))
        sites = attach_recurrent_policy_adapters(
            model,
            spec,
            lora_rank=lora_rank,
            lora_layers=lora_layers,
            lora_targets=("o_proj",),
            initialization_seed=(seed ^ 0x51F7A11) & 0xFFFFFFFF,
            lora_scale=1.0,
            lora_layer_placement="early",
        )
        adapter_before = adapter_tensor_fingerprint(adapter_tensor_dict(model))
        training_tasks = task_battery(
            list(training_families),
            list(training_depths),
            training_per_cell,
            seed=seed,
        )
        validation_tasks = task_battery(
            list(training_families),
            list(training_depths),
            validation_per_cell,
            seed=seed + 3_959,
            excluded_prompts=tuple(task.prompt for task in training_tasks),
            excluded_task_ids=tuple(task.task_id for task in training_tasks),
        )
        proxy_tasks = task_battery(
            list(training_families),
            list(training_depths),
            proxy_per_cell,
            seed=seed + 7_919,
            excluded_prompts=tuple(
                task.prompt for task in (*training_tasks, *validation_tasks)
            ),
            excluded_task_ids=tuple(
                task.task_id for task in (*training_tasks, *validation_tasks)
            ),
        )
        proxy_manifest, proxy_manifest_sha256 = build_recurrence_task_manifest(proxy_tasks)

        def probe_progress(event: dict[str, Any]) -> None:
            progress.emit("behavioral_probe", **event)

        teaching_pairs, teaching_manifest = _capture_training_inventory(
            model,
            tokenizer,
            training_tasks,
            spec=spec,
            rank=min(2, lora_rank),
            layers=lora_layers,
            seed=seed,
            progress=progress,
        )
        validation_pairs, validation_manifest = _capture_training_inventory(
            model,
            tokenizer,
            validation_tasks,
            spec=spec,
            rank=min(2, lora_rank),
            layers=lora_layers,
            seed=seed + 3_959,
            progress=progress,
        )
        decode_phases = {site: "decode" for site in teaching_pairs}
        fitted = fit_verified_trajectory_inventory(
            teaching_pairs,
            rank=lora_rank,
            regularization=regularization,
            gain=gain,
            adapter_scale=1.0,
            site_phases=decode_phases,
            normalize_corrections=False,
        )
        transfer_diagnostic = evaluate_verified_trajectory_transfer(
            fitted,
            validation_pairs,
            training_pairs=teaching_pairs,
        )
        progress.emit(
            "transfer_diagnostic",
            **transfer_diagnostic["aggregate"],
        )
        teaching_pairs_path = out_dir / "private_teaching_pairs.npz"
        np.savez_compressed(
            teaching_pairs_path,
            **{
                f"training__{site.replace('.', '__')}__{kind}": matrix
                for site, pair in teaching_pairs.items()
                for kind, matrix in (("inputs", pair[0]), ("corrections", pair[1]))
            },
            **{
                f"validation__{site.replace('.', '__')}__{kind}": matrix
                for site, pair in validation_pairs.items()
                for kind, matrix in (("inputs", pair[0]), ("corrections", pair[1]))
            },
        )
        teaching_pairs_path.chmod(0o600)
        teaching_pairs_payload = teaching_pairs_path.read_bytes()
        teaching_pairs_artifact = {
            "sha256": _sha256_bytes(teaching_pairs_payload),
            "size_bytes": len(teaching_pairs_payload),
            "mode": "0600",
        }
        if stop_after_transfer_diagnostic:
            preflight_gates = {
                "base_checkpoint_immutable": (
                    base_before == full_weight_checkpoint_identity(model_path)
                ),
                "heldout_operator_better_than_zero": bool(
                    transfer_diagnostic["aggregate"]["better_than_zero_operator"]
                ),
                "heldout_operator_direction_positive": (
                    float(transfer_diagnostic["aggregate"]["cosine"]) > 0.0
                ),
            }
            preflight_body = {
                "schema": CANARY_SCHEMA,
                "mode": "transfer_diagnostic_only",
                "source_commit": source_commit,
                "source_bindings": source_bindings,
                "model_path": str(model_path),
                "model_identity": base_before,
                "execution_spec": spec.to_dict(),
                "execution_spec_sha256": spec.sha256,
                "private_teaching_manifest": teaching_manifest,
                "private_validation_manifest": validation_manifest,
                "private_teaching_pairs_artifact": teaching_pairs_artifact,
                "factor_receipts": {
                    site: dict(value.receipt) for site, value in fitted.items()
                },
                "transfer_diagnostic": transfer_diagnostic,
                "gates": preflight_gates,
                "preflight_admitted": all(preflight_gates.values()),
                "admitted": False,
                "claim_boundary": (
                    "heldout_internal_operator_transfer_only_not_behavioral_gain"
                ),
            }
            preflight_receipt = _write_receipt(
                out_dir / "receipt.json",
                preflight_body,
            )
            progress.emit(
                "complete",
                preflight_admitted=preflight_receipt["preflight_admitted"],
                **preflight_gates,
            )
            return preflight_receipt
        ordinary_before, untreated = build_paired_full_engine_probe_reports(
            model,
            tokenizer,
            proxy_tasks,
            model_path=model_path,
            spec=spec,
            adapter_sha256=adapter_before,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
            objective_program_enabled=False,
            verified_objective_teacher_enabled=False,
            progress_callback=probe_progress,
        )
        progress.emit(
            "baseline_complete",
            ordinary=_report_score(ordinary_before),
            untreated=_report_score(untreated),
            observations=untreated["total_observations"],
        )
        del model
        mx.synchronize()
        mx.clear_cache()
        model, tokenizer = load(str(model_path))
        coda_sites = attach_coda_policy_adapters_at_sites(
            model,
            tuple(fitted),
            lora_rank=lora_rank,
            initialization_seed=(seed ^ 0xC0DA) & 0xFFFFFFFF,
            lora_scale=1.0,
        )
        if set(coda_sites) != set(sites):
            raise RuntimeError("trajectory coda topology differs from capture topology")
        installation = install_verified_trajectory_inventory(
            model,
            fitted,
            expected_sites=coda_sites,
        )
        adapter_after = adapter_tensor_fingerprint(adapter_tensor_dict(model))
        progress.emit("trajectory_installed", adapter_sha256=adapter_after)
        ordinary_after, treatment = build_paired_full_engine_probe_reports(
            model,
            tokenizer,
            proxy_tasks,
            model_path=model_path,
            spec=spec,
            adapter_sha256=adapter_after,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
            objective_program_enabled=False,
            verified_objective_teacher_enabled=False,
            progress_callback=probe_progress,
        )
        treatment_gain = _report_score(treatment) > max(
            _report_score(untreated),
            _report_score(ordinary_after),
        )
        progress.emit(
            "treatment_complete",
            ordinary=_report_score(ordinary_after),
            untreated=_report_score(untreated),
            treatment=_report_score(treatment),
            strict_gain=treatment_gain,
        )
        lesion = None
        sham = None
        restored_after_lesion = True
        restored_after_sham = True
        if treatment_gain:
            with _zeroed_recurrence_adapter(model):
                lesion_sha256 = adapter_tensor_fingerprint(adapter_tensor_dict(model))
                _ordinary_lesion, lesion = build_paired_full_engine_probe_reports(
                    model,
                    tokenizer,
                    proxy_tasks,
                    model_path=model_path,
                    spec=spec,
                    adapter_sha256=lesion_sha256,
                    task_manifest_sha256=proxy_manifest_sha256,
                    seed=seed,
                    objective_program_enabled=False,
                    verified_objective_teacher_enabled=False,
                    progress_callback=probe_progress,
                )
            restored_after_lesion = (
                adapter_tensor_fingerprint(adapter_tensor_dict(model)) == adapter_after
            )
            with _permuted_recurrence_adapter(model):
                sham_sha256 = adapter_tensor_fingerprint(adapter_tensor_dict(model))
                _ordinary_sham, sham = build_paired_full_engine_probe_reports(
                    model,
                    tokenizer,
                    proxy_tasks,
                    model_path=model_path,
                    spec=spec,
                    adapter_sha256=sham_sha256,
                    task_manifest_sha256=proxy_manifest_sha256,
                    seed=seed,
                    objective_program_enabled=False,
                    verified_objective_teacher_enabled=False,
                    progress_callback=probe_progress,
                )
            restored_after_sham = (
                adapter_tensor_fingerprint(adapter_tensor_dict(model)) == adapter_after
            )
        adapter_path = out_dir / "adapter.safetensors"
        mx.save_safetensors(str(adapter_path), adapter_tensor_dict(model))
        adapter_path.chmod(0o600)
        adapter_payload = adapter_path.read_bytes()
        adapter_artifact = {
            "sha256": _sha256_bytes(adapter_payload),
            "size_bytes": len(adapter_payload),
            "mode": "0600",
        }

    gates = {
        "base_checkpoint_immutable": base_before == full_weight_checkpoint_identity(model_path),
        "adapter_mutated": adapter_before != adapter_after,
        "heldout_operator_better_than_zero": bool(
            transfer_diagnostic["aggregate"]["better_than_zero_operator"]
        ),
        "heldout_operator_direction_positive": (
            float(transfer_diagnostic["aggregate"]["cosine"]) > 0.0
        ),
        "ordinary_control_stable": _report_score(ordinary_before)
        == _report_score(ordinary_after),
        "teacher_free_treatment_strict_gain": treatment_gain,
        "treatment_beats_lesion": bool(
            lesion is not None and _report_score(treatment) > _report_score(lesion)
        ),
        "treatment_beats_norm_preserving_sham": bool(
            sham is not None and _report_score(treatment) > _report_score(sham)
        ),
        "lesion_restored_exactly": restored_after_lesion,
        "sham_restored_exactly": restored_after_sham,
    }
    reports = {
        "ordinary_before": ordinary_before,
        "untreated": untreated,
        "ordinary_after": ordinary_after,
        "treatment": treatment,
        "lesion": lesion,
        "sham": sham,
    }
    persisted_reports = {
        name: (
            None
            if report is None
            else validate_free_generation_report(json.loads(_canonical_bytes(report)))
        )
        for name, report in reports.items()
    }
    body = {
        "schema": CANARY_SCHEMA,
        "source_commit": source_commit,
        "source_bindings": source_bindings,
        "model_path": str(model_path),
        "model_identity": base_before,
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "configuration": {
            "seed": seed,
            "training_families": list(training_families),
            "training_depths": list(training_depths),
            "training_per_cell": training_per_cell,
            "validation_per_cell": validation_per_cell,
            "proxy_per_cell": proxy_per_cell,
            "lora_rank": lora_rank,
            "lora_layers": lora_layers,
            "regularization": regularization,
            "gain": gain,
        },
        "private_teaching_manifest": teaching_manifest,
        "private_validation_manifest": validation_manifest,
        "private_teaching_pairs_artifact": teaching_pairs_artifact,
        "transfer_diagnostic": transfer_diagnostic,
        "factor_receipts": {site: dict(value.receipt) for site, value in fitted.items()},
        "installation": installation,
        "adapter_artifact": adapter_artifact,
        "proxy_manifest": proxy_manifest,
        "reports": persisted_reports,
        "gates": gates,
        "admitted": all(gates.values()),
        "claim_boundary": (
            "bounded_teacher_free_persistent_gain_only_not_general_or_frontier_gain"
        ),
    }
    receipt = _write_receipt(out_dir / "receipt.json", body)
    progress.emit("complete", admitted=receipt["admitted"], **gates)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081003)
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    parser.add_argument("--training-families", default="boolean,modular")
    parser.add_argument("--training-depths", default="2,3")
    parser.add_argument("--training-per-cell", type=int, default=2)
    parser.add_argument(
        "--sample-complexity-levels",
        default="",
        help=(
            "Comma-separated nested training examples per family/depth cell. "
            "When set, run the fixed-hyperparameter multi-seed transfer gate "
            "instead of behavioral generation."
        ),
    )
    parser.add_argument("--validation-per-cell", type=int, default=1)
    parser.add_argument("--validation-cohorts", type=int, default=3)
    parser.add_argument("--proxy-per-cell", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--gain", type=float, default=0.25)
    parser.add_argument("--stop-after-transfer-diagnostic", action="store_true")
    args = parser.parse_args()
    families = tuple(part.strip() for part in args.training_families.split(",") if part.strip())
    try:
        depths = tuple(int(part.strip()) for part in args.training_depths.split(",") if part.strip())
        levels = tuple(
            int(part.strip())
            for part in args.sample_complexity_levels.split(",")
            if part.strip()
        )
        if levels:
            if args.stop_after_transfer_diagnostic:
                raise ValueError(
                    "sample-complexity mode already stops before behavioral generation"
                )
            receipt = run_sample_complexity_canary(
                model_path=args.model.expanduser().resolve(strict=True),
                out_dir=args.out_dir.expanduser().resolve(strict=False),
                seed=args.seed,
                memory_fraction=args.memory_fraction,
                training_families=families,
                training_depths=depths,
                training_per_cell_levels=levels,
                validation_per_cell=args.validation_per_cell,
                validation_cohort_count=args.validation_cohorts,
                lora_rank=args.lora_rank,
                lora_layers=args.lora_layers,
                regularization=args.regularization,
                gain=args.gain,
            )
        else:
            receipt = run_canary(
                model_path=args.model.expanduser().resolve(strict=True),
                out_dir=args.out_dir.expanduser().resolve(strict=False),
                seed=args.seed,
                memory_fraction=args.memory_fraction,
                training_families=families,
                training_depths=depths,
                training_per_cell=args.training_per_cell,
                validation_per_cell=args.validation_per_cell,
                proxy_per_cell=args.proxy_per_cell,
                lora_rank=args.lora_rank,
                lora_layers=args.lora_layers,
                regularization=args.regularization,
                gain=args.gain,
                stop_after_transfer_diagnostic=args.stop_after_transfer_diagnostic,
            )
    except BaseException as exc:  # noqa: BLE001 - preserve diagnostic process status
        print(f"trajectory canary failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, indent=2))
    accepted = receipt["admitted"]
    if receipt.get("mode") == "transfer_diagnostic_only":
        accepted = receipt.get("preflight_admitted")
    return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
