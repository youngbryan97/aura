#!/usr/bin/env python3
"""Run a bounded real-checkpoint canary for the generated-prefix RLC objective.

This is an engineering discriminator, not a reasoning-gain experiment. It
loads a small MLX checkpoint, attaches the production recurrent adapter
topology, measures a held-out row, performs a bounded number of updates, and
remeasures the same row. The durable receipt proves that adapter tensors moved,
the base checkpoint files did not, generated-prefix evidence validates, and all
losses remained finite. It never promotes or fuses an adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
)
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    ExactAdjointTrajectoryConfig,
)
from core.learning.recurrence_native_objective_v5 import (  # noqa: E402
    GeneratedRollinSelectionConfig,
    derive_rollin_seed,
)
from core.learning.recurrence_native_objective_v6 import (  # noqa: E402
    RECURRENCE_NATIVE_OBJECTIVE_V6_SCHEMA,
    BranchSpecializationConfig,
    branch_specialization_live_path_loss,
    branch_specialization_live_path_value_and_grad,
    generated_rollin_specialization_loss,
    generated_rollin_specialization_value_and_grad,
    validate_branch_specialization_receipt,
    validate_generated_rollin_specialization_receipt,
)
from core.learning.recurrent_behavioral_probe import (  # noqa: E402
    build_behavioral_probe_report as _free_generation_report,
)
from core.learning.recurrent_behavioral_probe import (  # noqa: E402
    build_ordinary_decode_probe_report,
    free_generation_sampling_config,
    paired_generation_seed,
)
from core.learning.recurrent_behavioral_probe import (  # noqa: E402
    tokenize_task as _tokenize,
)
from core.learning.recurrent_checkpoint_admission import (  # noqa: E402
    build_checkpoint_behavioral_admission,
    build_recurrence_task_manifest,
    validate_checkpoint_behavioral_admission,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    attach_recurrent_policy_adapters,
    recurrent_policy_sha256,
)
from core.learning.recurrent_sft_execution import (  # noqa: E402
    adapter_tensor_dict,
    adapter_tensor_fingerprint,
)
from core.runtime.atomic_writer import atomic_append_text, atomic_write_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.generated_rollin_objective_canary.v6"
PROGRESS_SCHEMA: Final = "aura.generated_rollin_objective_canary.progress.v1"
SOURCE_PATHS: Final = (
    "core/learning/recurrence_native_objective_v2.py",
    "core/learning/recurrence_native_objective_v5.py",
    "core/learning/recurrence_native_objective_v4.py",
    "core/learning/recurrence_native_objective_v6.py",
    "core/learning/role_conditioned_lora.py",
    "core/learning/recurrent_grpo.py",
    "core/learning/recurrent_checkpoint_admission.py",
    "core/learning/recurrent_behavioral_probe.py",
    "core/learning/recurrence_curriculum.py",
    "core/brain/llm/latent_cortex/recurrence_adapter.py",
    "core/learning/depth_conditioned_lora.py",
    "tools/run_generated_rollin_objective_canary.py",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


class _ProgressLedger:
    """Durable, externally observable phase evidence for a bounded canary."""

    def __init__(self, out_dir: Path, *, started: float, source_commit: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=False)
        self._out_dir = out_dir
        self._started = started
        self._source_commit = source_commit
        self._sequence = 0

    @classmethod
    def resume(cls, out_dir: Path, latest: dict[str, Any]) -> _ProgressLedger:
        ledger = cls.__new__(cls)
        ledger._out_dir = out_dir
        ledger._started = time.time() - float(latest["elapsed_s"])
        ledger._source_commit = str(latest["source_commit"])
        ledger._sequence = int(latest["sequence"])
        return ledger

    def emit(
        self,
        phase: str,
        *,
        status: str = "running",
        **details: Any,
    ) -> dict[str, Any]:
        if not phase or status not in {"running", "completed", "failed"}:
            raise ValueError("progress phase/status is invalid")
        self._sequence += 1
        body = {
            "schema": PROGRESS_SCHEMA,
            "sequence": self._sequence,
            "phase": phase,
            "status": status,
            "pid": os.getpid(),
            "source_commit": self._source_commit,
            "wall_time_epoch_s": time.time(),
            "elapsed_s": time.time() - self._started,
            "details": details,
        }
        event = {
            **body,
            "event_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
        }
        payload = _canonical_json_bytes(event)
        atomic_append_text(
            self._out_dir / "progress.jsonl",
            payload.decode("ascii") + "\n",
        )
        atomic_write_bytes(self._out_dir / "progress.json", payload, mode=0o600)
        detail = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
        print(
            f"[canary] seq={self._sequence} phase={phase} status={status} {detail}".rstrip(),
            file=sys.stderr,
            flush=True,
        )
        return event


def _append_terminal_failure(out_dir: Path, exc: BaseException) -> None:
    """Append failure evidence only when this invocation created its ledger."""

    latest_path = out_dir / "progress.json"
    if not latest_path.is_file():
        return
    try:
        latest = json.loads(latest_path.read_text(encoding="ascii"))
        ledger = _ProgressLedger.resume(out_dir, latest)
        ledger.emit(
            str(latest.get("phase") or "canary"),
            status="failed",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
    except Exception as ledger_exc:  # noqa: BLE001 - preserve original failure
        print(
            f"[canary] failed to persist terminal failure: {ledger_exc}",
            file=sys.stderr,
            flush=True,
        )


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_state() -> tuple[str, dict[str, dict[str, Any]]]:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("canary requires a clean source worktree")
    head = _git("rev-parse", "HEAD")
    origin_main = _git("rev-parse", "origin/main")
    if head != origin_main:
        raise RuntimeError("canary source commit is not published on origin/main")
    bindings: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        payload = path.read_bytes()
        committed = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "show", f"{head}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if payload != committed:
            raise RuntimeError(f"canary source differs from commit: {relative}")
        bindings[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return head, bindings


def _evaluate(
    model: Any,
    row: dict[str, Any],
    *,
    spec: RLCExecutionSpec,
    generated_config: GeneratedRollinSelectionConfig,
    specialization_config: BranchSpecializationConfig,
    trajectory_config: ExactAdjointTrajectoryConfig,
    campaign_seed: int,
) -> dict[str, Any]:
    seed = derive_rollin_seed(
        campaign_seed=campaign_seed,
        phase="validation",
        example_id=row["task_id"],
        sample_ordinal=0,
        execution_spec_sha256=spec.sha256,
    )
    evaluation = generated_rollin_specialization_loss(
        model,
        row["prompt_tokens"],
        row["answer_tokens"],
        spec=spec,
        base_seed=seed,
        generated_config=generated_config,
        specialization_config=specialization_config,
        trajectory_config=trajectory_config,
        trajectory_policy_sha256=recurrent_policy_sha256(model, spec),
    )
    receipt = validate_generated_rollin_specialization_receipt(evaluation.receipt())
    return {
        "task_id": row["task_id"],
        "loss": evaluation.value,
        "lexical_loss": evaluation.generated.value,
        "specialization_loss": evaluation.specialization.value,
        "trajectory_loss": evaluation.trajectory.value,
        "branch_separations": list(evaluation.specialization.separations),
        "branch_values": list(evaluation.branch_values),
        "branch_weights": list(evaluation.branch_weights),
        "rollin_base_seed": seed,
        "objective_receipt": receipt,
    }


def _branch_separations(
    model: Any,
    row: dict[str, Any],
    *,
    spec: RLCExecutionSpec,
) -> list[float]:
    import mlx.core as mx

    from core.learning.recurrence_native_objective_v2 import live_path_forward
    from core.learning.recurrence_native_objective_v4 import pairwise_separations

    forward = live_path_forward(
        model,
        row["prompt_tokens"],
        row["answer_tokens"],
        spec=spec,
    )
    values = pairwise_separations(forward, comm_slot=spec.comm_slot)
    mx.eval(values)
    result = [float(value) for value in values]
    del forward, values
    mx.clear_cache()
    return result


def _branch_specialization_gates(
    loss_trail: list[dict[str, Any]],
    separation_after: list[float],
) -> dict[str, bool]:
    return {
        "branch_generated_prefix_distinct": bool(
            loss_trail
            and all(
                len(
                    {
                        branch["generated_tokens_sha256"]
                        for branch in entry["objective_receipt"]["generated_receipt"]["branches"]
                    }
                )
                == len(entry["objective_receipt"]["generated_receipt"]["branches"])
                for entry in loss_trail
            )
        ),
        "branch_state_specialized": bool(separation_after and min(separation_after) >= 0.30),
    }


def _warmup_target_reached(specialization_loss: Any) -> bool:
    return (
        not isinstance(specialization_loss, bool)
        and isinstance(specialization_loss, (int, float))
        and math.isfinite(float(specialization_loss))
        and float(specialization_loss) <= 1e-6
    )


def _cyclic_training_row(
    rows: list[dict[str, Any]],
    *,
    one_based_step: int,
) -> dict[str, Any]:
    if not rows or type(one_based_step) is not int or one_based_step < 1:
        raise ValueError("training-row cycle coordinates are invalid")
    return rows[(one_based_step - 1) % len(rows)]


def _paired_generation_seed(
    campaign_seed: int,
    task_ordinal: int,
    task_id: str,
    depth: int,
) -> int:
    return paired_generation_seed(campaign_seed, task_ordinal, task_id, depth)


def _free_generation_sampling_config() -> Any:
    return free_generation_sampling_config()


@contextmanager
def _permuted_coda_sham(model: Any) -> Iterator[dict[str, Any]]:
    """Apply a norm-preserving semantic sham and restore exact coda tensors."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedCodaLoRALinear

    snapshots: list[tuple[Any, Any]] = []
    sites: list[str] = []
    layers = getattr(getattr(model, "model", None), "layers", None) or []
    for layer_index, layer in enumerate(layers):
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for target in ("o_proj", "down_proj"):
                projection = getattr(parent, target, None)
                if not isinstance(projection, ScopedCodaLoRALinear):
                    continue
                snapshots.append((projection, projection.lora_b))
                sites.append(f"model.layers.{layer_index}.{parent_name}.{target}")
                width = int(projection.lora_b.shape[-1])
                permutation = mx.array(list(reversed(range(width))))
                projection.lora_b = projection.lora_b[:, permutation]
    if not snapshots:
        raise RuntimeError("coda sham found no coda-scoped projections")
    mx.eval(model.trainable_parameters())
    try:
        yield {
            "method": "reverse_output_basis_permutation_v1",
            "sites": sites,
            "norm_preserved": True,
        }
    finally:
        for projection, original in snapshots:
            projection.lora_b = original
        mx.eval(model.trainable_parameters())


@contextmanager
def _permuted_recurrence_sham(model: Any) -> Iterator[dict[str, Any]]:
    """Permute every learned recurrent output basis, then restore exactly."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    snapshots: list[tuple[Any, str, Any]] = []
    sites: list[str] = []
    layers = getattr(getattr(model, "model", None), "layers", None) or []
    for layer_index, layer in enumerate(layers):
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for target in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ):
                projection = getattr(parent, target, None)
                if not isinstance(projection, ScopedLoRALinear):
                    continue
                sites.append(f"model.layers.{layer_index}.{parent_name}.{target}")
                tensor_names = ["lora_b"]
                tensor_names.extend(
                    name
                    for name in ("depth_b", "role_b")
                    if hasattr(projection, name)
                )
                for name in tensor_names:
                    value = getattr(projection, name)
                    if isinstance(value, list):
                        for index, tensor in enumerate(value):
                            snapshots.append((projection, f"{name}.{index}", tensor))
                            width = int(tensor.shape[-1])
                            permutation = mx.array(list(reversed(range(width))))
                            value[index] = tensor[:, permutation]
                    else:
                        snapshots.append((projection, name, value))
                        width = int(value.shape[-1])
                        permutation = mx.array(list(reversed(range(width))))
                        setattr(projection, name, value[:, permutation])
    if not snapshots:
        raise RuntimeError("recurrent sham found no recurrence-scoped projections")
    mx.eval(model.trainable_parameters())
    try:
        yield {
            "method": "reverse_output_basis_permutation_v1",
            "sites": sites,
            "norm_preserved": True,
        }
    finally:
        for projection, name, original in reversed(snapshots):
            if "." in name:
                collection_name, index_text = name.split(".", 1)
                getattr(projection, collection_name)[int(index_text)] = original
            else:
                setattr(projection, name, original)
        mx.eval(model.trainable_parameters())


def run_canary(
    *,
    model_path: Path,
    out_dir: Path,
    steps: int,
    seed: int,
    memory_fraction: float,
    student_forcing_probability: float,
    sampling_temperature: float,
    specialization_weight: float,
    warmup_steps: int,
    warmup_learning_rate: float,
    joint_learning_rate: float,
    lora_rank: int = 2,
    lora_layers: int = 2,
    lora_targets: Sequence[str] = ("o_proj",),
    lora_layer_placement: str = "late",
    coda_lora_layers: int = 2,
    coda_lora_targets: Sequence[str] = ("o_proj", "down_proj"),
    training_families: Sequence[str] = ("boolean", "modular"),
    training_depths: Sequence[int] = (2,),
    training_per_cell: int = 2,
    proxy_per_cell: int = 2,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load

    from core.learning.recurrence_curriculum import (
        PROCESS_SUPERVISION_SCHEMA,
        task_battery,
    )

    if type(steps) is not int or not 1 <= steps <= 256:
        raise ValueError("steps must be inside [1, 256]")
    if type(warmup_steps) is not int or not 1 <= warmup_steps <= 64:
        raise ValueError("warmup_steps must be inside [1, 64]")
    if (
        type(lora_rank) is not int
        or not 1 <= lora_rank <= 64
        or type(lora_layers) is not int
        or not 1 <= lora_layers <= 64
        or not isinstance(lora_targets, Sequence)
        or isinstance(lora_targets, (str, bytes, bytearray))
        or not lora_targets
        or lora_layer_placement not in {"early", "distributed", "late"}
        or type(coda_lora_layers) is not int
        or not 0 <= coda_lora_layers <= 64
        or not isinstance(coda_lora_targets, Sequence)
        or isinstance(coda_lora_targets, (str, bytes, bytearray))
        or bool(coda_lora_layers) != bool(coda_lora_targets)
        or not isinstance(training_families, Sequence)
        or isinstance(training_families, (str, bytes, bytearray))
        or not training_families
        or not isinstance(training_depths, Sequence)
        or isinstance(training_depths, (str, bytes, bytearray))
        or not training_depths
        or type(training_per_cell) is not int
        or not 1 <= training_per_cell <= 128
        or type(proxy_per_cell) is not int
        or not 1 <= proxy_per_cell <= 128
    ):
        raise ValueError("canary topology or curriculum configuration is invalid")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be inside [0, 2^63-1]")
    for name, value in (
        ("warmup_learning_rate", warmup_learning_rate),
        ("joint_learning_rate", joint_learning_rate),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 1e-6 <= float(value) <= 1e-2
        ):
            raise ValueError(f"{name} must be inside [1e-6, 1e-2]")
    started = time.time()
    if out_dir.exists():
        raise FileExistsError(f"canary output directory already exists: {out_dir}")
    source_commit, source_bindings = _source_state()
    progress = _ProgressLedger(
        out_dir,
        started=started,
        source_commit=source_commit,
    )
    progress.emit(
        "source_bound",
        source_paths=len(source_bindings),
        model_path=str(model_path),
        seed=seed,
        warmup_steps=warmup_steps,
        joint_steps=steps,
    )
    base_before = full_weight_checkpoint_identity(model_path)
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=2,
        exchange_interval=1,
    )
    objective_config = GeneratedRollinSelectionConfig(
        student_forcing_probability=student_forcing_probability,
        sampling_temperature=sampling_temperature,
        branch_softmin_temperature=0.5,
    )
    specialization_config = BranchSpecializationConfig(
        weight=specialization_weight,
        target_separation=0.30,
    )
    trajectory_config = ExactAdjointTrajectoryConfig(
        probe_steps=(1, 2),
        improvement_weight=1.0,
        improvement_margin=0.05,
    )
    with (
        standalone_model_lane(
            owner_id=f"generated-rollin-canary:{out_dir.name}",
            model_path=str(model_path),
            purpose="training",
            preemptible=False,
            metadata={
                "tool": "run_generated_rollin_objective_canary",
                "source_commit": source_commit,
            },
        ),
        mlx_memory_envelope(
            fraction=memory_fraction,
            restore_limits_on_exit=True,
        ),
    ):
        progress.emit("model_load", model_path=str(model_path))
        model, tokenizer = load(str(model_path))
        attach_recurrent_policy_adapters(
            model,
            spec,
            lora_rank=lora_rank,
            lora_layers=lora_layers,
            lora_targets=tuple(lora_targets),
            initialization_seed=(seed ^ 0x51F7A11) & 0xFFFFFFFF,
            lora_scale=1.0,
            lora_layer_placement=lora_layer_placement,
            depth_conditioned_steps=spec.recurrent_steps,
            role_conditioned_branches=len(spec.branch_roles),
            coda_lora_layers=coda_lora_layers,
            coda_lora_targets=tuple(coda_lora_targets),
        )
        progress.emit(
            "model_ready",
            recurrent_adapter_layers=lora_layers,
            recurrent_adapter_targets=list(lora_targets),
            recurrent_adapter_placement=lora_layer_placement,
            coda_adapter_layers=coda_lora_layers,
            coda_adapter_targets=list(coda_lora_targets),
        )
        adapter_before = adapter_tensor_fingerprint(adapter_tensor_dict(model))
        training_tasks = task_battery(
            list(training_families),
            list(training_depths),
            training_per_cell,
            seed=seed,
        )
        training_rows = []
        training_target_manifest = []
        for task in training_tasks:
            prompt_tokens, answer_tokens = _tokenize(
                tokenizer,
                task.prompt,
                task.training_target,
            )
            training_rows.append(
                {
                    "task_id": task.task_id,
                    "prompt_tokens": prompt_tokens,
                    "answer_tokens": answer_tokens,
                }
            )
            training_target_manifest.append(
                {
                    "task_id": task.task_id,
                    "target_sha256": hashlib.sha256(
                        task.training_target.encode("utf-8")
                    ).hexdigest(),
                    "target_bytes": len(task.training_target.encode("utf-8")),
                }
            )
        proxy_tasks = task_battery(
            list(training_families),
            list(training_depths),
            proxy_per_cell,
            seed=seed + 7_919,
            excluded_prompts=tuple(task.prompt for task in training_tasks),
            excluded_task_ids=tuple(task.task_id for task in training_tasks),
        )
        validation_task = proxy_tasks[0]
        validation_prompt_tokens, validation_answer_tokens = _tokenize(
            tokenizer,
            validation_task.prompt,
            validation_task.training_target,
        )
        validation_row = {
            "task_id": validation_task.task_id,
            "prompt_tokens": validation_prompt_tokens,
            "answer_tokens": validation_answer_tokens,
        }
        proxy_manifest, proxy_manifest_sha256 = build_recurrence_task_manifest(proxy_tasks)

        def probe_progress(payload: dict[str, Any]) -> None:
            progress.emit("free_generation", **payload)

        free_generation_before = _free_generation_report(
            model,
            tokenizer,
            proxy_tasks,
            spec=spec,
            arm="initial_adapter",
            adapter_sha256=adapter_before,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
            progress_callback=probe_progress,
        )
        progress.emit(
            "initial_free_generation_complete",
            total_correct=free_generation_before["total_correct"],
            total_observations=free_generation_before["total_observations"],
        )
        progress.emit("heldout_objective_before")
        before = _evaluate(
            model,
            validation_row,
            spec=spec,
            generated_config=objective_config,
            specialization_config=specialization_config,
            trajectory_config=trajectory_config,
            campaign_seed=seed,
        )
        separation_before = _branch_separations(
            model,
            validation_row,
            spec=spec,
        )
        progress.emit(
            "heldout_objective_before_complete",
            loss=before["loss"],
            lexical_loss=before["lexical_loss"],
            trajectory_loss=before["trajectory_loss"],
            branch_separations=separation_before,
        )
        warmup_optimizer = optim.AdamW(
            learning_rate=warmup_learning_rate,
            weight_decay=0.0,
        )
        warmup_optimizer.init(model.trainable_parameters())
        warmup_trail: list[dict[str, Any]] = []
        for warmup_step in range(1, warmup_steps + 1):
            training_row = _cyclic_training_row(
                training_rows,
                one_based_step=warmup_step,
            )
            result = branch_specialization_live_path_value_and_grad(
                model,
                training_row["prompt_tokens"],
                spec=spec,
                config=specialization_config,
            )
            structural_receipt = validate_branch_specialization_receipt(result.evaluation.receipt())
            optimizer_before = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            warmup_optimizer.update(model, result.gradients)
            mx.eval(model.trainable_parameters(), warmup_optimizer.state)
            post_update = branch_specialization_live_path_loss(
                model,
                training_row["prompt_tokens"],
                spec=spec,
                config=specialization_config,
            )
            warmup_trail.append(
                {
                    "step": warmup_step,
                    "task_id": training_row["task_id"],
                    "loss_before": result.value,
                    "separations_before": list(result.evaluation.separations),
                    "separations_after": list(post_update.separations),
                    "objective_receipt": structural_receipt,
                    "adapter_before_sha256": optimizer_before,
                    "adapter_after_sha256": adapter_tensor_fingerprint(adapter_tensor_dict(model)),
                }
            )
            progress.emit(
                "specialization_warmup",
                step=warmup_step,
                total_steps=warmup_steps,
                task_id=training_row["task_id"],
                loss_before=result.value,
                separations_after=list(post_update.separations),
            )
        warmup_validation = _evaluate(
            model,
            validation_row,
            spec=spec,
            generated_config=objective_config,
            specialization_config=specialization_config,
            trajectory_config=trajectory_config,
            campaign_seed=seed,
        )
        progress.emit(
            "specialization_warmup_complete",
            validation_loss=warmup_validation["loss"],
            specialization_loss=warmup_validation["specialization_loss"],
        )
        if not _warmup_target_reached(warmup_validation["specialization_loss"]):
            progress.emit(
                "warmup_gate_rejected",
                observed_specialization_loss=warmup_validation["specialization_loss"],
                required_max_specialization_loss=1e-6,
                completed_steps=warmup_steps,
            )
            raise RuntimeError(
                "specialization warmup did not reach its preregistered target; "
                "later phases cannot make this canary admissible"
            )
        # Reset momentum when the structural constraint is met. Continuing an
        # Adam trajectory after the hinge reaches zero overshoots the target.
        optimizer = optim.AdamW(
            learning_rate=joint_learning_rate,
            weight_decay=0.0,
        )
        optimizer.init(model.trainable_parameters())
        loss_trail: list[dict[str, Any]] = []
        for step in range(1, steps + 1):
            training_row = _cyclic_training_row(
                training_rows,
                one_based_step=step,
            )
            rollin_seed = derive_rollin_seed(
                campaign_seed=seed,
                phase="train",
                example_id=training_row["task_id"],
                sample_ordinal=step,
                execution_spec_sha256=spec.sha256,
            )
            result = generated_rollin_specialization_value_and_grad(
                model,
                training_row["prompt_tokens"],
                training_row["answer_tokens"],
                spec=spec,
                base_seed=rollin_seed,
                generated_config=objective_config,
                specialization_config=specialization_config,
                trajectory_config=trajectory_config,
                trajectory_policy_sha256=recurrent_policy_sha256(model, spec),
            )
            receipt = validate_generated_rollin_specialization_receipt(result.evaluation.receipt())
            optimizer.update(model, result.gradients)
            mx.eval(model.trainable_parameters(), optimizer.state)
            loss_trail.append(
                {
                    "step": step,
                    "task_id": training_row["task_id"],
                    "loss": result.value,
                    "lexical_loss": result.evaluation.generated.value,
                    "specialization_loss": result.evaluation.specialization.value,
                    "trajectory_loss": result.evaluation.trajectory.value,
                    "branch_separations": list(result.evaluation.specialization.separations),
                    "branch_values": list(result.branch_values),
                    "branch_weights": list(result.branch_weights),
                    "rollin_base_seed": rollin_seed,
                    "objective_receipt": receipt,
                    "adapter_sha256": adapter_tensor_fingerprint(adapter_tensor_dict(model)),
                }
            )
            progress.emit(
                "joint_training",
                step=step,
                total_steps=steps,
                task_id=training_row["task_id"],
                loss=result.value,
                lexical_loss=result.evaluation.generated.value,
                specialization_loss=result.evaluation.specialization.value,
                trajectory_loss=result.evaluation.trajectory.value,
            )
        progress.emit("heldout_objective_after")
        after = _evaluate(
            model,
            validation_row,
            spec=spec,
            generated_config=objective_config,
            specialization_config=specialization_config,
            trajectory_config=trajectory_config,
            campaign_seed=seed,
        )
        separation_after = _branch_separations(
            model,
            validation_row,
            spec=spec,
        )
        adapter = adapter_tensor_dict(model)
        adapter_after = adapter_tensor_fingerprint(adapter)
        progress.emit(
            "heldout_objective_after_complete",
            loss=after["loss"],
            lexical_loss=after["lexical_loss"],
            trajectory_loss=after["trajectory_loss"],
            branch_separations=separation_after,
            adapter_sha256=adapter_after,
        )
        free_generation_after = _free_generation_report(
            model,
            tokenizer,
            proxy_tasks,
            spec=spec,
            arm="trained_adapter",
            adapter_sha256=adapter_after,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
            progress_callback=probe_progress,
        )
        progress.emit(
            "trained_free_generation_complete",
            total_correct=free_generation_after["total_correct"],
            total_observations=free_generation_after["total_observations"],
        )
        from core.brain.llm.latent_cortex.recurrence_adapter import (
            coda_adapter_disabled,
            recurrence_adapter_disabled,
        )

        with recurrence_adapter_disabled():
            free_generation_recurrence_lesion = _free_generation_report(
                model,
                tokenizer,
                proxy_tasks,
                spec=spec,
                arm="trained_adapter_lesion",
                adapter_sha256=adapter_after,
                task_manifest_sha256=proxy_manifest_sha256,
                seed=seed,
                progress_callback=probe_progress,
            )
        progress.emit(
            "recurrence_lesion_complete",
            total_correct=free_generation_recurrence_lesion["total_correct"],
            total_observations=free_generation_recurrence_lesion["total_observations"],
        )
        with _permuted_recurrence_sham(model) as recurrence_sham:
            recurrence_sham_sha256 = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            free_generation_recurrence_sham = _free_generation_report(
                model,
                tokenizer,
                proxy_tasks,
                spec=spec,
                arm="trained_adapter_sham",
                adapter_sha256=recurrence_sham_sha256,
                task_manifest_sha256=proxy_manifest_sha256,
                seed=seed,
                progress_callback=probe_progress,
            )
        recurrence_restore_exact = (
            adapter_tensor_fingerprint(adapter_tensor_dict(model)) == adapter_after
        )
        progress.emit(
            "recurrence_sham_complete",
            total_correct=free_generation_recurrence_sham["total_correct"],
            total_observations=free_generation_recurrence_sham["total_observations"],
            restored_exactly=recurrence_restore_exact,
        )
        free_generation_coda_lesion = None
        free_generation_coda_sham = None
        coda_sham = None
        coda_restore_exact = True
        if coda_lora_layers:
            with coda_adapter_disabled():
                free_generation_coda_lesion = _free_generation_report(
                    model,
                    tokenizer,
                    proxy_tasks,
                    spec=spec,
                    arm="trained_coda_lesion",
                    adapter_sha256=adapter_after,
                    task_manifest_sha256=proxy_manifest_sha256,
                    seed=seed,
                    progress_callback=probe_progress,
                )
            progress.emit(
                "coda_lesion_complete",
                total_correct=free_generation_coda_lesion["total_correct"],
                total_observations=free_generation_coda_lesion["total_observations"],
            )
            with _permuted_coda_sham(model) as coda_sham:
                coda_sham_sha256 = adapter_tensor_fingerprint(adapter_tensor_dict(model))
                free_generation_coda_sham = _free_generation_report(
                    model,
                    tokenizer,
                    proxy_tasks,
                    spec=spec,
                    arm="trained_coda_sham",
                    adapter_sha256=coda_sham_sha256,
                    task_manifest_sha256=proxy_manifest_sha256,
                    seed=seed,
                    progress_callback=probe_progress,
                )
            coda_restore_exact = (
                adapter_tensor_fingerprint(adapter_tensor_dict(model)) == adapter_after
            )
            progress.emit(
                "coda_sham_complete",
                total_correct=free_generation_coda_sham["total_correct"],
                total_observations=free_generation_coda_sham["total_observations"],
                restored_exactly=coda_restore_exact,
            )
        # The vanilla floor. Measured on the same weights and the same tasks,
        # after training, because the base weights never move -- only the
        # adapter does, and the ordinary path does not read it.
        ordinary_decode_report = build_ordinary_decode_probe_report(
            model,
            tokenizer,
            proxy_tasks,
            spec=spec,
            adapter_sha256=adapter_after,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
            progress_callback=probe_progress,
        )
        progress.emit(
            "ordinary_decode_complete",
            total_correct=ordinary_decode_report["total_correct"],
            total_observations=ordinary_decode_report["total_observations"],
        )
        behavioral_admission = build_checkpoint_behavioral_admission(
            initial_report=free_generation_before,
            trained_report=free_generation_after,
            task_manifest=proxy_manifest,
            ordinary_decode_report=ordinary_decode_report,
        )
        validate_checkpoint_behavioral_admission(
            behavioral_admission,
            initial_report=free_generation_before,
            trained_report=free_generation_after,
            task_manifest=proxy_manifest,
            ordinary_decode_report=ordinary_decode_report,
        )
        mx.save_safetensors(str(out_dir / "adapter.safetensors"), adapter)
        progress.emit("adapter_saved", adapter_sha256=adapter_after)

    base_after = full_weight_checkpoint_identity(model_path)
    finite_losses = [
        before["loss"],
        warmup_validation["loss"],
        after["loss"],
        *(entry["loss_before"] for entry in warmup_trail),
        *(entry["loss"] for entry in loss_trail),
    ]
    gates = {
        "base_checkpoint_immutable": base_before == base_after,
        "adapter_mutated": adapter_before != adapter_after,
        "losses_finite": all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in finite_losses
        ),
        "generated_prefix_exercised": all(
            any(
                branch["student_forced_positions"]
                for branch in entry["objective_receipt"]["generated_receipt"]["branches"]
            )
            for entry in loss_trail
        ),
        "branch_credit_normalized": all(
            math.isclose(
                sum(entry["branch_weights"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for entry in loss_trail
        ),
        "warmup_target_reached": bool(
            warmup_trail
            and _warmup_target_reached(warmup_validation["specialization_loss"])
        ),
        **_branch_specialization_gates(loss_trail, separation_after),
        "heldout_lexical_non_regression": after["lexical_loss"] <= before["lexical_loss"] + 1e-6,
        "heldout_depth_improvement_non_regression": after["trajectory_loss"]
        <= before["trajectory_loss"] + 1e-6,
        "heldout_free_generation_strict_gain": behavioral_admission["admitted"],
        "recurrence_causal_contribution": (
            free_generation_after["total_correct"]
            > free_generation_recurrence_lesion["total_correct"]
        ),
        "recurrence_outperforms_norm_preserving_sham": (
            free_generation_after["total_correct"]
            > free_generation_recurrence_sham["total_correct"]
        ),
        "recurrence_sham_restored_exactly": recurrence_restore_exact,
    }
    if coda_lora_layers:
        assert free_generation_coda_lesion is not None
        assert free_generation_coda_sham is not None
        gates.update(
            {
                "coda_causal_contribution": (
                    free_generation_after["total_correct"]
                    > free_generation_coda_lesion["total_correct"]
                ),
                "coda_outperforms_norm_preserving_sham": (
                    free_generation_after["total_correct"]
                    > free_generation_coda_sham["total_correct"]
                ),
                "coda_sham_restored_exactly": coda_restore_exact,
            }
        )
    body = {
        "schema": CANARY_SCHEMA,
        "objective_schema": RECURRENCE_NATIVE_OBJECTIVE_V6_SCHEMA,
        "process_supervision_schema": PROCESS_SUPERVISION_SCHEMA,
        "source_commit": source_commit,
        "source_bindings": source_bindings,
        "model_path": str(model_path),
        "base_before": base_before,
        "base_after": base_after,
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "objective_config": {
            "generated": objective_config.to_dict(),
            "specialization": specialization_config.to_dict(),
            "trajectory": trajectory_config.to_dict(),
            "lora_rank": lora_rank,
            "lora_layers": lora_layers,
            "lora_targets": list(lora_targets),
            "lora_layer_placement": lora_layer_placement,
            "coda_lora_layers": coda_lora_layers,
            "coda_lora_targets": list(coda_lora_targets),
            "training_families": list(training_families),
            "training_depths": list(training_depths),
            "training_per_cell": training_per_cell,
            "proxy_per_cell": proxy_per_cell,
            "warmup_steps": warmup_steps,
            "warmup_learning_rate": float(warmup_learning_rate),
            "joint_learning_rate": float(joint_learning_rate),
        },
        "seed": seed,
        "steps": steps,
        "adapter_before_sha256": adapter_before,
        "adapter_after_sha256": adapter_after,
        "training_task_ids": [task.task_id for task in training_tasks],
        "training_target_manifest": training_target_manifest,
        "validation_target_sha256": hashlib.sha256(
            validation_task.training_target.encode("utf-8")
        ).hexdigest(),
        "validation_task_id": validation_row["task_id"],
        "proxy_task_manifest": proxy_manifest,
        "proxy_task_manifest_sha256": proxy_manifest_sha256,
        "free_generation_before": free_generation_before,
        "free_generation_after": free_generation_after,
        "free_generation_recurrence_lesion": free_generation_recurrence_lesion,
        "free_generation_recurrence_sham": free_generation_recurrence_sham,
        "recurrence_sham": recurrence_sham,
        "free_generation_coda_lesion": free_generation_coda_lesion,
        "free_generation_coda_sham": free_generation_coda_sham,
        "coda_sham": coda_sham,
        "checkpoint_behavioral_admission": behavioral_admission,
        "validation_before": before,
        "branch_separation_before": separation_before,
        "warmup_trail": warmup_trail,
        "validation_after_warmup": warmup_validation,
        "loss_trail": loss_trail,
        "validation_after": after,
        "branch_separation_after": separation_after,
        "validation_loss_delta": after["loss"] - before["loss"],
        "validation_lexical_loss_delta": (after["lexical_loss"] - before["lexical_loss"]),
        "gates": gates,
        "passed": all(gates.values()),
        "claim_state": {
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "promotion_allowed": False,
            "fusion_allowed": False,
            "resident_campaign_admitted": False,
        },
        "elapsed_s": time.time() - started,
    }
    receipt = {
        **body,
        "receipt_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
    }
    atomic_write_bytes(
        out_dir / "receipt.json",
        _canonical_json_bytes(receipt),
        mode=0o600,
    )
    progress.emit(
        "canary_complete",
        status="completed",
        passed=receipt["passed"],
        receipt_sha256=receipt["receipt_sha256"],
        failed_gates=sorted(name for name, passed in gates.items() if not passed),
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026080207)
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    parser.add_argument("--student-forcing-probability", type=float, default=0.5)
    parser.add_argument("--sampling-temperature", type=float, default=0.8)
    parser.add_argument("--specialization-weight", type=float, default=8.0)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--warmup-learning-rate", type=float, default=1e-3)
    parser.add_argument("--joint-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-layers", type=int, default=2)
    parser.add_argument("--lora-targets", default="o_proj")
    parser.add_argument(
        "--lora-layer-placement",
        choices=("early", "distributed", "late"),
        default="late",
    )
    parser.add_argument("--coda-lora-layers", type=int, default=2)
    parser.add_argument("--coda-lora-targets", default="o_proj,down_proj")
    parser.add_argument("--training-families", default="boolean,modular")
    parser.add_argument("--training-depths", default="2")
    parser.add_argument("--training-per-cell", type=int, default=2)
    parser.add_argument("--proxy-per-cell", type=int, default=2)
    args = parser.parse_args()
    lora_targets = tuple(filter(None, (part.strip() for part in args.lora_targets.split(","))))
    coda_targets = tuple(
        filter(None, (part.strip() for part in args.coda_lora_targets.split(",")))
    )
    if args.coda_lora_layers == 0:
        coda_targets = ()
    training_families = tuple(
        filter(None, (part.strip() for part in args.training_families.split(",")))
    )
    try:
        training_depths = tuple(
            int(part.strip())
            for part in args.training_depths.split(",")
            if part.strip()
        )
    except ValueError as exc:
        parser.error(f"--training-depths must be comma-separated integers: {exc}")
    out_dir = args.out_dir.expanduser().resolve(strict=False)
    try:
        receipt = run_canary(
            model_path=args.model.expanduser().resolve(strict=True),
            out_dir=out_dir,
            steps=args.steps,
            seed=args.seed,
            memory_fraction=args.memory_fraction,
            student_forcing_probability=args.student_forcing_probability,
            sampling_temperature=args.sampling_temperature,
            specialization_weight=args.specialization_weight,
            warmup_steps=args.warmup_steps,
            warmup_learning_rate=args.warmup_learning_rate,
            joint_learning_rate=args.joint_learning_rate,
            lora_rank=args.lora_rank,
            lora_layers=args.lora_layers,
            lora_targets=lora_targets,
            lora_layer_placement=args.lora_layer_placement,
            coda_lora_layers=args.coda_lora_layers,
            coda_lora_targets=coda_targets,
            training_families=training_families,
            training_depths=training_depths,
            training_per_cell=args.training_per_cell,
            proxy_per_cell=args.proxy_per_cell,
        )
    except BaseException as exc:
        _append_terminal_failure(out_dir, exc)
        raise
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
