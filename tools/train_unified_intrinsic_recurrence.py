#!/usr/bin/env python3
"""Train the unified intrinsic recurrent controller on a bounded checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_map, tree_unflatten  # noqa: E402

from core.learning.frontier_process_supervision import (  # noqa: E402
    frontier_process_task_battery,
)
from core.learning.intrinsic_recurrence import _run, checkpointed_window  # noqa: E402
from core.learning.public_frontier_action_compiler import (  # noqa: E402
    compile_public_frontier_actions,
)
from core.learning.recurrent_action_schema import (  # noqa: E402
    ACTION_CARDINALITY,
    ACTION_SLOT_NAMES,
    MAX_RECURRENT_OPCODE,
    OP_FRONTIER_AUDIT,
    OP_FRONTIER_TRAVERSE,
    action_targets_from_program,
    action_value_semantic_label,
)
from core.learning.recurrent_answer_emission import (  # noqa: E402
    RecurrentAnswerEmissionContract,
    tokenizer_answer_emission_contract,
)
from core.learning.recurrent_literal_grounding import (  # noqa: E402
    LITERAL_MAX_VALUE,
    LiteralObservationContract,
    tokenizer_digit_token_ids,
)
from core.learning.recurrent_opcode_grounding import (  # noqa: E402
    tokenizer_frontier_family_contract,
    tokenizer_opcode_contract,
)
from core.learning.recurrent_state_schema import (  # noqa: E402
    SEMANTIC_STATE_SLOT_NAMES,
    STATE_SLOT_NAMES,
    state_slot_loss_weights,
    state_slot_names,
    state_targets_from_trace,
)
from core.learning.transition_identifiability import (  # noqa: E402
    audit_public_transition_identifiability,
)
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
    readout_fingerprint,
    structured_action_accuracy_breakdown,
    structured_action_loss,
    structured_initial_state_accuracy_breakdown,
    structured_initial_state_loss,
    structured_state_accuracy_breakdown,
    structured_state_loss,
    structured_state_trajectory_diagnostics,
    unified_answer_and_recurrent_trajectory,
    unified_intrinsic_training_loss,
    unified_process_training_loss,
    unified_typed_transition_processor_loss,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    ACTION_LITERAL_BINDING_PARAMETER_NAMES,
    ACTION_WORKSPACE_PARAMETER_NAMES,
    CAUSAL_ACTION_PARAMETER_NAMES,
    FAMILY_ACTION_PARAMETER_NAMES,
    INITIAL_STATE_PARAMETER_NAMES,
    MAX_PROCESS_INTEGER,
    PROCESS_READER_PARAMETER_NAMES,
    PROCESS_TAPE_SCHEMA,
    TRANSITION_COPY_PRIOR_LOGIT_BIAS,
    TRANSITION_MEMORY_PARAMETER_NAMES,
    TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES,
    TRANSITION_PROCESSOR_MODES,
    TRANSITION_PROCESSOR_PARAMETER_NAMES,
    TRANSITION_REPLAY_PARAMETER_NAMES,
    TRANSITION_TAPE_READER_PARAMETER_NAMES,
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
    unified_recurrent_hidden_states,
    unified_recurrent_logits,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_hardlink_replace,
    atomic_write_bytes,
    atomic_write_text,
    durable_unlink,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.mlx_memory_guard import host_pressure, mlx_memory_envelope  # noqa: E402
from tools.resident_recurrent_sft_bootstrap_identity import (  # noqa: E402
    resident_bootstrap_tokenizer_identity,
)
from tools.train_intrinsic_recurrence import encode_example  # noqa: E402
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    CHECKPOINT_GENERATION_SCHEMA,
    CHECKPOINT_POINTER_SCHEMA,
    TRAINING_SCHEMA,
    UnifiedCheckpointError,
    adopt_source_migration_identity,
    bootstrap_topology_mismatches,
    prune_checkpoint_generations,
    resolve_checkpoint_generation,
)
from tools.unified_intrinsic_preload_barrier import verify_release  # noqa: E402
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    CAMPAIGN_BINDING_SCHEMA,
)
from tools.unified_intrinsic_tokenization_contract import (  # noqa: E402
    TOKENIZED_DATASET_FILENAME,
    freeze_source_dataset,
    freeze_tokenized_dataset,
    verify_tokenized_dataset,
)

TRAINING_SOURCE_FILES = (
    "core/brain/llm/latent_cortex/frontier_tasks.py",
    "core/brain/llm/latent_cortex/recurrence_adapter.py",
    "core/brain/llm/latent_cortex/recurrence_adapter_identity_v2.py",
    "core/learning/depth_conditioned_lora.py",
    "core/learning/frontier_process_supervision.py",
    "core/learning/intrinsic_recurrence.py",
    "core/learning/public_frontier_action_compiler.py",
    "core/learning/protected_memory.py",
    "core/learning/recurrence_curriculum.py",
    "core/learning/recurrent_answer_emission.py",
    "core/learning/recurrent_action_schema.py",
    "core/learning/recurrent_literal_grounding.py",
    "core/learning/recurrent_opcode_grounding.py",
    "core/learning/recurrent_state_schema.py",
    "core/learning/transition_identifiability.py",
    "core/learning/unified_intrinsic_objective.py",
    "core/learning/unified_intrinsic_recurrence.py",
    "core/runtime/atomic_writer.py",
    "core/runtime/mlx_memory_guard.py",
    "core/runtime/model_lane_control.py",
    "pyproject.toml",
    "requirements_lock.txt",
    "tools/resident_recurrent_sft_bootstrap_identity.py",
    "tools/evaluate_unified_intrinsic_checkpoint.py",
    "tools/evaluate_unified_intrinsic_decoding.py",
    "tools/train_intrinsic_recurrence.py",
    "tools/train_unified_intrinsic_recurrence.py",
    "tools/unified_intrinsic_checkpoint.py",
    "tools/unified_intrinsic_preload_barrier.py",
    "tools/unified_intrinsic_resident_identity.py",
    "tools/unified_intrinsic_tokenization_contract.py",
)


class UnifiedTrainingBundle(nn.Module):
    def __init__(self, model: Any, controller: UnifiedRecurrentController) -> None:
        super().__init__()
        self.model = model
        self.controller = controller


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


def _parse_campaign_binding(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        binding = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("campaign checkpoint binding is invalid JSON") from exc
    required = {
        "schema",
        "campaign_id",
        "campaign_config_sha256",
        "source_commit",
        "source_tree",
        "source_manifest_sha256",
        "model_manifest_sha256",
        "runtime_identity_sha256",
        "dataset_identity_sha256",
        "tokenizer_identity_sha256",
        "tokenized_dataset_identity_sha256",
        "training_profile_sha256",
        "binding_sha256",
    }
    body = (
        {key: value for key, value in binding.items() if key != "binding_sha256"}
        if isinstance(binding, dict)
        else {}
    )
    if (
        not isinstance(binding, dict)
        or set(binding) != required
        or binding.get("schema") != CAMPAIGN_BINDING_SCHEMA
        or binding.get("binding_sha256") != _canonical_sha256(body)
        or any(
            not isinstance(value, str) or not value
            for key, value in body.items()
            if key != "schema"
        )
    ):
        raise ValueError("campaign checkpoint binding differs")
    return binding


def _adam(learning_rate: float) -> Any:
    return optim.Adam(
        learning_rate=learning_rate,
        betas=[0.9, 0.999],
        eps=1e-8,
        bias_correction=False,
    )


class _DisjointPathMultiOptimizer(optim.MultiOptimizer):
    """Merge optimizer outputs by leaf path, including siblings in module lists."""

    def apply_gradients(self, gradients: dict, parameters: dict) -> dict:
        updated: list[tuple[str, Any]] = []
        seen: set[str] = set()
        gradient_groups = self._split_dictionary(  # noqa: SLF001
            gradients
        )
        parameter_groups = self._split_dictionary(  # noqa: SLF001
            parameters
        )
        for optimizer, gradient_group, parameter_group in zip(
            self.optimizers,
            gradient_groups,
            parameter_groups,
            strict=True,
        ):
            # MLX otherwise initializes Adam moments from the potentially
            # sparse gradient group. Module-list experts can omit inactive
            # siblings on a step, making that state shorter than the owned
            # parameter tree and crashing the first or resumed update.
            if not optimizer._initialized:  # noqa: SLF001 - MLX has no public probe
                optimizer.init(parameter_group)
            for name, value in tree_flatten(
                optimizer.apply_gradients(gradient_group, parameter_group)
            ):
                if name in seen:
                    raise ValueError("optimizer ownership paths overlap")
                seen.add(name)
                updated.append((name, value))
        return tree_unflatten(updated)


def _ownership_optimizer(
    learning_rate: float,
    *,
    transformer_rate_scale: float = 1.0,
    query_rate_scale: float | None = None,
) -> Any:
    """Use distinct Adam moments and rates for query, bridge, and controller tissue."""

    scales = [("transformer", transformer_rate_scale)]
    if query_rate_scale is not None:
        scales.append(("query", query_rate_scale))
    for label, scale in scales:
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not 0.0 <= float(scale) <= 1.0
        ):
            raise ValueError(f"{label} optimizer rate scale must be inside [0, 1]")
    if query_rate_scale is None:
        return _DisjointPathMultiOptimizer(
            [
                _adam(float(learning_rate) * float(transformer_rate_scale)),
                _adam(float(learning_rate)),
            ],
            filters=[
                lambda name, _value: (
                    _gradient_ownership_group(name) == "scoped_transformer_bridge"
                )
            ],
        )
    return _DisjointPathMultiOptimizer(
        [
            _adam(float(learning_rate) * float(query_rate_scale)),
            _adam(float(learning_rate) * float(transformer_rate_scale)),
            _adam(float(learning_rate)),
        ],
        filters=[
            lambda name, _value: (
                _gradient_ownership_group(name) == "scoped_transformer_query"
            ),
            lambda name, _value: (
                _gradient_ownership_group(name) == "scoped_transformer_bridge"
            ),
        ],
    )


def _set_ownership_optimizer_rates(
    optimizer: Any,
    learning_rate: float,
    *,
    transformer_rate_scale: float,
    query_rate_scale: float | None,
) -> None:
    """Restore phase rates without collapsing MultiOptimizer group identity."""

    expected_optimizers = 3 if query_rate_scale is not None else 2
    if (
        not isinstance(optimizer, optim.MultiOptimizer)
        or len(optimizer.optimizers) != expected_optimizers
    ):
        raise TypeError("unified recurrence optimizer topology differs")
    scales = [("transformer", transformer_rate_scale)]
    if query_rate_scale is not None:
        scales.append(("query", query_rate_scale))
    for label, scale in scales:
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not 0.0 <= float(scale) <= 1.0
        ):
            raise ValueError(f"{label} optimizer rate scale must be inside [0, 1]")
    if query_rate_scale is None:
        optimizer.optimizers[0].learning_rate = float(learning_rate) * float(
            transformer_rate_scale
        )
        optimizer.optimizers[1].learning_rate = float(learning_rate)
    else:
        optimizer.optimizers[0].learning_rate = float(learning_rate) * float(query_rate_scale)
        optimizer.optimizers[1].learning_rate = float(learning_rate) * float(
            transformer_rate_scale
        )
        optimizer.optimizers[2].learning_rate = float(learning_rate)
    mx.eval(*(candidate.learning_rate for candidate in optimizer.optimizers))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(model_path: str) -> dict[str, Any]:
    directory = Path(model_path).expanduser().resolve(strict=True)
    config = directory / "config.json"
    weights = sorted(directory.glob("*.safetensors"))
    if not config.is_file() or not weights:
        raise ValueError("model checkpoint identity is incomplete")
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name != "README.md"
    )
    rows: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for path in files:
        if path.is_symlink():
            raise ValueError("model checkpoint contains a symlinked artifact")
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        row = {
            "name": path.name,
            "size": before.st_size,
            "sha256": _file_sha256(resolved),
        }
        after = resolved.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError("model checkpoint changed while hashing")
        rows.append(row)
        by_name[path.name] = row
    weight_rows = [by_name[path.name] for path in weights]
    behavior_rows = [row for row in rows if not row["name"].endswith(".safetensors")]
    body = {
        "canonical_path": str(directory),
        "config_sha256": by_name[config.name]["sha256"],
        "weights": weight_rows,
        "behavior_files": behavior_rows,
        "behavior_sha256": _canonical_sha256(behavior_rows),
    }
    return {**body, "identity_sha256": _canonical_sha256(body)}


def _runtime_identity() -> dict[str, Any]:
    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        runtime_environment_identity,
    )

    environment = runtime_environment_identity()
    executable = Path(os.path.abspath(sys.executable))
    real_executable = executable.resolve(strict=True)
    before = real_executable.stat()
    executable_sha256 = _file_sha256(real_executable)
    after = real_executable.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError("Python interpreter changed while hashing")
    body = {
        "environment": environment,
        "interpreter": {
            "executable": str(executable),
            "real_executable": str(real_executable),
            "sys_prefix": str(Path(sys.prefix).resolve(strict=True)),
            "base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
            "size_bytes": before.st_size,
            "sha256": executable_sha256,
        },
    }
    return {**body, "identity_sha256": _canonical_sha256(body)}


def _freeze_dataset(
    out_dir: Path,
    train_tasks: list[Any],
    holdout_tasks: list[Any],
) -> dict[str, Any]:
    from tools.unified_intrinsic_tokenization_contract import freeze_source_dataset

    return freeze_source_dataset(out_dir / "dataset.json", train_tasks, holdout_tasks)


def _load_frozen_dataset(path: Path) -> tuple[list[Any], list[Any]]:
    from tools.unified_intrinsic_tokenization_contract import load_source_dataset

    return load_source_dataset(path)


def _model_layer_count(model_path: str) -> int:
    config_path = Path(model_path).expanduser().resolve(strict=True) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise ValueError("model checkpoint config is unreadable") from exc
    if not isinstance(config, dict):
        raise ValueError("model checkpoint config differs")
    candidates = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        candidates.insert(0, text_config)
    for candidate in candidates:
        value = candidate.get("num_hidden_layers")
        if type(value) is int and value >= 3:
            return value
    raise ValueError("model checkpoint layer count is unavailable")


def _resolve_recurrent_window(
    model_path: str,
    *,
    prelude_end: int | None,
    coda_start: int | None,
    prelude_fraction: float,
    coda_fraction: float,
) -> tuple[int, int, dict[str, Any]]:
    layer_count = _model_layer_count(model_path)
    supplied = (prelude_end is not None, coda_start is not None)
    if supplied[0] != supplied[1]:
        raise ValueError("explicit recurrent window requires both boundaries")
    if all(supplied):
        resolved_prelude = int(prelude_end)
        resolved_coda = int(coda_start)
        mode = "explicit"
    else:
        for name, value in (
            ("prelude", prelude_fraction),
            ("coda", coda_fraction),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) < 0.5
            ):
                raise ValueError(f"{name} fraction must be finite and inside (0, 0.5)")
        if float(prelude_fraction) + float(coda_fraction) >= 1.0:
            raise ValueError("recurrent window fractions leave no middle block")
        resolved_prelude = max(1, int(layer_count * float(prelude_fraction)))
        resolved_coda = min(
            layer_count - 1,
            layer_count - max(1, int(layer_count * float(coda_fraction))),
        )
        mode = "fractional"
    if not 0 < resolved_prelude < resolved_coda < layer_count:
        raise ValueError("resolved recurrent window is outside the model")
    body = {
        "mode": mode,
        "layer_count": layer_count,
        "prelude_end": resolved_prelude,
        "coda_start": resolved_coda,
        "prelude_fraction": (float(prelude_fraction) if mode == "fractional" else None),
        "coda_fraction": float(coda_fraction) if mode == "fractional" else None,
    }
    return (
        resolved_prelude,
        resolved_coda,
        {
            **body,
            "contract_sha256": _canonical_sha256(body),
        },
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, encoded, encoding="utf-8", mode=0o600)


def _atomic_canonical_json(path: Path, value: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    atomic_write_text(path, encoded, encoding="ascii", mode=0o600)


def _await_resource_guard(
    marker_path: Path,
    *,
    trainer_sha256: str,
    startup_lethal_mb: float,
    steady_lethal_mb: float,
    timeout_s: float,
) -> dict[str, Any]:
    """Refuse the first optimizer graph until an external sentinel is armed."""

    from core.runtime.resource_stage_guard import (
        ResourceStageGuardError,
        ack_path,
        publish_ready_marker,
        read_armed_ack,
        sha256_bytes,
    )

    acknowledgement = ack_path(marker_path)
    if marker_path.exists() or acknowledgement.exists():
        raise ResourceStageGuardError("resource guard attempt artifacts already exist")
    marker, marker_raw = publish_ready_marker(
        marker_path,
        target_pid=os.getpid(),
        trainer_sha256=trainer_sha256,
    )
    print(f"resource guard marker published: {marker_path}", flush=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if acknowledgement.exists():
            acknowledgement_payload, acknowledgement_raw = read_armed_ack(
                marker_path,
                marker_raw=marker_raw,
                expected_target_pid=os.getpid(),
                startup_lethal_mb=startup_lethal_mb,
                steady_lethal_mb=steady_lethal_mb,
            )
            return {
                "marker": marker,
                "marker_sha256": sha256_bytes(marker_raw),
                "ack": acknowledgement_payload,
                "ack_sha256": sha256_bytes(acknowledgement_raw),
            }
        time.sleep(0.25)
    raise ResourceStageGuardError("external sentinel did not acknowledge unified training in time")


def _attach_window_adapters(
    model: Any,
    spec: UnifiedIntrinsicTrainingSpec,
    *,
    rank: int,
    targets: tuple[str, ...],
    depth_basis_size: int,
) -> dict[str, Any]:
    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear
    from core.learning.hybrid_recurrence_geometry import (
        LayerGeometry,
        expected_adapter_sites,
        geometry_receipt,
        window_alignment_errors,
    )

    # What the window is entitled to produce on THIS checkpoint, settled before
    # anything attaches. A hybrid model gives a layer self_attn only every
    # fourth index, so the loop below simply skips three layers in four and
    # still returns a non-empty site list -- a campaign that declared a
    # 32-layer window quietly training 8 layers of it.
    geometry = LayerGeometry.from_model(model)
    alignment = window_alignment_errors(geometry, spec.prelude_end, spec.coda_start)
    if alignment:
        raise RuntimeError(
            "unified recurrence window is misaligned for this checkpoint: "
            + "; ".join(alignment)
        )
    expected = expected_adapter_sites(
        geometry, spec.prelude_end, spec.coda_start, targets
    )

    sites = []
    for layer_index in range(spec.prelude_end, spec.coda_start):
        layer = model.model.layers[layer_index]
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for target in targets:
                projection = getattr(parent, target, None)
                if projection is None or isinstance(projection, ScopedLoRALinear):
                    continue
                site = f"model.layers.{layer_index}.{parent_name}.{target}"
                setattr(
                    parent,
                    target,
                    ScopedLoRALinear.from_base(
                        projection,
                        r=rank,
                        block_index=layer_index,
                        site=site,
                    ),
                )
                sites.append(site)
    if not sites:
        raise RuntimeError("unified recurrence attached no window projections")
    if sorted(sites) != sorted(expected):
        missing = sorted(set(expected) - set(sites))
        extra = sorted(set(sites) - set(expected))
        raise RuntimeError(
            "unified recurrence attachment does not match the declared geometry: "
            f"{len(sites)} attached, {len(expected)} expected"
            + (f"; missing {missing[:4]}" if missing else "")
            + (f"; unexpected {extra[:4]}" if extra else "")
        )
    from core.learning.depth_conditioned_lora import (
        wrap_continuous_depth_conditioned,
    )

    depth_operators = wrap_continuous_depth_conditioned(
        model,
        basis_size=depth_basis_size,
    )
    if set(depth_operators) != set(sites):
        raise RuntimeError("continuous depth operator inventory differs from adapters")
    return {
        "window_tissue_mode": "scoped_lora",
        "window": [spec.prelude_end, spec.coda_start],
        "geometry": geometry_receipt(
            geometry, spec.prelude_end, spec.coda_start, targets
        ),
        "adapted_sites": sorted(sites),
        "adapted_projection_count": len(sites),
        "continuous_depth_operator_count": len(depth_operators),
        "continuous_depth_basis_size": depth_basis_size,
        "coda_adapted": False,
        "readout_adapted": False,
        "ordinary_inference_requires_scope": True,
        "recurrence_phase_trains_shared_state_bridge": False,
        "state_transition_trains_shared_process_parser": True,
        "state_bridge": "continuous_depth_residual_preserves_t1",
    }


def _configure_window_tissue(
    model: Any,
    spec: UnifiedIntrinsicTrainingSpec,
    *,
    mode: str,
    rank: int,
    targets: tuple[str, ...],
    depth_basis_size: int,
) -> dict[str, Any]:
    """Build the declared recurrent tissue without silently adding base adapters."""

    if mode == "scoped_lora":
        return _attach_window_adapters(
            model,
            spec,
            rank=rank,
            targets=targets,
            depth_basis_size=depth_basis_size,
        )
    if mode != "controller_only":
        raise ValueError("unified recurrence window tissue mode is invalid")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or not 0 <= spec.prelude_end < spec.coda_start <= len(layers):
        raise ValueError("controller-only recurrent window is outside the model")
    return {
        "window_tissue_mode": "controller_only",
        "window": [spec.prelude_end, spec.coda_start],
        "adapted_sites": [],
        "adapted_projection_count": 0,
        "continuous_depth_operator_count": 0,
        "continuous_depth_basis_size": 0,
        "coda_adapted": False,
        "readout_adapted": False,
        "ordinary_inference_requires_scope": False,
        "recurrence_phase_trains_shared_state_bridge": False,
        "state_transition_trains_shared_process_parser": False,
        "state_bridge": "typed_recurrent_controller_only",
    }


def _model_lane_purpose(window_tissue_mode: str) -> str:
    """Select the physical-memory envelope that matches the trainable tissue."""

    if window_tissue_mode == "controller_only":
        return "train_frozen_controller"
    if window_tissue_mode == "scoped_lora":
        return "train"
    raise ValueError("unified recurrence window tissue mode is invalid")


def _fresh_public_transition_acquisition(
    *,
    window_tissue_mode: str,
    public_action_program: bool,
    direct_transition_processor: bool,
) -> bool:
    """Return whether transition tissue is complete without a parent checkpoint.

    Legacy transition-only campaigns refine a previously acquired process
    controller, so their parent checkpoint is part of the scientific object.
    A controller-only direct transition campaign is different: independently
    compiled public actions are its input and the fresh controller is the only
    trainable mechanism. Requiring unrelated parent tissue there would make
    the new state geometry unloadable and confound acquisition with the
    parent's learned behavior.
    """

    return (
        window_tissue_mode == "controller_only"
        and public_action_program
        and direct_transition_processor
    )


def _trainable(bundle: UnifiedTrainingBundle) -> dict[str, Any]:
    return dict(tree_flatten(bundle.trainable_parameters()))


def _ground_state_value_embeddings(
    model: Any,
    tokenizer: Any,
    controller: UnifiedRecurrentController,
    *,
    prelude_end: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Initialize typed values on the frozen model's native prelude manifold."""

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("grounding batch size must be a positive integer")

    def encode_label(label: str) -> list[int]:
        try:
            token_ids = tokenizer.encode(label, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(label)
        if not token_ids:
            raise RuntimeError("grounded semantic label encoded to no tokens")
        return [int(token_id) for token_id in token_ids]

    register_names = state_slot_names(controller.config.state_slots)
    labels = [
        f"Internal state {slot_name}={value}"
        for slot_name in register_names
        for value in range(controller.config.state_cardinality)
    ]
    labels.extend(
        action_value_semantic_label(slot_name, value)
        for slot_name in ACTION_SLOT_NAMES
        for value in range(controller.config.action_cardinality)
    )
    labels.extend(str(value) for value in range(LITERAL_MAX_VALUE + 1))
    encoded = [encode_label(label) for label in labels]
    buckets: dict[int, list[tuple[int, list[int]]]] = {}
    for index, token_ids in enumerate(encoded):
        buckets.setdefault(len(token_ids), []).append((index, token_ids))

    grounded_rows: list[Any | None] = [None] * len(labels)
    forward_batches = 0
    for token_count in sorted(buckets):
        bucket = buckets[token_count]
        for offset in range(0, len(bucket), batch_size):
            batch = bucket[offset : offset + batch_size]
            tokens = mx.array(
                [token_ids for _index, token_ids in batch],
                dtype=mx.int32,
            )
            hidden = model.model.embed_tokens(tokens)
            terminal = _run(model.model.layers[:prelude_end], hidden)[:, -1, :].astype(mx.float32)
            mx.eval(terminal)
            for row, (index, _token_ids) in enumerate(batch):
                grounded_rows[index] = terminal[row]
            forward_batches += 1

    if any(value is None for value in grounded_rows):
        raise RuntimeError("grounded semantic label inventory is incomplete")
    concrete_rows = [value for value in grounded_rows if value is not None]
    cursor = 0

    def take(shape: tuple[int, int]) -> Any:
        nonlocal cursor
        row_count, cardinality = shape
        count = row_count * cardinality
        selected = concrete_rows[cursor : cursor + count]
        cursor += count
        return mx.stack(selected).reshape(row_count, cardinality, -1)

    grounded = take((len(register_names), controller.config.state_cardinality))
    grounded_actions = take((len(ACTION_SLOT_NAMES), controller.config.action_cardinality))
    literal_count = LITERAL_MAX_VALUE + 1
    grounded_literals = mx.stack(concrete_rows[cursor : cursor + literal_count])
    cursor += literal_count
    if cursor != len(labels):
        raise RuntimeError("grounded semantic label cursor differs")
    if grounded.shape != controller.state_value_embeddings.shape:
        raise RuntimeError("grounded state codebook shape differs from controller")
    controller.state_value_embeddings = grounded
    if grounded_actions.shape != controller.action_value_embeddings.shape:
        raise RuntimeError("grounded action codebook shape differs from controller")
    controller.action_value_embeddings = grounded_actions
    if grounded_literals.shape != controller.literal_value_embeddings.shape:
        raise RuntimeError("grounded literal codebook shape differs from controller")
    controller.literal_value_embeddings = grounded_literals
    mx.eval(
        controller.state_value_embeddings,
        controller.action_value_embeddings,
        controller.literal_value_embeddings,
        controller.state_slot_embeddings,
        controller.action_slot_embeddings,
    )
    digest = hashlib.sha256(
        bytes(memoryview(controller.state_value_embeddings.astype(mx.float32)))
        + bytes(memoryview(controller.action_value_embeddings.astype(mx.float32)))
        + bytes(memoryview(controller.literal_value_embeddings.astype(mx.float32)))
        + bytes(memoryview(controller.state_slot_embeddings.astype(mx.float32)))
        + bytes(memoryview(controller.action_slot_embeddings.astype(mx.float32)))
    ).hexdigest()
    return {
        "sha256": digest,
        "label_count": len(labels),
        "forward_batches": forward_batches,
        "batch_size": batch_size,
        "token_length_buckets": sorted(buckets),
        "max_token_length": max(buckets),
    }


def _optimization_phase(
    step: int,
    semantic_warmup_steps: int,
    state_warmup_steps: int = 0,
    answer_bridge_steps: int = 0,
) -> str:
    if type(step) is not int or step < 0:
        raise ValueError("optimization step must be non-negative")
    if type(semantic_warmup_steps) is not int or semantic_warmup_steps < 0:
        raise ValueError("semantic warmup steps must be non-negative")
    if type(state_warmup_steps) is not int or state_warmup_steps < 0:
        raise ValueError("state warmup steps must be non-negative")
    if type(answer_bridge_steps) is not int or answer_bridge_steps < 0:
        raise ValueError("answer bridge steps must be non-negative")
    # State parsing must exist before a semantic adapter is asked to decode a
    # typed path.  Training the no-state decoder first produced an apparent CE
    # gain while the actual T1 inference path collapsed to punctuation.
    if step < state_warmup_steps:
        return "state_transition"
    if step < state_warmup_steps + semantic_warmup_steps:
        return "semantic_anchor"
    if step < state_warmup_steps + semantic_warmup_steps + answer_bridge_steps:
        return "answer_bridge"
    return "recurrence"


def _process_training_policy(
    step: int,
    total_steps: int,
    curriculum: str,
    *,
    initial_teacher_probability: float = 1.0,
    final_teacher_probability: float = 0.0,
    teacher_hold_fraction: float = 0.0,
) -> dict[str, Any]:
    """Assign exclusive process objectives before autonomous integration.

    A blended loss let the easy public-prefix initializer absorb useful
    gradient while typed actions and value transitions remained at chance.
    The factorized curriculum gives each causal mechanism an exclusive
    acquisition interval, then removes the teacher across a final joint tail.
    """

    if (
        type(step) is not int
        or type(total_steps) is not int
        or total_steps < 1
        or not 0 <= step < total_steps
    ):
        raise ValueError("process curriculum coordinates are invalid")
    if (
        isinstance(initial_teacher_probability, bool)
        or isinstance(final_teacher_probability, bool)
        or not isinstance(initial_teacher_probability, (int, float))
        or not isinstance(final_teacher_probability, (int, float))
        or not 0.0
        <= float(final_teacher_probability)
        <= float(initial_teacher_probability)
        <= 1.0
        or isinstance(teacher_hold_fraction, bool)
        or not isinstance(teacher_hold_fraction, (int, float))
        or not 0.0 <= float(teacher_hold_fraction) < 1.0
    ):
        raise ValueError("process teacher-forcing schedule must decrease inside [0, 1]")
    if curriculum == "joint":
        return {
            "component": "joint",
            "teacher_forcing_probability": 1.0,
            "stage_progress": (step + 1) / total_steps,
        }
    if curriculum == "action_workspace":
        autonomous_start = 3 * total_steps // 4
        teacher_probability = 1.0
        if step >= autonomous_start:
            autonomous_steps = total_steps - autonomous_start
            teacher_probability = max(
                0.0,
                1.0 - (step - autonomous_start + 1) / autonomous_steps,
            )
        return {
            "component": "action_workspace",
            "teacher_forcing_probability": teacher_probability,
            "stage_progress": (step + 1) / total_steps,
        }
    if curriculum == "transition_only":
        hold_steps = min(
            total_steps - 1,
            int(total_steps * float(teacher_hold_fraction)),
        )
        anneal_steps = total_steps - hold_steps
        progress = (
            0.0
            if step < hold_steps or anneal_steps == 1
            else (step - hold_steps) / (anneal_steps - 1)
        )
        teacher_probability = float(initial_teacher_probability) + progress * (
            float(final_teacher_probability) - float(initial_teacher_probability)
        )
        return {
            "component": "transition",
            "teacher_forcing_probability": teacher_probability,
            "stage_progress": (step + 1) / total_steps,
        }
    if curriculum != "factorized" or total_steps < 8:
        raise ValueError("process curriculum is invalid or too short")
    initializer_stop = max(1, total_steps // 8)
    action_stop = max(initializer_stop + 1, total_steps // 2)
    transition_stop = max(action_stop + 1, 7 * total_steps // 8)
    transition_stop = min(transition_stop, total_steps - 1)
    if step < initializer_stop:
        component = "initializer"
        stage_start, stage_stop = 0, initializer_stop
    elif step < action_stop:
        component = "action"
        stage_start, stage_stop = initializer_stop, action_stop
    elif step < transition_stop:
        component = "transition"
        stage_start, stage_stop = action_stop, transition_stop
    else:
        component = "joint"
        stage_start, stage_stop = transition_stop, total_steps
    stage_steps = stage_stop - stage_start
    stage_index = step - stage_start
    stage_progress = (stage_index + 1) / stage_steps
    teacher_probability = 1.0
    if component == "joint":
        teacher_probability = 0.0 if stage_steps == 1 else 1.0 - stage_index / (stage_steps - 1)
    return {
        "component": component,
        "teacher_forcing_probability": teacher_probability,
        "stage_progress": stage_progress,
    }


def _direct_transition_curriculum_window(
    step: int,
    total_steps: int,
    transition_depth: int,
    *,
    mode: str,
) -> dict[str, Any]:
    """Progress from local transition mastery to the deployed closed loop."""

    if (
        type(step) is not int
        or type(total_steps) is not int
        or type(transition_depth) is not int
        or total_steps < 1
        or transition_depth < 1
        or not 0 <= step < total_steps
        or mode not in {"closed_loop", "progressive"}
    ):
        raise ValueError("direct transition curriculum coordinates differ")
    if mode == "closed_loop":
        return {
            "stage": "closed_loop",
            "transition_start": 0,
            "transition_count": transition_depth,
            "training_only_midtrace_initial_state": False,
            "complete_public_prefix_visible": True,
            "corrupt_transition": None,
            "corrupt_state_mode": None,
            "corrupt_state_slot": None,
            "corrupt_state_offset": None,
        }

    stage_stop = (
        max(1, 15 * total_steps // 100),
        max(2, 30 * total_steps // 100),
        max(3, 45 * total_steps // 100),
    )
    widths = (1, 2, 4)
    stage_index = next(
        (index for index, stop in enumerate(stage_stop) if step < stop),
        None,
    )
    if stage_index is None:
        recovery_start = max(stage_stop[-1], 70 * total_steps // 100)
        recovery_stop = max(recovery_start, 85 * total_steps // 100)
        if recovery_start <= step < recovery_stop:
            recovery_index = step - recovery_start
            return {
                "stage": "controlled_recovery",
                "transition_start": 0,
                "transition_count": transition_depth,
                "training_only_midtrace_initial_state": False,
                "complete_public_prefix_visible": True,
                "corrupt_transition": recovery_index % transition_depth,
                "corrupt_state_mode": "coherent_trace_state",
                "corrupt_state_slot": None,
                "corrupt_state_offset": 1 + recovery_index % transition_depth,
            }
        return {
            "stage": "closed_loop",
            "transition_start": 0,
            "transition_count": transition_depth,
            "training_only_midtrace_initial_state": False,
            "complete_public_prefix_visible": True,
            "corrupt_transition": None,
            "corrupt_state_mode": None,
            "corrupt_state_slot": None,
            "corrupt_state_offset": None,
        }
    width = min(widths[stage_index], transition_depth)
    stage_start = 0 if stage_index == 0 else stage_stop[stage_index - 1]
    available_starts = transition_depth - width + 1
    transition_start = (step - stage_start) % available_starts
    return {
        "stage": f"verified_window_{width}",
        "transition_start": transition_start,
        "transition_count": width,
        "training_only_midtrace_initial_state": transition_start > 0,
        "complete_public_prefix_visible": True,
        "corrupt_transition": None,
        "corrupt_state_mode": None,
        "corrupt_state_slot": None,
        "corrupt_state_offset": None,
    }


def _phase_schedule(
    *,
    semantic_warmup_steps: int,
    state_warmup_steps: int,
    answer_bridge_steps: int,
    max_steps: int,
    bootstrap_output_dir: Path | None,
    process_only: bool = False,
    process_bootstrap: bool = False,
) -> dict[str, Any]:
    """Validate and bind a full campaign or a bootstrap-only bridge adaptation."""

    values = (
        semantic_warmup_steps,
        state_warmup_steps,
        answer_bridge_steps,
        max_steps,
    )
    if any(type(value) is not int or value < 0 for value in values) or max_steps < 1:
        raise ValueError("optimization phase steps must be non-negative integers")
    warmup_steps = semantic_warmup_steps + state_warmup_steps + answer_bridge_steps
    recurrence_steps = max_steps - warmup_steps
    if recurrence_steps < 0:
        raise ValueError("optimization phases exceed maximum steps")
    bridge_only = recurrence_steps == 0
    if bridge_only:
        process_contract = (
            process_only
            and state_warmup_steps == max_steps
            and semantic_warmup_steps == 0
            and answer_bridge_steps == 0
            and (bootstrap_output_dir is not None) is process_bootstrap
        )
        bridge_contract = answer_bridge_steps > 0 and bootstrap_output_dir is not None
        if not process_contract and not bridge_contract:
            raise ValueError(
                "zero-recurrence training requires either process-only acquisition "
                "or a bootstrapped answer-bridge campaign"
            )
    return {
        "schema": "aura.unified_intrinsic.phase_schedule.v1",
        "mode": (
            "bootstrap_process_acquisition_only"
            if bridge_only and process_only and process_bootstrap
            else "process_acquisition_only"
            if bridge_only and process_only
            else "bootstrap_answer_bridge_only"
            if bridge_only
            else "recurrent_training"
        ),
        "semantic_anchor_steps": semantic_warmup_steps,
        "state_transition_steps": state_warmup_steps,
        "answer_bridge_steps": answer_bridge_steps,
        "recurrence_steps": recurrence_steps,
        "max_steps": max_steps,
        "bootstrap_required": bridge_only and (not process_only or process_bootstrap),
    }


def _semantic_execution_depth(
    task_depth: int,
    spec: UnifiedIntrinsicTrainingSpec,
) -> int:
    """Return the public execution depth at which the task is complete."""

    if type(task_depth) is not int or task_depth not in spec.train_depths:
        raise ValueError("semantic task depth is outside the trained recurrence horizon")
    return task_depth


def _phase_gradients(gradients: Any, phase: str) -> Any:
    """Keep the T1 semantic anchor fixed while training residual recurrence.

    Shared adapters learn the scoped T1 anchor.  Typed-state interpretation is
    owned by the continuous depth residuals, so a joint update cannot erase a
    useful shallow candidate or alter ordinary model inference.
    """

    if phase not in {
        "semantic_anchor",
        "answer_bridge",
        "state_transition",
        "recurrence",
    }:
        raise ValueError("unified optimization phase is invalid")
    masked = []
    for name, value in tree_flatten(gradients):
        shared_adapter = (
            name.startswith("model.")
            and "continuous_depth_" not in name
            and (name.endswith(".lora_a") or name.endswith(".lora_b"))
        )
        neural_answer_bridge = name.startswith(("controller.answer_", "controller.process_reader_"))
        if phase == "semantic_anchor":
            keep = shared_adapter
        elif phase == "answer_bridge":
            keep = neural_answer_bridge
        elif phase == "state_transition":
            # Continuous-depth features are exactly zero at T1. The scoped
            # shared adapter is therefore the only transformer surface that
            # can learn the first process step. It remains inert outside an
            # explicit recurrence scope, preserving ordinary inference.
            keep = not neural_answer_bridge
        else:
            keep = not (shared_adapter or neural_answer_bridge)
        masked.append((name, value if keep else mx.zeros_like(value)))
    return tree_unflatten(masked)


def _clip_gradient_norm(gradients: Any, max_norm: float) -> tuple[Any, Any]:
    if (
        isinstance(max_norm, bool)
        or not isinstance(max_norm, (int, float))
        or not 0.0 < float(max_norm)
    ):
        raise ValueError("maximum gradient norm must be positive")
    flattened = tree_flatten(gradients)
    if not flattened:
        raise ValueError("gradient tree must not be empty")
    norm = mx.sqrt(
        mx.sum(mx.stack([mx.sum(value.astype(mx.float32) ** 2) for _name, value in flattened]))
    )
    scale = mx.minimum(1.0, float(max_norm) / mx.maximum(norm, 1e-12))
    return tree_unflatten(
        [(name, value * scale.astype(value.dtype)) for name, value in flattened]
    ), norm


def _gradient_ownership_group(name: str) -> str:
    if name.startswith("model."):
        if ".self_attn.q_proj." in name:
            return "scoped_transformer_query"
        return "scoped_transformer_bridge"
    if name.startswith("controller.initial_state_") or name == ("controller.state_slot_embeddings"):
        return "typed_state_initializer"
    if name.startswith(
        (
            "controller.process_reader_",
            "controller.answer_query",
            "controller.answer_key",
            "controller.answer_value",
            "controller.answer_output",
            "controller.answer_gate_query",
            "controller.answer_gate_logit",
            "controller.answer_role_projection",
            "controller.answer_role_bias",
            "controller.answer_place_projection",
            "controller.answer_place_state_projection",
            "controller.answer_place_width_projection",
            "controller.answer_place_bias",
            "controller.answer_digit_gate_logit",
        )
    ):
        return "state_answer_bridge"
    if name.startswith("controller.action_value_embeddings"):
        return "typed_action_codebook"
    if name.startswith(
        (
            "controller.action_workspace_",
            "controller.action_causal_",
            "controller.action_family_",
            "controller.action_literal_binding_",
        )
    ):
        return "typed_action_workspace"
    if name.startswith(
        (
            "controller.action_query",
            "controller.action_key",
            "controller.action_value",
            "controller.action_output",
            "controller.action_depth",
            "controller.action_bias",
            "controller.action_literal_copy_logit",
            "controller.opcode_copy_logit",
            "controller.action_slot_embeddings",
        )
    ):
        return "typed_action_transition"
    if name.startswith(
        (
            "controller.state_transition_",
            "controller.state_readout_",
            "controller.state_literal_copy_logit",
            "controller.state_action_projection",
            "controller.transition_memory_",
            "controller.transition_tape_",
            "controller.transition_processor_",
            "controller.transition_replay_",
        )
    ):
        return "typed_state_transition"
    if name.startswith(
        (
            "controller.state_value_embeddings",
            "controller.state_slot_embeddings",
            "controller.literal_value_embeddings",
            "controller.literal_grounding_logit",
        )
    ):
        return "typed_state_codebook"
    return "recurrent_controller"


def _process_component_gradients(
    gradients: Any,
    component: str,
) -> Any:
    """Prevent one typed-process role from rewriting another role's tissue."""

    transformer_groups = {"scoped_transformer_query", "scoped_transformer_bridge"}
    allowed = {
        "initializer": {*transformer_groups, "typed_state_initializer"},
        "action": {
            *transformer_groups,
            "typed_action_transition",
            "typed_action_workspace",
        },
        "action_workspace": {
            *transformer_groups,
            "typed_action_workspace",
        },
        "transition": {
            *transformer_groups,
            "typed_action_codebook",
            "typed_state_codebook",
            "typed_state_transition",
        },
        "joint": {
            *transformer_groups,
            "typed_action_codebook",
            "typed_action_transition",
            "typed_action_workspace",
            "typed_state_codebook",
            "typed_state_initializer",
            "typed_state_transition",
        },
    }
    if component not in allowed:
        raise ValueError("process gradient component is invalid")
    return tree_unflatten(
        [
            (
                name,
                value
                if _gradient_ownership_group(name) in allowed[component]
                else mx.zeros_like(value),
            )
            for name, value in tree_flatten(gradients)
        ]
    )


def _clip_gradient_groups(
    gradients: Any,
    max_norm: float,
) -> tuple[Any, Any, dict[str, Any]]:
    """Clip independent mechanisms without letting one starve the others."""

    if (
        isinstance(max_norm, bool)
        or not isinstance(max_norm, (int, float))
        or not 0.0 < float(max_norm)
    ):
        raise ValueError("maximum gradient norm must be positive")
    flattened = tree_flatten(gradients)
    if not flattened:
        raise ValueError("gradient tree must not be empty")
    grouped: dict[str, list[Any]] = {}
    for name, value in flattened:
        grouped.setdefault(_gradient_ownership_group(name), []).append(value)
    group_norms = {
        group: mx.sqrt(
            mx.sum(mx.stack([mx.sum(value.astype(mx.float32) ** 2) for value in values]))
        )
        for group, values in grouped.items()
    }
    scales = {
        group: mx.minimum(1.0, float(max_norm) / mx.maximum(norm, 1e-12))
        for group, norm in group_norms.items()
    }
    clipped = tree_unflatten(
        [
            (
                name,
                value * scales[_gradient_ownership_group(name)].astype(value.dtype),
            )
            for name, value in flattened
        ]
    )
    global_norm = mx.sqrt(
        mx.sum(mx.stack([norm.astype(mx.float32) ** 2 for norm in group_norms.values()]))
    )
    return clipped, global_norm, group_norms


def _apply_training_gradients(
    bundle: UnifiedTrainingBundle,
    optimizer: Any,
    gradients: Any,
    *,
    phase: str,
    max_norm: float,
    totals: dict[str, Any],
    loss: Any,
    process_component: str | None = None,
) -> None:
    """Apply one ownership-masked update and retain pre-clip diagnostics."""

    gradients = _phase_gradients(gradients, phase)
    if process_component is not None:
        if phase != "state_transition":
            raise ValueError("process component requires state-transition phase")
        gradients = _process_component_gradients(
            gradients,
            process_component,
        )
    gradients, gradient_norm, gradient_group_norms = _clip_gradient_groups(
        gradients,
        max_norm,
    )
    mx.eval(gradient_norm, *gradient_group_norms.values())
    totals["max_preclip_gradient_norm"] = max(
        float(totals["max_preclip_gradient_norm"]),
        float(gradient_norm.item()),
    )
    prior_group_norms = totals["max_preclip_gradient_norms"]
    for group, group_norm in gradient_group_norms.items():
        prior_group_norms[group] = max(
            float(prior_group_norms.get(group, 0.0)),
            float(group_norm.item()),
        )
    optimizer.update(bundle, gradients)
    mx.eval(bundle.parameters(), optimizer.state, loss)


def _streamed_recurrent_objective_gradients(
    bundle: UnifiedTrainingBundle,
    prompt: Any,
    answer: Any,
    spec: UnifiedIntrinsicTrainingSpec,
    *,
    readout_sha256: str,
    decoder_input_tokens: Any,
    transition_trace: Any,
    transition_program: Any,
    state_teacher_forcing_probability: float,
    envelope: Any,
    answer_digit_pointer_enabled: bool = True,
) -> tuple[Any, Any]:
    """Differentiate the exact recurrence objective one depth graph at a time."""

    accumulated = None
    total_loss = 0.0
    for depth in spec.train_depths:

        def depth_objective(
            candidate: UnifiedTrainingBundle,
            objective_prompt: Any,
            objective_answer: Any,
            objective_rollin: Any,
            objective_depth: int = depth,
        ) -> Any:
            return unified_intrinsic_training_loss(
                candidate.model,
                objective_prompt,
                objective_answer,
                candidate.controller,
                spec,
                readout_sha256=readout_sha256,
                decoder_input_tokens=objective_rollin,
                transition_trace=transition_trace,
                transition_program=transition_program,
                state_teacher_forcing_probability=state_teacher_forcing_probability,
                answer_digit_pointer_enabled=answer_digit_pointer_enabled,
                objective_depth=objective_depth,
            )[0]

        loss, gradients = nn.value_and_grad(bundle, depth_objective)(
            bundle,
            prompt,
            answer,
            decoder_input_tokens,
        )
        materialized = tree_map(mx.stop_gradient, gradients)
        mx.eval(loss, materialized)
        total_loss += float(loss.item())
        accumulated = (
            materialized
            if accumulated is None
            else tree_map(
                lambda prior, current: prior + current,
                accumulated,
                materialized,
            )
        )
        mx.eval(accumulated)
        del gradients, materialized, loss
        envelope.reclaim(force=True)
    if accumulated is None:  # pragma: no cover - the spec requires train depths
        raise RuntimeError("streamed recurrence objective emitted no gradients")
    return mx.array(total_loss, dtype=mx.float32), accumulated


def _student_rollin_probability(
    step: int,
    *,
    semantic_warmup_steps: int,
    max_steps: int,
    initial: float,
    final: float,
) -> float:
    if not semantic_warmup_steps <= step < max_steps:
        raise ValueError("student roll-in schedule step is outside recurrent phase")
    recurrent_steps = max_steps - semantic_warmup_steps
    progress = (
        (step - semantic_warmup_steps) / (recurrent_steps - 1) if recurrent_steps > 1 else 1.0
    )
    return float(initial + progress * (final - initial))


def _sha256_tokens(tokens: Any) -> str:
    values = [int(value) for value in tokens.tolist()[0]]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("ascii")).hexdigest()


def _deterministic_student_mix(
    answer_tokens: Any,
    generated_tokens: Any,
    *,
    probability: float,
    seed: int,
    interchangeable_token_ids: frozenset[int] | None = None,
) -> tuple[Any, tuple[int, ...]]:
    """Use generated history without relabeling or corrupting its grammar."""

    if answer_tokens.shape != generated_tokens.shape:
        raise ValueError("generated roll-in must be answer-aligned")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ValueError("student roll-in probability must be inside [0, 1]")
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError("student roll-in seed must be inside [0, 2^64)")
    answer = [int(value) for value in answer_tokens.tolist()[0]]
    generated = [int(value) for value in generated_tokens.tolist()[0]]
    threshold = int(float(probability) * (1 << 64))
    effective: list[int] = []
    selected: list[int] = []
    for position, (target, produced) in enumerate(zip(answer, generated, strict=True)):
        # The final decoder input has no successor label and cannot influence
        # this sequence loss, so leave it canonical.
        digest = hashlib.sha256(
            b"aura.unified.student-rollin.v1\0"
            + seed.to_bytes(8, "big")
            + position.to_bytes(8, "big")
        ).digest()
        use_generated = position + 1 < len(answer) and int.from_bytes(digest[:8], "big") < threshold
        if use_generated and interchangeable_token_ids is not None:
            # A generated digit may replace another digit because this exposes
            # the decoder to a wrong value while preserving the same grammar
            # state.  A digit may not replace syntax (or vice versa): doing so
            # shifts every later role/place target and trains against labels
            # that no longer describe the autoregressive prefix.
            use_generated = produced == target or (
                produced in interchangeable_token_ids and target in interchangeable_token_ids
            )
        effective.append(produced if use_generated else target)
        if use_generated:
            selected.append(position)
    return mx.array([effective], dtype=answer_tokens.dtype), tuple(selected)


def _record_student_rollin(
    totals: dict[str, Any],
    answer_tokens: Any,
    generated_tokens: Any,
    effective_tokens: Any,
    selected: tuple[int, ...],
    probability: float,
) -> None:
    """Record generated-history exposure without treating it as authority."""

    answer = [int(value) for value in answer_tokens.tolist()[0]]
    generated = [int(value) for value in generated_tokens.tolist()[0]]
    totals["examples"] += 1
    totals["answer_tokens"] += len(answer)
    totals["generated_positions"] += len(selected)
    totals["generated_matches"] += sum(generated[index] == answer[index] for index in selected)
    totals["last_generated_sha256"] = _sha256_tokens(generated_tokens)
    totals["last_effective_sha256"] = _sha256_tokens(effective_tokens)
    totals["last_probability"] = probability


def _answer_role_place_targets(
    family: str,
    answer_tokens: Any,
    contract: RecurrentAnswerEmissionContract,
) -> tuple[Any, Any] | None:
    """Label pointer-compatible answers; leave general schemas semantic-only.

    The pointer target means that a digit is an exact rendering of one of the
    five categorical recurrent state slots.  Broad frontier answers contain
    strings, arrays, booleans, and values assembled from multiple process
    fields, so inventing a slot label for them would be false supervision.
    They still train the complete answer bridge through token-level semantic
    cross-entropy; only this optional narrow auxiliary is omitted.
    """

    syntax = dict(contract.syntax)
    layouts = {
        # Role zero means no pointer. Positive role N selects categorical
        # state slot N-1, so public result register 1 is role class 2.
        "khop": ((syntax["khop"], None), ((), 2), (syntax["close"], None)),
        "modular": (
            (syntax["modular"], None),
            ((), 2),
            (syntax["close"], None),
        ),
        "register_trace": (
            (syntax["register_head"], None),
            ((), 2),
            (syntax["register_mid_r1"], None),
            ((), 3),
            (syntax["register_mid_r2"], None),
            ((), 4),
            (syntax["close"], None),
        ),
    }
    if family not in layouts:
        return None
    values = tuple(int(value) for value in answer_tokens.tolist()[0])
    roles = [0] * len(values)
    places = [0] * len(values)
    cursor = 0
    digit_ids = set(contract.digit_token_ids)
    for fixed, role in layouts[family]:
        if role is None:
            stop = cursor + len(fixed)
            if values[cursor:stop] != fixed:
                raise ValueError("answer tokens differ from the canonical grammar")
            cursor = stop
            continue
        start = cursor
        while cursor < len(values) and values[cursor] in digit_ids:
            cursor += 1
        width = cursor - start
        if width not in {1, 2}:
            raise ValueError("answer value is outside the admitted two-digit grammar")
        for position in range(start, cursor):
            roles[position] = role
        if width == 1:
            places[start] = 2
        else:
            places[start] = 1
            places[start + 1] = 2
    if values[cursor:] != (contract.eos_token_id,):
        raise ValueError("answer tokens do not terminate with the bound EOS token")
    return (
        mx.array([roles], dtype=mx.int32),
        mx.array([places], dtype=mx.int32),
    )


def _answer_binding_loss(
    role_logits: Any,
    place_logits: Any,
    role_targets: Any,
    place_targets: Any,
) -> Any:
    """Supervise neural slot selection without exposing answer values."""

    if role_targets.shape != place_targets.shape or len(role_targets.shape) != 2:
        raise ValueError("answer binding targets differ")
    token_count = int(role_targets.shape[-1])
    if (
        len(role_logits.shape) != 3
        or len(place_logits.shape) != 3
        or role_logits.shape[:2] != place_logits.shape[:2]
        or int(role_logits.shape[1]) < token_count
        or int(place_logits.shape[1]) < token_count
    ):
        raise ValueError("answer binding logits differ from target positions")
    role_terms = nn.losses.cross_entropy(
        role_logits[:, :token_count, :].astype(mx.float32),
        role_targets,
        reduction="none",
    )
    place_terms = nn.losses.cross_entropy(
        place_logits[:, :token_count, :].astype(mx.float32),
        place_targets,
        reduction="none",
    )
    role_weights = mx.where(role_targets == 0, 0.25, 1.0)
    place_weights = mx.where(place_targets == 0, 0.25, 1.0)
    role_loss = mx.sum(role_terms * role_weights) / mx.sum(role_weights)
    place_loss = mx.sum(place_terms * place_weights) / mx.sum(place_weights)
    return 0.5 * (role_loss + place_loss)


def _answer_bridge_task(tasks: list[Any], bridge_index: int) -> Any:
    """Cover every family, then every family/depth cell, before repetition."""

    if type(bridge_index) is not int or bridge_index < 0 or not tasks:
        raise ValueError("answer bridge task schedule is invalid")
    cells = sorted({(str(task.family), int(task.depth)) for task in tasks})
    first_by_family = [
        min((cell for cell in cells if cell[0] == family), key=lambda cell: cell[1])
        for family in sorted({family for family, _depth in cells})
    ]
    ordered_cells = first_by_family + [cell for cell in cells if cell not in first_by_family]
    cell = ordered_cells[bridge_index % len(ordered_cells)]
    cell_tasks = [task for task in tasks if (str(task.family), int(task.depth)) == cell]
    cycle = bridge_index // len(ordered_cells)
    return cell_tasks[cycle % len(cell_tasks)]


def _recurrent_training_task(
    tasks: list[Any],
    tokenizer: Any,
    bridge: str,
    recurrence_index: int,
    *,
    cover_all_cells: bool = False,
) -> Any:
    """Choose recurrence examples in deterministic memory-cost order.

    The historical closed curriculum emphasizes its maximum-depth cells. Broad
    process training must cover every family/depth cell because its domains have
    intentionally different natural program lengths.
    """

    if type(recurrence_index) is not int or recurrence_index < 0 or not tasks:
        raise ValueError("recurrent training task schedule is invalid")
    max_depth = max(int(task.depth) for task in tasks)
    ranked: list[tuple[int, str, str, Any]] = []
    for task in tasks:
        if not cover_all_cells and int(task.depth) != max_depth:
            continue
        prompt, answer = encode_example(tokenizer, task, bridge)
        ranked.append(
            (
                int(prompt.shape[-1]) + int(answer.shape[-1]),
                str(task.family),
                str(task.task_id),
                task,
            )
        )
    ranked.sort(key=lambda row: row[:3])
    return ranked[recurrence_index % len(ranked)][-1]


def _process_family_training_batch(
    tasks: list[Any],
    update_index: int,
    batch_size: int,
    *,
    mode: str = "same_family",
) -> tuple[Any, ...]:
    """Return a deterministic process cohort for one optimizer update.

    A family-specific head must fit several distinct programs at once. Applying
    one update per example allowed the last of two prototypes to overwrite its
    sibling. The shared transformer tissue has the complementary requirement:
    each update must represent every family so its parser cannot chase the
    latest domain. Both modes retain exact per-example execution traces.
    """

    if (
        type(update_index) is not int
        or update_index < 0
        or type(batch_size) is not int
        or batch_size < 1
        or not tasks
        or mode not in {"same_family", "balanced_families"}
    ):
        raise ValueError("process family batch schedule is invalid")
    by_family: dict[str, list[Any]] = {}
    for task in tasks:
        by_family.setdefault(str(task.family), []).append(task)
    families = sorted(by_family)
    for items in by_family.values():
        items.sort(key=lambda item: str(item.task_id))
    if mode == "balanced_families":
        if batch_size != len(families):
            raise ValueError("balanced process batch must contain exactly one example per family")
        return tuple(
            by_family[family][update_index % len(by_family[family])] for family in families
        )
    if any(len(items) < batch_size for items in by_family.values()):
        raise ValueError("process family batch exceeds an available family")
    family = families[update_index % len(families)]
    items = by_family[family]
    family_update = update_index // len(families)
    start = (family_update * batch_size) % len(items)
    return tuple(items[(start + offset) % len(items)] for offset in range(batch_size))


def _dual_ridge_residual_readout(
    features: Any,
    base_logits: Any,
    labels: Any,
    *,
    regularization: float,
    margin: float,
) -> tuple[Any, Any, dict[str, Any]]:
    """Fit an affine residual classifier without opening validation labels."""

    import numpy as np

    x = np.asarray(features, dtype=np.float64)
    base = np.asarray(base_logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if (
        x.ndim != 2
        or base.ndim != 2
        or target.ndim != 1
        or len(x) < 1
        or len(x) != len(base)
        or len(x) != len(target)
        or base.shape[1] < 2
        or np.any(target < 0)
        or np.any(target >= base.shape[1])
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(base))
    ):
        raise ValueError("analytic action readout observations are invalid")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = (x - mean) / scale
    design = np.concatenate(
        (normalized, np.ones((len(x), 1), dtype=np.float64)),
        axis=1,
    )
    desired = np.full((len(x), base.shape[1]), -float(margin), dtype=np.float64)
    desired[np.arange(len(x)), target] = float(margin)
    residual = desired - base
    width = float(design.shape[1])
    gram = (design @ design.T) / width
    ridge = float(regularization) * max(float(np.trace(gram)) / len(x), 1e-9)
    dual = np.linalg.solve(gram + ridge * np.eye(len(x)), residual)
    coefficients = (design.T @ dual) / width
    normalized_weight = coefficients[:-1]
    raw_weight = normalized_weight / scale[:, None]
    raw_bias = coefficients[-1] - (mean / scale) @ normalized_weight
    fitted = base + x @ raw_weight + raw_bias
    report = {
        "observations": len(x),
        "before_accuracy": float(np.mean(np.argmax(base, axis=1) == target)),
        "after_accuracy": float(np.mean(np.argmax(fitted, axis=1) == target)),
        "ridge": ridge,
    }
    return raw_weight.astype(np.float32), raw_bias.astype(np.float32), report


def _rbf_residual_readout(
    features: Any,
    base_logits: Any,
    labels: Any,
    *,
    capacity: int,
    regularization: float,
    margin: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a bounded nonlinear residual over training observations only."""

    import numpy as np

    x = np.asarray(features, dtype=np.float64)
    base = np.asarray(base_logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if (
        x.ndim != 2
        or base.ndim != 2
        or target.ndim != 1
        or len(x) < 1
        or len(x) > capacity
        or len(x) != len(base)
        or len(x) != len(target)
    ):
        raise ValueError("kernel action readout observations are invalid")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = (x - mean) / scale
    distance = np.mean(
        np.square(normalized[:, None, :] - normalized[None, :, :]),
        axis=-1,
    )
    nonzero = distance[distance > 1e-10]
    gamma = (
        float(np.log(2.0) / np.median(nonzero))
        if nonzero.size
        else 1.0
    )
    kernel = np.exp(-gamma * distance)
    desired = np.full((len(x), base.shape[1]), -float(margin), dtype=np.float64)
    desired[np.arange(len(x)), target] = float(margin)
    ridge = float(regularization) * max(float(np.trace(kernel)) / len(x), 1e-9)
    coefficients = np.linalg.solve(
        kernel + ridge * np.eye(len(x)),
        desired - base,
    )
    fitted = base + kernel @ coefficients
    prototypes = np.zeros((capacity, x.shape[1]), dtype=np.float32)
    output = np.zeros((capacity, base.shape[1]), dtype=np.float32)
    mask = np.zeros((capacity,), dtype=np.float32)
    prototypes[: len(x)] = normalized.astype(np.float32)
    output[: len(x)] = coefficients.astype(np.float32)
    mask[: len(x)] = 1.0
    parameters = {
        "mean": mean.astype(np.float32),
        "inv_scale": (1.0 / scale).astype(np.float32),
        "prototypes": prototypes,
        "coefficients": output,
        "mask": mask,
        "gamma": np.float32(gamma),
    }
    report = {
        "observations": len(x),
        "before_accuracy": float(np.mean(np.argmax(base, axis=1) == target)),
        "after_accuracy": float(np.mean(np.argmax(fitted, axis=1) == target)),
        "gamma": gamma,
        "ridge": ridge,
    }
    return parameters, report


def _fit_family_action_readout(
    bundle: UnifiedTrainingBundle,
    tokenizer: Any,
    tasks: list[Any],
    spec: UnifiedIntrinsicTrainingSpec,
    bridge: str,
    family_contract: Any,
    *,
    regularization: float,
    margin: float,
) -> dict[str, Any]:
    """Write the exact training-only workspace-to-instruction residual map."""

    import numpy as np

    if (
        not tasks
        or isinstance(regularization, bool)
        or not isinstance(regularization, (int, float))
        or not math.isfinite(float(regularization))
        or float(regularization) <= 0.0
        or isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or float(margin) <= 0.0
    ):
        raise ValueError("analytic family action readout configuration is invalid")
    controller = bundle.controller
    expert_count = int(controller.action_family_output.shape[0])
    slot_count = int(controller.action_family_output.shape[1])
    workspace_width = int(controller.action_family_output.shape[2])
    cardinality = int(controller.action_family_output.shape[3])
    kernel_capacity = int(controller.action_family_kernel_prototypes.shape[2])
    observations: dict[tuple[int, int], list[tuple[Any, Any, Any, int]]] = {}
    instruction_rows: list[
        tuple[int, list[tuple[int, Any, Any, Any, int]]]
    ] = []
    task_commitments: list[dict[str, Any]] = []
    depth = max(spec.train_depths)

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    with recurrence_adapter_scope(start=None, stop=None):
        for task in sorted(tasks, key=lambda item: (str(item.family), str(item.task_id))):
            prompt, _answer = encode_example(tokenizer, task, bridge)
            opcodes, recognized = family_contract.observe(prompt.tolist())
            if len(opcodes) != 1 or not recognized[0]:
                raise RuntimeError("analytic action readout task family is unavailable")
            expert = int(opcodes[0]) - OP_FRONTIER_TRAVERSE
            if not 0 <= expert < expert_count:
                raise RuntimeError("analytic action readout expert route differs")
            state_targets = state_targets_from_trace(
                task.transition_trace,
                depth,
                state_slots=controller.config.state_slots,
            )
            action_targets = action_targets_from_program(task.transition_program, depth)
            workspaces: list[Any] = []
            kernel_features: list[Any] = []
            base_logits: list[Any] = []
            unified_recurrent_hidden_states(
                bundle.model,
                prompt,
                spec.plan_at(depth),
                controller,
                state_slot_start=int(prompt.shape[-1]),
                action_logit_trajectory=base_logits,
                action_workspace_trajectory=workspaces,
                action_kernel_feature_trajectory=kernel_features,
                state_teacher_values=state_targets.values,
                action_teacher_values=action_targets.values,
                initial_state_teacher_values=state_targets.initial_values,
                state_teacher_forcing_probability=1.0,
                family_action_lesion=True,
                process_only=True,
            )
            if (
                len(workspaces) != depth
                or len(kernel_features) != depth
                or len(base_logits) != depth
            ):
                raise RuntimeError("analytic action readout capture differs from the process")
            mx.eval(*workspaces, *kernel_features, *base_logits)
            task_rows: list[tuple[int, Any, Any, Any, int]] = []
            for step, (workspace, kernel_feature, logits, values, masks) in enumerate(
                zip(
                    workspaces,
                    kernel_features,
                    base_logits,
                    action_targets.values,
                    action_targets.masks,
                    strict=True,
                )
            ):
                for slot, (target, active) in enumerate(zip(values, masks, strict=True)):
                    if not active:
                        continue
                    feature = np.asarray(workspace[0, slot], dtype=np.float64)
                    structured = np.asarray(
                        kernel_feature[0, slot], dtype=np.float64
                    )
                    base = np.asarray(logits[0, slot], dtype=np.float64)
                    observations.setdefault((expert, slot), []).append(
                        (feature, structured, base, int(target))
                    )
                    task_rows.append(
                        (
                            step * slot_count + slot,
                            feature,
                            structured,
                            base,
                            int(target),
                        )
                    )
            instruction_rows.append((expert, task_rows))
            task_commitments.append(
                {
                    "task_id": str(task.task_id),
                    "family": str(task.family),
                    "program_sha256": action_targets.program_sha256,
                    "target_sha256": action_targets.target_sha256,
                }
            )
            mx.clear_cache()

    fitted_weights = np.zeros(
        (expert_count, slot_count, workspace_width, cardinality),
        dtype=np.float32,
    )
    fitted_bias = np.zeros(
        (expert_count, slot_count, cardinality),
        dtype=np.float32,
    )
    kernel_mean = np.zeros(
        (expert_count, slot_count, workspace_width), dtype=np.float32
    )
    kernel_inv_scale = np.zeros_like(kernel_mean)
    kernel_prototypes = np.zeros(
        (expert_count, slot_count, kernel_capacity, workspace_width),
        dtype=np.float32,
    )
    kernel_coefficients = np.zeros(
        (expert_count, slot_count, kernel_capacity, cardinality),
        dtype=np.float32,
    )
    kernel_mask = np.zeros(
        (expert_count, slot_count, kernel_capacity), dtype=np.float32
    )
    kernel_gamma = np.zeros((expert_count, slot_count), dtype=np.float32)
    cell_receipts: dict[str, Any] = {}
    for (expert, slot), rows in sorted(observations.items()):
        features = np.stack([row[0] for row in rows])
        structured_features = np.stack([row[1] for row in rows])
        bases = np.stack([row[2] for row in rows])
        labels = np.asarray([row[3] for row in rows], dtype=np.int64)
        raw_weight, raw_bias, cell_report = _dual_ridge_residual_readout(
            features,
            bases,
            labels,
            regularization=float(regularization),
            margin=float(margin),
        )
        fitted_weights[expert, slot] = raw_weight
        fitted_bias[expert, slot] = raw_bias
        linear_logits = bases + features @ raw_weight + raw_bias
        kernel_parameters, kernel_report = _rbf_residual_readout(
            structured_features,
            linear_logits,
            labels,
            capacity=kernel_capacity,
            regularization=float(regularization),
            margin=float(margin),
        )
        kernel_mean[expert, slot] = kernel_parameters["mean"]
        kernel_inv_scale[expert, slot] = kernel_parameters["inv_scale"]
        kernel_prototypes[expert, slot] = kernel_parameters["prototypes"]
        kernel_coefficients[expert, slot] = kernel_parameters["coefficients"]
        kernel_mask[expert, slot] = kernel_parameters["mask"]
        kernel_gamma[expert, slot] = kernel_parameters["gamma"]
        cell_receipts[f"expert-{expert}:slot-{slot}"] = {
            "linear": cell_report,
            "kernel": kernel_report,
        }

    before_correct = 0
    after_correct = 0
    total = 0
    for expert, rows in instruction_rows:
        for coordinate, feature, structured, base, target in rows:
            slot = coordinate % slot_count
            before_correct += int(np.argmax(base) == target)
            corrected = (
                base
                + feature @ fitted_weights[expert, slot]
                + fitted_bias[expert, slot]
            )
            normalized = (
                structured - kernel_mean[expert, slot]
            ) * kernel_inv_scale[expert, slot]
            distances = np.mean(
                np.square(
                    normalized[None, :]
                    - kernel_prototypes[expert, slot]
                ),
                axis=-1,
            )
            kernel = (
                np.exp(-kernel_gamma[expert, slot] * distances)
                * kernel_mask[expert, slot]
            )
            corrected = (
                corrected
                + kernel @ kernel_coefficients[expert, slot]
            )
            after_correct += int(np.argmax(corrected) == target)
            total += 1
    before_sha256 = controller.parameter_sha256()
    controller.action_family_output = mx.array(fitted_weights, dtype=mx.float32)
    controller.action_family_bias = mx.array(fitted_bias, dtype=mx.float32)
    controller.action_family_kernel_mean = mx.array(kernel_mean, dtype=mx.float32)
    controller.action_family_kernel_inv_scale = mx.array(
        kernel_inv_scale, dtype=mx.float32
    )
    controller.action_family_kernel_prototypes = mx.array(
        kernel_prototypes, dtype=mx.float32
    )
    controller.action_family_kernel_coefficients = mx.array(
        kernel_coefficients, dtype=mx.float32
    )
    controller.action_family_kernel_mask = mx.array(kernel_mask, dtype=mx.float32)
    controller.action_family_kernel_gamma = mx.array(kernel_gamma, dtype=mx.float32)
    mx.eval(
        controller.action_family_output,
        controller.action_family_bias,
        controller.action_family_kernel_mean,
        controller.action_family_kernel_inv_scale,
        controller.action_family_kernel_prototypes,
        controller.action_family_kernel_coefficients,
        controller.action_family_kernel_mask,
        controller.action_family_kernel_gamma,
    )
    after_sha256 = controller.parameter_sha256()
    body = {
        "schema": "aura.unified_intrinsic.analytic_action_readout_fit.v1",
        "method": "training_only_affine_workspace_plus_public_signature_rbf_write",
        "affine_feature_source": "learned_recurrent_workspace",
        "kernel_feature_source": (
            "ordered_public_literals_current_state_depth_and_family"
        ),
        "regularization": float(regularization),
        "target_logit_margin": float(margin),
        "training_tasks": len(task_commitments),
        "holdout_tasks_accessed": 0,
        "observations": total,
        "before_accuracy": before_correct / total,
        "after_accuracy": after_correct / total,
        "controller_sha256_before": before_sha256,
        "controller_sha256_after": after_sha256,
        "task_commitments": task_commitments,
        "cells": cell_receipts,
        "private_targets_serialized_into_runtime_input": False,
    }
    return {**body, "receipt_sha256": _canonical_sha256(body)}


def _mean_gradient_trees(samples: list[Any]) -> Any:
    """Average compatible materialized gradient trees without hidden weighting."""

    if not samples:
        raise ValueError("gradient cohort is empty")
    flattened = [tree_flatten(sample) for sample in samples]
    signature = [(name, tuple(value.shape)) for name, value in flattened[0]]
    if any(
        [(name, tuple(value.shape)) for name, value in candidate] != signature
        for candidate in flattened[1:]
    ):
        raise ValueError("gradient cohort parameter topology differs")
    count = float(len(flattened))
    return tree_unflatten(
        [
            (
                name,
                sum(candidate[index][1] for candidate in flattened) / count,
            )
            for index, (name, _value) in enumerate(flattened[0])
        ]
    )


def _gradient_conflict_diagnostics(
    samples: list[Any],
    labels: list[str],
    *,
    ownership_group: str,
) -> dict[str, Any]:
    """Measure cross-task gradient alignment without inventing an SNR model."""

    if (
        not samples
        or len(samples) != len(labels)
        or len(set(labels)) != len(labels)
        or not ownership_group
    ):
        raise ValueError("gradient conflict cohort differs")
    flattened = [dict(tree_flatten(sample)) for sample in samples]
    names = sorted(
        name
        for name in flattened[0]
        if _gradient_ownership_group(name) == ownership_group
    )
    if not names or any(set(candidate) != set(flattened[0]) for candidate in flattened[1:]):
        raise ValueError("gradient conflict topology differs")
    norms = [
        mx.sqrt(
            mx.sum(
                mx.stack(
                    [
                        mx.sum(candidate[name].astype(mx.float32) ** 2)
                        for name in names
                    ]
                )
            )
        )
        for candidate in flattened
    ]
    pairs: list[dict[str, Any]] = []
    for left_index in range(len(flattened)):
        for right_index in range(left_index + 1, len(flattened)):
            denominator = norms[left_index] * norms[right_index]
            dot = mx.sum(
                mx.stack(
                    [
                        mx.sum(
                            flattened[left_index][name].astype(mx.float32)
                            * flattened[right_index][name].astype(mx.float32)
                        )
                        for name in names
                    ]
                )
            )
            mx.eval(denominator, dot)
            measured = float(denominator.item()) > 1e-12
            pairs.append(
                {
                    "left": labels[left_index],
                    "right": labels[right_index],
                    "cosine": float((dot / denominator).item()) if measured else None,
                    "measured": measured,
                }
            )
    mx.eval(*norms)
    measured_cosines = [
        float(pair["cosine"]) for pair in pairs if pair["cosine"] is not None
    ]
    return {
        "ownership_group": ownership_group,
        "parameter_count": len(names),
        "norms": {
            label: float(norm.item()) for label, norm in zip(labels, norms, strict=True)
        },
        "pairs": pairs,
        "measured_pairs": len(measured_cosines),
        "negative_pairs": sum(value < 0.0 for value in measured_cosines),
        "minimum_cosine": min(measured_cosines) if measured_cosines else None,
        "mean_cosine": (
            sum(measured_cosines) / len(measured_cosines)
            if measured_cosines
            else None
        ),
    }


def _combine_process_gradient_trees(
    samples: list[Any],
    labels: list[str],
    *,
    mode: str,
    ownership_group: str,
) -> tuple[Any, dict[str, Any]]:
    """Combine task gradients while removing only measured negative conflicts."""

    if (
        not samples
        or len(samples) != len(labels)
        or len(set(labels)) != len(labels)
        or mode not in {"mean", "balanced_mean", "pcgrad"}
        or not ownership_group
    ):
        raise ValueError("process gradient combiner cohort differs")
    if mode == "mean":
        return _mean_gradient_trees(samples), {
            "mode": mode,
            "ownership_group": ownership_group,
            "projection_count": 0,
            "projection_events": [],
            "unowned_parameters_combined_by": "arithmetic_mean",
        }

    original = [dict(tree_flatten(sample)) for sample in samples]
    names = [name for name, _value in tree_flatten(samples[0])]
    if any(list(candidate) != names for candidate in original[1:]):
        raise ValueError("process gradient combiner topology differs")
    owned = [
        name for name in names if _gradient_ownership_group(name) == ownership_group
    ]
    if not owned:
        raise ValueError("process gradient combiner owns no parameters")
    if mode == "balanced_mean":
        squared_norms = [
            mx.sum(
                mx.stack(
                    [
                        mx.sum(candidate[name].astype(mx.float32) ** 2)
                        for name in owned
                    ]
                )
            )
            for candidate in original
        ]
        mx.eval(*squared_norms)
        norms = [math.sqrt(max(0.0, float(value.item()))) for value in squared_norms]
        nonzero = [value for value in norms if value > 1e-12]
        target_norm = sum(nonzero) / len(nonzero) if nonzero else 0.0
        scales = [target_norm / value if value > 1e-12 else 0.0 for value in norms]
        count = float(len(original))
        combined = tree_unflatten(
            [
                (
                    name,
                    sum(
                        candidate[name] * (
                            scales[index]
                            if name in owned
                            else 1.0
                        )
                        for index, candidate in enumerate(original)
                    )
                    / count,
                )
                for name in names
            ]
        )
        return combined, {
            "mode": mode,
            "ownership_group": ownership_group,
            "target_owned_norm": target_norm,
            "owned_norms": dict(zip(labels, norms, strict=True)),
            "owned_scales": dict(zip(labels, scales, strict=True)),
            "projection_count": 0,
            "projection_events": [],
            "unowned_parameters_combined_by": "arithmetic_mean",
        }
    projected = [dict(candidate) for candidate in original]
    events: list[dict[str, Any]] = []
    for left_index in range(len(projected)):
        for right_index in range(len(original)):
            if left_index == right_index:
                continue
            dot = mx.sum(
                mx.stack(
                    [
                        mx.sum(
                            projected[left_index][name].astype(mx.float32)
                            * original[right_index][name].astype(mx.float32)
                        )
                        for name in owned
                    ]
                )
            )
            right_norm_squared = mx.sum(
                mx.stack(
                    [
                        mx.sum(original[right_index][name].astype(mx.float32) ** 2)
                        for name in owned
                    ]
                )
            )
            mx.eval(dot, right_norm_squared)
            dot_value = float(dot.item())
            norm_value = float(right_norm_squared.item())
            if dot_value >= 0.0 or norm_value <= 1e-12:
                continue
            coefficient = dot / right_norm_squared
            for name in owned:
                projected[left_index][name] = (
                    projected[left_index][name]
                    - coefficient * original[right_index][name]
                )
            events.append(
                {
                    "left": labels[left_index],
                    "right": labels[right_index],
                    "dot_before": dot_value,
                    "right_norm_squared": norm_value,
                }
            )
    count = float(len(projected))
    combined = tree_unflatten(
        [
            (name, sum(candidate[name] for candidate in projected) / count)
            for name in names
        ]
    )
    return combined, {
        "mode": mode,
        "ownership_group": ownership_group,
        "projection_count": len(events),
        "projection_events": events,
        "unowned_parameters_combined_by": "arithmetic_mean",
    }


def _cached_answer_binding_features(
    bundle: UnifiedTrainingBundle,
    prompt: Any,
    answer_tokens: Any,
    plan: Any,
    *,
    initial_state_teacher_values: tuple[int, ...] | None = None,
    state_teacher_values: tuple[tuple[int, ...], ...] | None = None,
    action_teacher_values: tuple[tuple[int, ...], ...] | None = None,
    state_teacher_forcing_probability: float = 0.0,
) -> tuple[Any, Any, Any]:
    """Run the expensive tissue once and detach its causal binding features."""

    full = mx.concatenate([prompt, answer_tokens], axis=1)
    features: list[tuple[Any, Any, Any]] = []
    unified_recurrent_hidden_states(
        bundle.model,
        full,
        plan,
        bundle.controller,
        state_slot_start=int(prompt.shape[-1]),
        answer_binding_feature_trajectory=features,
        initial_state_teacher_values=initial_state_teacher_values,
        state_teacher_values=state_teacher_values,
        action_teacher_values=action_teacher_values,
        state_teacher_forcing_probability=state_teacher_forcing_probability,
    )
    if not features:
        raise RuntimeError("answer bridge emitted no reusable causal features")
    selected = tuple(mx.stop_gradient(value) for value in features[-1])
    mx.eval(*selected)
    return selected


def _capture_autonomous_process(
    bundle: UnifiedTrainingBundle,
    prompt: Any,
    plan: Any,
    *,
    public_action_values: tuple[tuple[int, ...], ...] | None = None,
    microcode_lesion: bool = False,
    transition_processor_lesion: bool = False,
    transition_processor_mode: str = "residual",
    transition_copy_prior_logit_bias: float = TRANSITION_COPY_PRIOR_LOGIT_BIAS,
    transition_opcode_expert_routing: str = "opcode",
    transition_replay_mode: str = "disabled",
    transition_history_lesion: bool = False,
) -> dict[str, Any]:
    """Capture one prefix-only autonomous process execution without decoding."""

    initial_state_logits: list[Any] = []
    state_logits: list[Any] = []
    action_logits: list[Any] = []
    unified_recurrent_hidden_states(
        bundle.model,
        prompt,
        plan,
        bundle.controller,
        state_slot_start=int(prompt.shape[-1]),
        initial_state_logit_trajectory=initial_state_logits,
        state_logit_trajectory=state_logits,
        action_logit_trajectory=action_logits,
        public_action_values=public_action_values,
        microcode_lesion=microcode_lesion,
        transition_processor_lesion=transition_processor_lesion,
        transition_processor_mode=transition_processor_mode,
        transition_copy_prior_logit_bias=transition_copy_prior_logit_bias,
        transition_opcode_expert_routing=transition_opcode_expert_routing,
        transition_replay_mode=transition_replay_mode,
        transition_history_lesion=transition_history_lesion,
    )
    if len(initial_state_logits) != 1:
        raise RuntimeError("autonomous process emitted no initial state decision")
    mx.eval(initial_state_logits[0], *state_logits, *action_logits)
    if public_action_values is not None:
        action_logits = [
            bundle.controller.exact_probabilities(
                row,
                slots=bundle.controller.config.action_slots,
                cardinality=bundle.controller.config.action_cardinality,
            )
            for row in public_action_values
        ]
        mx.eval(*action_logits)
    return {
        "initial_state_logits": initial_state_logits[0],
        "state_logits": tuple(state_logits),
        "action_logits": tuple(action_logits),
    }


def _public_actions_for_task(task: Any, depth: int) -> tuple[tuple[int, ...], ...]:
    """Compile the task's public prompt without opening verifier evidence."""

    program = compile_public_frontier_actions(str(task.prompt), str(task.family))
    return program.values_for_iterations(depth)


def _answer_bridge_teacher_policy(
    step: int,
    *,
    bridge_start: int,
    bridge_steps: int,
    autonomous_tail_steps: int,
    process_exact: bool,
) -> dict[str, Any]:
    """Keep bad autonomous traces from becoming answer-bridge supervision."""

    if (
        type(step) is not int
        or type(bridge_start) is not int
        or type(bridge_steps) is not int
        or type(autonomous_tail_steps) is not int
        or type(process_exact) is not bool
        or bridge_steps < 1
        or not 1 <= autonomous_tail_steps <= bridge_steps
        or not bridge_start <= step < bridge_start + bridge_steps
    ):
        raise ValueError("answer bridge teacher policy coordinates are invalid")
    tail_start = bridge_start + bridge_steps - autonomous_tail_steps
    in_autonomous_tail = step >= tail_start
    if not in_autonomous_tail:
        probability = 1.0
    elif autonomous_tail_steps == 1:
        probability = 0.0
    else:
        tail_index = step - tail_start
        probability = 1.0 - tail_index / (autonomous_tail_steps - 1)
    return {
        "state_teacher_forcing_probability": probability,
        "autonomous_tail": in_autonomous_tail,
        "update_admitted": not in_autonomous_tail or process_exact,
    }


def _cached_answer_binding_loss(
    bundle: UnifiedTrainingBundle,
    features: tuple[Any, Any, Any],
    targets: tuple[Any, Any],
) -> Any:
    role_logits, place_logits = bundle.controller.answer_binding_logits(*features)
    return _answer_binding_loss(role_logits, place_logits, *targets)


def _generate_student_rollin(
    bundle: UnifiedTrainingBundle,
    prompt: Any,
    answer_tokens: Any,
    plan: Any,
    *,
    eos_token_id: int | None,
    answer_emission_contract: RecurrentAnswerEmissionContract | None = None,
    answer_digit_pointer_enabled: bool = True,
    state_slot_start: int | None = None,
    process_capture: dict[str, Any] | None = None,
    initial_state_teacher_values: tuple[int, ...] | None = None,
    state_teacher_values: tuple[tuple[int, ...], ...] | None = None,
    action_teacher_values: tuple[tuple[int, ...], ...] | None = None,
    state_teacher_forcing_probability: float = 0.0,
    process_tape_lesion: bool = False,
) -> Any:
    """Greedily materialize a fixed-length deep-policy decoder history."""

    token_count = int(answer_tokens.shape[-1])
    if token_count < 1:
        raise ValueError("student roll-in target must not be empty")
    tokens = prompt
    generated: list[int] = []
    stopped = False
    for position in range(token_count):
        if stopped and eos_token_id is not None:
            token = int(eos_token_id)
        else:
            initial_state_logits: list[Any] = []
            state_logits: list[Any] = []
            action_logits: list[Any] = []
            logits, _telemetry = unified_recurrent_logits(
                bundle.model,
                tokens,
                plan,
                bundle.controller,
                state_slot_start=state_slot_start,
                answer_emission_contract=answer_emission_contract,
                answer_digit_pointer_enabled=answer_digit_pointer_enabled,
                initial_state_logit_trajectory=(
                    initial_state_logits if process_capture is not None and position == 0 else None
                ),
                state_logit_trajectory=(
                    state_logits if process_capture is not None and position == 0 else None
                ),
                action_logit_trajectory=(
                    action_logits if process_capture is not None and position == 0 else None
                ),
                initial_state_teacher_values=initial_state_teacher_values,
                state_teacher_values=state_teacher_values,
                action_teacher_values=action_teacher_values,
                state_teacher_forcing_probability=state_teacher_forcing_probability,
                process_tape_lesion=process_tape_lesion,
            )
            if process_capture is not None and position == 0:
                if len(initial_state_logits) != 1:
                    raise RuntimeError("student roll-in emitted no initial state decision")
                mx.eval(
                    initial_state_logits[0],
                    *state_logits,
                    *action_logits,
                )
                process_capture.update(
                    {
                        "initial_state_logits": initial_state_logits[0],
                        "state_logits": tuple(state_logits),
                        "action_logits": tuple(action_logits),
                    }
                )
            token = int(mx.argmax(logits[0, -1]).item())
            stopped = eos_token_id is not None and token == eos_token_id
        generated.append(token)
        tokens = mx.concatenate(
            [tokens, mx.array([[token]], dtype=tokens.dtype)],
            axis=1,
        )
    return mx.array([generated], dtype=answer_tokens.dtype)


def _masked_process_decisions(
    logits: tuple[Any, ...],
    values: tuple[tuple[int, ...], ...],
    masks: tuple[tuple[bool, ...], ...],
) -> dict[str, Any]:
    """Measure exact typed decisions without serializing evaluator labels."""

    if not logits or len(logits) != len(values) or len(values) != len(masks):
        raise ValueError("process decision trajectory differs from verifier targets")
    correct = 0
    required = 0
    exact_steps = 0
    required_steps = 0
    predictions: list[list[int]] = []
    for decision, expected, active in zip(logits, values, masks, strict=True):
        predicted = tuple(int(value) for value in mx.argmax(decision[0], axis=-1).tolist())
        if len(predicted) != len(expected) or len(expected) != len(active):
            raise ValueError("process decision width differs from verifier targets")
        step_required = sum(active)
        step_correct = sum(
            enabled and observed == target
            for observed, target, enabled in zip(predicted, expected, active, strict=True)
        )
        required += step_required
        correct += step_correct
        if step_required > 0:
            required_steps += 1
            exact_steps += int(step_correct == step_required)
        predictions.append(list(predicted))
    return {
        "correct": correct,
        "required": required,
        "accuracy": correct / required if required else None,
        "exact_steps": exact_steps,
        "required_steps": required_steps,
        "steps": len(values),
        "exact": required > 0 and correct == required and exact_steps == required_steps,
        "prediction_sha256": _canonical_sha256(predictions),
    }


def _process_evidence_from_capture(
    task: Any,
    depth: int,
    capture: dict[str, Any],
) -> dict[str, Any]:
    """Bind autonomous process correctness to the private exact trace authority."""

    state_logits = tuple(capture["state_logits"])
    if not state_logits:
        raise RuntimeError("autonomous process capture emitted no state decisions")
    state_targets = state_targets_from_trace(
        task.transition_trace,
        depth,
        state_slots=int(state_logits[0].shape[-2]),
    )
    action_targets = action_targets_from_program(task.transition_program, depth)
    initial = _masked_process_decisions(
        (capture["initial_state_logits"],),
        (state_targets.initial_values,),
        (state_targets.initial_masks,),
    )
    states = _masked_process_decisions(
        tuple(capture["state_logits"]),
        state_targets.values,
        state_targets.masks,
    )
    actions = _masked_process_decisions(
        tuple(capture["action_logits"]),
        action_targets.values,
        action_targets.masks,
    )
    body = {
        "schema": "aura.unified_intrinsic.autonomous_process_evidence.v1",
        "depth": depth,
        "initial_state": initial,
        "states": states,
        "actions": actions,
        "trace_sha256": state_targets.trace_sha256,
        "state_target_sha256": state_targets.target_sha256,
        "program_sha256": action_targets.program_sha256,
        "action_target_sha256": action_targets.target_sha256,
        "process_exact": initial["exact"] and states["exact"] and actions["exact"],
        "private_values_exposed": False,
    }
    return {**body, "evidence_sha256": _canonical_sha256(body)}


def _answer_bridge_diagnostic_tasks(tasks: list[Any]) -> list[Any]:
    """Choose one deepest deterministic development example per family."""

    selected: dict[str, Any] = {}
    for task in sorted(
        tasks, key=lambda item: (str(item.family), int(item.depth), str(item.task_id))
    ):
        selected[str(task.family)] = task
    if not selected:
        raise ValueError("answer bridge diagnostic requires development tasks")
    return [selected[family] for family in sorted(selected)]


def _evaluate_answer_bridge_diagnostic(
    bundle: UnifiedTrainingBundle,
    tokenizer: Any,
    tasks: list[Any],
    spec: UnifiedIntrinsicTrainingSpec,
    bridge: str,
    contract: RecurrentAnswerEmissionContract,
    *,
    answer_digit_pointer_enabled: bool = True,
) -> dict[str, Any]:
    """Separate process, reader, and causal-tape failures on development tasks."""

    rows: list[dict[str, Any]] = []
    depth = max(spec.train_depths)
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    with recurrence_adapter_scope(start=None, stop=None):
        for task in _answer_bridge_diagnostic_tasks(tasks):
            prompt, answer = encode_example(tokenizer, task, bridge)
            process_capture: dict[str, Any] = {}
            autonomous = _generate_student_rollin(
                bundle,
                prompt,
                answer,
                spec.plan_at(depth),
                eos_token_id=tokenizer.eos_token_id,
                answer_emission_contract=contract,
                answer_digit_pointer_enabled=answer_digit_pointer_enabled,
                state_slot_start=int(prompt.shape[-1]),
                process_capture=process_capture,
            )
            process = _process_evidence_from_capture(task, depth, process_capture)
            state_targets = state_targets_from_trace(
                task.transition_trace,
                depth,
                state_slots=bundle.controller.config.state_slots,
            )
            action_targets = action_targets_from_program(task.transition_program, depth)
            oracle = _generate_student_rollin(
                bundle,
                prompt,
                answer,
                spec.plan_at(depth),
                eos_token_id=tokenizer.eos_token_id,
                answer_emission_contract=contract,
                answer_digit_pointer_enabled=answer_digit_pointer_enabled,
                state_slot_start=int(prompt.shape[-1]),
                initial_state_teacher_values=state_targets.initial_values,
                state_teacher_values=state_targets.values,
                action_teacher_values=action_targets.values,
                state_teacher_forcing_probability=1.0,
            )
            sham = _generate_student_rollin(
                bundle,
                prompt,
                answer,
                spec.plan_at(depth),
                eos_token_id=tokenizer.eos_token_id,
                answer_emission_contract=contract,
                answer_digit_pointer_enabled=answer_digit_pointer_enabled,
                state_slot_start=int(prompt.shape[-1]),
                process_tape_lesion=True,
            )
            expected = tuple(int(value) for value in answer.tolist()[0])
            arms = {
                "oracle": tuple(int(value) for value in oracle.tolist()[0]),
                "autonomous": tuple(int(value) for value in autonomous.tolist()[0]),
                "sham": tuple(int(value) for value in sham.tolist()[0]),
            }
            rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "task_depth": task.depth,
                    "process_exact": process["process_exact"],
                    "process_evidence_sha256": process["evidence_sha256"],
                    "oracle_exact": arms["oracle"] == expected,
                    "autonomous_exact": arms["autonomous"] == expected,
                    "sham_exact": arms["sham"] == expected,
                    "expected_sha256": _sha256_tokens(answer),
                    "oracle_sha256": _sha256_tokens(oracle),
                    "autonomous_sha256": _sha256_tokens(autonomous),
                    "sham_sha256": _sha256_tokens(sham),
                }
            )
    tasks_count = len(rows)
    oracle_exact = sum(row["oracle_exact"] for row in rows)
    autonomous_process_exact = sum(row["process_exact"] for row in rows)
    autonomous_exact = sum(row["autonomous_exact"] for row in rows)
    sham_exact = sum(row["sham_exact"] for row in rows)
    if oracle_exact < tasks_count:
        diagnosis = "reader_or_bridge_limited"
    elif autonomous_process_exact < tasks_count:
        diagnosis = "recurrent_process_limited"
    elif autonomous_exact < tasks_count:
        diagnosis = "autonomous_emission_limited"
    elif sham_exact == autonomous_exact:
        diagnosis = "causal_dependence_unresolved"
    else:
        diagnosis = "ready_for_joint_admission"
    body = {
        "schema": "aura.unified_intrinsic.answer_bridge_diagnostic.v1",
        "scope": "development_deepest_task_per_family",
        "depth": depth,
        "tasks": tasks_count,
        "oracle_exact": oracle_exact,
        "autonomous_process_exact": autonomous_process_exact,
        "autonomous_exact": autonomous_exact,
        "sham_exact": sham_exact,
        "diagnosis": diagnosis,
        "rows": rows,
    }
    return {**body, "diagnostic_sha256": _canonical_sha256(body)}


def _answer_bridge_process_preflight(
    diagnostic: dict[str, Any],
    *,
    identity_sha256: str,
    phase_schedule: dict[str, Any],
    start_step: int,
) -> dict[str, Any]:
    """Refuse bridge-only optimization until its autonomous process is exact."""

    if phase_schedule.get("mode") != "bootstrap_answer_bridge_only":
        raise ValueError("answer bridge process preflight requires a bridge-only schedule")
    tasks = diagnostic.get("tasks")
    process_exact = diagnostic.get("autonomous_process_exact")
    diagnostic_sha256 = diagnostic.get("diagnostic_sha256")
    diagnostic_body = {
        key: value for key, value in diagnostic.items() if key != "diagnostic_sha256"
    }
    if (
        type(tasks) is not int
        or tasks < 1
        or type(process_exact) is not int
        or not 0 <= process_exact <= tasks
        or not isinstance(diagnostic_sha256, str)
        or len(diagnostic_sha256) != 64
        or not isinstance(identity_sha256, str)
        or len(identity_sha256) != 64
        or type(start_step) is not int
        or start_step < 0
    ):
        raise ValueError("answer bridge process preflight evidence is invalid")
    if _canonical_sha256(diagnostic_body) != diagnostic_sha256:
        raise ValueError("answer bridge process preflight diagnostic was resealed")
    admitted = process_exact == tasks
    body = {
        "schema": "aura.unified_intrinsic.answer_bridge_process_preflight.v1",
        "identity_sha256": identity_sha256,
        "phase_schedule_sha256": _canonical_sha256(phase_schedule),
        "start_step": start_step,
        "tasks": tasks,
        "autonomous_process_exact": process_exact,
        "oracle_exact": diagnostic.get("oracle_exact"),
        "autonomous_answer_exact": diagnostic.get("autonomous_exact"),
        "sham_exact": diagnostic.get("sham_exact"),
        "diagnosis": diagnostic.get("diagnosis"),
        "diagnostic_sha256": diagnostic_sha256,
        "admitted": admitted,
        "reason": (
            "autonomous_process_exact"
            if admitted
            else "autonomous_process_not_exact_train_process_before_bridge"
        ),
        "optimizer_steps_executed": 0,
    }
    return {**body, "receipt_sha256": _canonical_sha256(body)}


def _evaluate_answer_bridge_admission(
    bundle: UnifiedTrainingBundle,
    tokenizer: Any,
    tasks: list[Any],
    spec: UnifiedIntrinsicTrainingSpec,
    bridge: str,
    contract: RecurrentAnswerEmissionContract,
    *,
    answer_digit_pointer_enabled: bool = True,
) -> dict[str, Any]:
    """Require exact autonomous process and emission on every unseen task."""

    if type(answer_digit_pointer_enabled) is not bool:
        raise TypeError("answer bridge admission pointer policy must be boolean")
    cells = {(str(task.family), int(task.depth)) for task in tasks}
    if not cells:
        raise ValueError("answer bridge admission requires unseen tasks")
    selected = sorted(
        tasks,
        key=lambda task: (str(task.family), int(task.depth), str(task.task_id)),
    )
    task_ids = [str(task.task_id) for task in selected]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("answer bridge admission task identities are duplicated")
    selected_cells = {(str(task.family), int(task.depth)) for task in selected}
    if selected_cells != cells:
        raise RuntimeError("answer bridge admission did not cover every task cell")
    rows: list[dict[str, Any]] = []
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    with recurrence_adapter_scope(start=None, stop=None):
        for task in selected:
            prompt, answer = encode_example(tokenizer, task, bridge)
            process_capture: dict[str, Any] = {}
            generated = _generate_student_rollin(
                bundle,
                prompt,
                answer,
                spec.plan_at(max(spec.train_depths)),
                eos_token_id=tokenizer.eos_token_id,
                answer_emission_contract=contract,
                answer_digit_pointer_enabled=answer_digit_pointer_enabled,
                state_slot_start=int(prompt.shape[-1]),
                process_capture=process_capture,
            )
            process = _process_evidence_from_capture(
                task,
                max(spec.train_depths),
                process_capture,
            )
            expected_values = tuple(int(value) for value in answer.tolist()[0])
            generated_values = tuple(int(value) for value in generated.tolist()[0])
            mismatches = [
                {
                    "position": position,
                    "expected_token_id": expected,
                    "generated_token_id": observed,
                }
                for position, (observed, expected) in enumerate(
                    zip(generated_values, expected_values, strict=True)
                )
                if observed != expected
            ]
            rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "task_depth": task.depth,
                    "exact": (generated_values == expected_values and process["process_exact"]),
                    "answer_exact": generated_values == expected_values,
                    "process_exact": process["process_exact"],
                    "process_evidence": process,
                    "matching_tokens": sum(
                        observed == expected
                        for observed, expected in zip(
                            generated_values,
                            expected_values,
                            strict=True,
                        )
                    ),
                    "token_count": len(expected_values),
                    "mismatches": mismatches,
                    "expected_sha256": _sha256_tokens(answer),
                    "generated_sha256": _sha256_tokens(generated),
                }
            )
    exact = sum(row["exact"] for row in rows)
    matching = sum(row["matching_tokens"] for row in rows)
    token_count = sum(row["token_count"] for row in rows)
    body = {
        "schema": "aura.unified_intrinsic.answer_bridge_admission.v6",
        "process_tape_enabled": True,
        "depth": max(spec.train_depths),
        "answer_digit_pointer_enabled": answer_digit_pointer_enabled,
        "cells": len(cells),
        "tasks": len(rows),
        "exact": exact,
        "answer_exact": sum(row["answer_exact"] for row in rows),
        "process_exact": sum(row["process_exact"] for row in rows),
        "exact_accuracy": exact / len(rows),
        "token_accuracy": matching / token_count,
        "admitted": exact == len(rows),
        "rows": rows,
    }
    return {**body, "admission_sha256": _canonical_sha256(body)}


def _evaluate_process_admission(
    bundle: UnifiedTrainingBundle,
    tokenizer: Any,
    tasks: list[Any],
    spec: UnifiedIntrinsicTrainingSpec,
    bridge: str,
    *,
    public_action_program: bool = False,
    transition_processor_lesion: bool = False,
    transition_processor_mode: str = "authoritative",
    transition_copy_prior_logit_bias: float = TRANSITION_COPY_PRIOR_LOGIT_BIAS,
    transition_opcode_expert_routing: str = "opcode",
    transition_replay_mode: str = "disabled",
    transition_history_lesion: bool = False,
) -> dict[str, Any]:
    """Require exact teacher-removed state/action execution on every unseen task."""

    selected = sorted(
        tasks,
        key=lambda task: (str(task.family), int(task.depth), str(task.task_id)),
    )
    task_ids = [str(task.task_id) for task in selected]
    cells = {(str(task.family), int(task.depth)) for task in selected}
    if not selected or len(task_ids) != len(set(task_ids)):
        raise ValueError("process admission requires unique unseen tasks")
    rows: list[dict[str, Any]] = []
    depth = max(spec.train_depths)
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    with recurrence_adapter_scope(start=None, stop=None):
        for task in selected:
            prompt, _answer = encode_example(tokenizer, task, bridge)
            capture = (
                _capture_autonomous_process(
                    bundle,
                    prompt,
                    spec.plan_at(depth),
                    public_action_values=_public_actions_for_task(task, depth),
                    microcode_lesion=True,
                    transition_processor_lesion=transition_processor_lesion,
                    transition_processor_mode=transition_processor_mode,
                    transition_copy_prior_logit_bias=(
                        transition_copy_prior_logit_bias
                    ),
                    transition_opcode_expert_routing=(
                        transition_opcode_expert_routing
                    ),
                    transition_replay_mode=transition_replay_mode,
                    transition_history_lesion=transition_history_lesion,
                )
                if public_action_program
                else _capture_autonomous_process(
                    bundle,
                    prompt,
                    spec.plan_at(depth),
                )
            )
            evidence = _process_evidence_from_capture(task, depth, capture)
            rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "task_depth": task.depth,
                    "process_exact": evidence["process_exact"],
                    "process_evidence": evidence,
                }
            )
    exact = sum(row["process_exact"] for row in rows)
    body = {
        "schema": "aura.unified_intrinsic.process_admission.v3",
        "teacher_available": False,
        "public_action_program": public_action_program,
        "exact_microcode_available": not public_action_program,
        "transition_processor_available": not transition_processor_lesion,
        "transition_processor_lesioned": transition_processor_lesion,
        "transition_processor_mode": (
            transition_processor_mode if public_action_program else "residual"
        ),
        "transition_copy_prior_logit_bias": (
            float(transition_copy_prior_logit_bias)
            if public_action_program
            and transition_processor_mode in {"copy_write", "masked_copy_write"}
            else 0.0
        ),
        "transition_opcode_expert_routing": transition_opcode_expert_routing,
        "transition_replay_mode": transition_replay_mode,
        "transition_action_history_available": not transition_history_lesion,
        "transition_action_history_lesioned": transition_history_lesion,
        "depth": depth,
        "cells": len(cells),
        "tasks": len(rows),
        "process_exact": exact,
        "exact_accuracy": exact / len(rows),
        "admitted": exact == len(rows),
        "rows": rows,
    }
    return {**body, "admission_sha256": _canonical_sha256(body)}


def _residual_hidden_size(model: Any) -> int:
    """Infer model width from an unquantized residual-space parameter.

    Quantized embedding weights expose their packed storage width, not the
    transformer hidden width. RMSNorm always has one scalar per residual
    channel and therefore remains representation independent.
    """

    layers = getattr(getattr(model, "model", None), "layers", None)
    weight = (
        getattr(getattr(layers[0], "input_layernorm", None), "weight", None) if layers else None
    )
    if weight is None or len(weight.shape) != 1 or int(weight.shape[0]) < 1:
        raise ValueError("model residual hidden size is unavailable")
    return int(weight.shape[0])


def _invocation_stop_step(
    start_step: int,
    max_steps: int,
    max_invocation_steps: int | None,
) -> int:
    """Return an operational stop boundary without changing campaign identity."""

    if start_step < 0 or max_steps < 1 or start_step > max_steps:
        raise ValueError("unified recurrence invocation step range is invalid")
    if max_invocation_steps is None:
        return max_steps
    if max_invocation_steps < 1:
        raise ValueError("maximum invocation steps must be positive")
    return min(max_steps, start_step + max_invocation_steps)


def _training_halt_reason(
    *,
    step: int,
    max_steps: int,
    invocation_stop_step: int,
) -> str:
    if step >= max_steps:
        return "max_steps"
    if step >= invocation_stop_step:
        return "invocation_step_limit"
    return "wall_clock"


def _training_verdict(
    *,
    complete: bool,
    answer_bridge_admission: dict[str, Any] | None,
    process_admission: dict[str, Any] | None,
    final: dict[str, Any] | None,
) -> str:
    """Label only terminal evidence as a scientific training verdict."""

    if not complete:
        return "incomplete_checkpoint"
    if answer_bridge_admission is not None and not answer_bridge_admission["admitted"]:
        return "answer_bridge_not_admitted"
    if process_admission is not None and not process_admission["admitted"]:
        return "autonomous_process_not_admitted"
    if final and final["heldout_depth_helps"]:
        return "heldout_depth_gain"
    if final and final["trained_depth_helps"]:
        return "trained_depth_gain_only"
    return "no_heldout_depth_gain"


def _initial_rollin_totals() -> dict[str, Any]:
    return {
        "examples": 0,
        "answer_tokens": 0,
        "generated_positions": 0,
        "generated_matches": 0,
        "last_generated_sha256": None,
        "last_effective_sha256": None,
        "max_preclip_gradient_norm": 0.0,
        "max_preclip_gradient_norms": {},
        "last_probability": None,
        "last_state_teacher_forcing_probability": None,
        "last_process_component": None,
        "last_process_stage_progress": None,
        "answer_bridge_inner_updates": 0,
        "answer_bridge_autonomous_tail_examples": 0,
        "answer_bridge_autonomous_process_exact": 0,
        "answer_bridge_wrong_process_updates_blocked": 0,
    }


def _restore_rollin_totals(training_state: dict[str, Any]) -> dict[str, Any]:
    if not training_state:
        return _initial_rollin_totals()
    candidate = training_state.get("rollin_totals")
    expected = _initial_rollin_totals()
    if not isinstance(candidate, dict) or set(candidate) != set(expected):
        raise RuntimeError("unified recurrence roll-in checkpoint state differs")
    for key in (
        "examples",
        "answer_tokens",
        "generated_positions",
        "generated_matches",
        "answer_bridge_autonomous_tail_examples",
        "answer_bridge_autonomous_process_exact",
        "answer_bridge_wrong_process_updates_blocked",
    ):
        value = candidate[key]
        if type(value) is not int or value < 0:
            raise RuntimeError("unified recurrence roll-in counters differ")
    for key in ("last_generated_sha256", "last_effective_sha256"):
        value = candidate[key]
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError("unified recurrence roll-in digest differs")
    maximum = candidate["max_preclip_gradient_norm"]
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or float(maximum) < 0.0
    ):
        raise RuntimeError("unified recurrence roll-in gradient maximum differs")
    group_norms = candidate["max_preclip_gradient_norms"]
    if not isinstance(group_norms, dict) or any(
        not isinstance(name, str)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for name, value in group_norms.items()
    ):
        raise RuntimeError("unified recurrence roll-in gradient groups differ")
    for key in ("last_probability", "last_state_teacher_forcing_probability"):
        value = candidate[key]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise RuntimeError("unified recurrence roll-in probability differs")
    component = candidate["last_process_component"]
    if component is not None and component not in {
        "initializer",
        "action",
        "action_workspace",
        "transition",
        "joint",
    }:
        raise RuntimeError("unified recurrence process component differs")
    progress = candidate["last_process_stage_progress"]
    if progress is not None and (
        isinstance(progress, bool)
        or not isinstance(progress, (int, float))
        or not 0.0 < float(progress) <= 1.0
    ):
        raise RuntimeError("unified recurrence process stage progress differs")
    return {
        **candidate,
        "max_preclip_gradient_norms": dict(group_norms),
    }


def _rollin_report(
    totals: dict[str, Any],
    *,
    initial_probability: float,
    final_probability: float,
) -> dict[str, Any]:
    snapshot = copy.deepcopy(totals)
    generated_positions = int(snapshot["generated_positions"])
    return {
        **snapshot,
        "initial_probability": initial_probability,
        "final_probability": final_probability,
        "generated_match_rate": (
            int(snapshot["generated_matches"]) / generated_positions
            if generated_positions
            else None
        ),
        "labels_from_generated_tokens": False,
    }


def _checkpoint_tensor_bytes(tensors: dict[str, Any], out_dir: Path) -> bytes:
    scratch = out_dir / f".checkpoint.{os.getpid()}.{uuid.uuid4().hex}.safetensors"
    try:
        mx.save_safetensors(str(scratch), tensors)
        return scratch.read_bytes()
    finally:
        durable_unlink(scratch, missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _discard_checkpoint_stage(path: Path) -> None:
    if not path.exists() or path.is_symlink():
        return
    for name in ("bundle.safetensors", "complete.json"):
        durable_unlink(path / name, missing_ok=True)
    try:
        path.rmdir()
    except OSError:
        pass


def _publish_latest_checkpoint_generation(
    out_dir: Path,
    *,
    stem: str,
    payload: bytes,
    step: int,
    history: list[dict[str, Any]],
    identity: dict[str, Any],
    optimization_phase: str,
    training_state: dict[str, Any],
) -> dict[str, Any]:
    """Publish immutable bytes, then atomically advance the latest pointer."""

    generations = ensure_private_directory(out_dir / "checkpoint_generations")
    checkpoint_id = f"{stem}-step-{step:08d}-{uuid.uuid4().hex}"
    generation_dir = generations / checkpoint_id
    stage_dir = ensure_private_directory(generations / f".checkpoint-stage-{uuid.uuid4().hex}")
    try:
        weights_path = stage_dir / "bundle.safetensors"
        atomic_write_bytes(weights_path, payload, mode=0o400)
        checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
        body = {
            "schema": TRAINING_SCHEMA,
            "checkpoint_generation_schema": CHECKPOINT_GENERATION_SCHEMA,
            "checkpoint_id": checkpoint_id,
            "stem": stem,
            "step": step,
            "optimization_phase": optimization_phase,
            "history": history,
            "training_state": training_state,
            "identity": identity,
            "checkpoint_file": weights_path.name,
            "checkpoint_size_bytes": len(payload),
            "checkpoint_sha256": checkpoint_sha256,
        }
        complete = {**body, "receipt_sha256": _canonical_sha256(body)}
        complete_bytes = (
            json.dumps(
                complete,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
        atomic_write_bytes(stage_dir / "complete.json", complete_bytes, mode=0o400)
        os.chmod(stage_dir, 0o500)
        os.rename(stage_dir, generation_dir)
        _fsync_directory(generations)
    finally:
        _discard_checkpoint_stage(stage_dir)
    pointer = {
        "schema": CHECKPOINT_POINTER_SCHEMA,
        "checkpoint": f"checkpoint_generations/{checkpoint_id}",
        "complete_sha256": hashlib.sha256(complete_bytes).hexdigest(),
        "identity_sha256": identity["identity_sha256"],
        "step": step,
        "stem": stem,
    }
    atomic_write_text(
        out_dir / f"{stem}_pointer.json",
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        mode=0o600,
    )

    # Preserve the historical fixed paths for evaluators without duplicating
    # the tensor payload. The immutable generation remains resume authority.
    compatibility_weights = out_dir / f"{stem}.safetensors"
    atomic_hardlink_replace(
        generation_dir / "bundle.safetensors",
        compatibility_weights,
    )
    legacy_body = {
        key: value
        for key, value in body.items()
        if key
        not in {
            "checkpoint_generation_schema",
            "checkpoint_id",
            "stem",
            "checkpoint_file",
            "checkpoint_size_bytes",
        }
    }
    _atomic_json(
        out_dir / f"{stem}.json",
        {**legacy_body, "receipt_sha256": _canonical_sha256(legacy_body)},
    )
    # Publication is complete before retention starts. If retention cannot
    # prove a generation is pointer-unreachable, it raises with this checkpoint
    # already resumable rather than letting an unattended campaign consume the
    # volume indefinitely.
    prune_checkpoint_generations(
        out_dir,
        rollback_generations_per_stem=2,
    )
    return complete


def _load_latest_checkpoint(
    out_dir: Path,
    *,
    required: bool,
) -> tuple[dict[str, Any], Path] | None:
    try:
        resolved = resolve_checkpoint_generation(
            out_dir,
            stem="checkpoint_latest",
            required=False,
        )
    except UnifiedCheckpointError as exc:
        raise RuntimeError(str(exc)) from exc
    if resolved is not None:
        return resolved.receipt, resolved.weights_path

    pointer_path = out_dir / "checkpoint_latest_pointer.json"
    legacy_receipt_path = out_dir / "checkpoint_latest.json"
    legacy_weights_path = out_dir / "checkpoint_latest.safetensors"
    if pointer_path.is_file():
        try:
            pointer = json.loads(pointer_path.read_text(encoding="ascii"))
        except (json.JSONDecodeError, OSError, UnicodeError) as exc:
            raise RuntimeError("unified recurrence checkpoint pointer is unreadable") from exc
        if (
            not isinstance(pointer, dict)
            or set(pointer)
            != {
                "schema",
                "checkpoint",
                "complete_sha256",
                "identity_sha256",
                "step",
            }
            or pointer.get("schema") != CHECKPOINT_POINTER_SCHEMA
        ):
            raise RuntimeError("unified recurrence checkpoint pointer differs")
        relative = pointer.get("checkpoint")
        if not isinstance(relative, str) or not relative.startswith("checkpoint_generations/"):
            raise RuntimeError("unified recurrence checkpoint pointer path is invalid")
        try:
            generation_dir = (out_dir / relative).resolve(strict=True)
            generation_root = (out_dir / "checkpoint_generations").resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError("unified recurrence checkpoint generation is unavailable") from exc
        if generation_dir.parent != generation_root or not generation_dir.is_dir():
            raise RuntimeError("unified recurrence checkpoint pointer escapes its root")
        complete_path = generation_dir / "complete.json"
        try:
            complete_bytes = complete_path.read_bytes()
            receipt = json.loads(complete_bytes.decode("ascii"))
        except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError) as exc:
            raise RuntimeError("unified recurrence checkpoint generation is unreadable") from exc
        identity = receipt.get("identity") if isinstance(receipt, dict) else None
        if (
            not isinstance(receipt, dict)
            or hashlib.sha256(complete_bytes).hexdigest() != pointer.get("complete_sha256")
            or receipt.get("checkpoint_generation_schema") != CHECKPOINT_GENERATION_SCHEMA
            or receipt.get("checkpoint_id") != generation_dir.name
            or receipt.get("step") != pointer.get("step")
            or not isinstance(identity, dict)
            or identity.get("identity_sha256") != pointer.get("identity_sha256")
        ):
            raise RuntimeError("unified recurrence checkpoint generation differs")
        receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if receipt.get("receipt_sha256") != _canonical_sha256(receipt_body):
            raise RuntimeError("unified recurrence checkpoint receipt differs")
        weights_name = receipt.get("checkpoint_file")
        if not isinstance(weights_name, str) or Path(weights_name).name != weights_name:
            raise RuntimeError("unified recurrence checkpoint weight path is invalid")
        weights_path = generation_dir / weights_name
        try:
            size = weights_path.stat().st_size
            digest = _file_sha256(weights_path)
        except OSError as exc:
            raise RuntimeError("unified recurrence checkpoint weights are unreadable") from exc
        if size != receipt.get("checkpoint_size_bytes") or digest != receipt.get(
            "checkpoint_sha256"
        ):
            raise RuntimeError("unified recurrence checkpoint weights differ")
        return receipt, weights_path

    legacy_present = (legacy_receipt_path.is_file(), legacy_weights_path.is_file())
    if any(legacy_present) and not all(legacy_present):
        raise RuntimeError("unified recurrence legacy checkpoint is incomplete")
    if not all(legacy_present):
        if required:
            raise RuntimeError("unified recurrence resume checkpoint is unavailable")
        return None
    try:
        receipt = json.loads(legacy_receipt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise RuntimeError("unified recurrence legacy checkpoint is unreadable") from exc
    if not isinstance(receipt, dict):
        raise RuntimeError("unified recurrence legacy checkpoint receipt differs")
    return receipt, legacy_weights_path


def _save_checkpoint(
    out_dir: Path,
    bundle: UnifiedTrainingBundle,
    optimizer: Any,
    *,
    step: int,
    history: list[dict[str, Any]],
    identity: dict[str, Any],
    stem: str = "checkpoint_latest",
    optimization_phase: str = "recurrence",
    training_state: dict[str, Any] | None = None,
) -> None:
    if not stem.startswith("checkpoint_") or not stem.replace("_", "").isalnum():
        raise ValueError("unified recurrence checkpoint stem is invalid")
    tensors = {f"bundle.{name}": value for name, value in _trainable(bundle).items()}
    tensors.update({f"optimizer.{name}": value for name, value in tree_flatten(optimizer.state)})
    payload = _checkpoint_tensor_bytes(tensors, out_dir)
    with interprocess_file_lock(out_dir / ".unified_checkpoint.lock"):
        _publish_latest_checkpoint_generation(
            out_dir,
            stem=stem,
            payload=payload,
            step=step,
            history=history,
            identity=identity,
            optimization_phase=optimization_phase,
            training_state=dict(training_state or {}),
        )


def _restore_checkpoint(
    out_dir: Path,
    bundle: UnifiedTrainingBundle,
    optimizer: Any,
    identity: dict[str, Any],
    *,
    semantic_warmup_steps: int = 0,
    state_warmup_steps: int = 0,
    answer_bridge_steps: int = 0,
    required: bool = False,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    with interprocess_file_lock(out_dir / ".unified_checkpoint.lock"):
        loaded = _load_latest_checkpoint(out_dir, required=required)
        if loaded is None:
            return 0, [], {}
        receipt, weights_path = loaded
        tensors = mx.load(str(weights_path))
        # MLX loads lazily. Evaluate every tensor while retention is excluded;
        # releasing the lock after merely resolving a pathname lets a publisher
        # rotate and delete the generation while the reader is still mapping it.
        mx.eval(tuple(tensors.values()))
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    stored_identity = receipt.get("identity")
    if not isinstance(stored_identity, dict):
        stored_identity = {}
    stored_identity_body = {
        key: value for key, value in stored_identity.items() if key != "identity_sha256"
    }
    if (
        receipt.get("receipt_sha256") != _canonical_sha256(body)
        or _canonical_sha256(stored_identity) != _canonical_sha256(identity)
        or stored_identity.get("identity_sha256") != _canonical_sha256(stored_identity_body)
    ):
        raise RuntimeError("unified recurrence checkpoint identity differs")
    expected_phase = _optimization_phase(
        int(receipt["step"]),
        semantic_warmup_steps,
        state_warmup_steps,
        answer_bridge_steps,
    )
    if receipt.get("optimization_phase") != expected_phase:
        raise RuntimeError("unified recurrence checkpoint phase differs")
    bundle_values = {
        name.removeprefix("bundle."): value
        for name, value in tensors.items()
        if name.startswith("bundle.")
    }
    expected = set(_trainable(bundle))
    if set(bundle_values) != expected:
        raise RuntimeError("unified recurrence checkpoint tensor inventory differs")
    bundle.update(tree_unflatten(list(bundle_values.items())))
    optimizer_values = [
        (name.removeprefix("optimizer."), value)
        for name, value in tensors.items()
        if name.startswith("optimizer.")
    ]
    if optimizer_values:
        optimizer.state = tree_unflatten(optimizer_values)
    mx.eval(bundle.parameters(), optimizer.state)
    training_state = receipt.get("training_state", {})
    if not isinstance(training_state, dict):
        raise RuntimeError("unified recurrence checkpoint training state differs")
    return (
        int(receipt["step"]),
        list(receipt.get("history", [])),
        dict(training_state),
    )


def _bootstrap_bundle_from_checkpoint(
    output_dir: Path,
    stem: str,
    bundle: UnifiedTrainingBundle,
    *,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Initialize new-campaign tissue from a verified parent, never its optimizer."""

    try:
        with interprocess_file_lock(output_dir / ".unified_checkpoint.lock"):
            resolved = resolve_checkpoint_generation(output_dir, stem=stem, required=True)
            if resolved is None:  # pragma: no cover - required=True is authoritative
                raise RuntimeError(
                    "unified recurrence bootstrap checkpoint is unavailable"
                )
            receipt = resolved.receipt
            tensors = mx.load(str(resolved.weights_path))
            mx.eval(tuple(tensors.values()))
    except (OSError, UnifiedCheckpointError, ValueError) as exc:
        raise RuntimeError("unified recurrence bootstrap checkpoint is invalid") from exc
    parent_identity = receipt.get("identity")
    if not isinstance(parent_identity, dict):
        raise RuntimeError("unified recurrence bootstrap identity is unavailable")
    mismatches = list(bootstrap_topology_mismatches(parent_identity, expected_identity))
    bundle_values = {
        name.removeprefix("bundle."): value
        for name, value in tensors.items()
        if name.startswith("bundle.")
    }
    child_values = _trainable(bundle)
    bootstrap_state_register_order = state_slot_names(bundle.controller.config.state_slots)
    expected = set(child_values)
    unexpected_tensors = sorted(set(bundle_values) - expected)
    if unexpected_tensors:
        raise RuntimeError(
            "unified recurrence bootstrap tensor inventory differs: " + ",".join(unexpected_tensors)
        )
    incompatible_tensors = sorted(
        name
        for name, value in bundle_values.items()
        if tuple(value.shape) != tuple(child_values[name].shape)
        or str(value.dtype) != str(child_values[name].dtype)
    )
    if incompatible_tensors:
        raise RuntimeError(
            "unified recurrence bootstrap tensor topology differs: "
            + ",".join(incompatible_tensors)
        )
    bundle_values, scoped_lora_extension, mismatches = (
        _merge_bootstrap_scoped_lora_target_extension(
            bundle_values,
            child_values,
            parent_identity=parent_identity,
            child_identity=expected_identity,
            mismatches=mismatches,
        )
    )
    bundle_values, initial_state_extension = _merge_bootstrap_initial_state_extension(
        bundle_values,
        child_values,
    )
    bundle_values, reader_extension = _merge_bootstrap_process_reader_extension(
        bundle_values,
        child_values,
    )
    bundle_values, action_workspace_extension = _merge_bootstrap_action_workspace_extension(
        bundle_values,
        child_values,
    )
    bundle_values, causal_action_extension = _merge_bootstrap_causal_action_extension(
        bundle_values,
        child_values,
    )
    bundle_values, family_action_extension = _merge_bootstrap_family_action_extension(
        bundle_values,
        child_values,
    )
    bundle_values, action_literal_binding_extension = (
        _merge_bootstrap_action_literal_binding_extension(
            bundle_values,
            child_values,
        )
    )
    bundle_values, transition_memory_extension = (
        _merge_bootstrap_transition_memory_extension(
            bundle_values,
            child_values,
            state_register_order=bootstrap_state_register_order,
        )
    )
    bundle_values, transition_tape_reader_extension = (
        _merge_bootstrap_transition_tape_reader_extension(
            bundle_values,
            child_values,
        )
    )
    bundle_values, transition_processor_extension = (
        _merge_bootstrap_transition_processor_extension(
            bundle_values,
            child_values,
            state_register_order=bootstrap_state_register_order,
        )
    )
    bundle_values, transition_opcode_expert_extension = (
        _merge_bootstrap_transition_opcode_expert_extension(
            bundle_values,
            child_values,
            state_register_order=bootstrap_state_register_order,
        )
    )
    bundle_values, transition_replay_extension = (
        _merge_bootstrap_transition_replay_extension(
            bundle_values,
            child_values,
        )
    )
    bundle_values, codebook_extension = _merge_bootstrap_codebook_extension(
        bundle_values,
        child_values,
        mismatches=mismatches,
        parent_identity=parent_identity,
        child_identity=expected_identity,
    )
    numeric_observation_extension = _bootstrap_numeric_observation_extension(
        parent_identity,
        expected_identity,
    )
    if set(bundle_values) != expected:
        raise RuntimeError("unified recurrence bootstrap tensor inventory differs")
    bundle.update(tree_unflatten(list(bundle_values.items())))
    mx.eval(bundle.parameters())
    result = {
        "schema": "aura.unified_intrinsic.bootstrap_tissue.v1",
        "stem": stem,
        "parent_step": int(receipt["step"]),
        "parent_checkpoint_sha256": receipt["checkpoint_sha256"],
        "parent_receipt_sha256": receipt["receipt_sha256"],
        "parent_identity_sha256": parent_identity["identity_sha256"],
        "optimizer_inherited": False,
        "history_inherited": False,
        "dataset_inherited": False,
        "dataset_transfer": "explicit_new_campaign",
        "tensor_shapes_verified": True,
        "tensor_dtypes_verified": True,
    }
    if codebook_extension is not None:
        result["semantic_codebook_extension"] = codebook_extension
    if scoped_lora_extension is not None:
        result["scoped_lora_target_extension"] = scoped_lora_extension
    if numeric_observation_extension is not None:
        result["numeric_observation_extension"] = numeric_observation_extension
    if initial_state_extension is not None:
        result["initial_state_extension"] = initial_state_extension
    if reader_extension is not None:
        result["process_reader_extension"] = reader_extension
    if action_workspace_extension is not None:
        result["action_workspace_extension"] = action_workspace_extension
    if causal_action_extension is not None:
        result["causal_action_extension"] = causal_action_extension
    if family_action_extension is not None:
        result["family_action_extension"] = family_action_extension
    if action_literal_binding_extension is not None:
        result["action_literal_binding_extension"] = action_literal_binding_extension
    if transition_memory_extension is not None:
        result["transition_memory_extension"] = transition_memory_extension
    if transition_tape_reader_extension is not None:
        result["transition_tape_reader_extension"] = transition_tape_reader_extension
    if transition_processor_extension is not None:
        result["transition_processor_extension"] = transition_processor_extension
    if transition_opcode_expert_extension is not None:
        result["transition_opcode_expert_extension"] = (
            transition_opcode_expert_extension
        )
    if transition_replay_extension is not None:
        result["transition_replay_extension"] = transition_replay_extension
    return result


def _tensor_sha256(value: Any) -> str:
    materialized = value.astype(mx.float32)
    mx.eval(materialized)
    return hashlib.sha256(bytes(memoryview(materialized))).hexdigest()


def _merge_bootstrap_scoped_lora_target_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
    *,
    parent_identity: dict[str, Any],
    child_identity: dict[str, Any],
    mismatches: list[str],
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    """Add a projection family only when every new path is an exact no-op."""

    if "lora_targets" not in mismatches:
        return dict(parent_values), None, list(mismatches)
    parent_targets = parent_identity.get("lora_targets")
    child_targets = child_identity.get("lora_targets")
    if (
        not isinstance(parent_targets, list)
        or not isinstance(child_targets, list)
        or len(parent_targets) != len(set(parent_targets))
        or len(child_targets) != len(set(child_targets))
        or not set(parent_targets) < set(child_targets)
        or set(child_targets) - set(parent_targets) != {"q_proj"}
    ):
        raise RuntimeError("unified recurrence bootstrap LoRA targets differ")
    parent_model = {name for name in parent_values if name.startswith("model.")}
    child_model = {name for name in child_values if name.startswith("model.")}
    new_names = child_model - parent_model
    if (
        not new_names
        or parent_model - child_model
        or any(".self_attn.q_proj." not in name for name in new_names)
    ):
        raise RuntimeError("unified recurrence bootstrap LoRA target inventory differs")
    inactive_names = {
        name
        for name in new_names
        if name.endswith(".lora_b") or ".continuous_depth_b." in name
    }
    if not inactive_names or any(bool(mx.any(child_values[name] != 0)) for name in inactive_names):
        raise RuntimeError("unified recurrence bootstrap LoRA target is not a no-op")

    migrated = dict(parent_values)
    tensors: dict[str, dict[str, Any]] = {}
    for name in sorted(new_names):
        value = child_values[name]
        migrated[name] = value
        tensors[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
            "zero_initialized": name in inactive_names,
        }
    remaining = [field for field in mismatches if field != "lora_targets"]
    return migrated, {
        "schema": "aura.unified_intrinsic.scoped_lora_target_extension.v1",
        "migration_rule": "parent_exact_plus_zero_output_query_projection",
        "parent_targets": list(parent_targets),
        "child_targets": list(child_targets),
        "added_targets": ["q_proj"],
        "parent_tensor_inventory_preserved": True,
        "behavior_before_training_preserved": True,
        "new_tensor_names": sorted(new_names),
        "tensors": tensors,
    }, remaining


def _bootstrap_numeric_observation_extension(
    parent_identity: dict[str, Any],
    child_identity: dict[str, Any],
) -> dict[str, Any] | None:
    """Audit a wider public-number sensor without pretending tensors changed."""

    parent = parent_identity.get(
        "numeric_observation_contract",
        parent_identity.get("literal_observation_contract"),
    )
    child = child_identity.get("numeric_observation_contract")
    if child is None or _canonical_sha256(parent) == _canonical_sha256(child):
        return None
    if not isinstance(parent, dict) or not isinstance(child, dict):
        raise RuntimeError("unified recurrence numeric observation extension differs")
    if (
        parent.get("digit_token_ids") != child.get("digit_token_ids")
        or type(parent.get("max_value")) is not int
        or type(child.get("max_value")) is not int
        or int(child["max_value"]) <= int(parent["max_value"])
        or child.get("encoding") != "direct_category_then_ordered_radix_pair"
        or child.get("radix") != 31
    ):
        raise RuntimeError("unified recurrence numeric observation extension differs")
    return {
        "schema": "aura.unified_intrinsic.numeric_observation_extension.v1",
        "parent_max_value": int(parent["max_value"]),
        "child_max_value": int(child["max_value"]),
        "digit_token_ids_preserved": True,
        "encoding": child["encoding"],
        "radix": child["radix"],
        "tensor_inventory_changed": False,
        "newly_observable_values": [
            int(parent["max_value"]) + 1,
            int(child["max_value"]),
        ],
    }


def _merge_bootstrap_initial_state_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Split legacy shared transition tensors into a behavior-identical parser."""

    expected = {f"controller.{name}" for name in INITIAL_STATE_PARAMETER_NAMES}
    missing = expected - set(parent_values)
    if not missing:
        return dict(parent_values), None
    if missing != expected:
        raise RuntimeError(
            "unified recurrence bootstrap initial-state inventory differs: "
            + ",".join(sorted(missing))
        )
    legacy_sources = {
        "controller.initial_state_query": "controller.state_transition_query",
        "controller.initial_state_key": "controller.state_transition_key",
        "controller.initial_state_value": "controller.state_transition_value",
        "controller.initial_state_output": "controller.state_transition_output",
        "controller.initial_state_bias": "controller.state_transition_bias",
        "controller.initial_state_literal_copy_logit": ("controller.state_literal_copy_logit"),
    }
    migrated = dict(parent_values)
    tensor_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        source = legacy_sources[name]
        if source not in parent_values or name not in child_values:
            raise RuntimeError("unified recurrence bootstrap initial-state source differs")
        value = parent_values[source]
        child = child_values[name]
        if tuple(value.shape) != tuple(child.shape) or str(value.dtype) != str(child.dtype):
            raise RuntimeError("unified recurrence bootstrap initial-state topology differs")
        migrated[name] = value
        tensor_receipts[name] = {
            "legacy_source": source,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    return migrated, {
        "schema": "aura.unified_intrinsic.initial_state_extension.v1",
        "migration_rule": "copy_legacy_shared_transition_tensors_exactly",
        "behavior_before_training_preserved": True,
        "new_tensor_names": sorted(expected),
        "tensors": tensor_receipts,
    }


def _merge_bootstrap_process_reader_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Add only the independently initialized causal-reader parameter family."""

    expected = {f"controller.{name}" for name in PROCESS_READER_PARAMETER_NAMES}
    missing = expected - set(parent_values)
    if not missing:
        return dict(parent_values), None
    if missing != expected:
        raise RuntimeError(
            "unified recurrence bootstrap tensor inventory differs: " + ",".join(sorted(missing))
        )
    missing_child = expected - set(child_values)
    if missing_child:
        raise RuntimeError(
            "unified recurrence bootstrap tensor inventory differs: "
            + ",".join(sorted(missing_child))
        )
    migrated = dict(parent_values)
    tensor_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        value = child_values[name]
        migrated[name] = value
        tensor_receipts[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    return migrated, {
        "schema": "aura.unified_intrinsic.process_reader_extension.v1",
        "migration_rule": "parent_exact_plus_deterministic_new_reader_tensors",
        "parent_tensor_inventory_preserved": True,
        "new_tensor_names": sorted(expected),
        "tensors": tensor_receipts,
    }


def _merge_bootstrap_action_workspace_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach an exact no-op action workspace without rewriting parent tissue."""

    expected = {f"controller.{name}" for name in ACTION_WORKSPACE_PARAMETER_NAMES}
    missing = expected - set(parent_values)
    if not missing:
        return dict(parent_values), None
    if missing != expected:
        raise RuntimeError(
            "unified recurrence bootstrap action-workspace inventory differs: "
            + ",".join(sorted(missing))
        )
    migrated = dict(parent_values)
    tensor_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        if name not in child_values:
            raise RuntimeError("unified recurrence bootstrap action-workspace source differs")
        value = child_values[name]
        migrated[name] = value
        tensor_receipts[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    output_name = "controller.action_workspace_output"
    if bool(mx.any(migrated[output_name] != 0)):
        raise RuntimeError("unified recurrence bootstrap action workspace is not a no-op")
    return migrated, {
        "schema": "aura.unified_intrinsic.action_workspace_extension.v1",
        "migration_rule": "parent_exact_plus_zero_output_recurrent_action_workspace",
        "parent_tensor_inventory_preserved": True,
        "behavior_before_training_preserved": True,
        "new_tensor_names": sorted(expected),
        "tensors": tensor_receipts,
    }


def _merge_bootstrap_causal_action_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach the causal decoder as an exact no-op over parent action logits."""

    expected = {f"controller.{name}" for name in CAUSAL_ACTION_PARAMETER_NAMES}
    missing = expected - set(parent_values)
    if not missing:
        return dict(parent_values), None
    if missing != expected:
        raise RuntimeError(
            "unified recurrence bootstrap causal-action inventory differs: "
            + ",".join(sorted(missing))
        )
    migrated = dict(parent_values)
    tensor_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        if name not in child_values:
            raise RuntimeError("unified recurrence bootstrap causal-action source differs")
        value = child_values[name]
        migrated[name] = value
        tensor_receipts[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    output_name = "controller.action_causal_output"
    if bool(mx.any(migrated[output_name] != 0)):
        raise RuntimeError("unified recurrence bootstrap causal action decoder is not a no-op")
    return migrated, {
        "schema": "aura.unified_intrinsic.causal_action_extension.v1",
        "migration_rule": "parent_exact_plus_zero_output_autoregressive_action_decoder",
        "parent_tensor_inventory_preserved": True,
        "behavior_before_training_preserved": True,
        "field_order": list(ACTION_SLOT_NAMES),
        "future_field_teacher_leakage": False,
        "new_tensor_names": sorted(expected),
        "tensors": tensor_receipts,
    }


def _merge_bootstrap_family_action_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach isolated public-family experts as an exact no-op."""

    expected = {f"controller.{name}" for name in FAMILY_ACTION_PARAMETER_NAMES}
    missing = expected - set(parent_values)
    if not missing:
        return dict(parent_values), None
    output_name = "controller.action_family_output"
    if output_name in missing and missing != expected:
        raise RuntimeError(
            "unified recurrence bootstrap family-action inventory differs: "
            + ",".join(sorted(missing))
        )
    migrated = dict(parent_values)
    tensor_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(missing):
        if name not in child_values:
            raise RuntimeError("unified recurrence bootstrap family-action source differs")
        value = child_values[name]
        if bool(mx.any(value != 0)):
            raise RuntimeError("unified recurrence bootstrap family action experts are not a no-op")
        migrated[name] = value
        tensor_receipts[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    return migrated, {
        "schema": "aura.unified_intrinsic.family_action_extension.v1",
        "migration_rule": "parent_exact_plus_zero_output_public_family_experts",
        "parent_tensor_inventory_preserved": True,
        "behavior_before_training_preserved": True,
        "private_transition_program_visible": False,
        "new_tensor_names": sorted(missing),
        "tensors": tensor_receipts,
    }


def _merge_bootstrap_action_literal_binding_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach public-literal binding as an exact no-op over parent logits."""

    expected = {
        f"controller.{name}" for name in ACTION_LITERAL_BINDING_PARAMETER_NAMES
    }
    missing = expected - set(parent_values)
    if not missing:
        return dict(parent_values), None
    family_output_name = "controller.action_literal_binding_family_output"
    if missing not in (expected, {family_output_name}):
        raise RuntimeError(
            "unified recurrence bootstrap action-literal-binding inventory differs: "
            + ",".join(sorted(missing))
        )
    migrated = dict(parent_values)
    tensor_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(missing):
        if name not in child_values:
            raise RuntimeError(
                "unified recurrence bootstrap action-literal-binding source differs"
            )
        value = child_values[name]
        migrated[name] = value
        tensor_receipts[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    output_names = {
        "controller.action_literal_binding_output",
        family_output_name,
    } & missing
    if any(bool(mx.any(migrated[name] != 0)) for name in output_names):
        raise RuntimeError(
            "unified recurrence bootstrap action literal binding is not a no-op"
        )
    return migrated, {
        "schema": "aura.unified_intrinsic.action_literal_binding_extension.v2",
        "migration_rule": (
            "parent_exact_plus_zero_output_public_family_conditioned_literal_binding"
        ),
        "parent_tensor_inventory_preserved": True,
        "behavior_before_training_preserved": True,
        "private_transition_program_visible": False,
        "new_tensor_names": sorted(expected),
        "tensors": tensor_receipts,
    }


def _merge_bootstrap_transition_memory_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
    *,
    state_register_order: tuple[str, ...] = STATE_SLOT_NAMES,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach typed gated transition memory as an exact parent no-op."""

    expected = {f"controller.{name}" for name in TRANSITION_MEMORY_PARAMETER_NAMES}
    missing = expected - set(parent_values)
    if not missing:
        return dict(parent_values), None
    if missing != expected:
        raise RuntimeError(
            "unified recurrence bootstrap transition-memory inventory differs: "
            + ",".join(sorted(missing))
        )
    migrated = dict(parent_values)
    tensor_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        if name not in child_values:
            raise RuntimeError(
                "unified recurrence bootstrap transition-memory source differs"
            )
        value = child_values[name]
        migrated[name] = value
        tensor_receipts[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    output_name = "controller.transition_memory_output"
    if bool(mx.any(migrated[output_name] != 0)):
        raise RuntimeError(
            "unified recurrence bootstrap transition memory is not a no-op"
        )
    return migrated, {
        "schema": "aura.unified_intrinsic.transition_memory_extension.v1",
        "migration_rule": (
            "parent_exact_plus_zero_output_slot_preserving_gated_transition_memory"
        ),
        "parent_tensor_inventory_preserved": True,
        "behavior_before_training_preserved": True,
        "field_order": list(ACTION_SLOT_NAMES),
        "state_register_order": list(state_register_order),
        "future_action_visible": False,
        "private_transition_trace_visible": False,
        "new_tensor_names": sorted(expected),
        "tensors": tensor_receipts,
    }


def _merge_bootstrap_transition_processor_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
    *,
    state_register_order: tuple[str, ...] = STATE_SLOT_NAMES,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach the typed state/action processor as an exact parent no-op."""

    expected = {f"controller.{name}" for name in TRANSITION_PROCESSOR_PARAMETER_NAMES}
    cross_name = "controller.transition_processor_state_cross_projection"
    missing = expected - set(parent_values)
    if not missing:
        return dict(parent_values), None
    if missing not in (expected, {cross_name}):
        raise RuntimeError(
            "unified recurrence bootstrap transition-processor inventory differs: "
            + ",".join(sorted(missing))
        )
    migrated = dict(parent_values)
    tensor_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(missing):
        if name not in child_values:
            raise RuntimeError(
                "unified recurrence bootstrap transition-processor source differs"
            )
        value = child_values[name]
        migrated[name] = value
        tensor_receipts[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    output_name = "controller.transition_processor_output"
    if output_name in missing and bool(mx.any(migrated[output_name] != 0)):
        raise RuntimeError(
            "unified recurrence bootstrap transition processor is not a no-op"
        )
    if cross_name in missing and bool(mx.any(migrated[cross_name] != 0)):
        raise RuntimeError(
            "unified recurrence bootstrap cross-register tissue is not a no-op"
        )
    return migrated, {
        "schema": "aura.unified_intrinsic.transition_processor_extension.v2",
        "migration_rule": (
            "parent_exact_plus_zero_output_processor_or_cross_register_extension"
        ),
        "parent_tensor_inventory_preserved": True,
        "behavior_before_training_preserved": True,
        "category_identity": "exact_one_hot_or_deterministic_fourier",
        "state_register_order": list(state_register_order),
        "action_field_order": list(ACTION_SLOT_NAMES),
        "future_action_visible": False,
        "private_transition_trace_visible": False,
        "new_tensor_names": sorted(missing),
        "tensors": tensor_receipts,
    }


def _merge_bootstrap_transition_tape_reader_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach causal public-tape attention as an exact parent no-op."""

    expected = {
        f"controller.{name}" for name in TRANSITION_TAPE_READER_PARAMETER_NAMES
    }
    missing = expected - set(parent_values)
    if not missing:
        return dict(parent_values), None
    if missing != expected:
        raise RuntimeError(
            "unified recurrence bootstrap transition-tape inventory differs: "
            + ",".join(sorted(missing))
        )
    migrated = dict(parent_values)
    tensor_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        if name not in child_values:
            raise RuntimeError(
                "unified recurrence bootstrap transition-tape source differs"
            )
        value = child_values[name]
        migrated[name] = value
        tensor_receipts[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    output_name = "controller.transition_tape_output"
    if bool(mx.any(migrated[output_name] != 0)):
        raise RuntimeError(
            "unified recurrence bootstrap transition tape reader is not a no-op"
        )
    return migrated, {
        "schema": "aura.unified_intrinsic.transition_tape_reader_extension.v1",
        "migration_rule": (
            "parent_exact_plus_zero_output_causal_public_tape_attention"
        ),
        "parent_tensor_inventory_preserved": True,
        "behavior_before_training_preserved": True,
        "current_prefix_retained_before_query": True,
        "future_action_visible": False,
        "private_transition_trace_visible": False,
        "new_tensor_names": sorted(expected),
        "tensors": tensor_receipts,
    }


def _merge_bootstrap_transition_opcode_expert_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
    *,
    state_register_order: tuple[str, ...] = STATE_SLOT_NAMES,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach operation-isolated transition heads as an exact parent no-op."""

    expected = {
        f"controller.{name}" for name in TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES
    }
    missing = expected - set(parent_values)
    if not missing:
        return dict(parent_values), None
    hidden_name = "controller.transition_processor_opcode_hidden"
    interaction_names = {
        "controller.transition_processor_opcode_interaction_up",
        "controller.transition_processor_opcode_interaction_down",
    }
    if missing not in (
        expected,
        {hidden_name},
        interaction_names,
        interaction_names | {hidden_name},
    ):
        raise RuntimeError(
            "unified recurrence bootstrap transition-opcode expert inventory differs: "
            + ",".join(sorted(missing))
        )
    migrated = dict(parent_values)
    tensor_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        if name not in child_values:
            raise RuntimeError(
                "unified recurrence bootstrap transition-opcode expert source differs"
            )
        value = child_values[name]
        if (
            name
            != "controller.transition_processor_opcode_interaction_up"
            and bool(mx.any(value != 0))
        ):
            raise RuntimeError(
                "unified recurrence bootstrap transition-opcode expert is not a no-op"
            )
        migrated[name] = value
        tensor_receipts[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    return migrated, {
        "schema": "aura.unified_intrinsic.transition_opcode_expert_extension.v1",
        "migration_rule": (
            "parent_exact_plus_zero_output_opcode_isolated_interaction_hidden_and_heads"
        ),
        "parent_tensor_inventory_preserved": True,
        "behavior_before_training_preserved": True,
        "matched_capacity_control": "uniform_public_opcode_router",
        "opcode_order": list(range(ACTION_CARDINALITY)),
        "state_register_order": list(state_register_order),
        "future_action_visible": False,
        "private_transition_trace_visible": False,
        "new_tensor_names": sorted(expected),
        "tensors": tensor_receipts,
    }


def _merge_bootstrap_transition_replay_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach state-independent public-prefix recovery as an exact no-op."""

    expected = {f"controller.{name}" for name in TRANSITION_REPLAY_PARAMETER_NAMES}
    missing = expected - set(parent_values)
    if not missing:
        return dict(parent_values), None
    if missing != expected:
        raise RuntimeError(
            "unified recurrence bootstrap transition-replay inventory differs: "
            + ",".join(sorted(missing))
        )
    migrated = dict(parent_values)
    tensor_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        if name not in child_values:
            raise RuntimeError(
                "unified recurrence bootstrap transition-replay source differs"
            )
        value = child_values[name]
        migrated[name] = value
        tensor_receipts[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    for output_name in (
        "controller.transition_replay_output",
        "controller.transition_replay_opcode_output",
    ):
        if bool(mx.any(migrated[output_name] != 0)):
            raise RuntimeError(
                "unified recurrence bootstrap transition replay is not a no-op"
            )
    return migrated, {
        "schema": "aura.unified_intrinsic.transition_replay_extension.v1",
        "migration_rule": (
            "parent_exact_plus_zero_candidate_state_independent_public_prefix_replay"
        ),
        "parent_tensor_inventory_preserved": True,
        "behavior_before_training_preserved": True,
        "query_conditioning": "fixed_state_register_and_action_field_queries",
        "future_action_visible": False,
        "private_transition_trace_visible": False,
        "new_tensor_names": sorted(expected),
        "tensors": tensor_receipts,
    }


def _merge_bootstrap_codebook_extension(
    parent_values: dict[str, Any],
    child_values: dict[str, Any],
    *,
    mismatches: list[str],
    parent_identity: dict[str, Any],
    child_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Migrate only newly assigned opcode semantics into proven parent tissue."""

    if not mismatches:
        return dict(parent_values), None
    if mismatches != ["state_codebook_sha256"]:
        raise RuntimeError("unified recurrence bootstrap topology differs: " + ",".join(mismatches))

    action_key = "controller.action_value_embeddings"
    if action_key not in parent_values or action_key not in child_values:
        raise RuntimeError("unified recurrence bootstrap semantic codebook is incomplete")

    parent_action = parent_values[action_key]
    child_action = child_values[action_key]
    start = OP_FRONTIER_TRAVERSE
    stop = MAX_RECURRENT_OPCODE + 1
    if (
        len(parent_action.shape) != 3
        or parent_action.shape[0] != len(ACTION_SLOT_NAMES)
        or parent_action.shape[1] <= stop
    ):
        raise RuntimeError("unified recurrence bootstrap action codebook shape differs")
    migrated_action = mx.concatenate(
        (
            mx.concatenate(
                (
                    parent_action[0:1, :start],
                    child_action[0:1, start:stop],
                    parent_action[0:1, stop:],
                ),
                axis=1,
            ),
            parent_action[1:],
        ),
        axis=0,
    )
    mx.eval(migrated_action)
    migrated = dict(parent_values)
    migrated[action_key] = migrated_action
    return migrated, {
        "schema": "aura.unified_intrinsic.semantic_codebook_extension.v2",
        "tensor": action_key,
        "slot": "opcode",
        "slot_index": 0,
        "value_start_inclusive": start,
        "value_stop_exclusive": stop,
        "migration_rule": "parent_exact_except_freshly_grounded_extension_rows",
        "parent_coordinates_preserved": True,
        "other_parent_tensors_preserved": True,
        "parent_state_codebook_sha256": parent_identity.get("state_codebook_sha256"),
        "child_state_codebook_sha256": child_identity.get("state_codebook_sha256"),
        "parent_tensor_sha256": _tensor_sha256(parent_action),
        "replacement_sha256": _tensor_sha256(child_action[0, start:stop]),
        "migrated_tensor_sha256": _tensor_sha256(migrated_action),
    }


def _evaluate_depth(
    bundle: UnifiedTrainingBundle,
    prompt: Any,
    answer: Any,
    task: Any,
    spec: UnifiedIntrinsicTrainingSpec,
    depth: int,
    *,
    public_action_program: bool = False,
    transition_processor_mode: str = "authoritative",
    transition_copy_prior_logit_bias: float = TRANSITION_COPY_PRIOR_LOGIT_BIAS,
    transition_opcode_expert_routing: str = "opcode",
    transition_replay_mode: str = "disabled",
    direct_transition_processor: bool = False,
) -> dict[str, float]:
    """Evaluate one depth and release its MLX graph before the next depth."""

    initial_state_logits: list[Any] = []
    action_logits: list[Any] = []
    public_actions = (
        _public_actions_for_task(task, depth) if public_action_program else None
    )
    trace = getattr(task, "transition_trace", None)
    if trace is None:
        if direct_transition_processor:
            raise ValueError("direct transition evaluation has no exact state trace")
        recurrent_states, _states, losses, state_logits = (
            unified_answer_and_recurrent_trajectory(
                bundle.model,
                prompt,
                answer,
                spec.plan_at(depth),
                bundle.controller,
                use_state_slots=False,
                initial_state_logit_trajectory=initial_state_logits,
                action_logit_trajectory=action_logits,
                public_action_values=public_actions,
                microcode_lesion=public_action_program,
                transition_processor_mode=(
                    transition_processor_mode if public_action_program else "residual"
                ),
                transition_copy_prior_logit_bias=transition_copy_prior_logit_bias,
                transition_opcode_expert_routing=transition_opcode_expert_routing,
                transition_replay_mode=transition_replay_mode,
                answer_digit_pointer_enabled=(
                    not str(getattr(task, "family", "")).startswith("frontier_")
                ),
                final_answer_only=True,
            )
        )
        loss = float(losses[-1].item())
        return {"loss": loss}

    targets = state_targets_from_trace(
        trace,
        depth,
        state_slots=bundle.controller.config.state_slots,
    )
    if direct_transition_processor:
        if not public_action_program or public_actions is None:
            raise ValueError("direct transition evaluation has no public action program")
        state = bundle.controller.exact_probabilities(
            targets.initial_values,
            slots=bundle.controller.config.state_slots,
            cardinality=bundle.controller.config.state_cardinality,
        )
        initial_state_logits.append(mx.log(mx.maximum(state, 1e-6)))
        action_history: list[Any] = []
        state_logits = []
        recurrent_states = []
        for action_values in public_actions:
            action = bundle.controller.exact_probabilities(
                action_values,
                slots=bundle.controller.config.action_slots,
                cardinality=bundle.controller.config.action_cardinality,
            )
            action_history.append(action)
            current_values = mx.argmax(state, axis=-1)
            if bool((current_values[0, -1] == 1).item()):
                decision = mx.log(mx.maximum(state, 1e-6))
            else:
                memory = bundle.controller._typed_transition_memory(
                    action_history,
                    state_probabilities=state,
                    action_probabilities=action,
                )
                decision = bundle.controller.resolve_transition_processor_logits(
                    None,
                    state,
                    action,
                    memory,
                    transition_processor_mode=transition_processor_mode,
                    opcode_expert_routing=transition_opcode_expert_routing,
                    transition_copy_prior_logit_bias=(
                        transition_copy_prior_logit_bias
                    ),
                )
                decision, _candidate, _gate = (
                    bundle.controller.typed_transition_replay_logits(
                        decision,
                        action_history,
                        action_probabilities=action,
                        replay_mode=transition_replay_mode,
                        opcode_expert_routing=transition_opcode_expert_routing,
                    )
                )
            mx.eval(decision)
            state_logits.append(decision)
            recurrent_states.append(decision)
            state = bundle.controller.straight_through_probabilities(decision)
        action_logits = [
            bundle.controller.exact_probabilities(
                row,
                slots=bundle.controller.config.action_slots,
                cardinality=bundle.controller.config.action_cardinality,
            )
            for row in public_actions
        ]
        state_loss, _state_accuracy, _step_accuracy = structured_state_loss(
            bundle.controller,
            recurrent_states,
            targets,
            public_token_count=0,
            state_logits=state_logits,
        )
        mx.eval(state_loss, *action_logits)
        loss = float(state_loss.item())
    else:
        recurrent_states, _states, losses, state_logits = (
            unified_answer_and_recurrent_trajectory(
                bundle.model,
                prompt,
                answer,
                spec.plan_at(depth),
                bundle.controller,
                use_state_slots=True,
                initial_state_logit_trajectory=initial_state_logits,
                action_logit_trajectory=action_logits,
                public_action_values=public_actions,
                microcode_lesion=public_action_program,
                transition_processor_mode=(
                    transition_processor_mode if public_action_program else "residual"
                ),
                transition_copy_prior_logit_bias=transition_copy_prior_logit_bias,
                transition_opcode_expert_routing=transition_opcode_expert_routing,
                transition_replay_mode=transition_replay_mode,
                answer_digit_pointer_enabled=(
                    not str(getattr(task, "family", "")).startswith("frontier_")
                ),
                # Evaluation consumes only the final answer loss. Decoding every
                # recurrent intermediate through the resident 32B coda retains
                # a depth-sized family of lazy Metal graphs.
                final_answer_only=True,
            )
        )
        loss = float(losses[-1].item())
    _state_loss, state_accuracy, _step_accuracy = structured_state_loss(
        bundle.controller,
        recurrent_states,
        targets,
        public_token_count=int(prompt.shape[-1]),
        state_slot_start=int(prompt.shape[-1]),
        state_logits=state_logits,
    )
    if len(initial_state_logits) != 1:
        raise RuntimeError("evaluation emitted no initial state decision")
    _initial_loss, initial_accuracy = structured_initial_state_loss(
        initial_state_logits[0],
        targets,
    )
    state_breakdown = structured_state_accuracy_breakdown(state_logits, targets)
    trajectory = structured_state_trajectory_diagnostics(
        state_logits,
        targets,
        active_steps=min(int(trace.depth), depth),
    )
    initial_breakdown = structured_initial_state_accuracy_breakdown(
        initial_state_logits[0], targets
    )
    program = getattr(task, "transition_program", None)
    if program is None:
        raise RuntimeError("evaluation task has no exact action program")
    action_targets = action_targets_from_program(program, depth)
    if public_actions is not None:
        action_logits = [
            bundle.controller.exact_probabilities(
                row,
                slots=bundle.controller.config.action_slots,
                cardinality=bundle.controller.config.action_cardinality,
            )
            for row in public_actions
        ]
        mx.eval(*action_logits)
    _action_loss, action_accuracy, _action_steps = structured_action_loss(
        action_logits, action_targets
    )
    action_breakdown = structured_action_accuracy_breakdown(action_logits, action_targets)
    return {
        "loss": loss,
        "state_accuracy": float(state_accuracy),
        "initial_state_accuracy": float(initial_accuracy),
        "state_value_accuracy": float(state_breakdown["value_accuracy"] or 0.0),
        "state_control_accuracy": float(state_breakdown["control_accuracy"] or 0.0),
        "initial_value_accuracy": float(initial_breakdown["value_accuracy"] or 0.0),
        "initial_control_accuracy": float(initial_breakdown["control_accuracy"] or 0.0),
        "state_value_exact_accuracy": float(state_breakdown["value_exact_accuracy"] or 0.0),
        "initial_value_exact_accuracy": float(initial_breakdown["value_exact_accuracy"] or 0.0),
        "action_accuracy": float(action_accuracy),
        "action_instruction_exact_accuracy": float(
            action_breakdown["instruction_exact_accuracy"] or 0.0
        ),
        **trajectory,
    }


def _evaluate(
    bundle: UnifiedTrainingBundle,
    tokenizer: Any,
    tasks: list[Any],
    spec: UnifiedIntrinsicTrainingSpec,
    bridge: str,
    depths: tuple[int, ...],
    *,
    envelope: Any,
    public_action_program: bool = False,
    transition_processor_mode: str = "authoritative",
    transition_copy_prior_logit_bias: float = TRANSITION_COPY_PRIOR_LOGIT_BIAS,
    transition_opcode_expert_routing: str = "opcode",
    transition_replay_mode: str = "disabled",
    direct_transition_processor: bool = False,
) -> dict[str, Any]:
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    totals = {depth: 0.0 for depth in depths}
    state_totals = {depth: 0.0 for depth in depths}
    state_counts = {depth: 0 for depth in depths}
    initial_state_totals = {depth: 0.0 for depth in depths}
    state_value_totals = {depth: 0.0 for depth in depths}
    state_control_totals = {depth: 0.0 for depth in depths}
    initial_value_totals = {depth: 0.0 for depth in depths}
    initial_control_totals = {depth: 0.0 for depth in depths}
    state_value_exact_totals = {depth: 0.0 for depth in depths}
    initial_value_exact_totals = {depth: 0.0 for depth in depths}
    action_totals = {depth: 0.0 for depth in depths}
    action_exact_totals = {depth: 0.0 for depth in depths}
    trajectory_metric_names = (
        "active_state_exact_accuracy",
        "active_value_exact_accuracy",
        "active_trajectory_exact",
        "first_error_fraction",
    )
    trajectory_totals = {
        name: {depth: 0.0 for depth in depths} for name in trajectory_metric_names
    }
    recovery_totals = {depth: 0.0 for depth in depths}
    sustained_recovery_totals = {depth: 0.0 for depth in depths}
    recovery_counts = {depth: 0 for depth in depths}
    terminal_correct_stability_totals = {depth: 0.0 for depth in depths}
    terminal_self_stability_totals = {depth: 0.0 for depth in depths}
    terminal_stability_counts = {depth: 0 for depth in depths}
    conditional_transition_counts = {
        name: {depth: 0 for depth in depths}
        for name in (
            "correct_after_correct",
            "correct_predecessors",
            "correct_after_wrong",
            "wrong_predecessors",
        )
    }
    first_error_histograms = {depth: {} for depth in depths}
    register_names = state_slot_names(bundle.controller.config.state_slots)
    register_totals = {
        name: {depth: 0.0 for depth in depths} for name in register_names
    }
    register_counts = {
        name: {depth: 0 for depth in depths} for name in register_names
    }
    diagnostic_depth = max(depths)
    family_process_totals: dict[str, dict[str, float | int]] = {}
    with recurrence_adapter_scope(start=None, stop=None):
        for task in tasks:
            prompt, answer = encode_example(tokenizer, task, bridge)
            for depth in depths:
                metrics = _evaluate_depth(
                    bundle,
                    prompt,
                    answer,
                    task,
                    spec,
                    depth,
                    public_action_program=public_action_program,
                    transition_processor_mode=transition_processor_mode,
                    transition_copy_prior_logit_bias=(
                        transition_copy_prior_logit_bias
                    ),
                    transition_opcode_expert_routing=(
                        transition_opcode_expert_routing
                    ),
                    transition_replay_mode=transition_replay_mode,
                    direct_transition_processor=direct_transition_processor,
                )
                totals[depth] += metrics["loss"]
                if "state_accuracy" in metrics:
                    state_totals[depth] += metrics["state_accuracy"]
                    initial_state_totals[depth] += metrics["initial_state_accuracy"]
                    state_value_totals[depth] += metrics["state_value_accuracy"]
                    state_control_totals[depth] += metrics["state_control_accuracy"]
                    initial_value_totals[depth] += metrics["initial_value_accuracy"]
                    initial_control_totals[depth] += metrics["initial_control_accuracy"]
                    state_value_exact_totals[depth] += metrics["state_value_exact_accuracy"]
                    initial_value_exact_totals[depth] += metrics["initial_value_exact_accuracy"]
                    action_totals[depth] += metrics["action_accuracy"]
                    action_exact_totals[depth] += metrics["action_instruction_exact_accuracy"]
                    for name in trajectory_metric_names:
                        trajectory_totals[name][depth] += float(metrics[name])
                    if metrics["recovery_observable"]:
                        recovery_totals[depth] += float(
                            metrics["recovered_after_first_error"]
                        )
                        sustained_recovery_totals[depth] += float(
                            metrics["sustained_recovery_after_first_error"]
                        )
                        recovery_counts[depth] += 1
                    if metrics["terminal_stability_observable"]:
                        terminal_correct_stability_totals[depth] += float(
                            metrics["terminal_correct_stable"]
                        )
                        terminal_self_stability_totals[depth] += float(
                            metrics["terminal_self_stable"]
                        )
                        terminal_stability_counts[depth] += 1
                    for name, value in metrics["conditional_transition_counts"].items():
                        conditional_transition_counts[name][depth] += int(value)
                    first_error_key = str(metrics["first_error_step"] or "none")
                    first_error_histograms[depth][first_error_key] = (
                        first_error_histograms[depth].get(first_error_key, 0) + 1
                    )
                    for name, value in metrics["per_register_accuracy"].items():
                        if value is not None:
                            register_totals[name][depth] += float(value)
                            register_counts[name][depth] += 1
                    state_counts[depth] += 1
                    if depth == diagnostic_depth:
                        family = str(task.family)
                        family_totals = family_process_totals.setdefault(
                            family,
                            {
                                "examples": 0,
                                "action_accuracy": 0.0,
                                "action_instruction_exact_accuracy": 0.0,
                                "state_value_exact_accuracy": 0.0,
                                "active_state_value_exact_accuracy": 0.0,
                            },
                        )
                        family_totals["examples"] += 1
                        family_totals["action_accuracy"] += metrics["action_accuracy"]
                        family_totals["action_instruction_exact_accuracy"] += metrics[
                            "action_instruction_exact_accuracy"
                        ]
                        family_totals["state_value_exact_accuracy"] += metrics[
                            "state_value_exact_accuracy"
                        ]
                        family_totals["active_state_value_exact_accuracy"] += metrics[
                            "active_value_exact_accuracy"
                        ]
                # Reclaim after each depth. Holding the full ladder's lazy MLX
                # graphs caused resident-32B evaluation to exceed 51 GiB.
                envelope.reclaim(force=True)
    count = len(tasks)
    ce = {f"T{depth}": totals[depth] / count for depth in depths}
    state_accuracy = {
        f"T{depth}": (state_totals[depth] / state_counts[depth] if state_counts[depth] else None)
        for depth in depths
    }
    initial_state_accuracy = {
        f"T{depth}": (
            initial_state_totals[depth] / state_counts[depth] if state_counts[depth] else None
        )
        for depth in depths
    }
    action_accuracy = {
        f"T{depth}": (action_totals[depth] / state_counts[depth] if state_counts[depth] else None)
        for depth in depths
    }
    state_value_accuracy = {
        f"T{depth}": state_value_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    state_control_accuracy = {
        f"T{depth}": state_control_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    initial_value_accuracy = {
        f"T{depth}": initial_value_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    initial_control_accuracy = {
        f"T{depth}": initial_control_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    state_value_exact_accuracy = {
        f"T{depth}": state_value_exact_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    initial_value_exact_accuracy = {
        f"T{depth}": initial_value_exact_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    action_instruction_exact_accuracy = {
        f"T{depth}": action_exact_totals[depth] / state_counts[depth]
        if state_counts[depth]
        else None
        for depth in depths
    }
    trajectory_metrics = {
        name: {
            f"T{depth}": (
                trajectory_totals[name][depth] / state_counts[depth]
                if state_counts[depth]
                else None
            )
            for depth in depths
        }
        for name in trajectory_metric_names
    }
    recovery_accuracy = {
        f"T{depth}": (
            recovery_totals[depth] / recovery_counts[depth]
            if recovery_counts[depth]
            else None
        )
        for depth in depths
    }
    sustained_recovery_accuracy = {
        f"T{depth}": (
            sustained_recovery_totals[depth] / recovery_counts[depth]
            if recovery_counts[depth]
            else None
        )
        for depth in depths
    }
    terminal_correct_stability_accuracy = {
        f"T{depth}": (
            terminal_correct_stability_totals[depth]
            / terminal_stability_counts[depth]
            if terminal_stability_counts[depth]
            else None
        )
        for depth in depths
    }
    terminal_self_stability_accuracy = {
        f"T{depth}": (
            terminal_self_stability_totals[depth] / terminal_stability_counts[depth]
            if terminal_stability_counts[depth]
            else None
        )
        for depth in depths
    }
    conditional_transition_accuracy = {
        "p_correct_given_previous_correct": {
            f"T{depth}": (
                conditional_transition_counts["correct_after_correct"][depth]
                / conditional_transition_counts["correct_predecessors"][depth]
                if conditional_transition_counts["correct_predecessors"][depth]
                else None
            )
            for depth in depths
        },
        "p_correct_given_previous_wrong": {
            f"T{depth}": (
                conditional_transition_counts["correct_after_wrong"][depth]
                / conditional_transition_counts["wrong_predecessors"][depth]
                if conditional_transition_counts["wrong_predecessors"][depth]
                else None
            )
            for depth in depths
        },
    }
    per_register_accuracy = {
        name: {
            f"T{depth}": (
                register_totals[name][depth] / register_counts[name][depth]
                if register_counts[name][depth]
                else None
            )
            for depth in depths
        }
        for name in register_names
    }
    process_by_family_at_max_depth = {
        family: {
            "examples": int(values["examples"]),
            "action_accuracy": float(values["action_accuracy"]) / int(values["examples"]),
            "action_instruction_exact_accuracy": float(
                values["action_instruction_exact_accuracy"]
            )
            / int(values["examples"]),
            "state_value_exact_accuracy": float(values["state_value_exact_accuracy"])
            / int(values["examples"]),
            "active_state_value_exact_accuracy": float(
                values["active_state_value_exact_accuracy"]
            )
            / int(values["examples"]),
        }
        for family, values in sorted(family_process_totals.items())
    }
    anchor = ce["T1"]
    trained_deeper = [ce[f"T{depth}"] for depth in spec.train_depths if depth != 1]
    heldout = [ce[f"T{depth}"] for depth in spec.heldout_depths]
    all_deeper = trained_deeper + heldout
    return {
        "examples": count,
        "ce": ce,
        "state_accuracy": state_accuracy,
        "initial_state_accuracy": initial_state_accuracy,
        "state_value_accuracy": state_value_accuracy,
        "state_control_accuracy": state_control_accuracy,
        "initial_value_accuracy": initial_value_accuracy,
        "initial_control_accuracy": initial_control_accuracy,
        "action_accuracy": action_accuracy,
        "state_value_exact_accuracy": state_value_exact_accuracy,
        "initial_value_exact_accuracy": initial_value_exact_accuracy,
        "action_instruction_exact_accuracy": action_instruction_exact_accuracy,
        **trajectory_metrics,
        "recovery_accuracy": recovery_accuracy,
        "sustained_recovery_accuracy": sustained_recovery_accuracy,
        "recovery_observation_count": {
            f"T{depth}": recovery_counts[depth] for depth in depths
        },
        "conditional_transition_accuracy": conditional_transition_accuracy,
        "conditional_transition_counts": {
            name: {f"T{depth}": values[depth] for depth in depths}
            for name, values in conditional_transition_counts.items()
        },
        "first_error_histogram": {
            f"T{depth}": first_error_histograms[depth] for depth in depths
        },
        "terminal_correct_stability_accuracy": (
            terminal_correct_stability_accuracy
        ),
        "terminal_self_stability_accuracy": terminal_self_stability_accuracy,
        "terminal_stability_observation_count": {
            f"T{depth}": terminal_stability_counts[depth] for depth in depths
        },
        "per_register_accuracy": per_register_accuracy,
        "process_by_family_at_max_depth": {
            "depth": diagnostic_depth,
            "families": process_by_family_at_max_depth,
        },
        "best_depth": min(ce, key=ce.__getitem__),
        "best_deep_relative_gain": (
            (anchor - min(all_deeper)) / max(anchor, 1e-9) if all_deeper else 0.0
        ),
        "best_heldout_relative_gain": (
            (anchor - min(heldout)) / max(anchor, 1e-9) if heldout else 0.0
        ),
        "trained_depth_helps": bool(trained_deeper and min(trained_deeper) < anchor),
        "heldout_depth_helps": bool(heldout and min(heldout) < anchor),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-model-identity-sha256")
    parser.add_argument(
        "--exclusive-model-lane",
        action="store_true",
        help="require an atomically exclusive model-memory lease before loading",
    )
    parser.add_argument(
        "--campaign-binding-json",
        help="canonical immutable source/model/runtime/training identity for checkpoints",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="create-once canonical dataset generated before resident model load",
    )
    parser.add_argument(
        "--tokenized-dataset",
        type=Path,
        help="create-once tokenizer-bound dataset generated before resident model load",
    )
    parser.add_argument("--prelude-end", type=int)
    parser.add_argument("--coda-start", type=int)
    parser.add_argument("--prelude-fraction", type=float, default=0.25)
    parser.add_argument("--coda-fraction", type=float, default=0.25)
    parser.add_argument("--train-depths", default="1,2,4")
    parser.add_argument("--heldout-depths", default="8,16")
    parser.add_argument("--families", default="khop,modular,register_trace")
    parser.add_argument(
        "--task-source",
        choices=("curriculum", "frontier_process"),
        default="curriculum",
        help="select the closed recurrence curriculum or verified broad process tasks",
    )
    parser.add_argument(
        "--frontier-difficulties",
        default="1,2,3",
        help="comma-separated frontier difficulty cells when --task-source=frontier_process",
    )
    parser.add_argument(
        "--frontier-registry-version",
        default="2026.08.06.1",
        help="versioned frontier generator registry used for broad process supervision",
    )
    parser.add_argument(
        "--task-depth",
        type=int,
        help="legacy single task depth; overrides --task-depths when supplied",
    )
    parser.add_argument("--task-depths", default="1,2,3,4")
    parser.add_argument("--per-cell", type=int, default=24)
    parser.add_argument("--holdout-per-cell", type=int, default=6)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--window-tissue-mode",
        choices=("scoped_lora", "controller_only"),
        default="scoped_lora",
        help=(
            "train scoped transformer adapters or leave every base-model tensor "
            "frozen and train only the recurrent controller"
        ),
    )
    parser.add_argument("--controller-rank", type=int, default=16)
    parser.add_argument(
        "--state-schema",
        choices=("legacy_v1", "semantic_v2"),
        default="legacy_v1",
        help=(
            "frozen recurrent-state topology; semantic_v2 carries the declared "
            "sufficient registers for bounded frontier transition families"
        ),
    )
    parser.add_argument("--state-weight", type=float, default=2.0)
    parser.add_argument("--stutter-weight", type=float, default=0.1)
    parser.add_argument("--depth-basis-size", type=int, default=4)
    parser.add_argument("--lora-targets", default="o_proj,v_proj")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--recurrent-learning-rate",
        type=float,
        help="recurrent-phase rate; defaults to --learning-rate",
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--semantic-warmup-steps", type=int, default=0)
    parser.add_argument("--state-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--process-curriculum",
        choices=("joint", "factorized", "action_workspace", "transition_only"),
        default="joint",
        help="typed-process acquisition policy during the state-transition phase",
    )
    parser.add_argument(
        "--process-family-batch-size",
        type=int,
        default=1,
        help=(
            "same-family examples whose independently computed process gradients "
            "are averaged before one optimizer update"
        ),
    )
    parser.add_argument(
        "--process-family-batch-mode",
        choices=("same_family", "balanced_families"),
        default="same_family",
        help=(
            "average a same-family cohort for an isolated expert or one example "
            "per family for shared recurrent-window tissue"
        ),
    )
    parser.add_argument(
        "--process-gradient-combiner",
        choices=("mean", "balanced_mean", "pcgrad"),
        default="mean",
        help=(
            "average family gradients, equalize transition-owned family norms, "
            "or project measured negative conflicts before averaging"
        ),
    )
    parser.add_argument(
        "--process-transformer-gradient-scale",
        type=float,
        default=1.0,
        help=(
            "relative Adam learning-rate scale applied only to scoped transformer "
            "adapters during typed-process optimization"
        ),
    )
    parser.add_argument(
        "--process-query-gradient-scale",
        type=float,
        default=1.0,
        help=(
            "relative Adam learning-rate scale applied only to scoped query "
            "projection adapters during typed-process optimization"
        ),
    )
    parser.add_argument(
        "--analytic-action-readout-fit",
        action="store_true",
        help=(
            "fit the public-family workspace readout once from training-only "
            "verified instructions before optimizer steps"
        ),
    )
    parser.add_argument(
        "--public-action-program",
        action="store_true",
        help=(
            "parse answer-blind public operands and train/evaluate the learned "
            "transition with exact microcode removed"
        ),
    )
    parser.add_argument(
        "--direct-transition-processor",
        action="store_true",
        help=(
            "train verified categorical state/action/history transitions without "
            "constructing a transformer graph"
        ),
    )
    parser.add_argument(
        "--transition-processor-mode",
        choices=tuple(
            mode for mode in TRANSITION_PROCESSOR_MODES if mode != "residual"
        ),
        default="authoritative",
        help=(
            "regenerate every state register or learn sparse writes over the "
            "committed categorical state"
        ),
    )
    parser.add_argument(
        "--transition-copy-prior-logit-bias",
        type=float,
        default=TRANSITION_COPY_PRIOR_LOGIT_BIAS,
        help=(
            "signed sparse-write retention margin; copy-write training requires "
            "a positive value and frozen diagnostics may evaluate zero"
        ),
    )
    parser.add_argument(
        "--direct-transition-curriculum",
        choices=("closed_loop", "progressive"),
        default="closed_loop",
        help=(
            "train only deployed full closed loops or first master verified "
            "one-, two-, and four-transition windows before the closed-loop tail"
        ),
    )
    parser.add_argument(
        "--direct-transition-weakest-register-weight",
        type=float,
        default=0.0,
        help=(
            "bounded extra weight for the worst active categorical register "
            "during direct transition acquisition"
        ),
    )
    parser.add_argument(
        "--transition-opcode-expert-routing",
        choices=("opcode", "uniform", "lesion"),
        default="opcode",
        help=(
            "route hidden and output transition experts by the public opcode, "
            "uniformly for a matched-capacity control, or lesion them"
        ),
    )
    parser.add_argument(
        "--transition-replay-mode",
        choices=("disabled", "active", "forced", "lesion"),
        default="disabled",
        help=(
            "blend, force, or lesion the state-independent public-prefix "
            "transition replay candidate"
        ),
    )
    parser.add_argument(
        "--transition-replay-auxiliary-weight",
        type=float,
        default=0.5,
        help="bounded auxiliary weight for the public-prefix replay candidate",
    )
    parser.add_argument(
        "--analytic-action-readout-ridge",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--analytic-action-readout-margin",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--answer-bridge-steps",
        type=int,
        default=0,
        help="isolated state-to-token adaptation steps after semantic warmup",
    )
    parser.add_argument(
        "--answer-bridge-inner-steps",
        type=int,
        default=1,
        help=(
            "head-only optimizer updates per expensive bridge feature pass; "
            "values above one use detached causal features"
        ),
    )
    parser.add_argument(
        "--answer-bridge-autonomous-tail-steps",
        type=int,
        help=(
            "final answer-bridge steps that anneal to zero process teacher forcing; "
            "defaults to min(8, answer bridge steps)"
        ),
    )
    parser.add_argument(
        "--state-learning-rate",
        type=float,
        help="state-transition phase rate; defaults to recurrent learning rate",
    )
    parser.add_argument(
        "--answer-bridge-learning-rate",
        type=float,
        help="isolated state-to-token rate; defaults to --learning-rate",
    )
    parser.add_argument(
        "--answer-bridge-rollin-probability",
        type=float,
        default=0.25,
        help="initial generated-history fraction during answer-bridge adaptation",
    )
    parser.add_argument(
        "--answer-bridge-rollin-final-probability",
        type=float,
        default=1.0,
        help="final generated-history fraction during answer-bridge adaptation",
    )
    parser.add_argument(
        "--student-rollin-probability",
        type=float,
        default=0.0,
        help="initial fraction of recurrent-phase history taken from the deep policy",
    )
    parser.add_argument(
        "--student-rollin-final-probability",
        type=float,
        help="final generated-history fraction; defaults to the initial fraction",
    )
    parser.add_argument(
        "--state-teacher-forcing-probability",
        type=float,
        default=1.0,
        help="initial training-only exact-state roll-in probability",
    )
    parser.add_argument(
        "--state-teacher-forcing-final-probability",
        type=float,
        default=0.25,
        help="final exact-state roll-in probability; inference is always zero",
    )
    parser.add_argument(
        "--state-teacher-forcing-hold-fraction",
        type=float,
        default=0.0,
        help="fraction of transition acquisition held at the initial exact-state rate",
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=1.0,
        help="global norm trust bound applied after phase masking",
    )
    parser.add_argument("--max-minutes", type=float, default=90.0)
    parser.add_argument(
        "--max-invocation-steps",
        type=int,
        help=(
            "stop this process after N additional durable steps without changing "
            "the scientific campaign identity; resume continues the same run"
        ),
    )
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--checkpoint-group", type=int, default=4)
    parser.add_argument(
        "--grounding-batch-size",
        type=int,
        default=32,
        help="equal-token-length batch size for frozen-prelude codebook grounding",
    )
    parser.add_argument("--seed", type=int, default=20260810198)
    parser.add_argument("--init-seed", type=int, default=20260810198)
    parser.add_argument("--bridge", default="assistant_answer")
    parser.add_argument("--memory-fraction", type=float, default=0.48)
    parser.add_argument("--memory-limit-gb", type=float)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    parser.add_argument("--wired-limit-gb", type=float)
    parser.add_argument("--resource-stage-path", type=Path)
    parser.add_argument("--resource-startup-lethal-mb", type=float)
    parser.add_argument("--resource-steady-lethal-mb", type=float)
    parser.add_argument("--resource-guard-timeout-s", type=float, default=120.0)
    parser.add_argument("--preload-ready-path", type=Path)
    parser.add_argument("--preload-release-path", type=Path)
    parser.add_argument("--preload-key-path", type=Path)
    parser.add_argument("--preload-config-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-if-available",
        action="store_true",
        help=(
            "restore a valid durable checkpoint when present, otherwise begin "
            "at step zero; intended for one immutable supervised replay command"
        ),
    )
    parser.add_argument(
        "--bootstrap-output-dir",
        type=Path,
        help="immutable imported parent checkpoint directory for new-campaign tissue",
    )
    parser.add_argument(
        "--bootstrap-stem",
        default="checkpoint_latest",
        help="authoritative parent checkpoint stem used only for initialization",
    )
    args = parser.parse_args()
    campaign_binding = _parse_campaign_binding(args.campaign_binding_json)
    if args.resume and args.resume_if_available:
        raise ValueError("resume modes are mutually exclusive")
    if args.bootstrap_output_dir is not None and args.resume:
        raise ValueError("bootstrap cannot accompany an unconditional local resume")
    phase_schedule = _phase_schedule(
        semantic_warmup_steps=args.semantic_warmup_steps,
        state_warmup_steps=args.state_warmup_steps,
        answer_bridge_steps=args.answer_bridge_steps,
        max_steps=args.max_steps,
        bootstrap_output_dir=args.bootstrap_output_dir,
        process_only=(
            args.task_source == "frontier_process"
            and args.semantic_warmup_steps == 0
            and args.state_warmup_steps == args.max_steps
            and args.answer_bridge_steps == 0
        ),
        process_bootstrap=args.bootstrap_output_dir is not None,
    )
    if args.process_curriculum == "factorized" and (
        args.task_source != "frontier_process" or args.state_warmup_steps < 8
    ):
        raise ValueError("factorized process curriculum requires frontier process acquisition")
    if args.process_curriculum == "action_workspace" and (
        args.task_source != "frontier_process"
        or args.state_warmup_steps != args.max_steps
        or args.bootstrap_output_dir is None
    ):
        raise ValueError(
            "action-workspace curriculum requires bootstrapped frontier process acquisition"
        )
    fresh_public_transition = _fresh_public_transition_acquisition(
        window_tissue_mode=args.window_tissue_mode,
        public_action_program=args.public_action_program,
        direct_transition_processor=args.direct_transition_processor,
    )
    if args.process_curriculum == "transition_only" and (
        args.task_source != "frontier_process"
        or args.state_warmup_steps != args.max_steps
        or (args.bootstrap_output_dir is None and not fresh_public_transition)
    ):
        raise ValueError(
            "transition-only curriculum requires bootstrapped frontier process acquisition"
        )
    if args.process_family_batch_size < 1 or (
        args.process_family_batch_size > 1
        and args.process_curriculum
        not in {"action_workspace", "factorized", "transition_only"}
    ):
        raise ValueError(
            "process family batching requires a supported process-acquisition curriculum"
        )
    if (
        args.process_family_batch_mode == "balanced_families"
        and args.process_family_batch_size
        != len([value for value in args.families.split(",") if value.strip()])
    ):
        raise ValueError(
            "balanced process batching requires exactly one example per selected family"
        )
    if args.process_gradient_combiner == "pcgrad" and (
        args.process_family_batch_size < 2 or not args.direct_transition_processor
    ):
        raise ValueError(
            "PCGrad requires a multi-family direct transition processor cohort"
        )
    if not 0.0 <= args.process_transformer_gradient_scale <= 1.0:
        raise ValueError("process transformer gradient scale must be inside [0, 1]")
    if not 0.0 <= args.process_query_gradient_scale <= 1.0:
        raise ValueError("process query gradient scale must be inside [0, 1]")
    if args.analytic_action_readout_fit and (
        args.task_source != "frontier_process"
        or args.bootstrap_output_dir is None
        or args.process_curriculum != "transition_only"
    ):
        raise ValueError(
            "analytic action readout fitting requires bootstrapped transition-only acquisition"
        )
    if args.public_action_program and (
        args.task_source != "frontier_process"
        or args.process_curriculum != "transition_only"
        or args.state_warmup_steps != args.max_steps
        or args.answer_bridge_steps != 0
        or any(
            family
            not in {"mathematics", "coding", "calibration", "misleading_premise"}
            for family in (value.strip() for value in args.families.split(",") if value.strip())
        )
    ):
        raise ValueError(
            "public action program requires a supported transition-only frontier campaign"
        )
    if args.direct_transition_processor and (
        not args.public_action_program
        or args.process_curriculum != "transition_only"
        or args.answer_bridge_steps != 0
        or args.process_transformer_gradient_scale != 0.0
        or args.process_query_gradient_scale != 0.0
    ):
        raise ValueError(
            "direct transition processor requires public transition-only controller training"
        )
    if args.transition_processor_mode != "authoritative" and (
        not args.direct_transition_processor or not args.public_action_program
    ):
        raise ValueError(
            "copy-write transition mode requires direct public transition training"
        )
    if (
        not math.isfinite(args.transition_copy_prior_logit_bias)
        or not 0.0 <= args.transition_copy_prior_logit_bias <= 8.0
        or args.transition_processor_mode in {"copy_write", "masked_copy_write"}
        and args.transition_copy_prior_logit_bias <= 0.0
    ):
        raise ValueError(
            "copy-write transition mode requires a finite positive prior in (0, 8]"
        )
    if (
        args.direct_transition_curriculum != "closed_loop"
        and not args.direct_transition_processor
    ):
        raise ValueError(
            "direct transition curriculum requires direct transition processor training"
        )
    if (
        not math.isfinite(args.direct_transition_weakest_register_weight)
        or not 0.0 <= args.direct_transition_weakest_register_weight <= 2.0
        or args.direct_transition_weakest_register_weight > 0.0
        and not args.direct_transition_processor
    ):
        raise ValueError("direct transition weakest-register weight differs")
    if args.transition_opcode_expert_routing != "opcode" and (
        not args.direct_transition_processor or not args.public_action_program
    ):
        raise ValueError(
            "transition opcode routing controls require direct public transition training"
        )
    if args.transition_replay_mode != "disabled" and (
        not args.direct_transition_processor or not args.public_action_program
    ):
        raise ValueError(
            "transition replay requires direct public transition training"
        )
    if (
        not math.isfinite(args.transition_replay_auxiliary_weight)
        or not 0.0 <= args.transition_replay_auxiliary_weight <= 2.0
    ):
        raise ValueError("transition replay auxiliary weight must be inside [0, 2]")
    if (
        not math.isfinite(args.analytic_action_readout_ridge)
        or args.analytic_action_readout_ridge <= 0.0
        or not math.isfinite(args.analytic_action_readout_margin)
        or args.analytic_action_readout_margin <= 0.0
    ):
        raise ValueError("analytic action readout fit bounds must be positive and finite")
    answer_bridge_autonomous_tail_steps = (
        min(8, args.answer_bridge_steps)
        if args.answer_bridge_autonomous_tail_steps is None
        else args.answer_bridge_autonomous_tail_steps
    )
    rollin_final_probability = (
        args.student_rollin_probability
        if args.student_rollin_final_probability is None
        else args.student_rollin_final_probability
    )
    if not (
        0.0 <= args.student_rollin_probability <= 1.0
        and 0.0 <= rollin_final_probability <= 1.0
        and args.student_rollin_probability <= rollin_final_probability
    ):
        raise ValueError("student roll-in probability must be inside [0, 1]")
    if not (
        0.0
        <= args.answer_bridge_rollin_probability
        <= args.answer_bridge_rollin_final_probability
        <= 1.0
    ):
        raise ValueError("answer bridge roll-in probability must increase inside [0, 1]")
    if not (
        0.0
        <= args.state_teacher_forcing_final_probability
        <= args.state_teacher_forcing_probability
        <= 1.0
    ):
        raise ValueError("state teacher-forcing schedule must decrease inside [0, 1]")
    if not 0.0 <= args.state_teacher_forcing_hold_fraction < 1.0:
        raise ValueError("state teacher-forcing hold fraction must be inside [0, 1)")
    if args.max_gradient_norm <= 0.0:
        raise ValueError("maximum gradient norm must be positive")
    if args.answer_bridge_inner_steps < 1:
        raise ValueError("answer bridge inner steps must be positive")
    if (
        type(answer_bridge_autonomous_tail_steps) is not int
        or (
            args.answer_bridge_steps > 0
            and not 1 <= answer_bridge_autonomous_tail_steps <= args.answer_bridge_steps
        )
        or (args.answer_bridge_steps == 0 and answer_bridge_autonomous_tail_steps != 0)
    ):
        raise ValueError("answer bridge autonomous tail must fit the answer bridge phase")
    if args.max_minutes <= 0.0:
        raise ValueError("maximum minutes must be positive")
    if any(
        value is not None and (not math.isfinite(value) or value <= 0.0)
        for value in (
            args.memory_limit_gb,
            args.cache_limit_gb,
            args.wired_limit_gb,
        )
    ):
        raise ValueError("explicit MLX memory limits must be finite and positive")
    if (
        args.memory_limit_gb is not None
        and args.cache_limit_gb is not None
        and args.cache_limit_gb >= args.memory_limit_gb
    ):
        raise ValueError("MLX cache limit must be below active memory limit")
    if (
        args.memory_limit_gb is not None
        and args.wired_limit_gb is not None
        and args.wired_limit_gb <= args.memory_limit_gb
    ):
        raise ValueError("MLX wired limit must exceed active memory limit")
    if args.max_invocation_steps is not None and args.max_invocation_steps < 1:
        raise ValueError("maximum invocation steps must be positive")
    if args.expected_model_identity_sha256 is not None and (
        len(args.expected_model_identity_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in args.expected_model_identity_sha256
        )
    ):
        raise ValueError("expected model identity SHA-256 is invalid")
    if args.grounding_batch_size < 1:
        raise ValueError("grounding batch size must be positive")
    recurrent_learning_rate = (
        args.learning_rate if args.recurrent_learning_rate is None else args.recurrent_learning_rate
    )
    state_learning_rate = (
        recurrent_learning_rate if args.state_learning_rate is None else args.state_learning_rate
    )
    answer_bridge_learning_rate = (
        args.learning_rate
        if args.answer_bridge_learning_rate is None
        else args.answer_bridge_learning_rate
    )
    if (
        args.learning_rate <= 0.0
        or recurrent_learning_rate <= 0.0
        or state_learning_rate <= 0.0
        or answer_bridge_learning_rate <= 0.0
    ):
        raise ValueError("learning rates must be positive")
    if args.state_weight <= 0.0 or args.stutter_weight < 0.0:
        raise ValueError("state weight must be positive and stutter weight non-negative")
    if args.window_tissue_mode == "controller_only" and args.semantic_warmup_steps:
        raise ValueError("controller-only tissue cannot schedule transformer semantic warmup")
    resource_guard_values = (
        args.resource_stage_path,
        args.resource_startup_lethal_mb,
        args.resource_steady_lethal_mb,
    )
    resource_guard_enabled = all(value is not None for value in resource_guard_values)
    if any(value is not None for value in resource_guard_values) != resource_guard_enabled:
        raise ValueError("resource guard arguments must be supplied together")
    if resource_guard_enabled and not (
        math.isfinite(float(args.resource_startup_lethal_mb))
        and math.isfinite(float(args.resource_steady_lethal_mb))
        and float(args.resource_startup_lethal_mb) > float(args.resource_steady_lethal_mb) > 0.0
        and math.isfinite(args.resource_guard_timeout_s)
        and args.resource_guard_timeout_s > 0.0
    ):
        raise ValueError("resource guard ceilings or timeout are invalid")
    preload_values = (
        args.preload_ready_path,
        args.preload_release_path,
        args.preload_key_path,
        args.preload_config_sha256,
    )
    preload_enabled = all(value is not None for value in preload_values)
    if any(value is not None for value in preload_values) != preload_enabled:
        raise ValueError("preload barrier arguments must be supplied together")
    if resource_guard_enabled and not preload_enabled:
        raise ValueError("external resource guard requires a signed preload barrier")
    preload_host_pressure: dict[str, Any] | None = None
    if resource_guard_enabled:
        preload_release = verify_release(
            args.preload_release_path.expanduser(),
            ready_path=args.preload_ready_path.expanduser(),
            key_path=args.preload_key_path.expanduser(),
            config_sha256=str(args.preload_config_sha256),
            require_live_evidence=True,
        )
        preload_host_pressure = dict(preload_release["host_pressure"])
    elif preload_enabled:
        raise ValueError("preload barrier requires the external resource guard")
    else:
        preload_host_pressure = host_pressure()
    if preload_enabled and campaign_binding is None:
        raise ValueError("resident preload requires a campaign checkpoint binding")
    if campaign_binding is not None and (
        campaign_binding["campaign_config_sha256"] != args.preload_config_sha256
    ):
        raise ValueError("campaign checkpoint and preload identities differ")
    if preload_enabled or resource_guard_enabled:
        if (
            preload_host_pressure.get("available") is not True
            or preload_host_pressure.get("under_pressure") is not False
        ):
            raise RuntimeError("resident unified training refused unavailable or pressured host")

    from mlx_lm import load

    from core.learning import recurrence_curriculum as curriculum
    from core.runtime.model_lane_control import standalone_model_lane

    train_depths = tuple(int(value) for value in args.train_depths.split(","))
    heldout_depths = tuple(int(value) for value in args.heldout_depths.split(","))
    prelude_end, coda_start, window_geometry = _resolve_recurrent_window(
        args.model,
        prelude_end=args.prelude_end,
        coda_start=args.coda_start,
        prelude_fraction=args.prelude_fraction,
        coda_fraction=args.coda_fraction,
    )
    spec = UnifiedIntrinsicTrainingSpec(
        prelude_end=prelude_end,
        coda_start=coda_start,
        train_depths=train_depths,
        heldout_depths=heldout_depths,
        state_weight=args.state_weight,
        stutter_weight=args.stutter_weight,
    )
    state_spec = replace(
        spec,
        answer_weight=0.0,
        anchor_weight=0.0,
        trajectory_weight=0.0,
        halt_weight=0.0,
        stutter_weight=0.0,
    )
    task_depths = (
        (args.task_depth,)
        if args.task_depth is not None
        else tuple(int(value) for value in args.task_depths.split(","))
    )
    if not task_depths or any(depth < 1 for depth in task_depths):
        raise ValueError("task depths must be positive")
    families = tuple(value.strip() for value in args.families.split(",") if value.strip())
    targets = tuple(value.strip() for value in args.lora_targets.split(",") if value.strip())
    query_optimizer_enabled = "q_proj" in targets
    bridge = {"assistant_answer": "\n\nFINAL_ANSWER: "}.get(
        args.bridge,
        args.bridge,
    )
    out_dir = args.out_dir.expanduser().resolve()
    ensure_private_directory(out_dir)
    if args.tokenized_dataset is not None and args.dataset is None:
        raise RuntimeError("tokenized dataset requires a frozen source dataset")
    frontier_difficulties = tuple(
        int(value) for value in args.frontier_difficulties.split(",") if value
    )
    if args.task_source == "frontier_process" and args.task_depth is not None:
        raise ValueError("frontier process tasks do not accept legacy --task-depth")
    if args.task_source == "frontier_process" and (
        not frontier_difficulties
        or any(value not in (1, 2, 3) for value in frontier_difficulties)
        or len(set(frontier_difficulties)) != len(frontier_difficulties)
    ):
        raise ValueError("frontier process difficulties must be unique values in {1,2,3}")
    expected_task_families = (
        {f"frontier_{domain}" for domain in families}
        if args.task_source == "frontier_process"
        else set(families)
    )
    expected_train_count = (
        len(families) * len(frontier_difficulties) * args.per_cell
        if args.task_source == "frontier_process"
        else len(families) * len(task_depths) * args.per_cell
    )
    expected_holdout_count = (
        len(families) * len(frontier_difficulties) * args.holdout_per_cell
        if args.task_source == "frontier_process"
        else len(families) * len(task_depths) * args.holdout_per_cell
    )
    if args.dataset is not None:
        dataset_path = args.dataset.expanduser().resolve(strict=True)
        train_tasks, holdout = _load_frozen_dataset(dataset_path)
        if (
            {task.family for task in train_tasks + holdout} != expected_task_families
            or len(train_tasks) != expected_train_count
            or len(holdout) != expected_holdout_count
        ):
            raise RuntimeError("unified recurrence frozen dataset differs from CLI")
        if args.task_source == "curriculum" and (
            {task.depth for task in train_tasks + holdout} != set(task_depths)
        ):
            raise RuntimeError("unified recurrence frozen task depths differ from CLI")
    elif args.task_source == "frontier_process":
        train_tasks = frontier_process_task_battery(
            families,
            frontier_difficulties,
            args.per_cell,
            seed=args.seed,
            registry_version=args.frontier_registry_version,
        )
        holdout = frontier_process_task_battery(
            families,
            frontier_difficulties,
            args.holdout_per_cell,
            seed=args.seed + 9_973,
            registry_version=args.frontier_registry_version,
            excluded_prompts={task.prompt for task in train_tasks},
        )
        random.Random(args.seed).shuffle(train_tasks)
        train_prompts = {task.prompt for task in train_tasks}
        holdout = [task for task in holdout if task.prompt not in train_prompts]
    else:
        train_tasks = curriculum.task_battery(
            families,
            task_depths,
            args.per_cell,
            seed=args.seed,
        )
        random.Random(args.seed).shuffle(train_tasks)
        holdout = curriculum.task_battery(
            families,
            task_depths,
            args.holdout_per_cell,
            seed=args.seed + 9_973,
        )
        train_prompts = {task.prompt for task in train_tasks}
        holdout = [task for task in holdout if task.prompt not in train_prompts]
    if not holdout:
        raise RuntimeError("unified recurrence holdout is empty")
    observed_task_depths = tuple(sorted({int(task.depth) for task in train_tasks + holdout}))
    frontier_depth_mismatch = args.task_source == "frontier_process" and any(
        depth not in spec.train_depths for depth in observed_task_depths
    )
    curriculum_depth_mismatch = args.task_source == "curriculum" and (
        max(observed_task_depths) > max(spec.train_depths)
    )
    if frontier_depth_mismatch or curriculum_depth_mismatch:
        raise ValueError(
            "task depths must all be explicitly present in --train-depths: "
            + ",".join(map(str, observed_task_depths))
        )
    if args.task_source == "frontier_process":
        task_depths = observed_task_depths
    missing_traces = [
        task.task_id for task in train_tasks + holdout if task.transition_trace is None
    ]
    if missing_traces:
        raise RuntimeError(
            "state-supervised curriculum contains tasks without exact traces: "
            + ",".join(missing_traces[:5])
        )
    missing_programs = [
        task.task_id for task in train_tasks + holdout if task.transition_program is None
    ]
    if missing_programs:
        raise RuntimeError(
            "action-supervised curriculum contains tasks without exact programs: "
            + ",".join(missing_programs[:5])
        )
    requested_state_register_order = (
        SEMANTIC_STATE_SLOT_NAMES
        if args.state_schema == "semantic_v2"
        else STATE_SLOT_NAMES
    )
    for task in train_tasks + holdout:
        state_targets_from_trace(
            task.transition_trace,
            1,
            state_slots=len(requested_state_register_order),
        )
    if args.state_schema == "semantic_v2" and args.task_source == "frontier_process":
        transition_identifiability = audit_public_transition_identifiability(
            train_tasks,
            holdout,
        )
        if not transition_identifiability["admission"][
            "state_recurrent_transition_admitted"
        ]:
            raise RuntimeError("semantic recurrent transition is not identifiable")

    dataset_identity = (
        _freeze_dataset(out_dir, train_tasks, holdout)
        if args.dataset is None
        else freeze_source_dataset(dataset_path, train_tasks, holdout)
    )
    source_sha256s = {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in TRAINING_SOURCE_FILES
    }
    runtime_identity = _runtime_identity()
    model_identity = _model_identity(args.model)
    if (
        args.expected_model_identity_sha256 is not None
        and model_identity["identity_sha256"] != args.expected_model_identity_sha256
    ):
        raise RuntimeError("resident unified model identity differs from campaign")
    started = time.time()
    deadline = started + args.max_minutes * 60.0
    with (
        standalone_model_lane(
            owner_id=f"train-unified-intrinsic:{out_dir.name}",
            model_path=args.model,
            purpose=_model_lane_purpose(args.window_tissue_mode),
            preemptible=False,
            require_exclusive=args.exclusive_model_lane,
            allow_owner_eviction=False,
            metadata={
                "tool": "train_unified_intrinsic_recurrence",
                "operator_launched": True,
                "window_tissue_mode": args.window_tissue_mode,
            },
        ),
        mlx_memory_envelope(
            fraction=args.memory_fraction,
            memory_gb=args.memory_limit_gb,
            cache_gb=args.cache_limit_gb,
            wired_gb=args.wired_limit_gb,
            restore_limits_on_exit=False,
        ) as envelope,
    ):
        mx.random.seed(args.init_seed)
        model, tokenizer = load(args.model)
        model.freeze()
        tokenizer_identity = resident_bootstrap_tokenizer_identity(
            Path(args.model),
            tokenizer,
        )
        if args.tokenized_dataset is not None:
            tokenized_path = args.tokenized_dataset.expanduser().resolve(strict=True)
            tokenized_dataset_identity = verify_tokenized_dataset(
                tokenized_path,
                tokenizer,
                train_tasks,
                holdout,
                bridge=bridge,
                dataset_identity=dataset_identity,
                tokenizer_identity_sha256=tokenizer_identity["identity_sha256"],
            )
        else:
            tokenized_dataset_identity = freeze_tokenized_dataset(
                out_dir / TOKENIZED_DATASET_FILENAME,
                tokenizer,
                train_tasks,
                holdout,
                bridge=bridge,
                dataset_identity=dataset_identity,
                tokenizer_identity_sha256=tokenizer_identity["identity_sha256"],
            )
        literal_digit_ids = tokenizer_digit_token_ids(tokenizer)
        literal_contract = LiteralObservationContract(literal_digit_ids)
        numeric_observation_contract = LiteralObservationContract(
            literal_digit_ids,
            max_value=MAX_PROCESS_INTEGER,
        )
        opcode_contract = tokenizer_opcode_contract(tokenizer)
        family_contract = tokenizer_frontier_family_contract(tokenizer)
        answer_emission_contract = tokenizer_answer_emission_contract(
            tokenizer,
            opcode_contract,
        )
        pointer_bound_families = frozenset(
            family for family, _marker in answer_emission_contract.family_markers
        )
        training_families = sorted({str(task.family) for task in train_tasks})
        hidden_size = _residual_hidden_size(model)
        state_register_order = requested_state_register_order
        register_loss_weights = state_slot_loss_weights(len(state_register_order))
        answer_bridge_supervision = {
            "schema": "aura.unified_intrinsic.answer_bridge_supervision.v6",
            "semantic_cross_entropy_families": training_families,
            "role_place_binding_families": sorted(
                family for family in training_families if family in pointer_bound_families
            ),
            "semantic_only_families": sorted(
                family for family in training_families if family not in pointer_bound_families
            ),
            "unsupported_pointer_targets_are_fabricated": False,
            "answer_digit_pointer_enabled": args.task_source != "frontier_process",
            "generated_history_policy": (
                "full_autonomous_prefix"
                if args.task_source == "frontier_process"
                else "grammar_preserving_digit_substitution"
            ),
            "process_teacher_policy": {
                "mapping_phase": "exact_verified_process",
                "autonomous_tail_steps": answer_bridge_autonomous_tail_steps,
                "tail_schedule": "linear_to_zero",
                "tail_update_authority": "exact_autonomous_process_only",
                "wrong_process_can_supervise_correct_answer": False,
            },
            "process_tape": {
                "schema": PROCESS_TAPE_SCHEMA,
                "ordering": "bounded_sinusoidal_step_and_transition_role",
                "reader": "two_independent_rank_expanded_causal_prefix_blocks",
                "reader_rank": min(hidden_size, 4 * args.controller_rank),
                "contents": [
                    "pre_state",
                    "typed_action",
                    "post_state",
                    "state_delta",
                ],
                "entries_per_live_step": (
                    len(ACTION_SLOT_NAMES) + 3 * len(state_register_order)
                ),
                "terminal_stutter_entries_masked": True,
                "next_action_reads_complete_prior_tape": True,
                "private_answer_exposed": False,
            },
        }
        wiring = _configure_window_tissue(
            model,
            spec,
            mode=args.window_tissue_mode,
            rank=args.lora_rank,
            targets=targets,
            depth_basis_size=args.depth_basis_size,
        )
        controller = UnifiedRecurrentController(
            UnifiedRecurrenceConfig(
                hidden_size=hidden_size,
                correction_rank=args.controller_rank,
                state_slots=len(state_register_order),
                minimum_iterations=1,
                initialization_seed=args.init_seed,
                literal_digit_token_ids=literal_digit_ids,
                numeric_observation_max_value=(numeric_observation_contract.max_value),
                opcode_token_patterns=opcode_contract.patterns,
                opcode_context_patterns=opcode_contract.contexts,
                frontier_family_token_patterns=family_contract.patterns,
            )
        )
        state_codebook_grounding = _ground_state_value_embeddings(
            model,
            tokenizer,
            controller,
            prelude_end=spec.prelude_end,
            batch_size=args.grounding_batch_size,
        )
        bundle = UnifiedTrainingBundle(model, controller)
        readout_sha256 = readout_fingerprint(model, spec.coda_start)
        identity = {
            "schema": TRAINING_SCHEMA,
            "model": model_identity,
            "runtime": runtime_identity,
            "dataset": dataset_identity,
            "tokenizer": tokenizer_identity,
            "tokenized_dataset": tokenized_dataset_identity,
            "spec": spec.to_dict(),
            "window_geometry": window_geometry,
            "families": list(families),
            "task_depths": list(task_depths),
            "task_source": args.task_source,
            "frontier_difficulties": list(frontier_difficulties),
            "frontier_registry_version": args.frontier_registry_version,
            "per_cell": args.per_cell,
            "holdout_per_cell": args.holdout_per_cell,
            "seed": args.seed,
            "init_seed": args.init_seed,
            "semantic_warmup_steps": args.semantic_warmup_steps,
            "state_warmup_steps": args.state_warmup_steps,
            "process_curriculum": args.process_curriculum,
            "process_family_batch_size": args.process_family_batch_size,
            "process_family_batch_mode": args.process_family_batch_mode,
            "process_gradient_combiner": args.process_gradient_combiner,
            "process_transformer_gradient_scale": (args.process_transformer_gradient_scale),
            "process_query_gradient_scale": args.process_query_gradient_scale,
            "direct_transition_processor": {
                "enabled": args.direct_transition_processor,
                "mode": args.transition_processor_mode,
                "copy_prior_logit_bias": args.transition_copy_prior_logit_bias,
                "objective": (
                    "actual_committed_state_public_action_history_to_exact_next_state"
                ),
                "curriculum": args.direct_transition_curriculum,
                "register_loss_weights": list(register_loss_weights),
                "weakest_register_weight": (
                    args.direct_transition_weakest_register_weight
                ),
                "curriculum_stages": (
                    [
                        "verified_window_1",
                        "verified_window_2",
                        "verified_window_4",
                        "closed_loop",
                        "controlled_recovery",
                        "closed_loop_final",
                    ]
                    if args.direct_transition_curriculum == "progressive"
                    else ["closed_loop"]
                ),
                "deployed_transition_policy": (
                    f"processor_{args.transition_processor_mode}"
                ),
                "opcode_expert_routing": args.transition_opcode_expert_routing,
                "transition_replay": {
                    "mode": args.transition_replay_mode,
                    "auxiliary_weight": args.transition_replay_auxiliary_weight,
                    "evidence": "ordered_public_action_prefix_only",
                    "state_independent_candidate": True,
                    "runtime_correctness_oracle_available": False,
                },
                "controlled_recovery_target": (
                    "true_transition_from_corrupted_state"
                ),
                "all_rollout_target_authority": (
                    "exact_transition_from_actual_committed_state"
                ),
                "gold_trace_after_initial": "consistency_check_only",
                "invalid_state_policy": "absorbing_structural_latch",
                "transformer_graph_constructed": False,
                "readout_graph_constructed": False,
            },
            "public_action_program": {
                "enabled": args.public_action_program,
                "source": "public_objective_literals_and_order_only",
                "verifier_answer_available": False,
                "exact_microcode_available": False,
            },
            "analytic_action_readout": {
                "enabled": args.analytic_action_readout_fit,
                "method": (
                    "training_only_affine_workspace_plus_public_signature_rbf_write"
                ),
                "affine_feature_source": "learned_recurrent_workspace",
                "kernel_feature_source": (
                    "ordered_public_literals_current_state_depth_and_family"
                ),
                "regularization": args.analytic_action_readout_ridge,
                "target_logit_margin": args.analytic_action_readout_margin,
                "holdout_fit_authority": False,
            },
            "answer_bridge_steps": args.answer_bridge_steps,
            "answer_bridge_inner_steps": args.answer_bridge_inner_steps,
            "answer_bridge_autonomous_tail_steps": (answer_bridge_autonomous_tail_steps),
            "max_steps": args.max_steps,
            "phase_schedule": phase_schedule,
            "answer_bridge_rollin_probability": (args.answer_bridge_rollin_probability),
            "answer_bridge_rollin_final_probability": (args.answer_bridge_rollin_final_probability),
            "student_rollin_probability": args.student_rollin_probability,
            "student_rollin_final_probability": rollin_final_probability,
            "state_teacher_forcing_probability": (args.state_teacher_forcing_probability),
            "state_teacher_forcing_final_probability": (
                args.state_teacher_forcing_final_probability
            ),
            "state_teacher_forcing_hold_fraction": (
                args.state_teacher_forcing_hold_fraction
            ),
            "max_gradient_norm": args.max_gradient_norm,
            "semantic_learning_rate": args.learning_rate,
            "answer_bridge_learning_rate": answer_bridge_learning_rate,
            "recurrent_learning_rate": recurrent_learning_rate,
            "state_learning_rate": state_learning_rate,
            "bridge": args.bridge,
            "window_tissue_mode": args.window_tissue_mode,
            "lora_rank": args.lora_rank,
            "controller_rank": args.controller_rank,
            "state_schema": args.state_schema,
            "state_slots": len(state_register_order),
            "state_register_order": list(state_register_order),
            "state_weight": args.state_weight,
            "stutter_weight": args.stutter_weight,
            "state_codebook": ("frozen_prelude_state_action_and_tokenizer_literal_labels"),
            "state_codebook_sha256": state_codebook_grounding["sha256"],
            "state_codebook_grounding": state_codebook_grounding,
            "literal_observation_contract": {
                **literal_contract.to_dict(),
                "contract_sha256": literal_contract.contract_sha256,
            },
            "numeric_observation_contract": {
                **numeric_observation_contract.to_dict(),
                "contract_sha256": numeric_observation_contract.contract_sha256,
                "encoding": "direct_category_then_ordered_radix_pair",
                "radix": 31,
            },
            "opcode_observation_contract": {
                **opcode_contract.to_dict(),
                "contract_sha256": opcode_contract.contract_sha256,
            },
            "frontier_family_observation_contract": {
                **family_contract.to_dict(),
                "contract_sha256": family_contract.contract_sha256,
            },
            "answer_emission_contract": {
                **answer_emission_contract.to_dict(),
                "contract_sha256": answer_emission_contract.contract_sha256,
            },
            "answer_bridge_supervision": answer_bridge_supervision,
            "depth_basis_size": args.depth_basis_size,
            "lora_targets": list(targets),
            "wiring": wiring,
            "readout_sha256": readout_sha256,
            "source_sha256s": source_sha256s,
            "campaign_binding": campaign_binding,
            "optimizer_contract": {
                "class": (
                    "mlx.optimizers.MultiOptimizer[Adam,Adam,Adam]"
                    if query_optimizer_enabled
                    else "mlx.optimizers.MultiOptimizer[Adam,Adam]"
                ),
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "bias_correction": False,
                "ownership_groups": {
                    **(
                        {"scoped_transformer_query": "independent_adam"}
                        if query_optimizer_enabled
                        else {}
                    ),
                    "scoped_transformer_bridge": "independent_adam",
                    "controller_and_readouts": "independent_adam",
                },
                "process_query_rate_scale": args.process_query_gradient_scale,
                "process_transformer_rate_scale": (
                    args.process_transformer_gradient_scale
                ),
                "phase_learning_rates": {
                    "semantic_anchor": args.learning_rate,
                    "answer_bridge": answer_bridge_learning_rate,
                    "state_transition": state_learning_rate,
                    "recurrence": recurrent_learning_rate,
                },
                "phase_transition_resets_optimizer_state": True,
                "process_component_transition_resets_optimizer_state": True,
            },
            "mlx_memory_envelope": envelope.to_receipt(),
        }
        bootstrap = (
            _bootstrap_bundle_from_checkpoint(
                args.bootstrap_output_dir.expanduser(),
                args.bootstrap_stem,
                bundle,
                expected_identity=identity,
            )
            if args.bootstrap_output_dir is not None
            else None
        )
        identity["initial_controller_sha256"] = controller.parameter_sha256()
        identity["bootstrap"] = bootstrap
        identity = adopt_source_migration_identity(out_dir, identity)
        if "identity_sha256" not in identity:
            identity["identity_sha256"] = _canonical_sha256(identity)

        def phase_learning_rate(phase: str) -> float:
            return {
                "semantic_anchor": args.learning_rate,
                "answer_bridge": answer_bridge_learning_rate,
                "state_transition": state_learning_rate,
                "recurrence": recurrent_learning_rate,
            }[phase]

        initial_phase = _optimization_phase(
            0,
            args.semantic_warmup_steps,
            args.state_warmup_steps,
            args.answer_bridge_steps,
        )
        optimizer = _ownership_optimizer(
            phase_learning_rate(
                initial_phase,
            ),
            transformer_rate_scale=(
                args.process_transformer_gradient_scale
                if initial_phase == "state_transition"
                else 1.0
            ),
            query_rate_scale=(
                args.process_query_gradient_scale
                if query_optimizer_enabled and initial_phase == "state_transition"
                else 1.0 if query_optimizer_enabled else None
            ),
        )
        should_restore = args.resume or args.resume_if_available
        step, history, restored_training_state = (
            _restore_checkpoint(
                out_dir,
                bundle,
                optimizer,
                identity,
                semantic_warmup_steps=args.semantic_warmup_steps,
                state_warmup_steps=args.state_warmup_steps,
                answer_bridge_steps=args.answer_bridge_steps,
                required=args.resume,
            )
            if should_restore
            else (0, [], {})
        )
        rollin_totals = _restore_rollin_totals(restored_training_state)
        invocation_start_step = step
        invocation_stop_step = _invocation_stop_step(
            invocation_start_step,
            args.max_steps,
            args.max_invocation_steps,
        )
        resumed_phase = _optimization_phase(
            step,
            args.semantic_warmup_steps,
            args.state_warmup_steps,
            args.answer_bridge_steps,
        )
        _set_ownership_optimizer_rates(
            optimizer,
            phase_learning_rate(resumed_phase),
            transformer_rate_scale=(
                args.process_transformer_gradient_scale
                if resumed_phase == "state_transition"
                else 1.0
            ),
            query_rate_scale=(
                args.process_query_gradient_scale
                if query_optimizer_enabled and resumed_phase == "state_transition"
                else 1.0 if query_optimizer_enabled else None
            ),
        )
        resource_guard_receipt: dict[str, Any] | None = None
        analytic_action_readout_receipt: dict[str, Any] | None = None
        if resource_guard_enabled:
            resource_guard_receipt = _await_resource_guard(
                args.resource_stage_path.expanduser(),
                trainer_sha256=_file_sha256(Path(__file__).resolve(strict=True)),
                startup_lethal_mb=float(args.resource_startup_lethal_mb),
                steady_lethal_mb=float(args.resource_steady_lethal_mb),
                timeout_s=float(args.resource_guard_timeout_s),
            )
        analytic_receipt_path = out_dir / "analytic_action_readout_fit.json"
        if args.analytic_action_readout_fit:
            if step == 0:
                analytic_action_readout_receipt = _fit_family_action_readout(
                    bundle,
                    tokenizer,
                    train_tasks,
                    state_spec,
                    bridge,
                    family_contract,
                    regularization=args.analytic_action_readout_ridge,
                    margin=args.analytic_action_readout_margin,
                )
                _atomic_canonical_json(
                    analytic_receipt_path,
                    analytic_action_readout_receipt,
                )
            else:
                try:
                    analytic_action_readout_receipt = json.loads(
                        analytic_receipt_path.read_text(encoding="ascii")
                    )
                except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
                    raise RuntimeError(
                        "analytic action readout resume receipt is unavailable"
                    ) from exc
                receipt_body = {
                    key: value
                    for key, value in analytic_action_readout_receipt.items()
                    if key != "receipt_sha256"
                }
                if analytic_action_readout_receipt.get(
                    "receipt_sha256"
                ) != _canonical_sha256(receipt_body):
                    raise RuntimeError("analytic action readout resume receipt differs")
        if phase_schedule["mode"] == "bootstrap_answer_bridge_only":
            bridge_preflight_diagnostic = _evaluate_answer_bridge_diagnostic(
                bundle,
                tokenizer,
                holdout,
                spec,
                bridge,
                answer_emission_contract,
                answer_digit_pointer_enabled=(args.task_source != "frontier_process"),
            )
            bridge_process_preflight = _answer_bridge_process_preflight(
                bridge_preflight_diagnostic,
                identity_sha256=identity["identity_sha256"],
                phase_schedule=phase_schedule,
                start_step=step,
            )
            _atomic_canonical_json(
                out_dir / "answer_bridge_process_preflight.json",
                {
                    **bridge_process_preflight,
                    "diagnostic": bridge_preflight_diagnostic,
                },
            )
            if not bridge_process_preflight["admitted"]:
                raise RuntimeError(bridge_process_preflight["reason"])
        print(
            f"[unified] step={step} trainable={sum(v.size for v in _trainable(bundle).values()):,} "
            f"readout={readout_sha256[:12]}",
            flush=True,
        )
        from core.brain.llm.latent_cortex.recurrence_adapter import (
            recurrence_adapter_scope,
        )

        with checkpointed_window(model, group_size=args.checkpoint_group):
            while step < invocation_stop_step and time.time() < deadline:
                phase = _optimization_phase(
                    step,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                    args.answer_bridge_steps,
                )
                if phase == "answer_bridge":
                    bridge_start = args.state_warmup_steps + args.semantic_warmup_steps
                    task = _answer_bridge_task(train_tasks, step - bridge_start)
                elif phase == "recurrence":
                    recurrent_start = (
                        args.semantic_warmup_steps
                        + args.state_warmup_steps
                        + args.answer_bridge_steps
                    )
                    task = _recurrent_training_task(
                        train_tasks,
                        tokenizer,
                        bridge,
                        step - recurrent_start,
                        cover_all_cells=args.task_source == "frontier_process",
                    )
                else:
                    task = train_tasks[step % len(train_tasks)]
                process_family_batch = (task,)
                if phase == "state_transition" and args.process_family_batch_size > 1:
                    process_family_batch = _process_family_training_batch(
                        train_tasks,
                        step,
                        args.process_family_batch_size,
                        mode=args.process_family_batch_mode,
                    )
                    task = process_family_batch[0]
                prompt, answer = encode_example(tokenizer, task, bridge)
                with recurrence_adapter_scope(start=None, stop=None):
                    update_applied = False
                    process_policy: dict[str, Any] | None = None
                    process_gradient_diagnostics: dict[str, Any] | None = None
                    process_gradient_combiner_receipt: dict[str, Any] | None = None
                    bridge_state_targets = None
                    bridge_action_targets = None
                    bridge_teacher_policy = None
                    semantic_depth = _semantic_execution_depth(task.depth, spec)
                    if phase == "answer_bridge":
                        bridge_state_targets = state_targets_from_trace(
                            task.transition_trace,
                            semantic_depth,
                            state_slots=controller.config.state_slots,
                        )
                        bridge_action_targets = action_targets_from_program(
                            task.transition_program,
                            semantic_depth,
                        )
                        process_capture = _capture_autonomous_process(
                            bundle,
                            prompt,
                            spec.plan_at(semantic_depth),
                        )
                        process_evidence = _process_evidence_from_capture(
                            task,
                            semantic_depth,
                            process_capture,
                        )
                        bridge_teacher_policy = _answer_bridge_teacher_policy(
                            step,
                            bridge_start=bridge_start,
                            bridge_steps=args.answer_bridge_steps,
                            autonomous_tail_steps=(answer_bridge_autonomous_tail_steps),
                            process_exact=process_evidence["process_exact"],
                        )
                        rollin_totals["last_state_teacher_forcing_probability"] = (
                            bridge_teacher_policy["state_teacher_forcing_probability"]
                        )
                        if bridge_teacher_policy["autonomous_tail"]:
                            rollin_totals["answer_bridge_autonomous_tail_examples"] += 1
                            rollin_totals["answer_bridge_autonomous_process_exact"] += int(
                                process_evidence["process_exact"]
                            )
                    binding_targets = (
                        _answer_role_place_targets(
                            task.family,
                            answer,
                            answer_emission_contract,
                        )
                        if phase == "answer_bridge"
                        else None
                    )
                    if phase == "answer_bridge" and not bridge_teacher_policy["update_admitted"]:
                        rollin_totals["answer_bridge_wrong_process_updates_blocked"] += 1
                        loss = mx.array(0.0, dtype=mx.float32)
                        update_applied = True
                    elif (
                        phase == "answer_bridge"
                        and args.answer_bridge_inner_steps > 1
                        and binding_targets is not None
                    ):
                        features = _cached_answer_binding_features(
                            bundle,
                            prompt,
                            answer,
                            spec.plan_at(semantic_depth),
                            initial_state_teacher_values=(bridge_state_targets.initial_values),
                            state_teacher_values=bridge_state_targets.values,
                            action_teacher_values=bridge_action_targets.values,
                            state_teacher_forcing_probability=(
                                bridge_teacher_policy["state_teacher_forcing_probability"]
                            ),
                        )
                        for _inner_step in range(args.answer_bridge_inner_steps):
                            loss, gradients = nn.value_and_grad(
                                bundle,
                                _cached_answer_binding_loss,
                            )(bundle, features, binding_targets)
                            _apply_training_gradients(
                                bundle,
                                optimizer,
                                gradients,
                                phase=phase,
                                max_norm=args.max_gradient_norm,
                                totals=rollin_totals,
                                loss=loss,
                            )
                        rollin_totals["answer_bridge_inner_updates"] += (
                            args.answer_bridge_inner_steps
                        )
                        update_applied = True
                    elif phase in {"semantic_anchor", "answer_bridge"}:
                        effective = None
                        if phase == "answer_bridge":
                            bridge_start = args.state_warmup_steps + args.semantic_warmup_steps
                            bridge_stop = bridge_start + args.answer_bridge_steps
                            rollin_probability = _student_rollin_probability(
                                step,
                                semantic_warmup_steps=bridge_start,
                                max_steps=bridge_stop,
                                initial=args.answer_bridge_rollin_probability,
                                final=args.answer_bridge_rollin_final_probability,
                            )
                            generated = _generate_student_rollin(
                                bundle,
                                prompt,
                                answer,
                                spec.plan_at(semantic_depth),
                                eos_token_id=tokenizer.eos_token_id,
                                answer_emission_contract=answer_emission_contract,
                                answer_digit_pointer_enabled=(
                                    args.task_source != "frontier_process"
                                ),
                                state_slot_start=int(prompt.shape[-1]),
                            )
                            effective, selected = _deterministic_student_mix(
                                answer,
                                generated,
                                probability=rollin_probability,
                                seed=args.seed * 1_000_003 + step,
                                interchangeable_token_ids=(
                                    None
                                    if args.task_source == "frontier_process"
                                    else frozenset(answer_emission_contract.digit_token_ids)
                                ),
                            )
                            _record_student_rollin(
                                rollin_totals,
                                answer,
                                generated,
                                effective,
                                selected,
                                rollin_probability,
                            )

                        def semantic_objective(
                            candidate: UnifiedTrainingBundle,
                            objective_prompt: Any,
                            objective_answer: Any,
                            objective_depth: int = semantic_depth,
                            objective_rollin: Any | None = effective,
                            objective_binding_targets: Any = binding_targets,
                            objective_state_targets: Any = bridge_state_targets,
                            objective_action_targets: Any = bridge_action_targets,
                            objective_teacher_probability: float = (
                                bridge_teacher_policy["state_teacher_forcing_probability"]
                                if bridge_teacher_policy is not None
                                else 0.0
                            ),
                        ):
                            role_logits: list[Any] = []
                            place_logits: list[Any] = []
                            _recurrent, _states, losses, _state_logits = (
                                unified_answer_and_recurrent_trajectory(
                                    candidate.model,
                                    objective_prompt,
                                    objective_answer,
                                    spec.plan_at(objective_depth),
                                    candidate.controller,
                                    decoder_input_tokens=objective_rollin,
                                    use_state_slots=True,
                                    answer_role_logit_trajectory=role_logits,
                                    answer_place_logit_trajectory=place_logits,
                                    initial_state_teacher_values=(
                                        objective_state_targets.initial_values
                                        if objective_state_targets is not None
                                        else None
                                    ),
                                    state_teacher_values=(
                                        objective_state_targets.values
                                        if objective_state_targets is not None
                                        else None
                                    ),
                                    action_teacher_values=(
                                        objective_action_targets.values
                                        if objective_action_targets is not None
                                        else None
                                    ),
                                    state_teacher_forcing_probability=(
                                        objective_teacher_probability
                                    ),
                                )
                            )
                            if objective_binding_targets is None:
                                return losses[-1]
                            if not role_logits or len(role_logits) != len(place_logits):
                                raise RuntimeError("answer bridge emitted no binding trajectory")
                            role_targets, place_targets = objective_binding_targets
                            binding_loss = _answer_binding_loss(
                                role_logits[-1],
                                place_logits[-1],
                                role_targets,
                                place_targets,
                            )
                            return losses[-1] + binding_loss

                        loss, gradients = nn.value_and_grad(
                            bundle,
                            semantic_objective,
                        )(bundle, prompt, answer)
                    else:
                        if phase == "state_transition":
                            effective = answer
                            objective_spec = state_spec
                            process_policy = _process_training_policy(
                                step - args.semantic_warmup_steps,
                                args.state_warmup_steps,
                                args.process_curriculum,
                                initial_teacher_probability=(
                                    args.state_teacher_forcing_probability
                                ),
                                final_teacher_probability=(
                                    args.state_teacher_forcing_final_probability
                                ),
                                teacher_hold_fraction=(
                                    args.state_teacher_forcing_hold_fraction
                                ),
                            )
                            state_teacher_probability = process_policy[
                                "teacher_forcing_probability"
                            ]
                            rollin_totals["last_process_component"] = process_policy["component"]
                            rollin_totals["last_process_stage_progress"] = process_policy[
                                "stage_progress"
                            ]
                        else:
                            recurrent_start = (
                                args.semantic_warmup_steps
                                + args.state_warmup_steps
                                + args.answer_bridge_steps
                            )
                            rollin_probability = _student_rollin_probability(
                                step,
                                semantic_warmup_steps=recurrent_start,
                                max_steps=args.max_steps,
                                initial=args.student_rollin_probability,
                                final=rollin_final_probability,
                            )
                            state_teacher_probability = _student_rollin_probability(
                                step,
                                semantic_warmup_steps=recurrent_start,
                                max_steps=args.max_steps,
                                initial=args.state_teacher_forcing_probability,
                                final=args.state_teacher_forcing_final_probability,
                            )
                            generated = _generate_student_rollin(
                                bundle,
                                prompt,
                                answer,
                                spec.plan_at(max(spec.train_depths)),
                                eos_token_id=tokenizer.eos_token_id,
                                answer_emission_contract=answer_emission_contract,
                                answer_digit_pointer_enabled=(
                                    args.task_source != "frontier_process"
                                ),
                                state_slot_start=int(prompt.shape[-1]),
                            )
                            effective, selected = _deterministic_student_mix(
                                answer,
                                generated,
                                probability=rollin_probability,
                                seed=args.seed * 1_000_003 + step,
                                interchangeable_token_ids=(
                                    None
                                    if args.task_source == "frontier_process"
                                    else frozenset(answer_emission_contract.digit_token_ids)
                                ),
                            )
                            _record_student_rollin(
                                rollin_totals,
                                answer,
                                generated,
                                effective,
                                selected,
                                rollin_probability,
                            )
                            objective_spec = spec
                        rollin_totals["last_state_teacher_forcing_probability"] = (
                            state_teacher_probability
                        )

                        def recurrent_objective(
                            candidate: UnifiedTrainingBundle,
                            objective_prompt: Any,
                            objective_answer: Any,
                            objective_rollin: Any,
                            transition_trace: Any = task.transition_trace,
                            transition_program: Any = task.transition_program,
                            state_teacher_forcing_probability: float = (state_teacher_probability),
                            training_spec: UnifiedIntrinsicTrainingSpec = (objective_spec),
                        ):
                            return unified_intrinsic_training_loss(
                                candidate.model,
                                objective_prompt,
                                objective_answer,
                                candidate.controller,
                                training_spec,
                                readout_sha256=readout_sha256,
                                decoder_input_tokens=objective_rollin,
                                transition_trace=transition_trace,
                                transition_program=transition_program,
                                state_teacher_forcing_probability=(
                                    state_teacher_forcing_probability
                                ),
                                answer_digit_pointer_enabled=(
                                    args.task_source != "frontier_process"
                                ),
                            )[0]

                        if phase == "state_transition":
                            cohort_losses: list[Any] = []
                            cohort_gradients: list[Any] = []
                            for cohort_task in process_family_batch:
                                cohort_prompt, _cohort_answer = encode_example(
                                    tokenizer,
                                    cohort_task,
                                    bridge,
                                )
                                cohort_public_actions = (
                                    _public_actions_for_task(
                                        cohort_task,
                                        max(state_spec.train_depths),
                                    )
                                    if args.public_action_program
                                    else None
                                )
                                direct_curriculum = (
                                    _direct_transition_curriculum_window(
                                        step,
                                        args.state_warmup_steps,
                                        min(
                                            max(state_spec.train_depths),
                                            int(cohort_task.transition_trace.depth),
                                            len(cohort_task.transition_program.actions),
                                        ),
                                        mode=args.direct_transition_curriculum,
                                    )
                                    if args.direct_transition_processor
                                    else None
                                )

                                def process_objective(
                                    candidate: UnifiedTrainingBundle,
                                    objective_prompt: Any,
                                    transition_trace: Any = cohort_task.transition_trace,
                                    transition_program: Any = cohort_task.transition_program,
                                    component: str = process_policy["component"],
                                    teacher_probability: float = state_teacher_probability,
                                    public_actions: tuple[tuple[int, ...], ...] | None = (
                                        cohort_public_actions
                                    ),
                                    curriculum: dict[str, Any] | None = direct_curriculum,
                                ) -> Any:
                                    if args.direct_transition_processor:
                                        if public_actions is None or curriculum is None:
                                            raise RuntimeError(
                                                "direct transition processor has no public actions"
                                            )
                                        return unified_typed_transition_processor_loss(
                                            candidate.controller,
                                            state_spec.plan_at(max(state_spec.train_depths)),
                                            transition_trace=transition_trace,
                                            transition_program=transition_program,
                                            public_action_values=public_actions,
                                            opcode_expert_routing=(
                                                args.transition_opcode_expert_routing
                                            ),
                                            transition_start=curriculum[
                                                "transition_start"
                                            ],
                                            transition_count=curriculum[
                                                "transition_count"
                                            ],
                                            corrupt_transition=curriculum[
                                                "corrupt_transition"
                                            ],
                                            corrupt_state_mode=(
                                                curriculum["corrupt_state_mode"]
                                                or "single_slot_offset"
                                            ),
                                            corrupt_state_slot=(
                                                1
                                                if curriculum[
                                                    "corrupt_state_slot"
                                                ]
                                                is None
                                                else curriculum[
                                                    "corrupt_state_slot"
                                                ]
                                            ),
                                            corrupt_state_offset=(
                                                1
                                                if curriculum[
                                                    "corrupt_state_offset"
                                                ]
                                                is None
                                                else curriculum[
                                                    "corrupt_state_offset"
                                                ]
                                            ),
                                            weakest_register_weight=(
                                                args.direct_transition_weakest_register_weight
                                            ),
                                            transition_processor_mode=(
                                                args.transition_processor_mode
                                            ),
                                            transition_copy_prior_logit_bias=(
                                                args.transition_copy_prior_logit_bias
                                            ),
                                            transition_replay_mode=(
                                                args.transition_replay_mode
                                            ),
                                            replay_auxiliary_weight=(
                                                args.transition_replay_auxiliary_weight
                                            ),
                                        )[0]
                                    return unified_process_training_loss(
                                        candidate.model,
                                        objective_prompt,
                                        candidate.controller,
                                        state_spec.plan_at(max(state_spec.train_depths)),
                                        transition_trace=transition_trace,
                                        transition_program=transition_program,
                                        state_teacher_forcing_probability=teacher_probability,
                                        state_weight=state_spec.state_weight,
                                        component=component,
                                        public_action_values=public_actions,
                                        microcode_lesion=args.public_action_program,
                                        transition_processor_mode=(
                                            args.transition_processor_mode
                                            if args.public_action_program
                                            else "residual"
                                        ),
                                        transition_copy_prior_logit_bias=(
                                            args.transition_copy_prior_logit_bias
                                        ),
                                        transition_opcode_expert_routing=(
                                            args.transition_opcode_expert_routing
                                        ),
                                        transition_replay_mode=(
                                            args.transition_replay_mode
                                        ),
                                    )[0]

                                cohort_loss, cohort_gradient = nn.value_and_grad(
                                    bundle,
                                    process_objective,
                                )(bundle, cohort_prompt)
                                cohort_gradient = tree_map(mx.stop_gradient, cohort_gradient)
                                mx.eval(
                                    cohort_loss,
                                    *[value for _name, value in tree_flatten(cohort_gradient)],
                                )
                                cohort_losses.append(mx.stop_gradient(cohort_loss))
                                cohort_gradients.append(cohort_gradient)
                            loss = mx.mean(mx.stack(cohort_losses))
                            if (
                                (step + 1) % args.eval_every == 0
                                or step + 1 == args.max_steps
                            ):
                                process_gradient_diagnostics = (
                                    _gradient_conflict_diagnostics(
                                        cohort_gradients,
                                        [
                                            str(item.task_id)
                                            for item in process_family_batch
                                        ],
                                        ownership_group="typed_state_transition",
                                    )
                                )
                            gradients, process_gradient_combiner_receipt = (
                                _combine_process_gradient_trees(
                                    cohort_gradients,
                                    [
                                        str(item.task_id)
                                        for item in process_family_batch
                                    ],
                                    mode=args.process_gradient_combiner,
                                    ownership_group="typed_state_transition",
                                )
                            )
                        elif phase == "recurrence":
                            loss, gradients = _streamed_recurrent_objective_gradients(
                                bundle,
                                prompt,
                                answer,
                                spec,
                                readout_sha256=readout_sha256,
                                decoder_input_tokens=effective,
                                transition_trace=task.transition_trace,
                                transition_program=task.transition_program,
                                state_teacher_forcing_probability=(state_teacher_probability),
                                answer_digit_pointer_enabled=(
                                    args.task_source != "frontier_process"
                                ),
                                envelope=envelope,
                            )
                        else:
                            loss, gradients = nn.value_and_grad(
                                bundle,
                                recurrent_objective,
                            )(
                                bundle,
                                prompt,
                                answer,
                                effective,
                            )
                    if not update_applied:
                        _apply_training_gradients(
                            bundle,
                            optimizer,
                            gradients,
                            phase=phase,
                            max_norm=args.max_gradient_norm,
                            totals=rollin_totals,
                            loss=loss,
                            process_component=(
                                process_policy["component"] if process_policy is not None else None
                            ),
                        )
                step += 1
                next_phase = _optimization_phase(
                    step,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                    args.answer_bridge_steps,
                )
                next_process_component = None
                if next_phase == "state_transition":
                    next_process_component = _process_training_policy(
                        step - args.semantic_warmup_steps,
                        args.state_warmup_steps,
                        args.process_curriculum,
                        initial_teacher_probability=(
                            args.state_teacher_forcing_probability
                        ),
                        final_teacher_probability=(
                            args.state_teacher_forcing_final_probability
                        ),
                        teacher_hold_fraction=(
                            args.state_teacher_forcing_hold_fraction
                        ),
                    )["component"]
                if next_phase != phase or (
                    process_policy is not None
                    and next_process_component != process_policy["component"]
                ):
                    optimizer = _ownership_optimizer(
                        phase_learning_rate(next_phase),
                        transformer_rate_scale=(
                            args.process_transformer_gradient_scale
                            if next_phase == "state_transition"
                            else 1.0
                        ),
                        query_rate_scale=(
                            args.process_query_gradient_scale
                            if query_optimizer_enabled and next_phase == "state_transition"
                            else 1.0 if query_optimizer_enabled else None
                        ),
                    )
                if step % 5 == 0:
                    print(
                        f"[step {step}] phase={phase} "
                        f"loss={float(loss.item()):.5f} "
                        f"elapsed_min={(time.time() - started) / 60.0:.1f}",
                        flush=True,
                    )
                if step % args.eval_every == 0 or step == args.max_steps:
                    report = _evaluate(
                        bundle,
                        tokenizer,
                        holdout,
                        spec,
                        bridge,
                        spec.depths,
                        envelope=envelope,
                        public_action_program=args.public_action_program,
                        transition_processor_mode=args.transition_processor_mode,
                        transition_copy_prior_logit_bias=(
                            args.transition_copy_prior_logit_bias
                        ),
                        transition_opcode_expert_routing=(
                            args.transition_opcode_expert_routing
                        ),
                        transition_replay_mode=args.transition_replay_mode,
                        direct_transition_processor=(
                            args.direct_transition_processor
                        ),
                    )
                    report["step"] = step
                    report["optimization_phase"] = next_phase
                    report["student_rollin"] = _rollin_report(
                        rollin_totals,
                        initial_probability=args.student_rollin_probability,
                        final_probability=rollin_final_probability,
                    )
                    report["process_gradient_diagnostics"] = (
                        process_gradient_diagnostics
                    )
                    report["process_gradient_combiner"] = (
                        process_gradient_combiner_receipt
                    )
                    history.append(report)
                    print(f"[eval {step}] {report}", flush=True)
                    prior = history[:-1]
                    if report["heldout_depth_helps"] and (
                        not prior
                        or report["best_heldout_relative_gain"]
                        > max(row.get("best_heldout_relative_gain", float("-inf")) for row in prior)
                    ):
                        _save_checkpoint(
                            out_dir,
                            bundle,
                            optimizer,
                            step=step,
                            history=history,
                            identity=identity,
                            stem="checkpoint_best_heldout",
                            optimization_phase=next_phase,
                            training_state={"rollin_totals": rollin_totals},
                        )
                    if report["trained_depth_helps"] and (
                        not prior
                        or report["best_deep_relative_gain"]
                        > max(row.get("best_deep_relative_gain", float("-inf")) for row in prior)
                    ):
                        _save_checkpoint(
                            out_dir,
                            bundle,
                            optimizer,
                            step=step,
                            history=history,
                            identity=identity,
                            stem="checkpoint_best_trained",
                            optimization_phase=next_phase,
                            training_state={"rollin_totals": rollin_totals},
                        )
                if step % args.checkpoint_every == 0 or step == args.max_steps:
                    _save_checkpoint(
                        out_dir,
                        bundle,
                        optimizer,
                        step=step,
                        history=history,
                        identity=identity,
                        optimization_phase=next_phase,
                        training_state={"rollin_totals": rollin_totals},
                    )
                envelope.reclaim(force=True)

        if (
            not history
            or int(history[-1].get("step", -1)) != step
            or set(history[-1].get("ce", {})) != {f"T{depth}" for depth in spec.depths}
            or "heldout_depth_helps" not in history[-1]
        ):
            final_ladder = _evaluate(
                bundle,
                tokenizer,
                holdout,
                spec,
                bridge,
                spec.depths,
                envelope=envelope,
                public_action_program=args.public_action_program,
                transition_processor_mode=args.transition_processor_mode,
                transition_copy_prior_logit_bias=(
                    args.transition_copy_prior_logit_bias
                ),
                transition_opcode_expert_routing=(
                    args.transition_opcode_expert_routing
                ),
                transition_replay_mode=args.transition_replay_mode,
                direct_transition_processor=args.direct_transition_processor,
            )
            final_ladder["step"] = step
            final_ladder["optimization_phase"] = _optimization_phase(
                step,
                args.semantic_warmup_steps,
                args.state_warmup_steps,
                args.answer_bridge_steps,
            )
            final_ladder["full_depth_ladder"] = True
            history.append(final_ladder)
            _save_checkpoint(
                out_dir,
                bundle,
                optimizer,
                step=step,
                history=history,
                identity=identity,
                optimization_phase=_optimization_phase(
                    step,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                    args.answer_bridge_steps,
                ),
                training_state={"rollin_totals": rollin_totals},
            )
        final_readout = readout_fingerprint(model, spec.coda_start)
        if final_readout != readout_sha256:
            raise RuntimeError("unified training changed the frozen readout")
        final = history[-1] if history else None
        answer_bridge_diagnostic = (
            _evaluate_answer_bridge_diagnostic(
                bundle,
                tokenizer,
                holdout,
                spec,
                bridge,
                answer_emission_contract,
                answer_digit_pointer_enabled=(args.task_source != "frontier_process"),
            )
            if args.answer_bridge_steps > 0 and step >= args.max_steps
            else None
        )
        answer_bridge_admission = (
            _evaluate_answer_bridge_admission(
                bundle,
                tokenizer,
                holdout,
                spec,
                bridge,
                answer_emission_contract,
                answer_digit_pointer_enabled=(args.task_source != "frontier_process"),
            )
            if args.answer_bridge_steps > 0 and step >= args.max_steps
            else None
        )
        process_admission = (
            _evaluate_process_admission(
                bundle,
                tokenizer,
                holdout,
                spec,
                bridge,
                public_action_program=args.public_action_program,
                transition_opcode_expert_routing=(
                    args.transition_opcode_expert_routing
                ),
                transition_replay_mode=args.transition_replay_mode,
            )
            if args.answer_bridge_steps == 0 and step >= args.max_steps
            else None
        )
        if answer_bridge_admission is not None and answer_bridge_admission["admitted"]:
            _save_checkpoint(
                out_dir,
                bundle,
                optimizer,
                step=step,
                history=history,
                identity=identity,
                stem="checkpoint_answer_bridge_admitted",
                optimization_phase=_optimization_phase(
                    step,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                    args.answer_bridge_steps,
                ),
                training_state={"rollin_totals": rollin_totals},
            )
        if process_admission is not None and process_admission["admitted"]:
            _save_checkpoint(
                out_dir,
                bundle,
                optimizer,
                step=step,
                history=history,
                identity=identity,
                stem="checkpoint_process_admitted",
                optimization_phase=_optimization_phase(
                    step,
                    args.semantic_warmup_steps,
                    args.state_warmup_steps,
                    args.answer_bridge_steps,
                ),
                training_state={"rollin_totals": rollin_totals},
            )
        halt_reason = _training_halt_reason(
            step=step,
            max_steps=args.max_steps,
            invocation_stop_step=invocation_stop_step,
        )
        with interprocess_file_lock(out_dir / ".unified_checkpoint.lock"):
            latest_checkpoint = _load_latest_checkpoint(out_dir, required=True)
            if latest_checkpoint is None:
                raise RuntimeError("unified recurrence final checkpoint is unavailable")
            checkpoint_receipt, checkpoint_weights_path = latest_checkpoint
            checkpoint_size_bytes = checkpoint_weights_path.stat().st_size
        body = {
            "schema": TRAINING_SCHEMA,
            "identity": identity,
            "steps": step,
            "history": history,
            "final": final,
            "readout_sha256_before": readout_sha256,
            "readout_sha256_after": final_readout,
            "readout_frozen": True,
            "complete": step >= args.max_steps,
            "halt_reason": halt_reason,
            "invocation": {
                "start_step": invocation_start_step,
                "end_step": step,
                "max_invocation_steps": args.max_invocation_steps,
                "planned_stop_step": invocation_stop_step,
                "max_minutes": args.max_minutes,
                "preload_host_pressure": preload_host_pressure,
                "resource_guard": resource_guard_receipt,
            },
            "latest_checkpoint": {
                "step": checkpoint_receipt["step"],
                "optimization_phase": checkpoint_receipt["optimization_phase"],
                "checkpoint_sha256": checkpoint_receipt["checkpoint_sha256"],
                "checkpoint_size_bytes": checkpoint_size_bytes,
                "receipt_sha256": checkpoint_receipt["receipt_sha256"],
            },
            "elapsed_minutes": round((time.time() - started) / 60.0, 3),
            "answer_bridge_diagnostic": answer_bridge_diagnostic,
            "answer_bridge_admission": answer_bridge_admission,
            "process_admission": process_admission,
            "analytic_action_readout_fit": analytic_action_readout_receipt,
            "phase_schedule": phase_schedule,
            "verdict": _training_verdict(
                complete=step >= args.max_steps,
                answer_bridge_admission=answer_bridge_admission,
                process_admission=process_admission,
                final=final,
            ),
        }
        receipt = {**body, "receipt_sha256": _canonical_sha256(body)}
        _atomic_canonical_json(out_dir / "training_receipt.json", receipt)
        print(f"[verdict] {receipt['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
