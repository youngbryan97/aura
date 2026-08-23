#!/usr/bin/env python3
"""Run all admitted candidate-cortex stages with durable stage boundaries."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.candidate_cortex_admission import (  # noqa: E402
    adjudicate_checkpoint_evidence,
)
from core.learning.candidate_cortex_training import (  # noqa: E402
    ADAPTIVE_RESULT_SCHEMA,
    ADMISSION_SCHEMA,
    JOURNAL_FILE,
    OBSERVATION_SCHEMA,
    CandidateCortexTrainingError,
    StagePolicy,
    adaptive_result_path,
    append_authenticated_event,
    build_stage_command,
    canonical_json_bytes,
    discover_exact_checkpoint,
    document_sha256,
    effective_stage_evidence,
    execution_admission,
    file_sha256,
    load_and_verify_plan,
    next_stage_plan,
    read_authenticated_journal,
    stage_adapter_root,
    validate_stage_observation,
)
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402
from tools import measure_candidate_cortex_checkpoint as measurement  # noqa: E402
from tools import run_detached_step as detached  # noqa: E402
from tools.run_candidate_cortex_canary_target import _mlx_arguments  # noqa: E402

STAGE_COMPLETION_SCHEMA = "aura.candidate_cortex_training.stage_completion.v1"
SEGMENTED_STAGE_COMPLETION_SCHEMA = (
    "aura.candidate_cortex_training.stage_completion.v2"
)
SEGMENT_COMPLETION_SCHEMA = "aura.candidate_cortex_training.segment_completion.v1"
PHASE_BOUNDARY_SCHEMA = "aura.candidate_cortex_training.phase_boundary.v1"
MAX_QWEN_HYBRID_SEGMENT_ITERATIONS = 48


@dataclass(frozen=True)
class StageSegment:
    index: int
    start_iteration: int
    iterations: int

    @property
    def end_iteration(self) -> int:
        return self.start_iteration + self.iterations


def _key(path: Path) -> bytes:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise CandidateCortexTrainingError("journal_key_invalid")
    payload = resolved.read_bytes()
    if len(payload) < 32:
        raise CandidateCortexTrainingError("journal_key_too_short")
    return payload


def _stage_parent(plan: Mapping[str, Any], stage_index: int) -> Path:
    return stage_adapter_root(plan, stage_index).parent


def _completion_path(plan: Mapping[str, Any], stage_index: int) -> Path:
    return _stage_parent(plan, stage_index) / "stage_completion.json"


def _evidence_path(plan: Mapping[str, Any], stage_index: int) -> Path:
    return _stage_parent(plan, stage_index) / "checkpoint_evidence.json"


def _detail_path(plan: Mapping[str, Any], stage_index: int) -> Path:
    return _stage_parent(plan, stage_index) / "checkpoint_measurement_detail.json"


def _segment_parent(
    plan: Mapping[str, Any], stage_index: int, segment_index: int
) -> Path:
    return (
        _stage_parent(plan, stage_index)
        / "segments"
        / f"segment-{segment_index:04d}"
    )


def _segment_adapter_root(
    plan: Mapping[str, Any], stage_index: int, segment_index: int
) -> Path:
    return _segment_parent(plan, stage_index, segment_index) / "adapter"


def _segment_completion_path(
    plan: Mapping[str, Any], stage_index: int, segment_index: int
) -> Path:
    return _segment_parent(plan, stage_index, segment_index) / "segment_completion.json"


def _segment_optimizer_path(
    plan: Mapping[str, Any], stage_index: int, segment_index: int
) -> Path:
    return _segment_parent(plan, stage_index, segment_index) / "optimizer_state.safetensors"


def _stage_segments(plan: Mapping[str, Any], stage_index: int) -> tuple[StageSegment, ...]:
    """Bound a hybrid-model stage below the known MLX Metal-handle horizon."""

    policy = StagePolicy(**dict(plan["stages"]))
    training = dict(plan["training"])
    accumulation = int(training["gradient_accumulation_steps"])
    if accumulation <= 0:
        raise CandidateCortexTrainingError("gradient_accumulation_invalid")
    horizon = MAX_QWEN_HYBRID_SEGMENT_ITERATIONS
    horizon -= horizon % accumulation
    if horizon <= 0:
        raise CandidateCortexTrainingError("segment_horizon_invalid")
    total = policy.iterations(stage_index)
    if total % accumulation:
        raise CandidateCortexTrainingError("stage_not_optimizer_aligned")
    segments: list[StageSegment] = []
    start = 0
    while start < total:
        iterations = min(horizon, total - start)
        segments.append(StageSegment(len(segments), start, iterations))
        start += iterations
    return tuple(segments)


def _strict_document(path: Path) -> dict[str, Any]:
    value = measurement._strict_json(path)  # noqa: SLF001 - shared strict boundary
    if not isinstance(value, dict):
        raise CandidateCortexTrainingError("adaptive_document_invalid")
    return value


def _write_once(path: Path, value: Mapping[str, Any], *, source: str) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    with local_internal_governed_scope(source, domain="file_write"):
        created = get_file_write_gateway().write_bytes_if_absent(
            path,
            payload,
            mode=0o600,
            source=source,
        )
    if not created and path.read_bytes() != payload:
        raise CandidateCortexTrainingError("adaptive_output_conflict")


def _validated_completion(
    plan: Mapping[str, Any], stage_index: int
) -> dict[str, Any] | None:
    path = _completion_path(plan, stage_index)
    if not path.is_file():
        return None
    document = _strict_document(path)
    material = dict(document)
    claimed = material.pop("completion_sha256", None)
    policy = StagePolicy(**dict(plan["stages"]))
    required_v1 = {
        "schema",
        "plan_sha256",
        "stage_index",
        "stage_iterations",
        "cumulative_iterations",
        "training_command_sha256",
        "checkpoint",
        "target_pid",
        "target_start_token",
        "host_metrics",
        "completion_sha256",
    }
    required_v2 = required_v1 | {
        "segments",
        "training_execution_sha256",
    }
    schema = document.get("schema")
    required = (
        required_v2
        if schema == SEGMENTED_STAGE_COMPLETION_SCHEMA
        else required_v1
    )
    if (
        set(document) != required
        or schema not in {
            STAGE_COMPLETION_SCHEMA,
            SEGMENTED_STAGE_COMPLETION_SCHEMA,
        }
        or document.get("plan_sha256") != plan["plan_sha256"]
        or document.get("stage_index") != stage_index
        or document.get("stage_iterations") != policy.iterations(stage_index)
        or document.get("cumulative_iterations")
        != policy.cumulative_iterations(stage_index)
        or claimed != document_sha256(material)
    ):
        raise CandidateCortexTrainingError("stage_completion_invalid")
    if schema == SEGMENTED_STAGE_COMPLETION_SCHEMA:
        segments = document.get("segments")
        expected_specs = _stage_segments(plan, stage_index)
        if not isinstance(segments, list) or len(segments) != len(expected_specs):
            raise CandidateCortexTrainingError("stage_segment_execution_invalid")
        rebuilt: list[dict[str, Any]] = []
        for spec, claimed_segment in zip(expected_specs, segments, strict=True):
            completion = _validated_segment_completion(plan, stage_index, spec)
            if completion is None:
                raise CandidateCortexTrainingError("stage_segment_execution_invalid")
            expected_segment = {
                "segment": asdict(spec),
                "completion_sha256": completion["completion_sha256"],
                "command_sha256": completion["command_sha256"],
                "adapter_sha256": completion["adapter"]["sha256"],
                "optimizer_state_sha256": completion["optimizer_state"]["sha256"],
            }
            if claimed_segment != expected_segment:
                raise CandidateCortexTrainingError("stage_segment_execution_invalid")
            rebuilt.append(expected_segment)
        if document.get("training_execution_sha256") != document_sha256(rebuilt):
            raise CandidateCortexTrainingError("stage_segment_execution_invalid")
    checkpoint = discover_exact_checkpoint(
        Path(str(plan["paths"]["checkpoint_root"])),
        expected_cumulative_iterations=policy.cumulative_iterations(stage_index),
    )
    if document.get("checkpoint") != checkpoint:
        raise CandidateCortexTrainingError("stage_completion_checkpoint_drift")
    return document


def _validated_segment_completion(
    plan: Mapping[str, Any], stage_index: int, segment: StageSegment
) -> dict[str, Any] | None:
    path = _segment_completion_path(plan, stage_index, segment.index)
    if not path.is_file():
        return None
    document = _strict_document(path)
    material = dict(document)
    claimed = material.pop("completion_sha256", None)
    required = {
        "schema",
        "plan_sha256",
        "stage_index",
        "segment",
        "command_sha256",
        "adapter",
        "optimizer_state",
        "host_metrics",
        "completion_sha256",
    }
    if (
        set(document) != required
        or document.get("schema") != SEGMENT_COMPLETION_SCHEMA
        or document.get("plan_sha256") != plan["plan_sha256"]
        or document.get("stage_index") != stage_index
        or document.get("segment") != asdict(segment)
        or claimed != document_sha256(material)
    ):
        raise CandidateCortexTrainingError("segment_completion_invalid")
    for field in ("adapter", "optimizer_state"):
        binding = document.get(field)
        if not isinstance(binding, dict):
            raise CandidateCortexTrainingError("segment_artifact_invalid")
        artifact = Path(str(binding.get("path"))).resolve(strict=True)
        if (
            not artifact.is_file()
            or file_sha256(artifact) != binding.get("sha256")
            or artifact.stat().st_size != binding.get("size_bytes")
        ):
            raise CandidateCortexTrainingError("segment_artifact_drift")
    return document


def _reset_incomplete_stage(plan: Mapping[str, Any], stage_index: int) -> None:
    parent = _stage_parent(plan, stage_index)
    policy = StagePolicy(**dict(plan["stages"]))
    canonical = (
        Path(str(plan["paths"]["checkpoint_root"]))
        / f"{policy.cumulative_iterations(stage_index):07d}_adapters.safetensors"
    )
    with local_internal_governed_scope(
        "candidate_cortex_adaptive.reset_stage", domain="file_write"
    ):
        gateway = get_file_write_gateway()
        if parent.exists() or parent.is_symlink():
            gateway.delete_path(
                parent,
                recursive=True,
                source="candidate_cortex_adaptive.reset_stage",
            )
        if canonical.exists() or canonical.is_symlink():
            gateway.delete_file(
                canonical,
                source="candidate_cortex_adaptive.reset_stage",
            )
        gateway.ensure_directory(
            stage_adapter_root(plan, stage_index),
            source="candidate_cortex_adaptive.reset_stage",
        )


def _sample_host(stop: threading.Event, state: dict[str, Any]) -> None:
    process = psutil.Process(os.getpid())
    while True:
        virtual = psutil.virtual_memory()
        try:
            rss = int(process.memory_info().rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            rss = 0
        state["sample_count"] += 1
        state["min_available_bytes"] = min(
            state["min_available_bytes"], int(virtual.available)
        )
        state["max_used_percent"] = max(
            state["max_used_percent"], float(virtual.percent)
        )
        state["max_process_rss_bytes"] = max(state["max_process_rss_bytes"], rss)
        if stop.wait(0.5):
            return


def _artifact_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise CandidateCortexTrainingError("segment_artifact_invalid")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise CandidateCortexTrainingError(
            f"stage_command_option_missing:{option}"
        ) from exc
    if index + 1 >= len(command):
        raise CandidateCortexTrainingError(f"stage_command_option_invalid:{option}")
    command[index + 1] = value


def _option_value(command: tuple[str, ...], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise CandidateCortexTrainingError(
            f"stage_command_option_missing:{option}"
        ) from exc
    if index + 1 >= len(command):
        raise CandidateCortexTrainingError(f"stage_command_option_invalid:{option}")
    return command[index + 1]


def _segment_command(
    plan: Mapping[str, Any],
    *,
    stage_index: int,
    segment: StageSegment,
    stage_resume_checkpoint: Mapping[str, Any] | None,
    actual_resume_checkpoint: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    command = list(
        build_stage_command(
            plan,
            stage_index=stage_index,
            resume_checkpoint=stage_resume_checkpoint,
        )
    )
    _replace_option(command, "--iters", str(segment.iterations))
    _replace_option(
        command,
        "--adapter-path",
        str(_segment_adapter_root(plan, stage_index, segment.index)),
    )
    _replace_option(command, "--save-every", str(segment.iterations))
    _replace_option(command, "--steps-per-eval", str(segment.iterations))
    if actual_resume_checkpoint is not None:
        path = Path(str(actual_resume_checkpoint.get("path"))).resolve(strict=True)
        if file_sha256(path) != actual_resume_checkpoint.get("sha256"):
            raise CandidateCortexTrainingError("segment_resume_checkpoint_drift")
        if "--resume-adapter-file" in command:
            _replace_option(command, "--resume-adapter-file", str(path))
        else:
            command.extend(("--resume-adapter-file", str(path)))
    return tuple(command)


def _iterate_batches_from_stage_offset(
    dataset,
    batch_size,
    max_seq_length,
    loop=False,
    seed=None,
    comm_group=None,
    *,
    start_iteration: int,
):
    """Match mlx-lm batching while resuming at an exact stage microstep."""

    import mlx.core as mx
    import numpy as np
    from mlx_lm.tuner.datasets import CacheDataset

    if isinstance(dataset, CacheDataset):
        def len_fn(index: int) -> int:
            return dataset.itemlen(index)
    else:
        def len_fn(index: int) -> int:
            return len(dataset[index][0])
    indices_by_length = sorted(range(len(dataset)), key=len_fn)
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} examples but "
            f"only has {len(dataset)}."
        )
    if comm_group is not None:
        offset = comm_group.rank()
        step = comm_group.size()
    else:
        offset = 0
        step = 1
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")
    batches = [
        indices_by_length[i + offset : i + offset + batch_size : step]
        for i in range(0, len(indices_by_length) - batch_size + 1, batch_size)
    ]
    rng = np.random.RandomState(seed)
    skipped = 0
    while True:
        permutation = rng.permutation(len(batches))
        for batch_position in permutation:
            if skipped < start_iteration:
                skipped += 1
                continue
            batch = [dataset[j] for j in batches[batch_position]]
            if len(batch[0]) == 2:
                batch, offsets = zip(*batch, strict=True)
            else:
                offsets = [0] * len(batch)
            lengths = [len(item) for item in batch]
            pad_to = 32
            padded_length = 1 + pad_to * (
                (max(lengths) + pad_to - 1) // pad_to
            )
            padded_length = min(padded_length, max_seq_length)
            batch_array = np.zeros(
                (batch_size // step, padded_length), np.int32
            )
            for item_index in range(batch_size // step):
                truncated = min(lengths[item_index], max_seq_length)
                batch_array[item_index, :truncated] = batch[item_index][:truncated]
                lengths[item_index] = truncated
            yield mx.array(batch_array), mx.array(
                list(zip(offsets, lengths, strict=True))
            )
        if not loop:
            break


def _save_optimizer_state(path: Path, optimizer: Any) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    values = {
        f"optimizer.{name}": value
        for name, value in tree_flatten(optimizer.state)
    }
    values.update(
        {f"mlx_random.{name}": value for name, value in tree_flatten(mx.random.state)}
    )
    mx.eval(*values.values())
    with local_internal_governed_scope(
        "candidate_cortex_adaptive.optimizer_state", domain="file_write"
    ):
        get_file_write_gateway().ensure_directory(
            path.parent,
            source="candidate_cortex_adaptive.optimizer_state",
        )
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.safetensors")
    mx.save_safetensors(str(temporary), values)
    with local_internal_governed_scope(
        "candidate_cortex_adaptive.optimizer_state", domain="file_write"
    ):
        gateway = get_file_write_gateway()
        if path.exists() or path.is_symlink():
            gateway.delete_file(
                temporary,
                source="candidate_cortex_adaptive.optimizer_state",
            )
            raise CandidateCortexTrainingError("optimizer_state_conflict")
        gateway.move_path(
            temporary,
            path,
            source="candidate_cortex_adaptive.optimizer_state",
        )


def _restore_optimizer_state(path: Path, optimizer: Any) -> None:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_unflatten

    mapped_values = mx.load(str(path.expanduser().resolve(strict=True)))
    if not isinstance(mapped_values, dict):
        raise CandidateCortexTrainingError("optimizer_state_invalid")
    values = {
        name: mx.array(np.array(value, copy=True))
        for name, value in mapped_values.items()
    }
    mx.eval(*values.values())
    del mapped_values
    optimizer_items = sorted(
        (
            name.removeprefix("optimizer."),
            value,
        )
        for name, value in values.items()
        if name.startswith("optimizer.")
    )
    random_items = sorted(
        (
            name.removeprefix("mlx_random."),
            value,
        )
        for name, value in values.items()
        if name.startswith("mlx_random.")
    )
    if not optimizer_items or not random_items:
        raise CandidateCortexTrainingError("optimizer_state_invalid")
    optimizer.state = tree_unflatten(optimizer_items)
    mx.random.state = tree_unflatten(random_items)


def _segment_batch_iterator(
    segment: StageSegment, *, data_seed: int
):
    """Return the exact deterministic batch lane for one stage segment."""

    def _iterator(*iterator_args, **iterator_kwargs):
        loop = bool(iterator_kwargs.get("loop", False))
        iterator_kwargs["seed"] = data_seed
        return _iterate_batches_from_stage_offset(
            *iterator_args,
            **iterator_kwargs,
            start_iteration=segment.start_iteration if loop else 0,
        )

    return _iterator


def _run_segment_training(
    command: tuple[str, ...],
    *,
    segment: StageSegment,
    prior_optimizer_state: Path | None,
    optimizer_output: Path,
) -> None:
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx_lm import lora
    from mlx_lm.tuner import trainer
    from mlx_lm.tuner.datasets import CacheDataset

    args = _mlx_arguments(command)
    data_seed = int(_option_value(command, "--seed"))
    np.random.seed(data_seed)
    training_callback = lora.get_reporting_callbacks(
        args.report_to,
        project_name=args.project_name,
        log_dir=args.adapter_path,
        config=vars(args),
    )

    print("Loading pretrained model", flush=True)
    model, tokenizer = lora.load(
        args.model,
        tokenizer_config={"trust_remote_code": True},
    )
    print("Loading datasets", flush=True)
    train_set, valid_set, _test_set = lora.load_dataset(args, tokenizer)
    print("Training", flush=True)

    mx.random.seed(data_seed)
    model.freeze()
    if args.num_layers > len(model.layers):
        raise CandidateCortexTrainingError("segment_num_layers_invalid")
    if args.fine_tune_type == "full":
        for layer in model.layers[-max(args.num_layers, 0) :]:
            layer.unfreeze()
        args.lora_parameters = None
    elif args.fine_tune_type in {"lora", "dora"}:
        lora.linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=args.fine_tune_type == "dora",
        )
    else:
        raise CandidateCortexTrainingError("segment_fine_tune_type_invalid")
    if args.resume_adapter_file is not None:
        print(
            f"Loading fine-tuned weights from {args.resume_adapter_file}",
            flush=True,
        )
        model.load_weights(args.resume_adapter_file, strict=False)
    lora.print_trainable_parameters(model)

    adapter_path = Path(args.adapter_path).expanduser().resolve(strict=False)
    with local_internal_governed_scope(
        "candidate_cortex_adaptive.segment_config", domain="file_write"
    ):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(
            adapter_path,
            source="candidate_cortex_adaptive.segment_config",
        )
        config = dict(vars(args))
        config.pop("_name_or_path", None)
        config.pop("vision_config", None)
        if "quantization" in config:
            config["quantization_config"] = config["quantization"]
        config_payload = json.dumps(dict(sorted(config.items())), indent=4).encode(
            "utf-8"
        )
        config_path = adapter_path / "adapter_config.json"
        created = gateway.write_bytes_if_absent(
            config_path,
            config_payload,
            mode=0o600,
            source="candidate_cortex_adaptive.segment_config",
        )
        if not created and config_path.read_bytes() != config_payload:
            raise CandidateCortexTrainingError("segment_adapter_config_conflict")

    training_args = trainer.TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_path / "adapters.safetensors",
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )
    learning_rate = (
        lora.build_schedule(args.lr_schedule)
        if args.lr_schedule
        else args.learning_rate
    )
    optimizer_classes = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "muon": optim.Muon,
        "sgd": optim.SGD,
        "adafactor": optim.Adafactor,
    }
    optimizer_name = args.optimizer.lower()
    optimizer_class = optimizer_classes.get(optimizer_name)
    if optimizer_class is None:
        raise CandidateCortexTrainingError("segment_optimizer_invalid")
    optimizer = optimizer_class(
        learning_rate=learning_rate,
        **args.optimizer_config.get(optimizer_name, {}),
    )
    if prior_optimizer_state is not None:
        _restore_optimizer_state(prior_optimizer_state, optimizer)
    trainer.train(
        model=model,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=training_args,
        iterate_batches=_segment_batch_iterator(segment, data_seed=data_seed),
        training_callback=training_callback,
    )
    _save_optimizer_state(optimizer_output, optimizer)


def _publish_checkpoint(
    plan: Mapping[str, Any],
    stage_index: int,
    command: tuple[str, ...],
    host_metrics: Mapping[str, Any],
    *,
    local_root: Path | None = None,
    local_iterations: int | None = None,
    segment_completions: tuple[Mapping[str, Any], ...] | None = None,
) -> dict[str, Any]:
    policy = StagePolicy(**dict(plan["stages"]))
    local_root = local_root or stage_adapter_root(plan, stage_index)
    local_iterations = local_iterations or policy.iterations(stage_index)
    local = discover_exact_checkpoint(
        local_root,
        expected_cumulative_iterations=local_iterations,
    )
    alias = local_root / "adapters.safetensors"
    if not alias.is_file() or file_sha256(alias) != local["sha256"]:
        raise CandidateCortexTrainingError("stage_final_adapter_mismatch")
    source_config = local_root / "adapter_config.json"
    config_payload = source_config.read_bytes()
    canonical_root = Path(str(plan["paths"]["adapter_root"])).resolve(strict=True)
    destination = (
        canonical_root
        / f"{policy.cumulative_iterations(stage_index):07d}_adapters.safetensors"
    )
    with local_internal_governed_scope(
        "candidate_cortex_adaptive.publish_stage", domain="file_write"
    ):
        gateway = get_file_write_gateway()
        if destination.exists() or destination.is_symlink():
            raise CandidateCortexTrainingError("stage_checkpoint_conflict")
        if segment_completions is None:
            gateway.move_path(
                Path(str(local["path"])),
                destination,
                source="candidate_cortex_adaptive.publish_stage",
            )
        else:
            gateway.copy_path(
                Path(str(local["path"])),
                destination,
                source="candidate_cortex_adaptive.publish_stage",
            )
        config_created = gateway.write_bytes_if_absent(
            canonical_root / "adapter_config.json",
            config_payload,
            mode=0o600,
            source="candidate_cortex_adaptive.publish_stage",
        )
        if (
            not config_created
            and (canonical_root / "adapter_config.json").read_bytes() != config_payload
        ):
            raise CandidateCortexTrainingError("stage_adapter_config_conflict")
        gateway.delete_file(
            alias,
            source="candidate_cortex_adaptive.publish_stage",
        )
    checkpoint = discover_exact_checkpoint(
        canonical_root,
        expected_cumulative_iterations=policy.cumulative_iterations(stage_index),
    )
    body: dict[str, Any] = {
        "schema": (
            SEGMENTED_STAGE_COMPLETION_SCHEMA
            if segment_completions is not None
            else STAGE_COMPLETION_SCHEMA
        ),
        "plan_sha256": plan["plan_sha256"],
        "stage_index": stage_index,
        "stage_iterations": policy.iterations(stage_index),
        "cumulative_iterations": policy.cumulative_iterations(stage_index),
        "training_command_sha256": document_sha256(list(command)),
        "checkpoint": checkpoint,
        "target_pid": os.getpid(),
        "target_start_token": detached._process_start_token(os.getpid()),  # noqa: SLF001
        "host_metrics": dict(host_metrics),
    }
    if segment_completions is not None:
        segment_values = [
            {
                "segment": dict(item["segment"]),
                "completion_sha256": str(item["completion_sha256"]),
                "command_sha256": str(item["command_sha256"]),
                "adapter_sha256": str(item["adapter"]["sha256"]),
                "optimizer_state_sha256": str(item["optimizer_state"]["sha256"]),
            }
            for item in segment_completions
        ]
        body["segments"] = segment_values
        body["training_execution_sha256"] = document_sha256(segment_values)
    completion = {**body, "completion_sha256": document_sha256(body)}
    _write_once(
        _completion_path(plan, stage_index),
        completion,
        source="candidate_cortex_adaptive.stage_completion",
    )
    return completion


def _publish_segment_completion(
    plan: Mapping[str, Any],
    stage_index: int,
    segment: StageSegment,
    command: tuple[str, ...],
    host_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    adapter_root = _segment_adapter_root(plan, stage_index, segment.index)
    checkpoint = discover_exact_checkpoint(
        adapter_root,
        expected_cumulative_iterations=segment.iterations,
    )
    alias = adapter_root / "adapters.safetensors"
    if not alias.is_file() or file_sha256(alias) != checkpoint["sha256"]:
        raise CandidateCortexTrainingError("segment_final_adapter_mismatch")
    optimizer = _artifact_binding(
        _segment_optimizer_path(plan, stage_index, segment.index)
    )
    body = {
        "schema": SEGMENT_COMPLETION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "stage_index": stage_index,
        "segment": asdict(segment),
        "command_sha256": document_sha256(list(command)),
        "adapter": checkpoint,
        "optimizer_state": optimizer,
        "host_metrics": dict(host_metrics),
    }
    completion = {**body, "completion_sha256": document_sha256(body)}
    _write_once(
        _segment_completion_path(plan, stage_index, segment.index),
        completion,
        source="candidate_cortex_adaptive.segment_completion",
    )
    return completion


def _reset_incomplete_segment(
    plan: Mapping[str, Any], stage_index: int, segment_index: int
) -> None:
    parent = _segment_parent(plan, stage_index, segment_index)
    with local_internal_governed_scope(
        "candidate_cortex_adaptive.reset_segment", domain="file_write"
    ):
        gateway = get_file_write_gateway()
        if parent.exists() or parent.is_symlink():
            gateway.delete_path(
                parent,
                recursive=True,
                source="candidate_cortex_adaptive.reset_segment",
            )
        gateway.ensure_directory(
            _segment_adapter_root(plan, stage_index, segment_index),
            source="candidate_cortex_adaptive.reset_segment",
        )


def _train_next_segment(
    plan: Mapping[str, Any],
    stage_index: int,
    stage_resume_checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Train one bounded segment; return stage completion only at the end."""

    existing = _validated_completion(plan, stage_index)
    if existing is not None:
        return existing
    segments = _stage_segments(plan, stage_index)
    completed: list[dict[str, Any]] = []
    for segment in segments:
        completion = _validated_segment_completion(plan, stage_index, segment)
        if completion is None:
            break
        completed.append(completion)
    if not completed:
        segment_root = _stage_parent(plan, stage_index) / "segments"
        if not segment_root.exists():
            _reset_incomplete_stage(plan, stage_index)
    segment = segments[len(completed)]
    _reset_incomplete_segment(plan, stage_index, segment.index)
    previous = completed[-1] if completed else None
    actual_resume = (
        previous["adapter"] if previous is not None else stage_resume_checkpoint
    )
    prior_optimizer = (
        Path(str(previous["optimizer_state"]["path"]))
        if previous is not None
        else None
    )
    command = _segment_command(
        plan,
        stage_index=stage_index,
        segment=segment,
        stage_resume_checkpoint=stage_resume_checkpoint,
        actual_resume_checkpoint=actual_resume,
    )
    state: dict[str, Any] = {
        "sample_count": 0,
        "min_available_bytes": 2**63 - 1,
        "max_used_percent": 0.0,
        "max_process_rss_bytes": 0,
    }
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_host, args=(stop, state), daemon=True)
    sampler.start()
    started = time.monotonic()
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        with standalone_model_lane(
            owner_id=f"candidate-cortex-stage:{plan['run_id']}:{stage_index}",
            model_path=str(plan["model"]["canonical_path"]),
            purpose="train",
            priority=100,
            preemptible=False,
            require_exclusive=True,
            allow_owner_eviction=True,
            metadata={
                "tool": "run_candidate_cortex_adaptive_target",
                "plan_sha256": plan["plan_sha256"],
                "stage_index": stage_index,
            },
        ):
            _run_segment_training(
                command,
                segment=segment,
                prior_optimizer_state=prior_optimizer,
                optimizer_output=_segment_optimizer_path(
                    plan, stage_index, segment.index
                ),
            )
    finally:
        stop.set()
        sampler.join(timeout=5.0)
        _release_model_memory()
    state["duration_seconds"] = max(0.0, time.monotonic() - started)
    segment_completion = _publish_segment_completion(
        plan,
        stage_index,
        segment,
        command,
        state,
    )
    completed.append(segment_completion)
    if len(completed) < len(segments):
        return None
    aggregate = {
        "sample_count": sum(
            int(item["host_metrics"]["sample_count"]) for item in completed
        ),
        "min_available_bytes": min(
            int(item["host_metrics"]["min_available_bytes"])
            for item in completed
        ),
        "max_used_percent": max(
            float(item["host_metrics"]["max_used_percent"])
            for item in completed
        ),
        "max_process_rss_bytes": max(
            int(item["host_metrics"]["max_process_rss_bytes"])
            for item in completed
        ),
        "duration_seconds": sum(
            float(item["host_metrics"]["duration_seconds"])
            for item in completed
        ),
        "segment_count": len(completed),
        "segment_completion_sha256": [
            str(item["completion_sha256"]) for item in completed
        ],
    }
    return _publish_checkpoint(
        plan,
        stage_index,
        build_stage_command(
            plan,
            stage_index=stage_index,
            resume_checkpoint=stage_resume_checkpoint,
        ),
        aggregate,
        local_root=_segment_adapter_root(plan, stage_index, segment.index),
        local_iterations=segment.iterations,
        segment_completions=tuple(completed),
    )


def _release_model_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass


def _restart_for_clean_model_phase(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    journal_key: Path,
    journal: Path,
    key: bytes,
    stage_index: int,
    next_phase: str,
    execution_id: str,
) -> None:
    """Replace the process image so one model-heavy phase cannot retain another.

    MLX cache clearing only releases allocations no Python object still owns.
    Training and checkpoint measurement intentionally use different process
    images, while ``execve`` preserves the detached supervisor's target PID and
    its trainer-bound sleep inhibitor.
    """

    if next_phase not in {"train", "measure", "decide"}:
        raise CandidateCortexTrainingError("adaptive_phase_invalid")
    launcher = Path(str(plan["python"])).expanduser()
    if not launcher.is_absolute() or not launcher.exists():
        raise CandidateCortexTrainingError("adaptive_phase_launcher_invalid")
    script = Path(__file__).resolve(strict=True)
    append_authenticated_event(
        journal,
        key=key,
        event_type="phase_restart_requested",
        payload={
            "schema": PHASE_BOUNDARY_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "stage_index": stage_index,
            "next_phase": next_phase,
            "target_pid": os.getpid(),
            "target_start_token": detached._process_start_token(os.getpid()),  # noqa: SLF001
        },
    )
    environment = dict(os.environ)
    environment["AURA_CANDIDATE_CORTEX_PHASE"] = next_phase
    environment["AURA_CANDIDATE_CORTEX_STAGE"] = str(stage_index)
    argv = [
        str(launcher),
        str(script),
        "--run-root",
        str(run_root.expanduser().resolve(strict=True)),
        "--journal-key",
        str(journal_key.expanduser().resolve(strict=True)),
        "--execution-id",
        execution_id,
    ]
    os.execve(str(launcher), argv, environment)
    raise CandidateCortexTrainingError("adaptive_phase_exec_returned")


def _measure_stage(
    plan: Mapping[str, Any], stage_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_path = _evidence_path(plan, stage_index)
    detail_path = _detail_path(plan, stage_index)
    if not evidence_path.is_file() or not detail_path.is_file():
        code = measurement.main(
            [
                "--run-root",
                str(plan["paths"]["run_root"]),
                "--stage-index",
                str(stage_index),
                "--evidence-output",
                str(evidence_path),
                "--detail-output",
                str(detail_path),
            ]
        )
        if code != 0:
            raise CandidateCortexTrainingError("stage_measurement_failed")
    evidence = _strict_document(evidence_path)
    detail = _strict_document(detail_path)
    admission = adjudicate_checkpoint_evidence(
        evidence,
        plan=plan,
        stage_index=stage_index,
    )
    if (
        detail.get("evidence_sha256") != evidence.get("measurement_sha256")
        or detail.get("stage_index") != stage_index
    ):
        raise CandidateCortexTrainingError("stage_measurement_detail_invalid")
    return evidence, admission


def _observation(
    plan: Mapping[str, Any],
    stage_index: int,
    completion: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _strict_document(Path(str(plan["paths"]["adapter_identity"])))
    eval_samples = (
        int(evidence["persona"]["samples"])
        + int(evidence["retention"]["samples"])
        + len(evidence["behavior"])
    )
    raw = {
        "schema": OBSERVATION_SCHEMA,
        "stage_index": stage_index,
        "cumulative_iterations": completion["cumulative_iterations"],
        "validation_loss": float(evidence["persona"]["candidate_loss"]),
        "eval_samples": eval_samples,
        "checkpoint": completion["checkpoint"],
        "adapter_identity_sha256": document_sha256(identity),
        "model_descriptor_sha256": plan["model"]["descriptor_sha256"],
        "dataset_receipt_sha256": plan["dataset"]["receipt_sha256"],
        "trainer_pid": completion["target_pid"],
        "trainer_start_token": completion["target_start_token"],
        "trainer_exit_code": 0,
    }
    return validate_stage_observation(
        raw,
        plan=plan,
        expected_stage_index=stage_index,
        launched_identity={
            "trainer_pid": completion["target_pid"],
            "trainer_start_token": completion["target_start_token"],
        },
    )


def _stage_events(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return effective_stage_evidence(events)


def _reconcile_partial_admission(
    plan: Mapping[str, Any], journal: Path, key: bytes
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = read_authenticated_journal(journal, key=key)
    observations, admissions = _stage_events(events)
    if len(observations) == len(admissions):
        return observations, admissions
    if len(observations) != len(admissions) + 1:
        raise CandidateCortexTrainingError("stage_journal_incomplete")
    stage_index = len(admissions)
    evidence = _strict_document(_evidence_path(plan, stage_index))
    admission = adjudicate_checkpoint_evidence(
        evidence,
        plan=plan,
        stage_index=stage_index,
    )
    if admission.get("schema") != ADMISSION_SCHEMA:
        raise CandidateCortexTrainingError("stage_admission_invalid")
    append_authenticated_event(
        journal,
        key=key,
        event_type="stage_admitted",
        payload=admission,
    )
    return observations, [*admissions, admission]


def _write_result(
    plan: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    execution_id: str,
) -> None:
    body = {
        "schema": ADAPTIVE_RESULT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "decision": dict(decision),
    }
    result = {**body, "result_sha256": document_sha256(body)}
    result_path = adaptive_result_path(plan, execution_id=execution_id)
    if execution_id != "primary":
        with local_internal_governed_scope(
            "candidate_cortex_adaptive.result_generation", domain="file_write"
        ):
            get_file_write_gateway().ensure_directory(
                result_path.parent,
                source="candidate_cortex_adaptive.result_generation",
            )
    _write_once(
        result_path,
        result,
        source="candidate_cortex_adaptive.result",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--journal-key", type=Path, required=True)
    parser.add_argument("--execution-id", default="primary")
    args = parser.parse_args(argv)
    plan = load_and_verify_plan(args.run_root, verify_full_model=True)
    key = _key(args.journal_key)
    journal = Path(str(plan["paths"]["run_root"])) / JOURNAL_FILE
    events = read_authenticated_journal(journal, key=key)
    admission = execution_admission(
        plan,
        execute=True,
        authenticated_events=events,
    )
    if admission.get("execution_authorized") is not True:
        raise CandidateCortexTrainingError("adaptive_execution_not_authorized")

    while True:
        observations, admissions = _reconcile_partial_admission(plan, journal, key)
        next_stage = next_stage_plan(
            plan,
            observations=observations,
            admissions=admissions,
        )
        if next_stage.get("decision") != "CONTINUE":
            _write_result(plan, next_stage, execution_id=args.execution_id)
            print(json.dumps(next_stage, indent=2, sort_keys=True), flush=True)
            return 0
        stage_index = int(next_stage["stage_index"])
        current_plan = load_and_verify_plan(args.run_root, verify_full_model=True)
        if current_plan != plan:
            raise CandidateCortexTrainingError("adaptive_plan_or_input_drift")
        completion = _validated_completion(plan, stage_index)
        trained_in_this_process = completion is None
        if completion is None:
            append_authenticated_event(
                journal,
                key=key,
                event_type="stage_started",
                payload={
                    "stage_index": stage_index,
                    "target_pid": os.getpid(),
                    "target_start_token": detached._process_start_token(os.getpid()),  # noqa: SLF001
                    "command_sha256": document_sha256(next_stage["command"]),
                },
            )
            completion = _train_next_segment(
                plan,
                stage_index,
                next_stage.get("resume_checkpoint"),
            )
            if completion is None:
                _restart_for_clean_model_phase(
                    plan,
                    run_root=args.run_root,
                    journal_key=args.journal_key,
                    journal=journal,
                    key=key,
                    stage_index=stage_index,
                    next_phase="train",
                    execution_id=args.execution_id,
                )
        if trained_in_this_process and (
            not _evidence_path(plan, stage_index).is_file()
            or not _detail_path(plan, stage_index).is_file()
        ):
            _restart_for_clean_model_phase(
                plan,
                run_root=args.run_root,
                journal_key=args.journal_key,
                journal=journal,
                key=key,
                stage_index=stage_index,
                next_phase="measure",
                execution_id=args.execution_id,
            )
        evidence, stage_admission = _measure_stage(plan, stage_index)
        observation = _observation(
            plan,
            stage_index,
            completion,
            evidence,
        )
        append_authenticated_event(
            journal,
            key=key,
            event_type="stage_observed",
            payload=observation,
        )
        append_authenticated_event(
            journal,
            key=key,
            event_type="stage_admitted",
            payload=stage_admission,
        )
        _restart_for_clean_model_phase(
            plan,
            run_root=args.run_root,
            journal_key=args.journal_key,
            journal=journal,
            key=key,
            stage_index=stage_index,
            next_phase="decide",
            execution_id=args.execution_id,
        )


if __name__ == "__main__":
    raise SystemExit(main())
