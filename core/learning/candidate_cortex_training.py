"""Candidate-bound, staged LoRA training control without model imports.

This module owns immutable plan, canary, checkpoint, and stage evidence.  It
deliberately does not import MLX or start a process; the CLI adapter hands its
exact commands to Aura's detached model-lane supervisor.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from core.brain.llm.model_artifact_profile import validate_model_artifact_descriptor
from core.governance_context import local_internal_governed_scope
from core.learning.candidate_cortex_kernel import (
    RECEIPT_SCHEMA as KERNEL_RECEIPT_SCHEMA,
)
from core.learning.candidate_cortex_kernel import (
    CandidateCortexKernelError,
    verify_candidate_cortex_kernel,
)
from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.file_write_gateway import get_file_write_gateway

PLAN_SCHEMA_V2: Final = "aura.candidate_cortex_training.plan.v2"
PLAN_SCHEMA: Final = "aura.candidate_cortex_training.plan.v3"
DATASET_SCHEMA: Final = KERNEL_RECEIPT_SCHEMA
JOURNAL_EVENT_SCHEMA: Final = "aura.candidate_cortex_training.journal_event.v1"
OBSERVATION_SCHEMA: Final = "aura.candidate_cortex_training.stage_observation.v1"
ADMISSION_SCHEMA: Final = "aura.candidate_cortex_training.admission.v1"
ADAPTER_IDENTITY_SCHEMA: Final = "aura.candidate_cortex_training.adapter_identity.v1"
SUPERVISION_SCHEMA: Final = "aura.candidate_cortex_training.supervision.v1"
CANARY_OBSERVATION_SCHEMA: Final = "aura.candidate_cortex_training.canary_observation.v1"
CANARY_ADMISSION_SCHEMA: Final = "aura.candidate_cortex_training.canary_admission.v1"
CANARY_HOST_METRICS_SCHEMA: Final = "aura.candidate_cortex_training.host_metrics.v1"
ADAPTIVE_RESULT_SCHEMA: Final = "aura.candidate_cortex_training.adaptive_result.v1"
STAGE_RECONCILIATION_SCHEMA: Final = (
    "aura.candidate_cortex_training.stage_reconciliation.v1"
)
PLAN_FILE: Final = "training_plan.json"
CONFIG_FILE: Final = "mlx_lora_config.json"
IDENTITY_FILE: Final = "adapter_identity.json"
JOURNAL_FILE: Final = "training_journal.jsonl"
LOCK_FILE: Final = ".training.lock"
MAX_DOCUMENT_BYTES: Final = 16 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT = re.compile(r"(?P<step>[0-9]+)_adapters\.safetensors")
_EXECUTION_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}")
_TRAIN_REPORT = re.compile(
    r"^Iter (?P<iteration>[0-9]+): Train loss (?P<loss>[0-9.eE+-]+), "
    r"Learning Rate (?P<learning_rate>[0-9.eE+-]+), "
    r"It/sec (?P<iterations_per_second>[0-9.eE+-]+), "
    r"Tokens/sec (?P<tokens_per_second>[0-9.eE+-]+), "
    r"Trained Tokens (?P<trained_tokens>[0-9.eE+-]+), "
    r"Peak mem (?P<peak_memory_gb>[0-9.eE+-]+) GB$"
)
_VAL_REPORT = re.compile(
    r"^Iter (?P<iteration>[0-9]+): Val loss (?P<loss>[0-9.eE+-]+), "
    r"Val took (?P<duration_seconds>[0-9.eE+-]+)s$"
)


class CandidateCortexTrainingError(ValueError):
    """A stable training-control contract failure."""


def _fail(code: str) -> None:
    raise CandidateCortexTrainingError(code)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise CandidateCortexTrainingError("canonical_json_invalid") from exc


def document_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _ensure_directory(path: Path) -> Path:
    with local_internal_governed_scope(
        "candidate_cortex_training.ensure_directory", domain="file_write"
    ):
        created = get_file_write_gateway().ensure_directory(
            path,
            source="candidate_cortex_training.ensure_directory",
        )
    return Path(created)


def _write_bytes_if_absent(path: Path, payload: bytes) -> bool:
    with local_internal_governed_scope(
        "candidate_cortex_training.write_once", domain="file_write"
    ):
        return get_file_write_gateway().write_bytes_if_absent(
            path,
            payload,
            mode=0o600,
            source="candidate_cortex_training.write_once",
        )


def _write_bytes(path: Path, payload: bytes) -> None:
    with local_internal_governed_scope(
        "candidate_cortex_training.write", domain="file_write"
    ):
        get_file_write_gateway().write_bytes(
            path,
            payload,
            source="candidate_cortex_training.write",
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CandidateCortexTrainingError("bound_file_unreadable") from exc
    return digest.hexdigest()


def adaptive_result_path(
    plan: Mapping[str, Any], *, execution_id: str = "primary"
) -> Path:
    """Immutable terminal-result location for one detached execution."""

    if not isinstance(execution_id, str) or _EXECUTION_ID.fullmatch(execution_id) is None:
        _fail("adaptive_execution_id_invalid")
    root = Path(str(plan["paths"]["run_root"])).expanduser().resolve(strict=True)
    if execution_id == "primary":
        return root / "adaptive_result.json"
    return root / "adaptive-results" / f"{execution_id}.json"


def _launcher_binding(path: Path) -> dict[str, Any]:
    """Bind an executable without erasing virtual-environment selection.

    Python discovers ``pyvenv.cfg`` from the launcher path. Resolving a venv's
    ``bin/python`` symlink before execution therefore changes the environment,
    even when the target binary bytes are identical. Keep the invocation path
    and independently freeze the target and environment marker.
    """

    launcher = path.expanduser().absolute()
    try:
        launcher_stat = launcher.lstat()
        resolved = launcher.resolve(strict=True)
    except OSError as exc:
        raise CandidateCortexTrainingError("python_executable_unavailable") from exc
    if not (stat.S_ISREG(launcher_stat.st_mode) or stat.S_ISLNK(launcher_stat.st_mode)):
        _fail("python_executable_invalid")
    if not resolved.is_file() or not os.access(launcher, os.X_OK):
        _fail("python_executable_invalid")
    pyvenv_path = launcher.parent.parent / "pyvenv.cfg"
    pyvenv: dict[str, Any] | None = None
    if pyvenv_path.exists() or pyvenv_path.is_symlink():
        if pyvenv_path.is_symlink() or not pyvenv_path.is_file():
            _fail("python_environment_invalid")
        pyvenv = {
            "path": str(pyvenv_path),
            "sha256": file_sha256(pyvenv_path),
            "size_bytes": pyvenv_path.stat().st_size,
        }
    body = {
        "invocation_path": str(launcher),
        "invocation_kind": (
            "symlink" if stat.S_ISLNK(launcher_stat.st_mode) else "file"
        ),
        "invocation_mode": stat.S_IMODE(launcher_stat.st_mode),
        "symlink_target": (
            os.readlink(launcher) if stat.S_ISLNK(launcher_stat.st_mode) else None
        ),
        "resolved_path": str(resolved),
        "resolved_sha256": file_sha256(resolved),
        "pyvenv": pyvenv,
    }
    return {**body, "binding_sha256": document_sha256(body)}


def _verify_launcher_binding(raw: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        _fail("python_executable_binding_missing")
    current = _launcher_binding(path)
    if dict(raw) != current:
        _fail("python_executable_binding_drift")
    return current


def _strict_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file() or resolved.is_symlink():
            _fail(f"{role}_not_regular")
        raw = resolved.read_bytes()
    except OSError as exc:
        raise CandidateCortexTrainingError(f"{role}_unreadable") from exc
    if not raw or len(raw) > MAX_DOCUMENT_BYTES:
        _fail(f"{role}_size_invalid")

    def reject_constant(_value: str) -> None:
        _fail(f"{role}_number_invalid")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except CandidateCortexTrainingError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CandidateCortexTrainingError(f"{role}_json_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_schema_invalid")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _line_count(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError as exc:
        raise CandidateCortexTrainingError("dataset_unreadable") from exc


def _file_binding(path: Path) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise CandidateCortexTrainingError("bound_file_unreadable") from exc
    if not stat.S_ISREG(info.st_mode) or resolved.is_symlink():
        _fail("bound_file_not_regular")
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": info.st_size,
        "lines": _line_count(resolved),
    }


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class TrainingConfig:
    """All model-facing settings that contribute to run identity."""

    rank: int = 32
    scale: float = 20.0
    dropout: float = 0.0
    num_layers: int = -1
    targets: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096
    learning_rate: float = 1e-5
    save_every: int = 25
    eval_every: int = 25
    report_every: int = 5
    val_batches: int = -1
    seed: int = 20260823

    def validate(self) -> None:
        if self.rank <= 0 or self.rank > 1024:
            _fail("rank_invalid")
        if not math.isfinite(self.scale) or self.scale <= 0:
            _fail("scale_invalid")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            _fail("dropout_invalid")
        if self.num_layers == 0 or self.num_layers < -1:
            _fail("num_layers_invalid")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            _fail("targets_invalid")
        if any(not re.fullmatch(r"[A-Za-z0-9_.]+", value) for value in self.targets):
            _fail("target_invalid")
        positive = (
            self.batch_size,
            self.gradient_accumulation_steps,
            self.max_seq_length,
            self.save_every,
            self.eval_every,
            self.report_every,
        )
        if any(value <= 0 for value in positive) or self.seed < 0:
            _fail("training_integer_invalid")
        if self.val_batches == 0 or self.val_batches < -1:
            _fail("val_batches_invalid")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            _fail("learning_rate_invalid")


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer-state policy bound into the training identity.

    Adafactor is deliberately fixed-rate here. Its factored second moment
    preserves per-matrix adaptation without Adam's two full-size moment
    tensors, which dominate host memory for a broad LoRA surface.
    """

    name: str = "adafactor"
    relative_step: bool = False
    scale_parameter: bool = False
    beta_1: float | None = None

    def validate(self) -> None:
        if self.name not in {"adam", "adafactor"}:
            _fail("optimizer_invalid")
        if self.name == "adam" and (
            self.relative_step is not False
            or self.scale_parameter is not False
            or self.beta_1 is not None
        ):
            _fail("adam_optimizer_config_invalid")
        if self.beta_1 is not None and (
            not math.isfinite(self.beta_1) or not 0 <= self.beta_1 < 1
        ):
            _fail("optimizer_beta_1_invalid")


@dataclass(frozen=True)
class StagePolicy:
    """Bounded geometric schedule and evidence thresholds."""

    initial_iterations: int = 100
    growth_factor: int = 2
    max_stages: int = 5
    min_stages: int = 2
    patience: int = 2
    min_loss_improvement: float = 0.002
    max_loss_regression_fraction: float = 0.02
    persona_floor: float = 0.90
    retention_floor: float = 0.98
    no_regression_floor: float = 1.0
    min_eval_samples: int = 32

    def validate(self) -> None:
        if self.initial_iterations <= 0 or self.growth_factor < 2:
            _fail("stage_schedule_invalid")
        if not 1 <= self.min_stages <= self.max_stages <= 16:
            _fail("stage_count_invalid")
        if not 1 <= self.patience <= self.max_stages:
            _fail("patience_invalid")
        finite = (
            self.min_loss_improvement,
            self.max_loss_regression_fraction,
            self.persona_floor,
            self.retention_floor,
            self.no_regression_floor,
        )
        if any(not math.isfinite(value) for value in finite):
            _fail("stage_threshold_non_finite")
        if self.min_loss_improvement < 0 or self.max_loss_regression_fraction < 0:
            _fail("stage_threshold_invalid")
        if (
            any(not 0 <= value <= 1 for value in finite[2:])
            or self.min_eval_samples <= 0
        ):
            _fail("admission_threshold_invalid")

    def iterations(self, stage_index: int) -> int:
        if not 0 <= stage_index < self.max_stages:
            _fail("stage_index_invalid")
        return self.initial_iterations * (self.growth_factor**stage_index)

    def cumulative_iterations(self, stage_index: int) -> int:
        return sum(self.iterations(index) for index in range(stage_index + 1))


@dataclass(frozen=True)
class CanaryPolicy:
    """A production-shaped resource and checkpoint gate before real training."""

    optimizer_steps: int = 10
    validation_batches: int = 4
    validation_interval_optimizer_steps: int = 5
    timeout_seconds: int = 7200
    min_host_available_gb: float = 4.0
    max_peak_mlx_gb: float = 40.0
    max_validation_loss_ratio: float = 1.5

    def validate(self) -> None:
        positive = (
            self.optimizer_steps,
            self.validation_batches,
            self.validation_interval_optimizer_steps,
            self.timeout_seconds,
        )
        if any(value <= 0 for value in positive):
            _fail("canary_integer_invalid")
        if self.validation_interval_optimizer_steps > self.optimizer_steps:
            _fail("canary_validation_interval_invalid")
        finite = (
            self.min_host_available_gb,
            self.max_peak_mlx_gb,
            self.max_validation_loss_ratio,
        )
        if any(not math.isfinite(value) or value <= 0 for value in finite):
            _fail("canary_threshold_invalid")


def validate_candidate_descriptor(
    descriptor_path: Path,
    *,
    expected_descriptor_sha256: str,
    verify_full_model: bool = True,
) -> tuple[dict[str, Any], Path]:
    descriptor = _strict_json(descriptor_path, role="candidate_descriptor")
    if descriptor.get("descriptor_sha256") != expected_descriptor_sha256:
        _fail("candidate_descriptor_not_admitted")
    try:
        model_root = Path(str(descriptor.get("canonical_path"))).expanduser().resolve(
            strict=True
        )
        validate_model_artifact_descriptor(
            descriptor,
            model_path=model_root,
            verify_full_hash=verify_full_model,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise CandidateCortexTrainingError(f"candidate_descriptor_invalid:{exc}") from exc
    if not model_root.is_dir() or model_root.is_symlink():
        _fail("candidate_model_root_invalid")
    return descriptor, model_root


def validate_compact_kernel_receipt(
    receipt_path: Path,
    *,
    expected_descriptor_sha256: str,
) -> dict[str, Any]:
    try:
        receipt = verify_candidate_cortex_kernel(
            receipt_path,
            expected_descriptor_sha256=expected_descriptor_sha256,
        )
        generation_root = Path(str(receipt["generation_root"])).resolve(strict=True)
        data_root = (generation_root / "data").resolve(strict=True)
    except CandidateCortexKernelError as exc:
        raise CandidateCortexTrainingError(f"compact_kernel_invalid:{exc}") from exc
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise CandidateCortexTrainingError("compact_kernel_root_unavailable") from exc
    if not data_root.is_dir() or data_root.is_symlink() or data_root.parent != generation_root:
        _fail("compact_kernel_root_invalid")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        _fail("compact_kernel_outputs_invalid")
    normalized_outputs: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        binding = outputs.get(split)
        if not isinstance(binding, dict) or set(binding) != {
            "path",
            "sha256",
            "size_bytes",
            "lines",
        }:
            _fail(f"compact_kernel_{split}_binding_invalid")
        relative = Path(str(binding.get("path")))
        expected_relative = Path("data") / f"{split}.jsonl"
        if relative != expected_relative:
            _fail(f"compact_kernel_{split}_path_invalid")
        path = (generation_root / relative).resolve(strict=True)
        if not _within(path, data_root) or path.parent != data_root:
            _fail(f"compact_kernel_{split}_path_escape")
        observed = _file_binding(path)
        observed.pop("path")
        if observed != {key: binding[key] for key in observed}:
            _fail(f"compact_kernel_{split}_drift")
        if observed["lines"] <= 0:
            _fail(f"compact_kernel_{split}_empty")
        normalized_outputs[split] = {"path": path.name, **observed}
    return {
        **receipt,
        "data_root": str(data_root),
        "outputs": normalized_outputs,
    }


def _training_config_document(
    config: TrainingConfig,
    optimizer: OptimizerConfig | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "lora_parameters": {
            "rank": config.rank,
            "scale": config.scale,
            "dropout": config.dropout,
            "keys": list(config.targets),
        }
    }
    if optimizer is not None:
        optimizer.validate()
        parameters: dict[str, Any] = {}
        if optimizer.name == "adafactor":
            parameters = {
                "relative_step": optimizer.relative_step,
                "scale_parameter": optimizer.scale_parameter,
                "beta_1": optimizer.beta_1,
            }
        document.update(
            {
                "optimizer": optimizer.name,
                "optimizer_config": {optimizer.name: parameters},
            }
        )
    return document


def _admission_command_identity(command: Sequence[str]) -> dict[str, Any]:
    if not command or any(not isinstance(value, str) or not value for value in command):
        _fail("admission_command_invalid")
    executable = Path(command[0]).expanduser()
    if executable.is_absolute():
        executable_binding: dict[str, Any] = _launcher_binding(executable)
    else:
        executable_binding = {"invocation_path": command[0], "binding_sha256": None}
    inputs: list[dict[str, Any]] = []
    for value in command[1:]:
        candidate = Path(value).expanduser()
        if candidate.is_absolute() and (candidate.exists() or candidate.is_symlink()):
            inputs.append(_file_binding(candidate))
        elif candidate.suffix.lower() in {".py", ".sh"}:
            _fail("admission_source_unavailable")
    return {
        "argv": list(command),
        "executable": executable_binding,
        "inputs": inputs,
    }


def prepare_training_run(
    *,
    descriptor_path: Path,
    expected_descriptor_sha256: str,
    dataset_receipt_path: Path,
    output_root: Path,
    python_executable: Path,
    admission_command: Sequence[str],
    config: TrainingConfig | None = None,
    optimizer: OptimizerConfig | None = None,
    policy: StagePolicy | None = None,
    canary_policy: CanaryPolicy | None = None,
    verify_full_model: bool = True,
) -> dict[str, Any]:
    """Validate immutable inputs and publish one content-addressed plan."""

    config = config or TrainingConfig()
    optimizer = optimizer or OptimizerConfig()
    policy = policy or StagePolicy()
    canary_policy = canary_policy or CanaryPolicy()
    config.validate()
    optimizer.validate()
    policy.validate()
    canary_policy.validate()
    descriptor, model_root = validate_candidate_descriptor(
        descriptor_path,
        expected_descriptor_sha256=expected_descriptor_sha256,
        verify_full_model=verify_full_model,
    )
    dataset = validate_compact_kernel_receipt(
        dataset_receipt_path,
        expected_descriptor_sha256=expected_descriptor_sha256,
    )
    python_binding = _launcher_binding(python_executable)
    python = Path(str(python_binding["invocation_path"]))
    admission = _admission_command_identity(admission_command)
    identity_material = {
        "model_descriptor_sha256": expected_descriptor_sha256,
        "dataset_receipt_sha256": dataset["receipt_sha256"],
        "training": asdict(config),
        "optimizer": asdict(optimizer),
        "stages": asdict(policy),
        "canary": asdict(canary_policy),
        "python": python_binding,
        "admission": admission,
    }
    run_id = document_sha256(identity_material)[:24]
    root = output_root.expanduser().resolve(strict=False)
    run_root = (root / expected_descriptor_sha256[:16] / run_id).resolve(strict=False)
    if not _within(run_root, root):
        _fail("training_run_path_escape")
    _ensure_directory(root)
    _ensure_directory(run_root.parent)
    _ensure_directory(run_root)
    data_root = Path(str(dataset["data_root"])).resolve(strict=True)
    canary_execution_root = run_root / "canary-execution"
    paths = {
        "run_root": str(run_root),
        "data_root": str(data_root),
        "adapter_root": str(run_root / "adapter"),
        "checkpoint_root": str(run_root / "adapter"),
        "journal": str(run_root / JOURNAL_FILE),
        "adapter_identity": str(run_root / IDENTITY_FILE),
        "mlx_config": str(run_root / CONFIG_FILE),
        "canary_execution_root": str(canary_execution_root),
        "canary_adapter_root": str(canary_execution_root / "adapter"),
        "canary_detached_root": str(canary_execution_root / "detached"),
        "canary_host_metrics": str(canary_execution_root / "host_metrics.json"),
        "canary_observation": str(canary_execution_root / "canary_observation.json"),
        "canary_admission": str(canary_execution_root / "canary_admission.json"),
    }
    plan_material: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "run_id": run_id,
        "model": {
            "descriptor_path": str(descriptor_path.expanduser().resolve(strict=True)),
            "descriptor_sha256": expected_descriptor_sha256,
            "canonical_path": str(model_root),
            "repository_id": descriptor.get("repository_id"),
            "revision": descriptor.get("revision"),
        },
        "dataset": {
            "receipt_path": str(dataset_receipt_path.expanduser().resolve(strict=True)),
            "receipt_sha256": dataset["receipt_sha256"],
            "data_root": str(data_root),
            "outputs": dataset["outputs"],
        },
        "training": asdict(config),
        "optimizer": asdict(optimizer),
        "stages": asdict(policy),
        "canary": asdict(canary_policy),
        "python": str(python),
        "python_binding": python_binding,
        "admission": admission,
        "paths": paths,
        "supervision": {
            "schema": SUPERVISION_SCHEMA,
            "required": True,
            "model_lane": "standalone_exclusive",
            "subprocess_gateway_source": "training_tooling:candidate_cortex_training",
            "accelerator_capability": "model",
            "detached_supervisor": "tools/run_detached_step.py",
            "sleep_inhibitor": "/usr/bin/caffeinate -i -w <trainer-pid>",
            "execution_adapter_available": True,
            "execution_adapter": "tools/run_candidate_cortex_training.py:launch-canary",
        },
    }
    plan_material["plan_sha256"] = document_sha256(plan_material)
    adapter_identity = {
        "schema": ADAPTER_IDENTITY_SCHEMA,
        "run_id": run_id,
        "model_descriptor_sha256": expected_descriptor_sha256,
        "dataset_receipt_sha256": dataset["receipt_sha256"],
        "training_identity_sha256": document_sha256(asdict(config)),
        "optimizer_identity_sha256": document_sha256(asdict(optimizer)),
        "stage_policy_sha256": document_sha256(asdict(policy)),
        "canary_policy_sha256": document_sha256(asdict(canary_policy)),
    }
    mlx_config = _training_config_document(config, optimizer)
    lock_path = run_root.parent / LOCK_FILE
    with interprocess_file_lock(lock_path):
        for path, payload, conflict in (
            (run_root / PLAN_FILE, plan_material, "training_plan_conflict"),
            (run_root / CONFIG_FILE, mlx_config, "mlx_config_conflict"),
            (run_root / IDENTITY_FILE, adapter_identity, "adapter_identity_conflict"),
        ):
            encoded = canonical_json_bytes(payload) + b"\n"
            if not _write_bytes_if_absent(path, encoded):
                if path.read_bytes() != encoded:
                    _fail(conflict)
        _ensure_directory(run_root / "adapter")
        _ensure_directory(canary_execution_root)
        _ensure_directory(canary_execution_root / "adapter")
    return plan_material


def load_and_verify_plan(
    run_root: Path,
    *,
    verify_full_model: bool = True,
) -> dict[str, Any]:
    root = run_root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink() or stat.S_IMODE(root.stat().st_mode) & 0o077:
        _fail("training_run_custody_invalid")
    plan = _strict_json(root / PLAN_FILE, role="training_plan")
    material = dict(plan)
    claimed = material.pop("plan_sha256", None)
    schema = plan.get("schema")
    if schema not in {PLAN_SCHEMA_V2, PLAN_SCHEMA} or not _is_sha256(claimed):
        _fail("training_plan_schema_invalid")
    if claimed != document_sha256(material):
        _fail("training_plan_digest_invalid")
    if plan.get("run_id") != root.name:
        _fail("training_plan_root_mismatch")
    model = plan.get("model")
    dataset = plan.get("dataset")
    paths = plan.get("paths")
    if not isinstance(model, dict) or not isinstance(dataset, dict) or not isinstance(paths, dict):
        _fail("training_plan_bindings_invalid")
    if paths.get("run_root") != str(root):
        _fail("training_plan_path_drift")
    for value in paths.values():
        if not _within(Path(str(value)), root) and value != paths.get("data_root"):
            _fail("training_plan_path_escape")
    _verify_launcher_binding(
        plan.get("python_binding"),
        Path(str(plan.get("python") or "")),
    )
    admission = plan.get("admission")
    if not isinstance(admission, Mapping):
        _fail("admission_command_invalid")
    admission_argv = admission.get("argv")
    admission_inputs = admission.get("inputs")
    if (
        not isinstance(admission_argv, list)
        or not admission_argv
        or not isinstance(admission_inputs, list)
    ):
        _fail("admission_command_invalid")
    admission_executable = Path(str(admission_argv[0])).expanduser()
    if admission_executable.is_absolute():
        _verify_launcher_binding(admission.get("executable"), admission_executable)
    current_inputs = []
    for value in admission_argv[1:]:
        candidate = Path(str(value)).expanduser()
        if candidate.is_absolute() and (candidate.exists() or candidate.is_symlink()):
            current_inputs.append(_file_binding(candidate))
        elif candidate.suffix.lower() in {".py", ".sh"}:
            _fail("admission_source_unavailable")
    if current_inputs != admission_inputs:
        _fail("admission_source_drift")
    validate_candidate_descriptor(
        Path(str(model.get("descriptor_path"))),
        expected_descriptor_sha256=str(model.get("descriptor_sha256")),
        verify_full_model=verify_full_model,
    )
    verified_dataset = validate_compact_kernel_receipt(
        Path(str(dataset.get("receipt_path"))),
        expected_descriptor_sha256=str(model.get("descriptor_sha256")),
    )
    if verified_dataset.get("receipt_sha256") != dataset.get("receipt_sha256"):
        _fail("training_dataset_receipt_drift")
    config = TrainingConfig(**plan["training"])
    optimizer = (
        OptimizerConfig(**plan["optimizer"])
        if schema == PLAN_SCHEMA
        else None
    )
    policy = StagePolicy(**plan["stages"])
    canary_policy = CanaryPolicy(**plan["canary"])
    config.validate()
    if optimizer is not None:
        optimizer.validate()
    policy.validate()
    canary_policy.validate()
    if _strict_json(root / CONFIG_FILE, role="mlx_config") != _training_config_document(
        config, optimizer
    ):
        _fail("mlx_config_drift")
    expected_identity = {
        "schema": ADAPTER_IDENTITY_SCHEMA,
        "run_id": plan["run_id"],
        "model_descriptor_sha256": model["descriptor_sha256"],
        "dataset_receipt_sha256": dataset["receipt_sha256"],
        "training_identity_sha256": document_sha256(asdict(config)),
        "stage_policy_sha256": document_sha256(asdict(policy)),
        "canary_policy_sha256": document_sha256(asdict(canary_policy)),
    }
    if optimizer is not None:
        expected_identity["optimizer_identity_sha256"] = document_sha256(
            asdict(optimizer)
        )
    if _strict_json(root / IDENTITY_FILE, role="adapter_identity") != expected_identity:
        _fail("adapter_identity_mismatch")
    return plan


def _journal_signature(material: Mapping[str, Any], key: bytes) -> str:
    if len(key) < 32:
        _fail("journal_key_too_short")
    return hmac.new(key, canonical_json_bytes(material), hashlib.sha256).hexdigest()


def read_authenticated_journal(path: Path, *, key: bytes) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        _fail("journal_custody_invalid")
    events: list[dict[str, Any]] = []
    previous = "0" * 64
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise CandidateCortexTrainingError("journal_unreadable") from exc
    for index, raw in enumerate(lines):
        try:
            event = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CandidateCortexTrainingError("journal_event_invalid") from exc
        if not isinstance(event, dict):
            _fail("journal_event_invalid")
        material = dict(event)
        signature = material.pop("hmac_sha256", None)
        event_sha = material.pop("event_sha256", None)
        if (
            event.get("schema") != JOURNAL_EVENT_SCHEMA
            or event.get("sequence") != index
            or event.get("previous_event_sha256") != previous
            or event_sha != document_sha256(material)
            or signature != _journal_signature({**material, "event_sha256": event_sha}, key)
        ):
            _fail("journal_authentication_failed")
        previous = str(event_sha)
        events.append(event)
    return events


def append_authenticated_event(
    path: Path,
    *,
    key: bytes,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z_]+", event_type):
        _fail("journal_event_type_invalid")
    _ensure_directory(path.parent)
    with interprocess_file_lock(path.parent / LOCK_FILE):
        events = read_authenticated_journal(path, key=key)
        previous = str(events[-1]["event_sha256"]) if events else "0" * 64
        material: dict[str, Any] = {
            "schema": JOURNAL_EVENT_SCHEMA,
            "sequence": len(events),
            "previous_event_sha256": previous,
            "event_type": event_type,
            "payload": dict(payload),
        }
        event_sha = document_sha256(material)
        event = {
            **material,
            "event_sha256": event_sha,
            "hmac_sha256": _journal_signature({**material, "event_sha256": event_sha}, key),
        }
        existing = path.read_bytes() if path.exists() else b""
        _write_bytes(path, existing + canonical_json_bytes(event) + b"\n")
        return event


def discover_exact_checkpoint(
    adapter_root: Path,
    *,
    expected_cumulative_iterations: int,
) -> dict[str, Any]:
    root = adapter_root.expanduser().resolve(strict=True)
    matches: list[Path] = []
    for path in root.iterdir():
        match = _CHECKPOINT.fullmatch(path.name)
        if match and int(match.group("step")) == expected_cumulative_iterations:
            matches.append(path)
    if len(matches) != 1:
        _fail("checkpoint_ambiguous" if matches else "checkpoint_missing")
    path = matches[0]
    binding = _file_binding(path)
    if binding["size_bytes"] <= 0:
        _fail("checkpoint_empty")
    return binding


def stage_adapter_root(plan: Mapping[str, Any], stage_index: int) -> Path:
    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 0:
        _fail("stage_index_invalid")
    run_root = Path(str(plan["paths"]["run_root"])).resolve(strict=True)
    root = (run_root / "stages" / f"stage-{stage_index:04d}" / "adapter").resolve(
        strict=False
    )
    if not _within(root, run_root):
        _fail("stage_adapter_path_escape")
    return root


def stage_detached_root(
    plan: Mapping[str, Any], *, execution_id: str = "primary"
) -> Path:
    run_root = Path(str(plan["paths"]["run_root"])).resolve(strict=True)
    if execution_id == "primary":
        root = (run_root / "adaptive-execution" / "detached").resolve(strict=False)
    else:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?", execution_id):
            _fail("adaptive_execution_id_invalid")
        root = (
            run_root
            / "adaptive-execution"
            / "executions"
            / execution_id
            / "detached"
        ).resolve(strict=False)
    if not _within(root, run_root):
        _fail("stage_detached_path_escape")
    return root


def build_stage_command(
    plan: Mapping[str, Any],
    *,
    stage_index: int,
    resume_checkpoint: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    config = TrainingConfig(**dict(plan["training"]))
    optimizer = OptimizerConfig(**dict(plan.get("optimizer") or {"name": "adam"}))
    policy = StagePolicy(**dict(plan["stages"]))
    config.validate()
    optimizer.validate()
    policy.validate()
    if stage_index == 0 and resume_checkpoint is not None:
        _fail("initial_stage_resume_forbidden")
    if stage_index > 0 and resume_checkpoint is None:
        _fail("resume_checkpoint_required")
    if resume_checkpoint is not None:
        resume = Path(str(resume_checkpoint.get("path"))).resolve(strict=True)
        adapter_root = Path(str(plan["paths"]["adapter_root"])).resolve(strict=True)
        if not _within(resume, adapter_root):
            _fail("resume_checkpoint_path_escape")
        current = _file_binding(resume)
        if any(current.get(key) != resume_checkpoint.get(key) for key in current):
            _fail("resume_checkpoint_drift")
    command = [
        str(plan["python"]),
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        str(plan["model"]["canonical_path"]),
        "--train",
        "--data",
        str(plan["dataset"]["data_root"]),
        "--fine-tune-type",
        "lora",
        "--optimizer",
        optimizer.name,
        "--mask-prompt",
        "-c",
        str(plan["paths"]["mlx_config"]),
        "--num-layers",
        str(config.num_layers),
        "--batch-size",
        str(config.batch_size),
        "--iters",
        str(policy.iterations(stage_index)),
        "--val-batches",
        str(config.val_batches),
        "--learning-rate",
        format(config.learning_rate, ".17g"),
        "--steps-per-report",
        str(config.report_every),
        "--steps-per-eval",
        str(config.eval_every),
        "--grad-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--adapter-path",
        str(stage_adapter_root(plan, stage_index)),
        "--save-every",
        str(policy.iterations(stage_index)),
        "--max-seq-length",
        str(config.max_seq_length),
        "--grad-checkpoint",
        "--seed",
        str(config.seed),
    ]
    if resume_checkpoint is not None:
        command.extend(("--resume-adapter-file", str(resume_checkpoint["path"])))
    return tuple(command)


def canary_micro_iterations(
    config: TrainingConfig,
    policy: CanaryPolicy,
) -> int:
    """Return microsteps needed to execute exactly N optimizer updates.

    One extra microstep triggers MLX-LM's pre-step validation after the final
    accumulated update.  Because that remainder never reaches an optimizer
    boundary, final adapter weights still represent exactly ``optimizer_steps``
    updates rather than an under-measured approximation.
    """

    config.validate()
    policy.validate()
    return config.gradient_accumulation_steps * policy.optimizer_steps + 1


def canary_checkpoint_iteration(
    config: TrainingConfig,
    policy: CanaryPolicy,
) -> int:
    config.validate()
    policy.validate()
    return config.gradient_accumulation_steps * policy.optimizer_steps


def build_canary_command(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Build the production-shaped canary command bound by ``plan``."""

    config = TrainingConfig(**dict(plan["training"]))
    optimizer = OptimizerConfig(**dict(plan.get("optimizer") or {"name": "adam"}))
    policy = CanaryPolicy(**dict(plan["canary"]))
    config.validate()
    optimizer.validate()
    policy.validate()
    checkpoint_iteration = canary_checkpoint_iteration(config, policy)
    validation_interval = (
        config.gradient_accumulation_steps * policy.validation_interval_optimizer_steps
    )
    return (
        str(plan["python"]),
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        str(plan["model"]["canonical_path"]),
        "--train",
        "--data",
        str(plan["dataset"]["data_root"]),
        "--fine-tune-type",
        "lora",
        "--optimizer",
        optimizer.name,
        "--mask-prompt",
        "-c",
        str(plan["paths"]["mlx_config"]),
        "--num-layers",
        str(config.num_layers),
        "--batch-size",
        str(config.batch_size),
        "--iters",
        str(canary_micro_iterations(config, policy)),
        "--val-batches",
        str(policy.validation_batches),
        "--learning-rate",
        format(config.learning_rate, ".17g"),
        "--steps-per-report",
        str(config.gradient_accumulation_steps),
        "--steps-per-eval",
        str(validation_interval),
        "--grad-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--adapter-path",
        str(plan["paths"]["canary_adapter_root"]),
        "--save-every",
        str(checkpoint_iteration),
        "--max-seq-length",
        str(config.max_seq_length),
        "--grad-checkpoint",
        "--seed",
        str(config.seed),
    )


def _finite_report_value(raw: str, *, role: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise CandidateCortexTrainingError(f"{role}_invalid") from exc
    if not math.isfinite(value):
        _fail(f"{role}_non_finite")
    return value


def parse_canary_training_log(
    payload: bytes,
    *,
    config: TrainingConfig,
    policy: CanaryPolicy,
) -> dict[str, Any]:
    """Parse MLX-LM's measured reports without treating prose as evidence."""

    config.validate()
    policy.validate()
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise CandidateCortexTrainingError("canary_log_utf8_invalid") from exc
    train_reports: dict[int, dict[str, Any]] = {}
    validation_reports: dict[int, dict[str, Any]] = {}
    for line in lines:
        train_match = _TRAIN_REPORT.fullmatch(line.strip())
        if train_match:
            iteration = int(train_match.group("iteration"))
            if iteration in train_reports:
                _fail("canary_train_report_duplicate")
            train_reports[iteration] = {
                "iteration": iteration,
                "loss": _finite_report_value(train_match.group("loss"), role="train_loss"),
                "learning_rate": _finite_report_value(
                    train_match.group("learning_rate"), role="train_learning_rate"
                ),
                "iterations_per_second": _finite_report_value(
                    train_match.group("iterations_per_second"), role="train_throughput"
                ),
                "tokens_per_second": _finite_report_value(
                    train_match.group("tokens_per_second"), role="token_throughput"
                ),
                "trained_tokens": _finite_report_value(
                    train_match.group("trained_tokens"), role="trained_tokens"
                ),
                "peak_memory_gb": _finite_report_value(
                    train_match.group("peak_memory_gb"), role="peak_memory"
                ),
            }
            continue
        validation_match = _VAL_REPORT.fullmatch(line.strip())
        if validation_match:
            iteration = int(validation_match.group("iteration"))
            if iteration in validation_reports:
                _fail("canary_validation_report_duplicate")
            validation_reports[iteration] = {
                "iteration": iteration,
                "loss": _finite_report_value(
                    validation_match.group("loss"), role="validation_loss"
                ),
                "duration_seconds": _finite_report_value(
                    validation_match.group("duration_seconds"),
                    role="validation_duration",
                ),
            }
    checkpoint_iteration = canary_checkpoint_iteration(config, policy)
    expected_optimizer_reports = tuple(
        config.gradient_accumulation_steps * index
        for index in range(1, policy.optimizer_steps + 1)
    )
    if any(iteration not in train_reports for iteration in expected_optimizer_reports):
        _fail("canary_optimizer_report_missing")
    final_iteration = canary_micro_iterations(config, policy)
    if 1 not in validation_reports or final_iteration not in validation_reports:
        _fail("canary_validation_endpoint_missing")
    ordered_train = [train_reports[index] for index in sorted(train_reports)]
    ordered_validation = [
        validation_reports[index] for index in sorted(validation_reports)
    ]
    return {
        "micro_iterations": final_iteration,
        "checkpoint_iteration": checkpoint_iteration,
        "optimizer_steps": policy.optimizer_steps,
        "optimizer_update_reports": [
            train_reports[index] for index in expected_optimizer_reports
        ],
        "all_training_reports": ordered_train,
        "validation_reports": ordered_validation,
        "initial_validation_loss": validation_reports[1]["loss"],
        "final_validation_loss": validation_reports[final_iteration]["loss"],
        "peak_mlx_memory_gb": max(
            report["peak_memory_gb"] for report in ordered_train
        ),
    }


def adjudicate_canary(
    plan: Mapping[str, Any],
    *,
    detached_receipt: Mapping[str, Any],
    expected_target_command: Sequence[str],
    detached_log_path: Path,
    host_metrics_path: Path,
    journal_key: bytes,
    verify_full_model: bool = True,
) -> dict[str, Any]:
    """Verify and publish one immutable canary observation and admission."""

    verified_plan = load_and_verify_plan(
        Path(str(plan["paths"]["run_root"])),
        verify_full_model=verify_full_model,
    )
    if verified_plan.get("plan_sha256") != plan.get("plan_sha256"):
        _fail("canary_plan_drift")
    plan = verified_plan
    receipt_body = dict(detached_receipt)
    claimed_receipt_sha = receipt_body.pop("receipt_sha256", None)
    if claimed_receipt_sha != document_sha256(receipt_body):
        _fail("canary_detached_receipt_digest_invalid")
    if (
        detached_receipt.get("command") != list(expected_target_command)
        or detached_receipt.get("status") != "passed"
        or detached_receipt.get("passed") is not True
        or detached_receipt.get("returncode") != 0
        or detached_receipt.get("restart_count") != 0
        or detached_receipt.get("containment_verified") is not True
        or detached_receipt.get("process_group_empty") is not True
        or detached_receipt.get("lineage_empty") is not True
    ):
        _fail("canary_detached_execution_invalid")
    log_binding = _file_binding(detached_log_path)
    parsed = parse_canary_training_log(
        detached_log_path.read_bytes(),
        config=TrainingConfig(**dict(plan["training"])),
        policy=CanaryPolicy(**dict(plan["canary"])),
    )
    metrics = _strict_json(host_metrics_path, role="canary_host_metrics")
    metrics_required = {
        "schema",
        "plan_sha256",
        "model_descriptor_sha256",
        "dataset_receipt_sha256",
        "training_command_sha256",
        "target_pid",
        "started_at_unix",
        "finished_at_unix",
        "duration_seconds",
        "sample_count",
        "min_available_bytes",
        "max_used_percent",
        "max_process_rss_bytes",
    }
    if set(metrics) != metrics_required or metrics.get("schema") != CANARY_HOST_METRICS_SCHEMA:
        _fail("canary_host_metrics_schema_invalid")
    command = build_canary_command(plan)
    if (
        metrics.get("plan_sha256") != plan["plan_sha256"]
        or metrics.get("model_descriptor_sha256")
        != plan["model"]["descriptor_sha256"]
        or metrics.get("dataset_receipt_sha256")
        != plan["dataset"]["receipt_sha256"]
        or metrics.get("training_command_sha256") != document_sha256(list(command))
        or metrics.get("target_pid") != detached_receipt.get("child_pid")
    ):
        _fail("canary_host_metrics_identity_mismatch")
    integer_metrics = (
        metrics.get("sample_count"),
        metrics.get("min_available_bytes"),
        metrics.get("max_process_rss_bytes"),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integer_metrics):
        _fail("canary_host_metrics_invalid")
    for field in ("started_at_unix", "finished_at_unix", "duration_seconds", "max_used_percent"):
        value = metrics.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            _fail("canary_host_metrics_invalid")
    if (
        float(metrics["duration_seconds"]) <= 0
        or float(metrics["finished_at_unix"]) < float(metrics["started_at_unix"])
        or not 0 <= float(metrics["max_used_percent"]) <= 100
    ):
        _fail("canary_host_metrics_invalid")
    config = TrainingConfig(**dict(plan["training"]))
    policy = CanaryPolicy(**dict(plan["canary"]))
    checkpoint = discover_exact_checkpoint(
        Path(str(plan["paths"]["canary_adapter_root"])),
        expected_cumulative_iterations=canary_checkpoint_iteration(config, policy),
    )
    final_adapter = _file_binding(
        Path(str(plan["paths"]["canary_adapter_root"])) / "adapters.safetensors"
    )
    observation_body = {
        "schema": CANARY_OBSERVATION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "model_descriptor_sha256": plan["model"]["descriptor_sha256"],
        "dataset_receipt_sha256": plan["dataset"]["receipt_sha256"],
        "training_command": list(command),
        "training_command_sha256": document_sha256(list(command)),
        "detached_receipt_sha256": claimed_receipt_sha,
        "detached_duration_seconds": detached_receipt["duration_s"],
        "trainer_pid": detached_receipt["child_pid"],
        "trainer_start_token": detached_receipt["child_start_token"],
        "log": log_binding,
        "host_metrics": metrics,
        "training_reports": parsed,
        "checkpoint": checkpoint,
        "final_adapter": final_adapter,
    }
    observation = {
        **observation_body,
        "observation_sha256": document_sha256(observation_body),
    }
    initial_loss = float(parsed["initial_validation_loss"])
    final_loss = float(parsed["final_validation_loss"])
    min_available_gb = float(metrics["min_available_bytes"]) / float(1024**3)
    checks = {
        "optimizer_updates_complete": len(parsed["optimizer_update_reports"])
        == policy.optimizer_steps,
        "checkpoint_exact": checkpoint["size_bytes"] > 0,
        "final_adapter_present": final_adapter["size_bytes"] > 0,
        "mlx_peak_within_bound": float(parsed["peak_mlx_memory_gb"])
        <= policy.max_peak_mlx_gb,
        "host_headroom_within_bound": min_available_gb
        >= policy.min_host_available_gb,
        "validation_stable": final_loss
        <= initial_loss * policy.max_validation_loss_ratio,
    }
    failed = sorted(key for key, value in checks.items() if not value)
    admission_body = {
        "schema": CANARY_ADMISSION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "observation_sha256": observation["observation_sha256"],
        "status": "PASS" if not failed else "REJECT",
        "failed_checks": failed,
        "checks": checks,
        "optimizer_steps": policy.optimizer_steps,
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "validation_loss_ratio": final_loss / initial_loss if initial_loss else None,
        "peak_mlx_memory_gb": parsed["peak_mlx_memory_gb"],
        "min_host_available_gb": min_available_gb,
        "detached_duration_seconds": detached_receipt["duration_s"],
    }
    admission = {
        **admission_body,
        "admission_sha256": document_sha256(admission_body),
    }
    for path, payload, conflict in (
        (
            Path(str(plan["paths"]["canary_observation"])),
            observation,
            "canary_observation_conflict",
        ),
        (
            Path(str(plan["paths"]["canary_admission"])),
            admission,
            "canary_admission_conflict",
        ),
    ):
        encoded = canonical_json_bytes(payload) + b"\n"
        if not _write_bytes_if_absent(path, encoded) and path.read_bytes() != encoded:
            _fail(conflict)
    journal_path = Path(str(plan["paths"]["journal"]))
    events = read_authenticated_journal(journal_path, key=journal_key)
    if not events:
        append_authenticated_event(
            journal_path,
            key=journal_key,
            event_type="canary_observed",
            payload=observation,
        )
        append_authenticated_event(
            journal_path,
            key=journal_key,
            event_type="canary_admitted",
            payload=admission,
        )
    else:
        expected = [
            ("canary_observed", observation),
            ("canary_admitted", admission),
        ]
        if len(events) != 2 or any(
            event.get("event_type") != event_type or event.get("payload") != payload
            for event, (event_type, payload) in zip(events, expected, strict=True)
        ):
            _fail("canary_journal_conflict")
    return {"observation": observation, "admission": admission}


def _validated_admission(
    raw: Mapping[str, Any],
    *,
    stage_index: int,
    policy: StagePolicy,
) -> dict[str, Any]:
    required = {
        "schema",
        "stage_index",
        "model_free",
        "persona_score",
        "retention_score",
        "no_regression_score",
        "regressions",
        "checks",
        "evidence_sha256",
    }
    if set(raw) != required or raw.get("schema") != ADMISSION_SCHEMA:
        _fail("admission_schema_invalid")
    if raw.get("stage_index") != stage_index or raw.get("model_free") is not True:
        _fail("admission_identity_invalid")
    scores: dict[str, float] = {}
    for field in ("persona_score", "retention_score", "no_regression_score"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail("admission_score_invalid")
        scores[field] = float(value)
        if not math.isfinite(scores[field]) or not 0 <= scores[field] <= 1:
            _fail("admission_score_invalid")
    regressions = raw.get("regressions")
    checks = raw.get("checks")
    if (
        isinstance(regressions, bool)
        or not isinstance(regressions, int)
        or regressions < 0
        or isinstance(checks, bool)
        or not isinstance(checks, int)
        or checks < policy.min_eval_samples
        or not _is_sha256(raw.get("evidence_sha256"))
    ):
        _fail("admission_evidence_invalid")
    return {**dict(raw), **scores}


def validate_stage_observation(
    raw: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected_stage_index: int,
    launched_identity: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema",
        "stage_index",
        "cumulative_iterations",
        "validation_loss",
        "eval_samples",
        "checkpoint",
        "adapter_identity_sha256",
        "model_descriptor_sha256",
        "dataset_receipt_sha256",
        "trainer_pid",
        "trainer_start_token",
        "trainer_exit_code",
    }
    if set(raw) != required or raw.get("schema") != OBSERVATION_SCHEMA:
        _fail("stage_observation_schema_invalid")
    policy = StagePolicy(**dict(plan["stages"]))
    if raw.get("stage_index") != expected_stage_index:
        _fail("stage_observation_index_mismatch")
    if raw.get("cumulative_iterations") != policy.cumulative_iterations(expected_stage_index):
        _fail("stage_iteration_mismatch")
    loss = raw.get("validation_loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
        _fail("validation_loss_non_finite")
    if float(loss) < 0:
        _fail("validation_loss_invalid")
    if (
        not isinstance(raw.get("eval_samples"), int)
        or isinstance(raw.get("eval_samples"), bool)
        or int(raw["eval_samples"]) < policy.min_eval_samples
    ):
        _fail("validation_evidence_missing")
    if raw.get("trainer_exit_code") != 0:
        _fail("trainer_failed")
    if (
        raw.get("trainer_pid") != launched_identity.get("trainer_pid")
        or raw.get("trainer_start_token") != launched_identity.get("trainer_start_token")
    ):
        _fail("trainer_process_identity_stale")
    if raw.get("model_descriptor_sha256") != plan["model"]["descriptor_sha256"]:
        _fail("stage_model_identity_mismatch")
    if raw.get("dataset_receipt_sha256") != plan["dataset"]["receipt_sha256"]:
        _fail("stage_dataset_identity_mismatch")
    expected_adapter_identity = document_sha256(
        _strict_json(Path(str(plan["paths"]["adapter_identity"])), role="adapter_identity")
    )
    if raw.get("adapter_identity_sha256") != expected_adapter_identity:
        _fail("stage_adapter_identity_mismatch")
    checkpoint = raw.get("checkpoint")
    if not isinstance(checkpoint, dict):
        _fail("stage_checkpoint_invalid")
    observed_checkpoint = discover_exact_checkpoint(
        Path(str(plan["paths"]["checkpoint_root"])),
        expected_cumulative_iterations=policy.cumulative_iterations(expected_stage_index),
    )
    if checkpoint != observed_checkpoint:
        _fail("stage_checkpoint_drift")
    return {**dict(raw), "validation_loss": float(loss)}


def decide_after_stage(
    *,
    policy: StagePolicy,
    observations: Sequence[Mapping[str, Any]],
    admissions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(observations) != len(admissions) or not observations:
        _fail("stage_evidence_incomplete")
    for index, (observation, admission) in enumerate(zip(observations, admissions, strict=True)):
        if observation.get("stage_index") != index or admission.get("stage_index") != index:
            _fail("stage_evidence_order_invalid")
        validated = _validated_admission(admission, stage_index=index, policy=policy)
        if (
            validated["persona_score"] < policy.persona_floor
            or validated["retention_score"] < policy.retention_floor
            or validated["no_regression_score"] < policy.no_regression_floor
            or validated["regressions"] > 0
        ):
            return {"decision": "REJECT", "reason": "admission_regression", "stage": index}
        if index:
            previous = float(observations[index - 1]["validation_loss"])
            current = float(observation["validation_loss"])
            if current > previous * (1.0 + policy.max_loss_regression_fraction):
                return {"decision": "REJECT", "reason": "validation_loss_regression", "stage": index}
    plateau = 0
    for previous, current in zip(observations, observations[1:], strict=False):
        improvement = float(previous["validation_loss"]) - float(current["validation_loss"])
        plateau = plateau + 1 if improvement < policy.min_loss_improvement else 0
    completed = len(observations)
    if completed >= policy.min_stages and plateau >= policy.patience:
        return {"decision": "COMPLETE", "reason": "convergence_patience_pass", "stage": completed - 1}
    if completed >= policy.max_stages:
        return {"decision": "REJECT", "reason": "max_stages_without_convergence", "stage": completed - 1}
    return {"decision": "CONTINUE", "reason": "more_evidence_required", "stage": completed}


def next_stage_plan(
    plan: Mapping[str, Any],
    *,
    observations: Sequence[Mapping[str, Any]],
    admissions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    policy = StagePolicy(**dict(plan["stages"]))
    decision = (
        {"decision": "CONTINUE", "reason": "initial_stage", "stage": 0}
        if not observations
        else decide_after_stage(policy=policy, observations=observations, admissions=admissions)
    )
    if decision["decision"] != "CONTINUE":
        return decision
    stage_index = len(observations)
    resume = None
    if stage_index:
        resume = dict(observations[-1]["checkpoint"])
    return {
        **decision,
        "stage_index": stage_index,
        "stage_iterations": policy.iterations(stage_index),
        "cumulative_iterations": policy.cumulative_iterations(stage_index),
        "resume_checkpoint": resume,
        "command": list(
            build_stage_command(
                plan,
                stage_index=stage_index,
                resume_checkpoint=resume,
            )
        ),
    }


def admitted_adaptive_checkpoint(
    plan: Mapping[str, Any],
    *,
    authenticated_events: Sequence[Mapping[str, Any]],
    adaptive_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the only checkpoint authorized for post-training fusion.

    Fusion changes the model's representation identity. It cannot infer a
    winner from filenames or accept the newest checkpoint merely because a
    trainer stopped. The authenticated observations, model-free admissions,
    terminal decision, and current checkpoint bytes must all name one stage.
    """

    if set(adaptive_result) != {
        "schema",
        "plan_sha256",
        "decision",
        "result_sha256",
    } or adaptive_result.get("schema") != ADAPTIVE_RESULT_SCHEMA:
        _fail("adaptive_result_invalid")
    material = dict(adaptive_result)
    claimed = material.pop("result_sha256", None)
    if claimed != document_sha256(material):
        _fail("adaptive_result_digest_invalid")
    if adaptive_result.get("plan_sha256") != plan.get("plan_sha256"):
        _fail("adaptive_result_plan_mismatch")

    observations, admissions = effective_stage_evidence(authenticated_events)
    if not observations or len(observations) != len(admissions):
        _fail("adaptive_stage_evidence_incomplete")

    policy = StagePolicy(**dict(plan["stages"]))
    decision = decide_after_stage(
        policy=policy,
        observations=observations,
        admissions=admissions,
    )
    if adaptive_result.get("decision") != decision:
        _fail("adaptive_result_decision_mismatch")
    if decision.get("decision") != "COMPLETE":
        _fail("adaptive_result_not_complete")
    stage_index = decision.get("stage")
    if isinstance(stage_index, bool) or not isinstance(stage_index, int):
        _fail("adaptive_result_stage_invalid")
    if stage_index != len(observations) - 1:
        _fail("adaptive_result_stage_mismatch")

    admission = _validated_admission(
        admissions[stage_index],
        stage_index=stage_index,
        policy=policy,
    )
    if (
        admission["persona_score"] < policy.persona_floor
        or admission["retention_score"] < policy.retention_floor
        or admission["no_regression_score"] < policy.no_regression_floor
        or admission["regressions"] != 0
    ):
        _fail("adaptive_final_stage_not_admitted")

    checkpoint = observations[stage_index].get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        _fail("adaptive_final_checkpoint_invalid")
    expected = discover_exact_checkpoint(
        Path(str(plan["paths"]["checkpoint_root"])),
        expected_cumulative_iterations=policy.cumulative_iterations(stage_index),
    )
    if dict(checkpoint) != expected:
        _fail("adaptive_final_checkpoint_drift")
    return {
        "stage_index": stage_index,
        "cumulative_iterations": policy.cumulative_iterations(stage_index),
        "checkpoint": expected,
        "admission": admission,
        "decision": decision,
        "adaptive_result_sha256": str(adaptive_result["result_sha256"]),
    }


def effective_stage_evidence(
    authenticated_events: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve append-only stage evidence, including explicit corrections.

    A reconciliation never deletes or edits the original admission. It must
    name that exact admission and its evidence digest before a corrected
    admission can supersede it. Multiple unrelated admissions for one stage or
    an out-of-order correction fail closed.
    """

    observations: dict[int, dict[str, Any]] = {}
    admissions: dict[int, dict[str, Any]] = {}
    for event in authenticated_events:
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        event_type = event.get("event_type")
        if event_type == "stage_observed":
            stage_index = payload.get("stage_index")
            if (
                isinstance(stage_index, bool)
                or not isinstance(stage_index, int)
                or stage_index < 0
                or stage_index in observations
            ):
                _fail("stage_observation_journal_conflict")
            observations[stage_index] = dict(payload)
        elif event_type == "stage_admitted":
            stage_index = payload.get("stage_index")
            if (
                isinstance(stage_index, bool)
                or not isinstance(stage_index, int)
                or stage_index < 0
                or stage_index in admissions
            ):
                _fail("stage_admission_journal_conflict")
            admissions[stage_index] = dict(payload)
        elif event_type == "stage_reconciled":
            required = {
                "schema",
                "plan_sha256",
                "stage_index",
                "prior_admission_sha256",
                "prior_evidence_sha256",
                "detail_sha256",
                "evaluator_source_sha256",
                "reconciled_evidence_sha256",
                "admission",
                "reconciliation_sha256",
            }
            material = dict(payload)
            claimed = material.pop("reconciliation_sha256", None)
            stage_index = payload.get("stage_index")
            prior = admissions.get(stage_index) if isinstance(stage_index, int) else None
            corrected = payload.get("admission")
            if (
                set(payload) != required
                or payload.get("schema") != STAGE_RECONCILIATION_SCHEMA
                or claimed != document_sha256(material)
                or isinstance(stage_index, bool)
                or not isinstance(stage_index, int)
                or prior is None
                or not isinstance(corrected, Mapping)
                or payload.get("prior_admission_sha256") != document_sha256(prior)
                or payload.get("prior_evidence_sha256") != prior.get("evidence_sha256")
                or corrected.get("stage_index") != stage_index
                or corrected.get("evidence_sha256")
                != payload.get("reconciled_evidence_sha256")
                or not all(
                    _is_sha256(payload.get(field))
                    for field in (
                        "detail_sha256",
                        "evaluator_source_sha256",
                        "reconciled_evidence_sha256",
                    )
                )
            ):
                _fail("stage_reconciliation_invalid")
            admissions[stage_index] = dict(corrected)

    if set(observations) != set(admissions):
        if not (
            len(observations) == len(admissions) + 1
            and set(admissions) == set(range(len(admissions)))
            and set(observations) == set(range(len(observations)))
        ):
            _fail("adaptive_stage_evidence_incomplete")
    expected = set(range(len(observations)))
    if set(observations) != expected or set(admissions) not in (expected, expected - {len(expected) - 1}):
        _fail("adaptive_stage_evidence_order_invalid")
    return (
        [observations[index] for index in sorted(observations)],
        [admissions[index] for index in sorted(admissions)],
    )


def run_admission_callback(
    callback: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    stage_index: int,
    policy: StagePolicy,
) -> dict[str, Any]:
    try:
        result = callback(dict(context))
    except Exception as exc:  # noqa: BLE001 - callback is a trust boundary
        raise CandidateCortexTrainingError("admission_callback_failed") from exc
    if not isinstance(result, Mapping):
        _fail("admission_callback_result_invalid")
    return _validated_admission(result, stage_index=stage_index, policy=policy)


def execution_admission(
    plan: Mapping[str, Any],
    *,
    execute: bool,
    authenticated_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return whether the detached canary adapter is available.

    Adaptive stages remain unauthorized until the independently adjudicated
    canary is present and admitted; callers cannot skip this transition by
    passing ``execute=True``.
    """

    supervision = plan.get("supervision")
    if not execute:
        return {"status": "DRY_RUN", "execution_authorized": False}
    if not isinstance(supervision, dict) or supervision.get("execution_adapter_available") is not True:
        _fail("durable_supervision_adapter_unavailable")
    admission_path = Path(str(plan["paths"]["canary_admission"]))
    if not admission_path.exists():
        return {
            "status": "CANARY_REQUIRED",
            "execution_authorized": False,
            "canary_launch_authorized": True,
        }
    admission = _strict_json(admission_path, role="canary_admission")
    material = dict(admission)
    claimed = material.pop("admission_sha256", None)
    if (
        admission.get("schema") != CANARY_ADMISSION_SCHEMA
        or claimed != document_sha256(material)
        or admission.get("plan_sha256") != plan.get("plan_sha256")
        or admission.get("status") != "PASS"
    ):
        _fail("canary_admission_invalid")
    observation = _strict_json(
        Path(str(plan["paths"]["canary_observation"])),
        role="canary_observation",
    )
    observed_events = [
        event
        for event in authenticated_events
        if event.get("event_type") in {"canary_observed", "canary_admitted"}
    ]
    if not observed_events:
        return {
            "status": "CANARY_AUTHENTICATION_REQUIRED",
            "execution_authorized": False,
            "canary_launch_authorized": False,
        }
    expected = [
        ("canary_observed", observation),
        ("canary_admitted", admission),
    ]
    if len(observed_events) != 2 or any(
        event.get("event_type") != event_type or event.get("payload") != payload
        for event, (event_type, payload) in zip(observed_events, expected, strict=True)
    ):
        _fail("canary_authenticated_evidence_mismatch")
    return {
        "status": "CANARY_PASSED",
        "execution_authorized": True,
        "canary_launch_authorized": False,
        "canary_admission_sha256": claimed,
    }


__all__ = [
    "ADAPTIVE_RESULT_SCHEMA",
    "ADAPTER_IDENTITY_SCHEMA",
    "ADMISSION_SCHEMA",
    "CANARY_ADMISSION_SCHEMA",
    "CANARY_HOST_METRICS_SCHEMA",
    "CANARY_OBSERVATION_SCHEMA",
    "CONFIG_FILE",
    "CanaryPolicy",
    "CandidateCortexTrainingError",
    "DATASET_SCHEMA",
    "IDENTITY_FILE",
    "JOURNAL_FILE",
    "OBSERVATION_SCHEMA",
    "OptimizerConfig",
    "PLAN_FILE",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_V2",
    "StagePolicy",
    "TrainingConfig",
    "append_authenticated_event",
    "adaptive_result_path",
    "admitted_adaptive_checkpoint",
    "adjudicate_canary",
    "build_canary_command",
    "build_stage_command",
    "canary_checkpoint_iteration",
    "canary_micro_iterations",
    "canonical_json_bytes",
    "decide_after_stage",
    "discover_exact_checkpoint",
    "document_sha256",
    "effective_stage_evidence",
    "execution_admission",
    "file_sha256",
    "load_and_verify_plan",
    "next_stage_plan",
    "parse_canary_training_log",
    "prepare_training_run",
    "read_authenticated_journal",
    "run_admission_callback",
    "stage_adapter_root",
    "stage_detached_root",
    "validate_candidate_descriptor",
    "validate_compact_kernel_receipt",
    "validate_stage_observation",
]
