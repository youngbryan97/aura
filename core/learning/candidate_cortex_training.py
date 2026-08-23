"""Candidate-bound, staged LoRA training control without model imports.

This module plans and verifies training.  It deliberately does not import MLX
or start a process.  A caller may hand the emitted command to Aura's durable
model-lane supervisor only after that supervisor has proved its own custody.
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

PLAN_SCHEMA: Final = "aura.candidate_cortex_training.plan.v1"
DATASET_SCHEMA: Final = KERNEL_RECEIPT_SCHEMA
JOURNAL_EVENT_SCHEMA: Final = "aura.candidate_cortex_training.journal_event.v1"
OBSERVATION_SCHEMA: Final = "aura.candidate_cortex_training.stage_observation.v1"
ADMISSION_SCHEMA: Final = "aura.candidate_cortex_training.admission.v1"
ADAPTER_IDENTITY_SCHEMA: Final = "aura.candidate_cortex_training.adapter_identity.v1"
SUPERVISION_SCHEMA: Final = "aura.candidate_cortex_training.supervision.v1"
PLAN_FILE: Final = "training_plan.json"
CONFIG_FILE: Final = "mlx_lora_config.json"
IDENTITY_FILE: Final = "adapter_identity.json"
JOURNAL_FILE: Final = "training_journal.jsonl"
LOCK_FILE: Final = ".training.lock"
MAX_DOCUMENT_BYTES: Final = 16 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT = re.compile(r"(?P<step>[0-9]+)_adapters\.safetensors")


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
        if any(not 0 <= value <= 1 for value in finite[2:]) or self.min_eval_samples <= 0:
            _fail("admission_threshold_invalid")

    def iterations(self, stage_index: int) -> int:
        if not 0 <= stage_index < self.max_stages:
            _fail("stage_index_invalid")
        return self.initial_iterations * (self.growth_factor**stage_index)

    def cumulative_iterations(self, stage_index: int) -> int:
        return sum(self.iterations(index) for index in range(stage_index + 1))


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


def _training_config_document(config: TrainingConfig) -> dict[str, Any]:
    return {
        "lora_parameters": {
            "rank": config.rank,
            "scale": config.scale,
            "dropout": config.dropout,
            "keys": list(config.targets),
        }
    }


def _admission_command_identity(command: Sequence[str]) -> dict[str, Any]:
    if not command or any(not isinstance(value, str) or not value for value in command):
        _fail("admission_command_invalid")
    executable = Path(command[0]).expanduser()
    if executable.is_absolute():
        try:
            resolved = executable.resolve(strict=True)
        except OSError as exc:
            raise CandidateCortexTrainingError("admission_executable_unavailable") from exc
        if not resolved.is_file() or resolved.is_symlink():
            _fail("admission_executable_invalid")
        executable_binding: dict[str, Any] = {
            "path": str(resolved),
            "sha256": file_sha256(resolved),
        }
    else:
        executable_binding = {"path": command[0], "sha256": None}
    return {"argv": list(command), "executable": executable_binding}


def prepare_training_run(
    *,
    descriptor_path: Path,
    expected_descriptor_sha256: str,
    dataset_receipt_path: Path,
    output_root: Path,
    python_executable: Path,
    admission_command: Sequence[str],
    config: TrainingConfig | None = None,
    policy: StagePolicy | None = None,
    verify_full_model: bool = True,
) -> dict[str, Any]:
    """Validate immutable inputs and publish one content-addressed plan."""

    config = config or TrainingConfig()
    policy = policy or StagePolicy()
    config.validate()
    policy.validate()
    descriptor, model_root = validate_candidate_descriptor(
        descriptor_path,
        expected_descriptor_sha256=expected_descriptor_sha256,
        verify_full_model=verify_full_model,
    )
    dataset = validate_compact_kernel_receipt(
        dataset_receipt_path,
        expected_descriptor_sha256=expected_descriptor_sha256,
    )
    try:
        python = python_executable.expanduser().resolve(strict=True)
    except OSError as exc:
        raise CandidateCortexTrainingError("python_executable_unavailable") from exc
    if not python.is_file() or python.is_symlink() or not os.access(python, os.X_OK):
        _fail("python_executable_invalid")
    admission = _admission_command_identity(admission_command)
    identity_material = {
        "model_descriptor_sha256": expected_descriptor_sha256,
        "dataset_receipt_sha256": dataset["receipt_sha256"],
        "training": asdict(config),
        "stages": asdict(policy),
        "python": str(python),
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
    paths = {
        "run_root": str(run_root),
        "data_root": str(data_root),
        "adapter_root": str(run_root / "adapter"),
        "checkpoint_root": str(run_root / "adapter"),
        "journal": str(run_root / JOURNAL_FILE),
        "adapter_identity": str(run_root / IDENTITY_FILE),
        "mlx_config": str(run_root / CONFIG_FILE),
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
        "stages": asdict(policy),
        "python": str(python),
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
            "execution_adapter_available": False,
        },
    }
    plan_material["plan_sha256"] = document_sha256(plan_material)
    adapter_identity = {
        "schema": ADAPTER_IDENTITY_SCHEMA,
        "run_id": run_id,
        "model_descriptor_sha256": expected_descriptor_sha256,
        "dataset_receipt_sha256": dataset["receipt_sha256"],
        "training_identity_sha256": document_sha256(asdict(config)),
        "stage_policy_sha256": document_sha256(asdict(policy)),
    }
    mlx_config = _training_config_document(config)
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
    if plan.get("schema") != PLAN_SCHEMA or not _is_sha256(claimed):
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
    policy = StagePolicy(**plan["stages"])
    config.validate()
    policy.validate()
    if _strict_json(root / CONFIG_FILE, role="mlx_config") != _training_config_document(config):
        _fail("mlx_config_drift")
    expected_identity = {
        "schema": ADAPTER_IDENTITY_SCHEMA,
        "run_id": plan["run_id"],
        "model_descriptor_sha256": model["descriptor_sha256"],
        "dataset_receipt_sha256": dataset["receipt_sha256"],
        "training_identity_sha256": document_sha256(asdict(config)),
        "stage_policy_sha256": document_sha256(asdict(policy)),
    }
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


def build_stage_command(
    plan: Mapping[str, Any],
    *,
    stage_index: int,
    resume_checkpoint: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    config = TrainingConfig(**dict(plan["training"]))
    policy = StagePolicy(**dict(plan["stages"]))
    config.validate()
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
        str(plan["paths"]["adapter_root"]),
        "--save-every",
        str(config.save_every),
        "--max-seq-length",
        str(config.max_seq_length),
        "--grad-checkpoint",
        "--seed",
        str(config.seed),
    ]
    if resume_checkpoint is not None:
        command.extend(("--resume-adapter-file", str(resume_checkpoint["path"])))
    return tuple(command)


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


def execution_admission(plan: Mapping[str, Any], *, execute: bool) -> dict[str, Any]:
    """Return dry-run status or refuse unsupported durable execution."""

    supervision = plan.get("supervision")
    if not execute:
        return {"status": "DRY_RUN", "execution_authorized": False}
    if not isinstance(supervision, dict) or supervision.get("execution_adapter_available") is not True:
        _fail("durable_supervision_adapter_unavailable")
    _fail("durable_supervision_adapter_unimplemented")


__all__ = [
    "ADAPTER_IDENTITY_SCHEMA",
    "ADMISSION_SCHEMA",
    "CONFIG_FILE",
    "CandidateCortexTrainingError",
    "DATASET_SCHEMA",
    "IDENTITY_FILE",
    "JOURNAL_FILE",
    "OBSERVATION_SCHEMA",
    "PLAN_FILE",
    "PLAN_SCHEMA",
    "StagePolicy",
    "TrainingConfig",
    "append_authenticated_event",
    "build_stage_command",
    "canonical_json_bytes",
    "decide_after_stage",
    "discover_exact_checkpoint",
    "document_sha256",
    "execution_admission",
    "file_sha256",
    "load_and_verify_plan",
    "next_stage_plan",
    "prepare_training_run",
    "read_authenticated_journal",
    "run_admission_callback",
    "validate_candidate_descriptor",
    "validate_compact_kernel_receipt",
    "validate_stage_observation",
]
