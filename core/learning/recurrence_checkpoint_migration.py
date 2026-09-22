"""Certified recurrence checkpoint migration after a resource-envelope abort."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

from core.runtime.atomic_writer import atomic_write_bytes, ensure_private_directory
from core.runtime.file_read_gateway import read_stable_bytes

MIGRATION_SCHEMA = "aura.recurrence_checkpoint_migration.v2"
CHECKPOINT_SCHEMA = "aura.recurrence_native_checkpoint.v3"
POINTER_SCHEMA = "aura.recurrence_native_checkpoint_pointer.v1"
_MAX_JSON_BYTES = 256 * 1024 * 1024
_MAX_TENSOR_BYTES = 1 << 40


class RecurrenceCheckpointMigrationError(RuntimeError):
    """Stable fail-closed checkpoint migration error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise RecurrenceCheckpointMigrationError(code)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, *, role: str) -> tuple[bytes, dict[str, Any]]:
    raw = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=_MAX_JSON_BYTES)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecurrenceCheckpointMigrationError(f"{role}_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    return raw, value


def _binding(path: Path, *, max_bytes: int = _MAX_JSON_BYTES) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        _fail("migration_artifact_storage_invalid")
    raw = read_stable_bytes(resolved, max_bytes=max_bytes)
    return {"path": str(resolved), "sha256": _sha(raw), "size_bytes": len(raw)}


def _binding_matches(binding: Mapping[str, Any], *, max_bytes: int) -> bool:
    try:
        observed = _binding(Path(str(binding["path"])), max_bytes=max_bytes)
    # not a failure: a binding whose file will not read does not match, and False is
    # the refusing direction.
    except (KeyError, OSError, RecurrenceCheckpointMigrationError):
        return False
    return observed == dict(binding)


def _copy_bound(source: Path, destination: Path, *, max_bytes: int) -> dict[str, Any]:
    payload = read_stable_bytes(source.resolve(strict=True), max_bytes=max_bytes)
    atomic_write_bytes(destination, payload, mode=0o600)
    return _binding(destination, max_bytes=max_bytes)


@dataclass(frozen=True)
class RecoveryAttemptEvidence:
    """Immutable evidence for one failed recovery generation."""

    checkpoint_migration: Path
    trainer_receipt: Path
    sentinel_receipt: Path
    tombstone: Path
    footprint_ring: Path


def _validated_resource_abort(
    *,
    trainer_receipt_path: Path,
    sentinel_receipt_path: Path,
    tombstone_path: Path,
) -> tuple[bytes, dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any]]:
    trainer_raw, trainer_receipt = _read_json(
        trainer_receipt_path,
        role="failed_trainer_receipt",
    )
    sentinel_raw, sentinel_document = _read_json(
        sentinel_receipt_path,
        role="sentinel_receipt",
    )
    tombstone_raw, tombstone_document = _read_json(
        tombstone_path,
        role="sentinel_tombstone",
    )
    final_sample = tombstone_document.get("final_sample")
    started_at = trainer_receipt.get("started_at")
    if (
        trainer_receipt.get("returncode") != -9
        or trainer_receipt.get("containment_verified") is not True
        or trainer_receipt.get("process_group_empty") is not True
        or trainer_receipt.get("lineage_empty") is not True
        or not isinstance(started_at, (int, float))
        or isinstance(started_at, bool)
        or started_at <= 0
        or sentinel_document.get("returncode") != 0
        or sentinel_document.get("containment_verified") is not True
        or tombstone_document.get("schema") != "aura.memory_sentinel.tombstone.v1"
        or tombstone_document.get("guard_stage") != "compute"
        or tombstone_document.get("reason")
        != "external sentinel killed process tree at lethal ceiling"
        or not isinstance(final_sample, Mapping)
        or final_sample.get("managed_mb", 0) < final_sample.get("active_lethal_mb", 1)
    ):
        _fail("resource_abort_evidence_invalid")
    return (
        trainer_raw,
        trainer_receipt,
        sentinel_raw,
        sentinel_document,
        tombstone_raw,
        tombstone_document,
    )


def _last_jsonl_record(path: Path) -> dict[str, Any]:
    raw = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=_MAX_JSON_BYTES)
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        _fail("recovery_footprint_ring_invalid")
    try:
        record = json.loads(lines[-1])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecurrenceCheckpointMigrationError("recovery_footprint_ring_invalid") from exc
    if not isinstance(record, dict):
        _fail("recovery_footprint_ring_invalid")
    return record


def _command_option(command: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        _fail("recovery_attempt_cross_binding_invalid")
    return command[positions[0] + 1]


def _recovery_attempt_record(
    evidence: RecoveryAttemptEvidence,
    *,
    ordinal: int,
) -> dict[str, Any]:
    migration_raw, migration = _read_json(
        evidence.checkpoint_migration,
        role="recovery_checkpoint_migration",
    )
    migration_material = dict(migration)
    migration_claimed = migration_material.pop("migration_sha256", None)
    destination = migration.get("destination")
    if (
        migration.get("schema")
        not in {"aura.recurrence_checkpoint_migration.v1", MIGRATION_SCHEMA}
        or migration_claimed != _sha(_canonical(migration_material))
        or not isinstance(destination, Mapping)
    ):
        _fail("recovery_checkpoint_migration_invalid")
    (
        trainer_raw,
        trainer,
        sentinel_raw,
        sentinel,
        tombstone_raw,
        tombstone,
    ) = _validated_resource_abort(
        trainer_receipt_path=evidence.trainer_receipt,
        sentinel_receipt_path=evidence.sentinel_receipt,
        tombstone_path=evidence.tombstone,
    )
    command = trainer.get("command")
    migration_path = str(evidence.checkpoint_migration.expanduser().resolve(strict=True))
    destination_value = destination.get("root")
    if not isinstance(destination_value, str):
        _fail("recovery_checkpoint_migration_invalid")
    destination_root = str(Path(destination_value).expanduser().resolve(strict=True))
    if (
        not isinstance(command, list)
        or not all(isinstance(item, str) for item in command)
    ):
        _fail("recovery_attempt_cross_binding_invalid")
    if (
        _command_option(command, "--resume-migration-evidence") != migration_path
        or _command_option(command, "--out-dir") != destination_root
        or _last_jsonl_record(evidence.footprint_ring) != tombstone["final_sample"]
    ):
        _fail("recovery_attempt_cross_binding_invalid")
    return {
        "ordinal": ordinal,
        "checkpoint_migration": _binding(evidence.checkpoint_migration),
        "checkpoint_migration_claimed_sha256": migration_claimed,
        "checkpoint_migration_payload_sha256": _sha(migration_raw),
        "trainer_receipt": _binding(evidence.trainer_receipt),
        "trainer_receipt_claimed_sha256": trainer.get("receipt_sha256"),
        "trainer_receipt_payload_sha256": _sha(trainer_raw),
        "sentinel_receipt": _binding(evidence.sentinel_receipt),
        "sentinel_receipt_claimed_sha256": sentinel.get("receipt_sha256"),
        "sentinel_receipt_payload_sha256": _sha(sentinel_raw),
        "tombstone": _binding(evidence.tombstone),
        "tombstone_payload_sha256": _sha(tombstone_raw),
        "footprint_ring": _binding(evidence.footprint_ring),
        "trainer_started_at": trainer["started_at"],
        "terminal_managed_mb": tombstone["final_sample"]["managed_mb"],
        "terminal_lethal_mb": tombstone["final_sample"]["active_lethal_mb"],
    }


def migration_identity_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable, path-free migration identity embedded in a bundle."""

    source = document.get("source")
    destination = document.get("destination")
    failure = document.get("failure")
    change = document.get("required_execution_change")
    trainer = document.get("new_trainer")
    recovery_attempts = document.get("recovery_attempts")
    if not all(
        isinstance(value, Mapping)
        for value in (source, destination, failure, change, trainer)
    ) or not isinstance(recovery_attempts, list):
        _fail("migration_evidence_invalid")
    complete = destination.get("complete")
    adapter = destination.get("adapter")
    optimizer = destination.get("optimizer")
    if not all(isinstance(value, Mapping) for value in (complete, adapter, optimizer)):
        _fail("migration_evidence_invalid")
    return {
        "schema": MIGRATION_SCHEMA,
        "migration_sha256": document.get("migration_sha256"),
        "source_checkpoint": source.get("checkpoint"),
        "source_step": source.get("step"),
        "source_config_sha256": source.get("config_sha256"),
        "dataset_sha256": source.get("dataset_sha256"),
        "execution_spec_sha256": source.get("execution_spec_sha256"),
        "checkpoint_complete_sha256": complete.get("sha256"),
        "adapter_sha256": adapter.get("sha256"),
        "optimizer_sha256": optimizer.get("sha256"),
        "failure_tombstone_sha256": failure.get("tombstone", {}).get("sha256"),
        "recovery_attempt_count": len(recovery_attempts),
        "recovery_attempts_sha256": _sha(_canonical(recovery_attempts)),
        "activation_rematerialization": change.get("activation_rematerialization"),
        "adjoint_schema": change.get("adjoint_schema"),
        "new_trainer_sha256": trainer.get("sha256"),
    }


def prepare_migration(
    *,
    source_root: Path,
    destination_root: Path,
    failed_trainer_receipt: Path,
    sentinel_receipt: Path,
    tombstone: Path,
    protocol: Path,
    amendment: Path,
    new_trainer: Path,
    output: Path,
    recovery_attempts: Sequence[RecoveryAttemptEvidence] = (),
) -> dict[str, Any]:
    source = source_root.expanduser().resolve(strict=True)
    destination = destination_root.expanduser().resolve(strict=False)
    if source == destination or destination.exists() or destination.is_symlink():
        _fail("migration_destination_invalid")
    latest_raw, latest = _read_json(source / "latest.json", role="source_latest")
    if latest.get("schema") != POINTER_SCHEMA:
        _fail("source_latest_invalid")
    checkpoint_relative = latest.get("checkpoint")
    if not isinstance(checkpoint_relative, str) or not checkpoint_relative.startswith(
        "checkpoints/"
    ):
        _fail("source_checkpoint_path_invalid")
    checkpoint = (source / checkpoint_relative).resolve(strict=True)
    if checkpoint.parent != (source / "checkpoints").resolve(strict=True):
        _fail("source_checkpoint_path_invalid")
    complete_raw, complete = _read_json(checkpoint / "complete.json", role="source_complete")
    if (
        complete.get("schema") != CHECKPOINT_SCHEMA
        or latest.get("complete_sha256") != _sha(complete_raw)
        or complete.get("step") is None
    ):
        _fail("source_checkpoint_invalid")
    training_config_raw, training_config = _read_json(
        source / "training_config.json",
        role="source_training_config",
    )
    if complete.get("config_sha256") != _sha(training_config_raw):
        _fail("source_training_config_mismatch")
    dataset_raw, dataset_document = _read_json(
        source / "dataset_manifest.json",
        role="source_dataset_manifest",
    )
    execution_spec_raw, execution_spec_document = _read_json(
        source / "execution_spec.json",
        role="source_execution_spec",
    )
    if (
        complete.get("dataset_sha256") != _sha(dataset_raw)
        or complete.get("execution_spec_sha256") != _sha(
            _canonical(execution_spec_document)
        )
        or training_config.get("dataset_sha256") != _sha(dataset_raw)
        or training_config.get("execution_spec_sha256")
        != complete.get("execution_spec_sha256")
    ):
        _fail("source_scientific_inputs_mismatch")
    (
        trainer_raw,
        trainer_receipt,
        sentinel_raw,
        sentinel_document,
        tombstone_raw,
        tombstone_document,
    ) = _validated_resource_abort(
        trainer_receipt_path=failed_trainer_receipt,
        sentinel_receipt_path=sentinel_receipt,
        tombstone_path=tombstone,
    )
    final_sample = tombstone_document["final_sample"]
    recovery_records = [
        _recovery_attempt_record(evidence, ordinal=index)
        for index, evidence in enumerate(recovery_attempts, start=1)
    ]
    previous_started_at = trainer_receipt["started_at"]
    for record in recovery_records:
        if record["trainer_started_at"] <= previous_started_at:
            _fail("recovery_attempt_chronology_invalid")
        previous_started_at = record["trainer_started_at"]
    destination_checkpoint = ensure_private_directory(
        destination / "checkpoints" / checkpoint.name
    )
    copied = {
        "complete": _copy_bound(
            checkpoint / "complete.json",
            destination_checkpoint / "complete.json",
            max_bytes=_MAX_JSON_BYTES,
        ),
        "adapter": _copy_bound(
            checkpoint / str(complete["adapter"]["path"]),
            destination_checkpoint / str(complete["adapter"]["path"]),
            max_bytes=_MAX_TENSOR_BYTES,
        ),
        "optimizer": _copy_bound(
            checkpoint / str(complete["optimizer"]["path"]),
            destination_checkpoint / str(complete["optimizer"]["path"]),
            max_bytes=_MAX_TENSOR_BYTES,
        ),
    }
    latest_copy = _copy_bound(
        source / "latest.json",
        destination / "latest.json",
        max_bytes=_MAX_JSON_BYTES,
    )
    material = {
        "schema": MIGRATION_SCHEMA,
        "reason": "compute_memory_envelope_exceeded_requires_exact_rematerialization",
        "prepared_at": time.time(),
        "source": {
            "root": str(source),
            "latest": _binding(source / "latest.json"),
            "checkpoint": checkpoint_relative,
            "complete": _binding(checkpoint / "complete.json"),
            "adapter": _binding(
                checkpoint / str(complete["adapter"]["path"]),
                max_bytes=_MAX_TENSOR_BYTES,
            ),
            "optimizer": _binding(
                checkpoint / str(complete["optimizer"]["path"]),
                max_bytes=_MAX_TENSOR_BYTES,
            ),
            "training_config": _binding(source / "training_config.json"),
            "training_config_document": training_config,
            "dataset_manifest": _binding(source / "dataset_manifest.json"),
            "dataset_manifest_document": dataset_document,
            "execution_spec": _binding(source / "execution_spec.json"),
            "execution_spec_document": execution_spec_document,
            "step": complete["step"],
            "config_sha256": complete["config_sha256"],
            "dataset_sha256": complete["dataset_sha256"],
            "execution_spec_sha256": complete["execution_spec_sha256"],
        },
        "destination": {
            "root": str(destination),
            "checkpoint": checkpoint_relative,
            "latest": latest_copy,
            **copied,
        },
        "failure": {
            "trainer_receipt": _binding(failed_trainer_receipt),
            "trainer_receipt_claimed_sha256": trainer_receipt.get("receipt_sha256"),
            "sentinel_receipt": _binding(sentinel_receipt),
            "sentinel_receipt_claimed_sha256": sentinel_document.get("receipt_sha256"),
            "tombstone": _binding(tombstone),
            "terminal_managed_mb": final_sample["managed_mb"],
            "terminal_lethal_mb": final_sample["active_lethal_mb"],
            "trainer_started_at": trainer_receipt["started_at"],
            "trainer_receipt_payload_sha256": _sha(trainer_raw),
            "sentinel_receipt_payload_sha256": _sha(sentinel_raw),
            "tombstone_payload_sha256": _sha(tombstone_raw),
        },
        "recovery_attempts": recovery_records,
        "protocol": _binding(protocol),
        "amendment": _binding(amendment),
        "new_trainer": _binding(new_trainer),
        "required_execution_change": {
            "gradient_schema": "aura.recurrence_streamed_depth_gradient.v6",
            "activation_rematerialization": "exact_discrete_adjoint",
            "adjoint_schema": "aura.recurrence_exact_discrete_adjoint.v1",
            "boundary_state_storage": "materialized_stop_gradient",
            "terminal_branch_graphs_concurrent": 1,
            "recurrent_transition_graphs_concurrent": 1,
            "optimizer_state_reset": False,
            "adapter_state_reset": False,
            "sample_cursor_reset": False,
        },
    }
    document = {**material, "migration_sha256": _sha(_canonical(material))}
    output = output.expanduser().resolve(strict=False)
    if output != destination / "checkpoint_migration.json":
        _fail("migration_output_path_invalid")
    atomic_write_bytes(output, _canonical(document) + b"\n", mode=0o600)
    return document


def verify_migration(
    path: Path,
    *,
    expected_destination_root: Path,
    expected_trainer_sha256: str,
    allow_destination_pointer_advance: bool = True,
) -> dict[str, Any]:
    _raw, document = _read_json(path, role="checkpoint_migration")
    claimed = document.get("migration_sha256")
    material = dict(document)
    material.pop("migration_sha256", None)
    source = document.get("source")
    destination = document.get("destination")
    change = document.get("required_execution_change")
    trainer = document.get("new_trainer")
    failure = document.get("failure")
    recovery_attempts = document.get("recovery_attempts")
    if (
        document.get("schema") != MIGRATION_SCHEMA
        or claimed != _sha(_canonical(material))
        or document.get("reason")
        != "compute_memory_envelope_exceeded_requires_exact_rematerialization"
        or not all(
            isinstance(value, Mapping)
            for value in (source, destination, change, trainer, failure)
        )
        or not isinstance(recovery_attempts, list)
        or Path(str(destination.get("root"))).resolve(strict=True)
        != expected_destination_root.resolve(strict=True)
        or trainer.get("sha256") != expected_trainer_sha256
        or change
        != {
            "gradient_schema": "aura.recurrence_streamed_depth_gradient.v6",
            "activation_rematerialization": "exact_discrete_adjoint",
            "adjoint_schema": "aura.recurrence_exact_discrete_adjoint.v1",
            "boundary_state_storage": "materialized_stop_gradient",
            "terminal_branch_graphs_concurrent": 1,
            "recurrent_transition_graphs_concurrent": 1,
            "optimizer_state_reset": False,
            "adapter_state_reset": False,
            "sample_cursor_reset": False,
        }
    ):
        _fail("migration_evidence_invalid")
    source_training_config = source.get("training_config")
    source_training_document = source.get("training_config_document")
    source_dataset_manifest = source.get("dataset_manifest")
    source_dataset_document = source.get("dataset_manifest_document")
    source_execution_spec = source.get("execution_spec")
    source_execution_document = source.get("execution_spec_document")
    if not all(
        isinstance(value, Mapping)
        for value in (
            source_training_config,
            source_training_document,
            source_dataset_manifest,
            source_dataset_document,
            source_execution_spec,
            source_execution_document,
        )
    ):
        _fail("migration_evidence_invalid")
    bindings = [
        (source.get("latest"), _MAX_JSON_BYTES),
        (source.get("complete"), _MAX_JSON_BYTES),
        (source.get("adapter"), _MAX_TENSOR_BYTES),
        (source.get("optimizer"), _MAX_TENSOR_BYTES),
        (source.get("training_config"), _MAX_JSON_BYTES),
        (source.get("dataset_manifest"), _MAX_JSON_BYTES),
        (source.get("execution_spec"), _MAX_JSON_BYTES),
        (destination.get("complete"), _MAX_JSON_BYTES),
        (destination.get("adapter"), _MAX_TENSOR_BYTES),
        (destination.get("optimizer"), _MAX_TENSOR_BYTES),
        (document.get("protocol"), _MAX_JSON_BYTES),
        (document.get("amendment"), _MAX_JSON_BYTES),
        (trainer, _MAX_JSON_BYTES),
        (failure.get("trainer_receipt"), _MAX_JSON_BYTES),
        (failure.get("sentinel_receipt"), _MAX_JSON_BYTES),
        (failure.get("tombstone"), _MAX_JSON_BYTES),
    ]
    destination_latest = destination.get("latest")
    destination_latest_matches = isinstance(
        destination_latest,
        Mapping,
    ) and _binding_matches(destination_latest, max_bytes=_MAX_JSON_BYTES)
    if not allow_destination_pointer_advance:
        bindings.append((destination.get("latest"), _MAX_JSON_BYTES))
    previous_started_at = failure.get("trainer_started_at")
    if (
        not isinstance(previous_started_at, (int, float))
        or isinstance(previous_started_at, bool)
        or previous_started_at <= 0
    ):
        _fail("migration_evidence_invalid")
    for index, attempt in enumerate(recovery_attempts, start=1):
        if not isinstance(attempt, Mapping) or attempt.get("ordinal") != index:
            _fail("recovery_attempt_evidence_invalid")
        try:
            evidence = RecoveryAttemptEvidence(
                checkpoint_migration=Path(str(attempt["checkpoint_migration"]["path"])),
                trainer_receipt=Path(str(attempt["trainer_receipt"]["path"])),
                sentinel_receipt=Path(str(attempt["sentinel_receipt"]["path"])),
                tombstone=Path(str(attempt["tombstone"]["path"])),
                footprint_ring=Path(str(attempt["footprint_ring"]["path"])),
            )
        except (KeyError, TypeError):
            _fail("recovery_attempt_evidence_invalid")
        observed_attempt = _recovery_attempt_record(evidence, ordinal=index)
        if observed_attempt != dict(attempt):
            _fail("recovery_attempt_evidence_changed")
        if observed_attempt["trainer_started_at"] <= previous_started_at:
            _fail("recovery_attempt_chronology_invalid")
        previous_started_at = observed_attempt["trainer_started_at"]
        bindings.extend(
            (
                (attempt.get("checkpoint_migration"), _MAX_JSON_BYTES),
                (attempt.get("trainer_receipt"), _MAX_JSON_BYTES),
                (attempt.get("sentinel_receipt"), _MAX_JSON_BYTES),
                (attempt.get("tombstone"), _MAX_JSON_BYTES),
                (attempt.get("footprint_ring"), _MAX_JSON_BYTES),
            )
        )
    if any(
        not isinstance(binding, Mapping)
        or not _binding_matches(binding, max_bytes=max_bytes)
        for binding, max_bytes in bindings
    ):
        _fail("migration_artifact_binding_changed")
    if allow_destination_pointer_advance and not destination_latest_matches:
        latest_binding = destination_latest
        if not isinstance(latest_binding, Mapping):
            _fail("migration_evidence_invalid")
        latest_path = Path(str(latest_binding.get("path")))
        destination_root = expected_destination_root.resolve(strict=True)
        if latest_path.resolve(strict=True) != destination_root / "latest.json":
            _fail("destination_advanced_pointer_invalid")
        latest_raw, latest = _read_json(latest_path, role="destination_latest")
        checkpoint_relative = latest.get("checkpoint")
        if (
            latest.get("schema") != POINTER_SCHEMA
            or not isinstance(checkpoint_relative, str)
            or not checkpoint_relative.startswith("checkpoints/")
        ):
            _fail("destination_advanced_pointer_invalid")
        checkpoint = (destination_root / checkpoint_relative).resolve(strict=True)
        if checkpoint.parent != (destination_root / "checkpoints").resolve(strict=True):
            _fail("destination_advanced_pointer_invalid")
        complete_raw, complete = _read_json(
            checkpoint / "complete.json",
            role="destination_advanced_complete",
        )
        step = complete.get("step")
        if (
            complete.get("schema") != CHECKPOINT_SCHEMA
            or type(step) is not int
            or step <= int(source.get("step", -1))
            or latest.get("complete_sha256") != _sha(complete_raw)
        ):
            _fail("destination_advanced_pointer_invalid")
        for role in ("adapter", "optimizer"):
            tensor = complete.get(role)
            if not isinstance(tensor, Mapping):
                _fail("destination_advanced_checkpoint_invalid")
            relative = tensor.get("path")
            if not isinstance(relative, str) or Path(relative).name != relative:
                _fail("destination_advanced_checkpoint_invalid")
            observed = _binding(checkpoint / relative, max_bytes=_MAX_TENSOR_BYTES)
            if (
                observed["sha256"] != tensor.get("sha256")
                or observed["size_bytes"] != tensor.get("size_bytes")
            ):
                _fail("destination_advanced_checkpoint_invalid")
        if not latest_raw:
            _fail("destination_advanced_pointer_invalid")
    training_config_raw, observed_training_config = _read_json(
        Path(str(source_training_config["path"])),
        role="source_training_config",
    )
    if (
        observed_training_config != source_training_document
        or _sha(training_config_raw) != source.get("config_sha256")
    ):
        _fail("migration_source_config_changed")
    dataset_raw, observed_dataset = _read_json(
        Path(str(source_dataset_manifest["path"])),
        role="source_dataset_manifest",
    )
    execution_raw, observed_execution = _read_json(
        Path(str(source_execution_spec["path"])),
        role="source_execution_spec",
    )
    if (
        observed_dataset != source_dataset_document
        or _sha(dataset_raw) != source.get("dataset_sha256")
        or observed_execution != source_execution_document
        or _sha(_canonical(observed_execution)) != source.get("execution_spec_sha256")
        or not execution_raw
    ):
        _fail("migration_source_scientific_inputs_changed")
    for role in ("complete", "adapter", "optimizer"):
        source_binding = source.get(role)
        destination_binding = destination.get(role)
        if (
            not isinstance(source_binding, Mapping)
            or not isinstance(destination_binding, Mapping)
            or source_binding.get("sha256") != destination_binding.get("sha256")
            or source_binding.get("size_bytes") != destination_binding.get("size_bytes")
        ):
            _fail("migration_checkpoint_copy_mismatch")
    return migration_identity_summary(document)


__all__ = [
    "MIGRATION_SCHEMA",
    "RecoveryAttemptEvidence",
    "RecurrenceCheckpointMigrationError",
    "migration_identity_summary",
    "prepare_migration",
    "verify_migration",
]
