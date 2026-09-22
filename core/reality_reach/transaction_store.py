"""Secure durable storage for Reality Reach actuation transactions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Never

from core.reality_reach.actuation import ActuationCommand, ActuationState
from core.runtime.atomic_writer import (
    atomic_write_bytes,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError

TRANSACTION_SCHEMA = "aura.reality-reach-actuation-transaction.v1"
COMMAND_CAPSULE_SCHEMA = "aura.reality-reach-actuation-command.v1"
RECOVERY_REPORT_SCHEMA = "aura.reality-reach-restart-recovery-report.v1"
MAX_TRANSACTION_BYTES = 1024 * 1024
MAX_RECOVERY_SCAN_ENTRIES = 8192
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_TRANSACTION_FILENAME = re.compile(r"^(?P<stem>[0-9a-f]{64})\.json$")
_COMMAND_CAPSULE_FILENAME = re.compile(r"^(?P<stem>[0-9a-f]{64})\.command$")


class RealityActuationError(RuntimeError):
    """Stable fail-closed Reality Reach transaction error."""


def transaction_sha256(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def error_evidence(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        error_type = type(error).__name__
        detail = str(error)
    else:
        error_type = "recovery"
        detail = str(error)
    return f"{error_type}:{transaction_sha256(detail)}"


def _record_path(root: Path, idempotency_key: str) -> Path:
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return root / f"{digest}.json"


def _command_capsule_filename(idempotency_key: str) -> str:
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"{digest}.command"


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_mapping(payload: bytes, *, role: str) -> dict[str, Any]:
    if not payload or len(payload) > MAX_TRANSACTION_BYTES:
        raise RealityActuationError(f"reality_actuation_{role}_size_invalid")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RealityActuationError(
                    f"reality_actuation_{role}_duplicate_json_key"
                )
            result[key] = value
        return result

    def reject_constant(_value: str) -> Never:
        raise RealityActuationError(f"reality_actuation_{role}_non_finite_number")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except RealityActuationError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise RealityActuationError(f"reality_actuation_{role}_unreadable") from exc
    if not isinstance(value, dict):
        raise RealityActuationError(f"reality_actuation_{role}_not_mapping")
    if payload != _canonical_bytes(value):
        raise RealityActuationError(f"reality_actuation_{role}_noncanonical")
    return value


def _private_root(path: Path) -> Path:
    requested = path.expanduser().absolute()
    if requested.is_symlink():
        raise RealityActuationError("reality_actuation_root_symlink_refused")
    root = Path(ensure_private_directory(requested))
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RealityActuationError("reality_actuation_root_custody_invalid")
    return root


def _read_record_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RealityActuationError("reality_actuation_transaction_open_failed") from exc
    try:
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size > MAX_TRANSACTION_BYTES
        ):
            raise RealityActuationError(
                "reality_actuation_transaction_custody_invalid"
            )
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if remaining or any(
            getattr(after, field) != getattr(metadata, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_uid",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
            )
        ):
            raise RealityActuationError(
                "reality_actuation_transaction_changed_during_read"
            )
        return b"".join(chunks)
    finally:
        os.close(fd)


def _validate_record(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "transaction_sha256",
        "command_id",
        "command_sha256",
        "idempotency_key",
        "adapter_id",
        "channel_id",
        "state",
        "revision",
        "lease_sha256",
        "preparation_sha256",
        "actuation_receipt_sha256",
        "effect_receipt_sha256",
        "rollback_receipt_sha256",
        "authority_receipt_id",
        "created_at_ns",
        "updated_at_ns",
        "manual_reconciliation_required",
        "last_error",
    }
    if set(value) != required or value.get("schema") != TRANSACTION_SCHEMA:
        raise RealityActuationError("reality_actuation_transaction_schema_invalid")
    body = dict(value)
    digest = body.pop("transaction_sha256", None)
    if digest != transaction_sha256(body):
        raise RealityActuationError("reality_actuation_transaction_digest_invalid")
    try:
        state = ActuationState(str(value["state"]))
    except ValueError as exc:
        raise RealityActuationError("reality_actuation_transaction_state_invalid") from exc
    if (
        not isinstance(value.get("revision"), int)
        or isinstance(value.get("revision"), bool)
        or int(value["revision"]) < 0
        or not isinstance(value.get("manual_reconciliation_required"), bool)
        or any(
            not isinstance(value.get(name), str)
            or not IDENTIFIER.fullmatch(str(value[name]))
            for name in ("command_id", "idempotency_key", "adapter_id", "channel_id")
        )
        or any(
            not isinstance(value.get(name), str)
            or (bool(value[name]) and not _DIGEST.fullmatch(str(value[name])))
            for name in (
                "lease_sha256",
                "preparation_sha256",
                "actuation_receipt_sha256",
                "effect_receipt_sha256",
                "rollback_receipt_sha256",
            )
        )
        or not isinstance(value.get("command_sha256"), str)
        or not _DIGEST.fullmatch(str(value["command_sha256"]))
        or not isinstance(value.get("authority_receipt_id"), str)
        or (
            bool(value["authority_receipt_id"])
            and not IDENTIFIER.fullmatch(str(value["authority_receipt_id"]))
        )
        or any(
            not isinstance(value.get(name), int)
            or isinstance(value.get(name), bool)
            or int(value[name]) <= 0
            for name in ("created_at_ns", "updated_at_ns")
        )
        or not isinstance(value.get("last_error"), str)
        or len(str(value["last_error"])) > 500
    ):
        raise RealityActuationError("reality_actuation_transaction_fields_invalid")
    required_by_state: dict[ActuationState, tuple[str, ...]] = {
        ActuationState.ADMITTED: (
            "lease_sha256",
            "preparation_sha256",
            "authority_receipt_id",
        ),
        ActuationState.DISPATCHED: (
            "lease_sha256",
            "preparation_sha256",
            "authority_receipt_id",
        ),
        ActuationState.EXECUTED: ("actuation_receipt_sha256",),
        ActuationState.EFFECT_VERIFIED: (
            "actuation_receipt_sha256",
            "effect_receipt_sha256",
        ),
        ActuationState.MANUALLY_RECONCILED: (
            "actuation_receipt_sha256",
            "effect_receipt_sha256",
            "authority_receipt_id",
        ),
        ActuationState.COMPENSATED: ("rollback_receipt_sha256",),
        ActuationState.ROLLED_BACK: ("rollback_receipt_sha256",),
        ActuationState.SAFE_STATE: ("rollback_receipt_sha256",),
    }
    if any(not value.get(name) for name in required_by_state.get(state, ())):
        raise RealityActuationError("reality_actuation_transaction_lineage_invalid")
    result = dict(value)
    result["state"] = state.value
    return result


class RealityActuationTransactionStore:
    """Owns transaction records, immutable command capsules, and custody checks."""

    def __init__(
        self,
        root: Path,
        *,
        wall_clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.root = _private_root(root)
        self.lock_path = self.root / ".transactions.lock"
        self._wall_clock_ns = wall_clock_ns

    def is_alive(self) -> bool:
        try:
            return self.root.is_dir() and not self.root.is_symlink()
        # not a failure: a root that cannot be stat'ed is not a live store, and False is
        # the refusing direction.
        except OSError:
            return False

    def load(self, command: ActuationCommand) -> dict[str, Any] | None:
        path = _record_path(self.root, command.idempotency_key)
        with interprocess_file_lock(self.lock_path):
            if path.is_symlink():
                raise RealityActuationError(
                    "reality_actuation_transaction_symlink_refused"
                )
            if not path.exists():
                return None
            if not path.is_file():
                raise RealityActuationError(
                    "reality_actuation_transaction_type_invalid"
                )
            try:
                raw = _read_record_bytes(path)
                value = _strict_json_mapping(raw, role="transaction")
            except OSError as exc:
                raise RealityActuationError(
                    "reality_actuation_transaction_unreadable"
                ) from exc
            record = _validate_record(value)
            if (
                record["command_id"] != command.command_id
                or record["command_sha256"] != command.sha256
                or record["idempotency_key"] != command.idempotency_key
            ):
                raise RealityActuationError("reality_actuation_idempotency_collision")
            return record

    @staticmethod
    def _command_capsule_document(command: ActuationCommand) -> dict[str, Any]:
        return {
            "schema": COMMAND_CAPSULE_SCHEMA,
            "command": command.to_dict(),
            "command_sha256": command.sha256,
        }

    @staticmethod
    def _verify_capsule_mode(custody: DirectoryCustody, filename: str) -> None:
        fd = custody.open_file(filename, os.O_RDONLY)
        try:
            if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                raise RealityActuationError(
                    "reality_actuation_command_capsule_mode_invalid"
                )
        finally:
            os.close(fd)

    def _persist_command_capsule(self, command: ActuationCommand) -> None:
        filename = _command_capsule_filename(command.idempotency_key)
        payload = _canonical_bytes(self._command_capsule_document(command))
        try:
            with DirectoryCustody.acquire(self.root, create=True, private=True) as custody:
                custody.write_bytes_once(filename, payload, mode=0o600)
                self._verify_capsule_mode(custody, filename)
                existing = custody.read_bytes(
                    filename,
                    max_bytes=MAX_TRANSACTION_BYTES,
                )
        except RealityActuationError:
            raise
        except SecurePathCustodyError as exc:
            raise RealityActuationError(
                "reality_actuation_command_capsule_custody_invalid"
            ) from exc
        if existing != payload:
            raise RealityActuationError("reality_actuation_command_capsule_collision")

    @classmethod
    def _load_command_capsule(
        cls,
        custody: DirectoryCustody,
        filename: str,
    ) -> ActuationCommand:
        if not _COMMAND_CAPSULE_FILENAME.fullmatch(filename):
            raise RealityActuationError(
                "reality_actuation_command_capsule_name_invalid"
            )
        cls._verify_capsule_mode(custody, filename)
        payload = custody.read_bytes(filename, max_bytes=MAX_TRANSACTION_BYTES)
        document = _strict_json_mapping(payload, role="command_capsule")
        if set(document) != {"schema", "command", "command_sha256"} or (
            document.get("schema") != COMMAND_CAPSULE_SCHEMA
            or not isinstance(document.get("command"), Mapping)
        ):
            raise RealityActuationError(
                "reality_actuation_command_capsule_schema_invalid"
            )
        try:
            command = ActuationCommand.from_dict(document["command"])
        except (TypeError, ValueError) as exc:
            raise RealityActuationError(
                "reality_actuation_command_capsule_command_invalid"
            ) from exc
        if document.get("command_sha256") != command.sha256:
            raise RealityActuationError(
                "reality_actuation_command_capsule_digest_invalid"
            )
        if filename != _command_capsule_filename(command.idempotency_key):
            raise RealityActuationError(
                "reality_actuation_command_capsule_identity_invalid"
            )
        return command

    def create(self, command: ActuationCommand) -> dict[str, Any]:
        path = _record_path(self.root, command.idempotency_key)
        with interprocess_file_lock(self.lock_path):
            self._persist_command_capsule(command)
            if path.is_symlink():
                raise RealityActuationError(
                    "reality_actuation_transaction_symlink_refused"
                )
            if path.exists():
                existing = self.load(command)
                if existing is None:
                    raise RealityActuationError("reality_actuation_create_race")
                return existing
            now_ns = int(self._wall_clock_ns())
            body: dict[str, Any] = {
                "schema": TRANSACTION_SCHEMA,
                "command_id": command.command_id,
                "command_sha256": command.sha256,
                "idempotency_key": command.idempotency_key,
                "adapter_id": command.adapter_id,
                "channel_id": command.channel_id,
                "state": ActuationState.PLANNED.value,
                "revision": 0,
                "lease_sha256": "",
                "preparation_sha256": "",
                "actuation_receipt_sha256": "",
                "effect_receipt_sha256": "",
                "rollback_receipt_sha256": "",
                "authority_receipt_id": "",
                "created_at_ns": now_ns,
                "updated_at_ns": now_ns,
                "manual_reconciliation_required": False,
                "last_error": "",
            }
            record = {**body, "transaction_sha256": transaction_sha256(body)}
            atomic_write_bytes(path, _canonical_bytes(record), mode=0o600)
            return _validate_record(record)

    def discover_recovery_commands(
        self,
        max_transactions: int,
    ) -> tuple[tuple[ActuationCommand, ...], tuple[str, ...], tuple[str, ...], int]:
        command_names: dict[str, str] = {}
        transaction_stems: set[str] = set()
        scanned = 0
        try:
            with (
                interprocess_file_lock(self.lock_path),
                DirectoryCustody.acquire(
                    self.root,
                    create=True,
                    private=True,
                ) as custody,
            ):
                with os.scandir(custody.fileno()) as entries:
                    for entry in entries:
                        scanned += 1
                        if scanned > MAX_RECOVERY_SCAN_ENTRIES:
                            raise RealityActuationError(
                                "reality_actuation_recovery_scan_limit_exceeded"
                            )
                        name = str(entry.name)
                        command_match = _COMMAND_CAPSULE_FILENAME.fullmatch(name)
                        transaction_match = _TRANSACTION_FILENAME.fullmatch(name)
                        if command_match is not None:
                            command_names[command_match.group("stem")] = name
                        elif transaction_match is not None:
                            transaction_stems.add(transaction_match.group("stem"))
                capsule_stems = set(command_names)
                legacy = tuple(sorted(transaction_stems - capsule_stems))
                capsule_without_transaction = tuple(
                    sorted(capsule_stems - transaction_stems)
                )
                commands: list[ActuationCommand] = []
                deferred = 0
                for stem in sorted(capsule_stems & transaction_stems):
                    command = self._load_command_capsule(custody, command_names[stem])
                    record = self.load(command)
                    if record is None or ActuationState(record["state"]) not in {
                        ActuationState.PLANNED,
                        ActuationState.ADMITTED,
                        ActuationState.DISPATCHED,
                        ActuationState.EXECUTED,
                        ActuationState.INDETERMINATE,
                    }:
                        continue
                    if len(commands) < max_transactions:
                        commands.append(command)
                    else:
                        deferred += 1
        except RealityActuationError:
            raise
        except (OSError, SecurePathCustodyError) as exc:
            raise RealityActuationError(
                "reality_actuation_recovery_scan_custody_invalid"
            ) from exc
        return tuple(commands), legacy, capsule_without_transaction, deferred

    def transition(
        self,
        command: ActuationCommand,
        *,
        expected: set[ActuationState],
        state: ActuationState,
        updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = _record_path(self.root, command.idempotency_key)
        with interprocess_file_lock(self.lock_path):
            record = self.load(command)
            if record is None:
                raise RealityActuationError("reality_actuation_transaction_missing")
            current = ActuationState(record["state"])
            if current not in expected:
                if current == state:
                    return record
                raise RealityActuationError(
                    f"reality_actuation_transition_invalid:{current.value}->{state.value}"
                )
            permitted_updates = {
                "lease_sha256",
                "preparation_sha256",
                "actuation_receipt_sha256",
                "effect_receipt_sha256",
                "rollback_receipt_sha256",
                "authority_receipt_id",
                "manual_reconciliation_required",
                "last_error",
            }
            update_fields = dict(updates or {})
            if not set(update_fields).issubset(permitted_updates):
                raise RealityActuationError(
                    "reality_actuation_transition_fields_invalid"
                )
            body = {
                key: value
                for key, value in record.items()
                if key != "transaction_sha256"
            }
            body.update(update_fields)
            body["state"] = state.value
            body["revision"] = int(record["revision"]) + 1
            body["updated_at_ns"] = int(self._wall_clock_ns())
            updated = {**body, "transaction_sha256": transaction_sha256(body)}
            atomic_write_bytes(path, _canonical_bytes(updated), mode=0o600)
            return _validate_record(updated)


__all__ = [
    "COMMAND_CAPSULE_SCHEMA",
    "IDENTIFIER",
    "RECOVERY_REPORT_SCHEMA",
    "RealityActuationError",
    "RealityActuationTransactionStore",
    "TRANSACTION_SCHEMA",
    "error_evidence",
    "transaction_sha256",
]
