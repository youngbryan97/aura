"""Crash-resumable, append-only journal for Recursive Latent Cortex campaigns.

The journal deliberately has no model or Aura runtime dependencies.  It is a
small durable state machine that campaign producers can use without making the
final evidence verifier trust mutable in-progress state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
import threading
from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Never, cast

from core.runtime.descriptor_owner import OwnsDescriptors
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError

PLAN_SCHEMA = "aura.latent_cortex.campaign_plan.v1"
PLAN_VERSION = 1
CELL_SCHEMA = "aura.latent_cortex.campaign_cell.v1"
EVENT_SCHEMA = "aura.latent_cortex.campaign_event.v1"
MANIFEST_SCHEMA = "aura.latent_cortex.campaign_manifest.v1"
MANIFEST_VERSION = 1

PLAN_EVENT = "PLAN"
STARTED = "STARTED"
ARM_RESULT = "ARM_RESULT"
VERIFIED = "VERIFIED"
COMMITTED = "COMMITTED"
FAILED = "FAILED"
ACTION_INTERVENTION_CLAIMED = "ACTION_INTERVENTION_CLAIMED"

_CELL_EVENTS = frozenset(
    {STARTED, ACTION_INTERVENTION_CLAIMED, ARM_RESULT, VERIFIED, COMMITTED, FAILED}
)
_EVENT_KEYS = frozenset(
    {
        "schema",
        "sequence",
        "plan_sha256",
        "previous_event_sha256",
        "event",
        "cell_id",
        "attempt_id",
        "payload",
        "event_sha256",
    }
)
_ZERO_SHA256 = "0" * 64
_MAX_JOURNAL_BYTES = 1024 * 1024 * 1024
_MAX_EVENT_BYTES = 16 * 1024 * 1024
_MAX_CELLS = 1_000_000

_PROCESS_LOCK_GUARD = threading.Lock()
_PROCESS_LOCKS: set[str] = set()


class CampaignJournalError(ValueError):
    """Stable fail-closed error raised for invalid journal operations."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    error = CampaignJournalError(code)
    raise error


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_int(raw: str) -> int:
    digits = raw.removeprefix("-")
    if not digits or len(digits) > 128:
        _fail("json_integer_out_of_bounds")
    return int(raw)


def _strict_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        _fail("json_non_finite_number")
    return value


def _strict_json_loads(raw: bytes, *, role: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail(f"{role}_not_utf8")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_json_key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        _fail(f"{role}_non_finite_number")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_int=_strict_int,
            parse_float=_strict_float,
            parse_constant=reject_constant,
        )
    except CampaignJournalError:
        raise
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail(f"{role}_json_invalid")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes, rejecting non-JSON and non-finite values."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail("value_not_canonical_json")


def _normalize_json(value: Any) -> Any:
    return _strict_json_loads(canonical_json_bytes(value), role="value")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _attempt_id(plan_sha256: str, cell_id: str, attempt_number: int) -> str:
    material = {
        "attempt_number": attempt_number,
        "cell_id": cell_id,
        "plan_sha256": plan_sha256,
        "schema": "aura.latent_cortex.campaign_attempt.v1",
    }
    return f"attempt-{_sha256(canonical_json_bytes(material))}"


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    """An immutable canonical campaign plan with deterministic cell identities."""

    campaign_name: str
    plan_sha256: str
    cell_ids: tuple[str, ...]
    _document_bytes: bytes = field(repr=False)
    _cell_definition_bytes: tuple[tuple[str, bytes], ...] = field(
        repr=False,
        compare=False,
    )

    @classmethod
    def build(
        cls,
        campaign_name: str,
        cells: Iterable[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CampaignPlan:
        if (
            not isinstance(campaign_name, str)
            or not campaign_name
            or campaign_name != campaign_name.strip()
            or len(campaign_name) > 512
        ):
            _fail("plan_campaign_name_invalid")

        normalized_metadata = _normalize_json({} if metadata is None else metadata)
        if not isinstance(normalized_metadata, dict):
            _fail("plan_metadata_invalid")

        normalized_cells: list[dict[str, Any]] = []
        for ordinal, definition in enumerate(cells):
            if ordinal >= _MAX_CELLS:
                _fail("plan_cell_count_exceeded")
            normalized_definition = _normalize_json(definition)
            if not isinstance(normalized_definition, dict):
                _fail("plan_cell_definition_invalid")
            identity_material = {
                "definition": normalized_definition,
                "ordinal": ordinal,
                "schema": CELL_SCHEMA,
            }
            cell_id = f"cell-{_sha256(canonical_json_bytes(identity_material))}"
            normalized_cells.append(
                {
                    "cell_id": cell_id,
                    "definition": normalized_definition,
                    "ordinal": ordinal,
                    "schema": CELL_SCHEMA,
                }
            )
        if not normalized_cells:
            _fail("plan_cells_empty")

        plan_material = {
            "campaign_name": campaign_name,
            "cells": normalized_cells,
            "metadata": normalized_metadata,
            "plan_version": PLAN_VERSION,
            "schema": PLAN_SCHEMA,
        }
        plan_sha256 = _sha256(canonical_json_bytes(plan_material))
        document = {**plan_material, "plan_sha256": plan_sha256}
        return cls(
            campaign_name=campaign_name,
            plan_sha256=plan_sha256,
            cell_ids=tuple(cell["cell_id"] for cell in normalized_cells),
            _document_bytes=canonical_json_bytes(document),
            _cell_definition_bytes=tuple(
                sorted(
                    (
                        cell["cell_id"],
                        canonical_json_bytes(cell["definition"]),
                    )
                    for cell in normalized_cells
                )
            ),
        )

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> CampaignPlan:
        normalized = _normalize_json(document)
        expected_keys = {
            "campaign_name",
            "cells",
            "metadata",
            "plan_sha256",
            "plan_version",
            "schema",
        }
        if not isinstance(normalized, dict) or set(normalized) != expected_keys:
            _fail("plan_document_invalid")
        if normalized.get("schema") != PLAN_SCHEMA or normalized.get("plan_version") != 1:
            _fail("plan_version_invalid")
        raw_cells = normalized.get("cells")
        if not isinstance(raw_cells, list):
            _fail("plan_cells_invalid")
        definitions: list[dict[str, Any]] = []
        for ordinal, cell in enumerate(raw_cells):
            if (
                not isinstance(cell, dict)
                or set(cell) != {"cell_id", "definition", "ordinal", "schema"}
                or cell.get("ordinal") != ordinal
                or cell.get("schema") != CELL_SCHEMA
                or not isinstance(cell.get("definition"), dict)
            ):
                _fail("plan_cell_invalid")
            definitions.append(cell["definition"])
        campaign_name = normalized.get("campaign_name")
        if not isinstance(campaign_name, str):
            _fail("plan_campaign_name_invalid")
        rebuilt = cls.build(
            campaign_name,
            definitions,
            metadata=normalized.get("metadata"),
        )
        if rebuilt.to_dict() != normalized:
            _fail("plan_hash_or_cell_identity_invalid")
        return rebuilt

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _strict_json_loads(self._document_bytes, role="plan"))

    def cell_definition(self, cell_id: str) -> dict[str, Any]:
        if not isinstance(cell_id, str):
            _fail("unknown_cell")
        position = bisect_left(
            self._cell_definition_bytes,
            (cell_id, b""),
        )
        if (
            position >= len(self._cell_definition_bytes)
            or self._cell_definition_bytes[position][0] != cell_id
        ):
            _fail("unknown_cell")
        definition = self._cell_definition_bytes[position][1]
        return cast(
            dict[str, Any],
            _strict_json_loads(definition, role="cell_definition"),
        )


@dataclass(frozen=True, slots=True)
class ResumeSnapshot:
    """Replay result: only committed cells are durable; all others are runnable."""

    committed_cell_ids: tuple[str, ...]
    runnable_cell_ids: tuple[str, ...]
    incomplete_cell_ids: tuple[str, ...]
    sealed_cell_ids: tuple[str, ...]
    journal_head_sha256: str


@dataclass(slots=True)
class _Attempt:
    cell_id: str
    attempt_id: str
    attempt_number: int
    state: str = STARTED
    intervention_claim_event_sha256: str | None = None
    intervention_claim: dict[str, Any] | None = None
    arm_result_event_sha256: str | None = None
    verified_event_sha256: str | None = None
    commit_event_sha256: str | None = None
    arm_result: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    commit: dict[str, Any] | None = None


@dataclass(slots=True)
class _ReplayState:
    sequence: int = -1
    event_count: int = 0
    head_sha256: str = _ZERO_SHA256
    size_bytes: int = 0
    start_counts: dict[str, int] = field(default_factory=dict)
    attempts: dict[str, _Attempt] = field(default_factory=dict)
    active_by_cell: dict[str, str] = field(default_factory=dict)
    committed_by_cell: dict[str, str] = field(default_factory=dict)


class CampaignJournal(OwnsDescriptors):
    """Single-writer append-only campaign journal."""

    def __init__(
        self,
        path: Path | str,
        plan: CampaignPlan,
        *,
        custody: DirectoryCustody | None = None,
    ) -> None:
        if not isinstance(plan, CampaignPlan):
            _fail("plan_type_invalid")
        self.path = Path(path).expanduser().absolute()
        self.plan = plan
        self._closed = True
        self._lock_fd: int | None = None
        self._lock_identity: tuple[int, int] | None = None
        self._journal_fd: int | None = None
        self._journal_payload = b""
        self._lock_key = str(self.path)
        self._process_lock_registered = False
        self._recovered_attempts: set[str] = set()
        self._custody = custody
        self._custody_relative: str | None = None

        if custody is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            try:
                relative = self.path.relative_to(custody.path)
            except ValueError:
                _fail("journal_outside_custody")
            if relative.parent != Path("."):
                custody.ensure_directory(relative.parent)
            self._custody_relative = relative.as_posix()
        self._acquire_lock()
        try:
            if custody is not None and self._custody_relative is not None:
                self._closed = False
                if custody.file_exists(self._custody_relative):
                    self._journal_payload = custody.read_bytes(
                        self._custody_relative,
                        max_bytes=_MAX_JOURNAL_BYTES,
                    )
                if not self._journal_payload:
                    self._write_genesis()
            else:
                self._journal_fd = self._open_regular_append_file(
                    self.path,
                    custody=None,
                    relative=None,
                )
                self._closed = False
                if os.fstat(self._journal_fd).st_size == 0:
                    self._write_genesis()
            self._state = self._replay()
            self._recovered_attempts = {
                attempt_id
                for attempt_id in self._state.active_by_cell.values()
                if self._state.attempts[attempt_id].state != ACTION_INTERVENTION_CLAIMED
            }
        except BaseException:  # noqa: BLE001 - resource cleanup on any exit; original re-raised
            self.close()
            raise

    @staticmethod
    def _open_regular_append_file(
        path: Path,
        *,
        custody: DirectoryCustody | None,
        relative: str | None,
    ) -> int:
        if path.is_symlink():
            _fail("journal_symlink_rejected")
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            if custody is not None and relative is not None:
                fd = custody.open_file(
                    relative,
                    flags,
                    mode=0o600,
                    create_parents=True,
                )
            else:
                fd = os.open(path, flags, 0o600)
        except (OSError, SecurePathCustodyError):
            _fail("journal_open_failed")
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            _fail("journal_not_regular_file")
        return fd

    def _acquire_lock(self) -> None:
        with _PROCESS_LOCK_GUARD:
            if self._lock_key in _PROCESS_LOCKS:
                _fail("journal_writer_locked")
            _PROCESS_LOCKS.add(self._lock_key)
            self._process_lock_registered = True

        lock_path = self.path.with_name(f"{self.path.name}.lock")
        try:
            if lock_path.is_symlink():
                _fail("journal_lock_symlink_rejected")
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            if self._custody is not None and self._custody_relative is not None:
                lock_relative = str(
                    Path(self._custody_relative).with_name(
                        f"{Path(self._custody_relative).name}.lock"
                    )
                )
                self._lock_fd = self._custody.open_file(
                    lock_relative,
                    flags,
                    mode=0o600,
                    create_parents=True,
                )
            else:
                self._lock_fd = os.open(lock_path, flags, 0o600)
            lock_stat = os.fstat(self._lock_fd)
            if (
                not stat.S_ISREG(lock_stat.st_mode)
                or lock_stat.st_uid != os.geteuid()
                or lock_stat.st_nlink != 1
            ):
                _fail("journal_lock_not_regular_file")
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                _fail("journal_writer_locked")
            self._lock_identity = (int(lock_stat.st_dev), int(lock_stat.st_ino))
            self._verify_lock_binding()
        except BaseException:  # noqa: BLE001 - resource cleanup on any exit; original re-raised
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
                self._lock_identity = None
            with _PROCESS_LOCK_GUARD:
                _PROCESS_LOCKS.discard(self._lock_key)
                self._process_lock_registered = False
            raise

    def _lock_relative_path(self) -> str | None:
        if self._custody_relative is None:
            return None
        relative = Path(self._custody_relative)
        return str(relative.with_name(f"{relative.name}.lock"))

    def _verify_lock_binding(self) -> None:
        fd = self._lock_fd
        expected = self._lock_identity
        if fd is None or expected is None:
            _fail("journal_lock_identity_drift")
        held = os.fstat(fd)
        if (
            not stat.S_ISREG(held.st_mode)
            or held.st_uid != os.geteuid()
            or held.st_nlink != 1
            or (int(held.st_dev), int(held.st_ino)) != expected
        ):
            _fail("journal_lock_identity_drift")
        entry_fd = -1
        try:
            relative = self._lock_relative_path()
            if self._custody is not None and relative is not None:
                entry_fd = self._custody.open_file(relative, os.O_RDONLY)
            else:
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                entry_fd = os.open(self.path.with_name(f"{self.path.name}.lock"), flags)
            entry = os.fstat(entry_fd)
            if (
                not stat.S_ISREG(entry.st_mode)
                or entry.st_uid != os.geteuid()
                or entry.st_nlink != 1
                or (int(entry.st_dev), int(entry.st_ino)) != expected
            ):
                _fail("journal_lock_identity_drift")
        except CampaignJournalError:
            raise
        except (OSError, SecurePathCustodyError):
            _fail("journal_lock_identity_drift")
        finally:
            if entry_fd >= 0:
                os.close(entry_fd)

    def _write_all(self, payload: bytes) -> None:
        self._verify_lock_binding()
        if self._custody is not None and self._custody_relative is not None:
            try:
                current = (
                    self._custody.read_bytes(
                        self._custody_relative,
                        max_bytes=_MAX_JOURNAL_BYTES,
                    )
                    if self._custody.file_exists(self._custody_relative)
                    else b""
                )
                if current != self._journal_payload:
                    _fail("journal_changed_during_session")
                combined = current + payload
                if len(combined) > _MAX_JOURNAL_BYTES:
                    _fail("journal_too_large")
                self._custody.atomic_write_bytes(
                    self._custody_relative,
                    combined,
                    mode=0o600,
                )
                self._journal_payload = combined
                return
            except SecurePathCustodyError:
                _fail("journal_append_failed")
        if self._journal_fd is None:
            _fail("journal_closed")
        fd = self._journal_fd
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                _fail("journal_append_failed")
            view = view[written:]
        os.fsync(fd)

    def _write_genesis(self) -> None:
        base = {
            "schema": EVENT_SCHEMA,
            "sequence": 0,
            "plan_sha256": self.plan.plan_sha256,
            "previous_event_sha256": _ZERO_SHA256,
            "event": PLAN_EVENT,
            "cell_id": None,
            "attempt_id": None,
            "payload": {"plan": self.plan.to_dict()},
        }
        record = {**base, "event_sha256": _sha256(canonical_json_bytes(base))}
        self._write_all(canonical_json_bytes(record) + b"\n")

    def _read_all(self) -> bytes:
        self._verify_lock_binding()
        if self._custody is not None and self._custody_relative is not None:
            try:
                payload = self._custody.read_bytes(
                    self._custody_relative,
                    max_bytes=_MAX_JOURNAL_BYTES,
                )
            except SecurePathCustodyError:
                _fail("journal_read_failed")
            self._journal_payload = payload
            return payload
        if self._journal_fd is None:
            _fail("journal_closed")
        fd = self._journal_fd
        size = os.fstat(fd).st_size
        if size > _MAX_JOURNAL_BYTES:
            _fail("journal_too_large")
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                _fail("journal_read_truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.fstat(fd).st_size != size:
            _fail("journal_changed_during_replay")
        return b"".join(chunks)

    def _replay(self) -> _ReplayState:
        raw = self._read_all()
        if not raw:
            _fail("journal_empty")
        if not raw.endswith(b"\n"):
            _fail("journal_torn_record")

        state = _ReplayState(size_bytes=len(raw))
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if len(line) > _MAX_EVENT_BYTES:
                _fail("journal_event_too_large")
            event_bytes = line[:-1]
            if not event_bytes:
                _fail("journal_blank_record")
            record = _strict_json_loads(event_bytes, role="journal_event")
            if not isinstance(record, dict) or set(record) != _EVENT_KEYS:
                _fail("journal_event_shape_invalid")
            if canonical_json_bytes(record) != event_bytes:
                _fail("journal_event_noncanonical")
            self._apply_replayed_record(state, record, expected_sequence=index)
        return state

    def _apply_replayed_record(
        self,
        state: _ReplayState,
        record: dict[str, Any],
        *,
        expected_sequence: int,
    ) -> None:
        event_sha256 = record.get("event_sha256")
        if not _is_sha256(event_sha256):
            _fail("journal_event_hash_invalid")
        event_sha256 = cast(str, event_sha256)
        base = {key: value for key, value in record.items() if key != "event_sha256"}
        if _sha256(canonical_json_bytes(base)) != event_sha256:
            _fail("journal_event_hash_mismatch")
        if record.get("schema") != EVENT_SCHEMA:
            _fail("journal_event_schema_invalid")
        if record.get("sequence") != expected_sequence:
            _fail("journal_sequence_drift")
        expected_previous = _ZERO_SHA256 if expected_sequence == 0 else state.head_sha256
        if record.get("previous_event_sha256") != expected_previous:
            _fail("journal_hash_chain_invalid")
        if record.get("plan_sha256") != self.plan.plan_sha256:
            _fail("journal_plan_drift")

        event = record.get("event")
        cell_id = record.get("cell_id")
        attempt_id = record.get("attempt_id")
        payload = record.get("payload")
        if expected_sequence == 0:
            if (
                event != PLAN_EVENT
                or cell_id is not None
                or attempt_id is not None
                or not isinstance(payload, dict)
                or set(payload) != {"plan"}
                or not isinstance(payload.get("plan"), dict)
            ):
                _fail("journal_genesis_invalid")
            persisted_plan = CampaignPlan.from_dict(payload["plan"])
            if persisted_plan.to_dict() != self.plan.to_dict():
                _fail("journal_plan_mismatch")
        else:
            if not isinstance(event, str) or event not in _CELL_EVENTS:
                _fail("journal_event_state_invalid")
            if not isinstance(cell_id, str) or cell_id not in self.plan.cell_ids:
                _fail("journal_cell_drift")
            if not isinstance(attempt_id, str):
                _fail("journal_attempt_invalid")
            if not isinstance(payload, dict):
                _fail("journal_event_state_invalid")
            self._apply_cell_event(state, event, cell_id, attempt_id, payload, event_sha256)

        state.sequence = expected_sequence
        state.event_count = expected_sequence + 1
        state.head_sha256 = event_sha256

    def _apply_cell_event(
        self,
        state: _ReplayState,
        event: str,
        cell_id: str,
        attempt_id: str,
        payload: dict[str, Any],
        event_sha256: str,
    ) -> None:
        if event == STARTED:
            if set(payload) != {"attempt_number"}:
                _fail("started_payload_invalid")
            attempt_number = payload.get("attempt_number")
            expected_number = state.start_counts.get(cell_id, 0) + 1
            if type(attempt_number) is not int or attempt_number != expected_number:
                _fail("journal_attempt_number_drift")
            if attempt_id != _attempt_id(self.plan.plan_sha256, cell_id, attempt_number):
                _fail("journal_attempt_identity_drift")
            if cell_id in state.committed_by_cell:
                _fail("journal_duplicate_commit")
            if cell_id in state.active_by_cell or attempt_id in state.attempts:
                _fail("journal_duplicate_attempt")
            if self._strict_execution_order() and cell_id != self._next_execution_cell(state):
                _fail("journal_execution_order_drift")
            state.start_counts[cell_id] = attempt_number
            state.attempts[attempt_id] = _Attempt(cell_id, attempt_id, attempt_number)
            state.active_by_cell[cell_id] = attempt_id
            return

        attempt = state.attempts.get(attempt_id)
        if attempt is None or attempt.cell_id != cell_id:
            _fail("journal_attempt_drift")
        if state.active_by_cell.get(cell_id) != attempt_id:
            _fail("journal_attempt_not_active")

        if event == ACTION_INTERVENTION_CLAIMED:
            if set(payload) != {
                "intervention_sha256",
                "request_payload_sha256",
                "signed_journal_head_sha256",
                "signed_journal_event_count",
            }:
                _fail("action_intervention_claim_payload_invalid")
            if (
                attempt.state != STARTED
                or not all(
                    _is_sha256(payload.get(name))
                    for name in (
                        "intervention_sha256",
                        "request_payload_sha256",
                        "signed_journal_head_sha256",
                    )
                )
                or type(payload.get("signed_journal_event_count")) is not int
                or payload["signed_journal_event_count"] != state.event_count
                or payload["signed_journal_head_sha256"] != state.head_sha256
            ):
                _fail("action_intervention_claim_invalid")
            attempt.state = ACTION_INTERVENTION_CLAIMED
            attempt.intervention_claim_event_sha256 = event_sha256
            attempt.intervention_claim = payload
        elif event == ARM_RESULT:
            if set(payload) != {"result"} or not isinstance(payload["result"], dict):
                _fail("arm_result_payload_invalid")
            required_state = (
                ACTION_INTERVENTION_CLAIMED if self._action_intervention_required() else None
            )
            if (required_state is not None and attempt.state != required_state) or (
                required_state is None
                and attempt.state not in {STARTED, ACTION_INTERVENTION_CLAIMED}
            ):
                _fail("journal_invalid_transition")
            attempt.state = ARM_RESULT
            attempt.arm_result_event_sha256 = event_sha256
            attempt.arm_result = payload["result"]
        elif event == VERIFIED:
            if set(payload) != {"verification"} or not isinstance(payload["verification"], dict):
                _fail("verified_payload_invalid")
            if attempt.state != ARM_RESULT:
                _fail("journal_invalid_transition")
            attempt.state = VERIFIED
            attempt.verified_event_sha256 = event_sha256
            attempt.verification = payload["verification"]
        elif event == COMMITTED:
            if set(payload) != {"commit"} or not isinstance(payload["commit"], dict):
                _fail("committed_payload_invalid")
            if attempt.state != VERIFIED:
                _fail("journal_invalid_transition")
            if cell_id in state.committed_by_cell:
                _fail("journal_duplicate_commit")
            attempt.state = COMMITTED
            attempt.commit_event_sha256 = event_sha256
            attempt.commit = payload["commit"]
            state.committed_by_cell[cell_id] = attempt_id
            del state.active_by_cell[cell_id]
        elif event == FAILED:
            if set(payload) != {"details", "reason"} or not isinstance(payload["details"], dict):
                _fail("failed_payload_invalid")
            reason = payload.get("reason")
            if not isinstance(reason, str) or not reason or reason != reason.strip():
                _fail("failed_reason_invalid")
            if attempt.state == ACTION_INTERVENTION_CLAIMED:
                _fail("claimed_attempt_requires_arm_result")
            attempt.state = FAILED
            del state.active_by_cell[cell_id]
        else:
            _fail("journal_event_state_invalid")

    def _assert_open(self) -> None:
        if self._closed or (self._custody is None and self._journal_fd is None):
            _fail("journal_closed")
        self._verify_lock_binding()

    def _protocol_metadata(self) -> dict[str, Any]:
        metadata = self.plan.to_dict().get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    def _action_intervention_required(self) -> bool:
        return self._protocol_metadata().get("action_intervention_required") is True

    def _strict_execution_order(self) -> bool:
        return self._protocol_metadata().get("strict_execution_order") is True

    def _next_execution_cell(self, state: _ReplayState) -> str | None:
        for cell_id in self.plan.cell_ids:
            if cell_id in state.committed_by_cell:
                continue
            active_id = state.active_by_cell.get(cell_id)
            if active_id is not None:
                active = state.attempts[active_id]
                if active.state in {STARTED, ACTION_INTERVENTION_CLAIMED}:
                    return cell_id
                continue
            if state.start_counts.get(cell_id, 0) == 0:
                return cell_id
            attempts = (
                attempt for attempt in state.attempts.values() if attempt.cell_id == cell_id
            )
            latest = max(attempts, key=lambda attempt: attempt.attempt_number, default=None)
            if latest is not None and latest.state == FAILED:
                return cell_id
        return None

    def _assert_unchanged_size(self) -> None:
        self._assert_open()
        self._verify_lock_binding()
        if self._custody is not None and self._custody_relative is not None:
            try:
                current = self._custody.read_bytes(
                    self._custody_relative,
                    max_bytes=_MAX_JOURNAL_BYTES,
                )
            except SecurePathCustodyError:
                _fail("journal_read_failed")
            if current != self._journal_payload or len(current) != self._state.size_bytes:
                _fail("journal_changed_during_session")
            return
        fd = self._journal_fd
        if fd is None:
            _fail("journal_closed")
        if os.fstat(fd).st_size != self._state.size_bytes:
            _fail("journal_changed_during_session")

    def _append_event(
        self,
        event: str,
        cell_id: str,
        attempt_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        self._assert_unchanged_size()
        normalized_payload = _normalize_json(payload)
        if not isinstance(normalized_payload, dict):
            _fail("event_payload_invalid")
        base = {
            "schema": EVENT_SCHEMA,
            "sequence": self._state.sequence + 1,
            "plan_sha256": self.plan.plan_sha256,
            "previous_event_sha256": self._state.head_sha256,
            "event": event,
            "cell_id": cell_id,
            "attempt_id": attempt_id,
            "payload": normalized_payload,
        }
        event_sha256 = _sha256(canonical_json_bytes(base))
        record = {**base, "event_sha256": event_sha256}
        line = canonical_json_bytes(record) + b"\n"
        if len(line) > _MAX_EVENT_BYTES:
            _fail("journal_event_too_large")
        self._write_all(line)
        self._state.sequence += 1
        self._state.event_count += 1
        self._state.head_sha256 = event_sha256
        self._state.size_bytes += len(line)
        return event_sha256

    def resume(self) -> ResumeSnapshot:
        """Return committed cells and mark every other cell as runnable."""

        self._assert_open()
        committed = tuple(
            cell_id for cell_id in self.plan.cell_ids if cell_id in self._state.committed_by_cell
        )
        incomplete = tuple(
            cell_id for cell_id in self.plan.cell_ids if cell_id in self._state.active_by_cell
        )
        sealed = tuple(
            cell_id
            for cell_id in self.plan.cell_ids
            if (
                cell_id in self._state.committed_by_cell
                or (
                    cell_id in self._state.active_by_cell
                    and self._state.attempts[self._state.active_by_cell[cell_id]].state
                    in {ACTION_INTERVENTION_CLAIMED, ARM_RESULT, VERIFIED}
                )
            )
        )
        runnable = tuple(cell_id for cell_id in self.plan.cell_ids if cell_id not in committed)
        return ResumeSnapshot(
            committed,
            runnable,
            incomplete,
            sealed,
            self._state.head_sha256,
        )

    def attempt_status(self, cell_id: str) -> dict[str, Any]:
        """Return durable attempt accounting for one cell without exposing mutable state."""

        self._assert_open()
        if cell_id not in self.plan.cell_ids:
            _fail("unknown_cell")
        active_id = self._state.active_by_cell.get(cell_id)
        active = self._state.attempts.get(active_id) if active_id is not None else None
        return cast(
            dict[str, Any],
            _normalize_json(
                {
                    "attempt_count": self._state.start_counts.get(cell_id, 0),
                    "active_attempt_id": active_id,
                    "active_attempt_number": active.attempt_number if active is not None else None,
                    "active_state": active.state if active is not None else None,
                    "recovered": active_id in self._recovered_attempts
                    if active_id is not None
                    else False,
                }
            ),
        )

    def start_cell(self, cell_id: str) -> str:
        """Start a cell, explicitly failing a crash-recovered partial attempt first."""

        self._assert_open()
        if cell_id not in self.plan.cell_ids:
            _fail("unknown_cell")
        if cell_id in self._state.committed_by_cell:
            _fail("cell_already_committed")
        if self._strict_execution_order() and cell_id != self._next_execution_cell(self._state):
            _fail("journal_execution_order_drift")
        active_id = self._state.active_by_cell.get(cell_id)
        if active_id is not None:
            if active_id not in self._recovered_attempts:
                _fail("cell_attempt_already_active")
            self._append_failed(
                cell_id,
                active_id,
                reason="recovered_incomplete_attempt",
                details={},
            )

        attempt_number = self._state.start_counts.get(cell_id, 0) + 1
        attempt_id = _attempt_id(self.plan.plan_sha256, cell_id, attempt_number)
        event_sha256 = self._append_event(
            STARTED,
            cell_id,
            attempt_id,
            {"attempt_number": attempt_number},
        )
        del event_sha256
        self._state.start_counts[cell_id] = attempt_number
        self._state.attempts[attempt_id] = _Attempt(cell_id, attempt_id, attempt_number)
        self._state.active_by_cell[cell_id] = attempt_id
        self._recovered_attempts.discard(active_id)
        return attempt_id

    def _active_attempt(self, cell_id: str, attempt_id: str) -> _Attempt:
        self._assert_open()
        if cell_id not in self.plan.cell_ids:
            _fail("unknown_cell")
        attempt = self._state.attempts.get(attempt_id)
        if attempt is None or attempt.cell_id != cell_id:
            _fail("attempt_not_found")
        if self._state.active_by_cell.get(cell_id) != attempt_id:
            _fail("attempt_not_active")
        return attempt

    def claim_action_intervention(
        self,
        cell_id: str,
        attempt_id: str,
        *,
        intervention_sha256: str,
        request_payload_sha256: str,
        expected_journal_head_sha256: str,
        expected_journal_event_count: int,
    ) -> str:
        """Durably reserve an active attempt before worker dispatch.

        The operation is idempotent only for the exact same signed claim.  A
        claimed attempt is sealed against crash-recovery retry until the
        runner records its result. An unresolved claim requires operator
        reconciliation; it cannot be failed into a retry while its worker may
        still execute.
        """

        attempt = self._active_attempt(cell_id, attempt_id)
        claim = {
            "intervention_sha256": intervention_sha256,
            "request_payload_sha256": request_payload_sha256,
            "signed_journal_head_sha256": expected_journal_head_sha256,
            "signed_journal_event_count": expected_journal_event_count,
        }
        if (
            not all(
                _is_sha256(claim[name])
                for name in (
                    "intervention_sha256",
                    "request_payload_sha256",
                    "signed_journal_head_sha256",
                )
            )
            or type(expected_journal_event_count) is not int
            or expected_journal_event_count < 2
        ):
            _fail("action_intervention_claim_invalid")
        if attempt.state == ACTION_INTERVENTION_CLAIMED:
            if (
                attempt.intervention_claim != claim
                or attempt.intervention_claim_event_sha256 is None
            ):
                _fail("action_intervention_claim_conflict")
            return attempt.intervention_claim_event_sha256
        if (
            attempt.state != STARTED
            or self._state.head_sha256 != expected_journal_head_sha256
            or self._state.event_count != expected_journal_event_count
        ):
            _fail("action_intervention_attempt_superseded")
        event_sha256 = self._append_event(
            ACTION_INTERVENTION_CLAIMED,
            cell_id,
            attempt_id,
            claim,
        )
        attempt.state = ACTION_INTERVENTION_CLAIMED
        attempt.intervention_claim_event_sha256 = event_sha256
        attempt.intervention_claim = claim
        self._recovered_attempts.discard(attempt_id)
        return event_sha256

    def record_arm_result(
        self,
        cell_id: str,
        attempt_id: str,
        result: Mapping[str, Any],
    ) -> str:
        attempt = self._active_attempt(cell_id, attempt_id)
        if attempt.state == ARM_RESULT:
            _fail("duplicate_arm_result")
        if self._action_intervention_required():
            if attempt.state != ACTION_INTERVENTION_CLAIMED:
                _fail("action_intervention_claim_required")
        elif attempt.state not in {STARTED, ACTION_INTERVENTION_CLAIMED}:
            _fail("invalid_transition")
        event_sha256 = self._append_event(
            ARM_RESULT,
            cell_id,
            attempt_id,
            {"result": result},
        )
        attempt.state = ARM_RESULT
        attempt.arm_result_event_sha256 = event_sha256
        attempt.arm_result = _normalize_json(result)
        return event_sha256

    def record_verified(
        self,
        cell_id: str,
        attempt_id: str,
        verification: Mapping[str, Any],
    ) -> str:
        attempt = self._active_attempt(cell_id, attempt_id)
        if attempt.state != ARM_RESULT:
            _fail("invalid_transition")
        event_sha256 = self._append_event(
            VERIFIED,
            cell_id,
            attempt_id,
            {"verification": verification},
        )
        attempt.state = VERIFIED
        attempt.verified_event_sha256 = event_sha256
        attempt.verification = _normalize_json(verification)
        return event_sha256

    def commit_cell(
        self,
        cell_id: str,
        attempt_id: str,
        commit: Mapping[str, Any] | None = None,
    ) -> str:
        if cell_id in self._state.committed_by_cell:
            _fail("duplicate_commit")
        attempt = self._active_attempt(cell_id, attempt_id)
        if attempt.state != VERIFIED:
            _fail("invalid_transition")
        event_sha256 = self._append_event(
            COMMITTED,
            cell_id,
            attempt_id,
            {"commit": {} if commit is None else commit},
        )
        attempt.state = COMMITTED
        attempt.commit_event_sha256 = event_sha256
        attempt.commit = _normalize_json({} if commit is None else commit)
        self._state.committed_by_cell[cell_id] = attempt_id
        del self._state.active_by_cell[cell_id]
        return event_sha256

    def import_committed_cell(
        self,
        cell_id: str,
        *,
        expected_attempt_id: str,
        result: Mapping[str, Any],
        verification: Mapping[str, Any],
        commit: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Idempotently finish one preverified staged cell after any crash boundary."""

        self._assert_open()
        if cell_id not in self.plan.cell_ids:
            _fail("unknown_cell")
        if not isinstance(expected_attempt_id, str) or not expected_attempt_id:
            _fail("import_attempt_id_invalid")
        normalized_result = _normalize_json(result)
        normalized_verification = _normalize_json(verification)
        normalized_commit = _normalize_json(commit)
        if not all(
            isinstance(value, dict)
            for value in (
                normalized_result,
                normalized_verification,
                normalized_commit,
            )
        ):
            _fail("import_payload_invalid")

        committed_attempt_id = self._state.committed_by_cell.get(cell_id)
        if committed_attempt_id is not None:
            attempt = self._state.attempts[committed_attempt_id]
            if (
                committed_attempt_id != expected_attempt_id
                or attempt.arm_result != normalized_result
                or attempt.verification != normalized_verification
                or attempt.commit != normalized_commit
                or attempt.arm_result_event_sha256 is None
                or attempt.verified_event_sha256 is None
                or attempt.commit_event_sha256 is None
            ):
                _fail("import_committed_cell_conflict")
            return {
                "attempt_id": committed_attempt_id,
                "arm_result_event_sha256": attempt.arm_result_event_sha256,
                "verified_event_sha256": attempt.verified_event_sha256,
                "commit_event_sha256": attempt.commit_event_sha256,
                "resumed_from_state": COMMITTED,
                "already_committed": True,
            }

        active_attempt_id = self._state.active_by_cell.get(cell_id)
        if active_attempt_id is None:
            next_attempt_number = self._state.start_counts.get(cell_id, 0) + 1
            derived_attempt_id = _attempt_id(
                self.plan.plan_sha256,
                cell_id,
                next_attempt_number,
            )
            if derived_attempt_id != expected_attempt_id:
                _fail("import_attempt_id_conflict")
            active_attempt_id = self.start_cell(cell_id)
        elif active_attempt_id != expected_attempt_id:
            _fail("import_attempt_id_conflict")

        attempt = self._state.attempts[active_attempt_id]
        resumed_from_state = attempt.state
        if attempt.state in {STARTED, ACTION_INTERVENTION_CLAIMED}:
            self.record_arm_result(
                cell_id,
                active_attempt_id,
                normalized_result,
            )
        elif attempt.state in {ARM_RESULT, VERIFIED}:
            if attempt.arm_result != normalized_result:
                _fail("import_arm_result_conflict")
        else:
            _fail("import_state_invalid")

        if attempt.state == ARM_RESULT:
            self.record_verified(
                cell_id,
                active_attempt_id,
                normalized_verification,
            )
        elif attempt.state == VERIFIED:
            if attempt.verification != normalized_verification:
                _fail("import_verification_conflict")
        else:
            _fail("import_state_invalid")

        if attempt.state != VERIFIED:
            _fail("import_state_invalid")
        commit_event_sha256 = self.commit_cell(
            cell_id,
            active_attempt_id,
            normalized_commit,
        )
        if attempt.arm_result_event_sha256 is None or attempt.verified_event_sha256 is None:
            _fail("import_receipt_incomplete")
        return {
            "attempt_id": active_attempt_id,
            "arm_result_event_sha256": attempt.arm_result_event_sha256,
            "verified_event_sha256": attempt.verified_event_sha256,
            "commit_event_sha256": commit_event_sha256,
            "resumed_from_state": resumed_from_state,
            "already_committed": False,
        }

    def import_staged_arm_result(
        self,
        cell_id: str,
        *,
        expected_attempt_id: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Idempotently import transport evidence without scoring it early."""

        self._assert_open()
        if cell_id not in self.plan.cell_ids:
            _fail("unknown_cell")
        if not isinstance(expected_attempt_id, str) or not expected_attempt_id:
            _fail("import_attempt_id_invalid")
        normalized_result = _normalize_json(result)
        if not isinstance(normalized_result, dict):
            _fail("import_payload_invalid")

        committed_attempt_id = self._state.committed_by_cell.get(cell_id)
        if committed_attempt_id is not None:
            attempt = self._state.attempts[committed_attempt_id]
            if (
                committed_attempt_id != expected_attempt_id
                or attempt.arm_result != normalized_result
                or attempt.arm_result_event_sha256 is None
            ):
                _fail("import_committed_cell_conflict")
            return {
                "attempt_id": committed_attempt_id,
                "arm_result_event_sha256": attempt.arm_result_event_sha256,
                "canonical_state": COMMITTED,
            }

        active_attempt_id = self._state.active_by_cell.get(cell_id)
        if active_attempt_id is None:
            next_attempt_number = self._state.start_counts.get(cell_id, 0) + 1
            derived_attempt_id = _attempt_id(
                self.plan.plan_sha256,
                cell_id,
                next_attempt_number,
            )
            if derived_attempt_id != expected_attempt_id:
                _fail("import_attempt_id_conflict")
            active_attempt_id = self.start_cell(cell_id)
        elif active_attempt_id != expected_attempt_id:
            _fail("import_attempt_id_conflict")

        attempt = self._state.attempts[active_attempt_id]
        if attempt.state in {STARTED, ACTION_INTERVENTION_CLAIMED}:
            self.record_arm_result(
                cell_id,
                active_attempt_id,
                normalized_result,
            )
        elif attempt.state in {ARM_RESULT, VERIFIED}:
            if attempt.arm_result != normalized_result:
                _fail("import_arm_result_conflict")
        else:
            _fail("import_state_invalid")
        if attempt.arm_result_event_sha256 is None:
            _fail("import_arm_result_missing")
        return {
            "attempt_id": active_attempt_id,
            "arm_result_event_sha256": attempt.arm_result_event_sha256,
            "canonical_state": attempt.state,
        }

    def committed_records(self) -> tuple[dict[str, Any], ...]:
        """Return immutable JSON copies of committed evidence in plan order."""

        self._assert_open()
        replayed = self._replay()
        records: list[dict[str, Any]] = []
        for cell_id in self.plan.cell_ids:
            attempt_id = replayed.committed_by_cell.get(cell_id)
            if attempt_id is None:
                continue
            attempt = replayed.attempts[attempt_id]
            if attempt.arm_result is None or attempt.verification is None or attempt.commit is None:
                _fail("campaign_commit_evidence_incomplete")
            records.append(
                _normalize_json(
                    {
                        "cell_id": cell_id,
                        "attempt_id": attempt_id,
                        "definition": self.plan.cell_definition(cell_id),
                        "result": attempt.arm_result,
                        "verification": attempt.verification,
                        "commit": attempt.commit,
                    }
                )
            )
        return tuple(records)

    def result_records(self) -> tuple[dict[str, Any], ...]:
        """Return current fsync-sealed results, including pre-verification cells."""

        self._assert_open()
        replayed = self._replay()
        records: list[dict[str, Any]] = []
        for cell_id in self.plan.cell_ids:
            attempt_id = replayed.committed_by_cell.get(cell_id)
            if attempt_id is None:
                attempt_id = replayed.active_by_cell.get(cell_id)
            if attempt_id is None:
                continue
            attempt = replayed.attempts[attempt_id]
            if attempt.state not in {ARM_RESULT, VERIFIED, COMMITTED}:
                continue
            if attempt.arm_result is None or attempt.arm_result_event_sha256 is None:
                _fail("campaign_result_evidence_incomplete")
            records.append(
                _normalize_json(
                    {
                        "arm_result_event_sha256": attempt.arm_result_event_sha256,
                        "attempt_id": attempt_id,
                        "cell_id": cell_id,
                        "definition": self.plan.cell_definition(cell_id),
                        "result": attempt.arm_result,
                        "state": attempt.state,
                        "verification": attempt.verification,
                    }
                )
            )
        return tuple(records)

    def _append_failed(
        self,
        cell_id: str,
        attempt_id: str,
        *,
        reason: str,
        details: Mapping[str, Any],
    ) -> str:
        attempt = self._active_attempt(cell_id, attempt_id)
        if attempt.state == ACTION_INTERVENTION_CLAIMED:
            _fail("claimed_attempt_requires_arm_result")
        if not isinstance(reason, str) or not reason or reason != reason.strip():
            _fail("failed_reason_invalid")
        event_sha256 = self._append_event(
            FAILED,
            cell_id,
            attempt_id,
            {"details": details, "reason": reason},
        )
        attempt.state = FAILED
        del self._state.active_by_cell[cell_id]
        self._recovered_attempts.discard(attempt_id)
        return event_sha256

    def fail_cell(
        self,
        cell_id: str,
        attempt_id: str,
        *,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> str:
        return self._append_failed(
            cell_id,
            attempt_id,
            reason=reason,
            details={} if details is None else details,
        )

    def finalize(self, manifest_path: Path | str) -> dict[str, Any]:
        """Atomically publish a manifest only for the exact complete plan cell set."""

        self._assert_open()
        replayed = self._replay()
        if replayed.head_sha256 != self._state.head_sha256:
            _fail("journal_changed_during_session")
        expected_cells = set(self.plan.cell_ids)
        if set(replayed.committed_by_cell) != expected_cells:
            _fail("campaign_incomplete")

        cells: list[dict[str, Any]] = []
        for cell_id in self.plan.cell_ids:
            attempt_id = replayed.committed_by_cell[cell_id]
            attempt = replayed.attempts[attempt_id]
            if not all(
                (
                    attempt.arm_result_event_sha256,
                    attempt.verified_event_sha256,
                    attempt.commit_event_sha256,
                )
            ):
                _fail("campaign_commit_evidence_incomplete")
            cells.append(
                {
                    "arm_result_event_sha256": attempt.arm_result_event_sha256,
                    "attempt_id": attempt_id,
                    "cell_id": cell_id,
                    "commit_event_sha256": attempt.commit_event_sha256,
                    "verified_event_sha256": attempt.verified_event_sha256,
                }
            )

        manifest_material = {
            "schema": MANIFEST_SCHEMA,
            "manifest_version": MANIFEST_VERSION,
            "plan_sha256": self.plan.plan_sha256,
            "journal_head_sha256": replayed.head_sha256,
            "journal_event_count": replayed.event_count,
            "journal_size_bytes": replayed.size_bytes,
            "cell_count": len(cells),
            "cells": cells,
        }
        manifest = {
            **manifest_material,
            "manifest_sha256": _sha256(canonical_json_bytes(manifest_material)),
        }
        target = Path(manifest_path).expanduser().absolute()
        if target in {self.path, self.path.with_name(f"{self.path.name}.lock")}:
            _fail("manifest_path_conflicts_with_journal")
        payload = canonical_json_bytes(manifest) + b"\n"
        if self._custody is not None:
            try:
                relative = target.relative_to(self._custody.path).as_posix()
            except ValueError:
                _fail("manifest_outside_custody")
            try:
                if not self._custody.write_bytes_once(relative, payload, mode=0o600):
                    existing = self._custody.read_bytes(
                        relative,
                        max_bytes=max(len(payload), 1),
                    )
                    if existing != payload:
                        _fail("manifest_already_exists_with_different_content")
            except SecurePathCustodyError:
                _fail("manifest_publish_failed")
        else:
            self._atomic_publish(target, payload)
        return manifest

    @staticmethod
    def _atomic_publish(target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            _fail("manifest_symlink_rejected")
        if target.exists():
            try:
                existing = target.read_bytes()
            except OSError:
                _fail("manifest_read_failed")
            if existing == payload:
                return
            _fail("manifest_already_exists_with_different_content")

        fd = -1
        temporary = ""
        try:
            fd, temporary = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
            )
            os.fchmod(fd, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    _fail("manifest_write_failed")
                view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(temporary, target)
            temporary = ""
            directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except CampaignJournalError:
            raise
        except OSError:
            _fail("manifest_publish_failed")
        finally:
            if fd >= 0:
                os.close(fd)
            if temporary:
                try:
                    os.unlink(temporary)
                # not a failure: a file that is already gone is the state this is reaching.
                except OSError:
                    pass

    def close(self) -> None:
        if self._journal_fd is not None:
            os.close(self._journal_fd)
            self._journal_fd = None
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None
                self._lock_identity = None
        if self._process_lock_registered:
            with _PROCESS_LOCK_GUARD:
                _PROCESS_LOCKS.discard(self._lock_key)
                self._process_lock_registered = False
        self._closed = True

    def __enter__(self) -> CampaignJournal:
        self._assert_open()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        # not a failure: the comment below says it: finalizers must never raise, and the
        # explicit close is the path that reports.
        except (OSError, RuntimeError, ValueError):
            # Finalizers must never raise; the explicit close/context-manager
            # paths are where real close errors surface.
            pass


__all__ = [
    "ACTION_INTERVENTION_CLAIMED",
    "ARM_RESULT",
    "COMMITTED",
    "FAILED",
    "STARTED",
    "VERIFIED",
    "CampaignJournal",
    "CampaignJournalError",
    "CampaignPlan",
    "ResumeSnapshot",
    "canonical_json_bytes",
]
