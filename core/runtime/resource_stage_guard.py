"""Create-once handshake for staged external resource guards."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import time
from pathlib import Path
from typing import Any, Never

MARKER_SCHEMA = "aura.resource_stage.marker.v1"
ACK_SCHEMA = "aura.resource_stage.ack.v1"
READY_STAGE = "ready_for_steady_memory_guard"
ARMED_STAGE = "steady_memory_guard_armed"
LEASE_REQUEST_SCHEMA = "aura.resource_stage.compute_lease_request.v1"
LEASE_ACK_SCHEMA = "aura.resource_stage.compute_lease_ack.v1"
LEASE_ACTIONS = frozenset({"acquire", "release"})
LEASE_STAGES = {
    "acquire": "compute_guard_armed",
    "release": "steady_memory_guard_rearmed",
}
MAX_DOCUMENT_BYTES = 16 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NONCE_RE = re.compile(r"[0-9a-f]{32}")
_WORKLOAD_RE = re.compile(r"[a-z][a-z0-9_]{0,47}")


class ResourceStageGuardError(RuntimeError):
    """A staged resource-guard artifact violated its fail-closed contract."""


def ack_path(marker_path: Path) -> Path:
    return marker_path.with_name(f"{marker_path.name}.armed.json")


def lease_request_path(marker_path: Path, *, sequence: int, action: str) -> Path:
    if type(sequence) is not int or sequence < 1 or action not in LEASE_ACTIONS:
        raise ResourceStageGuardError("compute lease path state is invalid")
    return marker_path.with_name(
        f"{marker_path.name}.lease-{sequence:06d}-{action}.json"
    )


def lease_ack_path(request_path: Path) -> Path:
    return request_path.with_name(f"{request_path.name}.ack.json")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResourceStageGuardError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Never:
    raise ResourceStageGuardError(f"non-finite JSON value: {value}")


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResourceStageGuardError("resource guard document is not canonical JSON") from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validated_parent(path: Path) -> Path:
    supplied = path.expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ResourceStageGuardError("resource guard path must be absolute and nonsymlinked")
    parent = supplied.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise ResourceStageGuardError("resource guard parent does not exist") from exc
    if resolved_parent != parent or not resolved_parent.is_dir():
        raise ResourceStageGuardError("resource guard parent must be a real directory")
    return supplied


def _write_create_once(path: Path, raw: bytes) -> None:
    destination = _validated_parent(path)
    if not raw or len(raw) > MAX_DOCUMENT_BYTES:
        raise ResourceStageGuardError("resource guard document size is invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(destination, flags, 0o600)
        created = True
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise ResourceStageGuardError("resource guard document already exists") from exc
    except OSError as exc:
        if created:
            try:
                destination.unlink()
            except OSError as unlink_exc:
                # A half-written guard document left behind will be read as a
                # real one on the next boot.
                exc.add_note(
                    f"partial guard document at {destination} could not be "
                    f"removed: {type(unlink_exc).__name__}: {unlink_exc}"
                )
        raise ResourceStageGuardError("resource guard document write failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    supplied = path.expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ResourceStageGuardError("resource guard path must be absolute and nonsymlinked")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ResourceStageGuardError("resource guard document is missing") from exc
    if resolved != supplied:
        raise ResourceStageGuardError("resource guard path changed during resolution")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_DOCUMENT_BYTES:
                raise ResourceStageGuardError("resource guard document size or type is invalid")
            raw = os.read(descriptor, MAX_DOCUMENT_BYTES + 1)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ResourceStageGuardError("resource guard document read failed") from exc
    if (
        len(raw) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ResourceStageGuardError("resource guard document changed during read")
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResourceStageGuardError("resource guard document is invalid JSON") from exc
    if not isinstance(parsed, dict) or canonical_bytes(parsed) != raw:
        raise ResourceStageGuardError("resource guard document is not canonical")
    return parsed, raw


def _valid_timestamp(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def publish_ready_marker(
    path: Path,
    *,
    target_pid: int,
    trainer_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    if type(target_pid) is not int or target_pid < 1:
        raise ResourceStageGuardError("target pid is invalid")
    if not isinstance(trainer_sha256, str) or _SHA256_RE.fullmatch(trainer_sha256) is None:
        raise ResourceStageGuardError("trainer sha256 is invalid")
    payload = {
        "schema": MARKER_SCHEMA,
        "target_pid": target_pid,
        "trainer_sha256": trainer_sha256,
        "nonce": secrets.token_hex(16),
        "stage": READY_STAGE,
        "written_at": time.time(),
    }
    raw = canonical_bytes(payload)
    _write_create_once(path, raw)
    return payload, raw


def read_ready_marker(
    path: Path,
    *,
    expected_target_pid: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    payload, raw = _read_canonical(path)
    if (
        set(payload) != {"schema", "target_pid", "trainer_sha256", "nonce", "stage", "written_at"}
        or payload.get("schema") != MARKER_SCHEMA
        or type(payload.get("target_pid")) is not int
        or int(payload["target_pid"]) < 1
        or (expected_target_pid is not None and payload["target_pid"] != expected_target_pid)
        or not isinstance(payload.get("trainer_sha256"), str)
        or _SHA256_RE.fullmatch(payload["trainer_sha256"]) is None
        or not isinstance(payload.get("nonce"), str)
        or _NONCE_RE.fullmatch(payload["nonce"]) is None
        or payload.get("stage") != READY_STAGE
        or not _valid_timestamp(payload.get("written_at"))
    ):
        raise ResourceStageGuardError("resource guard marker contract is invalid")
    return payload, raw


def publish_armed_ack(
    marker_path: Path,
    *,
    marker_raw: bytes,
    target_pid: int,
    sentinel_pid: int,
    startup_lethal_mb: float,
    steady_lethal_mb: float,
) -> tuple[Path, dict[str, Any], bytes]:
    if type(target_pid) is not int or target_pid < 1 or type(sentinel_pid) is not int or sentinel_pid < 1:
        raise ResourceStageGuardError("resource guard acknowledgement pid is invalid")
    if any(
        isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0
        for value in (startup_lethal_mb, steady_lethal_mb)
    ) or float(startup_lethal_mb) <= float(steady_lethal_mb):
        raise ResourceStageGuardError("resource guard acknowledgement limits are invalid")
    payload = {
        "schema": ACK_SCHEMA,
        "target_pid": target_pid,
        "sentinel_pid": sentinel_pid,
        "marker_sha256": sha256_bytes(marker_raw),
        "stage": ARMED_STAGE,
        "startup_lethal_mb": float(startup_lethal_mb),
        "steady_lethal_mb": float(steady_lethal_mb),
        "written_at": time.time(),
    }
    raw = canonical_bytes(payload)
    destination = ack_path(marker_path)
    _write_create_once(destination, raw)
    return destination, payload, raw


def read_armed_ack(
    marker_path: Path,
    *,
    marker_raw: bytes,
    expected_target_pid: int,
    startup_lethal_mb: float,
    steady_lethal_mb: float,
) -> tuple[dict[str, Any], bytes]:
    payload, raw = _read_canonical(ack_path(marker_path))
    if (
        set(payload)
        != {
            "schema",
            "target_pid",
            "sentinel_pid",
            "marker_sha256",
            "stage",
            "startup_lethal_mb",
            "steady_lethal_mb",
            "written_at",
        }
        or payload.get("schema") != ACK_SCHEMA
        or payload.get("target_pid") != expected_target_pid
        or type(payload.get("sentinel_pid")) is not int
        or int(payload["sentinel_pid"]) < 1
        or payload.get("marker_sha256") != sha256_bytes(marker_raw)
        or payload.get("stage") != ARMED_STAGE
        or payload.get("startup_lethal_mb") != float(startup_lethal_mb)
        or payload.get("steady_lethal_mb") != float(steady_lethal_mb)
        or not _valid_timestamp(payload.get("written_at"))
    ):
        raise ResourceStageGuardError("resource guard acknowledgement contract is invalid")
    return payload, raw


def publish_compute_lease_request(
    marker_path: Path,
    *,
    marker_raw: bytes,
    target_pid: int,
    sequence: int,
    workload: str,
    action: str,
    predecessor_ack_raw: bytes,
) -> tuple[Path, dict[str, Any], bytes]:
    marker, observed_marker_raw = read_ready_marker(
        marker_path,
        expected_target_pid=target_pid,
    )
    if observed_marker_raw != marker_raw:
        raise ResourceStageGuardError("compute lease marker binding changed")
    if (
        type(sequence) is not int
        or sequence < 1
        or not isinstance(workload, str)
        or _WORKLOAD_RE.fullmatch(workload) is None
        or action not in LEASE_ACTIONS
        or not predecessor_ack_raw
    ):
        raise ResourceStageGuardError("compute lease request state is invalid")
    payload = {
        "schema": LEASE_REQUEST_SCHEMA,
        "target_pid": target_pid,
        "trainer_sha256": marker["trainer_sha256"],
        "marker_sha256": sha256_bytes(marker_raw),
        "nonce": marker["nonce"],
        "sequence": sequence,
        "workload": workload,
        "action": action,
        "predecessor_ack_sha256": sha256_bytes(predecessor_ack_raw),
        "written_at": time.time(),
    }
    raw = canonical_bytes(payload)
    destination = lease_request_path(
        marker_path,
        sequence=sequence,
        action=action,
    )
    _write_create_once(destination, raw)
    return destination, payload, raw


def read_compute_lease_request(
    marker_path: Path,
    *,
    marker_raw: bytes,
    expected_target_pid: int,
    sequence: int,
    workload: str | None,
    action: str,
    predecessor_ack_raw: bytes,
) -> tuple[Path, dict[str, Any], bytes]:
    destination = lease_request_path(
        marker_path,
        sequence=sequence,
        action=action,
    )
    payload, raw = _read_canonical(destination)
    marker, observed_marker_raw = read_ready_marker(
        marker_path,
        expected_target_pid=expected_target_pid,
    )
    if (
        observed_marker_raw != marker_raw
        or set(payload)
        != {
            "schema",
            "target_pid",
            "trainer_sha256",
            "marker_sha256",
            "nonce",
            "sequence",
            "workload",
            "action",
            "predecessor_ack_sha256",
            "written_at",
        }
        or payload.get("schema") != LEASE_REQUEST_SCHEMA
        or payload.get("target_pid") != expected_target_pid
        or payload.get("trainer_sha256") != marker["trainer_sha256"]
        or payload.get("marker_sha256") != sha256_bytes(marker_raw)
        or payload.get("nonce") != marker["nonce"]
        or payload.get("sequence") != sequence
        or not isinstance(payload.get("workload"), str)
        or _WORKLOAD_RE.fullmatch(payload["workload"]) is None
        or (workload is not None and payload.get("workload") != workload)
        or payload.get("action") != action
        or payload.get("predecessor_ack_sha256")
        != sha256_bytes(predecessor_ack_raw)
        or not _valid_timestamp(payload.get("written_at"))
    ):
        raise ResourceStageGuardError("compute lease request contract is invalid")
    return destination, payload, raw


def publish_compute_lease_ack(
    request_path: Path,
    *,
    request_raw: bytes,
    target_pid: int,
    sentinel_pid: int,
    sequence: int,
    workload: str,
    action: str,
    active_lethal_mb: float,
) -> tuple[Path, dict[str, Any], bytes]:
    if (
        type(target_pid) is not int
        or target_pid < 1
        or type(sentinel_pid) is not int
        or sentinel_pid < 1
        or type(sequence) is not int
        or sequence < 1
        or not isinstance(workload, str)
        or _WORKLOAD_RE.fullmatch(workload) is None
        or action not in LEASE_ACTIONS
        or isinstance(active_lethal_mb, bool)
        or not math.isfinite(float(active_lethal_mb))
        or float(active_lethal_mb) <= 0.0
    ):
        raise ResourceStageGuardError("compute lease acknowledgement state is invalid")
    payload = {
        "schema": LEASE_ACK_SCHEMA,
        "target_pid": target_pid,
        "sentinel_pid": sentinel_pid,
        "request_sha256": sha256_bytes(request_raw),
        "sequence": sequence,
        "workload": workload,
        "action": action,
        "stage": LEASE_STAGES[action],
        "active_lethal_mb": float(active_lethal_mb),
        "written_at": time.time(),
    }
    raw = canonical_bytes(payload)
    destination = lease_ack_path(request_path)
    _write_create_once(destination, raw)
    return destination, payload, raw


def read_compute_lease_ack(
    request_path: Path,
    *,
    request_raw: bytes,
    expected_target_pid: int,
    sequence: int,
    workload: str,
    action: str,
    active_lethal_mb: float,
) -> tuple[dict[str, Any], bytes]:
    payload, raw = _read_canonical(lease_ack_path(request_path))
    if (
        set(payload)
        != {
            "schema",
            "target_pid",
            "sentinel_pid",
            "request_sha256",
            "sequence",
            "workload",
            "action",
            "stage",
            "active_lethal_mb",
            "written_at",
        }
        or payload.get("schema") != LEASE_ACK_SCHEMA
        or payload.get("target_pid") != expected_target_pid
        or type(payload.get("sentinel_pid")) is not int
        or int(payload["sentinel_pid"]) < 1
        or payload.get("request_sha256") != sha256_bytes(request_raw)
        or payload.get("sequence") != sequence
        or payload.get("workload") != workload
        or payload.get("action") != action
        or payload.get("stage") != LEASE_STAGES[action]
        or payload.get("active_lethal_mb") != float(active_lethal_mb)
        or not _valid_timestamp(payload.get("written_at"))
    ):
        raise ResourceStageGuardError(
            "compute lease acknowledgement contract is invalid"
        )
    return payload, raw
