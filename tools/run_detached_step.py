#!/usr/bin/env python3
"""Run one crash-observable step independently of the launching terminal.

The launcher spawns a fresh session before returning, so the supervisor is
independent of the caller's terminal teardown without forking a potentially
multithreaded Python process. The supervisor starts the
target in its own process group, publishes atomic heartbeat/status artifacts,
enforces one wall-clock timeout, and writes exactly one terminal receipt. It
never restarts a failed target: scientific verdicts and training failures are
results to inspect, not conditions that should silently create another run.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import fcntl
import hashlib
import hmac
import json
import math
import os
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Never

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignJournalError,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    CAMPAIGN_RUNNER,
    CampaignTrustError,
    VerifiedCampaignTrustPolicy,
    prepare_role_signature_request,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.worker_origin import (  # noqa: E402
    WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    ZERO_SHA256,
    WorkerOriginError,
    compute_allowed_cell_digest,
    validate_worker_authorization_payload,
    verify_worker_authorization,
    verify_worker_lifecycle_event_origin,
)
from core.runtime import detached_subprocess_broker as broker_protocol  # noqa: E402
from core.runtime.detached_worker_origin import (  # noqa: E402
    DetachedWorkerOriginAuthority,
    DetachedWorkerOriginError,
    DetachedWorkerOriginState,
)
from core.runtime.detached_worker_origin_channel import (  # noqa: E402
    WORKER_ORIGIN_FD_ENV,
    WORKER_ORIGIN_SESSION_ENV,
    DetachedWorkerOriginChannelError,
    DetachedWorkerOriginChannelServer,
    create_worker_origin_socketpair,
)
from core.runtime.secure_path_custody import (  # noqa: E402
    DirectoryCustody,
    SecurePathCustodyError,
    validate_directory_identity,
)

SCHEMA_PREFIX = "aura.detached_step"
PLAN_FILE = "detached_plan.json"
STATUS_FILE = "detached_status.json"
RECEIPT_FILE = "detached_receipt.json"
ATTEMPTS_FILE = "detached_attempts.jsonl"
LOG_FILE = "detached.log"
LOCK_FILE = ".detached.lock"
CONTROL_SOCKET_PREFIX = "aura-detached-control"
WORKER_ORIGIN_POLICY_SCHEMA = f"{SCHEMA_PREFIX}.worker_origin_policy.v1"
WORKER_ORIGIN_LIFECYCLE_ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}.worker_origin_lifecycle_artifact.v1"
WORKER_ORIGIN_QUARANTINE_RECEIPT_SCHEMA = f"{SCHEMA_PREFIX}.worker_origin_quarantine_receipt.v1"
_MAX_WORKER_ORIGIN_CELLS = 16_384
_MAX_WORKER_ORIGIN_TRUST_POLICY_BYTES = 1024 * 1024
_MAX_WORKER_ORIGIN_TRUST_ROOT_BYTES = 64 * 1024
_POLL_S = 1.0
_TERM_GRACE_S = 5.0
_IDENTITY_GRACE_S = 10.0
_ACTIVE_RUN_CUSTODIES: dict[Path, DirectoryCustody] = {}


def _custody_for_path(path: Path) -> tuple[DirectoryCustody, str] | None:
    absolute = path.expanduser().absolute()
    matches: list[tuple[int, DirectoryCustody, str]] = []
    for root, custody in _ACTIVE_RUN_CUSTODIES.items():
        try:
            relative = absolute.relative_to(root).as_posix()
        except ValueError:
            continue
        if relative and relative != ".":
            matches.append((len(root.parts), custody, relative))
    if not matches:
        return None
    _depth, custody, relative = max(matches, key=lambda item: item[0])
    return custody, relative


@contextmanager
def _run_directory_custody(
    path: Path,
    *,
    create: bool,
    expected_identity: dict[str, int] | None = None,
) -> Iterator[DirectoryCustody]:
    absolute = path.expanduser().absolute()
    custody = DirectoryCustody.acquire(
        absolute,
        create=create,
        expected_identity=expected_identity,
        private=True,
    )
    prior = _ACTIVE_RUN_CUSTODIES.get(custody.path)
    _ACTIVE_RUN_CUSTODIES[custody.path] = custody
    try:
        yield custody
    finally:
        if _ACTIVE_RUN_CUSTODIES.get(custody.path) is custody:
            if prior is None:
                del _ACTIVE_RUN_CUSTODIES[custody.path]
            else:
                _ACTIVE_RUN_CUSTODIES[custody.path] = prior
        custody.close()


_HANDOFF_WAIT_S = 5.0
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DARWIN_SANDBOX = Path("/usr/bin/sandbox-exec")
_DARWIN_CAFFEINATE = Path("/usr/bin/caffeinate")
_NO_FORK_SANDBOX_PROFILE = "(version 1) (allow default) (deny process-fork)"
_PRECONTAINED_SANDBOX_MODE = "precontained-sandbox"
_SUPERVISOR_SANDBOX_MODE = "supervisor-no-fork"
_CONTAINMENT_MODES = frozenset({_PRECONTAINED_SANDBOX_MODE, _SUPERVISOR_SANDBOX_MODE})
_MAX_SANDBOX_PROFILE_BYTES = 1024 * 1024
_REQUIRED_PRECONTAINED_PROFILE_MARKERS = (
    "(version 1)",
    "(deny default)",
    "(deny network*)",
    "(deny process-fork)",
)
_SOURCE_SUFFIXES = frozenset({".json", ".py", ".pyi", ".sb", ".sh", ".toml", ".yaml", ".yml"})
_EXECUTABLE_SOURCE_SUFFIXES = frozenset({".py", ".pyi", ".sh"})
_SAFE_ENVIRONMENT_KEYS = (
    "AURA_DATA_DIR",
    "AURA_HOME",
    "AURA_LATENT_CORTEX",
    "AURA_MODEL_PATH",
    "AURA_MODEL_LANE_STATE_PATH",
    "AURA_RLC_FULL_SHA",
    "COMMAND_MODE",
    "HF_HOME",
    "HOME",
    "HUGGINGFACE_HUB_CACHE",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "MallocNanoZone",
    "MLX_METAL_CACHE_DIR",
    "MLX_METAL_JIT",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "SHELL",
    "TMPDIR",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_CACHE",
    "USER",
    "VIRTUAL_ENV",
)


@dataclass(frozen=True)
class ProcessObservation:
    state: Literal["alive", "dead", "unknown"]
    token: str = ""
    process_group_id: int = 0
    executable: str = ""


@dataclass
class ActiveBrokerWorker:
    request_id: str
    policy_sha256: str
    command_sha256: str
    process: subprocess.Popen[Any]
    process_group_id: int
    start_token: str
    containment_token: str
    response_token: str
    reply_path: Path
    started_at: float
    started_monotonic_ns: int
    deadline_ns: int
    log: Any
    worker_origin: PreparedBrokerWorkerOrigin | None = None
    worker_origin_server: DetachedWorkerOriginChannelServer | None = None
    timed_out: bool = False


@dataclass
class PreparedBrokerWorkerOrigin:
    policy_sha256: str
    authority: DetachedWorkerOriginAuthority
    request: dict[str, Any]
    request_path: Path
    payload_path: Path
    attestation_path: Path
    lifecycle_path: Path
    policy: VerifiedCampaignTrustPolicy
    authorized: bool = False
    finalized: bool = False


class _ProcBSDInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


class DetachedStepError(RuntimeError):
    pass


class BrokerRequestError(DetachedStepError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise DetachedStepError(f"execution artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
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
        raise DetachedStepError(f"execution artifact changed while hashing: {path}")
    return digest.hexdigest()


def _git_root(cwd: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    candidate = Path(result.stdout.strip()).resolve(strict=True)
    return candidate if candidate.is_dir() else None


def _normalized_excluded_roots(paths: Iterable[Path]) -> tuple[Path, ...]:
    roots: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve(strict=False)
        if not resolved.is_absolute() or resolved == resolved.parent:
            raise DetachedStepError("execution exclusion root is invalid")
        roots.add(resolved)
    return tuple(sorted(roots, key=lambda value: os.fsencode(value)))


def _is_excluded(path: Path, excluded_roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in excluded_roots)


def _git_tracked_paths(
    root: Path,
    *,
    excluded_roots: tuple[Path, ...] = (),
) -> list[Path]:
    try:
        tracked_result = subprocess.run(
            ["/usr/bin/git", "-C", str(root), "ls-files", "-z", "--cached"],
            check=True,
            capture_output=True,
            timeout=30.0,
        )
        untracked_result = subprocess.run(
            [
                "/usr/bin/git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--others",
                "--exclude-standard",
            ],
            check=True,
            capture_output=True,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DetachedStepError(
            f"could not enumerate Git-tracked execution source: {root}"
        ) from exc
    paths: set[Path] = set()
    raw_paths = [(raw, True) for raw in tracked_result.stdout.split(b"\0")]
    raw_paths.extend((raw, False) for raw in untracked_result.stdout.split(b"\0"))
    for raw, tracked in raw_paths:
        if not raw:
            continue
        try:
            relative = Path(os.fsdecode(raw))
        except UnicodeDecodeError as exc:
            raise DetachedStepError("Git-tracked execution path is not decodable") from exc
        if relative.is_absolute() or ".." in relative.parts:
            raise DetachedStepError(f"unsafe Git-tracked execution path: {relative}")
        absolute = root / relative
        if _is_excluded(absolute, excluded_roots):
            if tracked:
                raise DetachedStepError("execution exclusion contains Git-tracked source")
            continue
        if tracked or relative.suffix.lower() in _EXECUTABLE_SOURCE_SUFFIXES:
            paths.add(relative)
    return sorted(paths, key=lambda value: os.fsencode(value))


def _fingerprint_paths(root: Path, relative_paths: list[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for relative in relative_paths:
        path = root / relative
        entry: dict[str, Any]
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            entry = {"path": str(relative), "kind": "missing"}
        else:
            if stat.S_ISLNK(metadata.st_mode):
                entry = {
                    "path": str(relative),
                    "kind": "symlink",
                    "target": os.readlink(path),
                }
            elif stat.S_ISREG(metadata.st_mode):
                content_sha = _sha256_file(path)
                entry = {
                    "path": str(relative),
                    "kind": "file",
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "size": metadata.st_size,
                    "sha256": content_sha,
                }
                total_bytes += metadata.st_size
                file_count += 1
            elif stat.S_ISDIR(metadata.st_mode):
                entry = {"path": str(relative), "kind": "gitlink_or_directory"}
            else:
                raise DetachedStepError(f"unsupported execution source artifact: {path}")
        digest.update(_canonical_bytes(entry))
        digest.update(b"\n")
    return {
        "tree_sha256": digest.hexdigest(),
        "entry_count": len(relative_paths),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _fingerprint_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    return {
        "path": str(resolved),
        "kind": "file",
        "size": metadata.st_size,
        "sha256": _sha256_file(resolved),
    }


def _source_tree_paths(
    root: Path,
    *,
    excluded_roots: tuple[Path, ...] = (),
) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(
            part in {".git", ".mypy_cache", ".pytest_cache", "__pycache__"}
            for part in relative.parts
        ):
            continue
        if _is_excluded(path, excluded_roots):
            continue
        if path.is_file() and path.suffix.lower() in _EXECUTABLE_SOURCE_SUFFIXES:
            paths.append(relative)
    return sorted(paths, key=lambda value: os.fsencode(value))


def _source_file_arguments(command: list[str], cwd: Path) -> list[Path]:
    paths: set[Path] = set()
    for argument in command[1:]:
        candidate_value = (
            argument.split("=", 1)[1] if argument.startswith("--") and "=" in argument else argument
        )
        if not candidate_value or candidate_value.startswith("-"):
            continue
        candidate = Path(candidate_value).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if resolved.is_file() and resolved.suffix.lower() in _SOURCE_SUFFIXES:
            paths.add(resolved)
    return sorted(paths, key=lambda value: os.fsencode(value))


def _build_execution_manifest(
    command: list[str],
    cwd: Path,
    *,
    excluded_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    exclusions = _normalized_excluded_roots(excluded_roots)
    roots: list[dict[str, Any]] = [_fingerprint_file(Path(command[0]))]
    source_paths = _source_file_arguments(command, cwd)
    executable_source_paths = tuple(
        path
        for path in source_paths
        if path.suffix.lower() in _EXECUTABLE_SOURCE_SUFFIXES
    )
    if any(_is_excluded(path, exclusions) for path in executable_source_paths):
        raise DetachedStepError("execution exclusion contains target source")
    git_roots: set[Path] = set()
    for candidate in [cwd, *(path.parent for path in source_paths)]:
        if (root := _git_root(candidate)) is not None:
            git_roots.add(root)
    for raw_path in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        try:
            candidate = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if candidate.is_dir() and (root := _git_root(candidate)) is not None:
            git_roots.add(root)
    if "-m" in command[1:3] and not git_roots:
        raise DetachedStepError(
            "Python -m execution requires a Git-tracked working or PYTHONPATH source root"
        )
    for git_root in sorted(git_roots, key=lambda value: os.fsencode(value)):
        if _is_excluded(git_root, exclusions):
            raise DetachedStepError("execution exclusion contains a source root")
        tracked = _git_tracked_paths(git_root, excluded_roots=exclusions)
        roots.append(
            {
                "path": str(git_root),
                "kind": "git_tracked_tree",
                **_fingerprint_paths(git_root, tracked),
            }
        )
    source_tree_roots: set[Path] = set()
    for source_path in source_paths:
        if source_path.suffix.lower() in _EXECUTABLE_SOURCE_SUFFIXES:
            if any(source_path.is_relative_to(root) for root in git_roots):
                continue
            source_tree_roots.add(source_path.parent)
        else:
            # Untracked data/config artifacts are not executable source and
            # must not make unrelated live evidence writers invalidate the
            # whole repository. Explicit command inputs remain exact-bound.
            roots.append(_fingerprint_file(source_path))
    for source_root in sorted(source_tree_roots, key=lambda value: os.fsencode(value)):
        roots.append(
            {
                "path": str(source_root),
                "kind": "source_tree",
                **_fingerprint_paths(
                    source_root,
                    _source_tree_paths(source_root, excluded_roots=exclusions),
                ),
            }
        )
    body = {
        "schema": f"{SCHEMA_PREFIX}.execution_manifest.v1",
        "excluded_roots": [str(path) for path in exclusions],
        "roots": roots,
    }
    return {**body, "manifest_sha256": _sha256(body)}


def _verify_execution_manifest_structure(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise DetachedStepError("execution manifest is missing")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema") != f"{SCHEMA_PREFIX}.execution_manifest.v1"
        or manifest.get("manifest_sha256") != _sha256(body)
        or not isinstance(manifest.get("excluded_roots"), list)
        or not isinstance(manifest.get("roots"), list)
        or not manifest["roots"]
    ):
        raise DetachedStepError("execution manifest binding is invalid")
    raw_exclusions = manifest["excluded_roots"]
    if any(not isinstance(value, str) or not value for value in raw_exclusions):
        raise DetachedStepError("execution manifest exclusion is invalid")
    exclusions = _normalized_excluded_roots(Path(value) for value in raw_exclusions)
    if [str(path) for path in exclusions] != raw_exclusions:
        raise DetachedStepError("execution manifest exclusion is not canonical")
    return manifest


def _refresh_execution_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    exclusions = _normalized_excluded_roots(
        Path(value) for value in manifest.get("excluded_roots", [])
    )
    roots: list[dict[str, Any]] = []
    for root in manifest["roots"]:
        if not isinstance(root, dict):
            raise DetachedStepError("execution manifest root is invalid")
        path = Path(str(root.get("path") or ""))
        kind = root.get("kind")
        if not path.is_absolute():
            raise DetachedStepError("execution manifest root must be absolute")
        if kind == "file":
            roots.append(_fingerprint_file(path))
        elif kind == "git_tracked_tree":
            tracked = _git_tracked_paths(path, excluded_roots=exclusions)
            roots.append(
                {
                    "path": str(path),
                    "kind": kind,
                    **_fingerprint_paths(path, tracked),
                }
            )
        elif kind == "source_tree":
            roots.append(
                {
                    "path": str(path),
                    "kind": kind,
                    **_fingerprint_paths(
                        path,
                        _source_tree_paths(path, excluded_roots=exclusions),
                    ),
                }
            )
        else:
            raise DetachedStepError(f"unsupported execution manifest root kind: {kind}")
    body = {
        "schema": f"{SCHEMA_PREFIX}.execution_manifest.v1",
        "excluded_roots": [str(path) for path in exclusions],
        "roots": roots,
    }
    return {**body, "manifest_sha256": _sha256(body)}


def _verify_execution_manifest_current(manifest: Any) -> None:
    expected = _verify_execution_manifest_structure(manifest)
    current = _refresh_execution_manifest(expected)
    if current != expected:
        raise DetachedStepError("execution source changed after the detached plan was frozen")


def _frozen_environment() -> dict[str, str]:
    environment = {
        key: value for key in _SAFE_ENVIRONMENT_KEYS if (value := os.environ.get(key)) is not None
    }
    environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
    environment.setdefault("LANG", "C.UTF-8")
    return dict(sorted(environment.items()))


def _resolve_command(command: list[str], cwd: Path, environment: dict[str, str]) -> list[str]:
    executable = command[0]
    if "/" in executable:
        candidate = Path(executable).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
    else:
        located = shutil.which(executable, path=environment["PATH"])
        if located is None:
            raise DetachedStepError(f"command executable is unavailable: {executable}")
        candidate = Path(located)
    # Preserve the final launcher component. Python uses a virtualenv
    # launcher's location to discover pyvenv.cfg; resolving that symlink here
    # silently changes the interpreter environment even when the binary bytes
    # are identical. The binding below still freezes the resolved target.
    launcher = candidate.parent.resolve(strict=True) / candidate.name
    resolved_target = launcher.resolve(strict=True)
    if not resolved_target.is_file() or not os.access(launcher, os.X_OK):
        raise DetachedStepError(f"command executable is not an executable file: {launcher}")
    return [str(launcher), *command[1:]]


def _launcher_binding(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        raise DetachedStepError("command launcher path must be absolute")
    try:
        launcher_stat = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DetachedStepError(f"command launcher is unavailable: {path}") from exc
    if not (stat.S_ISREG(launcher_stat.st_mode) or stat.S_ISLNK(launcher_stat.st_mode)):
        raise DetachedStepError(f"command launcher type is invalid: {path}")
    if not resolved.is_file() or not os.access(path, os.X_OK):
        raise DetachedStepError(f"command launcher is not executable: {path}")
    pyvenv_path = path.parent.parent / "pyvenv.cfg"
    pyvenv: dict[str, Any] | None = None
    if pyvenv_path.exists() or pyvenv_path.is_symlink():
        if pyvenv_path.is_symlink() or not pyvenv_path.is_file():
            raise DetachedStepError("command launcher pyvenv.cfg is unsafe")
        pyvenv = {
            "path": str(pyvenv_path),
            "sha256": _sha256_file(pyvenv_path),
            "size": pyvenv_path.stat().st_size,
        }
    body = {
        "schema": f"{SCHEMA_PREFIX}.launcher_binding.v1",
        "invocation_path": str(path),
        "invocation_kind": "symlink" if stat.S_ISLNK(launcher_stat.st_mode) else "file",
        "invocation_mode": stat.S_IMODE(launcher_stat.st_mode),
        "symlink_target": os.readlink(path) if stat.S_ISLNK(launcher_stat.st_mode) else None,
        "resolved_path": str(resolved),
        "resolved_sha256": _sha256_file(resolved),
        "pyvenv": pyvenv,
    }
    return {**body, "binding_sha256": _sha256(body)}


def _verify_launcher_binding(binding: Any, path: Path) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise DetachedStepError("command launcher binding is missing")
    current = _launcher_binding(path)
    if binding != current:
        raise DetachedStepError("command launcher binding changed")
    return current


def _fsync_directory(path: Path) -> None:
    absolute = path.expanduser().absolute()
    root_custody = _ACTIVE_RUN_CUSTODIES.get(absolute)
    if root_custody is not None:
        root_custody.fsync()
        return
    bound = _custody_for_path(path)
    if bound is not None:
        custody, relative = bound
        descriptor = custody.open_directory(relative)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, value: Any, *, replace: bool = True) -> None:
    _atomic_write_bytes(path, _canonical_bytes(value) + b"\n", replace=replace)


def _atomic_write_bytes(path: Path, payload: bytes, *, replace: bool = True) -> None:
    bound = _custody_for_path(path)
    if bound is not None:
        custody, relative = bound
        try:
            if replace:
                custody.atomic_write_bytes(relative, payload, mode=0o600)
            elif not custody.write_bytes_once(relative, payload, mode=0o600):
                raise DetachedStepError(f"artifact already exists: {path}")
        except SecurePathCustodyError as exc:
            raise DetachedStepError(f"custodied artifact write failed: {path}") from exc
        return
    if path.is_symlink():
        raise DetachedStepError(f"symlink artifact rejected: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise DetachedStepError(f"short artifact write: {temporary}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    except FileExistsError as exc:
        raise DetachedStepError(f"artifact already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        bound = _custody_for_path(path)
        if bound is not None:
            custody, relative = bound
            payload = custody.read_bytes(relative, max_bytes=256 * 1024 * 1024)
            value = json.loads(payload)
        else:
            if path.is_symlink() or not path.is_file():
                raise DetachedStepError(f"artifact is unavailable: {path}")
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, SecurePathCustodyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetachedStepError(f"artifact is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise DetachedStepError(f"artifact must contain an object: {path}")
    return value


def _read_json_artifact_with_digest(path: Path, *, max_bytes: int) -> tuple[dict[str, Any], str]:
    bound = _custody_for_path(path)
    if bound is not None:
        custody, relative = bound
        try:
            descriptor = custody.open_file(relative, os.O_RDONLY)
        except SecurePathCustodyError as exc:
            raise DetachedStepError(f"evidence artifact is unavailable: {path}") from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > max_bytes
            ):
                raise DetachedStepError(f"evidence artifact ownership or size is invalid: {path}")
            payload = os.read(descriptor, before.st_size + 1)
            after = os.fstat(descriptor)
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
            ) or len(payload) != before.st_size:
                raise DetachedStepError(f"evidence artifact changed while reading: {path}")
        finally:
            os.close(descriptor)
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DetachedStepError(f"evidence artifact is invalid: {path}") from exc
        if not isinstance(value, dict):
            raise DetachedStepError(f"evidence artifact must contain an object: {path}")
        return value, hashlib.sha256(payload).hexdigest()
    if path.is_symlink() or not path.is_file():
        raise DetachedStepError(f"evidence artifact is unavailable: {path}")
    before = path.stat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > max_bytes
    ):
        raise DetachedStepError(f"evidence artifact ownership or size is invalid: {path}")
    payload = path.read_bytes()
    after = path.stat()
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
        raise DetachedStepError(f"evidence artifact changed while reading: {path}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetachedStepError(f"evidence artifact is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise DetachedStepError(f"evidence artifact must contain an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DetachedStepError("canonical JSON artifact contains a duplicate key")
        value[key] = item
    return value


def _read_stable_private_bytes(path: Path, *, max_bytes: int, role: str) -> bytes:
    bound = _custody_for_path(path)
    if bound is not None:
        custody, relative = bound
        try:
            descriptor = custody.open_file(relative, os.O_RDONLY)
        except SecurePathCustodyError as exc:
            raise DetachedStepError(f"{role} is unavailable: {path}") from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or before.st_mode & 0o022
                or before.st_size <= 0
                or before.st_size > max_bytes
            ):
                raise DetachedStepError(f"{role} ownership, mode, or size is invalid: {path}")
            payload = os.read(descriptor, before.st_size + 1)
            after = os.fstat(descriptor)
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
            ) or len(payload) != before.st_size:
                raise DetachedStepError(f"{role} changed while reading: {path}")
            return payload
        finally:
            os.close(descriptor)
    if path.is_symlink() or not path.is_file():
        raise DetachedStepError(f"{role} is unavailable: {path}")
    before = path.stat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or before.st_mode & 0o022
        or before.st_size <= 0
        or before.st_size > max_bytes
    ):
        raise DetachedStepError(f"{role} ownership, mode, or size is invalid: {path}")
    payload = path.read_bytes()
    after = path.stat()
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
        raise DetachedStepError(f"{role} changed while reading: {path}")
    return payload


def _read_canonical_private_json(
    path: Path,
    *,
    max_bytes: int,
    role: str,
) -> dict[str, Any]:
    payload = _read_stable_private_bytes(path, max_bytes=max_bytes, role=role)

    def reject_constant(_value: str) -> Never:
        raise DetachedStepError(f"{role} contains a non-finite number")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_json_pairs,
            parse_constant=reject_constant,
        )
    except DetachedStepError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DetachedStepError(f"{role} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DetachedStepError(f"{role} must contain an object: {path}")
    try:
        canonical = canonical_json_bytes(value)
    except CampaignJournalError as exc:
        raise DetachedStepError(f"{role} is not canonical JSON: {path}") from exc
    if payload not in {canonical, canonical + b"\n"}:
        raise DetachedStepError(f"{role} bytes are not canonical: {path}")
    return value


def _sha256_identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DetachedStepError(f"{role} must be a lowercase SHA-256 identifier")
    return value


def _positive_integer(value: Any, *, role: str, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or (maximum is not None and value > maximum)
    ):
        raise DetachedStepError(f"{role} must be a bounded positive integer")
    return int(value)


def _ensure_private_directory(path: Path) -> None:
    bound = _custody_for_path(path)
    if bound is not None:
        custody, relative = bound
        try:
            custody.ensure_directory(relative)
        except SecurePathCustodyError as exc:
            raise DetachedStepError(f"worker-origin artifact directory is invalid: {path}") from exc
        return
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise DetachedStepError(f"worker-origin artifact directory is invalid: {path}")
    else:
        parent = path.parent
        if parent.is_symlink() or not parent.is_dir():
            raise DetachedStepError(f"worker-origin artifact parent is unavailable: {parent}")
        parent_stat = parent.stat()
        if parent_stat.st_uid != os.geteuid() or parent_stat.st_mode & 0o022:
            raise DetachedStepError(
                f"worker-origin artifact parent permissions are unsafe: {parent}"
            )
        path.mkdir(mode=0o700)
        _fsync_directory(parent)
    directory_stat = path.stat()
    if directory_stat.st_uid != os.geteuid() or directory_stat.st_mode & 0o077:
        raise DetachedStepError(f"worker-origin artifact directory is not private: {path}")


def _read_attempts(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / ATTEMPTS_FILE
    try:
        bound = _custody_for_path(path)
        if bound is not None:
            custody, relative = bound
            if not custody.file_exists(relative):
                return []
            payload = custody.read_bytes(relative, max_bytes=256 * 1024 * 1024)
            raw_lines = payload.decode("utf-8").splitlines()
        else:
            if not path.exists():
                return []
            if path.is_symlink() or not path.is_file():
                raise DetachedStepError(f"attempt journal is invalid: {path}")
            raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, SecurePathCustodyError, UnicodeDecodeError) as exc:
        raise DetachedStepError(f"attempt journal is unreadable: {path}") from exc
    events: list[dict[str, Any]] = []
    previous = ""
    launched_attempts: set[int] = set()
    control_attempts: set[int] = set()
    target_attempts: set[int] = set()
    terminal_attempts: set[int] = set()
    broker_started: set[tuple[int, str]] = set()
    broker_terminal: set[tuple[int, str]] = set()
    broker_origin_quarantined: set[tuple[int, str]] = set()
    for sequence, line in enumerate(raw_lines, start=1):
        if not line.strip():
            raise DetachedStepError(f"attempt journal contains an empty record: {path}")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DetachedStepError(f"attempt journal contains invalid JSON: {path}") from exc
        if not isinstance(event, dict):
            raise DetachedStepError(f"attempt journal record must be an object: {path}")
        if event.get("schema") != f"{SCHEMA_PREFIX}.attempt_event.v1":
            raise DetachedStepError(f"attempt journal schema mismatch: {path}")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        if event.get("sequence") != sequence:
            raise DetachedStepError(f"attempt journal sequence mismatch: {path}")
        if event.get("previous_event_sha256") != previous:
            raise DetachedStepError(f"attempt journal chain mismatch: {path}")
        expected_hash = _sha256(body)
        if event.get("event_sha256") != expected_hash:
            raise DetachedStepError(f"attempt journal hash mismatch: {path}")
        attempt = int(event.get("attempt") or 0)
        event_type = str(event.get("event") or "")
        if attempt <= 0:
            raise DetachedStepError(f"attempt journal has an invalid attempt: {path}")
        if event_type == "LAUNCHED":
            if terminal_attempts:
                raise DetachedStepError(f"attempt journal continues after terminal state: {path}")
            if attempt != len(launched_attempts) + 1:
                raise DetachedStepError(f"attempt journal launch order mismatch: {path}")
            if attempt in launched_attempts or attempt in terminal_attempts:
                raise DetachedStepError(f"attempt journal has duplicate launch: {path}")
            launched_attempts.add(attempt)
        elif event_type == "CONTROL_READY":
            if terminal_attempts:
                raise DetachedStepError(f"attempt journal continues after terminal state: {path}")
            if (
                attempt not in launched_attempts
                or attempt in control_attempts
                or attempt != max(launched_attempts)
            ):
                raise DetachedStepError(f"attempt journal has invalid control record: {path}")
            control_attempts.add(attempt)
        elif event_type == "TARGET_STARTED":
            if terminal_attempts:
                raise DetachedStepError(f"attempt journal continues after terminal state: {path}")
            if (
                attempt not in launched_attempts
                or attempt not in control_attempts
                or attempt in target_attempts
                or attempt != max(launched_attempts)
            ):
                raise DetachedStepError(f"attempt journal has invalid target record: {path}")
            target_attempts.add(attempt)
        elif event_type == "BROKER_STARTED":
            request_id = str(event.get("request_id") or "")
            key = (attempt, request_id)
            if (
                terminal_attempts
                or attempt not in target_attempts
                or attempt != max(launched_attempts)
                or len(request_id) != 32
                or key in broker_started
            ):
                raise DetachedStepError(f"attempt journal has invalid broker start: {path}")
            broker_started.add(key)
        elif event_type == "BROKER_TERMINAL":
            request_id = str(event.get("request_id") or "")
            key = (attempt, request_id)
            if (
                terminal_attempts
                or key not in broker_started
                or key in broker_terminal
                or key in broker_origin_quarantined
                or attempt != max(launched_attempts)
            ):
                raise DetachedStepError(f"attempt journal has invalid broker terminal: {path}")
            broker_terminal.add(key)
        elif event_type == "BROKER_ORIGIN_QUARANTINED":
            request_id = str(event.get("request_id") or "")
            key = (attempt, request_id)
            receipt = event.get("quarantine_receipt")
            if (
                terminal_attempts
                or key not in broker_started
                or key in broker_terminal
                or key in broker_origin_quarantined
                or attempt != max(launched_attempts)
                or not isinstance(receipt, dict)
            ):
                raise DetachedStepError(
                    f"attempt journal has invalid worker-origin quarantine: {path}"
                )
            receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            if receipt.get("receipt_sha256") != _sha256(receipt_body):
                raise DetachedStepError(
                    f"attempt journal worker-origin quarantine hash mismatch: {path}"
                )
            broker_origin_quarantined.add(key)
        elif event_type == "TERMINAL":
            if (
                attempt not in launched_attempts
                or attempt in terminal_attempts
                or attempt != max(launched_attempts)
            ):
                raise DetachedStepError(f"attempt journal has invalid terminal record: {path}")
            receipt = event.get("receipt")
            if not isinstance(receipt, dict):
                raise DetachedStepError(f"attempt journal terminal receipt is invalid: {path}")
            receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            if receipt.get("receipt_sha256") != _sha256(receipt_body):
                raise DetachedStepError(f"attempt journal terminal receipt hash mismatch: {path}")
            terminal_attempts.add(attempt)
        else:
            raise DetachedStepError(f"attempt journal has an unknown event: {path}")
        events.append(event)
        previous = expected_hash
    return events


def _append_attempt_event_locked(run_dir: Path, event_body: dict[str, Any]) -> dict[str, Any]:
    events = _read_attempts(run_dir)
    body = {
        "schema": f"{SCHEMA_PREFIX}.attempt_event.v1",
        "sequence": len(events) + 1,
        "previous_event_sha256": events[-1]["event_sha256"] if events else "",
        **event_body,
    }
    event = {**body, "event_sha256": _sha256(body)}
    combined = [*events, event]
    payload = b"".join(_canonical_bytes(item) + b"\n" for item in combined)
    _atomic_write_bytes(run_dir / ATTEMPTS_FILE, payload)
    _read_attempts(run_dir)
    return event


def _append_attempt_event(run_dir: Path, event_body: dict[str, Any]) -> dict[str, Any]:
    with _locked(run_dir):
        return _append_attempt_event_locked(run_dir, event_body)


@contextmanager
def _locked(run_dir: Path) -> Iterator[None]:
    root_custody = _ACTIVE_RUN_CUSTODIES.get(run_dir.expanduser().absolute())
    if root_custody is not None:
        try:
            with root_custody.file_lock(LOCK_FILE):
                yield
            return
        except SecurePathCustodyError as exc:
            raise DetachedStepError(f"run directory custody failed: {run_dir}") from exc
    run_dir.mkdir(parents=True, exist_ok=True)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise DetachedStepError(f"run directory is invalid: {run_dir}")
    directory_stat = run_dir.stat()
    if directory_stat.st_uid != os.geteuid() or directory_stat.st_mode & 0o022:
        raise DetachedStepError(f"run directory ownership or permissions are unsafe: {run_dir}")
    descriptor = os.open(
        run_dir / LOCK_FILE,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | _NOFOLLOW,
        0o600,
    )
    try:
        lock_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_uid != os.geteuid()
            or lock_stat.st_nlink != 1
        ):
            raise DetachedStepError(f"lock file ownership or type is unsafe: {run_dir / LOCK_FILE}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _pid_signal_state(pid: int) -> Literal["alive", "dead", "unknown"]:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "unknown"
    return "alive"


def _darwin_process_observation(pid: int) -> ProcessObservation:
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        proc_pidpath = libproc.proc_pidpath
        proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        proc_pidpath.restype = ctypes.c_int
        info = _ProcBSDInfo()
        copied = proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
    except (AttributeError, OSError):
        return ProcessObservation("unknown")
    if copied != ctypes.sizeof(info) or int(info.pbi_pid) != pid:
        signal_state = _pid_signal_state(pid)
        return ProcessObservation("dead" if signal_state == "dead" else "unknown")
    if int(info.pbi_status) == 5:
        return ProcessObservation("dead")
    path_buffer = ctypes.create_string_buffer(4096)
    path_length = proc_pidpath(pid, path_buffer, len(path_buffer))
    if path_length <= 0:
        return ProcessObservation("unknown")
    executable = path_buffer.value.decode("utf-8", errors="surrogateescape")
    identity = {
        "pid": pid,
        "uid": int(info.pbi_uid),
        "start_seconds": int(info.pbi_start_tvsec),
        "start_microseconds": int(info.pbi_start_tvusec),
    }
    return ProcessObservation(
        "alive",
        token=_sha256(identity),
        process_group_id=int(info.pbi_pgid),
        executable=executable,
    )


def _portable_process_observation(pid: int) -> ProcessObservation:
    try:
        result = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "lstart=", "-o", "pgid=", "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ProcessObservation("unknown")
    line = result.stdout.strip()
    if result.returncode != 0 or not line:
        signal_state = _pid_signal_state(pid)
        return ProcessObservation("dead" if signal_state == "dead" else "unknown")
    fields = line.split(maxsplit=6)
    if len(fields) != 7:
        return ProcessObservation("unknown")
    try:
        process_group_id = int(fields[5])
    except ValueError:
        return ProcessObservation("unknown")
    executable = fields[6]
    return ProcessObservation(
        "alive",
        token=_sha256(
            {
                "pid": pid,
                "start": " ".join(fields[:5]),
            }
        ),
        process_group_id=process_group_id,
        executable=executable,
    )


def _inspect_process(pid: int) -> ProcessObservation:
    if pid <= 0:
        return ProcessObservation("dead")
    if sys.platform == "darwin":
        return _darwin_process_observation(pid)
    return _portable_process_observation(pid)


def _process_start_token(pid: int) -> str:
    observation = _inspect_process(pid)
    return observation.token if observation.state == "alive" else ""


def _identity_state(pid: int, token: str) -> Literal["alive", "dead", "unknown"]:
    if not token:
        return "unknown"
    observation = _inspect_process(pid)
    if observation.state != "alive":
        return observation.state
    if not observation.token:
        return "unknown"
    return "alive" if observation.token == token else "dead"


def _pid_matches(pid: int, start_token: str) -> bool:
    return _identity_state(pid, start_token) == "alive"


def _wait_for_pid_exit(pid: int, start_token: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = _identity_state(pid, start_token)
        if state == "dead":
            return True
        if state == "unknown":
            raise DetachedStepError("process identity became unobservable")
        time.sleep(0.05)
    state = _identity_state(pid, start_token)
    if state == "unknown":
        raise DetachedStepError("process identity became unobservable")
    return state == "dead"


def _observe_direct_child(
    child: subprocess.Popen[Any],
    child_token: str,
    timeout_s: float,
) -> tuple[ProcessObservation, int | None]:
    """Reconcile waitpid and libproc across transient Darwin exit states."""
    deadline = time.monotonic() + timeout_s
    observation = ProcessObservation("unknown")
    # Each non-terminal round sleeps 10ms, so the round bound can only trip
    # after the wall-clock deadline check inside the body has returned; it
    # exists as the hard guarantee that this loop is finite.
    for _round in range(max(1, int(timeout_s * 100) + 2)):
        returncode = child.poll()
        if returncode is not None:
            return ProcessObservation("dead"), returncode
        observation = _inspect_process(child.pid)
        if (
            os.environ.get("AURA_DETACHED_TEST_LIBPROC_DARK") == "direct_child"
            and "PYTEST_CURRENT_TEST" in os.environ
        ):
            observation = ProcessObservation("unknown")
        if observation.state == "alive":
            if not observation.token:
                observation = ProcessObservation("unknown")
            elif observation.token != child_token:
                raise DetachedStepError("target process identity changed before reap")
            else:
                return observation, None
        if time.monotonic() >= deadline:
            return observation, None
        time.sleep(0.01)
    return observation, None


def _process_group_exists(process_group_id: int) -> bool:
    if process_group_id <= 1:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise DetachedStepError("cannot inspect child process group") from exc
    return True


def _terminate_group_id(process_group_id: int) -> bool:
    if not _process_group_exists(process_group_id):
        return False
    os.killpg(process_group_id, signal.SIGTERM)
    deadline = time.monotonic() + _TERM_GRACE_S
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            return True
        time.sleep(0.05)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + _TERM_GRACE_S
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            return True
        time.sleep(0.05)
    raise DetachedStepError("child process group survived TERM and KILL")


def _tagged_processes(containment_token: str) -> list[tuple[int, ProcessObservation]]:
    marker = f"AURA_DETACHED_RUN_TOKEN={containment_token}"
    try:
        result = subprocess.run(
            ["/bin/ps", "eww", "-axo", "pid=,uid=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DetachedStepError("cannot inspect tagged target lineage") from exc
    if result.returncode != 0:
        raise DetachedStepError("tagged target lineage inspection failed")
    tagged: list[tuple[int, ProcessObservation]] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or marker not in fields[2]:
            continue
        try:
            pid = int(fields[0])
            uid = int(fields[1])
        except ValueError:
            continue
        if uid != os.geteuid():
            continue
        observation = _inspect_process(pid)
        if observation.state == "unknown":
            raise DetachedStepError("tagged target identity is unobservable")
        if observation.state == "alive":
            tagged.append((pid, observation))
    return tagged


def _terminate_tagged_processes(containment_token: str) -> int:
    tagged = _tagged_processes(containment_token)
    terminated_pids: set[int] = set()
    for pid, observation in tagged:
        if _identity_state(pid, observation.token) != "alive":
            continue
        os.kill(pid, signal.SIGTERM)
        terminated_pids.add(pid)
    deadline = time.monotonic() + _TERM_GRACE_S
    while time.monotonic() < deadline:
        remaining = _tagged_processes(containment_token)
        if not remaining:
            return len(terminated_pids)
        time.sleep(0.05)
    for pid, observation in _tagged_processes(containment_token):
        if _identity_state(pid, observation.token) != "alive":
            continue
        os.kill(pid, signal.SIGKILL)
        terminated_pids.add(pid)
    deadline = time.monotonic() + _TERM_GRACE_S
    while time.monotonic() < deadline:
        if not _tagged_processes(containment_token):
            return len(terminated_pids)
        time.sleep(0.05)
    raise DetachedStepError("tagged target lineage survived TERM and KILL")


def _cleanup_child_process(
    child: subprocess.Popen[Any],
    child_token: str,
    process_group_id: int,
    containment_token: str,
) -> tuple[bool, int]:
    # waitpid is authoritative for a direct child. On Darwin, proc_pidpath can
    # disappear during the short zombie window before waitpid reaps the child,
    # which makes libproc observation indeterminate even though the supervisor
    # can prove that its own child exited.
    observation, direct_returncode = _observe_direct_child(
        child,
        child_token,
        _TERM_GRACE_S,
    )
    if direct_returncode is not None:
        lineage_cleanup_count = _terminate_tagged_processes(containment_token)
        if _tagged_processes(containment_token):
            raise DetachedStepError("tagged target lineage is not empty")
        if _process_group_exists(process_group_id):
            raise DetachedStepError("exited target process group is not empty")
        return lineage_cleanup_count > 0, lineage_cleanup_count

    groups = {process_group_id}
    if observation.state == "alive" and observation.token == child_token:
        if observation.process_group_id > 1:
            groups.add(observation.process_group_id)
    elif observation.state == "unknown":
        raise DetachedStepError("cannot prove target identity during cleanup")
    cleanup_performed = any(_process_group_exists(group) for group in groups)
    for group in groups:
        if _process_group_exists(group):
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                pass
    try:
        child.wait(timeout=_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        for group in groups:
            if _process_group_exists(group):
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        child.wait(timeout=_TERM_GRACE_S)
    for group in groups:
        if _process_group_exists(group):
            _terminate_group_id(group)
    identity_state = _identity_state(child.pid, child_token)
    if identity_state == "unknown":
        raise DetachedStepError("cannot prove target exit after cleanup")
    if identity_state == "alive":
        raise DetachedStepError("target survived process-group cleanup")
    lineage_cleanup_count = _terminate_tagged_processes(containment_token)
    if _tagged_processes(containment_token):
        raise DetachedStepError("tagged target lineage is not empty")
    return cleanup_performed or lineage_cleanup_count > 0, lineage_cleanup_count


def _terminate_stale_target(target: dict[str, Any]) -> bool:
    child_pid = int(target.get("child_pid") or 0)
    child_group = int(target.get("child_process_group_id") or 0)
    child_start = str(target.get("child_start_token") or "")
    containment_token = str(target.get("containment_token") or "")
    if len(containment_token) != 64:
        raise DetachedStepError("stale target containment token is invalid")
    state = _identity_state(child_pid, child_start)
    if state == "unknown":
        raise DetachedStepError("stale target identity is unobservable; refusing resume")
    lineage_cleanup_count = _terminate_tagged_processes(containment_token)
    state = _identity_state(child_pid, child_start)
    if state == "dead":
        if _process_group_exists(child_group):
            raise DetachedStepError(
                "stale target leader is gone but its process group still exists; refusing unsafe resume"
            )
        return lineage_cleanup_count > 0
    if child_group <= 1:
        raise DetachedStepError("refusing to terminate an invalid stale child process group")
    observation = _inspect_process(child_pid)
    if observation.state != "alive" or observation.process_group_id != child_group:
        raise DetachedStepError("stale target process-group identity mismatch")
    cleaned = _terminate_group_id(child_group)
    if not _wait_for_pid_exit(child_pid, child_start, _TERM_GRACE_S):
        raise DetachedStepError("stale child process survived process-group termination")
    return cleaned or lineage_cleanup_count > 0


def _terminate_stale_broker_worker(started: dict[str, Any]) -> bool:
    worker_pid = int(started.get("worker_pid") or 0)
    worker_group = int(started.get("worker_process_group_id") or 0)
    worker_start = str(started.get("worker_start_token") or "")
    containment_token = str(started.get("containment_token") or "")
    if len(containment_token) != 64:
        raise DetachedStepError("stale broker worker containment token is invalid")
    state = _identity_state(worker_pid, worker_start)
    if state == "unknown":
        raise DetachedStepError("stale broker worker identity is unobservable; refusing resume")
    lineage_cleanup_count = _terminate_tagged_processes(containment_token)
    state = _identity_state(worker_pid, worker_start)
    if state == "dead":
        if _process_group_exists(worker_group):
            raise DetachedStepError("stale broker worker group survived lineage cleanup")
        return lineage_cleanup_count > 0
    observation = _inspect_process(worker_pid)
    if (
        worker_group <= 1
        or observation.state != "alive"
        or observation.process_group_id != worker_group
    ):
        raise DetachedStepError("stale broker worker process-group identity mismatch")
    cleaned = _terminate_group_id(worker_group)
    if not _wait_for_pid_exit(worker_pid, worker_start, _TERM_GRACE_S):
        raise DetachedStepError("stale broker worker survived process-group termination")
    return cleaned or lineage_cleanup_count > 0


def _kill_group(process: subprocess.Popen[Any], sig: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (OSError, ProcessLookupError):
        pass


def _sandboxed_command(plan: dict[str, Any], command: list[str]) -> list[str]:
    sandbox = plan.get("execution_sandbox")
    if sandbox is None:
        return list(command)
    if sandbox.get("mode") == _PRECONTAINED_SANDBOX_MODE:
        return list(command)
    return [str(sandbox["path"]), "-p", str(sandbox["profile"]), *command]


def _executed_command(plan: dict[str, Any]) -> list[str]:
    return _sandboxed_command(plan, list(plan["command"]))


def _start_power_assertion(
    child_pid: int, log: Any, plan: dict[str, Any]
) -> subprocess.Popen[Any] | None:
    power_assertion = plan.get("power_assertion")
    if power_assertion is None:
        return None
    return subprocess.Popen(
        [str(power_assertion["path"]), "-i", "-w", str(child_pid)],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
        env=dict(plan["execution_environment"]),
    )


def _stop_power_assertion(assertion: subprocess.Popen[Any] | None) -> None:
    if assertion is None:
        return
    try:
        assertion.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass
    _kill_group(assertion, signal.SIGTERM)
    try:
        assertion.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        _kill_group(assertion, signal.SIGKILL)
        assertion.wait(timeout=1.0)


def _open_secure_log(path: Path) -> Any:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    bound = _custody_for_path(path)
    if bound is not None:
        custody, relative = bound
        descriptor = custody.open_file(
            relative,
            flags,
            mode=0o600,
            create_parents=True,
        )
    else:
        descriptor = os.open(
            path,
            flags | getattr(os, "O_CLOEXEC", 0) | _NOFOLLOW,
            0o600,
        )
    try:
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.geteuid()
            or file_stat.st_nlink != 1
        ):
            raise DetachedStepError(f"log file ownership or type is unsafe: {path}")
        return os.fdopen(descriptor, "ab", buffering=0, closefd=True)
    except BaseException:  # noqa: BLE001 - fd cleanup on any exit; original re-raised
        os.close(descriptor)
        raise


def _gated_exec(gate_fd: int, command: list[str]) -> int:
    try:
        release = os.read(gate_fd, 1)
    finally:
        os.close(gate_fd)
    if release != b"G":
        return 125
    os.execvp(command[0], command)
    return 126


def _spawn_gated_target(
    command: list[str],
    *,
    cwd: str,
    environment: dict[str, str],
    log: Any,
    inherited_fds: tuple[int, ...] = (),
) -> tuple[subprocess.Popen[Any], int]:
    gate_read_fd, gate_write_fd = os.pipe()
    wrapper = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_exec_gate",
        str(gate_read_fd),
        *command,
    ]
    try:
        child = subprocess.Popen(
            wrapper,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            pass_fds=(gate_read_fd, *inherited_fds),
            env=environment,
        )
    except BaseException:  # noqa: BLE001 - fd cleanup on any exit; original re-raised
        os.close(gate_read_fd)
        os.close(gate_write_fd)
        raise
    os.close(gate_read_fd)
    return child, gate_write_fd


def _publish_status(run_dir: Path, body: dict[str, Any]) -> None:
    payload = {
        "schema": f"{SCHEMA_PREFIX}.status.v1",
        **body,
    }
    _atomic_write(run_dir / STATUS_FILE, payload)


def _create_control_socket(
    run_dir: Path,
    plan: dict[str, Any],
    attempt: int,
    supervisor_pid: int,
    supervisor_start_token: str,
) -> tuple[socket.socket, Path, str, str]:
    filename = (
        f"{CONTROL_SOCKET_PREFIX}-{os.geteuid()}-{str(plan['plan_sha256'])[:16]}-{attempt}.sock"
    )
    socket_path = Path("/tmp") / filename
    if socket_path.exists() or socket_path.is_symlink():
        raise DetachedStepError(f"control socket path already exists: {socket_path}")
    control_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        control_socket.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        control_socket.setblocking(False)
        control_token = secrets.token_hex(32)
        broker_token = secrets.token_hex(32) if plan["broker_policy"] else ""
        _append_attempt_event(
            run_dir,
            {
                "event": "CONTROL_READY",
                "attempt": attempt,
                "plan_sha256": plan["plan_sha256"],
                "supervisor_pid": supervisor_pid,
                "supervisor_start_token": supervisor_start_token,
                "socket_path": str(socket_path),
                "control_token": control_token,
                "broker_token": broker_token,
                "broker_enabled": bool(plan["broker_policy"]),
                "recorded_at": time.time(),
            },
        )
        return control_socket, socket_path, control_token, broker_token
    except BaseException:  # noqa: BLE001 - socket cleanup on any exit; original re-raised
        control_socket.close()
        socket_path.unlink(missing_ok=True)
        raise


def _poll_control_socket(control_socket: socket.socket) -> dict[str, Any] | None:
    try:
        payload = control_socket.recv(4096)
    except BlockingIOError:
        return None
    try:
        request = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return request if isinstance(request, dict) else None


def _broker_reply_path(request: dict[str, Any], target_pid: int) -> Path:
    reply_path = Path(str(request.get("reply_path") or ""))
    expected_prefix = f"aura-broker-reply-{os.geteuid()}-{target_pid}-"
    try:
        reply_stat = reply_path.lstat()
        expected_parent = Path("/tmp").resolve(strict=True)
        actual_parent = reply_path.parent.resolve(strict=True)
    except OSError as exc:
        raise BrokerRequestError("broker reply socket is unavailable") from exc
    if (
        not reply_path.is_absolute()
        or actual_parent != expected_parent
        or not reply_path.name.startswith(expected_prefix)
        or not stat.S_ISSOCK(reply_stat.st_mode)
        or reply_stat.st_uid != os.geteuid()
    ):
        raise BrokerRequestError("broker reply socket identity is invalid")
    return reply_path


def _matching_broker_policy(plan: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    command_sha = str(request.get("command_sha256") or "")
    request_binding_sha = str(request.get("request_binding_sha256") or "")
    timeout_s = request.get("timeout_s")
    if (
        request.get("schema") != broker_protocol.REQUEST_SCHEMA
        or request.get("action") != "run"
        or len(command_sha) != 64
        or any(character not in "0123456789abcdef" for character in command_sha)
        or len(request_binding_sha) != 64
        or any(character not in "0123456789abcdef" for character in request_binding_sha)
        or not isinstance(timeout_s, (int, float))
        or isinstance(timeout_s, bool)
        or not math.isfinite(float(timeout_s))
        or float(timeout_s) <= 0.0
    ):
        raise BrokerRequestError("broker request binding is invalid")
    policies = plan.get("broker_policy")
    if not isinstance(policies, list):
        raise DetachedStepError("broker policy is unavailable")
    for policy_value in policies:
        if not isinstance(policy_value, dict):
            raise DetachedStepError("broker policy entry is invalid")
        policy: dict[str, Any] = policy_value
        if policy["command_sha256"] != command_sha:
            continue
        expected_binding = broker_protocol.compute_broker_request_binding(
            policy["command"],
            cwd=policy["cwd"],
            stdout_path=policy["stdout_path"],
        )
        if (
            request_binding_sha != expected_binding
            or float(timeout_s) > float(policy["timeout_s_max"])
        ):
            raise BrokerRequestError("broker request exceeds its frozen policy")
        return policy
    raise BrokerRequestError("broker command is not in the frozen policy")


def _worker_origin_artifact_paths(
    contract: dict[str, Any],
    *,
    supervisor_attempt: int,
    broker_policy_sha256: str,
) -> dict[str, Path]:
    artifact_dir = Path(contract["artifact_dir"])
    prefix = (
        f"worker-origin-attempt-{supervisor_attempt:04d}-"
        f"slot-{int(contract['worker_attempt_slot']):04d}-"
        f"{broker_policy_sha256[:16]}"
    )
    return {
        "payload": artifact_dir / f"{prefix}.payload.json",
        "request": artifact_dir / f"{prefix}.request.json",
        "attestation": artifact_dir / f"{prefix}.attestation.json",
        "lifecycle": artifact_dir / f"{prefix}.lifecycle.json",
    }


def _record_worker_origin_quarantine_locked(
    run_dir: Path,
    plan: dict[str, Any],
    *,
    attempt: int,
    launched: dict[str, Any],
    broker_start: dict[str, Any],
    cleanup_action_performed: bool,
) -> dict[str, Any] | None:
    policy_sha256 = str(broker_start.get("policy_sha256") or "")
    policy = next(
        (
            candidate
            for candidate in plan["broker_policy"]
            if candidate["policy_sha256"] == policy_sha256
        ),
        None,
    )
    if policy is None:
        raise DetachedStepError("worker-origin quarantine policy is unavailable")
    contract = _verify_worker_origin_policy(
        policy.get("worker_origin"),
        require_current=False,
    )
    if contract is None:
        return None

    request_id = str(broker_start.get("request_id") or "")
    events = _read_attempts(run_dir)
    attempt_events = _events_by_attempt(events).get(attempt, {})
    existing = [
        event
        for event in _broker_events(
            attempt_events,
            "BROKER_ORIGIN_QUARANTINED",
        )
        if event.get("request_id") == request_id
    ]
    if existing:
        if len(existing) != 1:
            raise DetachedStepError("duplicate worker-origin quarantine records")
        return existing[0]
    if any(
        event.get("request_id") == request_id
        for event in _broker_events(attempt_events, "BROKER_TERMINAL")
    ):
        raise DetachedStepError("cannot quarantine a terminal broker origin")

    supervisor_pid = int(launched.get("supervisor_pid") or 0)
    supervisor_start_token = str(launched.get("supervisor_start_token") or "")
    worker_pid = int(broker_start.get("worker_pid") or 0)
    worker_start_token = str(broker_start.get("worker_start_token") or "")
    worker_process_group_id = int(broker_start.get("worker_process_group_id") or 0)
    if _identity_state(supervisor_pid, supervisor_start_token) != "dead":
        raise DetachedStepError("worker-origin quarantine requires a dead prior supervisor")
    if _identity_state(worker_pid, worker_start_token) != "dead":
        raise DetachedStepError("worker-origin quarantine requires a contained worker")
    if _process_group_exists(worker_process_group_id):
        raise DetachedStepError("worker-origin quarantine requires an empty worker process group")

    start_origin = broker_start.get("worker_origin")
    if not isinstance(start_origin, dict):
        raise DetachedStepError("worker-origin quarantine start metadata is unavailable")
    paths = _worker_origin_artifact_paths(
        contract,
        supervisor_attempt=attempt,
        broker_policy_sha256=policy_sha256,
    )
    lifecycle_sha256: str | None = None
    if paths["lifecycle"].exists() or paths["lifecycle"].is_symlink():
        lifecycle = _read_canonical_private_json(
            paths["lifecycle"],
            max_bytes=_MAX_WORKER_ORIGIN_TRUST_POLICY_BYTES,
            role="worker-origin lifecycle artifact",
        )
        lifecycle_sha256 = str(lifecycle.get("artifact_sha256") or "")
        if len(lifecycle_sha256) != 64:
            raise DetachedStepError("worker-origin quarantine lifecycle digest is invalid")

    prior_journal_head_sha256 = str(events[-1]["event_sha256"]) if events else ""
    quarantined_at_unix = int(time.time())
    receipt_body = {
        "schema": WORKER_ORIGIN_QUARANTINE_RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "broker_policy_sha256": policy_sha256,
        "request_id": request_id,
        "supervisor_attempt": attempt,
        "supervisor_pid": supervisor_pid,
        "supervisor_start_token": supervisor_start_token,
        "worker_pid": worker_pid,
        "worker_process_group_id": worker_process_group_id,
        "worker_start_token": worker_start_token,
        "containment_token": broker_start["containment_token"],
        "worker_origin_contract_sha256": contract["contract_sha256"],
        "session_id": start_origin["session_id"],
        "authorization_request_sha256": start_origin["authorization_request_sha256"],
        "authorization_attestation_sha256": start_origin["authorization_attestation_sha256"],
        "payload_path": start_origin["payload_path"],
        "request_path": start_origin["request_path"],
        "attestation_path": start_origin["attestation_path"],
        "lifecycle_path": start_origin["lifecycle_path"],
        "lifecycle_artifact_sha256": lifecycle_sha256,
        "prior_journal_head_sha256": prior_journal_head_sha256,
        "supervisor_identity_observed": "dead",
        "worker_identity_observed": "dead",
        "worker_process_group_empty": True,
        "cleanup_action_performed": bool(cleanup_action_performed),
        "authority_key_recoverable": False,
        "lifecycle_recoverable": False,
        "claim_eligible": False,
        "reason": "supervisor_ephemeral_authority_lost",
        "quarantined_at_unix": quarantined_at_unix,
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": _sha256(receipt_body),
    }
    return _append_attempt_event_locked(
        run_dir,
        {
            "event": "BROKER_ORIGIN_QUARANTINED",
            "attempt": attempt,
            "plan_sha256": plan["plan_sha256"],
            "request_id": request_id,
            "policy_sha256": policy_sha256,
            "quarantine_receipt": receipt,
            "recorded_at": float(quarantined_at_unix),
        },
    )


def _prepare_broker_worker_origin(
    plan: dict[str, Any],
    broker_policy: dict[str, Any],
    *,
    supervisor_attempt: int,
) -> PreparedBrokerWorkerOrigin | None:
    contract = _verify_worker_origin_policy(
        broker_policy.get("worker_origin"),
        require_current=True,
    )
    if contract is None:
        return None
    _ensure_private_directory(Path(contract["artifact_dir"]))
    paths = _worker_origin_artifact_paths(
        contract,
        supervisor_attempt=supervisor_attempt,
        broker_policy_sha256=broker_policy["policy_sha256"],
    )
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise BrokerRequestError("worker-origin artifact slot already exists")
    try:
        trust_root_pem = base64.b64decode(
            contract["trust_root_public_key_pem_b64"],
            validate=True,
        )
        verified_policy = validate_campaign_trust_policy(
            contract["trust_policy_document"],
            trusted_root_public_key_pem=trust_root_pem,
            expected_campaign_name=contract["campaign_name"],
            expected_policy_sha256=contract["trust_policy_sha256"],
            expected_protocol_sha256=contract["protocol_sha256"],
            now_unix=int(time.time()),
        )
        authority = DetachedWorkerOriginAuthority(
            policy=verified_policy,
            campaign_name=contract["campaign_name"],
            protocol_sha256=contract["protocol_sha256"],
            detached_plan_sha256=plan["plan_sha256"],
            broker_policy_sha256=broker_policy["policy_sha256"],
            executable_binding_sha256=broker_policy["executable_binding"]["binding_sha256"],
            environment_sha256=plan["execution_environment_sha256"],
            sandbox_sha256=_sha256(plan["execution_sandbox"]),
            source_manifest_sha256=broker_policy["execution_manifest"]["manifest_sha256"],
            session_id=secrets.token_hex(16),
            supervisor_attempt=supervisor_attempt,
            arm=contract["arm"],
            worker_attempt_slot=contract["worker_attempt_slot"],
            allowed_cells=contract["allowed_cells"],
            model_identity_sha256=contract["model_identity_sha256"],
            adapter_identity_sha256=contract["adapter_identity_sha256"],
            authorization_ttl_seconds=contract["authorization_ttl_seconds"],
        )
        request = authority.request_authorization(signed_at_unix=int(time.time()))
    except (CampaignTrustError, DetachedWorkerOriginError, ValueError) as exc:
        raise BrokerRequestError(f"worker-origin authority preparation failed: {exc}") from exc
    _atomic_write(paths["payload"], authority.authorization_payload, replace=False)
    _atomic_write(paths["request"], request, replace=False)
    return PreparedBrokerWorkerOrigin(
        policy_sha256=broker_policy["policy_sha256"],
        authority=authority,
        request=request,
        request_path=paths["request"],
        payload_path=paths["payload"],
        attestation_path=paths["attestation"],
        lifecycle_path=paths["lifecycle"],
        policy=verified_policy,
    )


def _admit_broker_worker_origin(
    prepared: PreparedBrokerWorkerOrigin | None,
) -> None:
    if prepared is None or prepared.authorized:
        return
    if not prepared.attestation_path.is_file():
        raise BrokerRequestError(
            f"worker-origin external authorization required at {prepared.attestation_path}"
        )
    attestation = _read_canonical_private_json(
        prepared.attestation_path,
        max_bytes=_MAX_WORKER_ORIGIN_TRUST_POLICY_BYTES,
        role="worker-origin authorization attestation",
    )
    try:
        prepared.authority.accept_authorization(
            attestation,
            now_unix=int(time.time()),
        )
    except DetachedWorkerOriginError as exc:
        raise BrokerRequestError(f"worker-origin authorization rejected: {exc.code}") from exc
    prepared.authorized = True


def _finalize_broker_worker_origin(
    prepared: PreparedBrokerWorkerOrigin,
    *,
    successful: bool,
    occurred_at_unix: int,
    reason: str,
) -> dict[str, Any]:
    if prepared.finalized:
        return _read_canonical_private_json(
            prepared.lifecycle_path,
            max_bytes=_MAX_WORKER_ORIGIN_TRUST_POLICY_BYTES,
            role="worker-origin lifecycle artifact",
        )
    event_origin: dict[str, Any]
    completion_error: str | None = None
    if prepared.authority.state in {
        DetachedWorkerOriginState.TERMINAL,
        DetachedWorkerOriginState.ABANDONED,
    }:
        existing_receipt = prepared.authority.lifecycle_receipt
        if existing_receipt is None:
            raise DetachedStepError("finalized worker-origin authority has no lifecycle receipt")
        event_origin = existing_receipt
    elif successful:
        try:
            event_origin = prepared.authority.complete(
                occurred_at_unix=occurred_at_unix,
                return_code=0,
            )
        except DetachedWorkerOriginError as exc:
            completion_error = exc.code
            event_origin = prepared.authority.abandon(
                reason=f"completion_rejected:{exc.code}",
                occurred_at_unix=occurred_at_unix,
            )
    else:
        event_origin = prepared.authority.abandon(
            reason=reason[:2048],
            occurred_at_unix=occurred_at_unix,
        )
    body = {
        "schema": WORKER_ORIGIN_LIFECYCLE_ARTIFACT_SCHEMA,
        "broker_policy_sha256": prepared.policy_sha256,
        "authorization_payload": prepared.authority.authorization_payload,
        "authorization_request": prepared.request,
        "authorization_attestation": prepared.authority.authorization_attestation,
        "event_origin": event_origin,
        "completion_error": completion_error,
    }
    artifact = {**body, "artifact_sha256": _sha256(body)}
    _atomic_write(prepared.lifecycle_path, artifact, replace=False)
    prepared.finalized = True
    return artifact


def _send_broker_rejection(
    request: dict[str, Any],
    target_pid: int,
    error: BaseException,
) -> None:
    try:
        reply_path = _broker_reply_path(request, target_pid)
    except BrokerRequestError:
        return
    request_id = str(request.get("request_id") or "")
    command_sha = str(request.get("command_sha256") or "")
    body = {
        "schema": f"{SCHEMA_PREFIX}.broker_response.v1",
        "request_id": request_id,
        "policy_sha256": None,
        "command_sha256": command_sha,
        "worker_pid": 0,
        "worker_process_group_id": 0,
        "worker_start_token": "",
        "returncode": 70,
        "timed_out": False,
        "containment_verified": True,
        "status": "rejected",
        "error": f"{type(error).__name__}: {error}"[:1000],
    }
    signed = {**body, "receipt_sha256": _sha256(body)}
    broker_token = str(request.get("broker_token") or "")
    if len(broker_token) != 64:
        return
    response = {
        **signed,
        "response_hmac_sha256": hmac.new(
            bytes.fromhex(broker_token),
            _canonical_bytes(signed),
            hashlib.sha256,
        ).hexdigest(),
    }
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            sender.sendto(_canonical_bytes(response), str(reply_path))
    except OSError:
        return


def _start_broker_worker(
    run_dir: Path,
    plan: dict[str, Any],
    attempt: int,
    supervisor_pid: int,
    supervisor_start_token: str,
    target_pid: int,
    request: dict[str, Any],
    prepared_origins: dict[str, PreparedBrokerWorkerOrigin],
) -> ActiveBrokerWorker:
    request_id = str(request.get("request_id") or "")
    if len(request_id) != 32 or any(
        character not in "0123456789abcdef" for character in request_id
    ):
        raise BrokerRequestError("broker request id is invalid")
    policy = _matching_broker_policy(plan, request)
    prepared_origin = prepared_origins.get(policy["policy_sha256"])
    if prepared_origin is None and policy.get("worker_origin") is not None:
        prepared_origin = _prepare_broker_worker_origin(
            plan,
            policy,
            supervisor_attempt=attempt,
        )
        if prepared_origin is None:
            raise DetachedStepError("worker-origin preparation returned no authority")
        prepared_origins[policy["policy_sha256"]] = prepared_origin
    _admit_broker_worker_origin(prepared_origin)
    reply_path = _broker_reply_path(request, target_pid)
    _verify_execution_manifest_current(policy["execution_manifest"])
    log = _open_secure_log(Path(policy["stdout_path"]))
    gate_write_fd: int | None = None
    worker: subprocess.Popen[Any] | None = None
    worker_start_token = ""
    worker_group = 0
    containment_token = secrets.token_hex(32)
    origin_server: DetachedWorkerOriginChannelServer | None = None
    origin_supervisor_socket: socket.socket | None = None
    origin_worker_socket: socket.socket | None = None
    try:
        environment = dict(plan["execution_environment"])
        environment["AURA_DETACHED_RUN_TOKEN"] = containment_token
        inherited_fds: tuple[int, ...] = ()
        if prepared_origin is not None:
            origin_supervisor_socket, origin_worker_socket = create_worker_origin_socketpair()
            origin_server = DetachedWorkerOriginChannelServer(
                origin_supervisor_socket,
                prepared_origin.authority,
            )
            environment[WORKER_ORIGIN_FD_ENV] = str(origin_worker_socket.fileno())
            environment[WORKER_ORIGIN_SESSION_ENV] = origin_server.session_id
            inherited_fds = (origin_worker_socket.fileno(),)
        worker, gate_write_fd = _spawn_gated_target(
            _sandboxed_command(plan, list(policy["command"])),
            cwd=policy["cwd"],
            environment=environment,
            log=log,
            inherited_fds=inherited_fds,
        )
        if origin_worker_socket is not None:
            origin_worker_socket.close()
            origin_worker_socket = None
        observation = _inspect_process(worker.pid)
        if observation.state != "alive" or observation.process_group_id != worker.pid:
            raise DetachedStepError("broker worker identity could not be established")
        worker_start_token = observation.token
        worker_group = observation.process_group_id
        with _locked(run_dir):
            events = _read_attempts(run_dir)
            attempt_events = _events_by_attempt(events).get(attempt, {})
            invocations = sum(
                event.get("policy_sha256") == policy["policy_sha256"]
                for event in _broker_events(attempt_events, "BROKER_STARTED")
            )
            if invocations >= int(policy["max_invocations"]):
                raise BrokerRequestError("broker policy invocation bound is exhausted")
            if any(
                event.get("request_id") == request_id
                for event in _broker_events(attempt_events, "BROKER_STARTED")
            ):
                raise BrokerRequestError("broker request id has already been used")
            _append_attempt_event_locked(
                run_dir,
                {
                    "event": "BROKER_STARTED",
                    "attempt": attempt,
                    "plan_sha256": plan["plan_sha256"],
                    "supervisor_pid": supervisor_pid,
                    "supervisor_start_token": supervisor_start_token,
                    "request_id": request_id,
                    "policy_sha256": policy["policy_sha256"],
                    "command_sha256": policy["command_sha256"],
                    "worker_pid": worker.pid,
                    "worker_process_group_id": observation.process_group_id,
                    "worker_start_token": observation.token,
                    "containment_token": containment_token,
                    "reply_path": str(reply_path),
                    "timeout_s": float(request["timeout_s"]),
                    "worker_origin": (
                        {
                            "contract_sha256": policy["worker_origin"]["contract_sha256"],
                            "session_id": origin_server.session_id,
                            "authorization_payload": prepared_origin.authority.authorization_payload,
                            "authorization_request_sha256": prepared_origin.request[
                                "request_sha256"
                            ],
                            "authorization_attestation_sha256": hashlib.sha256(
                                canonical_json_bytes(
                                    prepared_origin.authority.authorization_attestation
                                )
                            ).hexdigest(),
                            "request_path": str(prepared_origin.request_path),
                            "payload_path": str(prepared_origin.payload_path),
                            "attestation_path": str(prepared_origin.attestation_path),
                            "lifecycle_path": str(prepared_origin.lifecycle_path),
                        }
                        if prepared_origin is not None and origin_server is not None
                        else None
                    ),
                    "recorded_at": time.time(),
                },
            )
        if prepared_origin is not None:
            prepared_origin.authority.start()
        if os.write(gate_write_fd, b"G") != 1:
            raise DetachedStepError("broker worker release gate failed")
        os.close(gate_write_fd)
        gate_write_fd = None
        if (
            os.environ.get("AURA_DETACHED_TEST_CRASH_POINT") == "after_broker_release"
            and "PYTEST_CURRENT_TEST" in os.environ
        ):
            os._exit(95)
        started_monotonic_ns = time.monotonic_ns()
        return ActiveBrokerWorker(
            request_id=request_id,
            policy_sha256=str(policy["policy_sha256"]),
            command_sha256=str(policy["command_sha256"]),
            process=worker,
            process_group_id=observation.process_group_id,
            start_token=observation.token,
            containment_token=containment_token,
            response_token=str(request["broker_token"]),
            reply_path=reply_path,
            started_at=time.time(),
            started_monotonic_ns=started_monotonic_ns,
            deadline_ns=started_monotonic_ns + int(float(request["timeout_s"]) * 1_000_000_000),
            log=log,
            worker_origin=prepared_origin,
            worker_origin_server=origin_server,
        )
    except BaseException as start_exc:  # noqa: BLE001 - start-failure cleanup; strongest error re-raised
        if gate_write_fd is not None:
            os.close(gate_write_fd)
        cleanup_exc: BaseException | None = None
        if worker is not None:
            try:
                if worker_start_token and worker_group > 1:
                    _cleanup_child_process(
                        worker,
                        worker_start_token,
                        worker_group,
                        containment_token,
                    )
                else:
                    _kill_group(worker, signal.SIGKILL)
                    worker.wait(timeout=_TERM_GRACE_S)
                    if _process_group_exists(worker.pid):
                        raise DetachedStepError("unidentified broker worker group survived cleanup")
            except BaseException as exc:  # noqa: BLE001 - cleanup error captured for the receipt, never masked
                cleanup_exc = exc
        if origin_worker_socket is not None:
            origin_worker_socket.close()
        if origin_server is not None:
            origin_server.close()
        elif origin_supervisor_socket is not None:
            origin_supervisor_socket.close()
        if prepared_origin is not None and not prepared_origin.finalized:
            try:
                _finalize_broker_worker_origin(
                    prepared_origin,
                    successful=False,
                    occurred_at_unix=int(time.time()),
                    reason=f"worker_start_failed:{type(start_exc).__name__}",
                )
            except BaseException as exc:  # noqa: BLE001 - preserve strongest cleanup failure below
                cleanup_exc = cleanup_exc or exc
        log.close()
        if cleanup_exc is not None:
            raise DetachedStepError(
                f"broker start failed ({start_exc}); cleanup failed ({cleanup_exc})"
            ) from cleanup_exc
        raise


def _broker_response_body(
    worker: ActiveBrokerWorker,
    *,
    returncode: int,
    containment_verified: bool,
    cleanup_performed: bool,
    cleanup_count: int,
    error: str | None = None,
    worker_origin_lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}.broker_response.v1",
        "request_id": worker.request_id,
        "policy_sha256": worker.policy_sha256,
        "command_sha256": worker.command_sha256,
        "worker_pid": worker.process.pid,
        "worker_process_group_id": worker.process_group_id,
        "worker_start_token": worker.start_token,
        "started_at": worker.started_at,
        "finished_at": time.time(),
        "duration_s": round(
            max(0.0, (time.monotonic_ns() - worker.started_monotonic_ns) / 1_000_000_000),
            6,
        ),
        "returncode": int(returncode),
        "timed_out": worker.timed_out,
        "cleanup_performed": cleanup_performed,
        "lineage_cleanup_count": cleanup_count,
        "containment_verified": containment_verified,
        "status": (
            "containment_failed"
            if not containment_verified
            else "timed_out"
            if worker.timed_out
            else "passed"
            if returncode == 0
            else "failed"
        ),
        "error": error,
        "worker_origin_lifecycle": worker_origin_lifecycle,
    }


def _finish_broker_worker(
    run_dir: Path,
    plan: dict[str, Any],
    attempt: int,
    worker: ActiveBrokerWorker,
    *,
    force_returncode: int | None = None,
) -> dict[str, Any]:
    returncode = force_returncode
    origin_channel_error: str | None = None
    if worker.worker_origin_server is not None:
        try:
            while worker.worker_origin_server.poll_once():
                pass
        except DetachedWorkerOriginChannelError as exc:
            returncode = 70
            origin_channel_error = exc.code
    if returncode is None:
        observed = worker.process.poll()
        returncode = observed if observed is not None else 70
    containment_verified = False
    cleanup_performed = False
    cleanup_count = 0
    cleanup_error: str | None = None
    worker_origin_summary: dict[str, Any] | None = None
    try:
        cleanup_performed, cleanup_count = _cleanup_child_process(
            worker.process,
            worker.start_token,
            worker.process_group_id,
            worker.containment_token,
        )
        containment_verified = True
    except BaseException as exc:  # noqa: BLE001 - crash-observability: every failure becomes a receipt
        returncode = 70
        cleanup_error = f"{type(exc).__name__}: {exc}"[:1000]
    finally:
        worker.log.close()
    if origin_channel_error is not None:
        origin_detail = f"worker-origin channel failed: {origin_channel_error}"
        cleanup_error = f"{cleanup_error}; {origin_detail}" if cleanup_error else origin_detail
    if worker.worker_origin is not None:
        try:
            lifecycle = _finalize_broker_worker_origin(
                worker.worker_origin,
                successful=(
                    returncode == 0
                    and containment_verified
                    and not worker.timed_out
                    and cleanup_error is None
                ),
                occurred_at_unix=int(time.time()),
                reason=(
                    "worker_timed_out"
                    if worker.timed_out
                    else "worker_containment_failed"
                    if not containment_verified
                    else f"worker_exit_{returncode}"
                ),
            )
            event_origin = lifecycle["event_origin"]
            signed_payload = event_origin["signed_payload"]
            worker_origin_summary = {
                "artifact_path": str(worker.worker_origin.lifecycle_path),
                "artifact_sha256": lifecycle["artifact_sha256"],
                "event_type": signed_payload["event_type"],
                "event_sha256": event_origin["event_sha256"],
                "result_count": signed_payload["result_count"],
                "session_id": signed_payload["session_id"],
            }
            if lifecycle.get("completion_error") is not None:
                returncode = 70
                cleanup_error = (
                    f"worker-origin completion rejected: {lifecycle['completion_error']}"
                )
        except BaseException as exc:  # noqa: BLE001 - origin finalization is part of the broker verdict
            returncode = 70
            cleanup_error = (f"worker-origin finalization failed: {type(exc).__name__}: {exc}")[
                :1000
            ]
        finally:
            if worker.worker_origin_server is not None:
                worker.worker_origin_server.close()
    body = _broker_response_body(
        worker,
        returncode=returncode,
        containment_verified=containment_verified,
        cleanup_performed=cleanup_performed,
        cleanup_count=cleanup_count,
        error=cleanup_error,
        worker_origin_lifecycle=worker_origin_summary,
    )
    signed = {**body, "receipt_sha256": _sha256(body)}
    response = {
        **signed,
        "response_hmac_sha256": hmac.new(
            bytes.fromhex(worker.response_token),
            _canonical_bytes(signed),
            hashlib.sha256,
        ).hexdigest(),
    }
    _append_attempt_event(
        run_dir,
        {
            "event": "BROKER_TERMINAL",
            "attempt": attempt,
            "plan_sha256": plan["plan_sha256"],
            "request_id": worker.request_id,
            "policy_sha256": worker.policy_sha256,
            "response": response,
            "recorded_at": time.time(),
        },
    )
    try:
        reply_stat = worker.reply_path.lstat()
        if stat.S_ISSOCK(reply_stat.st_mode) and reply_stat.st_uid == os.geteuid():
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                sender.sendto(_canonical_bytes(response), str(worker.reply_path))
    except OSError:
        pass
    return response


def _cleanup_stale_control(control: dict[str, Any] | None) -> None:
    if control is None:
        return
    socket_path = Path(str(control.get("socket_path") or ""))
    try:
        socket_stat = socket_path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(socket_stat.st_mode) or socket_stat.st_uid != os.geteuid():
        raise DetachedStepError("stale control socket identity is invalid")
    socket_path.unlink()


def _terminal_receipt(
    *,
    plan: dict[str, Any],
    attempt: int,
    supervisor_pid: int,
    supervisor_start_token: str,
    child_pid: int,
    child_process_group_id: int,
    child_start_token: str,
    started_at: float,
    started_monotonic_ns: int,
    returncode: int,
    timed_out: bool,
    stop_signal: int | None,
    descendant_cleanup_performed: bool,
    lineage_cleanup_count: int,
    containment_verified: bool,
    supervisor_error: BaseException | None,
) -> dict[str, Any]:
    finished_at = time.time()
    duration_s = max(0.0, (time.monotonic_ns() - started_monotonic_ns) / 1_000_000_000)
    status = (
        "containment_failed"
        if not containment_verified
        else "supervisor_failed"
        if supervisor_error is not None
        else "timed_out"
        if timed_out
        else "stopped"
        if stop_signal
        else "passed"
        if returncode == 0
        else "failed"
    )
    body = {
        "schema": f"{SCHEMA_PREFIX}.receipt.v1",
        "name": plan["name"],
        "plan_sha256": plan["plan_sha256"],
        "command_sha256": plan["command_sha256"],
        "command": plan["command"],
        "executed_command": _executed_command(plan),
        "cwd": plan["cwd"],
        "timeout_s": plan["timeout_s"],
        "supervisor_attempt": attempt,
        "supervisor_pid": supervisor_pid,
        "supervisor_start_token": supervisor_start_token,
        "child_pid": child_pid,
        "child_process_group_id": child_process_group_id,
        "child_start_token": child_start_token,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(duration_s, 6),
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "stop_signal": stop_signal,
        "restart_count": 0,
        "descendant_cleanup_performed": descendant_cleanup_performed,
        "lineage_cleanup_count": lineage_cleanup_count,
        "lineage_empty": containment_verified,
        "process_group_empty": containment_verified,
        "fork_policy": plan["fork_policy"],
        "containment_verified": containment_verified,
        "status": status,
        "passed": status == "passed",
        "supervisor_error_type": type(supervisor_error).__name__ if supervisor_error else None,
        "supervisor_error": str(supervisor_error)[:1000] if supervisor_error else None,
    }
    return {**body, "receipt_sha256": _sha256(body)}


def _validate_supervisor_reservation(
    run_dir: Path,
    plan: dict[str, Any],
    attempt: int,
    supervisor_pid: int,
    supervisor_start_token: str,
) -> None:
    with _locked(run_dir):
        events = _read_attempts(run_dir)
        grouped = _events_by_attempt(events)
        launched = grouped.get(attempt, {}).get("LAUNCHED")
        if launched is None:
            raise DetachedStepError("supervisor launch reservation is missing")
        if (
            launched.get("plan_sha256") != plan["plan_sha256"]
            or int(launched.get("supervisor_pid") or 0) != supervisor_pid
            or launched.get("supervisor_start_token") != supervisor_start_token
            or "TARGET_STARTED" in grouped[attempt]
            or "TERMINAL" in grouped[attempt]
        ):
            raise DetachedStepError("supervisor launch reservation does not match the live process")


def _record_target_started(
    run_dir: Path,
    plan: dict[str, Any],
    attempt: int,
    supervisor_pid: int,
    supervisor_start_token: str,
    child: subprocess.Popen[Any],
    child_observation: ProcessObservation,
    containment_token: str,
) -> None:
    with _locked(run_dir):
        events = _read_attempts(run_dir)
        grouped = _events_by_attempt(events)
        launched = grouped.get(attempt, {}).get("LAUNCHED")
        if (
            launched is None
            or "TARGET_STARTED" in grouped[attempt]
            or "TERMINAL" in grouped[attempt]
        ):
            raise DetachedStepError("target start violates the attempt state machine")
        if (
            launched.get("plan_sha256") != plan["plan_sha256"]
            or int(launched.get("supervisor_pid") or 0) != supervisor_pid
            or launched.get("supervisor_start_token") != supervisor_start_token
            or child_observation.state != "alive"
            or child_observation.process_group_id != child.pid
        ):
            raise DetachedStepError("target start identity does not match its launch reservation")
        _append_attempt_event_locked(
            run_dir,
            {
                "event": "TARGET_STARTED",
                "attempt": attempt,
                "plan_sha256": plan["plan_sha256"],
                "supervisor_pid": supervisor_pid,
                "supervisor_start_token": supervisor_start_token,
                "child_pid": child.pid,
                "child_process_group_id": child_observation.process_group_id,
                "child_start_token": child_observation.token,
                "containment_token": containment_token,
                "gated_executable": child_observation.executable,
                "recorded_at": time.time(),
            },
        )


def _publish_terminal_receipt(
    run_dir: Path,
    plan: dict[str, Any],
    attempt: int,
    receipt: dict[str, Any],
) -> None:
    with _locked(run_dir):
        events = _read_attempts(run_dir)
        grouped = _events_by_attempt(events)
        existing = grouped.get(attempt, {}).get("TERMINAL")
        if existing is not None:
            if existing.get("receipt") != receipt:
                raise DetachedStepError("attempt already has a different terminal receipt")
            _materialize_terminal_receipt_locked(run_dir, existing)
            return
        launched = grouped.get(attempt, {}).get("LAUNCHED")
        if launched is None:
            raise DetachedStepError(
                "cannot publish a terminal receipt without a launch reservation"
            )
        terminal = _append_attempt_event_locked(
            run_dir,
            {
                "event": "TERMINAL",
                "attempt": attempt,
                "plan_sha256": plan["plan_sha256"],
                "supervisor_pid": receipt["supervisor_pid"],
                "supervisor_start_token": receipt["supervisor_start_token"],
                "receipt": receipt,
                "recorded_at": time.time(),
            },
        )
        if (
            os.environ.get("AURA_DETACHED_TEST_CRASH_POINT") == "after_terminal_journal"
            and "PYTEST_CURRENT_TEST" in os.environ
        ):
            os._exit(92)
        _materialize_terminal_receipt_locked(run_dir, terminal)


def _supervise(run_dir: Path, plan: dict[str, Any], attempt: int) -> None:
    supervisor_pid = os.getpid()
    supervisor_observation = _inspect_process(supervisor_pid)
    if supervisor_observation.state != "alive":
        raise DetachedStepError("could not establish supervisor process identity")
    supervisor_start_token = supervisor_observation.token
    _validate_supervisor_reservation(
        run_dir,
        plan,
        attempt,
        supervisor_pid,
        supervisor_start_token,
    )
    started_at = time.time()
    started_monotonic_ns = time.monotonic_ns()
    executed = _executed_command(plan)
    log_path = run_dir / LOG_FILE
    stop_signal: int | None = None
    child: subprocess.Popen[Any] | None = None
    child_group = 0
    child_start_token = ""
    gate_write_fd: int | None = None
    returncode: int | None = None
    timed_out = False
    sequence = 0
    supervisor_error: BaseException | None = None
    descendant_cleanup_performed = False
    lineage_cleanup_count = 0
    containment_verified = False
    containment_token = secrets.token_hex(32)
    control_socket: socket.socket | None = None
    control_socket_path: Path | None = None
    control_token = ""
    broker_token = ""
    power_assertion: subprocess.Popen[Any] | None = None
    active_broker: ActiveBrokerWorker | None = None
    prepared_origins: dict[str, PreparedBrokerWorkerOrigin] = {}
    broker_containment_verified = True

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_signal
        stop_signal = signum

    for watched in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(watched, request_stop)

    try:
        control_socket, control_socket_path, control_token, broker_token = _create_control_socket(
            run_dir,
            plan,
            attempt,
            supervisor_pid,
            supervisor_start_token,
        )
        with _open_secure_log(log_path) as log:
            target_environment = dict(plan["execution_environment"])
            target_environment.update(
                {
                    "AURA_DETACHED_RUN_TOKEN": containment_token,
                    "AURA_DETACHED_RUN_DIR": str(run_dir),
                    "AURA_DETACHED_PLAN_PATH": str(run_dir / PLAN_FILE),
                    "AURA_DETACHED_ATTEMPTS_PATH": str(run_dir / ATTEMPTS_FILE),
                    "AURA_DETACHED_PLAN_SHA256": str(plan["plan_sha256"]),
                    "AURA_DETACHED_SUPERVISOR_ATTEMPT": str(attempt),
                }
            )
            if broker_token:
                target_environment["AURA_DETACHED_BROKER_SOCKET"] = str(control_socket_path)
                target_environment["AURA_DETACHED_BROKER_TOKEN"] = broker_token
            child, gate_write_fd = _spawn_gated_target(
                executed,
                cwd=plan["cwd"],
                environment=target_environment,
                log=log,
            )
            child_observation = _inspect_process(child.pid)
            if child_observation.state != "alive":
                raise DetachedStepError("could not establish gated target process identity")
            child_group = child_observation.process_group_id
            child_start_token = child_observation.token
            _record_target_started(
                run_dir,
                plan,
                attempt,
                supervisor_pid,
                supervisor_start_token,
                child,
                child_observation,
                containment_token,
            )
            _verify_execution_manifest_current(plan["target_execution_manifest"])
            power_assertion = _start_power_assertion(child.pid, log, plan)
            os.write(gate_write_fd, b"G")
            os.close(gate_write_fd)
            gate_write_fd = None
            if (
                os.environ.get("AURA_DETACHED_TEST_FAULT_POINT") == "after_target_release"
                and "PYTEST_CURRENT_TEST" in os.environ
            ):
                raise RuntimeError("injected supervisor failure after target release")
            deadline_ns = started_monotonic_ns + int(float(plan["timeout_s"]) * 1_000_000_000)
        while returncode is None:
            sequence += 1
            request = _poll_control_socket(control_socket) if control_socket is not None else None
            if request is not None and request.get("action") == "stop":
                provided = request.get("control_token")
                if isinstance(provided, str) and secrets.compare_digest(provided, control_token):
                    stop_signal = signal.SIGTERM
            elif request is not None and request.get("action") == "run":
                provided = request.get("broker_token")
                if (
                    broker_token
                    and isinstance(provided, str)
                    and secrets.compare_digest(provided, broker_token)
                ):
                    if active_broker is not None:
                        _send_broker_rejection(
                            request,
                            child.pid,
                            BrokerRequestError("broker already has an active worker"),
                        )
                    else:
                        try:
                            active_broker = _start_broker_worker(
                                run_dir,
                                plan,
                                attempt,
                                supervisor_pid,
                                supervisor_start_token,
                                child.pid,
                                request,
                                prepared_origins,
                            )
                        except BrokerRequestError as broker_start_exc:
                            _send_broker_rejection(request, child.pid, broker_start_exc)
            if active_broker is not None and active_broker.worker_origin_server is not None:
                try:
                    active_broker.worker_origin_server.poll_once()
                except DetachedWorkerOriginChannelError as origin_channel_exc:
                    finished_broker = active_broker
                    active_broker = None
                    broker_response = _finish_broker_worker(
                        run_dir,
                        plan,
                        attempt,
                        finished_broker,
                        force_returncode=70,
                    )
                    if not broker_response["containment_verified"]:
                        broker_containment_verified = False
                        raise DetachedStepError(
                            "malformed worker-origin channel containment failed"
                        ) from origin_channel_exc
            if active_broker is not None:
                broker_returncode = active_broker.process.poll()
                if broker_returncode is not None:
                    finished_broker = active_broker
                    active_broker = None
                    broker_response = _finish_broker_worker(
                        run_dir,
                        plan,
                        attempt,
                        finished_broker,
                        force_returncode=broker_returncode,
                    )
                    if not broker_response["containment_verified"]:
                        broker_containment_verified = False
                        raise DetachedStepError("broker worker containment could not be verified")
                elif time.monotonic_ns() >= active_broker.deadline_ns:
                    active_broker.timed_out = True
                    finished_broker = active_broker
                    active_broker = None
                    broker_response = _finish_broker_worker(
                        run_dir,
                        plan,
                        attempt,
                        finished_broker,
                        force_returncode=124,
                    )
                    if not broker_response["containment_verified"]:
                        broker_containment_verified = False
                        raise DetachedStepError("timed-out broker worker containment failed")
            live_child, direct_returncode = _observe_direct_child(
                child,
                child_start_token,
                _IDENTITY_GRACE_S,
            )
            if direct_returncode is not None:
                returncode = direct_returncode
                if (
                    os.environ.get("AURA_DETACHED_TEST_CRASH_POINT") == "after_target_exit"
                    and "PYTEST_CURRENT_TEST" in os.environ
                ):
                    os._exit(91)
                break
            if (
                live_child.state == "alive"
                and live_child.token == child_start_token
                and live_child.process_group_id != child_group
            ):
                raise DetachedStepError("target escaped its declared process-group containment")
            if live_child.state == "unknown":
                # A direct child cannot be pid-reused while unreaped (the
                # kernel holds the zombie until waitpid), so libproc going
                # dark with poll() still pending is Darwin exit teardown —
                # a resident-scale model can spend >10s releasing Metal
                # buffers — not an identity violation. waitpid stays the
                # only exit authority; the step timeout still bounds the
                # wait, and containment is re-proven at cleanup.
                pass
            _publish_status(
                run_dir,
                {
                    "name": plan["name"],
                    "plan_sha256": plan["plan_sha256"],
                    "command_sha256": plan["command_sha256"],
                    "state": "stopping" if stop_signal else "running",
                    "supervisor_attempt": attempt,
                    "supervisor_pid": supervisor_pid,
                    "supervisor_start_token": supervisor_start_token,
                    "child_pid": child.pid,
                    "child_process_group_id": child_group,
                    "child_start_token": child_start_token,
                    "started_at": started_at,
                    "heartbeat_at": time.time(),
                    "heartbeat_sequence": sequence,
                    "restart_count": 0,
                },
            )
            if stop_signal is not None:
                break
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                timed_out = True
                returncode = 124
                break
            try:
                poll_interval = (
                    0.01
                    if active_broker is not None and active_broker.worker_origin_server is not None
                    else _POLL_S
                )
                returncode = child.wait(
                    timeout=min(
                        poll_interval,
                        remaining_ns / 1_000_000_000,
                    )
                )
                if (
                    returncode is not None
                    and os.environ.get("AURA_DETACHED_TEST_CRASH_POINT") == "after_target_exit"
                    and "PYTEST_CURRENT_TEST" in os.environ
                ):
                    os._exit(91)
            except subprocess.TimeoutExpired:
                continue
    except BaseException as exc:  # noqa: BLE001 - supervisor crash-observability: failure becomes receipt
        supervisor_error = exc
        if returncode is None:
            returncode = 70
    finally:
        if gate_write_fd is not None:
            try:
                os.close(gate_write_fd)
            except OSError:
                pass
        if child is not None:
            if active_broker is not None:
                try:
                    broker_response = _finish_broker_worker(
                        run_dir,
                        plan,
                        attempt,
                        active_broker,
                        force_returncode=70,
                    )
                    broker_containment_verified = (
                        bool(broker_response["containment_verified"])
                        and broker_containment_verified
                    )
                except BaseException as broker_cleanup_exc:  # noqa: BLE001 - broker cleanup error captured for the receipt
                    if supervisor_error is None:
                        supervisor_error = broker_cleanup_exc
                    returncode = 70
                active_broker = None
            for prepared_origin in prepared_origins.values():
                if prepared_origin.finalized:
                    continue
                try:
                    _finalize_broker_worker_origin(
                        prepared_origin,
                        successful=False,
                        occurred_at_unix=int(time.time()),
                        reason="detached_supervisor_shutdown",
                    )
                except BaseException as origin_cleanup_exc:  # noqa: BLE001 - authority cleanup is part of containment
                    if supervisor_error is None:
                        supervisor_error = origin_cleanup_exc
                    returncode = 70
            try:
                descendant_cleanup_performed, lineage_cleanup_count = _cleanup_child_process(
                    child,
                    child_start_token,
                    child_group,
                    containment_token,
                )
                containment_verified = (
                    plan.get("fork_policy") == "kernel_denied" and broker_containment_verified
                )
            except BaseException as cleanup_exc:  # noqa: BLE001 - cleanup error captured for the receipt
                if supervisor_error is None:
                    supervisor_error = cleanup_exc
                else:
                    supervisor_error = DetachedStepError(
                        f"{supervisor_error}; cleanup failed: {cleanup_exc}"
                    )
                returncode = 70
        else:
            containment_verified = (
                plan.get("fork_policy") == "kernel_denied" and broker_containment_verified
            )
        try:
            _stop_power_assertion(power_assertion)
        except BaseException as assertion_cleanup_exc:  # noqa: BLE001 - power-assertion cleanup error captured for the receipt
            containment_verified = False
            if supervisor_error is None:
                supervisor_error = assertion_cleanup_exc
            returncode = 70
        try:
            if control_socket is not None:
                control_socket.close()
            if control_socket_path is not None:
                control_socket_path.unlink(missing_ok=True)
        except OSError as control_cleanup_exc:
            if supervisor_error is None:
                supervisor_error = control_cleanup_exc
            returncode = 70
    if returncode is None:
        returncode = child.returncode if child is not None and child.returncode is not None else 1
    if stop_signal is not None and returncode == 0:
        returncode = 128 + stop_signal
    receipt = _terminal_receipt(
        plan=plan,
        attempt=attempt,
        supervisor_pid=supervisor_pid,
        supervisor_start_token=supervisor_start_token,
        child_pid=child.pid if child is not None else 0,
        child_process_group_id=child_group if child is not None else 0,
        child_start_token=child_start_token if child is not None else "",
        started_at=started_at,
        started_monotonic_ns=started_monotonic_ns,
        returncode=returncode,
        timed_out=timed_out,
        stop_signal=stop_signal,
        descendant_cleanup_performed=descendant_cleanup_performed,
        lineage_cleanup_count=lineage_cleanup_count,
        containment_verified=containment_verified,
        supervisor_error=supervisor_error,
    )
    _publish_terminal_receipt(run_dir, plan, attempt, receipt)
    _publish_status(
        run_dir,
        {
            "name": plan["name"],
            "plan_sha256": plan["plan_sha256"],
            "command_sha256": plan["command_sha256"],
            "state": receipt["status"],
            "supervisor_attempt": attempt,
            "supervisor_pid": supervisor_pid,
            "supervisor_start_token": supervisor_start_token,
            "child_pid": receipt["child_pid"],
            "child_process_group_id": receipt["child_process_group_id"],
            "child_start_token": receipt["child_start_token"],
            "started_at": started_at,
            "heartbeat_at": receipt["finished_at"],
            "heartbeat_sequence": sequence + 1,
            "restart_count": 0,
            "receipt_sha256": receipt["receipt_sha256"],
        },
    )


def _supervisor_bootstrap(
    release_fd_text: str,
    run_dir_text: str,
    attempt_text: str,
    run_dir_identity_json: str,
) -> int:
    """Enter supervisor custody in a clean interpreter spawned by the kernel."""

    try:
        release_fd = int(release_fd_text)
        attempt = int(attempt_text)
        identity_value = json.loads(run_dir_identity_json)
        identity = validate_directory_identity(identity_value)
    except (json.JSONDecodeError, TypeError, ValueError, SecurePathCustodyError):
        return 126
    if release_fd < 3 or attempt < 1:
        return 126
    run_dir = Path(run_dir_text).expanduser().absolute()
    plan_path = run_dir / PLAN_FILE
    plan: dict[str, Any] = {}
    try:
        with _run_directory_custody(
            run_dir,
            create=False,
            expected_identity=identity,
        ):
            plan = _read_json(plan_path)
            _verify_plan(plan, plan_path)
            try:
                release = os.read(release_fd, 1)
            finally:
                os.close(release_fd)
            if release != b"G":
                return 0
            os.chdir(plan["cwd"])
            os.umask(0o077)
            _supervise(run_dir, plan, attempt)
            return 0
    except BaseException as exc:  # noqa: BLE001 - supervisor last-resort: failure becomes status file
        supervisor_pid = os.getpid()
        supervisor_observation = _inspect_process(supervisor_pid)
        supervisor_start_token = supervisor_observation.token
        target: dict[str, Any] = {}
        try:
            attempts = _read_attempts(run_dir)
            grouped = _events_by_attempt(attempts)
            existing_terminal = grouped.get(attempt, {}).get("TERMINAL")
            if existing_terminal is not None:
                with _locked(run_dir):
                    _materialize_terminal_receipt_locked(run_dir, existing_terminal)
                os._exit(0)
            target = grouped.get(attempt, {}).get("TARGET_STARTED", {})
        except (OSError, DetachedStepError, ValueError):
            target = {}
        target_group = int(target.get("child_process_group_id") or 0)
        try:
            process_group_empty = not _process_group_exists(target_group)
        except DetachedStepError:
            process_group_empty = False
        failure_containment_verified = (
            process_group_empty and plan.get("fork_policy") == "kernel_denied"
        )
        failure_body = {
            "schema": f"{SCHEMA_PREFIX}.receipt.v1",
            "name": plan.get("name", ""),
            "plan_sha256": plan.get("plan_sha256", ""),
            "command_sha256": plan.get("command_sha256", ""),
            "command": plan.get("command", []),
            "executed_command": _executed_command(plan),
            "cwd": plan.get("cwd", ""),
            "timeout_s": plan.get("timeout_s", 0.0),
            "supervisor_attempt": attempt,
            "supervisor_pid": supervisor_pid,
            "supervisor_start_token": supervisor_start_token,
            "child_pid": int(target.get("child_pid") or 0),
            "child_process_group_id": target_group,
            "child_start_token": str(target.get("child_start_token") or ""),
            "started_at": time.time(),
            "finished_at": time.time(),
            "duration_s": 0.0,
            "returncode": 70,
            "timed_out": False,
            "stop_signal": None,
            "restart_count": 0,
            "descendant_cleanup_performed": False,
            "lineage_cleanup_count": 0,
            "lineage_empty": failure_containment_verified,
            "process_group_empty": failure_containment_verified,
            "fork_policy": plan.get("fork_policy", "unknown"),
            "containment_verified": failure_containment_verified,
            "status": "supervisor_failed" if target else "bootstrap_failed",
            "passed": False,
            "supervisor_error_type": type(exc).__name__,
            "supervisor_error": str(exc)[:1000],
        }
        failure = {**failure_body, "receipt_sha256": _sha256(failure_body)}
        try:
            _publish_terminal_receipt(run_dir, plan, attempt, failure)
        except (OSError, DetachedStepError):
            pass
        return 70


def _daemonize(run_dir: Path, plan: dict[str, Any], attempt: int) -> tuple[int, int]:
    """Spawn a detached supervisor without calling ``fork`` in this process."""

    custody = _ACTIVE_RUN_CUSTODIES.get(run_dir)
    if custody is None:
        raise DetachedStepError("detached supervisor lacks run-directory custody")
    release_read_fd, release_write_fd = os.pipe()
    null_fd = os.open(os.devnull, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
    # Preserve the virtualenv launcher path. Resolving its symlink would make
    # the clean interpreter lose pyvenv.cfg discovery and its site-packages.
    executable_path = Path(sys.executable).expanduser().absolute()
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        os.close(release_read_fd)
        os.close(release_write_fd)
        os.close(null_fd)
        raise DetachedStepError("detached supervisor interpreter is unavailable")
    executable = str(executable_path)
    launcher = str(Path(__file__).resolve(strict=True))
    argv = [
        executable,
        launcher,
        "_supervise",
        str(release_read_fd),
        str(run_dir),
        str(attempt),
        _canonical_bytes(custody.identity).decode("ascii"),
    ]
    file_actions = (
        (os.POSIX_SPAWN_DUP2, null_fd, 0),
        (os.POSIX_SPAWN_DUP2, null_fd, 1),
        (os.POSIX_SPAWN_DUP2, null_fd, 2),
        (os.POSIX_SPAWN_CLOSE, null_fd),
    )
    try:
        os.set_inheritable(release_read_fd, True)
        supervisor_pid = os.posix_spawn(
            executable,
            argv,
            os.environ.copy(),
            file_actions=file_actions,
            setsid=True,
            setsigmask=(),
            setsigdef=(signal.SIGINT, signal.SIGTERM, signal.SIGHUP),
        )
    except (OSError, NotImplementedError) as exc:
        os.close(release_write_fd)
        raise DetachedStepError("detached supervisor spawn failed") from exc
    finally:
        os.set_inheritable(release_read_fd, False)
        os.close(release_read_fd)
        os.close(null_fd)
    return supervisor_pid, release_write_fd


def _normalize_command(parser: argparse.ArgumentParser, raw: list[str]) -> list[str]:
    command = list(raw)
    if command and command[0] == "--":
        command = command[1:]
    if not command or any(not isinstance(item, str) or not item for item in command):
        parser.error("a non-empty command is required after --")
    return command


def _parse_optional_command_json(
    parser: argparse.ArgumentParser,
    raw: str,
) -> list[str] | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        parser.error("--resume-verifier-json must contain a JSON string array")
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        parser.error("--resume-verifier-json must contain a non-empty JSON string array")
    return value


def _parse_broker_policy_json(
    parser: argparse.ArgumentParser,
    raw: str,
) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        parser.error("--broker-policy-json must contain a JSON object array")
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) for item in value)
    ):
        parser.error("--broker-policy-json must contain a non-empty JSON object array")
    return value


def _build_worker_origin_policy(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    expected_keys = {
        "schema",
        "campaign_name",
        "protocol_sha256",
        "trust_policy_path",
        "trust_root_path",
        "artifact_dir",
        "arm",
        "worker_attempt_slot",
        "allowed_cells",
        "model_identity_sha256",
        "adapter_identity_sha256",
        "authorization_ttl_seconds",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema") != WORKER_ORIGIN_POLICY_SCHEMA
    ):
        raise DetachedStepError("worker-origin policy specification is invalid")
    campaign_name = str(value.get("campaign_name") or "")
    arm = str(value.get("arm") or "")
    if (
        not campaign_name
        or campaign_name != campaign_name.strip()
        or len(campaign_name) > 200
        or not arm
        or arm != arm.strip()
        or len(arm) > 200
    ):
        raise DetachedStepError("worker-origin campaign or arm identity is invalid")
    protocol_sha = _sha256_identifier(
        value.get("protocol_sha256"),
        role="worker-origin protocol",
    )
    model_identity_sha = _sha256_identifier(
        value.get("model_identity_sha256"),
        role="worker-origin model identity",
    )
    adapter_identity_sha = _sha256_identifier(
        value.get("adapter_identity_sha256"),
        role="worker-origin adapter identity",
    )
    attempt_slot = _positive_integer(
        value.get("worker_attempt_slot"),
        role="worker-origin attempt slot",
    )
    authorization_ttl = _positive_integer(
        value.get("authorization_ttl_seconds"),
        role="worker-origin authorization TTL",
        maximum=7 * 24 * 60 * 60,
    )
    allowed_cells = value.get("allowed_cells")
    if (
        not isinstance(allowed_cells, list)
        or not allowed_cells
        or len(allowed_cells) > _MAX_WORKER_ORIGIN_CELLS
    ):
        raise DetachedStepError("worker-origin allowed cells are invalid")
    try:
        normalized_cells = json.loads(canonical_json_bytes(allowed_cells))
        allowed_cell_digest = compute_allowed_cell_digest(normalized_cells)
    except (CampaignJournalError, WorkerOriginError) as exc:
        raise DetachedStepError("worker-origin allowed cells are invalid") from exc

    trust_policy_path = (
        Path(str(value.get("trust_policy_path") or "")).expanduser().resolve(strict=True)
    )
    trust_root_path = (
        Path(str(value.get("trust_root_path") or "")).expanduser().resolve(strict=True)
    )
    trust_policy_document = _read_canonical_private_json(
        trust_policy_path,
        max_bytes=_MAX_WORKER_ORIGIN_TRUST_POLICY_BYTES,
        role="worker-origin trust policy",
    )
    trust_root_pem = _read_stable_private_bytes(
        trust_root_path,
        max_bytes=_MAX_WORKER_ORIGIN_TRUST_ROOT_BYTES,
        role="worker-origin trust root",
    )
    try:
        verified_policy = validate_campaign_trust_policy(
            trust_policy_document,
            trusted_root_public_key_pem=trust_root_pem,
            expected_campaign_name=campaign_name,
            expected_protocol_sha256=protocol_sha,
            now_unix=int(time.time()),
        )
    except CampaignTrustError as exc:
        raise DetachedStepError(
            f"worker-origin trust policy is not admissible: {exc.code}"
        ) from exc

    artifact_dir = Path(str(value.get("artifact_dir") or "")).expanduser().resolve(strict=False)
    if (
        not artifact_dir.is_absolute()
        or artifact_dir == artifact_dir.parent
        or artifact_dir.parent.is_symlink()
        or not artifact_dir.parent.is_dir()
    ):
        raise DetachedStepError("worker-origin artifact directory is invalid")
    body = {
        "schema": WORKER_ORIGIN_POLICY_SCHEMA,
        "campaign_name": campaign_name,
        "protocol_sha256": protocol_sha,
        "trust_policy_path": str(trust_policy_path),
        "trust_policy_binding": _fingerprint_file(trust_policy_path),
        "trust_policy_document": verified_policy.document,
        "trust_policy_sha256": verified_policy.policy_sha256,
        "trust_root_path": str(trust_root_path),
        "trust_root_binding": _fingerprint_file(trust_root_path),
        "trust_root_public_key_pem_b64": base64.b64encode(trust_root_pem).decode("ascii"),
        "trust_root_key_id": verified_policy.root_key_id,
        "artifact_dir": str(artifact_dir),
        "arm": arm,
        "worker_attempt_slot": attempt_slot,
        "allowed_cells": normalized_cells,
        "allowed_cell_digest": allowed_cell_digest,
        "model_identity_sha256": model_identity_sha,
        "adapter_identity_sha256": adapter_identity_sha,
        "authorization_ttl_seconds": authorization_ttl,
    }
    return {**body, "contract_sha256": _sha256(body)}


def _verify_worker_origin_policy(value: Any, *, require_current: bool) -> dict[str, Any] | None:
    if value is None:
        return None
    expected_keys = {
        "schema",
        "campaign_name",
        "protocol_sha256",
        "trust_policy_path",
        "trust_policy_binding",
        "trust_policy_document",
        "trust_policy_sha256",
        "trust_root_path",
        "trust_root_binding",
        "trust_root_public_key_pem_b64",
        "trust_root_key_id",
        "artifact_dir",
        "arm",
        "worker_attempt_slot",
        "allowed_cells",
        "allowed_cell_digest",
        "model_identity_sha256",
        "adapter_identity_sha256",
        "authorization_ttl_seconds",
        "contract_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema") != WORKER_ORIGIN_POLICY_SCHEMA
    ):
        raise DetachedStepError("detached worker-origin contract is invalid")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    if value.get("contract_sha256") != _sha256(body):
        raise DetachedStepError("detached worker-origin contract hash is invalid")
    for role in (
        "protocol_sha256",
        "trust_policy_sha256",
        "allowed_cell_digest",
        "model_identity_sha256",
        "adapter_identity_sha256",
        "contract_sha256",
    ):
        _sha256_identifier(value.get(role), role=f"worker-origin {role}")
    _positive_integer(
        value.get("worker_attempt_slot"),
        role="worker-origin attempt slot",
    )
    _positive_integer(
        value.get("authorization_ttl_seconds"),
        role="worker-origin authorization TTL",
        maximum=7 * 24 * 60 * 60,
    )
    allowed_cells = value.get("allowed_cells")
    if (
        not isinstance(allowed_cells, list)
        or not allowed_cells
        or len(allowed_cells) > _MAX_WORKER_ORIGIN_CELLS
    ):
        raise DetachedStepError("detached worker-origin allowed cells are invalid")
    try:
        if compute_allowed_cell_digest(allowed_cells) != value["allowed_cell_digest"]:
            raise DetachedStepError("detached worker-origin allowed-cell digest is invalid")
    except WorkerOriginError as exc:
        raise DetachedStepError("detached worker-origin allowed cells are invalid") from exc
    trust_root_encoded = value.get("trust_root_public_key_pem_b64")
    if not isinstance(trust_root_encoded, str):
        raise DetachedStepError("detached worker-origin trust root is invalid")
    try:
        trust_root_pem = base64.b64decode(trust_root_encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise DetachedStepError("detached worker-origin trust root is invalid") from exc
    try:
        policy = validate_campaign_trust_policy(
            value.get("trust_policy_document"),
            trusted_root_public_key_pem=trust_root_pem,
            expected_campaign_name=str(value.get("campaign_name") or ""),
            expected_policy_sha256=str(value.get("trust_policy_sha256") or ""),
            expected_protocol_sha256=str(value.get("protocol_sha256") or ""),
            now_unix=int(time.time()) if require_current else None,
        )
    except CampaignTrustError as exc:
        raise DetachedStepError(
            f"detached worker-origin trust policy is invalid: {exc.code}"
        ) from exc
    if policy.root_key_id != value.get("trust_root_key_id"):
        raise DetachedStepError("detached worker-origin trust-root identity drifted")
    trust_policy_path = Path(str(value.get("trust_policy_path") or ""))
    trust_root_path = Path(str(value.get("trust_root_path") or ""))
    current_policy_document = _read_canonical_private_json(
        trust_policy_path,
        max_bytes=_MAX_WORKER_ORIGIN_TRUST_POLICY_BYTES,
        role="worker-origin trust policy",
    )
    current_trust_root = _read_stable_private_bytes(
        trust_root_path,
        max_bytes=_MAX_WORKER_ORIGIN_TRUST_ROOT_BYTES,
        role="worker-origin trust root",
    )
    if (
        current_policy_document != value.get("trust_policy_document")
        or current_trust_root != trust_root_pem
        or value.get("trust_policy_binding") != _fingerprint_file(trust_policy_path)
        or value.get("trust_root_binding") != _fingerprint_file(trust_root_path)
    ):
        raise DetachedStepError("detached worker-origin trust files changed")
    artifact_dir = Path(str(value.get("artifact_dir") or ""))
    if (
        not artifact_dir.is_absolute()
        or artifact_dir == artifact_dir.parent
        or not isinstance(value.get("arm"), str)
        or not value["arm"]
    ):
        raise DetachedStepError("detached worker-origin execution identity is invalid")
    if artifact_dir.exists() or artifact_dir.is_symlink():
        if artifact_dir.is_symlink() or not artifact_dir.is_dir():
            raise DetachedStepError("detached worker-origin artifact directory is invalid")
        artifact_dir_stat = artifact_dir.stat()
        if artifact_dir_stat.st_uid != os.geteuid() or artifact_dir_stat.st_mode & 0o077:
            raise DetachedStepError("detached worker-origin artifact directory is not private")
    return value


def _verify_persisted_worker_origin_quarantine(
    *,
    plan: dict[str, Any],
    policy: dict[str, Any],
    contract: dict[str, Any],
    attempt: int,
    broker_start: dict[str, Any],
    quarantine_event: dict[str, Any] | None,
    authorization: dict[str, Any],
    paths: dict[str, Path],
    lifecycle_artifact_sha256: str | None,
) -> None:
    if quarantine_event is None:
        return
    receipt = quarantine_event.get("quarantine_receipt")
    start_origin = broker_start.get("worker_origin")
    if not isinstance(receipt, dict) or not isinstance(start_origin, dict):
        raise DetachedStepError("worker-origin quarantine receipt is invalid")
    receipt_keys = {
        "schema",
        "plan_sha256",
        "broker_policy_sha256",
        "request_id",
        "supervisor_attempt",
        "supervisor_pid",
        "supervisor_start_token",
        "worker_pid",
        "worker_process_group_id",
        "worker_start_token",
        "containment_token",
        "worker_origin_contract_sha256",
        "session_id",
        "authorization_request_sha256",
        "authorization_attestation_sha256",
        "payload_path",
        "request_path",
        "attestation_path",
        "lifecycle_path",
        "lifecycle_artifact_sha256",
        "prior_journal_head_sha256",
        "supervisor_identity_observed",
        "worker_identity_observed",
        "worker_process_group_empty",
        "cleanup_action_performed",
        "authority_key_recoverable",
        "lifecycle_recoverable",
        "claim_eligible",
        "reason",
        "quarantined_at_unix",
        "receipt_sha256",
    }
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    quarantined_at = receipt.get("quarantined_at_unix")
    started_at = broker_start.get("recorded_at")
    if (
        set(receipt) != receipt_keys
        or receipt.get("schema") != WORKER_ORIGIN_QUARANTINE_RECEIPT_SCHEMA
        or receipt.get("receipt_sha256") != _sha256(receipt_body)
        or quarantine_event.get("request_id") != broker_start.get("request_id")
        or quarantine_event.get("policy_sha256") != policy["policy_sha256"]
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("broker_policy_sha256") != policy["policy_sha256"]
        or receipt.get("request_id") != broker_start.get("request_id")
        or int(receipt.get("supervisor_attempt") or 0) != attempt
        or int(receipt.get("supervisor_pid") or 0) != int(broker_start.get("supervisor_pid") or 0)
        or receipt.get("supervisor_start_token") != broker_start.get("supervisor_start_token")
        or int(receipt.get("worker_pid") or 0) != int(broker_start.get("worker_pid") or 0)
        or int(receipt.get("worker_process_group_id") or 0)
        != int(broker_start.get("worker_process_group_id") or 0)
        or receipt.get("worker_start_token") != broker_start.get("worker_start_token")
        or receipt.get("containment_token") != broker_start.get("containment_token")
        or receipt.get("worker_origin_contract_sha256") != contract["contract_sha256"]
        or receipt.get("session_id") != authorization["session_id"]
        or receipt.get("authorization_request_sha256")
        != start_origin.get("authorization_request_sha256")
        or receipt.get("authorization_attestation_sha256")
        != start_origin.get("authorization_attestation_sha256")
        or receipt.get("payload_path") != str(paths["payload"])
        or receipt.get("request_path") != str(paths["request"])
        or receipt.get("attestation_path") != str(paths["attestation"])
        or receipt.get("lifecycle_path") != str(paths["lifecycle"])
        or receipt.get("lifecycle_artifact_sha256") != lifecycle_artifact_sha256
        or receipt.get("prior_journal_head_sha256") != quarantine_event.get("previous_event_sha256")
        or receipt.get("supervisor_identity_observed") != "dead"
        or receipt.get("worker_identity_observed") != "dead"
        or receipt.get("worker_process_group_empty") is not True
        or not isinstance(receipt.get("cleanup_action_performed"), bool)
        or receipt.get("authority_key_recoverable") is not False
        or receipt.get("lifecycle_recoverable") is not False
        or receipt.get("claim_eligible") is not False
        or receipt.get("reason") != "supervisor_ephemeral_authority_lost"
        or isinstance(quarantined_at, bool)
        or not isinstance(quarantined_at, int)
        or isinstance(started_at, bool)
        or not isinstance(started_at, (int, float))
        or quarantined_at < int(float(started_at))
        or quarantine_event.get("recorded_at") != float(quarantined_at)
    ):
        raise DetachedStepError("worker-origin quarantine receipt binding is invalid")


def _verify_persisted_worker_origin_bundle(
    plan: dict[str, Any],
    policy: dict[str, Any],
    *,
    attempt: int,
    broker_start: dict[str, Any] | None,
    broker_response: dict[str, Any] | None,
    broker_quarantine: dict[str, Any] | None,
    attempt_terminal: bool,
) -> None:
    contract = _verify_worker_origin_policy(
        policy.get("worker_origin"),
        require_current=False,
    )
    start_origin = broker_start.get("worker_origin") if broker_start else None
    response_origin = (
        broker_response.get("worker_origin_lifecycle") if broker_response is not None else None
    )
    if contract is None:
        if start_origin is not None or response_origin is not None or broker_quarantine is not None:
            raise DetachedStepError(
                "broker journal asserts worker-origin custody without a contract"
            )
        return

    paths = _worker_origin_artifact_paths(
        contract,
        supervisor_attempt=attempt,
        broker_policy_sha256=policy["policy_sha256"],
    )
    present = {role: path.exists() or path.is_symlink() for role, path in paths.items()}
    if not any(present.values()):
        if broker_start is not None:
            raise DetachedStepError("broker worker-origin artifacts are missing")
        return
    if not present["payload"] or not present["request"]:
        raise DetachedStepError("worker-origin authority artifacts are incomplete")

    payload = _read_canonical_private_json(
        paths["payload"],
        max_bytes=_MAX_WORKER_ORIGIN_TRUST_POLICY_BYTES,
        role="worker-origin authorization payload",
    )
    request = _read_canonical_private_json(
        paths["request"],
        max_bytes=_MAX_WORKER_ORIGIN_TRUST_POLICY_BYTES,
        role="worker-origin authorization request",
    )
    try:
        authorization = validate_worker_authorization_payload(payload)
        trust_root_pem = base64.b64decode(
            contract["trust_root_public_key_pem_b64"],
            validate=True,
        )
        verified_policy = validate_campaign_trust_policy(
            contract["trust_policy_document"],
            trusted_root_public_key_pem=trust_root_pem,
            expected_campaign_name=contract["campaign_name"],
            expected_policy_sha256=contract["trust_policy_sha256"],
            expected_protocol_sha256=contract["protocol_sha256"],
        )
    except (CampaignTrustError, WorkerOriginError, TypeError, ValueError) as exc:
        raise DetachedStepError("persisted worker-origin authorization is invalid") from exc

    expected_authorization = {
        "campaign_name": contract["campaign_name"],
        "policy_sha256": contract["trust_policy_sha256"],
        "protocol_sha256": contract["protocol_sha256"],
        "detached_plan_sha256": plan["plan_sha256"],
        "broker_policy_sha256": policy["policy_sha256"],
        "executable_binding_sha256": policy["executable_binding"]["binding_sha256"],
        "environment_sha256": plan["execution_environment_sha256"],
        "sandbox_sha256": _sha256(plan["execution_sandbox"]),
        "source_manifest_sha256": policy["execution_manifest"]["manifest_sha256"],
        "supervisor_attempt": attempt,
        "arm": contract["arm"],
        "worker_attempt_slot": contract["worker_attempt_slot"],
        "allowed_cell_digest": contract["allowed_cell_digest"],
        "model_identity_sha256": contract["model_identity_sha256"],
        "adapter_identity_sha256": contract["adapter_identity_sha256"],
        "worker_key_custody": WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    }
    if any(authorization.get(key) != value for key, value in expected_authorization.items()):
        raise DetachedStepError("persisted worker-origin authorization binding is invalid")

    signed_payload = request.get("signed_payload")
    signed_at_unix = (
        signed_payload.get("signed_at_unix") if isinstance(signed_payload, dict) else None
    )
    if isinstance(signed_at_unix, bool) or not isinstance(signed_at_unix, int):
        raise DetachedStepError("worker-origin authorization request time is invalid")
    try:
        expected_request = prepare_role_signature_request(
            verified_policy,
            role=CAMPAIGN_RUNNER,
            payload=authorization,
            signed_at_unix=signed_at_unix,
        )
    except ValueError as exc:
        raise DetachedStepError("worker-origin authorization request is invalid") from exc
    if request != expected_request:
        raise DetachedStepError("worker-origin authorization request binding is invalid")

    attestation: dict[str, Any] | None = None
    if present["attestation"]:
        attestation = _read_canonical_private_json(
            paths["attestation"],
            max_bytes=_MAX_WORKER_ORIGIN_TRUST_POLICY_BYTES,
            role="worker-origin authorization attestation",
        )
        try:
            verify_worker_authorization(
                verified_policy,
                attestation,
                expected_payload=authorization,
                not_before_unix=signed_at_unix,
                not_after_unix=signed_at_unix,
            )
        except WorkerOriginError as exc:
            raise DetachedStepError("persisted worker-origin attestation is invalid") from exc
    if broker_start is not None:
        expected_start_keys = {
            "contract_sha256",
            "session_id",
            "authorization_payload",
            "authorization_request_sha256",
            "authorization_attestation_sha256",
            "request_path",
            "payload_path",
            "attestation_path",
            "lifecycle_path",
        }
        if (
            not isinstance(start_origin, dict)
            or set(start_origin) != expected_start_keys
            or attestation is None
            or start_origin.get("contract_sha256") != contract["contract_sha256"]
            or start_origin.get("session_id") != authorization["session_id"]
            or start_origin.get("authorization_payload") != authorization
            or start_origin.get("authorization_request_sha256") != request["request_sha256"]
            or start_origin.get("authorization_attestation_sha256")
            != hashlib.sha256(canonical_json_bytes(attestation)).hexdigest()
            or start_origin.get("request_path") != str(paths["request"])
            or start_origin.get("payload_path") != str(paths["payload"])
            or start_origin.get("attestation_path") != str(paths["attestation"])
            or start_origin.get("lifecycle_path") != str(paths["lifecycle"])
        ):
            raise DetachedStepError("attempt journal worker-origin start binding is invalid")
    elif start_origin is not None:
        raise DetachedStepError("orphaned worker-origin start metadata")

    if not isinstance(broker_start, dict):
        raise DetachedStepError("worker-origin broker start metadata is invalid")

    if not present["lifecycle"]:
        _verify_persisted_worker_origin_quarantine(
            plan=plan,
            policy=policy,
            contract=contract,
            attempt=attempt,
            broker_start=broker_start,
            quarantine_event=broker_quarantine,
            authorization=authorization,
            paths=paths,
            lifecycle_artifact_sha256=None,
        )
        if broker_response is not None or attempt_terminal:
            raise DetachedStepError("worker-origin lifecycle artifact is missing")
        if response_origin is not None:
            raise DetachedStepError("worker-origin lifecycle summary is orphaned")
        return

    lifecycle = _read_canonical_private_json(
        paths["lifecycle"],
        max_bytes=_MAX_WORKER_ORIGIN_TRUST_POLICY_BYTES,
        role="worker-origin lifecycle artifact",
    )
    lifecycle_keys = {
        "schema",
        "broker_policy_sha256",
        "authorization_payload",
        "authorization_request",
        "authorization_attestation",
        "event_origin",
        "completion_error",
        "artifact_sha256",
    }
    lifecycle_body = {key: value for key, value in lifecycle.items() if key != "artifact_sha256"}
    event_origin = lifecycle.get("event_origin")
    lifecycle_signed = (
        event_origin.get("signed_payload") if isinstance(event_origin, dict) else None
    )
    if (
        set(lifecycle) != lifecycle_keys
        or lifecycle.get("schema") != WORKER_ORIGIN_LIFECYCLE_ARTIFACT_SCHEMA
        or lifecycle.get("broker_policy_sha256") != policy["policy_sha256"]
        or lifecycle.get("authorization_payload") != authorization
        or lifecycle.get("authorization_request") != request
        or lifecycle.get("authorization_attestation") != attestation
        or lifecycle.get("artifact_sha256") != _sha256(lifecycle_body)
        or not isinstance(lifecycle_signed, dict)
    ):
        raise DetachedStepError("worker-origin lifecycle artifact binding is invalid")
    if not isinstance(event_origin, dict):
        raise DetachedStepError("worker-origin lifecycle event origin is invalid")
    _verify_persisted_worker_origin_quarantine(
        plan=plan,
        policy=policy,
        contract=contract,
        attempt=attempt,
        broker_start=broker_start,
        quarantine_event=broker_quarantine,
        authorization=authorization,
        paths=paths,
        lifecycle_artifact_sha256=str(lifecycle["artifact_sha256"]),
    )

    broker_passed = (
        broker_response is not None
        and broker_response.get("status") == "passed"
        and broker_response.get("returncode") == 0
        and broker_response.get("containment_verified") is True
        and broker_response.get("timed_out") is False
    )
    persisted_event_type = lifecycle_signed.get("event_type")
    expected_event_type = (
        "terminal"
        if broker_passed
        else "abandoned"
        if broker_response is not None
        else persisted_event_type
    )
    if expected_event_type not in {"terminal", "abandoned"}:
        raise DetachedStepError("worker-origin lifecycle event type is invalid")
    result_count = lifecycle_signed.get("result_count")
    occurred_at_unix = lifecycle_signed.get("occurred_at_unix")
    previous_origin_sha256 = lifecycle_signed.get("previous_origin_sha256")
    if (
        isinstance(result_count, bool)
        or not isinstance(result_count, int)
        or result_count < 0
        or result_count > len(contract["allowed_cells"])
        or isinstance(occurred_at_unix, bool)
        or not isinstance(occurred_at_unix, int)
        or occurred_at_unix < signed_at_unix
        or not isinstance(previous_origin_sha256, str)
        or len(previous_origin_sha256) != 64
    ):
        raise DetachedStepError("worker-origin lifecycle state is invalid")
    if broker_response is not None:
        started_at = broker_response.get("started_at")
        finished_at = broker_response.get("finished_at")
        if (
            isinstance(started_at, bool)
            or not isinstance(started_at, (int, float))
            or not math.isfinite(float(started_at))
            or isinstance(finished_at, bool)
            or not isinstance(finished_at, (int, float))
            or not math.isfinite(float(finished_at))
            or float(finished_at) < float(started_at)
            or occurred_at_unix < int(float(started_at)) - 1
            or occurred_at_unix > int(float(finished_at)) + 1
        ):
            raise DetachedStepError("worker-origin lifecycle time binding is invalid")
    if (
        (expected_event_type == "terminal" and result_count != len(contract["allowed_cells"]))
        or (result_count == 0 and previous_origin_sha256 != ZERO_SHA256)
        or (result_count > 0 and previous_origin_sha256 == ZERO_SHA256)
        or (broker_start is None and result_count != 0)
    ):
        raise DetachedStepError("worker-origin lifecycle result binding is invalid")
    expected_prior_state = (
        "running"
        if broker_start is not None
        else "authorized"
        if attestation is not None
        else "awaiting_external_signature"
    )
    expected_return_code = 0 if expected_event_type == "terminal" else None
    expected_reason = None if expected_event_type == "terminal" else lifecycle_signed.get("reason")
    completion_error = lifecycle.get("completion_error")
    if (expected_event_type == "terminal" and completion_error is not None) or (
        expected_event_type == "abandoned"
        and (
            (
                completion_error is not None
                and (
                    not isinstance(completion_error, str)
                    or expected_reason != f"completion_rejected:{completion_error}"
                )
            )
            or (
                completion_error is None
                and isinstance(expected_reason, str)
                and expected_reason.startswith("completion_rejected:")
            )
        )
    ):
        raise DetachedStepError("worker-origin lifecycle completion binding is invalid")
    try:
        verify_worker_lifecycle_event_origin(
            policy=verified_policy if attestation is not None else None,
            authorization_payload=authorization,
            authorization_attestation=attestation,
            event_origin=event_origin,
            expected_event_type=expected_event_type,
            expected_prior_state=expected_prior_state,
            expected_result_count=result_count,
            expected_previous_origin_sha256=previous_origin_sha256,
            expected_completed_cell_ids=[
                cell["cell_id"] for cell in contract["allowed_cells"][:result_count]
            ],
            expected_occurred_at_unix=occurred_at_unix,
            expected_return_code=expected_return_code,
            expected_reason=expected_reason,
        )
    except WorkerOriginError as exc:
        raise DetachedStepError("worker-origin lifecycle signature is invalid") from exc

    expected_summary = {
        "artifact_path": str(paths["lifecycle"]),
        "artifact_sha256": lifecycle["artifact_sha256"],
        "event_type": expected_event_type,
        "event_sha256": event_origin["event_sha256"],
        "result_count": result_count,
        "session_id": authorization["session_id"],
    }
    if broker_response is not None and response_origin != expected_summary:
        raise DetachedStepError("attempt journal worker-origin lifecycle summary is invalid")
    if broker_response is None and response_origin is not None:
        raise DetachedStepError("orphaned worker-origin lifecycle summary")


def _build_broker_policy(
    specifications: list[dict[str, Any]],
    environment: dict[str, str],
    *,
    execution_exclusion_roots: Iterable[Path] = (),
) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    seen_commands: set[str] = set()
    for specification in specifications:
        command = specification.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise DetachedStepError("broker policy command must be a non-empty string array")
        cwd = Path(str(specification.get("cwd") or "")).expanduser().resolve(strict=True)
        if not cwd.is_dir():
            raise DetachedStepError("broker policy cwd must be a directory")
        resolved_command = _resolve_command(command, cwd, environment)
        command_sha = _sha256(resolved_command)
        if command_sha in seen_commands:
            raise DetachedStepError("broker policy contains a duplicate command")
        seen_commands.add(command_sha)
        stdout_path = Path(str(specification.get("stdout_path") or "")).expanduser()
        if not stdout_path.is_absolute():
            stdout_path = cwd / stdout_path
        stdout_path = stdout_path.resolve(strict=False)
        if not stdout_path.parent.is_dir():
            raise DetachedStepError("broker policy stdout parent must exist")
        timeout_s_max = specification.get("timeout_s_max")
        max_invocations = specification.get("max_invocations")
        worker_origin = _build_worker_origin_policy(specification.get("worker_origin"))
        if (
            not isinstance(timeout_s_max, (int, float))
            or isinstance(timeout_s_max, bool)
            or not math.isfinite(float(timeout_s_max))
            or float(timeout_s_max) <= 0.0
            or not isinstance(max_invocations, int)
            or isinstance(max_invocations, bool)
            or max_invocations <= 0
            or max_invocations > 4096
            or (worker_origin is not None and max_invocations != 1)
        ):
            raise DetachedStepError("broker policy timeout or invocation bound is invalid")
        body = {
            "command": resolved_command,
            "command_sha256": command_sha,
            "executable_binding": _launcher_binding(Path(resolved_command[0])),
            "cwd": str(cwd),
            "stdout_path": str(stdout_path),
            "timeout_s_max": float(timeout_s_max),
            "max_invocations": max_invocations,
            "execution_manifest": _build_execution_manifest(
                resolved_command,
                cwd,
                excluded_roots=execution_exclusion_roots,
            ),
            "worker_origin": worker_origin,
        }
        policies.append({**body, "policy_sha256": _sha256(body)})
    return policies


def _build_plan(
    name: str,
    command: list[str],
    cwd: Path,
    timeout_s: float,
    resume_contract: str,
    resume_verifier: list[str] | None = None,
    broker_policy_specs: list[dict[str, Any]] | None = None,
    execution_exclusion_roots: Iterable[Path] = (),
    containment_mode: str = _SUPERVISOR_SANDBOX_MODE,
) -> dict[str, Any]:
    exclusions = _normalized_excluded_roots(execution_exclusion_roots)
    environment = _frozen_environment()
    resolved_command = _resolve_command(command, cwd, environment)
    executable_path = Path(resolved_command[0])
    executable_binding = _launcher_binding(executable_path)
    if sys.platform != "darwin" or not _DARWIN_SANDBOX.is_file():
        raise DetachedStepError(
            "strong detached containment requires the macOS sandbox-exec process-fork boundary"
        )
    if containment_mode not in _CONTAINMENT_MODES:
        raise DetachedStepError("detached containment mode is invalid")
    if containment_mode == _PRECONTAINED_SANDBOX_MODE:
        if (
            len(resolved_command) < 4
            or Path(resolved_command[0]) != _DARWIN_SANDBOX
            or resolved_command[1] != "-f"
        ):
            raise DetachedStepError("precontained target must start with exact sandbox-exec -f")
        profile_path = Path(resolved_command[2])
        if not profile_path.is_absolute():
            raise DetachedStepError("precontained sandbox profile path must be absolute")
        profile_payload = _read_stable_private_bytes(
            profile_path,
            max_bytes=_MAX_SANDBOX_PROFILE_BYTES,
            role="precontained sandbox profile",
        )
        try:
            profile = profile_payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DetachedStepError("precontained sandbox profile is not UTF-8") from exc
        if (
            any(marker not in profile for marker in _REQUIRED_PRECONTAINED_PROFILE_MARKERS)
            or "(allow default)" in profile
            or "(allow network" in profile
            or "(allow process-fork" in profile
        ):
            raise DetachedStepError("precontained sandbox profile lacks the required deny boundary")
        sandbox = {
            "mode": _PRECONTAINED_SANDBOX_MODE,
            "path": str(_DARWIN_SANDBOX),
            "sha256": _sha256_file(_DARWIN_SANDBOX),
            "profile_path": str(profile_path),
            "profile_sha256": hashlib.sha256(profile_payload).hexdigest(),
            "required_markers": list(_REQUIRED_PRECONTAINED_PROFILE_MARKERS),
        }
    else:
        sandbox = {
            "path": str(_DARWIN_SANDBOX),
            "sha256": _sha256_file(_DARWIN_SANDBOX),
            "profile": _NO_FORK_SANDBOX_PROFILE,
            "profile_sha256": hashlib.sha256(_NO_FORK_SANDBOX_PROFILE.encode("utf-8")).hexdigest(),
        }
    power_assertion = (
        {"path": str(_DARWIN_CAFFEINATE), "sha256": _sha256_file(_DARWIN_CAFFEINATE)}
        if _DARWIN_CAFFEINATE.is_file()
        else None
    )
    if resume_contract == "target_checkpoint" and resume_verifier is None:
        raise DetachedStepError("target_checkpoint contract requires a resume verifier command")
    if resume_contract == "none" and resume_verifier is not None:
        raise DetachedStepError("resume verifier command requires target_checkpoint contract")
    resolved_verifier = (
        _resolve_command(resume_verifier, cwd, environment) if resume_verifier is not None else None
    )
    command_sha256 = _sha256(resolved_command)
    target_execution_manifest = _build_execution_manifest(
        resolved_command,
        cwd,
        excluded_roots=exclusions,
    )
    verifier_execution_manifest = (
        _build_execution_manifest(
            resolved_verifier,
            cwd,
            excluded_roots=exclusions,
        )
        if resolved_verifier
        else None
    )
    broker_policy = _build_broker_policy(
        broker_policy_specs or [],
        environment,
        execution_exclusion_roots=exclusions,
    )
    body = {
        "schema": f"{SCHEMA_PREFIX}.plan.v2",
        "name": name,
        "command": resolved_command,
        "command_sha256": command_sha256,
        "executable_sha256": executable_binding["resolved_sha256"],
        "executable_binding": executable_binding,
        "execution_sandbox": sandbox,
        "power_assertion": power_assertion,
        "target_execution_manifest": target_execution_manifest,
        "execution_environment": environment,
        "execution_environment_sha256": _sha256(environment),
        "resume_verifier_command": resolved_verifier,
        "resume_verifier_command_sha256": _sha256(resolved_verifier) if resolved_verifier else None,
        "resume_verifier_executable_sha256": (
            _launcher_binding(Path(resolved_verifier[0]))["resolved_sha256"]
            if resolved_verifier
            else None
        ),
        "resume_verifier_executable_binding": (
            _launcher_binding(Path(resolved_verifier[0])) if resolved_verifier else None
        ),
        "resume_verifier_execution_manifest": verifier_execution_manifest,
        "broker_policy": broker_policy,
        "broker_policy_sha256": _sha256(broker_policy),
        "cwd": str(cwd),
        "timeout_s": float(timeout_s),
        "restart_policy": "never",
        "resume_contract": resume_contract,
        "session_escape_policy": "prohibited",
        "fork_policy": "kernel_denied",
        "containment_policy": (
            "precontained_deny_default_plus_process_identity_and_group"
            if containment_mode == _PRECONTAINED_SANDBOX_MODE
            else "sandbox_no_fork_plus_process_identity_and_group"
        ),
        "containment_environment_key": "AURA_DETACHED_RUN_TOKEN",
        "created_at": time.time(),
    }
    return {**body, "plan_sha256": _sha256(body)}


def _verified_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_json(path)
    claimed_hash = str(receipt.get("receipt_sha256") or "")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if not claimed_hash or claimed_hash != _sha256(body):
        raise DetachedStepError(f"terminal receipt hash mismatch: {path}")
    return receipt


def _comparable_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: plan.get(key)
        for key in (
            "name",
            "command",
            "command_sha256",
            "executable_sha256",
            "executable_binding",
            "execution_sandbox",
            "power_assertion",
            "target_execution_manifest",
            "execution_environment",
            "execution_environment_sha256",
            "resume_verifier_command",
            "resume_verifier_command_sha256",
            "resume_verifier_executable_sha256",
            "resume_verifier_executable_binding",
            "resume_verifier_execution_manifest",
            "broker_policy",
            "broker_policy_sha256",
            "cwd",
            "timeout_s",
            "restart_policy",
            "resume_contract",
            "session_escape_policy",
            "fork_policy",
            "containment_policy",
            "containment_environment_key",
        )
    }


def _verify_plan(plan: dict[str, Any], path: Path) -> None:
    if plan.get("schema") != f"{SCHEMA_PREFIX}.plan.v2":
        raise DetachedStepError(f"detached plan schema mismatch: {path}")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != _sha256(body):
        raise DetachedStepError(f"detached plan hash mismatch: {path}")
    command = plan.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise DetachedStepError(f"detached plan command is invalid: {path}")
    if plan.get("command_sha256") != _sha256(command):
        raise DetachedStepError(f"detached plan command hash mismatch: {path}")
    environment = plan.get("execution_environment")
    if (
        not isinstance(environment, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        )
        or plan.get("execution_environment_sha256") != _sha256(environment)
    ):
        raise DetachedStepError(f"detached plan environment binding is invalid: {path}")
    executable = Path(command[0])
    if not executable.is_absolute() or not executable.is_file():
        raise DetachedStepError(f"detached plan executable is unavailable: {path}")
    executable_binding = _verify_launcher_binding(plan.get("executable_binding"), executable)
    if plan.get("executable_sha256") != executable_binding["resolved_sha256"]:
        raise DetachedStepError(f"detached plan executable hash mismatch: {path}")
    sandbox = plan.get("execution_sandbox")
    if not isinstance(sandbox, dict):
        raise DetachedStepError(f"detached plan sandbox binding is invalid: {path}")
    sandbox_path = Path(str(sandbox.get("path") or ""))
    if sandbox.get("mode") == _PRECONTAINED_SANDBOX_MODE:
        profile_path = Path(str(sandbox.get("profile_path") or ""))
        if (
            sandbox_path != _DARWIN_SANDBOX
            or sandbox.get("sha256") != _sha256_file(sandbox_path)
            or len(command) < 4
            or command[0] != str(_DARWIN_SANDBOX)
            or command[1] != "-f"
            or command[2] != str(profile_path)
            or not profile_path.is_absolute()
            or sandbox.get("required_markers") != list(_REQUIRED_PRECONTAINED_PROFILE_MARKERS)
        ):
            raise DetachedStepError(f"detached precontained sandbox binding mismatch: {path}")
        profile_payload = _read_stable_private_bytes(
            profile_path,
            max_bytes=_MAX_SANDBOX_PROFILE_BYTES,
            role="precontained sandbox profile",
        )
        try:
            profile = profile_payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DetachedStepError(
                f"detached precontained sandbox profile is invalid: {path}"
            ) from exc
        if (
            sandbox.get("profile_sha256") != hashlib.sha256(profile_payload).hexdigest()
            or any(marker not in profile for marker in _REQUIRED_PRECONTAINED_PROFILE_MARKERS)
            or "(allow default)" in profile
            or "(allow network" in profile
            or "(allow process-fork" in profile
        ):
            raise DetachedStepError(f"detached precontained sandbox profile drift: {path}")
    elif (
        sandbox_path != _DARWIN_SANDBOX
        or sandbox.get("sha256") != _sha256_file(sandbox_path)
        or sandbox.get("profile") != _NO_FORK_SANDBOX_PROFILE
        or sandbox.get("profile_sha256")
        != hashlib.sha256(_NO_FORK_SANDBOX_PROFILE.encode("utf-8")).hexdigest()
    ):
        raise DetachedStepError(f"detached plan sandbox hash mismatch: {path}")
    power_assertion = plan.get("power_assertion")
    if power_assertion is not None:
        if not isinstance(power_assertion, dict):
            raise DetachedStepError(f"detached plan power assertion binding is invalid: {path}")
        assertion_path = Path(str(power_assertion.get("path") or ""))
        if assertion_path != _DARWIN_CAFFEINATE or power_assertion.get("sha256") != _sha256_file(
            assertion_path
        ):
            raise DetachedStepError(f"detached plan power assertion hash mismatch: {path}")
    _verify_execution_manifest_structure(plan.get("target_execution_manifest"))
    if plan.get("restart_policy") != "never":
        raise DetachedStepError(f"detached plan restart policy is invalid: {path}")
    if plan.get("resume_contract") not in {"none", "target_checkpoint"}:
        raise DetachedStepError(f"detached plan resume contract is invalid: {path}")
    verifier = plan.get("resume_verifier_command")
    if plan.get("resume_contract") == "target_checkpoint":
        if not isinstance(verifier, list) or not verifier:
            raise DetachedStepError(f"detached plan resume verifier is missing: {path}")
        if (
            plan.get("resume_verifier_command_sha256") != _sha256(verifier)
            or not Path(verifier[0]).is_absolute()
            or plan.get("resume_verifier_executable_sha256")
            != _verify_launcher_binding(
                plan.get("resume_verifier_executable_binding"),
                Path(verifier[0]),
            )["resolved_sha256"]
        ):
            raise DetachedStepError(f"detached plan resume verifier binding is invalid: {path}")
        _verify_execution_manifest_structure(plan.get("resume_verifier_execution_manifest"))
    elif any(
        plan.get(key) is not None
        for key in (
            "resume_verifier_command",
            "resume_verifier_command_sha256",
            "resume_verifier_executable_sha256",
            "resume_verifier_executable_binding",
            "resume_verifier_execution_manifest",
        )
    ):
        raise DetachedStepError(f"detached plan has an unexpected resume verifier: {path}")
    broker_policy = plan.get("broker_policy")
    if not isinstance(broker_policy, list) or plan.get("broker_policy_sha256") != _sha256(
        broker_policy
    ):
        raise DetachedStepError(f"detached plan broker policy binding is invalid: {path}")
    seen_broker_commands: set[str] = set()
    for policy in broker_policy:
        if not isinstance(policy, dict):
            raise DetachedStepError(f"detached plan broker policy entry is invalid: {path}")
        body = {key: value for key, value in policy.items() if key != "policy_sha256"}
        command = policy.get("command")
        command_sha = str(policy.get("command_sha256") or "")
        cwd_value = Path(str(policy.get("cwd") or ""))
        stdout_path = Path(str(policy.get("stdout_path") or ""))
        timeout_max = policy.get("timeout_s_max")
        max_invocations = policy.get("max_invocations")
        worker_origin = policy.get("worker_origin")
        if (
            policy.get("policy_sha256") != _sha256(body)
            or not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
            or command_sha != _sha256(command)
            or command_sha in seen_broker_commands
            or not cwd_value.is_absolute()
            or not stdout_path.is_absolute()
            or not isinstance(timeout_max, (int, float))
            or isinstance(timeout_max, bool)
            or not math.isfinite(float(timeout_max))
            or float(timeout_max) <= 0.0
            or not isinstance(max_invocations, int)
            or isinstance(max_invocations, bool)
            or max_invocations <= 0
            or (worker_origin is not None and max_invocations != 1)
        ):
            raise DetachedStepError(f"detached plan broker policy entry binding is invalid: {path}")
        _verify_launcher_binding(policy.get("executable_binding"), Path(command[0]))
        _verify_execution_manifest_structure(policy.get("execution_manifest"))
        _verify_worker_origin_policy(worker_origin, require_current=False)
        seen_broker_commands.add(command_sha)
    if plan.get("session_escape_policy") != "prohibited":
        raise DetachedStepError(f"detached plan containment policy is invalid: {path}")
    expected_containment_policy = (
        "precontained_deny_default_plus_process_identity_and_group"
        if sandbox.get("mode") == _PRECONTAINED_SANDBOX_MODE
        else "sandbox_no_fork_plus_process_identity_and_group"
    )
    if plan.get("fork_policy") != "kernel_denied" or (
        plan.get("containment_policy") != expected_containment_policy
        or plan.get("containment_environment_key") != "AURA_DETACHED_RUN_TOKEN"
    ):
        raise DetachedStepError(f"detached plan lineage containment policy is invalid: {path}")


def _events_by_attempt(events: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for event in events:
        event_type = str(event["event"])
        key = (
            f"{event_type}:{event.get('request_id')}"
            if event_type
            in {
                "BROKER_STARTED",
                "BROKER_TERMINAL",
                "BROKER_ORIGIN_QUARANTINED",
            }
            else event_type
        )
        grouped.setdefault(int(event["attempt"]), {})[key] = event
    return grouped


def _broker_events(
    attempt_events: dict[str, dict[str, Any]], event_type: str
) -> list[dict[str, Any]]:
    prefix = f"{event_type}:"
    return [event for key, event in attempt_events.items() if key.startswith(prefix)]


def _materialize_terminal_receipt_locked(run_dir: Path, terminal: dict[str, Any]) -> dict[str, Any]:
    receipt_value = terminal.get("receipt")
    if not isinstance(receipt_value, dict):
        raise DetachedStepError("authoritative terminal receipt is invalid")
    receipt: dict[str, Any] = receipt_value
    receipt_path = run_dir / RECEIPT_FILE
    if receipt_path.exists():
        materialized = _verified_receipt(receipt_path)
        if materialized != receipt:
            raise DetachedStepError(
                "terminal receipt differs from the authoritative journal record"
            )
    else:
        _atomic_write(receipt_path, receipt, replace=False)
    return receipt


def _verify_persisted_resume_verdict(
    run_dir: Path,
    plan: dict[str, Any],
    attempt: int,
    launched: dict[str, Any],
) -> None:
    verdict = launched.get("resume_verdict")
    if attempt == 1:
        if verdict is not None:
            raise DetachedStepError("initial attempt has an unexpected resume verdict")
        return
    prior_head = str(launched.get("previous_event_sha256") or "")
    if (
        plan.get("resume_contract") != "target_checkpoint"
        or not isinstance(verdict, dict)
        or verdict.get("schema") != f"{SCHEMA_PREFIX}.resume_verdict.v3"
        or verdict.get("plan_sha256") != plan["plan_sha256"]
        or verdict.get("command_sha256") != plan.get("command_sha256")
        or verdict.get("prior_attempt") != attempt - 1
        or verdict.get("prior_journal_head_sha256") != prior_head
        or verdict.get("verdict") != "safe_to_resume"
    ):
        raise DetachedStepError("attempt journal resume verdict is invalid")
    evidence = verdict.get("evidence")
    if not isinstance(evidence, dict):
        raise DetachedStepError("attempt journal resume evidence is invalid")
    evidence_sha = _sha256(evidence)
    checkpoint_sequence = verdict.get("checkpoint_sequence")
    if (
        evidence_sha != verdict.get("evidence_sha256")
        or not isinstance(checkpoint_sequence, int)
        or isinstance(checkpoint_sequence, bool)
        or checkpoint_sequence < 0
        or evidence.get("checkpoint_sequence") != checkpoint_sequence
    ):
        raise DetachedStepError("attempt journal resume evidence binding is invalid")
    expected_identity = _sha256(
        {
            "prior_attempt": attempt - 1,
            "prior_journal_head_sha256": prior_head,
            "checkpoint_sequence": checkpoint_sequence,
            "evidence_sha256": evidence_sha,
        }
    )
    if verdict.get("checkpoint_identity") != expected_identity:
        raise DetachedStepError("attempt journal checkpoint identity is invalid")


def _verify_run_locked(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    plan_path = run_dir / PLAN_FILE
    plan = _read_json(plan_path)
    _verify_plan(plan, plan_path)
    events = _read_attempts(run_dir)
    grouped = _events_by_attempt(events)
    latest_attempt = max(grouped, default=0)
    plan_sha = plan["plan_sha256"]
    for attempt, attempt_events in grouped.items():
        launched = attempt_events.get("LAUNCHED")
        if launched is None:
            raise DetachedStepError("attempt journal is missing a launch reservation")
        for event in attempt_events.values():
            if event.get("plan_sha256") != plan_sha:
                raise DetachedStepError("attempt journal plan binding mismatch")
        supervisor_pid = int(launched.get("supervisor_pid") or 0)
        supervisor_token = str(launched.get("supervisor_start_token") or "")
        if supervisor_pid <= 0 or not supervisor_token:
            raise DetachedStepError("attempt journal supervisor identity is invalid")
        _verify_persisted_resume_verdict(run_dir, plan, attempt, launched)
        control = attempt_events.get("CONTROL_READY")
        if control is not None and (
            int(control.get("supervisor_pid") or 0) != supervisor_pid
            or control.get("supervisor_start_token") != supervisor_token
            or not isinstance(control.get("socket_path"), str)
            or len(str(control.get("control_token") or "")) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(control.get("control_token") or "")
            )
            or bool(control.get("broker_enabled")) != bool(plan["broker_policy"])
            or (
                len(str(control.get("broker_token") or "")) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(control.get("broker_token") or "")
                )
                if plan["broker_policy"]
                else bool(control.get("broker_token"))
            )
        ):
            raise DetachedStepError("attempt journal control identity is invalid")
        target = attempt_events.get("TARGET_STARTED")
        if target is not None:
            if (
                control is None
                or int(target.get("supervisor_pid") or 0) != supervisor_pid
                or target.get("supervisor_start_token") != supervisor_token
                or int(target.get("child_pid") or 0) <= 0
                or int(target.get("child_process_group_id") or 0) <= 1
                or not target.get("child_start_token")
                or len(str(target.get("containment_token") or "")) != 64
            ):
                raise DetachedStepError("attempt journal target identity is invalid")
        policy_by_sha = {str(policy["policy_sha256"]): policy for policy in plan["broker_policy"]}
        broker_starts = _broker_events(attempt_events, "BROKER_STARTED")
        broker_terminals = {
            str(event.get("request_id") or ""): event
            for event in _broker_events(attempt_events, "BROKER_TERMINAL")
        }
        broker_quarantines = {
            str(event.get("request_id") or ""): event
            for event in _broker_events(
                attempt_events,
                "BROKER_ORIGIN_QUARANTINED",
            )
        }
        attempt_terminal = attempt_events.get("TERMINAL") is not None
        invocation_counts: dict[str, int] = {}
        for broker_start in broker_starts:
            request_id = str(broker_start.get("request_id") or "")
            policy_sha = str(broker_start.get("policy_sha256") or "")
            policy = policy_by_sha.get(policy_sha)
            invocation_counts[policy_sha] = invocation_counts.get(policy_sha, 0) + 1
            if (
                target is None
                or policy is None
                or int(broker_start.get("supervisor_pid") or 0) != supervisor_pid
                or broker_start.get("supervisor_start_token") != supervisor_token
                or broker_start.get("command_sha256") != policy["command_sha256"]
                or int(broker_start.get("worker_pid") or 0) <= 0
                or int(broker_start.get("worker_process_group_id") or 0)
                != int(broker_start.get("worker_pid") or 0)
                or not broker_start.get("worker_start_token")
                or len(str(broker_start.get("containment_token") or "")) != 64
                or len(request_id) != 32
            ):
                raise DetachedStepError("attempt journal broker worker identity is invalid")
            broker_terminal = broker_terminals.get(request_id)
            broker_quarantine = broker_quarantines.get(request_id)
            if broker_terminal is None:
                _verify_persisted_worker_origin_bundle(
                    plan,
                    policy,
                    attempt=attempt,
                    broker_start=broker_start,
                    broker_response=None,
                    broker_quarantine=broker_quarantine,
                    attempt_terminal=attempt_terminal,
                )
                if (
                    policy.get("worker_origin") is not None
                    and attempt < latest_attempt
                    and broker_quarantine is None
                ):
                    raise DetachedStepError(
                        "historical unfinished worker-origin slot is not quarantined"
                    )
                continue
            response = broker_terminal.get("response")
            if not isinstance(response, dict):
                raise DetachedStepError("attempt journal broker response is invalid")
            response_signed = {
                key: value for key, value in response.items() if key != "response_hmac_sha256"
            }
            response_body = {
                key: value for key, value in response_signed.items() if key != "receipt_sha256"
            }
            response_hmac = str(response.get("response_hmac_sha256") or "")
            broker_token = str(control.get("broker_token") or "") if control is not None else ""
            if (
                broker_terminal.get("policy_sha256") != policy_sha
                or response.get("receipt_sha256") != _sha256(response_body)
                or len(response_hmac) != 64
                or len(broker_token) != 64
                or not hmac.compare_digest(
                    response_hmac,
                    hmac.new(
                        bytes.fromhex(broker_token),
                        _canonical_bytes(response_signed),
                        hashlib.sha256,
                    ).hexdigest(),
                )
                or response.get("schema") != f"{SCHEMA_PREFIX}.broker_response.v1"
                or response.get("request_id") != request_id
                or response.get("policy_sha256") != policy_sha
                or response.get("command_sha256") != policy["command_sha256"]
                or int(response.get("worker_pid") or 0) != int(broker_start.get("worker_pid") or 0)
                or response.get("worker_start_token") != broker_start.get("worker_start_token")
                or not isinstance(response.get("containment_verified"), bool)
                or (
                    not response.get("containment_verified")
                    and response.get("status") != "containment_failed"
                )
                or (
                    response.get("containment_verified")
                    and response.get("status") == "containment_failed"
                )
            ):
                raise DetachedStepError("attempt journal broker terminal binding is invalid")
            _verify_persisted_worker_origin_bundle(
                plan,
                policy,
                attempt=attempt,
                broker_start=broker_start,
                broker_response=response,
                broker_quarantine=broker_quarantine,
                attempt_terminal=attempt_terminal,
            )
        for policy_sha, policy in policy_by_sha.items():
            if invocation_counts.get(policy_sha, 0) == 0:
                _verify_persisted_worker_origin_bundle(
                    plan,
                    policy,
                    attempt=attempt,
                    broker_start=None,
                    broker_response=None,
                    broker_quarantine=None,
                    attempt_terminal=attempt_terminal,
                )
        if any(
            count > int(policy_by_sha[policy_sha]["max_invocations"])
            for policy_sha, count in invocation_counts.items()
            if policy_sha in policy_by_sha
        ):
            raise DetachedStepError("attempt journal broker invocation bound exceeded")
        terminal = attempt_events.get("TERMINAL")
        if terminal is not None:
            terminal_receipt = terminal["receipt"]
            if (
                terminal_receipt.get("schema") != f"{SCHEMA_PREFIX}.receipt.v1"
                or terminal_receipt.get("plan_sha256") != plan_sha
                or int(terminal_receipt.get("supervisor_attempt") or 0) != attempt
                or int(terminal_receipt.get("supervisor_pid") or 0) != supervisor_pid
                or terminal_receipt.get("supervisor_start_token") != supervisor_token
                or int(terminal.get("supervisor_pid") or 0) != supervisor_pid
                or terminal.get("supervisor_start_token") != supervisor_token
                or terminal_receipt.get("command_sha256") != plan.get("command_sha256")
                or terminal_receipt.get("command") != plan.get("command")
                or terminal_receipt.get("restart_count") != 0
                or terminal_receipt.get("fork_policy") != plan.get("fork_policy")
                or terminal_receipt.get("process_group_empty")
                != terminal_receipt.get("containment_verified")
                or terminal_receipt.get("lineage_empty")
                != terminal_receipt.get("containment_verified")
            ):
                raise DetachedStepError("terminal receipt supervisor binding mismatch")
            if target is not None and (
                int(terminal_receipt.get("child_pid") or 0) != int(target.get("child_pid") or 0)
                or int(terminal_receipt.get("child_process_group_id") or 0)
                != int(target.get("child_process_group_id") or 0)
                or terminal_receipt.get("child_start_token") != target.get("child_start_token")
            ):
                raise DetachedStepError("terminal receipt target binding mismatch")
            if len(broker_terminals) != len(broker_starts):
                raise DetachedStepError("terminal attempt has an unfinished broker worker")
    terminal_events = [event for event in events if event.get("event") == "TERMINAL"]
    receipt: dict[str, Any] | None = None
    if terminal_events:
        receipt = _materialize_terminal_receipt_locked(run_dir, terminal_events[-1])
    elif (run_dir / RECEIPT_FILE).exists():
        raise DetachedStepError("terminal receipt exists without an authoritative journal record")
    status_path = run_dir / STATUS_FILE
    status = _read_json(status_path) if status_path.is_file() else {}
    if status:
        if (
            status.get("schema") != f"{SCHEMA_PREFIX}.status.v1"
            or status.get("plan_sha256") != plan_sha
        ):
            raise DetachedStepError("detached status binding mismatch")
        status_attempt = int(status.get("supervisor_attempt") or 0)
        if status_attempt not in grouped:
            raise DetachedStepError("detached status references an unknown attempt")
        launched = grouped[status_attempt]["LAUNCHED"]
        if int(status.get("supervisor_pid") or 0) != int(
            launched.get("supervisor_pid") or 0
        ) or status.get("supervisor_start_token") != launched.get("supervisor_start_token"):
            raise DetachedStepError("detached status supervisor identity mismatch")
        target = grouped[status_attempt].get("TARGET_STARTED")
        if int(status.get("child_pid") or 0) > 0 and (
            target is None
            or int(status.get("child_pid") or 0) != int(target.get("child_pid") or 0)
            or int(status.get("child_process_group_id") or 0)
            != int(target.get("child_process_group_id") or 0)
            or status.get("child_start_token") != target.get("child_start_token")
        ):
            raise DetachedStepError("detached status target identity mismatch")
    return plan, events, status, receipt


def validate_resume_verdict(
    verdict: Any,
    *,
    plan_sha256: str,
    command_sha256: str,
    prior_attempt: int,
    prior_journal_head_sha256: str,
) -> dict[str, Any]:
    """Accept one resume verdict exactly as the supervisor does.

    Every resume verifier is a separate process, so nothing links them to this
    contract at import time -- a version bump here strands its implementations
    silently until a campaign resumes at 3am.  Verifier tests call this to prove
    their real output against the only consumer that matters.

    Structural acceptance only: an ``indeterminate`` or ``already_completed``
    verdict is well-formed.  Whether resuming is permitted is the caller's call.
    """
    if not isinstance(verdict, dict):
        raise DetachedStepError("target checkpoint verifier verdict must be an object")
    evidence_sha = str(verdict.get("evidence_sha256") or "")
    evidence = verdict.get("evidence")
    checkpoint_sequence = verdict.get("checkpoint_sequence")
    checkpoint_identity = str(verdict.get("checkpoint_identity") or "")
    if (
        verdict.get("schema") != f"{SCHEMA_PREFIX}.resume_verdict.v3"
        or verdict.get("plan_sha256") != plan_sha256
        or verdict.get("command_sha256") != command_sha256
        or verdict.get("prior_attempt") != prior_attempt
        or verdict.get("prior_journal_head_sha256") != prior_journal_head_sha256
        or not isinstance(evidence, dict)
        or not isinstance(checkpoint_sequence, int)
        or isinstance(checkpoint_sequence, bool)
        or checkpoint_sequence < 0
        or verdict.get("verdict") not in {"safe_to_resume", "already_completed", "indeterminate"}
        or len(evidence_sha) != 64
        or len(checkpoint_identity) != 64
        or any(
            character not in "0123456789abcdef" for character in evidence_sha + checkpoint_identity
        )
    ):
        raise DetachedStepError("target checkpoint verifier verdict binding is invalid")
    artifact_sha = _sha256(evidence)
    if artifact_sha != evidence_sha:
        raise DetachedStepError("target checkpoint evidence binding is invalid")
    if (
        evidence.get("schema") != f"{SCHEMA_PREFIX}.resume_evidence.v2"
        or evidence.get("plan_sha256") != plan_sha256
        or evidence.get("command_sha256") != command_sha256
        or evidence.get("prior_attempt") != prior_attempt
        or evidence.get("prior_journal_head_sha256") != prior_journal_head_sha256
        or evidence.get("checkpoint_sequence") != checkpoint_sequence
    ):
        raise DetachedStepError("target checkpoint evidence content is not attempt-bound")
    expected_checkpoint_identity = _sha256(
        {
            "prior_attempt": prior_attempt,
            "prior_journal_head_sha256": prior_journal_head_sha256,
            "checkpoint_sequence": checkpoint_sequence,
            "evidence_sha256": evidence_sha,
        }
    )
    if checkpoint_identity != expected_checkpoint_identity:
        raise DetachedStepError("target checkpoint identity is invalid")
    return verdict


def _run_resume_verifier(
    plan: dict[str, Any],
    run_dir: Path,
    prior_attempt: int,
    prior_journal_head_sha256: str,
) -> dict[str, Any]:
    verifier = plan.get("resume_verifier_command")
    if not isinstance(verifier, list) or not verifier:
        raise DetachedStepError("target_checkpoint plan has no verified resume command")
    if len(prior_journal_head_sha256) != 64:
        raise DetachedStepError("target checkpoint prior journal head is invalid")
    environment = dict(plan["execution_environment"])
    environment.update(
        {
            "AURA_DETACHED_PLAN_SHA256": str(plan["plan_sha256"]),
            "AURA_DETACHED_COMMAND_SHA256": str(plan["command_sha256"]),
            "AURA_DETACHED_PRIOR_ATTEMPT": str(prior_attempt),
            "AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256": prior_journal_head_sha256,
            "AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT": "stdout-v3",
        }
    )
    _verify_execution_manifest_current(plan["resume_verifier_execution_manifest"])
    try:
        result = subprocess.run(
            verifier,
            cwd=plan["cwd"],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DetachedStepError("target checkpoint verifier could not execute") from exc
    if result.returncode != 0 or len(result.stdout.encode("utf-8")) > 65_536:
        raise DetachedStepError("target checkpoint verifier did not return an admissible verdict")
    try:
        verdict = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise DetachedStepError("target checkpoint verifier returned invalid JSON") from exc
    verdict = validate_resume_verdict(
        verdict,
        plan_sha256=str(plan["plan_sha256"]),
        command_sha256=str(plan["command_sha256"]),
        prior_attempt=prior_attempt,
        prior_journal_head_sha256=prior_journal_head_sha256,
    )
    if verdict["verdict"] != "safe_to_resume":
        raise DetachedStepError(f"target checkpoint verifier returned {verdict['verdict']}")
    return verdict


def _launch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().absolute()
    expected_identity: dict[str, int] | None = None
    if args.run_dir_identity_json:
        try:
            parsed_identity = json.loads(args.run_dir_identity_json)
            if not isinstance(parsed_identity, dict):
                raise ValueError
            expected_identity = validate_directory_identity(parsed_identity)
        except (json.JSONDecodeError, ValueError, SecurePathCustodyError) as exc:
            parser.error(f"--run-dir-identity-json is invalid: {exc}")
    with _run_directory_custody(
        run_dir,
        create=True,
        expected_identity=expected_identity,
    ):
        return _launch_custodied(args, parser)


def _launch_custodied(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    if not args.name or len(args.name) > 120 or any(ch.isspace() for ch in args.name):
        parser.error("--name must be a non-empty whitespace-free identifier")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be finite and positive")
    command = _normalize_command(parser, args.command)
    cwd = Path(args.cwd).expanduser().resolve(strict=True)
    if not cwd.is_dir():
        parser.error("--cwd must resolve to a directory")
    run_dir = Path(args.run_dir).expanduser().absolute()
    output_roots: list[Path] = []
    for value in args.execution_output_root:
        output_root = Path(value).expanduser().resolve(strict=False)
        if output_root == cwd or not output_root.is_relative_to(cwd):
            parser.error("--execution-output-root must be a strict child of --cwd")
        output_roots.append(output_root)
    receipt_path = run_dir / RECEIPT_FILE
    resume_verifier = _parse_optional_command_json(parser, args.resume_verifier_json)
    broker_policy_specs = _parse_broker_policy_json(parser, args.broker_policy_json)
    requested_plan = _build_plan(
        args.name,
        command,
        cwd,
        args.timeout,
        args.resume_contract,
        resume_verifier,
        broker_policy_specs,
        (run_dir, *output_roots),
        args.containment_mode,
    )
    recovered_stale_child = False
    prior_completion_indeterminate = False
    resume_verdict: dict[str, Any] | None = None
    release_fd: int | None = None
    with _locked(run_dir):
        plan_path = run_dir / PLAN_FILE
        status_path = run_dir / STATUS_FILE
        if plan_path.exists():
            plan, attempts, _prior_status, prior_receipt = _verify_run_locked(run_dir)
            if prior_receipt is not None:
                raise DetachedStepError("terminal receipt already exists; run is immutable")
            if not args.resume:
                raise DetachedStepError(
                    "existing detached plan has no terminal receipt; use --resume explicitly"
                )
            if _comparable_plan(plan) != _comparable_plan(requested_plan):
                raise DetachedStepError("existing detached plan differs")
            if plan.get("resume_contract") != "target_checkpoint":
                raise DetachedStepError(
                    "incomplete generic execution is completion-indeterminate; target_checkpoint contract required"
                )
            grouped = _events_by_attempt(attempts)
            if grouped:
                latest_attempt = max(grouped)
                latest = grouped[latest_attempt]
                launched = latest["LAUNCHED"]
                supervisor_state = _identity_state(
                    int(launched["supervisor_pid"]),
                    str(launched["supervisor_start_token"]),
                )
                if supervisor_state == "alive":
                    raise DetachedStepError("detached supervisor is already alive")
                if supervisor_state == "unknown":
                    raise DetachedStepError("supervisor identity is unobservable; refusing resume")
                _cleanup_stale_control(latest.get("CONTROL_READY"))
                target = latest.get("TARGET_STARTED")
                if target is not None:
                    recovered_stale_child = _terminate_stale_target(target)
                broker_terminal_ids = {
                    str(event.get("request_id") or "")
                    for event in _broker_events(latest, "BROKER_TERMINAL")
                }
                for broker_start in _broker_events(latest, "BROKER_STARTED"):
                    if str(broker_start.get("request_id") or "") in broker_terminal_ids:
                        continue
                    worker_cleanup_performed = _terminate_stale_broker_worker(broker_start)
                    recovered_stale_child = worker_cleanup_performed or recovered_stale_child
                    _record_worker_origin_quarantine_locked(
                        run_dir,
                        plan,
                        attempt=latest_attempt,
                        launched=launched,
                        broker_start=broker_start,
                        cleanup_action_performed=worker_cleanup_performed,
                    )
                if (
                    os.environ.get("AURA_DETACHED_TEST_CRASH_POINT")
                    == "after_worker_origin_quarantine"
                    and "PYTEST_CURRENT_TEST" in os.environ
                ):
                    os._exit(95)
                prior_completion_indeterminate = True
            if grouped:
                prior_attempt = max(grouped)
                attempts = _read_attempts(run_dir)
                prior_journal_head_sha256 = attempts[-1]["event_sha256"]
                resume_verdict = _run_resume_verifier(
                    plan,
                    run_dir,
                    prior_attempt,
                    prior_journal_head_sha256,
                )
                attempt = prior_attempt + 1
            else:
                attempt = 1
        else:
            if args.resume:
                raise DetachedStepError("--resume requires an existing detached plan")
            plan = requested_plan
            _atomic_write(plan_path, plan, replace=False)
            attempt = 1

        supervisor_pid, release_fd = _daemonize(run_dir, plan, attempt)
        if (
            os.environ.get("AURA_DETACHED_TEST_CRASH_POINT")
            == "after_supervisor_fork_before_reservation"
            and "PYTEST_CURRENT_TEST" in os.environ
        ):
            os._exit(93)
        supervisor_observation = ProcessObservation("unknown")
        identity_deadline = time.monotonic() + 2.0
        while time.monotonic() < identity_deadline:
            supervisor_observation = _inspect_process(supervisor_pid)
            if supervisor_observation.state == "alive":
                break
            if supervisor_observation.state == "dead":
                break
            time.sleep(0.02)
        if supervisor_observation.state != "alive":
            os.close(release_fd)
            release_fd = None
            raise DetachedStepError("detached supervisor identity could not be established")
        supervisor_start = supervisor_observation.token
        _append_attempt_event_locked(
            run_dir,
            {
                "event": "LAUNCHED",
                "attempt": attempt,
                "plan_sha256": plan["plan_sha256"],
                "supervisor_pid": supervisor_pid,
                "supervisor_start_token": supervisor_start,
                "supervisor_process_group_id": supervisor_observation.process_group_id,
                "supervisor_executable": supervisor_observation.executable,
                "resume_verdict": resume_verdict,
                "recorded_at": time.time(),
            },
        )
        _publish_status(
            run_dir,
            {
                "name": plan["name"],
                "plan_sha256": plan["plan_sha256"],
                "command_sha256": plan["command_sha256"],
                "state": "handoff",
                "supervisor_attempt": attempt,
                "supervisor_pid": supervisor_pid,
                "supervisor_start_token": supervisor_start,
                "child_pid": 0,
                "child_process_group_id": 0,
                "child_start_token": "",
                "started_at": time.time(),
                "heartbeat_at": time.time(),
                "heartbeat_sequence": 0,
                "restart_count": 0,
            },
        )
        if (
            os.environ.get("AURA_DETACHED_TEST_CRASH_POINT") == "after_reservation_before_release"
            and "PYTEST_CURRENT_TEST" in os.environ
        ):
            os._exit(94)
        try:
            if os.write(release_fd, b"G") != 1:
                raise DetachedStepError("detached supervisor release handoff was incomplete")
        finally:
            os.close(release_fd)
            release_fd = None

    deadline = time.monotonic() + _HANDOFF_WAIT_S
    status: dict[str, Any] = {}
    observed_receipt: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if receipt_path.is_file():
            with _locked(run_dir):
                _verified_plan, _verified_events, _verified_status, observed_receipt = (
                    _verify_run_locked(run_dir)
                )
            if (
                observed_receipt is not None
                and observed_receipt.get("plan_sha256") == plan["plan_sha256"]
                and int(observed_receipt.get("supervisor_attempt") or 0) == attempt
                and int(observed_receipt.get("supervisor_pid") or 0) == supervisor_pid
            ):
                break
        if status_path.is_file():
            status = _read_json(status_path)
            if (
                status.get("plan_sha256") == plan["plan_sha256"]
                and int(status.get("supervisor_attempt") or 0) == attempt
                and int(status.get("supervisor_pid") or 0) == supervisor_pid
            ):
                start_token = str(status.get("supervisor_start_token") or "")
                if _pid_matches(supervisor_pid, start_token):
                    break
        time.sleep(0.05)

    if observed_receipt is not None and (
        observed_receipt.get("plan_sha256") == plan["plan_sha256"]
        and int(observed_receipt.get("supervisor_attempt") or 0) == attempt
        and int(observed_receipt.get("supervisor_pid") or 0) == supervisor_pid
    ):
        start_token = str(observed_receipt.get("supervisor_start_token") or "")
        terminal = True
    else:
        start_token = str(status.get("supervisor_start_token") or "")
        terminal = False
        if (
            status.get("plan_sha256") != plan["plan_sha256"]
            or int(status.get("supervisor_attempt") or 0) != attempt
            or int(status.get("supervisor_pid") or 0) != supervisor_pid
            or not _pid_matches(supervisor_pid, start_token)
        ):
            raise DetachedStepError("detached supervisor did not become observable")
    return {
        "schema": f"{SCHEMA_PREFIX}.launch.v1",
        "run_dir": str(run_dir),
        "plan_sha256": plan["plan_sha256"],
        "command_sha256": plan["command_sha256"],
        "supervisor_attempt": attempt,
        "supervisor_pid": supervisor_pid,
        "supervisor_start_token": start_token,
        "terminal": terminal,
        "resumed": bool(args.resume),
        "recovered_stale_child": recovered_stale_child,
        "prior_completion_indeterminate": prior_completion_indeterminate,
        "resume_verdict": resume_verdict,
        "status_path": str(run_dir / STATUS_FILE),
        "receipt_path": str(run_dir / RECEIPT_FILE),
        "attempts_path": str(run_dir / ATTEMPTS_FILE),
        "log_path": str(run_dir / LOG_FILE),
        "restart_policy": "never",
    }


def _status(
    run_dir: Path,
    *,
    expected_identity: dict[str, int] | None = None,
) -> dict[str, Any]:
    with _run_directory_custody(
        run_dir,
        create=False,
        expected_identity=expected_identity,
    ) as custody:
        return _status_custodied(custody.path)


def _status_custodied(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().absolute()
    with _locked(run_dir):
        plan, attempts, status, receipt = _verify_run_locked(run_dir)
    grouped = _events_by_attempt(attempts)
    latest_attempt = max(grouped, default=0)
    latest = grouped.get(latest_attempt, {})
    launched = latest.get("LAUNCHED", {})
    target = latest.get("TARGET_STARTED", {})
    supervisor_pid = int(launched.get("supervisor_pid") or 0)
    supervisor_start = str(launched.get("supervisor_start_token") or "")
    supervisor_state = _identity_state(supervisor_pid, supervisor_start)
    child_state = (
        _identity_state(
            int(target.get("child_pid") or 0),
            str(target.get("child_start_token") or ""),
        )
        if target
        else "dead"
    )
    return {
        "schema": f"{SCHEMA_PREFIX}.inspection.v1",
        "run_dir": str(run_dir),
        "state": (
            receipt.get("status")
            if receipt
            else "completion_indeterminate"
            if supervisor_state == "dead"
            else status.get("state", "handoff")
        ),
        "plan_sha256": plan.get("plan_sha256"),
        "resume_contract": plan.get("resume_contract"),
        "supervisor_attempt": latest_attempt,
        "supervisor_pid": supervisor_pid,
        "supervisor_start_token": supervisor_start,
        "supervisor_state": supervisor_state,
        "supervisor_alive": supervisor_state == "alive",
        "child_pid": target.get("child_pid"),
        "child_process_group_id": target.get("child_process_group_id"),
        "child_start_token": target.get("child_start_token"),
        "containment_token": target.get("containment_token"),
        "child_state": child_state,
        "heartbeat_at": status.get("heartbeat_at"),
        "heartbeat_sequence": status.get("heartbeat_sequence"),
        "restart_count": receipt.get("restart_count") if receipt else status.get("restart_count"),
        "attempt_event_count": len(attempts),
        "attempt_journal_head_sha256": attempts[-1]["event_sha256"] if attempts else "",
        "terminal": receipt is not None,
        "completion_indeterminate": receipt is None and supervisor_state == "dead",
        "receipt": receipt,
    }


def _stop(
    run_dir: Path,
    *,
    expected_identity: dict[str, int] | None = None,
) -> dict[str, Any]:
    with _run_directory_custody(
        run_dir,
        create=False,
        expected_identity=expected_identity,
    ) as custody:
        return _stop_custodied(custody.path)


def _stop_custodied(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().absolute()
    with _locked(run_dir):
        _plan, attempts, _status_body, receipt = _verify_run_locked(run_dir)
        if receipt is not None:
            return {"stopped": False, "reason": "already_terminal"}
        grouped = _events_by_attempt(attempts)
        if not grouped:
            return {"stopped": False, "reason": "supervisor_not_reserved"}
        launched = grouped[max(grouped)]["LAUNCHED"]
        supervisor_pid = int(launched["supervisor_pid"])
        start_token = str(launched["supervisor_start_token"])
        state = _identity_state(supervisor_pid, start_token)
        if state == "unknown":
            raise DetachedStepError("supervisor identity is unobservable; refusing signal")
        if state == "dead":
            return {"stopped": False, "reason": "supervisor_not_alive"}
        control = grouped[max(grouped)].get("CONTROL_READY")
        if control is None:
            raise DetachedStepError("supervisor control channel is not ready")
        socket_path = Path(str(control.get("socket_path") or ""))
        try:
            socket_stat = socket_path.lstat()
        except OSError as exc:
            raise DetachedStepError("supervisor control socket is unavailable") from exc
        if not stat.S_ISSOCK(socket_stat.st_mode) or socket_stat.st_uid != os.geteuid():
            raise DetachedStepError("supervisor control socket identity is invalid")
        request = _canonical_bytes(
            {
                "action": "stop",
                "control_token": str(control.get("control_token") or ""),
            }
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.sendto(request, str(socket_path))
    return {"stopped": True, "supervisor_pid": supervisor_pid, "control": "authenticated_socket"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    launch = subparsers.add_parser("launch", help="launch one detached step")
    launch.add_argument("--run-dir", required=True)
    launch.add_argument("--run-dir-identity-json", default="")
    launch.add_argument("--name", required=True)
    launch.add_argument("--cwd", default=str(Path(__file__).resolve().parents[1]))
    launch.add_argument("--timeout", type=float, required=True)
    launch.add_argument(
        "--resume-contract",
        choices=("none", "target_checkpoint"),
        default="none",
        help="declare whether explicit replay is safe because the target owns durable resume state",
    )
    launch.add_argument(
        "--resume-verifier-json",
        default="",
        help="JSON string array for the frozen target checkpoint verifier command",
    )
    launch.add_argument(
        "--broker-policy-json",
        default="",
        help="JSON object array of exact bounded subprocess commands the target may broker",
    )
    launch.add_argument(
        "--containment-mode",
        choices=tuple(sorted(_CONTAINMENT_MODES)),
        default=_SUPERVISOR_SANDBOX_MODE,
        help=(
            "use the supervisor no-fork wrapper, or verify and execute one "
            "already deny-default sandboxed target without nesting"
        ),
    )
    launch.add_argument(
        "--execution-output-root",
        action="append",
        default=[],
        help=(
            "strict child of --cwd whose untracked generated outputs may change; "
            "tracked source and explicit command inputs remain ineligible for exclusion"
        ),
    )
    launch.add_argument("--resume", action="store_true")
    launch.add_argument("command", nargs=argparse.REMAINDER)
    status = subparsers.add_parser("status", help="inspect status and receipt")
    status.add_argument("--run-dir", required=True)
    stop = subparsers.add_parser("stop", help="request one supervised stop")
    stop.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv and effective_argv[0] == "_exec_gate":
        if len(effective_argv) < 3:
            return 126
        try:
            gate_fd = int(effective_argv[1])
        except ValueError:
            return 126
        return _gated_exec(gate_fd, effective_argv[2:])
    if effective_argv and effective_argv[0] == "_supervise":
        if len(effective_argv) != 5:
            return 126
        return _supervisor_bootstrap(*effective_argv[1:])
    parser = build_parser()
    args = parser.parse_args(effective_argv)
    try:
        if args.action == "launch":
            payload = _launch(args, parser)
        elif args.action == "status":
            payload = _status(Path(args.run_dir))
        else:
            payload = _stop(Path(args.run_dir))
    except (DetachedStepError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"run_detached_step: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
