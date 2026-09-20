#!/usr/bin/env python3
"""Run the source-bound resident recurrent-SFT bootstrap campaign.

The controller owns process rotation, durable campaign journaling, exact
checkpoint resume, liveness receipts, and the two-consecutive-no-progress stop
rule. The MLX trainer remains a separate supervised child for every bounded
invocation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import plistlib
import shlex
import stat
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignJournal,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
)
from core.learning.resident_recurrent_sft_bootstrap_authority import (  # noqa: E402
    sha256_bytes,
    sha256_json,
    validate_authority,
)
from core.learning.resident_recurrent_sft_bootstrap_state import (  # noqa: E402
    ResidentSFTBootstrapStateError,
    authority_state_bindings,
    inspect_checkpoint,
    inspect_checkpoint_generation,
    validate_checkpoint_descendant,
)
from core.learning.resident_recurrent_sft_checkpoint_migration import (  # noqa: E402
    ResidentSFTCheckpointMigrationError,
    verify_migration,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from core.runtime.secure_path_custody import (  # noqa: E402
    DirectoryCustody,
    SecurePathCustodyError,
    validate_directory_identity,
    validate_path_custody_threat_model,
)
from tools import run_detached_step as detached  # noqa: E402
from tools.prepare_resident_recurrent_sft_bootstrap_campaign import (  # noqa: E402
    CONTROLLER_CONFIG_SCHEMA,
    SOURCE_PATHS,
)
from tools.resident_recurrent_sft_bootstrap_identity import (  # noqa: E402
    resident_bootstrap_runtime_identity,
)

STATUS_SCHEMA: Final = "aura.resident_recurrent_sft_controller_status.v1"
ATTEMPT_SCHEMA: Final = "aura.resident_recurrent_sft_controller_attempt.v1"
COMPLETION_SCHEMA: Final = "aura.resident_recurrent_sft_controller_completion.v1"
LAUNCH_SCHEMA: Final = "aura.resident_recurrent_sft_controller_launchd.v1"
MAX_DOCUMENT_BYTES: Final = 512 * 1024 * 1024
RESTARTABLE_CONTROLLER_ERRORS: Final = frozenset(
    {
        "resident_sft_controller_detached_launch_failed",
        "resident_sft_controller_detached_resume_failed",
        "resident_sft_controller_attempt_timeout",
    }
)
_ACTIVE_CUSTODIES: tuple[DirectoryCustody, ...] = ()
_RESIDENT_TRAINING_LABEL_PREFIXES: Final = (
    "com.aura.resident-sft.",
    "com.aura.resident-32b-recurrent-grpo",
)
_RESIDENT_TRAINING_STATE_DIR: Final = Path.home() / ".aura/state/resident-training"


class ResidentSFTCampaignControllerError(RuntimeError):
    """Stable fail-closed controller error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentSFTCampaignControllerError(code)


def _resident_training_label(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not any(value.startswith(prefix) for prefix in _RESIDENT_TRAINING_LABEL_PREFIXES)
        or len(value) > 255
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        _fail("resident_training_label_invalid")
    return value


@contextmanager
def _resident_training_lock(path: Path, *, busy_code: str) -> Iterator[int]:
    ensure_private_directory(path.parent)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ResidentSFTCampaignControllerError(
            "resident_training_lock_open_failed"
        ) from exc
    acquired = False
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.getuid():
            _fail("resident_training_lock_identity_invalid")
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise ResidentSFTCampaignControllerError(busy_code) from exc
        yield fd
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def _resident_training_host_lease(
    *,
    label: str,
    config_sha256: str,
    lease_path: Path | None = None,
) -> Iterator[dict[str, Any]]:
    label = _resident_training_label(label)
    if (
        not isinstance(config_sha256, str)
        or len(config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in config_sha256)
    ):
        _fail("resident_training_config_identity_invalid")
    path = lease_path or (_RESIDENT_TRAINING_STATE_DIR / "host.lock")
    with _resident_training_lock(path, busy_code="resident_training_host_busy") as fd:
        body = {
            "schema": "aura.resident_training_host_lease.v1",
            "active": True,
            "label": label,
            "config_sha256": config_sha256,
            "pid": os.getpid(),
            "acquired_at_unix_ns": time.time_ns(),
        }
        lease = {**body, "lease_sha256": sha256_json(body)}
        payload = _canonical(lease)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload)
        os.fsync(fd)
        try:
            yield lease
        finally:
            released_body = {
                **body,
                "active": False,
                "released_at_unix_ns": time.time_ns(),
            }
            released = {
                **released_body,
                "lease_sha256": sha256_json(released_body),
            }
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, _canonical(released))
            os.fsync(fd)


def _loaded_resident_training_labels(*, timeout_s: float = 30.0) -> set[str]:
    result = subprocess.run(
        ["/bin/launchctl", "list"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        _fail("resident_training_launchd_inventory_failed")
    labels: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        label = fields[-1]
        if any(label.startswith(prefix) for prefix in _RESIDENT_TRAINING_LABEL_PREFIXES):
            labels.add(_resident_training_label(label))
    return labels


def _retire_resident_training_jobs(
    *,
    active_label: str,
    launch_agents: Path | None = None,
    quarantine_root: Path | None = None,
) -> dict[str, Any]:
    active_label = _resident_training_label(active_label)
    uid = os.getuid()
    domain = f"gui/{uid}"
    launch_agents = launch_agents or (Path.home() / "Library/LaunchAgents")
    quarantine_root = quarantine_root or (
        Path.home() / ".aura/quarantine/resident-training-launchagents"
    )
    ensure_private_directory(launch_agents)
    ensure_private_directory(quarantine_root)
    loaded = _loaded_resident_training_labels()
    discovered: dict[str, Path] = {}
    for path in sorted(launch_agents.glob("*.plist")):
        if not any(path.name.startswith(prefix) for prefix in _RESIDENT_TRAINING_LABEL_PREFIXES):
            continue
        if path.is_symlink() or not path.is_file():
            _fail("resident_training_launchd_plist_identity_invalid")
        observed = path.stat()
        if observed.st_uid != uid or observed.st_size > 1024 * 1024:
            _fail("resident_training_launchd_plist_identity_invalid")
        try:
            document = plistlib.loads(read_stable_bytes(path, max_bytes=1024 * 1024))
        except (OSError, ValueError, plistlib.InvalidFileException) as exc:
            raise ResidentSFTCampaignControllerError(
                "resident_training_launchd_plist_invalid"
            ) from exc
        label = _resident_training_label(document.get("Label"))
        if path.name != f"{label}.plist":
            _fail("resident_training_launchd_plist_label_mismatch")
        discovered[label] = path

    retired_labels: list[str] = []
    quarantined: list[str] = []
    for label in sorted((loaded | set(discovered)) - {active_label}):
        if label in loaded:
            stopped = subprocess.run(
                ["/bin/launchctl", "bootout", f"{domain}/{label}"],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            if stopped.returncode != 0:
                _fail(f"resident_training_launchd_retirement_failed:{label}")
        discovered_path = discovered.get(label)
        if discovered_path is not None:
            destination = quarantine_root / f"{time.time_ns()}-{discovered_path.name}"
            os.replace(discovered_path, destination)
            os.chmod(destination, 0o600)
            quarantined.append(str(destination))
        retired_labels.append(label)

    remaining = _loaded_resident_training_labels() - {active_label}
    if remaining:
        _fail("resident_training_launchd_retirement_incomplete")
    return {
        "schema": "aura.resident_training_launchd_retirement.v1",
        "active_label": active_label,
        "retired_labels": retired_labels,
        "quarantined_plists": quarantined,
    }


def _canonical(value: Any) -> bytes:
    payload: bytes = canonical_json_bytes(value)
    return payload


def _custody_for(path: Path) -> tuple[DirectoryCustody, str] | None:
    absolute = path.expanduser().absolute()
    matches: list[tuple[int, DirectoryCustody, str]] = []
    for custody in _ACTIVE_CUSTODIES:
        try:
            relative = absolute.relative_to(custody.path).as_posix()
        except ValueError:
            continue
        if relative and relative != ".":
            matches.append((len(custody.path.parts), custody, relative))
    if not matches:
        return None
    _depth, custody, relative = max(matches, key=lambda item: item[0])
    return custody, relative


def _exact_custody(path: Path) -> DirectoryCustody | None:
    absolute = path.expanduser().absolute()
    for custody in _ACTIVE_CUSTODIES:
        if custody.path == absolute:
            return custody
    return None


def _read_canonical(path: Path, *, role: str) -> dict[str, Any]:
    try:
        bound = _custody_for(path)
        if bound is not None:
            custody, relative = bound
            payload = custody.read_bytes(relative, max_bytes=MAX_DOCUMENT_BYTES)
        else:
            if path.is_symlink() or not path.is_file():
                _fail(f"resident_sft_controller_{role}_file_invalid")
            payload = read_stable_bytes(path, max_bytes=MAX_DOCUMENT_BYTES)
        value = json.loads(payload)
    except (
        OSError,
        SecurePathCustodyError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ResidentSFTCampaignControllerError(
            f"resident_sft_controller_{role}_unreadable"
        ) from exc
    if not isinstance(value, dict) or _canonical(value) != payload:
        _fail(f"resident_sft_controller_{role}_noncanonical")
    return value


def _repo_path(value: Any, *, role: str, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(f"resident_sft_controller_{role}_path_invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or str(pure) != value or ".." in pure.parts:
        _fail(f"resident_sft_controller_{role}_path_invalid")
    lexical = REPO_ROOT / pure
    lexical = lexical.absolute()
    active = _exact_custody(lexical)
    if active is None:
        bound = _custody_for(lexical)
        active = bound[0] if bound is not None else None
    if active is not None:
        try:
            active.verify()
        except SecurePathCustodyError as exc:
            raise ResidentSFTCampaignControllerError(
                f"resident_sft_controller_{role}_custody_invalid"
            ) from exc
        return lexical
    current = REPO_ROOT
    for component in pure.parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            _fail(f"resident_sft_controller_{role}_symlink_forbidden")
    resolved = lexical.resolve(strict=must_exist)
    try:
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ResidentSFTCampaignControllerError(
            f"resident_sft_controller_{role}_outside_repo"
        ) from exc
    return resolved


def _verify_binding(binding: Any, *, role: str) -> tuple[Path, bytes]:
    required = {"path", "sha256", "size_bytes"}
    if not isinstance(binding, Mapping) or not required.issubset(binding):
        _fail(f"resident_sft_controller_{role}_binding_invalid")
    path = _repo_path(binding["path"], role=role)
    payload = read_stable_bytes(path, max_bytes=MAX_DOCUMENT_BYTES)
    if len(payload) != binding["size_bytes"] or sha256_bytes(payload) != binding["sha256"]:
        _fail(f"resident_sft_controller_{role}_binding_drift")
    return path, payload


def _trainer_import_closure() -> frozenset[str]:
    python = str(Path(os.path.abspath(sys.executable)))
    probe = """
import json
import sys
from pathlib import Path
import tools.train_resident_recurrent_sft_bootstrap
root = Path.cwd().resolve()
paths = set()
for module in tuple(sys.modules.values()):
    value = getattr(module, '__file__', None)
    if not isinstance(value, str):
        continue
    try:
        path = Path(value).resolve(strict=True)
        relative = path.relative_to(root).as_posix()
    except (OSError, ValueError):
        continue
    if path.is_file():
        paths.add(relative)
print(json.dumps(sorted(paths), separators=(',', ':')))
"""
    result = subprocess.run(
        [python, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120.0,
        check=False,
    )
    if result.returncode != 0:
        _fail("resident_sft_controller_import_closure_unavailable")
    lines = [line for line in result.stdout.splitlines() if line.startswith("[")]
    try:
        paths = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError, TypeError):
        _fail("resident_sft_controller_import_closure_invalid")
    if (
        not isinstance(paths, list)
        or not paths
        or any(
            not isinstance(path, str)
            or not path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            for path in paths
        )
    ):
        _fail("resident_sft_controller_import_closure_invalid")
    return frozenset(paths)


def _protected_source_changes(
    changed_paths: Sequence[str],
    import_closure: frozenset[str],
) -> tuple[str, ...]:
    protected = import_closure | frozenset(SOURCE_PATHS.values())
    return tuple(sorted(path for path in changed_paths if path in protected))


def _verify_source_lineage(expected: Mapping[str, Any]) -> dict[str, str]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if result.returncode != 0 and args[0] != "merge-base":
            _fail("resident_sft_controller_git_state_unavailable")
        return result

    branch = run("branch", "--show-current").stdout.strip()
    head = run("rev-parse", "HEAD").stdout.strip()
    upstream = run("rev-parse", "origin/main").stdout.strip()
    frozen = expected.get("commit")
    if (
        not isinstance(frozen, str)
        or len(frozen) != 40
        or run("merge-base", "--is-ancestor", frozen, head).returncode != 0
        or run("merge-base", "--is-ancestor", frozen, upstream).returncode != 0
    ):
        _fail("resident_sft_controller_source_lineage_drift")
    committed_changes = run("diff", "--name-only", f"{frozen}..{head}", "--").stdout.splitlines()
    dirty_changes = run("diff", "--name-only", "HEAD", "--").stdout.splitlines()
    protected_changes = _protected_source_changes(
        [*committed_changes, *dirty_changes],
        _trainer_import_closure(),
    )
    if protected_changes:
        _fail("resident_sft_controller_source_closure_drift")
    return {
        "branch": branch,
        "frozen_commit": frozen,
        "observed_head": head,
        "observed_origin_main": upstream,
    }


def _verify_authority_artifacts(authority: Mapping[str, Any]) -> None:
    for split in ("train", "validation"):
        _verify_binding(
            authority["dataset_artifacts"][split],
            role=f"dataset_{split}",
        )
    _verify_binding(authority["execution_spec"], role="execution_spec")
    _verify_binding(authority["trust_policy"], role="trust_policy")
    sources = authority.get("sources")
    if not isinstance(sources, Mapping):
        _fail("resident_sft_controller_sources_invalid")
    for role, binding in sorted(sources.items()):
        _verify_binding(binding, role=f"source_{role}")


def _launchd_job(label: str) -> dict[str, Any]:
    target = f"gui/{os.getuid()}/{label}"
    result = subprocess.run(
        ["/bin/launchctl", "print", target],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if result.returncode != 0:
        _fail("resident_sft_controller_launchd_job_unavailable")
    pid: int | None = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("pid = "):
            try:
                pid = int(stripped.removeprefix("pid = "))
            except ValueError:
                _fail("resident_sft_controller_launchd_pid_invalid")
            break
    if pid is None or pid <= 1:
        _fail("resident_sft_controller_launchd_pid_missing")
    return {"target": target, "job_pid": pid}


def _wait_launchd_job(label: str, *, timeout_s: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: ResidentSFTCampaignControllerError | None = None
    while time.monotonic() < deadline:
        try:
            return _launchd_job(label)
        except ResidentSFTCampaignControllerError as exc:
            if exc.code not in {
                "resident_sft_controller_launchd_job_unavailable",
                "resident_sft_controller_launchd_pid_missing",
            }:
                raise
            last_error = exc
            time.sleep(0.25)
    if last_error is not None:
        raise last_error
    _fail("resident_sft_controller_launchd_job_unavailable")


def _verify_execution_supervision(
    config: Mapping[str, Any], *, launchd_supervised: bool
) -> dict[str, Any]:
    if not launchd_supervised:
        _fail("resident_sft_controller_requires_launchd_entrypoint")
    job = _launchd_job(str(config["launch"]["label"]))
    controller_pid = os.getpid()
    if controller_pid != job["job_pid"]:
        _fail("resident_sft_controller_launchd_parent_mismatch")

    processes = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,command="],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    expected = (
        "/usr/bin/caffeinate",
        "-i",
        sys.executable,
        str(Path(__file__).resolve(strict=True)),
        *sys.argv[1:],
    )
    inhibitor_pids: list[int] = []
    if processes.returncode == 0:
        for line in processes.stdout.splitlines():
            fields = line.strip().split(maxsplit=2)
            if len(fields) != 3:
                continue
            try:
                pid = int(fields[0])
                parent_pid = int(fields[1])
                argv = tuple(shlex.split(fields[2]))
            except (ValueError, TypeError):
                continue
            if parent_pid == controller_pid and argv == expected:
                inhibitor_pids.append(pid)
    if len(inhibitor_pids) != 1:
        _fail("resident_sft_controller_caffeinate_parent_missing")
    return {
        "mode": "launchd_caffeinate",
        "launchd": True,
        "caffeinate": True,
        "launchd_target": job["target"],
        "launchd_pid": job["job_pid"],
        "controller_pid": controller_pid,
        "controller_parent_pid": os.getppid(),
        "caffeinate_pid": inhibitor_pids[0],
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_canonical(path.expanduser().resolve(strict=True), role="config")
    required = {
        "schema",
        "campaign_id",
        "profile",
        "source",
        "authority",
        "plan",
        "paths",
        "path_custody",
        "path_custody_threat_model",
        "watchdog",
        "launch",
        "claim_state",
        "config_sha256",
    }
    if set(config) != required or config.get("schema") != CONTROLLER_CONFIG_SCHEMA:
        _fail("resident_sft_controller_config_schema_invalid")
    body = dict(config)
    claimed = body.pop("config_sha256")
    if claimed != sha256_json(body):
        _fail("resident_sft_controller_config_digest_mismatch")
    if config.get("profile") not in {"canary", "full"}:
        _fail("resident_sft_controller_profile_invalid")
    campaign_id = config.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.startswith(
        "resident-32b-recurrent-sft-bootstrap-cp"
    ):
        _fail("resident_sft_controller_campaign_id_invalid")
    source = config.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"branch", "commit", "origin_main"}
        or not isinstance(source.get("branch"), str)
        or len(source["branch"]) > 255
        or any(ord(character) < 32 for character in source["branch"])
        or any(
            not isinstance(source.get(key), str)
            or len(source[key]) != 40
            or any(character not in "0123456789abcdef" for character in source[key])
            for key in ("commit", "origin_main")
        )
        or source.get("commit") != source.get("origin_main")
    ):
        _fail("resident_sft_controller_source_binding_invalid")
    for role in ("authority", "plan"):
        binding = config.get(role)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256", "size_bytes", "semantic_sha256"}
            or not isinstance(binding.get("path"), str)
            or not isinstance(binding.get("size_bytes"), int)
            or binding["size_bytes"] < 1
            or any(
                not isinstance(binding.get(field), str)
                or len(binding[field]) != 64
                or any(character not in "0123456789abcdef" for character in binding[field])
                for field in ("sha256", "semantic_sha256")
            )
        ):
            _fail(f"resident_sft_controller_{role}_binding_invalid")
    paths = config.get("paths")
    required_paths = {
        "artifact_root",
        "training_output",
        "controller_root",
        "journal",
        "manifest",
        "detached_attempts",
    }
    if (
        not isinstance(paths, Mapping)
        or set(paths) != required_paths
        or any(not isinstance(paths.get(key), str) or not paths[key] for key in required_paths)
    ):
        _fail("resident_sft_controller_paths_invalid")
    custody = config.get("path_custody")
    if not isinstance(custody, Mapping) or set(custody) != {
        "artifact_root",
        "training_output",
        "controller_root",
    }:
        _fail("resident_sft_controller_path_custody_invalid")
    try:
        for role in ("artifact_root", "training_output", "controller_root"):
            validate_directory_identity(custody[role])
    except (KeyError, SecurePathCustodyError) as exc:
        raise ResidentSFTCampaignControllerError(
            "resident_sft_controller_path_custody_invalid"
        ) from exc
    threat_model = config.get("path_custody_threat_model")
    if not isinstance(threat_model, Mapping):
        _fail("resident_sft_controller_path_custody_threat_model_invalid")
    try:
        validate_path_custody_threat_model(threat_model)
    except SecurePathCustodyError as exc:
        raise ResidentSFTCampaignControllerError(
            "resident_sft_controller_path_custody_threat_model_invalid"
        ) from exc
    if config.get("claim_state") != {
        "reasoning_gain_proven": False,
        "frontier_level_proven": False,
        "grpo_admission": False,
        "promotion_allowed": False,
    }:
        _fail("resident_sft_controller_claim_boundary_invalid")
    watchdog = config.get("watchdog")
    required_watchdog = {
        "schema",
        "max_attempts_per_cell",
        "max_consecutive_no_progress_failures",
        "poll_interval_s",
        "heartbeat_stale_s",
        "attempt_timeout_s",
        "retry_backoff_s",
        "resume_exact_checkpoint_only",
    }
    if (
        not isinstance(watchdog, Mapping)
        or set(watchdog) != required_watchdog
        or watchdog.get("schema") != "aura.resident_recurrent_sft_controller_watchdog.v1"
        or watchdog.get("resume_exact_checkpoint_only") is not True
        or watchdog.get("max_consecutive_no_progress_failures") != 2
        or type(watchdog.get("max_attempts_per_cell")) is not int
        or not 2 <= watchdog["max_attempts_per_cell"] <= 32
    ):
        _fail("resident_sft_controller_watchdog_invalid")
    for key in (
        "poll_interval_s",
        "heartbeat_stale_s",
        "attempt_timeout_s",
        "retry_backoff_s",
    ):
        value = watchdog.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            _fail("resident_sft_controller_watchdog_invalid")
    launch = config.get("launch")
    if (
        not isinstance(launch, Mapping)
        or set(launch) != {"label", "launchd_required", "caffeinate_required"}
        or launch.get("label") != f"com.aura.resident-sft.{campaign_id}"
        or launch.get("launchd_required") is not True
        or launch.get("caffeinate_required") is not True
    ):
        _fail("resident_sft_controller_launch_policy_invalid")
    return config


def _load_contracts(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], CampaignPlan]:
    config = _load_config(config_path)
    _verify_source_lineage(config["source"])
    authority_path, authority_payload = _verify_binding(config["authority"], role="authority")
    authority = json.loads(authority_payload)
    training_path = _repo_path(
        config["paths"]["training_output"], role="training", must_exist=False
    )
    authority = validate_authority(
        authority,
        expected_authority_sha256=config["authority"]["semantic_sha256"],
        now=datetime.now(UTC),
        allow_expired_resume=(training_path / "latest.json").exists(),
    )
    if resident_bootstrap_runtime_identity() != authority.get("runtime"):
        _fail("resident_sft_controller_runtime_identity_drift")
    _verify_authority_artifacts(authority)
    if (
        authority_path.as_posix()
        != _repo_path(config["authority"]["path"], role="authority").as_posix()
    ):
        _fail("resident_sft_controller_authority_path_drift")
    _plan_path, plan_payload = _verify_binding(config["plan"], role="plan")
    try:
        plan = CampaignPlan.from_dict(json.loads(plan_payload))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ResidentSFTCampaignControllerError("resident_sft_controller_plan_invalid") from exc
    if (
        plan.plan_sha256 != config["plan"]["semantic_sha256"]
        or plan.campaign_name != config["campaign_id"]
        or authority["campaign_id"] != config["campaign_id"]
        or authority["campaign_scope"]
        != ("canary_lifecycle" if config["profile"] == "canary" else "full_bootstrap")
        or authority["artifact_root_identity"] != config["path_custody"]["training_output"]
    ):
        _fail("resident_sft_controller_campaign_binding_drift")
    return config, authority, plan


def _acquire_campaign_custodies(config: Mapping[str, Any]) -> tuple[DirectoryCustody, ...]:
    paths = config["paths"]
    identities = config["path_custody"]
    acquired: list[DirectoryCustody] = []
    try:
        for role in ("artifact_root", "training_output", "controller_root"):
            acquired.append(
                DirectoryCustody.acquire(
                    _repo_path(paths[role], role=role),
                    expected_identity=identities[role],
                    private=True,
                )
            )
    except BaseException:
        for custody in reversed(acquired):
            custody.close()
        raise
    return tuple(acquired)


def _verify_campaign_custodies(custodies: Sequence[DirectoryCustody]) -> None:
    for custody in custodies:
        custody.verify()


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical(value)
    bound = _custody_for(path)
    if bound is not None:
        custody, relative = bound
        try:
            if custody.write_bytes_once(relative, payload, mode=0o600):
                return
            observed = custody.read_bytes(relative, max_bytes=max(len(payload), 1))
        except SecurePathCustodyError as exc:
            raise ResidentSFTCampaignControllerError(
                "resident_sft_controller_custodied_write_failed"
            ) from exc
        if observed != payload:
            _fail("resident_sft_controller_immutable_artifact_drift")
        return
    ensure_private_directory(path.parent)
    if atomic_write_bytes_if_absent(path, payload, mode=0o600):
        return
    observed = read_stable_bytes(path, max_bytes=max(len(payload), 1))
    if observed != payload:
        _fail("resident_sft_controller_immutable_artifact_drift")


def _note_status_publish_failure(exc: BaseException, *, original: str) -> None:
    """The status file did not get written while a campaign was already dying.

    This runs inside an ``except`` block that is about to re-raise, so it must
    not raise and must not replace the original failure — a controller that
    reported "could not write the status file" instead of "training aborted"
    would send the reader to the wrong subsystem.

    It must also not be silent. A campaign whose terminal status never landed
    looks, to every downstream reader, exactly like a campaign still running;
    the whole point of the status file is that absence means something. So
    the failure is written to stderr, where the launchd log already collects
    this controller's output.
    """
    print(
        f"resident_sft_controller: could not publish terminal status "
        f"after {original}: {type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )


def _publish_status(
    root: Path, config: Mapping[str, Any], state: str, details: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "schema": STATUS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "state": state,
        "controller_pid": os.getpid(),
        "observed_at_unix_ns": time.time_ns(),
        "details": dict(details),
    }
    status = {**body, "status_sha256": sha256_json(body)}
    status_path = root / "status.json"
    bound = _custody_for(status_path)
    if bound is not None:
        custody, relative = bound
        try:
            custody.atomic_write_bytes(relative, _canonical(status), mode=0o600)
        except SecurePathCustodyError as exc:
            raise ResidentSFTCampaignControllerError(
                "resident_sft_controller_custodied_status_failed"
            ) from exc
    else:
        atomic_write_bytes(status_path, _canonical(status), mode=0o600)
    return status


@contextmanager
def _controller_lock(path: Path) -> Iterator[bool]:
    bound = _custody_for(path)
    if bound is not None:
        custody, relative = bound
        fd = custody.open_file(
            relative,
            os.O_RDWR | os.O_CREAT,
            mode=0o600,
            create_parents=True,
        )
        acquired = False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            yield acquired
        finally:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        return
    with path.open("a+b") as lock:
        acquired = False
        try:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            yield acquired
        finally:
            if acquired:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _checkpoint_snapshot(authority: Mapping[str, Any]) -> dict[str, Any]:
    output = _repo_path(authority["artifact_root"], role="training", must_exist=False)
    custody = _exact_custody(output)
    latest_present = (
        custody.file_exists("latest.json")
        if custody is not None
        else (output / "latest.json").is_file()
    )
    if not latest_present:
        return {
            "present": False,
            "step": 0,
            "checkpoint_sequence": 0,
            "invocation_count": 0,
            "terminal": False,
            "complete_sha256": "",
        }
    inspected = inspect_checkpoint(
        output,
        expected_bindings=authority_state_bindings(authority),
        custody=custody,
    )
    state = inspected.state
    return {
        "present": True,
        "step": state["step"],
        "checkpoint_sequence": state["checkpoint_sequence"],
        "invocation_count": state["invocation_count"],
        "terminal": state["terminal"],
        "halt_reason": state["halt_reason"],
        "elapsed_training_s": state["elapsed_training_s"],
        "complete_sha256": inspected.complete_sha256,
        "model_identity_sha256": state["model_identity_sha256"],
    }


def _verified_migration_start_step(authority: Mapping[str, Any]) -> int | None:
    root = _repo_path(
        authority["artifact_root"], role="artifact_root", must_exist=False
    )
    if not root.is_dir():
        return None
    receipt_path = root / "checkpoint-migration.json"
    if not receipt_path.is_file():
        return None
    try:
        receipt = verify_migration(
            receipt_path,
            destination_repo_root=REPO_ROOT,
            destination_authority=authority,
        )
        destination = receipt.get("destination")
        if not isinstance(destination, Mapping):
            _fail("resident_sft_controller_checkpoint_migration_invalid")
        checkpoint_path = Path(str(destination.get("checkpoint", ""))).resolve(strict=True)
        checkpoint_ref = checkpoint_path.relative_to(root).as_posix()
        migrated = inspect_checkpoint_generation(
            root,
            checkpoint=checkpoint_ref,
            expected_bindings=authority_state_bindings(authority),
        )
        current = inspect_checkpoint(
            root,
            expected_bindings=authority_state_bindings(authority),
        )
        validate_checkpoint_descendant(migrated.state, current.state)
    except (
        FileNotFoundError,
        OSError,
        ResidentSFTBootstrapStateError,
        ResidentSFTCheckpointMigrationError,
        ValueError,
    ) as exc:
        raise ResidentSFTCampaignControllerError(
            "resident_sft_controller_checkpoint_migration_invalid"
        ) from exc
    return int(migrated.state["step"])


def _initial_checkpoint_matches_plan(
    *,
    observed_step: int,
    required_start: int,
    required_end: int,
    migration_start: int | None,
) -> bool:
    return observed_step == required_start or (
        migration_start == observed_step
        and required_start < observed_step <= required_end
    )


def _commit_migration_covered_cell(
    journal: CampaignJournal,
    *,
    cell_id: str,
    required_start: int,
    required_end: int,
    migration_start: int,
    snapshot: Mapping[str, Any],
) -> None:
    """Record a plan cell already covered by one verified migrated checkpoint."""

    if (
        migration_start != int(snapshot.get("step", -1))
        or required_start < 0
        or required_end <= required_start
        or required_end > migration_start
        or snapshot.get("present") is not True
    ):
        _fail("resident_sft_controller_migration_covered_cell_invalid")
    evidence = {
        "schema": "aura.resident_recurrent_sft_migration_covered_cell.v1",
        "migration_start_step": migration_start,
        "required_start_step": required_start,
        "required_end_step": required_end,
        "checkpoint_sequence": int(snapshot["checkpoint_sequence"]),
        "checkpoint_complete_sha256": str(snapshot["complete_sha256"]),
    }
    attempt_id = journal.start_cell(cell_id)
    result_sha = journal.record_arm_result(
        cell_id,
        attempt_id,
        {**evidence, "result": "covered_by_verified_migration"},
    )
    verification_sha = journal.record_verified(
        cell_id,
        attempt_id,
        {
            **evidence,
            "verified": True,
            "result_event_sha256": result_sha,
        },
    )
    journal.commit_cell(
        cell_id,
        attempt_id,
        {
            **evidence,
            "verification_event_sha256": verification_sha,
        },
    )


def _invocation_receipt(
    authority: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    if snapshot.get("present") is not True:
        _fail("resident_sft_controller_checkpoint_missing")
    output = _repo_path(authority["artifact_root"], role="training")
    path = output / f"invocation-{int(snapshot['invocation_count']):04d}.json"
    receipt = _read_canonical(path, role="invocation_receipt")
    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    if (
        claimed != sha256_json(body)
        or receipt.get("authority_sha256") != authority["authority_sha256"]
        or receipt.get("campaign_id") != authority["campaign_id"]
        or receipt.get("campaign_scope") != authority["campaign_scope"]
        or receipt.get("checkpoint_complete_sha256") != snapshot["complete_sha256"]
        or receipt.get("step") != snapshot["step"]
        or receipt.get("required_end_step") != snapshot["step"]
        or receipt.get("base_checkpoint_immutable") is not True
        or receipt.get("base_checkpoint_before") != authority["model"]["base_checkpoint"]
        or receipt.get("base_checkpoint_after") != authority["model"]["base_checkpoint"]
        or receipt.get("claim_state", {}).get("promotion_allowed") is not False
    ):
        _fail("resident_sft_controller_invocation_receipt_invalid")
    return receipt


def _attempt_result_paths(root: Path, cell_ordinal: int) -> list[Path]:
    directory = root / "attempt-results"
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"cell-{cell_ordinal:04d}-attempt-*.json"))


def _attempt_result_path(root: Path, cell_ordinal: int, attempt_number: int) -> Path:
    return root / "attempt-results" / f"cell-{cell_ordinal:04d}-attempt-{attempt_number:04d}.json"


def _attempt_reservation_path(root: Path, cell_ordinal: int, attempt_number: int) -> Path:
    return (
        root / "attempt-reservations" / f"cell-{cell_ordinal:04d}-attempt-{attempt_number:04d}.json"
    )


def _reserve_attempt(
    *,
    root: Path,
    config: Mapping[str, Any],
    cell_id: str,
    cell_ordinal: int,
    attempt_id: str,
    attempt_number: int,
    before: Mapping[str, Any],
    required_end_step: int,
) -> dict[str, Any]:
    body = {
        "schema": "aura.resident_recurrent_sft_attempt_reservation.v1",
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "cell_id": cell_id,
        "cell_ordinal": cell_ordinal,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "progress_before": dict(before),
        "required_end_step": required_end_step,
    }
    reservation = {**body, "reservation_sha256": sha256_json(body)}
    _write_once(
        _attempt_reservation_path(root, cell_ordinal, attempt_number),
        reservation,
    )
    return reservation


def _load_attempt_reservation(
    root: Path,
    *,
    cell_ordinal: int,
    attempt_number: int,
) -> dict[str, Any] | None:
    path = _attempt_reservation_path(root, cell_ordinal, attempt_number)
    if not path.is_file():
        return None
    reservation = _read_canonical(path, role="attempt_reservation")
    body = dict(reservation)
    claimed = body.pop("reservation_sha256", None)
    if claimed != sha256_json(body):
        _fail("resident_sft_controller_attempt_reservation_digest_mismatch")
    return reservation


def _trailing_no_progress(root: Path, cell_ordinal: int) -> int:
    count = 0
    for path in reversed(_attempt_result_paths(root, cell_ordinal)):
        record = _read_canonical(path, role="attempt_result")
        if record.get("durable_progress") is True:
            break
        count += 1
    return count


def _attempt_record(
    *,
    config: Mapping[str, Any],
    cell_id: str,
    cell_ordinal: int,
    attempt_id: str,
    attempt_number: int,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    detached_status: Mapping[str, Any],
    required_end_step: int,
) -> dict[str, Any]:
    receipt = detached_status.get("receipt")
    returncode = receipt.get("returncode") if isinstance(receipt, Mapping) else None
    progressed = int(after.get("checkpoint_sequence") or 0) > int(
        before.get("checkpoint_sequence") or 0
    ) and int(after.get("step") or 0) > int(before.get("step") or 0)
    reached = int(after.get("step") or 0) == required_end_step
    body = {
        "schema": ATTEMPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "cell_id": cell_id,
        "cell_ordinal": cell_ordinal,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "progress_before": dict(before),
        "progress_after": dict(after),
        "durable_progress": progressed,
        "required_end_step": required_end_step,
        "required_end_reached": reached,
        "detached_plan_sha256": detached_status.get("plan_sha256"),
        "detached_receipt_sha256": (
            receipt.get("receipt_sha256") if isinstance(receipt, Mapping) else None
        ),
        "returncode": returncode,
        "terminal_success": returncode == 0 and reached,
        "recorded_at_unix_ns": time.time_ns(),
    }
    return {**body, "attempt_sha256": sha256_json(body)}


def _verify_commit_evidence(
    *,
    authority: Mapping[str, Any],
    record: Mapping[str, Any],
    required_end_step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if record.get("terminal_success") is not True:
        _fail("resident_sft_controller_attempt_not_successful")
    expected = record.get("progress_after")
    if not isinstance(expected, Mapping):
        _fail("resident_sft_controller_attempt_checkpoint_invalid")
    current = _checkpoint_snapshot(authority)
    if (
        current.get("present") is not True
        or current.get("step") != required_end_step
        or current.get("step") != expected.get("step")
        or current.get("checkpoint_sequence") != expected.get("checkpoint_sequence")
        or current.get("complete_sha256") != expected.get("complete_sha256")
    ):
        _fail("resident_sft_controller_attempt_checkpoint_drift")
    invocation = _invocation_receipt(authority, current)
    observed_base = full_weight_checkpoint_identity(
        _repo_path(authority["model"]["path"], role="model")
    )
    if observed_base != authority["model"]["base_checkpoint"]:
        _fail("resident_sft_controller_attempt_base_checkpoint_drift")
    verification = {
        "schema": "aura.resident_recurrent_sft_attempt_verification.v1",
        "attempt_sha256": record["attempt_sha256"],
        "checkpoint_complete_sha256": current["complete_sha256"],
        "invocation_receipt_sha256": invocation["receipt_sha256"],
        "required_end_reached": True,
        "base_checkpoint_fingerprint": observed_base["fingerprint"],
        "base_checkpoint_immutable": True,
    }
    return current, verification


def _run_dir(root: Path, cell_ordinal: int, attempt_number: int) -> Path:
    return root / "detached-attempts" / (f"cell-{cell_ordinal:04d}-attempt-{attempt_number:04d}")


def _launch_args(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    run_dir: Path,
    run_dir_identity: Mapping[str, Any] | None,
    name: str,
    minimum_step: int,
    invocation_step_budget: int,
    required_end_step: int,
    resume: bool,
) -> list[str]:
    interpreter = authority.get("runtime", {}).get("interpreter", {})
    python = interpreter.get("executable")
    if not isinstance(python, str) or not python:
        _fail("resident_sft_controller_interpreter_identity_invalid")
    controller = str(Path(__file__).resolve(strict=True))
    trainer = str(
        (REPO_ROOT / "tools/train_resident_recurrent_sft_bootstrap.py").resolve(strict=True)
    )
    authority_path = str(_repo_path(config["authority"]["path"], role="authority"))
    training_output = str(
        _repo_path(config["paths"]["training_output"], role="training", must_exist=False)
    )
    verifier = json.dumps(
        [
            python,
            controller,
            "verify-resume",
            "--config",
            str(config_path),
            "--minimum-step",
            str(minimum_step),
        ],
        separators=(",", ":"),
    )
    args = [
        "launch",
        "--run-dir",
        str(run_dir),
        "--run-dir-identity-json",
        (
            json.dumps(run_dir_identity, sort_keys=True, separators=(",", ":"))
            if run_dir_identity is not None
            else ""
        ),
        "--name",
        name,
        "--cwd",
        str(REPO_ROOT),
        "--timeout",
        str(config["watchdog"]["attempt_timeout_s"]),
        "--resume-contract",
        "target_checkpoint",
        "--resume-verifier-json",
        verifier,
        "--execution-output-root",
        training_output,
    ]
    if resume:
        args.append("--resume")
    args.extend(
        [
            python,
            trainer,
            "--authority",
            authority_path,
            "--expected-authority-sha256",
            authority["authority_sha256"],
            "--resume-policy",
            "auto",
            "--invocation-step-budget",
            str(invocation_step_budget),
            "--required-end-step",
            str(required_end_step),
        ]
    )
    return args


def _invoke_detached(args: list[str], *, failure_code: str) -> int:
    try:
        return int(detached.main(args))
    except SystemExit as exc:
        raise ResidentSFTCampaignControllerError(failure_code) from exc


def _wait_attempt(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    run_dir: Path,
    name: str,
    minimum_step: int,
    invocation_step_budget: int,
    required_end_step: int,
) -> dict[str, Any]:
    run_custody: DirectoryCustody | None = None
    bound_run = _custody_for(run_dir)
    if bound_run is not None:
        parent_custody, relative = bound_run
        parent_custody.ensure_directory(relative)
        run_custody = DirectoryCustody.acquire(run_dir, private=True)
    run_dir_identity = run_custody.identity if run_custody is not None else None
    try:
        return _wait_attempt_custodied(
            config_path=config_path,
            config=config,
            authority=authority,
            run_dir=run_dir,
            run_dir_identity=run_dir_identity,
            name=name,
            minimum_step=minimum_step,
            invocation_step_budget=invocation_step_budget,
            required_end_step=required_end_step,
        )
    finally:
        if run_custody is not None:
            run_custody.close()


def _wait_attempt_custodied(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    run_dir: Path,
    run_dir_identity: dict[str, int] | None,
    name: str,
    minimum_step: int,
    invocation_step_budget: int,
    required_end_step: int,
) -> dict[str, Any]:
    if not (run_dir / detached.PLAN_FILE).exists():
        result = _invoke_detached(
            _launch_args(
                config_path=config_path,
                config=config,
                authority=authority,
                run_dir=run_dir,
                run_dir_identity=run_dir_identity,
                name=name,
                minimum_step=minimum_step,
                invocation_step_budget=invocation_step_budget,
                required_end_step=required_end_step,
                resume=False,
            ),
            failure_code="resident_sft_controller_detached_cli_contract_invalid",
        )
        if result != 0:
            _fail("resident_sft_controller_detached_launch_failed")
    deadline = time.monotonic() + float(config["watchdog"]["attempt_timeout_s"]) + 120.0
    resume_used = False
    stale_stop_requested = False
    last_heartbeat_sequence: Any = None
    heartbeat_progress_at = time.monotonic()
    while time.monotonic() < deadline:
        try:
            status = detached._status(  # noqa: SLF001 - verified supervisor API
                run_dir,
                expected_identity=run_dir_identity,
            )
        except (detached.DetachedStepError, OSError, ValueError):
            time.sleep(float(config["watchdog"]["poll_interval_s"]))
            continue
        if status.get("terminal") is True:
            return dict(status)
        heartbeat_at = status.get("heartbeat_at")
        heartbeat_sequence = status.get("heartbeat_sequence")
        if heartbeat_sequence != last_heartbeat_sequence:
            last_heartbeat_sequence = heartbeat_sequence
            heartbeat_progress_at = time.monotonic()
        stale_by_wall_clock = (
            isinstance(heartbeat_at, (int, float))
            and not isinstance(heartbeat_at, bool)
            and time.time() - float(heartbeat_at) > float(config["watchdog"]["heartbeat_stale_s"])
        )
        stale_by_sequence = time.monotonic() - heartbeat_progress_at > float(
            config["watchdog"]["heartbeat_stale_s"]
        )
        if not stale_stop_requested and (stale_by_wall_clock or stale_by_sequence):
            try:
                stop = detached._stop(  # noqa: SLF001 - authenticated supervisor API
                    run_dir,
                    expected_identity=run_dir_identity,
                )
            except (detached.DetachedStepError, OSError, ValueError) as exc:
                raise ResidentSFTCampaignControllerError(
                    "resident_sft_controller_stale_supervisor_stop_failed"
                ) from exc
            if stop.get("stopped") is not True and stop.get("reason") not in {
                "already_terminal",
                "supervisor_not_alive",
            }:
                _fail("resident_sft_controller_stale_supervisor_stop_failed")
            stale_stop_requested = True
        if status.get("completion_indeterminate") is True:
            if resume_used:
                _fail("resident_sft_controller_detached_resume_failed")
            result = _invoke_detached(
                _launch_args(
                    config_path=config_path,
                    config=config,
                    authority=authority,
                    run_dir=run_dir,
                    run_dir_identity=run_dir_identity,
                    name=name,
                    minimum_step=minimum_step,
                    invocation_step_budget=invocation_step_budget,
                    required_end_step=required_end_step,
                    resume=True,
                ),
                failure_code="resident_sft_controller_detached_cli_contract_invalid",
            )
            if result != 0:
                _fail("resident_sft_controller_detached_resume_failed")
            resume_used = True
        time.sleep(float(config["watchdog"]["poll_interval_s"]))
    try:
        stop = detached._stop(  # noqa: SLF001 - authenticated supervisor API
            run_dir,
            expected_identity=run_dir_identity,
        )
    except (detached.DetachedStepError, OSError, ValueError) as exc:
        raise ResidentSFTCampaignControllerError(
            "resident_sft_controller_timeout_containment_failed"
        ) from exc
    if stop.get("stopped") is not True and stop.get("reason") not in {
        "already_terminal",
        "supervisor_not_alive",
    }:
        _fail("resident_sft_controller_timeout_containment_failed")
    containment_deadline = time.monotonic() + 60.0
    while time.monotonic() < containment_deadline:
        try:
            status = detached._status(  # noqa: SLF001 - verified supervisor API
                run_dir,
                expected_identity=run_dir_identity,
            )
        except (detached.DetachedStepError, OSError, ValueError):
            time.sleep(float(config["watchdog"]["poll_interval_s"]))
            continue
        if status.get("terminal") is True:
            return dict(status)
        if (
            status.get("completion_indeterminate") is True
            and status.get("supervisor_state") == "dead"
            and status.get("child_state") == "dead"
        ):
            _fail("resident_sft_controller_attempt_timeout")
        time.sleep(float(config["watchdog"]["poll_interval_s"]))
    _fail("resident_sft_controller_timeout_containment_failed")


def _reconcile_staged_results(
    journal: CampaignJournal,
    root: Path,
    plan: CampaignPlan,
    authority: Mapping[str, Any],
) -> None:
    committed = set(journal.resume().committed_cell_ids)
    for ordinal, cell_id in enumerate(plan.cell_ids, start=1):
        if cell_id in committed:
            continue
        for path in _attempt_result_paths(root, ordinal):
            record = _read_canonical(path, role="attempt_result")
            body = dict(record)
            claimed = body.pop("attempt_sha256", None)
            if claimed != sha256_json(body):
                _fail("resident_sft_controller_attempt_result_digest_mismatch")
            if (
                record.get("cell_id") != cell_id
                or record.get("required_end_reached") is not True
                or record.get("terminal_success") is not True
            ):
                continue
            required_end = int(plan.cell_definition(cell_id)["required_end_step"])
            current, verification = _verify_commit_evidence(
                authority=authority,
                record=record,
                required_end_step=required_end,
            )
            journal.import_committed_cell(
                cell_id,
                expected_attempt_id=record["attempt_id"],
                result=record,
                verification=verification,
                commit={
                    "schema": "aura.resident_recurrent_sft_attempt_commit.v1",
                    "step": current["step"],
                    "checkpoint_sequence": current["checkpoint_sequence"],
                },
            )
            committed.add(cell_id)
            break


def run_controller(config_path: Path, *, launchd_supervised: bool = False) -> dict[str, Any]:
    global _ACTIVE_CUSTODIES
    config_path = config_path.expanduser().resolve(strict=True)
    config_probe = _load_config(config_path)
    with _resident_training_host_lease(
        label=config_probe["launch"]["label"],
        config_sha256=config_probe["config_sha256"],
    ):
        config, authority, plan = _load_contracts(config_path)
        custodies = _acquire_campaign_custodies(config)
        _ACTIVE_CUSTODIES = custodies
        try:
            return _run_controller_custodied(
                config_path=config_path,
                config=config,
                authority=authority,
                plan=plan,
                custodies=custodies,
                launchd_supervised=launchd_supervised,
            )
        except ResidentSFTCampaignControllerError as exc:
            try:
                _publish_status(
                    custodies[-1].path,
                    config,
                    (
                        "restartable_failure"
                        if exc.code in RESTARTABLE_CONTROLLER_ERRORS
                        else "failed"
                    ),
                    {"error": exc.code},
                )
            except Exception as publish_exc:  # noqa: BLE001 — see _note_status_publish_failure
                _note_status_publish_failure(publish_exc, original=exc.code)
            raise
        except BaseException as exc:
            try:
                _publish_status(
                    custodies[-1].path,
                    config,
                    "failed",
                    {"error": str(exc) or "no_message", "error_type": type(exc).__name__},
                )
            except Exception as publish_exc:  # noqa: BLE001 — see _note_status_publish_failure
                _note_status_publish_failure(
                    publish_exc, original=type(exc).__name__
                )
            raise
        finally:
            for custody in reversed(custodies):
                custody.close()
            _ACTIVE_CUSTODIES = ()


def _run_controller_custodied(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    plan: CampaignPlan,
    custodies: Sequence[DirectoryCustody],
    launchd_supervised: bool,
) -> dict[str, Any]:
    _verify_campaign_custodies(custodies)
    supervision = _verify_execution_supervision(
        config,
        launchd_supervised=launchd_supervised,
    )
    root = (
        custodies[-1].path
        if custodies
        else ensure_private_directory(
            _repo_path(config["paths"]["controller_root"], role="controller", must_exist=False)
        )
    )
    lock_path = root / "controller.lock"
    with _controller_lock(lock_path) as acquired:
        if not acquired:
            return _publish_status(root, config, "already_running", {})
        journal_path = _repo_path(config["paths"]["journal"], role="journal", must_exist=False)
        manifest_path = _repo_path(config["paths"]["manifest"], role="manifest", must_exist=False)
        with CampaignJournal(
            journal_path,
            plan,
            custody=(custodies[-1] if custodies else None),
        ) as journal:
            _reconcile_staged_results(journal, root, plan, authority)
            migration_start = _verified_migration_start_step(authority)
            for ordinal, cell_id in enumerate(plan.cell_ids, start=1):
                if cell_id in journal.resume().committed_cell_ids:
                    continue
                _verify_source_lineage(config["source"])
                _verify_authority_artifacts(authority)
                _verify_campaign_custodies(custodies)
                definition = plan.cell_definition(cell_id)
                required_start = int(definition["expected_start_step"])
                required_end = int(definition["required_end_step"])
                before = _checkpoint_snapshot(authority)
                if before.get("terminal") is True and before.get("halt_reason") == "wall_clock":
                    _publish_status(
                        root,
                        config,
                        "budget_exhausted",
                        {
                            "step": before["step"],
                            "max_steps": authority["trainer"]["max_steps"],
                            "elapsed_training_s": before.get("elapsed_training_s"),
                            "max_minutes": authority["trainer"]["max_minutes"],
                        },
                    )
                    _fail("resident_sft_controller_training_budget_exhausted")
                initial_attempt_status = journal.attempt_status(cell_id)
                if migration_start is not None and required_end <= migration_start:
                    if (
                        initial_attempt_status["active_attempt_id"] is not None
                        or int(initial_attempt_status["attempt_count"]) != 0
                    ):
                        _fail("resident_sft_controller_migration_covered_cell_not_fresh")
                    _commit_migration_covered_cell(
                        journal,
                        cell_id=cell_id,
                        required_start=required_start,
                        required_end=required_end,
                        migration_start=migration_start,
                        snapshot=before,
                    )
                    continue
                fresh_start_valid = _initial_checkpoint_matches_plan(
                    observed_step=int(before["step"]),
                    required_start=required_start,
                    required_end=required_end,
                    migration_start=migration_start,
                )
                if (
                    initial_attempt_status["active_attempt_id"] is None
                    and not fresh_start_valid
                ) or (
                    initial_attempt_status["active_attempt_id"] is not None
                    and not required_start <= int(before["step"]) <= required_end
                ):
                    _fail("resident_sft_controller_checkpoint_plan_drift")
                while True:
                    attempt_status = journal.attempt_status(cell_id)
                    active_attempt_id = attempt_status["active_attempt_id"]
                    if active_attempt_id is None:
                        attempt_number = int(attempt_status["attempt_count"]) + 1
                    else:
                        attempt_number = int(attempt_status["active_attempt_number"])
                    if attempt_number > int(config["watchdog"]["max_attempts_per_cell"]):
                        _fail("resident_sft_controller_attempt_budget_exhausted")
                    if _trailing_no_progress(root, ordinal) >= 2:
                        _fail("resident_sft_controller_no_progress_limit_exhausted")
                    if active_attempt_id is None:
                        attempt_id = journal.start_cell(cell_id)
                        before = _checkpoint_snapshot(authority)
                        if not required_start <= int(before["step"]) <= required_end:
                            _fail("resident_sft_controller_checkpoint_plan_drift")
                        reservation = _reserve_attempt(
                            root=root,
                            config=config,
                            cell_id=cell_id,
                            cell_ordinal=ordinal,
                            attempt_id=attempt_id,
                            attempt_number=attempt_number,
                            before=before,
                            required_end_step=required_end,
                        )
                    else:
                        attempt_id = str(active_attempt_id)
                        loaded_reservation = _load_attempt_reservation(
                            root,
                            cell_ordinal=ordinal,
                            attempt_number=attempt_number,
                        )
                        if loaded_reservation is None:
                            recovered = _checkpoint_snapshot(authority)
                            record = _attempt_record(
                                config=config,
                                cell_id=cell_id,
                                cell_ordinal=ordinal,
                                attempt_id=attempt_id,
                                attempt_number=attempt_number,
                                before=recovered,
                                after=recovered,
                                detached_status={
                                    "plan_sha256": None,
                                    "receipt": None,
                                    "controller_error": "dispatch_contract_missing_after_restart",
                                },
                                required_end_step=required_end,
                            )
                            _write_once(
                                root
                                / "attempt-results"
                                / f"cell-{ordinal:04d}-attempt-{attempt_number:04d}.json",
                                record,
                            )
                            journal.fail_cell(
                                cell_id,
                                attempt_id,
                                reason="reserved_attempt_missing_dispatch_contract",
                                details={"attempt_sha256": record["attempt_sha256"]},
                            )
                            continue
                        reservation = loaded_reservation
                        if (
                            reservation.get("cell_id") != cell_id
                            or reservation.get("attempt_id") != attempt_id
                            or reservation.get("required_end_step") != required_end
                            or not isinstance(reservation.get("progress_before"), Mapping)
                        ):
                            _fail("resident_sft_controller_attempt_reservation_drift")
                        before = dict(reservation["progress_before"])
                        existing_result_path = _attempt_result_path(
                            root,
                            ordinal,
                            attempt_number,
                        )
                        if existing_result_path.is_file():
                            existing_result = _read_canonical(
                                existing_result_path,
                                role="attempt_result",
                            )
                            body = dict(existing_result)
                            claimed = body.pop("attempt_sha256", None)
                            if (
                                claimed != sha256_json(body)
                                or existing_result.get("cell_id") != cell_id
                                or existing_result.get("attempt_id") != attempt_id
                                or existing_result.get("attempt_number") != attempt_number
                                or existing_result.get("terminal_success") is True
                            ):
                                _fail("resident_sft_controller_recovered_attempt_result_invalid")
                            journal.fail_cell(
                                cell_id,
                                attempt_id,
                                reason="recovered_recorded_failure",
                                details={"attempt_sha256": claimed},
                            )
                            continue
                    run_dir = _run_dir(root, ordinal, attempt_number)
                    _publish_status(
                        root,
                        config,
                        "training",
                        {
                            "cell_ordinal": ordinal,
                            "cell_count": len(plan.cell_ids),
                            "attempt_number": attempt_number,
                            "step": before["step"],
                            "required_end_step": required_end,
                            "run_dir": str(run_dir),
                        },
                    )
                    try:
                        status = _wait_attempt(
                            config_path=config_path,
                            config=config,
                            authority=authority,
                            run_dir=run_dir,
                            name=(f"{config['campaign_id']}-c{ordinal:04d}-a{attempt_number:04d}"),
                            minimum_step=int(before["step"]),
                            invocation_step_budget=max(1, required_end - int(before["step"])),
                            required_end_step=required_end,
                        )
                    except ResidentSFTCampaignControllerError as exc:
                        if exc.code not in RESTARTABLE_CONTROLLER_ERRORS:
                            raise
                        status = {
                            "plan_sha256": None,
                            "receipt": None,
                            "controller_error": exc.code,
                        }
                    _verify_campaign_custodies(custodies)
                    after = _checkpoint_snapshot(authority)
                    detached_receipt = status.get("receipt")
                    if (
                        after["present"]
                        and isinstance(detached_receipt, Mapping)
                        and detached_receipt.get("returncode") == 0
                    ):
                        _invocation_receipt(authority, after)
                    record = _attempt_record(
                        config=config,
                        cell_id=cell_id,
                        cell_ordinal=ordinal,
                        attempt_id=attempt_id,
                        attempt_number=attempt_number,
                        before=before,
                        after=after,
                        detached_status=status,
                        required_end_step=required_end,
                    )
                    result_path = _attempt_result_path(root, ordinal, attempt_number)
                    _write_once(result_path, record)
                    if record["terminal_success"] is True:
                        current, verification = _verify_commit_evidence(
                            authority=authority,
                            record=record,
                            required_end_step=required_end,
                        )
                        journal.import_committed_cell(
                            cell_id,
                            expected_attempt_id=attempt_id,
                            result=record,
                            verification=verification,
                            commit={
                                "schema": "aura.resident_recurrent_sft_attempt_commit.v1",
                                "step": current["step"],
                                "checkpoint_sequence": current["checkpoint_sequence"],
                            },
                        )
                        before = current
                        break
                    journal.fail_cell(
                        cell_id,
                        attempt_id,
                        reason=(
                            "partial_durable_progress"
                            if record["durable_progress"]
                            else "no_durable_progress"
                        ),
                        details={
                            "attempt_sha256": record["attempt_sha256"],
                            "step": after["step"],
                        },
                    )
                    if record["durable_progress"]:
                        before = after
                    if _trailing_no_progress(root, ordinal) >= 2:
                        _fail("resident_sft_controller_no_progress_limit_exhausted")
                    time.sleep(float(config["watchdog"]["retry_backoff_s"]))

            final = _checkpoint_snapshot(authority)
            _verify_campaign_custodies(custodies)
            trainer_config = authority["trainer"]
            if (
                final["step"] != trainer_config["max_steps"]
                or final["terminal"] is not True
                or final.get("halt_reason") != "max_steps"
            ):
                _fail("resident_sft_controller_terminal_checkpoint_invalid")
            invocation = _invocation_receipt(authority, final)
            canary_lifecycle_complete = config["profile"] == "canary"
            bootstrap_complete = config["profile"] == "full"
            if (
                invocation.get("canary_lifecycle_complete") is not canary_lifecycle_complete
                or invocation.get("bootstrap_complete") is not bootstrap_complete
                or invocation.get("claim_state", {}).get("resident_sft_complete")
                is not bootstrap_complete
            ):
                _fail("resident_sft_controller_completion_scope_invalid")
            manifest = journal.finalize(manifest_path)
            body = {
                "schema": COMPLETION_SCHEMA,
                "campaign_id": config["campaign_id"],
                "config_sha256": config["config_sha256"],
                "authority_sha256": authority["authority_sha256"],
                "plan_sha256": plan.plan_sha256,
                "journal_manifest_sha256": manifest["manifest_sha256"],
                "checkpoint": final,
                "campaign_scope": authority["campaign_scope"],
                "canary_lifecycle_complete": canary_lifecycle_complete,
                "bootstrap_complete": bootstrap_complete,
                "base_checkpoint_immutable": True,
                "post_training_gates_required": True,
                "execution_supervision": supervision,
                "claim_state": config["claim_state"],
                "claims_supported": [
                    (
                        "resident_recurrent_sft_canary_lifecycle_completed"
                        if config["profile"] == "canary"
                        else "resident_recurrent_sft_bootstrap_completed"
                    )
                ],
            }
            completion = {**body, "completion_sha256": sha256_json(body)}
            _write_once(root / "completion-receipt.json", completion)
            _publish_status(root, config, "completed", completion)
            _verify_campaign_custodies(custodies)
            return completion


def verify_resume(config_path: Path, minimum_step: int) -> dict[str, Any]:
    global _ACTIVE_CUSTODIES
    config_path = config_path.expanduser().resolve(strict=True)
    config, authority, _plan = _load_contracts(config_path)
    custodies = _acquire_campaign_custodies(config)
    _ACTIVE_CUSTODIES = custodies
    try:
        return _verify_resume_custodied(config, authority, minimum_step)
    finally:
        for custody in reversed(custodies):
            custody.close()
        _ACTIVE_CUSTODIES = ()


def _verify_resume_custodied(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    minimum_step: int,
) -> dict[str, Any]:
    if type(minimum_step) is not int or minimum_step < 0:
        _fail("resident_sft_controller_resume_minimum_step_invalid")
    plan_sha = os.environ.get("AURA_DETACHED_PLAN_SHA256", "")
    command_sha = os.environ.get("AURA_DETACHED_COMMAND_SHA256", "")
    prior_head = os.environ.get("AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256", "")
    try:
        prior_attempt = int(os.environ.get("AURA_DETACHED_PRIOR_ATTEMPT", ""))
    except ValueError as exc:
        raise ResidentSFTCampaignControllerError(
            "resident_sft_controller_resume_context_invalid"
        ) from exc
    if (
        os.environ.get("AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT") != "stdout-v3"
        or prior_attempt < 1
        or any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in (plan_sha, command_sha, prior_head)
        )
    ):
        _fail("resident_sft_controller_resume_context_invalid")
    snapshot = _checkpoint_snapshot(authority)
    if snapshot["present"] is not True or int(snapshot["step"]) < minimum_step:
        _fail("resident_sft_controller_resume_checkpoint_unavailable")
    observed_base = full_weight_checkpoint_identity(
        _repo_path(authority["model"]["path"], role="model")
    )
    if observed_base != authority["model"]["base_checkpoint"]:
        _fail("resident_sft_controller_resume_base_checkpoint_drift")
    checkpoint_sequence = int(snapshot["checkpoint_sequence"])
    evidence = {
        "schema": "aura.detached_step.resume_evidence.v2",
        "plan_sha256": plan_sha,
        "command_sha256": command_sha,
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": prior_head,
        "checkpoint_sequence": checkpoint_sequence,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "minimum_step": minimum_step,
        "checkpoint": snapshot,
        "base_checkpoint": observed_base,
    }
    evidence_sha = sha256_json(evidence)
    checkpoint_identity = sha256_json(
        {
            "prior_attempt": prior_attempt,
            "prior_journal_head_sha256": prior_head,
            "checkpoint_sequence": checkpoint_sequence,
            "evidence_sha256": evidence_sha,
        }
    )
    return {
        "schema": "aura.detached_step.resume_verdict.v3",
        "plan_sha256": plan_sha,
        "command_sha256": command_sha,
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": prior_head,
        "checkpoint_sequence": checkpoint_sequence,
        "checkpoint_identity": checkpoint_identity,
        "verdict": "safe_to_resume",
        "evidence_sha256": evidence_sha,
        "evidence": evidence,
    }


def install_launchd(config_path: Path) -> dict[str, Any]:
    global _ACTIVE_CUSTODIES
    config, authority, _plan = _load_contracts(config_path.expanduser().resolve(strict=True))
    if config["launch"] != {
        "label": f"com.aura.resident-sft.{config['campaign_id']}",
        "launchd_required": True,
        "caffeinate_required": True,
    }:
        _fail("resident_sft_controller_launchd_policy_invalid")
    label = config["launch"]["label"]
    custodies = _acquire_campaign_custodies(config)
    _ACTIVE_CUSTODIES = custodies
    try:
        return _install_launchd_custodied(
            config_path=config_path,
            config=config,
            authority=authority,
            label=label,
            custodies=custodies,
        )
    finally:
        for custody in reversed(custodies):
            custody.close()
        _ACTIVE_CUSTODIES = ()


def _install_launchd_custodied(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    label: str,
    custodies: Sequence[DirectoryCustody],
) -> dict[str, Any]:
    root = custodies[-1].path
    interpreter = authority.get("runtime", {}).get("interpreter", {})
    python = interpreter.get("executable")
    if not isinstance(python, str) or not python:
        _fail("resident_sft_controller_interpreter_identity_invalid")
    install_lock = _RESIDENT_TRAINING_STATE_DIR / "install.lock"
    with _resident_training_lock(
        install_lock,
        busy_code="resident_training_install_busy",
    ):
        retirement = _retire_resident_training_jobs(active_label=label)
        return _install_exclusive_launchd_custodied(
            config_path=config_path,
            config=config,
            authority=authority,
            label=label,
            root=root,
            retirement=retirement,
        )


def _install_exclusive_launchd_custodied(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    label: str,
    root: Path,
    retirement: Mapping[str, Any],
) -> dict[str, Any]:
    interpreter = authority.get("runtime", {}).get("interpreter", {})
    python = interpreter.get("executable")
    if not isinstance(python, str) or not python:
        _fail("resident_sft_controller_interpreter_identity_invalid")
    payload = {
        "Label": label,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-i",
            python,
            str(Path(__file__).resolve(strict=True)),
            "run",
            "--config",
            str(config_path.expanduser().resolve(strict=True)),
            "--launchd-supervised",
        ],
        "WorkingDirectory": str(REPO_ROOT),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "StandardOutPath": str(root / "controller.log"),
        "StandardErrorPath": str(root / "controller.log"),
    }
    launch_agents = ensure_private_directory(Path.home() / "Library/LaunchAgents")
    plist_path = launch_agents / f"{label}.plist"
    plist_bytes = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)
    atomic_write_bytes(plist_path, plist_bytes, mode=0o600)
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    started = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if started.returncode != 0:
        _fail(f"resident_sft_controller_launchd_bootstrap_failed:{started.returncode}")
    job = _wait_launchd_job(label)
    receipt_path = root / "launchd-receipt.json"
    if receipt_path.is_file():
        existing = _read_canonical(receipt_path, role="launchd_receipt")
        existing_body = dict(existing)
        existing_digest = existing_body.pop("launch_sha256", None)
        if (
            existing_digest != sha256_json(existing_body)
            or existing.get("campaign_id") != config["campaign_id"]
            or existing.get("config_sha256") != config["config_sha256"]
            or existing.get("plist_sha256") != sha256_bytes(plist_bytes)
        ):
            _fail("resident_sft_controller_launchd_receipt_drift")
        return existing
    body = {
        "schema": LAUNCH_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "label": label,
        "domain": domain,
        "plist_path": str(plist_path),
        "plist_sha256": sha256_bytes(plist_bytes),
        "caffeinate": True,
        "launchd_keepalive": True,
        "launchd_target": job["target"],
        "launchd_pid": job["job_pid"],
        "retirement": dict(retirement),
        "installed_at_unix_ns": time.time_ns(),
    }
    receipt = {**body, "launch_sha256": sha256_json(body)}
    _write_once(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--launchd-supervised", action="store_true")
    verify = commands.add_parser("verify-resume")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--minimum-step", type=int, required=True)
    install = commands.add_parser("install")
    install.add_argument("--config", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "run":
            payload = run_controller(
                args.config,
                launchd_supervised=args.launchd_supervised,
            )
        elif args.action == "verify-resume":
            payload = verify_resume(args.config, args.minimum_step)
        elif args.action == "install":
            payload = install_launchd(args.config)
        else:
            config = _load_config(args.config.expanduser().resolve(strict=True))
            root = _repo_path(config["paths"]["controller_root"], role="controller")
            payload = _read_canonical(root / "status.json", role="status")
    except ResidentSFTCampaignControllerError as exc:
        error = {
            "schema": "aura.resident_recurrent_sft_controller_error.v1",
            "error_type": type(exc).__name__,
            "error": exc.code,
            "claims_supported": [],
        }
        print(json.dumps(error, sort_keys=True), file=sys.stderr, flush=True)
        if args.action == "run":
            if args.launchd_supervised and exc.code not in RESTARTABLE_CONTROLLER_ERRORS:
                return 0
            return 1
        return 2
    except Exception as exc:  # noqa: BLE001 - printed to stderr and returned non-zero
        error = {
            "schema": "aura.resident_recurrent_sft_controller_crash.v1",
            "error_type": type(exc).__name__,
            "error": str(exc) or "no_message",
            "traceback": traceback.format_exc(),
            "claims_supported": [],
        }
        print(json.dumps(error, sort_keys=True), file=sys.stderr, flush=True)
        if args.action == "run" and args.launchd_supervised:
            return 0
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
