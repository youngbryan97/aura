"""core/actuators/process_supervisor.py
======================================
Spawns and manages long-running background processes.
Streams stdout/stderr to files under the configured log directory.
Integrates with SubstrateGovernor to scale clock speed during execution.

Hardening (CP126): spawns are constrained to an executable allowlist, run in a
bounded launch workspace with a scrubbed environment and OS resource limits,
are killed by process group (no orphan descendants), and every spawn/exit/kill
is receipted. The registry is lock-synchronized and completed jobs are reaped.
"""

import hashlib
import json
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from core.actuators.actuator_registry import ActuatorResult, BaseActuator
from core.actuators.authority import verify_actuator_authority
from core.container import ServiceContainer
from core.runtime.subprocess_gateway import get_subprocess_gateway

try:  # POSIX-only; guarded so the module imports on every platform.
    import resource as _resource
except ImportError:  # pragma: no cover - non-POSIX
    # not a failure: the comment above says POSIX-only, and every use
    # checks the name before calling into it.
    _resource = None  # type: ignore[assignment]

logger = logging.getLogger("Aura.ProcessSupervisor")

# ── Spawn constraints ────────────────────────────────────────────────────────

# Only these executables (matched by basename) may be spawned. Extend for a
# deployment via AURA_PROCESS_SPAWN_ALLOWLIST (comma-separated basenames).
_DEFAULT_ALLOWED_COMMANDS = frozenset({
    "python", "python3", "python3.12", "python3.13", "python3.14",
    "pip", "pip3", "pytest", "git", "make", "ruff", "mypy",
    "node", "npm", "caffeinate",
})

# Environment keys whose mere name marks a secret — never copied to a child.
_SENSITIVE_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "KEY", "CREDENTIAL", "AUTH")

# Loader-injection variables a caller must never set on a child.
_DANGEROUS_ENV_KEYS = frozenset({
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
    "BASH_ENV", "ENV", "PYTHONSTARTUP",
})

_MAX_JOBS = 16                     # concurrent running jobs
_MAX_ENV_VARS = 256
_MAX_ENV_VALUE_LEN = 8192
_PREVIEW_TAIL_BYTES = 16 * 1024    # bytes read from the tail for a log preview
_PREVIEW_LINES = 20
_PREVIEW_MAX_LINE = 2048           # per-line clamp in a preview
_COMPLETED_RETENTION_S = 3600.0    # keep finished job metadata this long
_RLIMIT_CPU_SECONDS = int(os.environ.get("AURA_PROCESS_RLIMIT_CPU_S", "3600") or "3600")
_RLIMIT_FSIZE_BYTES = int(os.environ.get("AURA_PROCESS_RLIMIT_FSIZE", str(512 * 1024 * 1024)) or str(512 * 1024 * 1024))
_RLIMIT_NPROC = int(os.environ.get("AURA_PROCESS_RLIMIT_NPROC", "256") or "256")


def _allowed_commands() -> frozenset[str]:
    extra = os.environ.get("AURA_PROCESS_SPAWN_ALLOWLIST", "")
    if not extra.strip():
        return _DEFAULT_ALLOWED_COMMANDS
    names = {part.strip() for part in extra.split(",") if part.strip()}
    return _DEFAULT_ALLOWED_COMMANDS | names


def _resolve_logs_dir(logs_dir: str) -> str:
    """Anchor logs under the configured log dir, not the launch cwd."""
    base = os.environ.get("AURA_LOG_DIR", "").strip()
    if base and not os.path.isabs(logs_dir):
        return os.path.abspath(os.path.join(base, logs_dir))
    return os.path.abspath(logs_dir)


def _resolve_workspace_root() -> str:
    """The immutable root a job may run in (never the live process cwd)."""
    root = os.environ.get("AURA_PROCESS_WORKSPACE", "").strip() or os.getcwd()
    return os.path.realpath(os.path.abspath(root))


def _scrub_base_env() -> dict[str, str]:
    """A clean base environment with secret-bearing variables removed."""
    env = {
        key: value
        for key, value in os.environ.items()
        if isinstance(value, str)
        and not any(marker in key.upper() for marker in _SENSITIVE_ENV_MARKERS)
        and key not in _DANGEROUS_ENV_KEYS
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _merge_caller_env(base: dict[str, str], caller_env: Any) -> tuple[dict[str, str] | None, str]:
    """Validate and merge caller-supplied environment overrides."""
    if caller_env is None:
        return base, ""
    if not isinstance(caller_env, dict):
        return None, "env must be a mapping of string->string"
    if len(caller_env) > _MAX_ENV_VARS:
        return None, f"env has too many entries (>{_MAX_ENV_VARS})"
    for key, value in caller_env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return None, f"env entry {key!r} must be string->string"
        if key in _DANGEROUS_ENV_KEYS or any(m in key.upper() for m in _SENSITIVE_ENV_MARKERS):
            return None, f"env key {key!r} is not permitted"
        if "\x00" in key or "\n" in key or "=" in key:
            return None, f"env key {key!r} contains an illegal character"
        if len(value) > _MAX_ENV_VALUE_LEN or "\x00" in value:
            return None, f"env value for {key!r} is invalid or too large"
        base[key] = value
    return base, ""


def _validate_command(command: Any) -> tuple[list[str] | None, str]:
    """Parse and allowlist a spawn command. Returns (argv, error)."""
    if isinstance(command, str):
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return None, f"invalid command syntax: {exc}"
    elif isinstance(command, list):
        argv = command
    else:
        return None, "command must be a string or list of strings"
    if not argv or not all(isinstance(x, str) and x for x in argv):
        return None, "command must be a non-empty list of strings"
    for part in argv:
        if "\x00" in part or "\n" in part or "\r" in part:
            return None, "command contains control characters"
    binary = Path(argv[0]).name
    if binary not in _allowed_commands():
        return None, f"command '{binary}' is not in the spawn allowlist"
    return argv, ""


def _validate_cwd(requested: Any, workspace_root: str) -> tuple[str | None, str]:
    """Confine a job's working directory to the immutable workspace root."""
    if requested is None:
        return workspace_root, ""
    if not isinstance(requested, str) or not requested.strip():
        return None, "cwd must be a non-empty string"
    resolved = os.path.realpath(os.path.abspath(requested))
    if resolved != workspace_root and not resolved.startswith(workspace_root + os.sep):
        return None, f"cwd escapes the process workspace root ({workspace_root})"
    if not os.path.isdir(resolved):
        return None, f"cwd does not exist: {resolved}"
    return resolved, ""


def _rlimit_preexec() -> None:  # pragma: no cover - runs in the child
    """Apply conservative resource limits in the child before exec."""
    if _resource is None:
        return
    for what, soft in (
        (getattr(_resource, "RLIMIT_CPU", None), _RLIMIT_CPU_SECONDS),
        (getattr(_resource, "RLIMIT_FSIZE", None), _RLIMIT_FSIZE_BYTES),
        (getattr(_resource, "RLIMIT_NPROC", None), _RLIMIT_NPROC),
    ):
        if what is None:
            continue
        try:
            hard = _resource.getrlimit(what)[1]
            cap = soft if hard == _resource.RLIM_INFINITY else min(soft, hard)
            _resource.setrlimit(what, (cap, hard))
        except (ValueError, OSError):
            continue


def _redact_command(argv: list[str]) -> list[str]:
    """Redact obvious secrets in a command echoed back to a caller."""
    out: list[str] = []
    for part in argv:
        low = part.lower()
        if any(m.lower() in low for m in _SENSITIVE_ENV_MARKERS) and "=" in part:
            key, _, _ = part.partition("=")
            out.append(f"{key}=***redacted***")
        else:
            out.append(part)
    return out


def _tail_preview(path: str) -> str:
    """Read only the tail of a log for a bounded preview."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > _PREVIEW_TAIL_BYTES:
                fh.seek(-_PREVIEW_TAIL_BYTES, os.SEEK_END)
            data = fh.read()
    except (OSError, ValueError):
        return ""
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()[-_PREVIEW_LINES:]
    return "\n".join(line[:_PREVIEW_MAX_LINE] for line in lines)


class ProcessSupervisorActuator(BaseActuator):
    """Actuator to manage, inspect, and kill background jobs."""

    requires_authority = True

    def __init__(self, logs_dir: str = "logs/processes"):
        self.logs_dir = _resolve_logs_dir(logs_dir)
        os.makedirs(self.logs_dir, exist_ok=True)
        self._workspace_root = _resolve_workspace_root()
        self._receipt_path = os.path.join(self.logs_dir, "process_receipts.jsonl")
        self._processes: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._governor_leased = False

    @property
    def name(self) -> str:
        return "process_supervisor"

    @property
    def description(self) -> str:
        return "Spawns, lists, queries, and kills background tasks, streaming output to log files."

    def validate_params(self, params: dict[str, Any]) -> bool:
        if not isinstance(params, dict) or "action" not in params:
            return False
        action = params["action"]
        if action not in ("spawn", "list", "query", "kill"):
            return False
        if action == "spawn":
            if not bool(params.get("allow_spawn")):
                return False
            if "command" not in params:
                return False
            argv, _err = _validate_command(params["command"])
            if argv is None:
                return False
        if action in ("query", "kill"):
            if "process_id" not in params or not isinstance(params["process_id"], str):
                return False
        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        _authorized, _auth_reason = verify_actuator_authority(params, actuator=self.name)
        if not _authorized:
            return ActuatorResult(False, _auth_reason, {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Parameter validation failed for process supervisor.", {})

        action = params["action"]

        if action == "spawn":
            return self._handle_spawn(params)
        elif action == "list":
            return self._handle_list()
        elif action == "query":
            return self._handle_query(params["process_id"])
        elif action == "kill":
            return self._handle_kill(params["process_id"])

        return ActuatorResult(False, f"Unsupported action: {action}", {})

    # ── Spawn ───────────────────────────────────────────────────────────

    def _handle_spawn(self, params: dict[str, Any]) -> ActuatorResult:
        argv, err = _validate_command(params["command"])
        if argv is None:
            return ActuatorResult(False, f"Refused spawn: {err}", {})

        cwd, cwd_err = _validate_cwd(params.get("cwd"), self._workspace_root)
        if cwd is None:
            return ActuatorResult(False, f"Refused spawn: {cwd_err}", {})

        env, env_err = _merge_caller_env(_scrub_base_env(), params.get("env"))
        if env is None:
            return ActuatorResult(False, f"Refused spawn: {env_err}", {})

        # Concurrency budget — count only jobs that are still running.
        with self._lock:
            running = sum(1 for p in self._processes.values() if p["proc"].poll() is None)
            if running >= _MAX_JOBS:
                return ActuatorResult(
                    False, f"Refused spawn: max concurrent jobs ({_MAX_JOBS}) reached.", {}
                )

        process_id = f"proc_{uuid.uuid4().hex[:8]}"
        stdout_path = os.path.join(self.logs_dir, f"{process_id}_stdout.log")
        stderr_path = os.path.join(self.logs_dir, f"{process_id}_stderr.log")

        try:
            proc = get_subprocess_gateway().spawn(
                argv,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cwd=cwd,
                env=env,
                text=True,
                start_new_session=True,
                preexec_fn=_rlimit_preexec if _resource is not None else None,
                source="process_supervisor",
                accelerator_capability="auto",
            )
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            return ActuatorResult(False, f"Process spawn failed: {e}", {})

        with self._lock:
            self._processes[process_id] = {
                "proc": proc,
                "argv": argv,
                "command": argv,
                "start_time": time.time(),
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "log_streams": getattr(proc, "_aura_gateway_streams", ()),
                "cwd": cwd,
                "completed_at": None,
                "streams_closed": False,
            }
            self._lease_governor()

        self._write_receipt(
            "spawn", process_id, argv=argv, cwd=cwd, env=env, pid=proc.pid
        )

        msg = f"Background process spawned successfully with ID: {process_id} (PID: {proc.pid})."
        return ActuatorResult(
            success=True,
            message=msg,
            updates={
                "process_id": process_id,
                "pid": proc.pid,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
            },
        )

    # ── List / Query ────────────────────────────────────────────────────

    def _handle_list(self) -> ActuatorResult:
        result = {}
        now = time.time()
        with self._lock:
            for pid, pinfo in list(self._processes.items()):
                exit_code = pinfo["proc"].poll()
                is_running = exit_code is None
                if not is_running:
                    self._on_completed(pid, pinfo, exit_code)
                    if pinfo["completed_at"] and (now - pinfo["completed_at"]) > _COMPLETED_RETENTION_S:
                        self._processes.pop(pid, None)
                        continue
                result[pid] = {
                    "command": _redact_command(pinfo["argv"]),
                    "start_time": pinfo["start_time"],
                    "is_running": is_running,
                    "exit_code": exit_code,
                    "stdout_path": pinfo["stdout_path"],
                    "stderr_path": pinfo["stderr_path"],
                }
            running_count = sum(1 for v in result.values() if v["is_running"])
            if running_count == 0:
                self._release_governor()

        return ActuatorResult(True, f"Listed {len(result)} managed processes.", {"processes": result})

    def _handle_query(self, process_id: str) -> ActuatorResult:
        with self._lock:
            pinfo = self._processes.get(process_id)
            if pinfo is None:
                return ActuatorResult(False, f"Process with ID {process_id} not found.", {})
            exit_code = pinfo["proc"].poll()
            is_running = exit_code is None
            if not is_running:
                self._on_completed(process_id, pinfo, exit_code)
            stdout_path = pinfo["stdout_path"]
            stderr_path = pinfo["stderr_path"]
            pid = pinfo["proc"].pid
            argv = pinfo["argv"]
            start_time = pinfo["start_time"]

        details = {
            "process_id": process_id,
            "pid": pid,
            "command": _redact_command(argv),
            "start_time": start_time,
            "is_running": is_running,
            "exit_code": exit_code,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "stdout_preview": _tail_preview(stdout_path),
            "stderr_preview": _tail_preview(stderr_path),
        }

        msg = (
            f"Process {process_id} is running (PID: {pid})."
            if is_running
            else f"Process {process_id} completed with exit code {exit_code}."
        )
        return ActuatorResult(True, msg, {"process_details": details})

    # ── Kill ────────────────────────────────────────────────────────────

    def _handle_kill(self, process_id: str) -> ActuatorResult:
        with self._lock:
            pinfo = self._processes.get(process_id)
            if pinfo is None:
                return ActuatorResult(False, f"Process with ID {process_id} not found.", {})
            proc = pinfo["proc"]
            exit_code = proc.poll()
            if exit_code is not None:
                self._on_completed(process_id, pinfo, exit_code)
                return ActuatorResult(
                    True,
                    f"Process {process_id} was already stopped (exit code {exit_code}).",
                    {"exit_code": exit_code},
                )

        try:
            self._terminate_group(proc, graceful_deadline_s=3.0)
            exit_code = proc.poll()
            with self._lock:
                self._on_completed(process_id, pinfo, exit_code)
            self._write_receipt("kill", process_id, exit_code=exit_code)
            self._handle_list()  # refresh governor lease
            return ActuatorResult(True, f"Process {process_id} killed successfully.", {"exit_code": exit_code})
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            return ActuatorResult(False, f"Failed to kill process {process_id}: {e}", {})

    @staticmethod
    def _terminate_group(proc: Any, *, graceful_deadline_s: float) -> None:
        """Terminate the whole process group so descendants don't orphan."""
        def _signal(sig: int) -> None:
            sent = False
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                    sent = True
                except (ProcessLookupError, PermissionError, OSError):
                    # not a failure: the group signal is the first rung, and
                    # `if not sent` below signals the process directly.
                    sent = False
            if not sent:
                try:
                    proc.send_signal(sig)
                except (ProcessLookupError, OSError):
                    # not a failure: a process already gone needs no signal,
                    # and the poll loop below is what decides it ended.
                    pass

        _signal(signal.SIGTERM)
        deadline = time.time() + graceful_deadline_s
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        else:
            _signal(signal.SIGKILL)
            try:
                proc.wait(timeout=5.0)
            except (subprocess.TimeoutExpired, OSError) as exc:
                # SIGKILL was the last rung. A child that does not reap
                # after it is a zombie this supervisor left behind.
                logger.warning("Process %s did not reap after SIGKILL: %s", proc.pid, exc)

    # ── Lifecycle bookkeeping ───────────────────────────────────────────

    def _on_completed(self, process_id: str, pinfo: dict[str, Any], exit_code: Any) -> None:
        """Idempotently mark a job done: close streams once, receipt once."""
        if pinfo.get("streams_closed"):
            return
        pinfo["completed_at"] = time.time()
        pinfo["streams_closed"] = True
        for stream in pinfo.get("log_streams", ()):
            try:
                stream.close()
            except (OSError, ValueError) as exc:
                logger.debug("Process log close failed for %s: %s", process_id, exc)
        self._write_receipt("exit", process_id, exit_code=exit_code)

    def _lease_governor(self) -> None:
        if self._governor_leased:
            return
        governor = ServiceContainer.get("governor", default=None)
        if governor:
            try:
                governor.apply_volition_profile(3)  # 20.0 Hz focus while jobs run
                self._governor_leased = True
            except (RuntimeError, AttributeError, TypeError, ValueError) as ex:
                logger.warning("Failed to scale substrate governor frequency: %s", ex)

    def _release_governor(self) -> None:
        if not self._governor_leased:
            return
        governor = ServiceContainer.get("governor", default=None)
        if governor:
            try:
                governor.apply_volition_profile(1)  # revert to 10.0 Hz reflective
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("Failed to restore substrate governor frequency: %s", exc)
        self._governor_leased = False

    def close(self) -> None:
        """Drain: terminate every live child, close streams, restore governor."""
        with self._lock:
            items = list(self._processes.items())
        for process_id, pinfo in items:
            proc = pinfo["proc"]
            try:
                if proc.poll() is None:
                    self._terminate_group(proc, graceful_deadline_s=2.0)
            except (OSError, ValueError) as exc:
                logger.debug("Drain terminate failed for %s: %s", process_id, exc)
            with self._lock:
                self._on_completed(process_id, pinfo, proc.poll())
        with self._lock:
            self._processes.clear()
            self._release_governor()

    # ── Receipts ────────────────────────────────────────────────────────

    def _write_receipt(self, event: str, process_id: str, **fields: Any) -> None:
        """Append a tamper-visible spawn/exit/kill receipt (best-effort)."""
        record: dict[str, Any] = {
            "event": event,
            "process_id": process_id,
            "timestamp": time.time(),
        }
        if "argv" in fields:
            argv = fields.pop("argv")
            record["command_sha256"] = hashlib.sha256(
                "\x1f".join(argv).encode()
            ).hexdigest()
            record["command_redacted"] = _redact_command(argv)
        if "env" in fields:
            env = fields.pop("env")
            record["env_sha256"] = hashlib.sha256(
                json.dumps(dict(sorted(env.items())), default=str).encode()
            ).hexdigest()
            record["env_var_count"] = len(env)
        if "cwd" in fields:
            record["cwd"] = fields.pop("cwd")
        record.update({k: v for k, v in fields.items()})
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope(
                "actuators.process_supervisor.receipt",
                domain="file_write",
                receipt_prefix="process-supervisor-receipt",
            ):
                get_file_write_gateway().append_text(
                    self._receipt_path,
                    json.dumps(record, default=str) + "\n",
                    encoding="utf-8",
                    source="actuators.process_supervisor.receipt",
                )
        except (OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
            logger.debug("Process receipt write failed (%s): %s", event, exc)
