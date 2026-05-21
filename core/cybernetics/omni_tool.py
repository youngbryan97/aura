from __future__ import annotations

import asyncio
import inspect
import logging
import shlex
import time
from collections.abc import Callable
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Cybernetics.OmniTool")

MAX_ACTION_NAME_CHARS = 128
MAX_COMMAND_CHARS = 4096
MAX_LOG_ENTRIES = 50
MAX_DAEMON_OUTPUT_BYTES = 65536
MAX_DAEMON_DRAIN_CHUNKS = 256
DEFAULT_DAEMON_TIMEOUT_S = 3600.0

_OMNI_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_omni_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "omni_tool",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation(
                "omni_tool",
                error,
                severity=severity,
                action=action or "omni tool degraded",
            )
        except TypeError:
            logger.warning(
                "OmniTool degradation could not be recorded: %s",
                signature_exc,
            )


def _safe_text(value: object, *, default: str = "", max_chars: int = MAX_COMMAND_CHARS) -> str:
    try:
        text = str(value if value is not None else default)
    except (RuntimeError, TypeError, ValueError):
        text = default
    return text.replace("\x00", "")[:max_chars].strip()


def _safe_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(0.0, parsed)


def _trim_logs(logs: list[dict[str, Any]]) -> None:
    while len(logs) > MAX_LOG_ENTRIES:
        logs.pop(0)


async def _read_limited_stream(stream: asyncio.StreamReader | None) -> str:
    if stream is None:
        return ""
    captured = bytearray()
    for _ in range(MAX_DAEMON_DRAIN_CHUNKS):
        chunk = await stream.read(4096)
        if not chunk:
            break
        remaining = MAX_DAEMON_OUTPUT_BYTES - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
    else:
        captured.extend(b"\n[omni_tool: output drain budget exhausted]\n")
    return captured.decode("utf-8", errors="replace")


class OmniTool:
    """
    [ZENITH] The Omni-Tool Command Interface (Mass Effect inspired).
    A unified router for safe, permissioned, and cooldown-protected tool execution.
    """
    def __init__(self, kernel: Any = None):
        self.kernel = kernel
        self._event_bus = None
        self._cooldowns: dict[str, float] = {}
        # Consolidation: Unified Execution Logs
        self._execution_logs: dict[str, list[dict[str, Any]]] = {}
        self._permissions: dict[str, bool] = {
            "reboot": False,
            "kernel_patch": False,
            "external_request": True,
        }
        # Daemon Supervisor: Proactive Task Management
        self._daemons: dict[str, dict[str, Any]] = {}

    async def load(self):
        try:
            from core.event_bus import get_event_bus
            self._event_bus = get_event_bus()
        except _OMNI_ERRORS as exc:
            self._event_bus = None
            _record_omni_degradation(
                exc,
                action="loaded omni tool without event bus publication support",
                severity="warning",
            )
        logger.info("🔋 [OMNI] Omni-Tool Interface ENGAGED. Field actions READY.")

    async def execute_action(self, action_name: str, handler: Callable, *args, **kwargs) -> Any:
        """
        Executes a field action with standardized safety guardrails.
        """
        now = time.time()
        action_name = _safe_text(action_name, max_chars=MAX_ACTION_NAME_CHARS)
        if not action_name:
            return {"error": "invalid_action_name"}
        if not callable(handler):
            return {"error": "invalid_handler"}
        
        # 1. Cooldown Enforcement
        last_run = self._cooldowns.get(action_name, 0.0)
        cooldown_period = min(_safe_float(kwargs.pop("cooldown", 5.0), 5.0), 86400.0)
        if now - last_run < cooldown_period:
            logger.warning("⏳ [OMNI] Action '%s' is cooling down.", action_name)
            return {"error": "cooldown_active", "remaining": cooldown_period - (now - last_run)}

        # 2. Permission Validation
        if not self._permissions.get(action_name, True):
            logger.error("🚫 [OMNI] Action '%s' DENIED by system security policy.", action_name)
            return {"error": "permission_denied"}

        # 3. Execution with Error Catching
        try:
            logger.info("🚀 [OMNI] Executing Field Action: %s", action_name)
            self._cooldowns[action_name] = now
            
            if inspect.iscoroutinefunction(handler):
                result = await handler(*args, **kwargs)
            else:
                result = handler(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            
            # Log Success
            logs = self._execution_logs.setdefault(action_name, [])
            logs.append({"ts": now, "status": "success"})
            _trim_logs(logs)
            return result
            
        except _OMNI_ERRORS as e:
            _record_omni_degradation(
                e,
                action="returned structured field action failure",
                severity="degraded",
                extra={"action_name": action_name},
            )
            logger.error("💥 [OMNI] Action '%s' FAILED: %s", action_name, e)
            # Log Failure
            logs = self._execution_logs.setdefault(action_name, [])
            logs.append({"ts": now, "status": "error", "error": str(e)})
            _trim_logs(logs)
            return {"error": str(e)}

    async def spawn_daemon(
        self,
        name: str,
        command: str,
        *,
        timeout_s: float = DEFAULT_DAEMON_TIMEOUT_S,
    ) -> dict[str, Any]:
        """[DAEMON] Spawns a proactive background task."""
        name = _safe_text(name, max_chars=MAX_ACTION_NAME_CHARS)
        command = _safe_text(command)
        timeout_s = min(_safe_float(timeout_s, DEFAULT_DAEMON_TIMEOUT_S), 86400.0)
        if not name:
            return {"status": "error", "message": "Daemon name is required."}
        if not command:
            return {"status": "error", "message": "Daemon command is required."}
        if not self._permissions.get("external_request", True):
            return {"status": "error", "message": "Daemon execution denied by policy."}
        if name in self._daemons:
            return {"status": "error", "message": f"Daemon '{name}' already exists."}

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            _record_omni_degradation(
                exc,
                action="rejected malformed daemon command",
                severity="warning",
                extra={"daemon": name},
            )
            return {"status": "error", "message": "Malformed daemon command."}
        if not argv:
            return {"status": "error", "message": "Daemon command is empty."}

        metadata = {
            "name": name,
            "command": command,
            "start_time": time.time(),
            "status": "starting",
            "pid": None,
            "returncode": None,
        }
        self._daemons[name] = metadata

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except _OMNI_ERRORS as exc:
            metadata["status"] = "failed_to_start"
            metadata["end_time"] = time.time()
            metadata["error"] = str(exc)
            _record_omni_degradation(
                exc,
                action="failed to start supervised daemon process",
                severity="degraded",
                extra={"daemon": name, "command": command},
            )
            return {"status": "error", "daemon": metadata}

        metadata["status"] = "running"
        metadata["pid"] = process.pid
        logger.info("🕯️ [DAEMON] LIGHTING: %s -> %s", name, command)

        get_task_tracker().create_task(
            self._watch_daemon(name, process, timeout_s),
            name=f"omni_tool.daemon.{name}",
        )

        if self._event_bus:
            await self._publish_daemon_event("core/cybernetics/daemon_spawned", metadata)

        return {"status": "spawned", "daemon": metadata}

    async def _watch_daemon(
        self,
        name: str,
        process: asyncio.subprocess.Process,
        timeout_s: float,
    ) -> None:
        metadata = self._daemons.get(name)
        if metadata is None:
            return
        stdout_task = get_task_tracker().create_task(
            _read_limited_stream(process.stdout),
            name=f"omni_tool.daemon.{name}.stdout",
        )
        stderr_task = get_task_tracker().create_task(
            _read_limited_stream(process.stderr),
            name=f"omni_tool.daemon.{name}.stderr",
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_s)
            metadata["status"] = "completed" if process.returncode == 0 else "failed"
        except TimeoutError:
            metadata["status"] = "timed_out"
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                process.kill()
                await process.wait()
            _record_omni_degradation(
                TimeoutError(f"daemon {name} exceeded {timeout_s:.1f}s"),
                action="terminated daemon after timeout",
                severity="warning",
                extra={"daemon": name},
            )
        except _OMNI_ERRORS as exc:
            metadata["status"] = "watch_failed"
            metadata["error"] = str(exc)
            _record_omni_degradation(
                exc,
                action="daemon watcher failed after process start",
                severity="degraded",
                extra={"daemon": name},
            )
        finally:
            metadata["returncode"] = process.returncode
            metadata["end_time"] = time.time()
            metadata["stdout_tail"] = await stdout_task
            metadata["stderr_tail"] = await stderr_task
            logger.info("✨ [DAEMON] EXTINGUISHED: %s status=%s", name, metadata["status"])
            if self._event_bus:
                await self._publish_daemon_event("core/cybernetics/daemon_finished", metadata)

    async def _publish_daemon_event(self, topic: str, metadata: dict[str, Any]) -> None:
        try:
            await self._event_bus.publish(topic, dict(metadata))
        except _OMNI_ERRORS as exc:
            _record_omni_degradation(
                exc,
                action="continued daemon lifecycle after event publication failed",
                severity="warning",
                extra={"topic": topic, "daemon": metadata.get("name", "")},
            )

    def check_daemons(self) -> list[dict[str, Any]]:
        """[DAEMON] Review all active background outcomes."""
        return list(self._daemons.values())

    def get_status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "active_cooldowns": list(self._cooldowns.keys()),
            "restricted_actions": [k for k, v in self._permissions.items() if not v],
            "daemons": {
                name: {
                    "status": daemon.get("status"),
                    "pid": daemon.get("pid"),
                    "returncode": daemon.get("returncode"),
                }
                for name, daemon in self._daemons.items()
            },
        }
