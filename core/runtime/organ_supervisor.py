"""Command-organ transport managed by the canonical runtime control plane.

``OrganSupervisor`` is retained as the public registration and IPC facade, but
it no longer owns a watchdog or restart policy. Each command organ becomes a
``RuntimeControlPlane`` desired-state service; this module owns only subprocess
launch/stop mechanics and bounded framed Unix-socket I/O.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import struct
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.control_plane import (
    DesiredServiceSpec,
    DesiredServiceState,
    ObservedServiceState,
    RuntimeControlPlane,
    WorkClass,
    get_runtime_control_plane,
)
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.flags import FlagKind, declare
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.OrganSupervisor")

_SOCK_DIR = Path("/tmp")
_VALID_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_IPC_MAX_BYTES_FLAG = declare(
    "AURA_ORGAN_IPC_MAX_BYTES",
    kind=FlagKind.INT,
    default=16 * 1024 * 1024,
    description="Maximum framed request or response size for command-organ IPC",
    owner="core.runtime.organ_supervisor",
)


@dataclass(frozen=True)
class RestartPolicy:
    max_restarts: int = 5
    window_s: float = 60.0
    backoff_initial_s: float = 0.5
    backoff_factor: float = 2.0
    backoff_max_s: float = 30.0

    def __post_init__(self) -> None:
        if self.max_restarts < 0:
            raise ValueError("max_restarts must be non-negative")
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if self.backoff_initial_s < 0 or self.backoff_max_s < 0:
            raise ValueError("restart backoff values must be non-negative")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be at least 1")


@dataclass
class OrganRecord:
    name: str
    cmd: list[str]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    proc: asyncio.subprocess.Process | None = None
    started_at: float = 0.0
    last_stopped_at: float = 0.0
    sock_path: str = ""
    service_name: str = ""
    policy: RestartPolicy = field(default_factory=RestartPolicy)
    critical: bool = False
    stop_requested: bool = False

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None


class OrganSupervisor:
    """Register command organs as desired-state services and provide IPC."""

    def __init__(self, *, control_plane: RuntimeControlPlane | None = None) -> None:
        self._organs: dict[str, OrganRecord] = {}
        self._control_plane = control_plane or get_runtime_control_plane()
        self._last_report: dict[str, Any] = {}

    @staticmethod
    def _service_name(name: str) -> str:
        return f"organ:{name}"

    def register_organ(
        self,
        name: str,
        *,
        cmd: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        policy: RestartPolicy | None = None,
        critical: bool = False,
    ) -> None:
        normalized = str(name or "").strip()
        if not _VALID_NAME.fullmatch(normalized):
            raise ValueError(
                "organ name must be 1-64 characters of letters, digits, dot, dash, or underscore"
            )
        command = [str(part) for part in cmd if str(part)]
        if not command:
            raise ValueError("organ command must be non-empty")
        if normalized in self._organs:
            raise ValueError(f"organ already registered: {normalized}")

        sock = _SOCK_DIR / f"aura-{os.getpid()}-{normalized}.sock"
        if sock.exists() or sock.is_symlink():
            try:
                with local_internal_governed_scope(
                    "runtime.organ_supervisor.stale_socket",
                    domain="file_write",
                ):
                    get_file_write_gateway().delete_file(
                        sock,
                        source="runtime.organ_supervisor.stale_socket",
                    )
            except OSError as exc:
                raise RuntimeError(
                    f"cannot clear stale organ socket for {normalized}: {sock}"
                ) from exc
        effective_policy = policy or RestartPolicy()
        service_name = self._service_name(normalized)
        record = OrganRecord(
            name=normalized,
            cmd=command,
            cwd=str(cwd) if cwd is not None else None,
            env={str(key): str(value) for key, value in dict(env or {}).items()},
            sock_path=str(sock),
            service_name=service_name,
            policy=effective_policy,
            critical=bool(critical),
        )
        self._organs[normalized] = record

        async def start_registered_organ() -> None:
            await self._start_organ(record)

        async def stop_registered_organ() -> None:
            await self._stop_organ(record)

        self._control_plane.register_service(
            DesiredServiceSpec(
                name=service_name,
                critical=record.critical,
                desired_state=DesiredServiceState.STOPPED,
                start_timeout_s=15.0,
                stop_timeout_s=10.0,
                restart_limit=effective_policy.max_restarts,
                restart_window_s=effective_policy.window_s,
                backoff_initial_s=effective_policy.backoff_initial_s,
                backoff_factor=effective_policy.backoff_factor,
                backoff_max_s=effective_policy.backoff_max_s,
                admission_class=WorkClass.SERVICE_START,
                metadata={"domain": "command_organ", "organ": normalized},
            ),
            start=start_registered_organ,
            stop=stop_registered_organ,
            probe=record.is_alive,
        )

    async def start_all(self) -> dict[str, Any]:
        for record in self._organs.values():
            record.stop_requested = False
            self._control_plane.set_desired_state(
                record.service_name,
                DesiredServiceState.RUNNING,
            )
        report = await self._control_plane.reconcile_once()
        self._last_report = report
        critical_failures = [
            record.name
            for record in self._organs.values()
            if record.critical
            and report.get("services", {})
            .get(record.service_name, {})
            .get("observed_state")
            != ObservedServiceState.READY.value
        ]
        if critical_failures:
            raise RuntimeError(
                "critical command organs failed desired-state convergence: "
                + ",".join(sorted(critical_failures))
            )
        return report

    async def stop_all(self) -> dict[str, Any]:
        for record in self._organs.values():
            record.stop_requested = True
            self._control_plane.set_desired_state(
                record.service_name,
                DesiredServiceState.STOPPED,
            )
        report = await self._control_plane.reconcile_once()
        self._last_report = report
        return report

    async def start(self) -> None:
        await self.start_all()

    async def stop(self) -> None:
        await self.stop_all()

    async def _start_organ(self, record: OrganRecord) -> None:
        if record.is_alive():
            return
        if is_shutdown_requested():
            raise RuntimeError(f"runtime shutdown blocks organ start: {record.name}")
        child_env = os.environ.copy()
        child_env.update(record.env)
        child_env["AURA_ORGAN_SOCK"] = record.sock_path
        child_env["AURA_ORGAN_NAME"] = record.name
        record.stop_requested = False
        try:
            record.proc = await get_subprocess_gateway().spawn_async(
                record.cmd,
                cwd=record.cwd,
                env=child_env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
                source="environment_action:organ_supervisor.launch",
                accelerator_capability="auto",
            )
            record.started_at = time.time()
            logger.info(
                "Organ %s launched under desired state (pid=%s)",
                record.name,
                record.proc.pid,
            )
        except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
            record.proc = None
            record_degradation(
                "organ_supervisor",
                exc,
                severity="warning",
                action="command-organ launch failed; control plane owns bounded retry",
                extra={"organ": record.name, "cmd": record.cmd[:3]},
            )
            raise

    async def _stop_organ(self, record: OrganRecord) -> None:
        record.stop_requested = True
        process = record.proc
        if process is None:
            return
        try:
            if process.returncode is None:
                process.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except TimeoutError:
                    process.kill()
                    await asyncio.wait_for(process.wait(), timeout=2.0)
        except (ProcessLookupError, RuntimeError, AttributeError, OSError) as exc:
            record_degradation(
                "organ_supervisor",
                exc,
                severity="warning",
                action="continued command-organ stop after process teardown error",
                extra={"organ": record.name},
            )
        finally:
            record.proc = None
            record.last_stopped_at = time.time()

    async def ipc_call(
        self,
        organ_name: str,
        payload: dict[str, Any],
        *,
        timeout_s: float = 8.0,
    ) -> dict[str, Any]:
        record = self._organs.get(str(organ_name))
        if record is None:
            raise KeyError(organ_name)
        if not record.is_alive():
            raise RuntimeError(f"organ is not ready: {organ_name}")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        max_bytes = max(1024, int(_IPC_MAX_BYTES_FLAG.value()))
        if len(body) > max_bytes:
            raise ValueError(
                f"organ request exceeds frame limit: {len(body)} > {max_bytes}"
            )
        header = struct.pack(">I", len(body))

        async def _do() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(record.sock_path)
            try:
                writer.write(header + body)
                await writer.drain()
                response_header = await reader.readexactly(4)
                response_size = struct.unpack(">I", response_header)[0]
                if response_size > max_bytes:
                    raise ValueError(
                        f"organ response exceeds frame limit: {response_size} > {max_bytes}"
                    )
                data = await reader.readexactly(response_size)
            finally:
                writer.close()
                try:
                    await asyncio.wait_for(
                        writer.wait_closed(),
                        timeout=min(1.0, float(timeout_s)),
                    )
                except (RuntimeError, TimeoutError, AttributeError) as exc:
                    # In a finally: this must not replace whatever is already
                    # on its way up, but a socket that will not close is worth
                    # saying out loud.
                    logger.debug(
                        "organ socket did not close cleanly (%s: %s)",
                        type(exc).__name__,
                        exc,
                    )
            decoded = json.loads(data.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("organ response must be a JSON object")
            return decoded

        return await asyncio.wait_for(_do(), timeout=float(timeout_s))

    def health(self) -> dict[str, Any]:
        plane_status = self._control_plane.service_status()
        return {
            name: {
                "alive": record.is_alive(),
                "pid": record.proc.pid if record.proc else None,
                "started_at": record.started_at,
                "last_stopped_at": record.last_stopped_at,
                "critical": record.critical,
                "desired_state": plane_status.get(record.service_name, {}).get(
                    "desired_state",
                    DesiredServiceState.STOPPED.value,
                ),
                "observed_state": plane_status.get(record.service_name, {}).get(
                    "observed_state",
                    ObservedServiceState.UNKNOWN.value,
                ),
                "reason": plane_status.get(record.service_name, {}).get("reason", ""),
                "restart_attempts_in_window": len(
                    plane_status.get(record.service_name, {}).get("restart_times", [])
                ),
                "next_retry_at": plane_status.get(record.service_name, {}).get(
                    "next_retry_at",
                    0.0,
                ),
                "policy": asdict(record.policy),
                "sock": record.sock_path,
            }
            for name, record in sorted(self._organs.items())
        }

    def is_alive(self) -> bool:
        return self._control_plane.is_alive()

    def is_ready(self) -> bool:
        return all(
            status["desired_state"] == DesiredServiceState.STOPPED.value
            or status["observed_state"] == ObservedServiceState.READY.value
            for status in self.health().values()
        )

    def get_status(self) -> dict[str, Any]:
        organs = self.health()
        return {
            "alive": self.is_alive(),
            "ready": self.is_ready(),
            "organs": organs,
            "summary": {
                "registered": len(organs),
                "running": sum(bool(status["alive"]) for status in organs.values()),
                "open_circuits": sum(
                    status["observed_state"] == ObservedServiceState.CIRCUIT_OPEN.value
                    for status in organs.values()
                ),
            },
            "last_report_digest": (
                hashlib.sha256(
                    json.dumps(self._last_report, sort_keys=True, default=str).encode(
                        "utf-8"
                    )
                ).hexdigest()
                if self._last_report
                else ""
            ),
        }


_SUPERVISOR: OrganSupervisor | None = None
_SUPERVISOR_LOCK = threading.Lock()


def get_supervisor() -> OrganSupervisor:
    global _SUPERVISOR
    if _SUPERVISOR is None:
        with _SUPERVISOR_LOCK:
            if _SUPERVISOR is None:
                _SUPERVISOR = OrganSupervisor()
    return _SUPERVISOR


def reset_supervisor() -> None:
    global _SUPERVISOR
    with _SUPERVISOR_LOCK:
        _SUPERVISOR = None


__all__ = [
    "OrganRecord",
    "OrganSupervisor",
    "RestartPolicy",
    "get_supervisor",
    "reset_supervisor",
]
