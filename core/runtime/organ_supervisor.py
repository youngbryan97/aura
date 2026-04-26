"""core/runtime/organ_supervisor.py

Multiprocess Organ Supervisor (Chromium-style)
================================================
Decouples Aura's "organs" — MLX worker, motor cortex, voice engine,
phi_core compute — into separate OS-level processes. The supervisor
restarts a crashed organ silently without bringing down the parent
process; one organ's segfault does not flinch the kernel.

The supervisor offers:

  * register_organ(name, cmd, *, restart_policy)
  * start_all() / stop_all()
  * health() — per-organ liveness, age, last restart, restart count
  * watchdog_loop() — restarts unhealthy organs subject to the
                      restart policy (max_restarts_per_window, window_s)
  * ipc_call(organ_name, payload, timeout) — request/response over a
    framed pipe to the organ's controller stub

Organs are launched with stdin/stdout closed; they communicate over a
unix domain socket created by the supervisor at boot in
``/tmp/aura-<pid>-<organ>.sock``. The protocol is JSON-lines with a
length prefix.

This module provides the *supervisor* and the *health/restart* layer.
The organ-side controller stub is a small entry-point that knows how to
open its socket, accept requests, and dispatch to the local handler;
each organ ships its own stub (e.g. ``core/brain/llm/mlx_controller.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("Aura.OrganSupervisor")


_SOCK_DIR = Path("/tmp")


@dataclass
class RestartPolicy:
    max_restarts: int = 5
    window_s: float = 60.0
    backoff_initial_s: float = 0.5
    backoff_factor: float = 2.0
    backoff_max_s: float = 30.0


@dataclass
class OrganRecord:
    name: str
    cmd: List[str]
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    proc: Optional[asyncio.subprocess.Process] = None
    started_at: float = 0.0
    last_restart_at: float = 0.0
    restart_count_window: List[float] = field(default_factory=list)
    sock_path: str = ""
    policy: RestartPolicy = field(default_factory=RestartPolicy)
    stop_requested: bool = False

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None


class OrganSupervisor:
    def __init__(self) -> None:
        self._organs: Dict[str, OrganRecord] = {}
        self._watchdog_task: Optional[asyncio.Task] = None
        self._running = False

    def register_organ(
        self,
        name: str,
        *,
        cmd: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        policy: Optional[RestartPolicy] = None,
    ) -> None:
        sock = _SOCK_DIR / f"aura-{os.getpid()}-{name}.sock"
        if sock.exists():
            try:
                sock.unlink()
            except Exception:
                pass
        record = OrganRecord(
            name=name,
            cmd=list(cmd),
            cwd=cwd,
            env=dict(env or {}),
            sock_path=str(sock),
            policy=policy or RestartPolicy(),
        )
        self._organs[name] = record

    async def start_all(self) -> None:
        for record in self._organs.values():
            await self._start_organ(record)
        self._running = True
        self._watchdog_task = asyncio.create_task(self._watchdog(), name="OrganSupervisorWatchdog")

    async def stop_all(self) -> None:
        self._running = False
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None
        for record in self._organs.values():
            record.stop_requested = True
            if record.is_alive():
                try:
                    record.proc.send_signal(signal.SIGTERM)  # type: ignore[union-attr]
                    try:
                        await asyncio.wait_for(record.proc.wait(), timeout=5.0)  # type: ignore[union-attr]
                    except asyncio.TimeoutError:
                        record.proc.kill()  # type: ignore[union-attr]
                except Exception as exc:
                    logger.debug("organ stop %s failed: %s", record.name, exc)

    async def _start_organ(self, record: OrganRecord) -> None:
        env = os.environ.copy()
        env.update(record.env)
        env["AURA_ORGAN_SOCK"] = record.sock_path
        env["AURA_ORGAN_NAME"] = record.name
        try:
            record.proc = await asyncio.create_subprocess_exec(
                *record.cmd,
                cwd=record.cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )
            record.started_at = time.time()
            logger.info("🩻 organ '%s' launched (pid=%s)", record.name, record.proc.pid)
        except Exception as exc:
            logger.warning("organ '%s' failed to launch: %s", record.name, exc)

    async def _watchdog(self) -> None:
        while self._running:
            for record in list(self._organs.values()):
                if record.stop_requested:
                    continue
                if not record.is_alive():
                    await self._restart(record)
            await asyncio.sleep(2.0)

    async def _restart(self, record: OrganRecord) -> None:
        now = time.time()
        # prune window
        record.restart_count_window = [t for t in record.restart_count_window if (now - t) <= record.policy.window_s]
        if len(record.restart_count_window) >= record.policy.max_restarts:
            logger.error(
                "💀 organ '%s' exceeded restart budget (%d in %ss); leaving down",
                record.name,
                record.policy.max_restarts,
                record.policy.window_s,
            )
            return
        # backoff
        n = len(record.restart_count_window)
        delay = min(record.policy.backoff_max_s, record.policy.backoff_initial_s * (record.policy.backoff_factor ** n))
        await asyncio.sleep(delay)
        record.restart_count_window.append(now)
        record.last_restart_at = now
        await self._start_organ(record)

    # ─── IPC ─────────────────────────────────────────────────────────────

    async def ipc_call(self, organ_name: str, payload: Dict[str, Any], *, timeout: float = 8.0) -> Dict[str, Any]:
        record = self._organs.get(organ_name)
        if record is None or not record.sock_path:
            raise KeyError(organ_name)
        body = json.dumps(payload).encode("utf-8")
        header = struct.pack(">I", len(body))
        async def _do():
            reader, writer = await asyncio.open_unix_connection(record.sock_path)
            writer.write(header + body)
            await writer.drain()
            n_bytes = await reader.readexactly(4)
            n = struct.unpack(">I", n_bytes)[0]
            data = await reader.readexactly(n)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return json.loads(data.decode("utf-8"))
        return await asyncio.wait_for(_do(), timeout=timeout)

    # ─── introspection ──────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        out = {}
        for name, r in self._organs.items():
            out[name] = {
                "alive": r.is_alive(),
                "pid": r.proc.pid if r.proc else None,
                "started_at": r.started_at,
                "restarts_in_window": len(r.restart_count_window),
                "policy": asdict(r.policy),
                "sock": r.sock_path,
            }
        return out


_SUPERVISOR: Optional[OrganSupervisor] = None


def get_supervisor() -> OrganSupervisor:
    global _SUPERVISOR
    if _SUPERVISOR is None:
        _SUPERVISOR = OrganSupervisor()
    return _SUPERVISOR


__all__ = ["OrganSupervisor", "OrganRecord", "RestartPolicy", "get_supervisor"]
