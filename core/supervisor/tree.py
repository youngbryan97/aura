"""Canonical multiprocessing actor transport beneath runtime desired state."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from core.bus.pipe_control import send_supervisor_stop
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.process_privilege import Privilege, ProcessRole
from core.runtime.resource_observation import get_resource_observer
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.subprocess_gateway import (
    AcceleratorCapability,
    PythonProcessSpec,
    get_subprocess_gateway,
)
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Supervisor")

_ACTOR_SHUTDOWN_GRACE_FLAG = declare(
    "AURA_ACTOR_SHUTDOWN_GRACE_S",
    kind=FlagKind.FLOAT,
    default=2.0,
    description="Cooperative stop grace for supervised actor processes",
    owner="core.supervisor.tree",
)
_ACTOR_FINAL_SWEEP_GRACE_FLAG = declare(
    "AURA_ACTOR_FINAL_SWEEP_GRACE_S",
    kind=FlagKind.FLOAT,
    default=0.75,
    description="Final join grace before terminating orphan actor children",
    owner="core.supervisor.tree",
)

SERVICE_NAME = "actor_supervision"


def _shutdown_requested() -> bool:
    return bool(is_shutdown_requested())


class _ProcessHandle(Protocol):
    pid: int | None
    exitcode: int | None
    name: str

    def start(self) -> None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def close(self) -> None: ...


@dataclass
class ActorSpec:
    name: str
    entry_point: Callable[..., Any] | None = None
    target: Callable[..., Any] | None = None
    args: tuple[Any, ...] = field(default_factory=tuple)
    restart_policy: str = "always"  # always, transient, never
    max_restarts: int = 3
    restart_delay: float = 1.0
    backoff_factor: float = 2.0
    window_seconds: int = 60
    health_timeout: float = 30.0
    grace_period: float = 45.0
    role: ProcessRole = ProcessRole.COORDINATOR
    requested_privileges: frozenset[Privilege] = field(default_factory=frozenset)
    accelerator_capability: AcceleratorCapability = AcceleratorCapability.NONE

    def __post_init__(self) -> None:
        if self.target and not self.entry_point:
            self.entry_point = self.target
        if not self.entry_point:
            raise ValueError("ActorSpec requires either entry_point or target")
        self.restart_policy = str(self.restart_policy or "").strip().lower()
        if self.restart_policy not in {"always", "transient", "never"}:
            raise ValueError(
                "restart_policy must be one of: always, transient, never"
            )
        if self.max_restarts < 0:
            raise ValueError("max_restarts must be non-negative")
        if self.restart_delay < 0:
            raise ValueError("restart_delay must be non-negative")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be at least 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not isinstance(self.role, ProcessRole):
            raise TypeError("role must be a ProcessRole")
        if any(
            not isinstance(privilege, Privilege)
            for privilege in self.requested_privileges
        ):
            raise TypeError("requested_privileges must contain Privilege values")
        if not isinstance(self.accelerator_capability, AcceleratorCapability):
            raise TypeError("accelerator_capability must be an AcceleratorCapability")
        if self.health_timeout <= 0:
            raise ValueError("health_timeout must be positive")
        if self.grace_period < 0:
            raise ValueError("grace_period must be non-negative")

class ActorHealthGate:
    """
    ZENITH LOCKDOWN: Health gating for actors.
    Provides grace periods and miss thresholds for heartbeats.
    """
    def __init__(self, grace_period: float = 15.0, timeout: float = 10.0) -> None:
        self.start_time = time.monotonic()
        self.last_heartbeat = time.monotonic()
        self.grace_period = grace_period
        self.timeout = timeout
        self.miss_count = 0
        self.max_misses = 3
        self._last_miss_window = -1

    def record_heartbeat(self) -> None:
        self.last_heartbeat = time.monotonic()
        self.miss_count = 0
        self._last_miss_window = -1

    def is_healthy(self) -> bool:
        now = time.monotonic()
        # Grace period for boot
        if now - self.start_time < self.grace_period:
            return True

        overdue = now - self.last_heartbeat - self.timeout
        if overdue <= 0:
            self.miss_count = 0
            self._last_miss_window = -1
            return True

        miss_window = int(overdue // self.timeout)
        if miss_window != self._last_miss_window:
            self.miss_count += 1
            self._last_miss_window = miss_window
        return self.miss_count <= self.max_misses
    
@dataclass
class ManagedActor:
    spec: ActorSpec
    process: _ProcessHandle | None = None
    pipe: Any | None = None
    restarts: int = 0
    consecutive_failures: int = 0
    last_restart: float = 0.0
    next_restart_time: float = 0.0
    is_circuit_broken: bool = False
    health_gate: ActorHealthGate | None = None
    monitor_health: bool = False
    desired_running: bool = False
    last_exit_code: int | None = None
    last_error: str = ""
    last_failure_at: float = 0.0

class SupervisionTree:
    """
    The 'Immune System' of Aura.
    Hierarchical supervisor for sovereign processes.
    """
    _active_lock: ClassVar[threading.Lock] = threading.Lock()
    _active_instance: ClassVar[SupervisionTree | None] = None

    def __init__(self) -> None:
        self._actors: dict[str, ManagedActor] = {}
        self._is_running = False
        self._shutting_down = False
        self._restart_callback: Callable[[str, Any], None] | None = None
        self._lock = threading.RLock()
        self._spawn_lock = threading.RLock()
        self._monitor_task: asyncio.Task[Any] | None = None

    def set_restart_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Set a callback for when an actor is restarted with a new pipe."""
        with self._lock:
            self._restart_callback = callback

    def add_actor(self, spec: ActorSpec) -> None:
        """Register a new actor spec."""
        with self._lock:
            existing = self._actors.get(spec.name)
            if existing is not None:
                same_contract = (
                    existing.spec.entry_point is spec.entry_point
                    and existing.spec.args == spec.args
                    and existing.spec.restart_policy == spec.restart_policy
                )
                if same_contract:
                    return
                if existing.process is not None and self._process_is_alive(existing.process):
                    raise RuntimeError(
                        f"cannot replace live supervised actor contract: {spec.name}"
                    )
                raise ValueError(
                    f"conflicting supervised actor registration: {spec.name}"
                )
            self._actors[spec.name] = ManagedActor(spec=spec)
        logger.info("🛡️ Actor Registered for Supervision: %s", spec.name)

    def is_actor_running(self, name: str) -> bool:
        with self._lock:
            actor = self._actors.get(name)
        return bool(actor and actor.process and actor.process.is_alive())

    def get_actor_pipe(self, name: str) -> Any | None:
        with self._lock:
            actor = self._actors.get(name)
        return actor.pipe if actor else None

    def _close_pipe(self, pipe: Any) -> None:
        if pipe is None:
            return
        if isinstance(pipe, tuple):
            for endpoint in pipe:
                try:
                    endpoint.close()
                except (RuntimeError, AttributeError, TypeError, ValueError):
                    pass  # no-op: intentional
            return
        try:
            pipe.close()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass  # no-op: intentional

    def _send_actor_stop(self, pipe: Any, name: str) -> bool:
        """Best-effort cooperative actor stop over the existing pipe transport."""
        try:
            return bool(send_supervisor_stop(pipe, name))
        except (BrokenPipeError, EOFError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("Cooperative actor stop send failed for %s: %s", name, exc)
            return False

    def _process_is_alive(self, process: _ProcessHandle) -> bool:
        if getattr(process, "exitcode", None) is not None:
            return False
        try:
            if not process.is_alive():
                return False
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass
        pid = getattr(process, "pid", None)
        if not pid:
            return True
        observed = get_resource_observer().process(pid)
        if observed is not None:
            return observed.status.lower() not in {"dead", "zombie"}
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def record_activity(self, name: str) -> None:
        """Mark an actor as alive without directly reading from its IPC pipe."""
        with self._lock:
            actor = self._actors.get(name)
            if not actor:
                return
            if actor.health_gate is None:
                actor.health_gate = ActorHealthGate(
                    grace_period=actor.spec.grace_period,
                    timeout=actor.spec.health_timeout,
                )
            actor.monitor_health = True
            actor.health_gate.record_heartbeat()

    def start_actor(self, name: str) -> Any | None:
        """Spin up one actor and fold launch failures into bounded recovery."""
        with self._spawn_lock:
            if self._shutting_down or _shutdown_requested():
                logger.info("🛑 Not starting Actor %s: runtime is shutting down.", name)
                return None
            with self._lock:
                actor = self._actors.get(name)
                if not actor:
                    raise ValueError(f"Unknown actor: {name}")

                actor.desired_running = True
                if actor.process and self._process_is_alive(actor.process):
                    logger.debug(
                        "🛡️ start_actor: %s is already alive (PID: %s). Returning existing pipe.",
                        name,
                        actor.process.pid,
                    )
                    return actor.pipe

            parent_pipe: tuple[Any, Any] | None = None
            child_pipe: tuple[Any, Any] | None = None
            proc: _ProcessHandle | None = None
            try:
                ctx = multiprocessing.get_context("spawn")
                parent_read, child_write = ctx.Pipe(duplex=False)
                child_read, parent_write = ctx.Pipe(duplex=False)
                parent_pipe = (parent_read, parent_write)
                child_pipe = (child_read, child_write)
                proc = get_subprocess_gateway().spawn_python_process(
                    PythonProcessSpec(
                        target=actor.spec.entry_point,
                        args=(*actor.spec.args, child_pipe),
                        source=f"actor_supervisor:{name}",
                        name=f"AuraActor:{name}",
                        role=actor.spec.role,
                        requested_privileges=actor.spec.requested_privileges,
                        accelerator_capability=actor.spec.accelerator_capability,
                        daemon=True,
                        start_method="spawn",
                    ),
                    context=ctx,
                )
                if self._shutting_down or _shutdown_requested():
                    logger.info(
                        "Actor %s crossed the shutdown boundary during spawn; terminating pid=%s",
                        name,
                        proc.pid,
                    )
                    if self._process_is_alive(proc):
                        proc.terminate()
                        proc.join(timeout=1.0)
                    if self._process_is_alive(proc):
                        proc.kill()
                        proc.join(timeout=1.0)
                    raise RuntimeError("runtime_shutdown_after_actor_start")
            except (
                AssertionError,
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                self._close_pipe(parent_pipe)
                self._close_pipe(child_pipe)
                if proc is not None:
                    try:
                        proc.close()
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass
                record_degradation(
                    "tree",
                    exc,
                    severity="error",
                    action="actor launch failed; scheduled bounded supervisor retry",
                    extra={"actor": name},
                )
                self._handle_failure(name, error=exc)
                self._publish_conditions()
                return None

            # The child endpoints must not remain open in the parent process;
            # otherwise EOF-based actor shutdown can remain permanently masked.
            self._close_pipe(child_pipe)
            if proc is None:
                raise RuntimeError(f"actor process construction returned no handle: {name}")
            with self._lock:
                actor.process = proc
                actor.pipe = parent_pipe
                actor.last_restart = time.time()
                actor.last_exit_code = None
                actor.is_circuit_broken = False
                actor.health_gate = ActorHealthGate(
                    grace_period=actor.spec.grace_period,
                    timeout=actor.spec.health_timeout,
                )
                actor.health_gate.record_heartbeat()

            logger.info("🚀 Actor Started: %s (PID: %s)", name, proc.pid)
            return parent_pipe

    def stop_actor(
        self,
        name: str,
        *,
        graceful_timeout: float = 0.0,
        terminate_timeout: float = 1.0,
        kill_timeout: float = 1.0,
        preserve_desired: bool = False,
    ) -> None:
        """Stop an actor and reap its process handle before shutdown returns.

        The caller can first give a cooperative bus-level ``stop`` handler time
        to exit. If the process is still alive, the supervisor escalates to
        terminate and finally kill, keeping the old bounded cleanup guarantee
        without creating hard-kill noise for actors that can stop cleanly.
        """
        with self._lock:
            actor = self._actors.get(name)
            if actor is not None and not preserve_desired:
                actor.desired_running = False
        if actor and actor.process:
            logger.info("🛑 Stopping Actor: %s", name)
            process = actor.process
            pipe_closed = False
            try:
                if not self._process_is_alive(process):
                    process.join(timeout=0.0)
                    logger.info("Actor %s already exited; reaped process handle.", name)
                    return

                if graceful_timeout > 0:
                    cooperative_sent = self._send_actor_stop(actor.pipe, name)
                    process.join(timeout=max(0.0, graceful_timeout))
                    if not self._process_is_alive(process):
                        logger.info(
                            "Actor %s exited cooperatively%s.",
                            name,
                            " after stop request" if cooperative_sent else "",
                        )
                        return

                    if actor.pipe is not None:
                        self._close_pipe(actor.pipe)
                        pipe_closed = True
                        with self._lock:
                            actor.pipe = None
                    process.join(timeout=min(0.25, max(0.0, graceful_timeout)))
                    if not self._process_is_alive(process):
                        logger.info("Actor %s exited cooperatively after pipe close.", name)
                        return

                process.terminate()
                process.join(timeout=max(0.0, terminate_timeout))
                if self._process_is_alive(process):
                    logger.warning(
                        "Actor %s did not exit after terminate within timeout; escalating to kill.",
                        name,
                    )
                    process.kill()
                    process.join(timeout=max(0.0, kill_timeout))
                if self._process_is_alive(process):
                    process.join(timeout=0.0)
                if self._process_is_alive(process) and process.exitcode is None:
                    logger.warning("Actor %s did not exit after kill within timeout.", name)
                elif process.exitcode is not None:
                    process.join(timeout=0.0)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('tree', e)
                logger.debug("Error stopping actor %s: %s", name, e)
            finally:
                if not pipe_closed:
                    self._close_pipe(actor.pipe)
                with self._lock:
                    actor.process = None
                    actor.pipe = None

    async def start(self) -> None:
        """Start one background monitor and restore desired actor processes."""
        with self._active_lock:
            active = self._active_instance
            if active is self and self._is_running:
                return
            if active is not None and active is not self:
                raise RuntimeError(
                    "another SupervisionTree already owns the actor monitor"
                )
            self.__class__._active_instance = self
            self._is_running = True
            self._shutting_down = False
        with self._lock:
            desired = [
                name
                for name, actor in self._actors.items()
                if actor.desired_running
                and (actor.process is None or not self._process_is_alive(actor.process))
            ]
        for name in desired:
            self.start_actor(name)
        monitor = self._monitor_loop()
        try:
            self._monitor_task = get_task_tracker().create_task(
                monitor,
                name="ActorSupervisionMonitor",
            )
        except (RuntimeError, AttributeError, TypeError, ValueError):
            monitor.close()
            self._is_running = False
            with self._active_lock:
                if self._active_instance is self:
                    self.__class__._active_instance = None
            raise
        self._register_with_control_plane()
        self._publish_conditions()
        logger.info("🛡️ Supervision Tree initialized with one managed monitor.")

    async def stop(self) -> None:
        """Stop the monitor and actors while preserving desired state for restart."""
        self._is_running = False
        self._shutting_down = True
        task, self._monitor_task = self._monitor_task, None
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, TimeoutError):
                pass
        try:
            await asyncio.to_thread(self.stop_all, preserve_desired=True)
            self._publish_conditions()
        finally:
            with self._active_lock:
                if self._active_instance is self:
                    self.__class__._active_instance = None

    async def _monitor_loop(self) -> None:
        logger.info("🛡️ Supervision Tree ACTIVE. Monitoring actors...")
        try:
            while self._is_running and not _shutdown_requested():
                self._poll_health()
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise
        finally:
            if _shutdown_requested():
                self._shutting_down = True
            self._is_running = False

    async def wait_forever(self, *, until: asyncio.Event | None = None) -> None:
        """Wait for an owner's lifetime or, for legacy callers, this monitor."""
        await self.start()
        task = self._monitor_task
        try:
            if until is not None:
                # A managed monitor may be replaced by the control plane.
                # Its cancellation must not become the desktop's lifetime.
                while not until.is_set() and not _shutdown_requested():
                    try:
                        await asyncio.wait_for(until.wait(), timeout=0.5)
                    except TimeoutError:
                        continue
            elif task is not None:
                await task
        except asyncio.CancelledError:
            raise
        finally:
            if _shutdown_requested():
                await self.stop()

    def _register_with_control_plane(self) -> None:
        try:
            from core.runtime.control_plane import (
                DesiredServiceSpec,
                WorkClass,
                get_runtime_control_plane,
            )
            from core.runtime.service_registry import register_runtime_service

            plane = get_runtime_control_plane()
            if not plane.has_service(SERVICE_NAME):
                plane.register_service(
                    DesiredServiceSpec(
                        name=SERVICE_NAME,
                        critical=True,
                        restart_limit=3,
                        restart_window_s=300.0,
                        backoff_initial_s=1.0,
                        backoff_max_s=30.0,
                        admission_class=WorkClass.SERVICE_START,
                        metadata={"domain": "multiprocessing_actors"},
                    ),
                    start=self.start,
                    stop=self.stop,
                    probe=self.is_alive,
                    adopt_running=True,
                )
            register_runtime_service(
                SERVICE_NAME,
                self,
                required=True,
                owner="core/supervisor/tree.py",
                registered_by="SupervisionTree.start",
                required_for="canonical actor process lifecycle and restart policy",
                failure_policy="fail-closed",
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "tree",
                exc,
                severity="error",
                action="actor supervision started but control-plane adoption failed",
            )

    def is_alive(self) -> bool:
        task = self._monitor_task
        return bool(self._is_running and task is not None and not task.done())

    def is_ready(self) -> bool:
        return self.is_alive()

    def run_forever(self) -> None:
        """Blocking compatibility entry point over the managed async monitor."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("run_forever cannot block an active event loop")

        async def run_managed() -> None:
            try:
                await self.wait_forever()
            finally:
                await self.stop()

        asyncio.run(run_managed())

    def _poll_health(self) -> None:
        """Check all actors and restart if needed."""
        if self._shutting_down or _shutdown_requested():
            self._shutting_down = True
            return
        now = time.time()
        with self._lock:
            actors = list(self._actors.items())

        for name, actor in actors:
            if actor.is_circuit_broken:
                continue

            if actor.process and not self._process_is_alive(actor.process):
                exit_code = actor.process.exitcode
                logger.warning("⚠️ Actor CRASHED: %s (Exit Code: %s)", name, exit_code)
                self._handle_failure(name, exit_code=exit_code)
            
            elif actor.process and actor.health_gate and actor.monitor_health:
                if not actor.health_gate.is_healthy():
                    logger.error("🚨 Actor STALLED (Liveness Failure): %s", name)
                    self.stop_actor(name, preserve_desired=True)
                    self._handle_failure(name, error="actor heartbeat timeout")
            
            elif actor.process is None and actor.next_restart_time > 0 and now >= actor.next_restart_time:
                logger.info("♻️ Restarting Actor %s after backoff...", name)
                actor.next_restart_time = 0 # Reset
                self._restart_actor(name)
        self._publish_conditions()

    def _handle_failure(
        self,
        name: str,
        *,
        exit_code: int | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        """Apply restart policy with circuit breaker and backoff."""
        with self._lock:
            actor = self._actors[name]

            # Mark process as gone
            self._close_pipe(actor.pipe)
            actor.process = None
            actor.pipe = None

            now = time.time()
            actor.last_exit_code = exit_code
            actor.last_failure_at = now
            if error is not None:
                actor.last_error = str(error)[:1000]
            elif exit_code not in (None, 0):
                actor.last_error = f"actor process exited with code {exit_code}"

            if self._shutting_down or _shutdown_requested():
                actor.next_restart_time = 0.0
                return
            policy = str(actor.spec.restart_policy or "always").strip().lower()
            if policy == "never" or (policy == "transient" and exit_code == 0):
                actor.desired_running = False
                actor.next_restart_time = 0.0
                return

            # 1. Update Failure Tracking
            if now - actor.last_restart < actor.spec.window_seconds:
                actor.consecutive_failures += 1
            else:
                actor.consecutive_failures = 1

            if actor.consecutive_failures > actor.spec.max_restarts:
                logger.error("🛑 CIRCUIT BROKEN: Actor %s failed too many times in window.", name)
                actor.is_circuit_broken = True
                actor.next_restart_time = 0.0
                return

            # 2. Calculate Exponential Backoff
            delay = actor.spec.restart_delay * (
                actor.spec.backoff_factor ** (actor.consecutive_failures - 1)
            )
            delay = min(delay, 60.0)

            actor.next_restart_time = now + delay
            attempt = actor.consecutive_failures
            max_restarts = actor.spec.max_restarts
        logger.info(
            "⏳ Scheduling Restart for %s (Attempt %s/%s) in %ss...",
            name,
            attempt,
            max_restarts,
            f"{delay:.1f}",
        )

    def _restart_actor(self, name: str) -> None:
        """Internal helper to start actor and trigger callback."""
        if self._shutting_down or _shutdown_requested():
            self._shutting_down = True
            return
        new_pipe = self.start_actor(name)
        with self._lock:
            if new_pipe and name in self._actors:
                self._actors[name].restarts += 1
            callback = self._restart_callback
        if callback and new_pipe:
            try:
                callback(name, new_pipe)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation(
                    "tree",
                    e,
                    severity="warning",
                    action="kept restarted actor running after restart callback failed",
                    extra={"actor": name},
                )
                logger.error("❌ Restart callback failed for %s: %s", name, e)

    def stop_all(self, *, preserve_desired: bool = False) -> None:
        """Kill everything."""
        self._shutting_down = True
        self._is_running = False
        graceful_timeout = float(_ACTOR_SHUTDOWN_GRACE_FLAG.value())
        graceful_timeout = min(10.0, max(0.0, graceful_timeout))
        final_sweep_grace = float(_ACTOR_FINAL_SWEEP_GRACE_FLAG.value())
        final_sweep_grace = min(2.0, max(0.0, final_sweep_grace))
        with self._lock:
            actor_names = list(self._actors.keys())
        for name in actor_names:
            self.stop_actor(
                name,
                graceful_timeout=graceful_timeout,
                preserve_desired=preserve_desired,
            )
        # Ensure all multiprocess children are reaped (ORPHAN-04)
        import multiprocessing
        for p in multiprocessing.active_children():
            if p.name.startswith("AuraActor:"):
                if not p.is_alive():
                    p.join(timeout=0.0)
                    logger.debug("Reaped already-exited actor child: %s", p.name)
                    continue
                if final_sweep_grace > 0.0:
                    p.join(timeout=final_sweep_grace)
                    if not p.is_alive():
                        logger.debug(
                            "Joined actor child during final supervisor grace: %s",
                            p.name,
                        )
                        continue
                logger.info("🧹 Stopping active actor child after supervisor shutdown: %s", p.name)
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=0.2)
        logger.info("🛡️ Supervision Tree Shutdown Complete.")

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            actors = {
                name: {
                    "desired_running": actor.desired_running,
                    "alive": bool(
                        actor.process is not None
                        and self._process_is_alive(actor.process)
                    ),
                    "pid": getattr(actor.process, "pid", None),
                    "exitcode": getattr(actor.process, "exitcode", None),
                    "restart_policy": actor.spec.restart_policy,
                    "restarts": actor.restarts,
                    "consecutive_failures": actor.consecutive_failures,
                    "last_restart": actor.last_restart,
                    "next_restart_time": actor.next_restart_time,
                    "circuit_open": actor.is_circuit_broken,
                    "last_exit_code": actor.last_exit_code,
                    "last_error": actor.last_error,
                    "last_failure_at": actor.last_failure_at,
                    "health_monitored": actor.monitor_health,
                    "last_heartbeat_monotonic": (
                        actor.health_gate.last_heartbeat
                        if actor.health_gate is not None
                        else 0.0
                    ),
                }
                for name, actor in sorted(self._actors.items())
            }
        return {
            "alive": self.is_alive(),
            "ready": self.is_ready(),
            "running": self._is_running,
            "shutting_down": self._shutting_down,
            "monitor_task_active": bool(
                self._monitor_task is not None and not self._monitor_task.done()
            ),
            "actors": actors,
            "summary": {
                "registered": len(actors),
                "desired_running": sum(
                    bool(actor["desired_running"]) for actor in actors.values()
                ),
                "alive": sum(bool(actor["alive"]) for actor in actors.values()),
                "open_circuits": sum(
                    bool(actor["circuit_open"]) for actor in actors.values()
                ),
            },
        }

    def _publish_conditions(self) -> None:
        try:
            from core.runtime.conditions import (
                ConditionType,
                get_component_conditions,
            )

            status = self.get_status()
            summary = status["summary"]
            open_circuits = int(summary["open_circuits"])
            progressing = int(summary["desired_running"]) > int(summary["alive"])
            conditions = get_component_conditions(SERVICE_NAME)
            conditions.set(
                ConditionType.READY,
                self.is_ready(),
                reason="MonitorRunning" if self.is_ready() else "MonitorStopped",
                message=f"actors_alive={summary['alive']}/{summary['desired_running']}",
            )
            conditions.set(
                ConditionType.PROGRESSING,
                progressing and not open_circuits,
                reason="ActorRestartPending" if progressing else "ActorsConverged",
                message=f"desired={summary['desired_running']} alive={summary['alive']}",
            )
            conditions.set(
                ConditionType.DEGRADED,
                open_circuits > 0,
                reason="ActorCircuitOpen" if open_circuits else "NoOpenCircuits",
                message=f"open_circuits={open_circuits}",
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            return

_tree_instance: SupervisionTree | None = None
_tree_instance_lock = threading.Lock()

def get_tree() -> SupervisionTree:
    global _tree_instance
    if _tree_instance is None:
        with _tree_instance_lock:
            if _tree_instance is None:
                _tree_instance = SupervisionTree()
    return _tree_instance


def reset_tree() -> None:
    """Reset the process supervisor singleton to a clean state."""
    global _tree_instance
    if _tree_instance is not None:
        if _tree_instance.is_alive():
            raise RuntimeError("stop the live actor supervisor before resetting it")
        _tree_instance.stop_all()
        with SupervisionTree._active_lock:
            if SupervisionTree._active_instance is _tree_instance:
                SupervisionTree._active_instance = None
    _tree_instance = None
