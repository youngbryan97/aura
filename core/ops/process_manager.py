"""Enterprise Process Management with Health Monitoring and Graceful Shutdown.

Features:
1. Process lifecycle management
2. Automatic health checks and restart with exponential backoff + jitter
3. Resource monitoring and limits
4. Graceful shutdown with timeouts
5. Process isolation and sandboxing
6. Comprehensive metrics and logging
7. PERMANENTLY_FAILED terminal state with incident tracking
8. Incident manager integration for structured failure reporting
"""

import asyncio
import atexit
import logging
import multiprocessing as mp
import os
import random
import resource  # For Unix resource limits
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.process_privilege import Privilege, ProcessRole
from core.runtime.resource_observation import get_resource_observer
from core.runtime.shutdown_coordinator import (
    is_shutdown_requested,
    record_shutdown_admission_event,
    request_shutdown,
    shutdown_request_snapshot,
)
from core.runtime.shutdown_execution import run_sync_shutdown_callable
from core.runtime.subprocess_gateway import (
    AcceleratorCapability,
    PythonProcessSpec,
    get_subprocess_gateway,
)
from core.utils.concurrency import cancel_and_join
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Kernel.ProcessManager")

_PROCESS_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)


def _record_process_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    process_name: str = "",
    severity: Severity = "warning",
) -> None:
    extra = {"stage": stage}
    if process_name:
        extra["process_name"] = process_name
    try:
        record_degradation(
            "process_manager",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=extra,
        )
    except TypeError:
        record_degradation(
            "process_manager",
            error,
            severity=severity,
            action=action,
        )


def _managed_process_entrypoint(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    memory_limit: int | None,
    priority: int,
) -> None:
    """Pickle-safe child entrypoint for multiprocessing spawn runtimes."""

    process_name = mp.current_process().name
    try:
        if memory_limit and os.name == "posix":
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        if priority and os.name == "posix":
            # Priority belongs to the child. Applying os.nice() before
            # Process.start() permanently changed Aura's parent process.
            os.nice(priority)
        logger.info("Process %s executing target function", process_name)
        target(*args, **kwargs)
    except KeyboardInterrupt:
        logger.info("Process %s interrupted gracefully", process_name)
    except _PROCESS_RECOVERABLE_ERRORS as exc:
        _record_process_degradation(
            exc,
            action="re-raised supervised child crash after recording process failure",
            stage="managed_process.wrapper",
            process_name=process_name,
            severity="degraded",
        )
        logger.error("Process %s crashed: %s", process_name, exc, exc_info=True)
        raise
    finally:
        logger.info("Process %s exiting", process_name)


class ProcessState(Enum):
    """Process lifecycle states."""

    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    RESTARTING = "restarting"
    PERMANENTLY_FAILED = "permanently_failed"  # Terminal: exceeded all retries


@dataclass
class ProcessConfig:
    """Process configuration."""

    name: str
    target: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    daemon: bool = False
    max_restarts: int = 3
    restart_window: int = 300  # seconds
    health_check_interval: int = 30  # seconds
    startup_timeout: int = 30  # seconds
    shutdown_timeout: int = 10  # seconds
    cpu_limit: float | None = None  # percentage
    memory_limit: int | None = None  # bytes
    priority: int = 0  # Process priority (0 = normal)
    role: ProcessRole = ProcessRole.COORDINATOR
    requested_privileges: frozenset[Privilege] = field(default_factory=frozenset)
    accelerator_capability: AcceleratorCapability = AcceleratorCapability.NONE
    start_method: str = "spawn"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Process name must be non-empty string")

        if self.max_restarts < 0:
            raise ValueError("max_restarts cannot be negative")

        if self.restart_window <= 0:
            raise ValueError("restart_window must be positive")

        if self.cpu_limit is not None and (self.cpu_limit < 0 or self.cpu_limit > 100):
            raise ValueError("cpu_limit must be between 0 and 100")

        if self.memory_limit is not None and self.memory_limit <= 0:
            raise ValueError("memory_limit must be positive")

        if not isinstance(self.role, ProcessRole):
            raise TypeError("role must be a ProcessRole")
        if any(
            not isinstance(privilege, Privilege)
            for privilege in self.requested_privileges
        ):
            raise TypeError("requested_privileges must contain Privilege values")

        if not isinstance(self.accelerator_capability, AcceleratorCapability):
            raise TypeError("accelerator_capability must be an AcceleratorCapability")
        if self.start_method not in {"spawn", "forkserver"}:
            raise ValueError("start_method must be spawn or forkserver")


@dataclass
class ProcessStats:
    """Process statistics."""

    start_time: float
    restarts: int = 0
    total_uptime: float = 0.0
    cpu_usage: list[float] = field(default_factory=list)
    memory_usage: list[int] = field(default_factory=list)
    last_health_check: float | None = None
    restart_timestamps: list[float] = field(default_factory=list)
    last_backoff_s: float = 0.0
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""


class ManagedProcess:
    """Managed process with supervision."""

    def __init__(self, config: ProcessConfig):
        self.config = config
        self.process: Any | None = None
        self.state = ProcessState.INITIALIZING
        self.stats = ProcessStats(start_time=time.time())
        self.last_restart_attempt: float | None = None
        self._lock = threading.RLock()
        self._health_check_thread: threading.Thread | None = None
        self._health_check_task: asyncio.Task[Any] | None = None
        self._health_check_loop: asyncio.AbstractEventLoop | None = None
        self._stop_health_check = threading.Event()

    async def start(self) -> bool:
        """Start the process."""
        if is_shutdown_requested():
            record_shutdown_admission_event(
                f"process_manager.start:{self.config.name}",
                resource_kind="multiprocessing",
                outcome="suppressed",
                detail="pre_factory",
            )
            logger.info("Process %s start refused: runtime shutdown requested", self.config.name)
            with self._lock:
                self.state = ProcessState.STOPPED
            return False

        with self._lock:
            if self.state in [ProcessState.STARTING, ProcessState.RUNNING]:
                logger.warning("Process %s already running", self.config.name)
                return False

            logger.info("Starting process: %s", self.config.name)
            self.state = ProcessState.STARTING

        try:
            source = f"process_manager:{self.config.name}"
            process = get_subprocess_gateway().spawn_python_process(
                PythonProcessSpec(
                    target=_managed_process_entrypoint,
                    args=(
                        self.config.target,
                        self.config.args,
                        self.config.kwargs,
                        self.config.memory_limit,
                        self.config.priority,
                    ),
                    name=self.config.name,
                    daemon=self.config.daemon,
                    source=source,
                    role=self.config.role,
                    requested_privileges=self.config.requested_privileges,
                    accelerator_capability=self.config.accelerator_capability,
                    start_method=self.config.start_method,
                ),
            )
            with self._lock:
                self.process = process

            if is_shutdown_requested():
                record_shutdown_admission_event(
                    f"process_manager.start:{self.config.name}",
                    resource_kind="multiprocessing",
                    outcome="crossed",
                    detail=f"pid={process.pid}",
                )
                logger.info(
                    "Process %s crossed the shutdown boundary during spawn; stopping it",
                    self.config.name,
                )
                reaped = await self.stop_async(force=True, timeout_s=1.0)
                record_shutdown_admission_event(
                    f"process_manager.start:{self.config.name}",
                    resource_kind="multiprocessing",
                    outcome="reaped" if reaped else "survived",
                    detail=f"pid={process.pid}",
                )
                return False

            start_time = time.monotonic()
            while time.monotonic() - start_time < self.config.startup_timeout:
                if is_shutdown_requested():
                    await self.stop_async(force=True, timeout_s=1.0)
                    return False
                if process.is_alive():
                    with self._lock:
                        shutdown_after_start = is_shutdown_requested()
                        if not shutdown_after_start:
                            self.state = ProcessState.RUNNING
                            self.stats.start_time = time.time()
                    if shutdown_after_start:
                        await self.stop_async(force=True, timeout_s=1.0)
                        return False
                    logger.info(
                        "Process %s started (PID: %s)", self.config.name, process.pid
                    )
                    await self._start_health_monitoring()
                    return True

                exitcode = getattr(process, "exitcode", None)
                if exitcode is not None:
                    with self._lock:
                        self.state = ProcessState.FAILED
                    logger.error(
                        "Process %s exited during startup (exitcode=%s)",
                        self.config.name,
                        exitcode,
                    )
                    return False
                await asyncio.sleep(0.1)

            with self._lock:
                self.state = ProcessState.FAILED
            logger.error("Process %s failed to start within timeout", self.config.name)
            return False

        except _PROCESS_RECOVERABLE_ERRORS as exc:
            if is_shutdown_requested():
                with self._lock:
                    self.state = ProcessState.STOPPED
                logger.info(
                    "Process %s start cancelled at shutdown boundary: %s",
                    self.config.name,
                    exc,
                )
                return False
            _record_process_degradation(
                exc,
                action="marked process failed after startup exception",
                stage="managed_process.start",
                process_name=self.config.name,
                severity="degraded",
            )
            with self._lock:
                self.state = ProcessState.FAILED
            logger.error(
                "Failed to start process %s: %s", self.config.name, exc, exc_info=True
            )
            return False

    def _process_wrapper(
        self,
        target: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Backward-compatible wrapper around the pickle-safe child entrypoint."""

        _managed_process_entrypoint(
            target,
            args,
            kwargs,
            self.config.memory_limit,
            self.config.priority,
        )

    def _request_health_monitor_stop(self) -> None:
        self._stop_health_check.set()
        task = self._health_check_task
        loop = self._health_check_loop
        if task is None or task.done():
            return
        try:
            running_loop = asyncio.get_running_loop()
        # not a failure: the question was whether a loop is running here, and it is not.
        except RuntimeError:
            running_loop = None
        if loop is not None and loop.is_running() and running_loop is not loop:
            loop.call_soon_threadsafe(task.cancel)
        else:
            task.cancel()

    def stop(self, force: bool = False, *, timeout_s: float | None = None) -> bool:
        """Stop the process from synchronous shutdown paths."""

        self._request_health_monitor_stop()
        return self._stop_process_blocking(force=force, timeout_s=timeout_s)

    async def stop_async(
        self,
        force: bool = False,
        *,
        timeout_s: float | None = None,
    ) -> bool:
        """Stop health ownership and the child without blocking the event loop."""

        await self._stop_health_monitoring()
        return bool(await run_sync_shutdown_callable(
            lambda: self._stop_process_blocking(
                force=force,
                timeout_s=timeout_s,
            ),
            timeout_s=max(
                0.1,
                float(
                    timeout_s
                    if timeout_s is not None
                    else self.config.shutdown_timeout + 2.0
                ),
            ),
            name=f"managed-process:{self.config.name}",
        ))

    def _stop_process_blocking(
        self,
        *,
        force: bool,
        timeout_s: float | None,
    ) -> bool:
        with self._lock:
            process = self.process
            prior_state = self.state
            if not process or not process.is_alive():
                self.state = ProcessState.STOPPED
                return True
            if self.state != ProcessState.STOPPING:
                logger.info("Stopping process: %s", self.config.name)
                self.state = ProcessState.STOPPING

        total_budget = None if timeout_s is None else max(0.0, float(timeout_s))
        stop_deadline = None if total_budget is None else time.monotonic() + total_budget
        graceful_timeout = float(self.config.shutdown_timeout)
        if total_budget is not None:
            # Preserve part of the caller's budget for forced reap instead of
            # spending the entire deadline on SIGTERM and overrunning on kill.
            graceful_timeout = min(graceful_timeout, total_budget * 0.8)

        try:
            # Signal the retained handle. Looking the child up by PID creates a
            # reuse race: after a fast exit the same integer can identify an
            # unrelated process by the time psutil resolves it.
            process.terminate()

            process.join(timeout=graceful_timeout)
            if process.is_alive():
                if not force:
                    with self._lock:
                        self.state = ProcessState.FAILED
                    logger.error("Process %s did not stop gracefully", self.config.name)
                    return False
                logger.warning("Process %s not responding, forcing kill", self.config.name)
                process.kill()
                force_join_timeout = 2.0
                if stop_deadline is not None:
                    force_join_timeout = max(0.0, stop_deadline - time.monotonic())
                process.join(timeout=force_join_timeout)

            stopped = not process.is_alive()
            with self._lock:
                self.state = ProcessState.STOPPED if stopped else ProcessState.FAILED
                if stopped and prior_state in {
                    ProcessState.RUNNING,
                    ProcessState.RESTARTING,
                    ProcessState.STARTING,
                }:
                    self.stats.total_uptime += max(0.0, time.time() - self.stats.start_time)
            if stopped:
                logger.info("Process %s stopped", self.config.name)
            else:
                logger.error("Process %s survived forced shutdown", self.config.name)
            return stopped
        except _PROCESS_RECOVERABLE_ERRORS as exc:
            _record_process_degradation(
                exc,
                action="marked process failed after stop operation failed",
                stage="managed_process.stop",
                process_name=self.config.name,
                severity="degraded",
            )
            logger.error("Error stopping process %s: %s", self.config.name, exc)
            with self._lock:
                self.state = ProcessState.FAILED
            return False

    def _compute_backoff(self) -> float:
        """Exponential backoff with jitter for restart delays.

        Uses capped exponential: min(base * 2^attempts, max_backoff) + jitter
        Jitter is ±25% to prevent thundering herd.
        """
        base_delay = 2.0  # seconds
        max_backoff = 120.0  # 2 minutes cap
        attempt = self.stats.restarts

        backoff = min(base_delay * (2**attempt), max_backoff)
        jitter = backoff * 0.25 * (2 * random.random() - 1)  # ±25%
        final = max(0.5, backoff + jitter)
        self.stats.last_backoff_s = final
        return float(final)

    async def restart(self) -> bool:
        """Restart the process with exponential backoff."""
        if is_shutdown_requested():
            logger.info("Process %s restart refused: runtime shutdown requested", self.config.name)
            return False

        with self._lock:
            # Check restart limits
            now = time.time()
            if self.last_restart_attempt:
                time_since_last_restart = now - self.last_restart_attempt

                # Reset counter if outside window
                if time_since_last_restart > self.config.restart_window:
                    self.stats.restarts = 0
                elif self.stats.restarts >= self.config.max_restarts:
                    logger.error(
                        "Process %s exceeded max restarts (%s) in %ss — PERMANENTLY FAILED",
                        self.config.name,
                        self.config.max_restarts,
                        self.config.restart_window,
                    )
                    self.state = ProcessState.PERMANENTLY_FAILED
                    # Report to incident manager
                    try:
                        from core.resilience.incident_manager import (
                            IncidentSeverity,
                            get_incident_manager,
                        )

                        get_incident_manager().report(
                            category=f"process_permanently_failed:{self.config.name}",
                            description=(
                                f"Process '{self.config.name}' exceeded {self.config.max_restarts} "
                                f"restarts in {self.config.restart_window}s window. "
                                f"Marked PERMANENTLY_FAILED. Manual intervention required."
                            ),
                            severity=IncidentSeverity.CRITICAL,
                            root_cause_hint="repeated_crash_loop",
                            mitigation_taken="process_disabled",
                            metadata={
                                "restarts": self.stats.restarts,
                                "window_s": self.config.restart_window,
                                "restart_timestamps": self.stats.restart_timestamps[-5:],
                            },
                        )
                    except _PROCESS_RECOVERABLE_ERRORS as exc:
                        _record_process_degradation(
                            exc,
                            action="kept terminal process state after incident report failed",
                            stage="managed_process.restart.incident_report",
                            process_name=self.config.name,
                            severity="degraded",
                        )
                        logger.debug(
                            "Incident report failed for permanent process failure %s: %s",
                            self.config.name,
                            exc,
                        )
                    # Report to metrics
                    try:
                        from core.observability.metrics import get_metrics

                        get_metrics().record_process_restart(self.config.name)
                    except _PROCESS_RECOVERABLE_ERRORS as exc:
                        _record_process_degradation(
                            exc,
                            action="kept terminal process state after restart metric failed",
                            stage="managed_process.restart.metrics",
                            process_name=self.config.name,
                        )
                        logger.debug(
                            "Process restart metric failed for %s: %s", self.config.name, exc
                        )
                    return False

            # Apply exponential backoff delay
            backoff_s = self._compute_backoff()
            if backoff_s > 1.0:
                logger.info(
                    "Process %s restart backoff: %.1fs before attempt %d",
                    self.config.name,
                    backoff_s,
                    self.stats.restarts + 1,
                )

            should_stop = bool(self.process and self.process.is_alive())

        if should_stop and not await self.stop_async(force=False):
            return False

        # Backoff is interruptible so a stop request never waits behind a
        # restart timer and then resurrects the child.
        backoff_deadline = time.monotonic() + backoff_s
        while time.monotonic() < backoff_deadline:
            if is_shutdown_requested():
                logger.info(
                    "Process %s restart cancelled during shutdown", self.config.name
                )
                return False
            remaining_backoff = max(0.0, backoff_deadline - time.monotonic())
            if remaining_backoff:
                await asyncio.sleep(min(0.25, remaining_backoff))

        with self._lock:
            if is_shutdown_requested():
                return False
            self.last_restart_attempt = time.time()
            self.stats.restarts += 1
            self.stats.restart_timestamps.append(time.time())
            # Keep only last 20 timestamps
            if len(self.stats.restart_timestamps) > 20:
                self.stats.restart_timestamps = self.stats.restart_timestamps[-20:]
            self.state = ProcessState.RESTARTING

            logger.info(
                "Restarting process %s (attempt %s, backoff=%.1fs)",
                self.config.name,
                self.stats.restarts,
                backoff_s,
            )
        return await self.start()

    async def _start_health_monitoring(self) -> None:
        """Start health monitoring task."""
        if is_shutdown_requested():
            return
        if self._health_check_task and not self._health_check_task.done():
            return

        self._stop_health_check.clear()
        self._health_check_loop = asyncio.get_running_loop()
        self._health_check_task = get_task_tracker().create_task(
            self._health_monitor_loop(),
            name=f"process_manager.{self.config.name}.health_monitor",
        )
        logger.debug("Started health monitoring for %s", self.config.name)

    async def _stop_health_monitoring(self) -> None:
        """Stop health monitoring task."""
        self._stop_health_check.set()
        task = self._health_check_task
        if task and task is not asyncio.current_task() and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5)
            except TimeoutError:
                logger.warning(
                    "Health monitoring task for %s did not stop gracefully, cancelling.",
                    self.config.name,
                )
                await cancel_and_join(task, owner="core.ops.process_manager")
            except asyncio.CancelledError:
                logger.debug("Exception caught during execution", exc_info=True)
        self._health_check_task = None
        self._health_check_loop = None

    async def _health_monitor_loop(self) -> None:
        """Continuous health monitoring loop."""
        while not self._stop_health_check.is_set() and not is_shutdown_requested():
            try:
                await asyncio.to_thread(self._check_health)
            except _PROCESS_RECOVERABLE_ERRORS as e:
                _record_process_degradation(
                    e,
                    action="continued health monitor loop after health check raised",
                    stage="managed_process.health_monitor",
                    process_name=self.config.name,
                )
                logger.error("Health check failed for %s: %s", self.config.name, e, exc_info=True)

            # Wait for next check or stop signal
            try:
                await asyncio.to_thread(
                    self._stop_health_check.wait,
                    self.config.health_check_interval,
                )
            except asyncio.CancelledError:
                break  # Task was cancelled, exit loop

    def _check_health(self) -> None:
        """Perform health check."""
        if not self.process or not self.process.is_alive():
            logger.warning("Process %s is not alive", self.config.name)
            return

        try:
            observer = get_resource_observer()
            observed = observer.process(self.process.pid)
            if observed is None:
                logger.warning("Process %s PID %s not found", self.config.name, self.process.pid)
                return
            self.stats.observation_source = observed.provenance.source.value
            self.stats.observation_scenario_id = observed.provenance.scenario_id

            # Check CPU usage
            cpu_percent = observed.cpu_percent
            self.stats.cpu_usage.append(cpu_percent)
            if len(self.stats.cpu_usage) > 100:  # Keep last 100 samples
                self.stats.cpu_usage.pop(0)

            # Check memory usage
            memory_rss = observed.rss_bytes
            self.stats.memory_usage.append(memory_rss)
            if len(self.stats.memory_usage) > 100:
                self.stats.memory_usage.pop(0)

            # Check against limits
            if self.config.cpu_limit and cpu_percent > self.config.cpu_limit:
                logger.warning(
                    "Process %s CPU usage %.1f%% exceeds limit %.1f%%",
                    self.config.name,
                    cpu_percent,
                    self.config.cpu_limit,
                )

            if self.config.memory_limit and memory_rss > self.config.memory_limit:
                logger.warning(
                    "Process %s memory usage %d exceeds limit %d",
                    self.config.name,
                    memory_rss,
                    self.config.memory_limit,
                )

            self.stats.last_health_check = time.time()

        except _PROCESS_RECOVERABLE_ERRORS as e:
            _record_process_degradation(
                e,
                action="kept last health sample after psutil health check failed",
                stage="managed_process.check_health",
                process_name=self.config.name,
            )
            logger.error("Health check error for %s: %s", self.config.name, e)

    def get_status(self) -> dict[str, Any]:
        """Get process status."""
        with self._lock:
            pid = self.process.pid if self.process else None
            alive = self.process.is_alive() if self.process else False

            # Calculate CPU and memory averages
            avg_cpu = (
                sum(self.stats.cpu_usage) / len(self.stats.cpu_usage) if self.stats.cpu_usage else 0
            )
            avg_memory = (
                sum(self.stats.memory_usage) / len(self.stats.memory_usage)
                if self.stats.memory_usage
                else 0
            )

            return {
                "name": self.config.name,
                "state": self.state.value,
                "pid": pid,
                "alive": alive,
                "restarts": self.stats.restarts,
                "uptime": time.time() - self.stats.start_time if alive else self.stats.total_uptime,
                "avg_cpu": round(avg_cpu, 1),
                "avg_memory": avg_memory,
                "last_health_check": self.stats.last_health_check,
                "observation_source": self.stats.observation_source,
                "observation_scenario_id": self.stats.observation_scenario_id,
            }


class ProcessManager:
    """Enterprise process manager with supervision and monitoring.

    Features:
    1. Process lifecycle management
    2. Automatic health monitoring
    3. Resource limit enforcement
    4. Graceful shutdown coordination
    5. Comprehensive metrics collection
    """

    def __init__(
        self,
        *,
        install_signal_handlers: bool = False,
        register_atexit: bool = False,
        cleanup_timeout_s: float = 6.0,
    ):
        self.processes: dict[str, ManagedProcess] = {}
        self.shutdown_event = threading.Event()
        self._lock = threading.RLock()
        self._monitor_thread: threading.Thread | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self.cleanup_timeout_s = max(1.0, float(cleanup_timeout_s))
        self._cleanup_lock = threading.Lock()
        self._cleanup_started = threading.Event()
        self._cleanup_complete = threading.Event()
        self._cleanup_owner_thread_id: int | None = None
        self._last_cleanup_summary: dict[str, Any] = {
            "status": "not_started",
            "successful": 0,
            "total": 0,
            "failed": [],
        }
        self._installed_signal_handlers: dict[signal.Signals, Any] = {}
        if install_signal_handlers:
            self._register_signal_handlers()
        if register_atexit:
            atexit.register(self.cleanup)

    def _register_signal_handlers(self) -> None:
        """Opt-in fallback for standalone use.

        Aura's root runtime owns signals. ProcessManager must not replace an
        event-loop handler merely because the service was instantiated.
        """
        try:
            signals = [signal.SIGTERM, signal.SIGINT]
            if hasattr(signal, "SIGHUP"):
                signals.append(signal.SIGHUP)
            for sig in signals:
                if sig not in self._installed_signal_handlers:
                    previous = signal.signal(sig, self._signal_handler)
                    self._installed_signal_handlers[sig] = previous
        except ValueError:
            logger.info("ProcessManager signal handlers skipped outside the main thread")

    def _restore_signal_handlers(self) -> None:
        for sig, previous in list(self._installed_signal_handlers.items()):
            try:
                if signal.getsignal(sig) == self._signal_handler:
                    signal.signal(sig, previous)
            except (OSError, RuntimeError, ValueError) as exc:
                logger.debug("Unable to restore ProcessManager signal %s: %s", sig, exc)
        self._installed_signal_handlers.clear()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Latch global shutdown before standalone fallback cleanup."""
        signal_name = signal.Signals(signum).name
        logger.info("Received signal %s (%s), initiating shutdown...", signal_name, signum)
        request_shutdown(f"process_manager_signal:{signal_name}")
        self.shutdown_event.set()
        self.cleanup()

    def _shutdown_started_now(self) -> bool:
        return self.shutdown_event.is_set() or is_shutdown_requested()

    def register_process(self, config: ProcessConfig) -> bool:
        """Register a process for management.

        Args:
            config: Process configuration

        Returns:
            True if registered successfully

        """
        if self._shutdown_started_now():
            logger.info("Process %s registration refused during shutdown", config.name)
            return False
        with self._lock:
            if self._shutdown_started_now():
                logger.info("Process %s registration refused during shutdown", config.name)
                return False
            if config.name in self.processes:
                logger.warning("Process %s already registered", config.name)
                return False

            try:
                process = ManagedProcess(config)
                self.processes[config.name] = process
                logger.info("Registered process: %s", config.name)
                return True

            except ValueError as e:
                logger.error("Invalid process configuration for %s: %s", config.name, e)
                return False

    async def start_process(self, name: str) -> bool:
        """Start a managed process.

        Args:
            name: Process name

        Returns:
            True if started successfully

        """
        if self._shutdown_started_now():
            logger.info("Process %s start refused during shutdown", name)
            return False
        self._event_loop = asyncio.get_running_loop()
        with self._lock:
            process = self.processes.get(name)
            if process is None:
                logger.error("Process %s not registered", name)
                return False
        return await process.start()

    def stop_process(
        self,
        name: str,
        force: bool = False,
        *,
        timeout_s: float | None = None,
    ) -> bool:
        """Stop a managed process.

        Args:
            name: Process name
            force: Force kill if graceful stop fails

        Returns:
            True if stopped successfully

        """
        with self._lock:
            process = self.processes.get(name)
            if process is None:
                logger.error("Process %s not found", name)
                return False
        return process.stop(force=force, timeout_s=timeout_s)

    async def restart_process(self, name: str) -> bool:
        """Restart a managed process.

        Args:
            name: Process name

        Returns:
            True if restarted successfully

        """
        if self._shutdown_started_now():
            logger.info("Process %s restart refused during shutdown", name)
            return False
        with self._lock:
            process = self.processes.get(name)
            if process is None:
                logger.error("Process %s not found", name)
                return False
        return await process.restart()

    async def start_all(self) -> dict[str, bool]:
        """Start all registered processes."""
        if self._shutdown_started_now():
            return {}
        results: dict[str, bool] = {}
        self._event_loop = asyncio.get_running_loop()
        with self._lock:
            names = list(self.processes)
        for name in names:
            results[name] = await self.start_process(name)
        return results

    def stop_all(
        self,
        force: bool = False,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, bool]:
        """Stop all registered processes within one manager-level budget."""

        results: dict[str, bool] = {}
        with self._lock:
            processes = list(self.processes.items())
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, timeout_s)
        for index, (name, process) in enumerate(processes):
            per_process_timeout = None
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                remaining_processes = max(1, len(processes) - index)
                per_process_timeout = remaining / remaining_processes
            results[name] = process.stop(
                force=force,
                timeout_s=per_process_timeout,
            )
        return results

    def start_monitoring(self, interval: int = 60) -> bool:
        """Start process monitoring thread."""
        if self._shutdown_started_now():
            logger.info("Process monitoring start refused during shutdown")
            return False
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Monitor thread already running")
            return False

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), name="ProcessMonitor", daemon=True
        )
        self._monitor_thread.start()
        logger.info("Process monitoring started (interval: %ss)", interval)
        return True

    def _monitor_loop(self, interval: int) -> None:
        """Process monitoring loop."""
        while not self._shutdown_started_now():
            try:
                self._check_all_processes()
            except _PROCESS_RECOVERABLE_ERRORS as e:
                _record_process_degradation(
                    e,
                    action="continued process monitor loop after global check failed",
                    stage="process_manager.monitor_loop",
                )
                logger.error("Process monitor error: %s", e)

            # Wait for next check or shutdown
            self.shutdown_event.wait(interval)

    def _check_all_processes(self) -> None:
        """Check health of all processes."""
        if self._shutdown_started_now():
            return
        with self._lock:
            processes = list(self.processes.items())
        for name, process in processes:
            if self._shutdown_started_now():
                return
            try:
                status = process.get_status()

                if status["state"] == ProcessState.RUNNING.value and not status["alive"]:
                    logger.error("Process %s died unexpectedly", name)
                    process.state = ProcessState.FAILED

                    if (
                        process.stats.restarts < process.config.max_restarts
                        and not self._shutdown_started_now()
                    ):
                        logger.info("Auto-restarting process %s", name)
                        loop = self._event_loop
                        if loop and not loop.is_closed() and not self._shutdown_started_now():
                            asyncio.run_coroutine_threadsafe(process.restart(), loop)
                        else:
                            logger.warning(
                                "No live event loop available to restart process %s", name
                            )

            except _PROCESS_RECOVERABLE_ERRORS as exc:
                _record_process_degradation(
                    exc,
                    action="continued checking remaining processes after status probe failed",
                    stage="process_manager.check_all",
                    process_name=name,
                )
                logger.error("Error checking process %s: %s", name, exc)

    async def on_stop_async(self) -> dict[str, Any]:
        """Container/coordinator hook that keeps blocking joins off the loop."""

        self.shutdown_event.set()
        if self._cleanup_complete.is_set():
            return dict(self._last_cleanup_summary)
        summary = await run_sync_shutdown_callable(
            lambda: self.cleanup(timeout_s=self.cleanup_timeout_s),
            timeout_s=self.cleanup_timeout_s + 1.0,
            name="process-manager",
        )
        return dict(summary)

    def cleanup(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Clean up all processes exactly once within a bounded budget."""

        if not is_shutdown_requested():
            request_shutdown("process_manager.cleanup")
        self.shutdown_event.set()
        if self._cleanup_complete.is_set():
            return dict(self._last_cleanup_summary)

        budget = self.cleanup_timeout_s if timeout_s is None else max(0.1, float(timeout_s))
        acquired = self._cleanup_lock.acquire(blocking=False)
        if not acquired:
            if self._cleanup_owner_thread_id == threading.get_ident():
                return {**self._last_cleanup_summary, "status": "in_progress"}
            self._cleanup_complete.wait(timeout=budget)
            return dict(self._last_cleanup_summary)

        started = time.monotonic()
        self._cleanup_owner_thread_id = threading.get_ident()
        self._cleanup_started.set()
        self._last_cleanup_summary = {
            "status": "in_progress",
            "successful": 0,
            "total": len(self.processes),
            "failed": [],
        }
        try:
            self._restore_signal_handlers()

            monitor = self._monitor_thread
            if (
                monitor
                and monitor.is_alive()
                and monitor is not threading.current_thread()
            ):
                monitor.join(timeout=min(2.0, budget))

            remaining = max(0.0, budget - (time.monotonic() - started))
            stop_results = self.stop_all(force=True, timeout_s=remaining)
            failed = sorted(name for name, success in stop_results.items() if not success)
            successful = sum(1 for success in stop_results.values() if success)
            self._last_cleanup_summary = {
                "status": "complete" if not failed else "degraded",
                "successful": successful,
                "total": len(stop_results),
                "failed": failed,
                "duration_seconds": round(time.monotonic() - started, 6),
            }
            return dict(self._last_cleanup_summary)
        except _PROCESS_RECOVERABLE_ERRORS as exc:
            _record_process_degradation(
                exc,
                action="completed bounded process cleanup with an internal failure",
                stage="process_manager.cleanup",
                severity="degraded",
            )
            self._last_cleanup_summary = {
                "status": "degraded",
                "successful": 0,
                "total": len(self.processes),
                "failed": sorted(self.processes),
                "duration_seconds": round(time.monotonic() - started, 6),
                "error": repr(exc),
            }
            return dict(self._last_cleanup_summary)
        finally:
            self._cleanup_owner_thread_id = None
            self._cleanup_complete.set()
            self._cleanup_lock.release()

    def get_status(self) -> dict[str, Any]:
        """Get status of all processes."""
        with self._lock:
            processes_status = {}
            for name, process in self.processes.items():
                processes_status[name] = process.get_status()

            return {
                "total_processes": len(self.processes),
                "running_processes": sum(1 for p in processes_status.values() if p["alive"]),
                "processes": processes_status,
                "shutdown_initiated": self._shutdown_started_now(),
                "shutdown_request": shutdown_request_snapshot(),
                "cleanup_started": self._cleanup_started.is_set(),
                "cleanup_complete": self._cleanup_complete.is_set(),
                "cleanup_summary": dict(self._last_cleanup_summary),
            }

    def get_process_stats(self, name: str) -> dict[str, Any] | None:
        """Get detailed statistics for a process."""
        with self._lock:
            if name not in self.processes:
                return None

            process = self.processes[name]
            status = process.get_status()
            stats = process.stats

            return {
                **status,
                "cpu_samples": len(stats.cpu_usage),
                "memory_samples": len(stats.memory_usage),
                "cpu_history": stats.cpu_usage[-20:],  # Last 20 samples
                "memory_history": stats.memory_usage[-20:],
                "config": {
                    "max_restarts": process.config.max_restarts,
                    "restart_window": process.config.restart_window,
                    "cpu_limit": process.config.cpu_limit,
                    "memory_limit": process.config.memory_limit,
                },
            }

    def export_metrics(self) -> dict[str, Any]:
        """Export metrics for monitoring systems."""
        with self._lock:
            metrics: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "process_manager": {
                    "total_processes": len(self.processes),
                    "shutdown_initiated": self._shutdown_started_now(),
                    "cleanup_started": self._cleanup_started.is_set(),
                    "cleanup_complete": self._cleanup_complete.is_set(),
                },
                "processes": {},
            }

            for name, process in self.processes.items():
                status = process.get_status()
                metrics["processes"][name] = {
                    "state": status["state"],
                    "alive": status["alive"],
                    "pid": status["pid"],
                    "restarts": status["restarts"],
                    "uptime": status["uptime"],
                    "avg_cpu": status["avg_cpu"],
                    "avg_memory": status["avg_memory"],
                }

            return metrics
