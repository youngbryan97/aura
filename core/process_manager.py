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

import psutil

from core.runtime.errors import FallbackClassification, record_degradation
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
    severity: str = "warning",
) -> None:
    extra = {"stage": stage}
    if process_name:
        extra["process_name"] = process_name
    try:
        record_degradation(
            "process_manager",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=extra,
        )
    except TypeError:
        record_degradation(
            "process_manager",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


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
    target: Callable
    args: tuple = ()
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

    def __post_init__(self):
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


class ManagedProcess:
    """Managed process with supervision."""

    def __init__(self, config: ProcessConfig):
        self.config = config
        self.process: mp.Process | None = None
        self.state = ProcessState.INITIALIZING
        self.stats = ProcessStats(start_time=time.time())
        self.last_restart_attempt: float | None = None
        self._lock = threading.RLock()
        self._health_check_thread: threading.Thread | None = None
        self._health_check_task: asyncio.Task | None = None
        self._stop_health_check = threading.Event()

    async def start(self) -> bool:
        """Start the process."""
        with self._lock:
            if self.state in [ProcessState.STARTING, ProcessState.RUNNING]:
                logger.warning("Process %s already running", self.config.name)
                return False

            logger.info("Starting process: %s", self.config.name)
            self.state = ProcessState.STARTING

            try:
                # Create process
                self.process = mp.Process(
                    target=self._process_wrapper,
                    args=(self.config.target, self.config.args, self.config.kwargs),
                    name=self.config.name,
                    daemon=self.config.daemon,
                )

                # Set process priority if supported
                if os.name == "posix" and self.config.priority != 0:
                    # Lower nice value = higher priority
                    os.nice(self.config.priority)

                # Start process
                self.process.start()

                # Wait for startup
                start_time = time.time()
                while time.time() - start_time < self.config.startup_timeout:
                    if self.process.is_alive():
                        self.state = ProcessState.RUNNING
                        self.stats.start_time = time.time()
                        logger.info(
                            "Process %s started (PID: %s)", self.config.name, self.process.pid
                        )

                        # Start health monitoring
                        self._start_health_monitoring()
                        return True
                    await asyncio.sleep(0.1)

                # Startup timeout
                self.state = ProcessState.FAILED
                logger.error("Process %s failed to start within timeout", self.config.name)
                return False

            except _PROCESS_RECOVERABLE_ERRORS as e:
                _record_process_degradation(
                    e,
                    action="marked process failed after startup exception",
                    stage="managed_process.start",
                    process_name=self.config.name,
                    severity="degraded",
                )
                self.state = ProcessState.FAILED
                logger.error("Failed to start process %s: %s", self.config.name, e, exc_info=True)
                return False

    def _process_wrapper(self, target: Callable, args: tuple, kwargs: dict[str, Any]):
        """Wrapper for process execution with error handling."""
        process_name = mp.current_process().name

        try:
            # Set resource limits if specified
            if self.config.memory_limit and os.name == "posix":
                resource.setrlimit(
                    resource.RLIMIT_AS, (self.config.memory_limit, self.config.memory_limit)
                )

            logger.info("Process %s executing target function", process_name)
            target(*args, **kwargs)

        except KeyboardInterrupt:
            logger.info("Process %s interrupted gracefully", process_name)
        except _PROCESS_RECOVERABLE_ERRORS as e:
            _record_process_degradation(
                e,
                action="re-raised supervised child crash after recording process failure",
                stage="managed_process.wrapper",
                process_name=process_name,
                severity="degraded",
            )
            logger.error("Process %s crashed: %s", process_name, e, exc_info=True)
            raise
        finally:
            logger.info("Process %s exiting", process_name)

    def stop(self, force: bool = False) -> bool:
        """Stop the process gracefully."""
        with self._lock:
            if self.state in [ProcessState.STOPPING, ProcessState.STOPPED]:
                logger.debug("Process %s already stopping/stopped", self.config.name)
                return True

            if not self.process or not self.process.is_alive():
                self.state = ProcessState.STOPPED
                return True

            logger.info("Stopping process: %s", self.config.name)
            self.state = ProcessState.STOPPING

            # Stop health monitoring
            self._stop_health_monitoring()

            try:
                # Try graceful termination
                if self.process.pid:
                    try:
                        psutil_process = psutil.Process(self.process.pid)
                        psutil_process.terminate()  # SIGTERM
                    except psutil.NoSuchProcess:
                        logger.debug("Exception caught during execution", exc_info=True)

                # Wait for graceful shutdown
                self.process.join(timeout=self.config.shutdown_timeout)

                if self.process.is_alive():
                    if force:
                        # Force kill
                        logger.warning("Process %s not responding, forcing kill", self.config.name)
                        self.process.kill()  # SIGKILL
                        self.process.join(timeout=5)
                    else:
                        logger.error("Process %s didn't stop gracefully", self.config.name)
                        return False

                self.state = ProcessState.STOPPED
                logger.info("Process %s stopped", self.config.name)

                # Update stats
                self.stats.total_uptime += time.time() - self.stats.start_time

                return True

            except _PROCESS_RECOVERABLE_ERRORS as e:
                _record_process_degradation(
                    e,
                    action="marked process failed after stop operation failed",
                    stage="managed_process.stop",
                    process_name=self.config.name,
                    severity="degraded",
                )
                logger.error("Error stopping process %s: %s", self.config.name, e)
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
        return final

    async def restart(self) -> bool:
        """Restart the process with exponential backoff."""
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

            # Stop if running
            if self.process and self.process.is_alive():
                self.stop()

        # Backoff delay OUTSIDE the lock
        if backoff_s > 1.0:
            await asyncio.sleep(backoff_s)

        with self._lock:
            # Start again
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

    async def _start_health_monitoring(self):
        """Start health monitoring task."""
        if self._health_check_task and not self._health_check_task.done():
            return

        self._stop_health_check.clear()
        self._health_check_task = get_task_tracker().create_task(
            self._health_monitor_loop(),
            name=f"process_manager.{self.config.name}.health_monitor",
        )
        logger.debug("Started health monitoring for %s", self.config.name)

    async def _stop_health_monitoring(self):
        """Stop health monitoring task."""
        self._stop_health_check.set()
        if self._health_check_task and not self._health_check_task.done():
            try:
                await asyncio.wait_for(self._health_check_task, timeout=5)
            except TimeoutError:
                logger.warning(
                    "Health monitoring task for %s did not stop gracefully, cancelling.",
                    self.config.name,
                )
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    logger.debug("Exception caught during execution", exc_info=True)
            except asyncio.CancelledError:
                logger.debug("Exception caught during execution", exc_info=True)
        self._health_check_task = None

    async def _health_monitor_loop(self):
        """Continuous health monitoring loop."""
        while not self._stop_health_check.is_set():
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

    def _check_health(self):
        """Perform health check."""
        if not self.process or not self.process.is_alive():
            logger.warning("Process %s is not alive", self.config.name)
            return

        try:
            psutil_process = psutil.Process(self.process.pid)

            # Check CPU usage
            cpu_percent = psutil_process.cpu_percent(interval=0.1)
            self.stats.cpu_usage.append(cpu_percent)
            if len(self.stats.cpu_usage) > 100:  # Keep last 100 samples
                self.stats.cpu_usage.pop(0)

            # Check memory usage
            memory_info = psutil_process.memory_info()
            self.stats.memory_usage.append(memory_info.rss)
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

            if self.config.memory_limit and memory_info.rss > self.config.memory_limit:
                logger.warning(
                    "Process %s memory usage %d exceeds limit %d",
                    self.config.name,
                    memory_info.rss,
                    self.config.memory_limit,
                )

            self.stats.last_health_check = time.time()

        except psutil.NoSuchProcess:
            logger.warning("Process %s PID %s not found", self.config.name, self.process.pid)
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

    def __init__(self):
        self.processes: dict[str, ManagedProcess] = {}
        self.shutdown_event = threading.Event()
        self._lock = threading.RLock()
        self._monitor_thread: threading.Thread | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._register_signal_handlers()
        atexit.register(self.cleanup)

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

            # Unix-specific signals
            if hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, self._signal_handler)
        except ValueError:
            logger.info("Signal handlers skipped (not main thread — desktop mode)")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info("Received signal %s (%s), initiating shutdown...", signal_name, signum)
        self.shutdown_event.set()
        self.cleanup()

    def register_process(self, config: ProcessConfig) -> bool:
        """Register a process for management.

        Args:
            config: Process configuration

        Returns:
            True if registered successfully

        """
        with self._lock:
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
        with self._lock:
            try:
                self._event_loop = asyncio.get_running_loop()
            except RuntimeError as _exc:
                logger.debug("Suppressed RuntimeError: %s", _exc)
            if name not in self.processes:
                logger.error("Process %s not registered", name)
                return False

            return await self.processes[name].start()

    def stop_process(self, name: str, force: bool = False) -> bool:
        """Stop a managed process.

        Args:
            name: Process name
            force: Force kill if graceful stop fails

        Returns:
            True if stopped successfully

        """
        with self._lock:
            if name not in self.processes:
                logger.error("Process %s not found", name)
                return False

            return self.processes[name].stop(force=force)

    async def restart_process(self, name: str) -> bool:
        """Restart a managed process.

        Args:
            name: Process name

        Returns:
            True if restarted successfully

        """
        with self._lock:
            if name not in self.processes:
                logger.error("Process %s not found", name)
                return False

            return await self.processes[name].restart()

    async def start_all(self) -> dict[str, bool]:
        """Start all registered processes."""
        results = {}
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError as _exc:
            logger.debug("Suppressed RuntimeError: %s", _exc)
        with self._lock:
            for name in list(self.processes.keys()):
                results[name] = await self.start_process(name)
        return results

    def stop_all(self, force: bool = False) -> dict[str, bool]:
        """Stop all registered processes."""
        results = {}
        with self._lock:
            for name in list(self.processes.keys()):
                results[name] = self.stop_process(name, force=force)
        return results

    def start_monitoring(self, interval: int = 60):
        """Start process monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Monitor thread already running")
            return

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), name="ProcessMonitor", daemon=True
        )
        self._monitor_thread.start()
        logger.info("Process monitoring started (interval: %ss)", interval)

    def _monitor_loop(self, interval: int):
        """Process monitoring loop."""
        while not self.shutdown_event.is_set():
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

    def _check_all_processes(self):
        """Check health of all processes."""
        with self._lock:
            for name, process in list(self.processes.items()):
                try:
                    status = process.get_status()

                    # Check if process died unexpectedly
                    if status["state"] == ProcessState.RUNNING.value and not status["alive"]:
                        logger.error("Process %s died unexpectedly", name)
                        process.state = ProcessState.FAILED

                        # Auto-restart if enabled
                        if process.stats.restarts < process.config.max_restarts:
                            logger.info("Auto-restarting process %s", name)
                            loop = self._event_loop
                            if loop and not loop.is_closed():
                                asyncio.run_coroutine_threadsafe(process.restart(), loop)
                            else:
                                logger.warning(
                                    "No live event loop available to restart process %s", name
                                )

                except _PROCESS_RECOVERABLE_ERRORS as e:
                    _record_process_degradation(
                        e,
                        action="continued checking remaining processes after status probe failed",
                        stage="process_manager.check_all",
                        process_name=name,
                    )
                    logger.error("Error checking process %s: %s", name, e)

    def cleanup(self):
        """Clean up all processes gracefully."""
        if self.shutdown_event.is_set():
            return  # Already cleaning up
        self.shutdown_event.set()

        # Stop monitoring
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

        # Stop all processes
        stop_results = self.stop_all(force=True)

        # Log results
        successful = sum(1 for success in stop_results.values() if success)
        total = len(stop_results)
        self._last_cleanup_summary = {"successful": successful, "total": total}

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
                "shutdown_initiated": self.shutdown_event.is_set(),
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
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "process_manager": {
                    "total_processes": len(self.processes),
                    "shutdown_initiated": self.shutdown_event.is_set(),
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
