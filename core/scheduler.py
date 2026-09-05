from __future__ import annotations

import asyncio
import enum
import inspect
import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import describe_error, record_degradation
from core.runtime.numeric_safety import is_usable
from core.runtime.service_registry import get_runtime_service, register_runtime_service
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.utils.task_tracker import get_task_tracker, mark_task_protected

logger = logging.getLogger("Aura.Scheduler")

class Lifecycle(enum.Enum):
    INITIALIZING = "initializing"
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    MODIFYING = "modifying"
    RECOVERING = "recovering"
    SHUTTING_DOWN = "shutting_down"

@dataclass
class TaskSpec:
    name: str
    coro: Callable[[], Any]  # Awaitable factory or coroutine function
    tick_interval: float | None = None  # None means one-shot or event-driven.
    last_run: float = field(default_factory=lambda: 0.0)
    running_task: asyncio.Task | None = None
    critical: bool = False
    priority: int = 0 # Higher = more urgent
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_s: float | None = None
    last_started_at: float = 0.0
    last_completed_at: float = 0.0
    last_duration_s: float = 0.0
    run_count: int = 0
    failure_count: int = 0
    last_error: str = ""

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("scheduled task name must be non-empty")
        # CP126 (high): "Intervals accept zero and non-finite values."
        #
        # The non-finite half is a real defect: `float(nan) < 0` is False, so
        # a NaN interval passed validation, and then `now - last_run >=
        # interval` is False forever — the task was registered, reported
        # healthy, and never ran again. Silent permanent non-execution is the
        # worst outcome a scheduler has.
        #
        # Zero is NOT a defect here, and rejecting it was wrong. The run loop
        # treats `now - last_run >= 0` as "run on every scheduler tick", which
        # is bounded by the scheduler's own cadence rather than being a busy
        # loop, and it is an idiom the suite relies on. Negative intervals
        # stay rejected because they mean nothing.
        if self.tick_interval is not None:
            if not is_usable(self.tick_interval):
                raise ValueError("scheduled task interval must be a finite number")
            if float(self.tick_interval) < 0:
                raise ValueError("scheduled task interval must be non-negative")
        if self.timeout_s is not None:
            if not is_usable(self.timeout_s):
                raise ValueError("scheduled task timeout must be a finite number")
            if float(self.timeout_s) <= 0:
                raise ValueError("scheduled task timeout must be positive")

class Scheduler:
    """
    Central Scheduler for Aura's autonomic nervous system.
    Manages background loops, heartbeats, and metabolic tasks with 
    deterministic concurrency and error isolation.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Audit-38: Ensure __init__ only runs once across singleton access
        if getattr(self, "_initialized", False):
            return
        self._tasks: dict[str, TaskSpec] = {}
        self.state = Lifecycle.INITIALIZING
        self._lock: asyncio.Lock | None = None
        self._stop: asyncio.Event | None = None
        self._health: dict[str, str] = {}
        self._main_loop_task: asyncio.Task | None = None
        self._initialized = True
        try:
            if get_runtime_service("scheduler", default=None) is None:
                register_runtime_service("scheduler", self, required=False, owner="core/scheduler.py", registered_by="Scheduler.__init__")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("scheduler", exc)
            logger.debug("Scheduler registry publication unavailable during import: %s", exc)
        logger.info("Scheduler substrate initialized.")

    def _ensure_async_primitives(self) -> tuple[asyncio.Lock, asyncio.Event]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._stop is None:
            self._stop = asyncio.Event()
        return self._lock, self._stop

    async def register(self, spec: TaskSpec):
        """Register a subsystem task with the scheduler."""
        lock, _stop = self._ensure_async_primitives()
        async with lock:
            if spec.name in self._tasks:
                logger.warning("Task %s already registered. Updating spec.", spec.name)
            self._tasks[spec.name] = spec
            self._health[spec.name] = "registered"
            logger.debug("Registered task: %s (interval=%s)", spec.name, spec.tick_interval)

    async def start(self):
        """Ignite the scheduling loop."""
        if self._main_loop_task and not self._main_loop_task.done():
            logger.warning("Scheduler already running.")
            return

        _lock, stop_event = self._ensure_async_primitives()
        self.state = Lifecycle.IDLE
        stop_event.clear()
        self._main_loop_task = get_task_tracker().create_task(
            self._main_loop(),
            name="aura.scheduler.main_loop",
        )
        mark_task_protected(self._main_loop_task, owner="scheduler")
        logger.info("🚀 Scheduler started.")

    async def _main_loop(self):
        """The heartbeat of the scheduler."""
        lock, stop_event = self._ensure_async_primitives()
        while not stop_event.is_set():
            try:
                now = time.monotonic()
                async with lock:
                    pending_tasks = sorted(
                        self._tasks.values(), 
                        key=lambda x: (x.critical, x.priority), 
                        reverse=True
                    )
                
                for spec in pending_tasks:
                    if spec.tick_interval is None:
                        continue
                    
                    # Check if it's time to run and not already running
                    if now - spec.last_run >= spec.tick_interval:
                        if spec.running_task is None or spec.running_task.done():
                            spec.last_run = now
                            spec.running_task = get_task_tracker().create_task(
                                self._run_task(spec),
                                name=f"scheduler.{spec.name}",
                            )
                
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                if stop_event.is_set() or is_shutdown_requested():
                    logger.info("Scheduler main loop cancelled cleanly.")
                    break
                logger.warning("Scheduler loop spuriously cancelled. Ignoring.")
                continue
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('scheduler', e)
                logger.error("Scheduler Fatal Crash: %s", e)
                logger.error(traceback.format_exc())
                self.state = Lifecycle.RECOVERING
                await asyncio.sleep(1.0)

    async def _run_task(self, spec: TaskSpec):
        """Safely execute a task with structured monitoring."""
        started = time.monotonic()
        spec.last_started_at = time.time()
        spec.run_count += 1
        try:
            self._health[spec.name] = "running"

            # Handle both coro functions and direct coroutines
            res = spec.coro()
            if inspect.isawaitable(res):
                if spec.timeout_s is None:
                    await res
                else:
                    await asyncio.wait_for(res, timeout=float(spec.timeout_s))

            self._health[spec.name] = "ok"
            spec.last_error = ""
        except asyncio.CancelledError:
            self._health[spec.name] = "cancelled"
            raise
        except Exception as e:  # noqa: BLE001 - final isolation boundary for periodic work
            self._health[spec.name] = f"error: {type(e).__name__}"
            spec.failure_count += 1
            spec.last_error = f"{type(e).__name__}: {e}".rstrip()
            record_degradation(
                "scheduler",
                e,
                severity="warning" if not spec.critical else "degraded",
                action="marked scheduled task failed and escalated critical task to recovery",
                extra={"task": spec.name, "critical": spec.critical},
            )
            # describe_error, not str(e): a bare RuntimeError() renders as
            # nothing, and "Task web_search failed: " with an empty cause is
            # exactly what made the 2026-07-18 soak's tool failures
            # undiagnosable.
            logger.error("Task %s failed: %s", spec.name, describe_error(e))
            if spec.critical:
                logger.critical("CRITICAL Task %s failed! Triggering recovery.", spec.name)
                self.state = Lifecycle.RECOVERING
        finally:
            spec.last_duration_s = max(0.0, time.monotonic() - started)
            spec.last_completed_at = time.time()
            spec.running_task = None

    async def stop(self):
        """Gracefully shut down all scheduled tasks."""
        logger.info("Shutting down scheduler...")
        lock, stop_event = self._ensure_async_primitives()
        stop_event.set()
        self.state = Lifecycle.SHUTTING_DOWN
        
        async with lock:
            for spec in self._tasks.values():
                if spec.running_task:
                    spec.running_task.cancel()
        
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in scheduler.py: %s', _e)
        
        logger.info("Scheduler disengaged.")

    def get_health(self):
        """Return structured health check data for the system API."""
        now = time.time()
        task_details: dict[str, dict[str, Any]] = {}
        for name, spec in self._tasks.items():
            running = bool(spec.running_task and not spec.running_task.done())
            completion_age = (
                max(0.0, now - spec.last_completed_at)
                if spec.last_completed_at > 0.0
                else None
            )
            if running:
                freshness = "running"
            elif spec.run_count == 0:
                freshness = "never_run"
            elif spec.tick_interval is None:
                freshness = "completed"
            else:
                max_age = max(float(spec.tick_interval) * 2.0, float(spec.tick_interval) + 1.0)
                freshness = (
                    "fresh"
                    if completion_age is not None and completion_age <= max_age
                    else "stale"
                )
            task_details[name] = {
                "status": self._health.get(name, "unknown"),
                "critical": spec.critical,
                "priority": spec.priority,
                "tick_interval_s": spec.tick_interval,
                "timeout_s": spec.timeout_s,
                "running": running,
                "freshness": freshness,
                "last_started_at": spec.last_started_at or None,
                "last_completed_at": spec.last_completed_at or None,
                "completion_age_s": (
                    round(completion_age, 3) if completion_age is not None else None
                ),
                "last_duration_s": round(spec.last_duration_s, 3),
                "run_count": spec.run_count,
                "failure_count": spec.failure_count,
                "last_error": spec.last_error,
                "metadata": dict(spec.metadata),
            }
        return {
            "state": self.state.value,
            "tasks": dict(self._health),
            "task_details": task_details,
            "active_tasks": len(
                [
                    task
                    for task in self._tasks.values()
                    if task.running_task and not task.running_task.done()
                ]
            ),
        }

    def is_alive(self) -> bool:
        """Deep liveness probe for the runtime health contract."""
        if self.state == Lifecycle.SHUTTING_DOWN:
            return False
        return bool(self._main_loop_task is not None and not self._main_loop_task.done())

# Global Instance
scheduler = Scheduler()
