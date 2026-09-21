"""Canonical ShutdownCoordinator.

Codifies the shutdown phase ordering required by the runtime invariants
audit:

    output flush -> memory commit -> state vault -> actors
    -> model runtime -> bus -> task supervisor

Each phase is a list of registered handlers. Handlers in the same phase
run concurrently; phases run sequentially. A handler that raises is logged
and treated as a failed phase, but does not abort the remaining phases —
the goal is to flush as much as possible during shutdown rather than abort
early on the first error.

Strict mode (AURA_STRICT_RUNTIME=1) elevates phase failures so they are
visible in tests and conformance harnesses, while still completing the
remaining phases.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from core.runtime.lifecycle_probe import (
    hold_shutdown_probe_async,
    shutdown_probe_hold_seconds,
)
from core.runtime.shutdown_artifact_store import (
    delete_shutdown_artifact,
    write_shutdown_artifact,
)
from core.runtime.shutdown_execution import run_sync_shutdown_callable
from core.runtime.state_ownership import state_root
from core.utils.task_tracker import (
    get_task_tracker,
)

logger = logging.getLogger("Aura.ShutdownCoordinator")

ShutdownHandler = Callable[[], None | Awaitable[None]]


# Canonical phases. Order matters.
SHUTDOWN_PHASES: tuple[str, ...] = (
    "output_flush",
    "memory_commit",
    "state_vault",
    "actors",
    "model_runtime",
    "event_bus",
    "task_supervisor",
)


@dataclass
class _RegisteredHandler:
    name: str
    handler: ShutdownHandler
    phase: str
    timeout: float = 15.0


@dataclass
class ShutdownReport:
    completed_phases: list[str] = field(default_factory=list)
    failed_phases: list[str] = field(default_factory=list)
    handler_failures: dict[str, str] = field(default_factory=dict)
    phase_durations_seconds: dict[str, float] = field(default_factory=dict)
    handler_durations_seconds: dict[str, float] = field(default_factory=dict)
    handler_statuses: dict[str, str] = field(default_factory=dict)
    escalations: list[dict[str, object]] = field(default_factory=list)
    started_at_unix: float | None = None
    completed_at_unix: float | None = None
    duration_seconds: float | None = None
    current_phase: str | None = None
    repeated_call_count: int = 0
    artifact_path: str | None = None

    @property
    def clean(self) -> bool:
        return not self.failed_phases and not self.handler_failures

    def clone(self) -> ShutdownReport:
        """Return a detached report so callers cannot mutate coordinator state."""

        return ShutdownReport(
            completed_phases=list(self.completed_phases),
            failed_phases=list(self.failed_phases),
            handler_failures=dict(self.handler_failures),
            phase_durations_seconds=dict(self.phase_durations_seconds),
            handler_durations_seconds=dict(self.handler_durations_seconds),
            handler_statuses=dict(self.handler_statuses),
            escalations=[dict(item) for item in self.escalations],
            started_at_unix=self.started_at_unix,
            completed_at_unix=self.completed_at_unix,
            duration_seconds=self.duration_seconds,
            current_phase=self.current_phase,
            repeated_call_count=self.repeated_call_count,
            artifact_path=self.artifact_path,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "clean": self.clean,
            "completed_phases": list(self.completed_phases),
            "failed_phases": list(self.failed_phases),
            "handler_failures": dict(self.handler_failures),
            "phase_durations_seconds": dict(self.phase_durations_seconds),
            "handler_durations_seconds": dict(self.handler_durations_seconds),
            "handler_statuses": dict(self.handler_statuses),
            "escalations": [dict(item) for item in self.escalations],
            "started_at_unix": self.started_at_unix,
            "completed_at_unix": self.completed_at_unix,
            "duration_seconds": self.duration_seconds,
            "current_phase": self.current_phase,
            "repeated_call_count": self.repeated_call_count,
            "artifact_path": self.artifact_path,
        }


class ShutdownCoordinator:
    """Single owner of teardown ordering."""

    # Coordinator phase -> verified-lifecycle state entered when that phase
    # begins. The FSM formalizes the coarse teardown arc (drain -> stop
    # services -> flush) on top of the fine-grained phase loop. Repeated
    # shutdown callers share one execution and replay its final report.
    _LIFECYCLE_BOUNDARIES: dict[str, str] = {
        "actors": "STOPPING_SERVICES",
        "task_supervisor": "FLUSHING_STATE",
    }

    def __init__(self, phases: tuple[str, ...] = SHUTDOWN_PHASES):
        self._phases = phases
        self._handlers: dict[str, list[_RegisteredHandler]] = {p: [] for p in phases}
        self._lock = threading.RLock()
        self._running = False
        self._shutdown_task: asyncio.Task[ShutdownReport] | None = None
        self._task_loop: asyncio.AbstractEventLoop | None = None
        self._completion = threading.Event()
        self._report: ShutdownReport | None = None
        self._working_report: ShutdownReport | None = None
        self._repeated_call_count = 0
        self._current_phase_started_monotonic: float | None = None
        self._current_phase_timeout_seconds: float | None = None
        self._current_phase_deadline_monotonic: float | None = None
        self._active_handlers: set[str] = set()
        self._handler_progress: dict[str, dict[str, object]] = {}
        self._lifecycle = self._build_lifecycle()

    @staticmethod
    def _build_lifecycle() -> Any:
        try:
            from core.resilience.verified_state_machine import (
                create_shutdown_lifecycle_machine,
            )
            return create_shutdown_lifecycle_machine()
        except (ImportError, RuntimeError, ValueError) as exc:
            # Lifecycle bookkeeping must never block teardown.
            logger.debug("Shutdown lifecycle machine unavailable: %s", exc)
            return None

    def lifecycle_state(self) -> str:
        """Current verified-lifecycle state (for diagnostics)."""
        return self._lifecycle.current if self._lifecycle is not None else "UNTRACKED"

    def _lifecycle_transition(self, to_state: str) -> None:
        """Advance the lifecycle machine; F17 is recorded on illegal moves,
        and teardown continues regardless — bookkeeping never blocks it."""
        if self._lifecycle is None:
            return
        try:
            from core.resilience.verified_state_machine import IllegalTransitionError
            try:
                self._lifecycle.transition(to_state)
            except IllegalTransitionError:
                pass  # already recorded as F17 by the machine
        except ImportError as exc:
            logger.debug("Lifecycle transition skipped: %s", exc)

    # --- Registration ---------------------------------------------------

    def register(
        self,
        handler: ShutdownHandler,
        *,
        phase: str,
        name: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        if phase not in self._handlers:
            raise ValueError(
                f"unknown shutdown phase '{phase}'; expected one of {self._phases}"
            )
        if not callable(handler):
            raise TypeError("shutdown handler must be callable")
        record = _RegisteredHandler(
            name=str(name or getattr(handler, "__name__", "anonymous")),
            handler=handler,
            phase=phase,
            timeout=timeout,
        )
        with self._lock:
            if self._running or self._report is not None:
                raise RuntimeError("cannot register shutdown handlers after teardown has started")
            self._handlers[phase].append(record)

    def clear(self) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("cannot clear shutdown handlers while teardown is running")
            for phase in self._handlers:
                self._handlers[phase] = []

    def phases(self) -> tuple[str, ...]:
        return self._phases

    def handler_names(self, phase: str) -> list[str]:
        with self._lock:
            return [h.name for h in self._handlers.get(phase, [])]

    def get_status(self) -> dict[str, object]:
        """Return a stable operator snapshot of latch and teardown progress."""

        with self._lock:
            now = time.monotonic()
            report = self._report or self._working_report
            report_payload = report.clone().as_dict() if report is not None else None
            registered_handlers = {
                phase: [record.name for record in records]
                for phase, records in self._handlers.items()
            }
            phase_elapsed = None
            if self._current_phase_started_monotonic is not None:
                phase_elapsed = max(0.0, now - self._current_phase_started_monotonic)
            phase_remaining = None
            if self._current_phase_deadline_monotonic is not None:
                phase_remaining = max(0.0, self._current_phase_deadline_monotonic - now)
            active_handlers = sorted(self._active_handlers)
            blocker_owner = active_handlers[0] if active_handlers else None
            if blocker_owner is None and report is not None and report.current_phase:
                blocker_owner = f"phase:{report.current_phase}"
            return {
                "running": self._running,
                "lifecycle_state": self.lifecycle_state(),
                "request": shutdown_request_snapshot(),
                "report": report_payload,
                "progress": {
                    "current_phase": report.current_phase if report is not None else None,
                    "phase_elapsed_seconds": (
                        round(phase_elapsed, 6) if phase_elapsed is not None else None
                    ),
                    "phase_timeout_seconds": self._current_phase_timeout_seconds,
                    "phase_remaining_seconds": (
                        round(phase_remaining, 6) if phase_remaining is not None else None
                    ),
                    "blocker_owner": blocker_owner,
                    "active_handlers": active_handlers,
                    "handlers": {
                        key: dict(value) for key, value in self._handler_progress.items()
                    },
                },
                "admission": shutdown_admission_snapshot(),
                "registered_handlers": registered_handlers,
            }

    # --- Execution ------------------------------------------------------

    async def shutdown(self, *, timeout_per_phase: float | None = None) -> ShutdownReport:
        request_shutdown("coordinator")
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._report is not None:
                self._repeated_call_count += 1
                replay = self._report.clone()
                replay.repeated_call_count = self._repeated_call_count
                return replay

            task: asyncio.Task[ShutdownReport] | None = self._shutdown_task
            task_loop = self._task_loop
            if task is None:
                started_at = time.time()
                self._working_report = ShutdownReport(started_at_unix=started_at)
                self._running = True
                self._completion.clear()
                task = cast(
                    asyncio.Task[ShutdownReport],
                    get_task_tracker().create_task(
                        self._execute_shutdown(timeout_per_phase=timeout_per_phase),
                        name="shutdown_coordinator.execute",
                        allow_during_shutdown=True,
                    ),
                )
                self._shutdown_task = task
                self._task_loop = loop
                task_loop = loop
            else:
                self._repeated_call_count += 1

        if task_loop is loop:
            report = await asyncio.shield(task)
            with self._lock:
                replay = report.clone()
                replay.repeated_call_count = self._repeated_call_count
                return replay

        # A coordinator may be reached from a second loop during process exit.
        # The teardown owner remains the first loop; other loops wait on a
        # thread-safe completion event instead of attempting duplicate cleanup.
        with self._lock:
            cross_loop_timeout = min(
                300.0,
                max(
                    10.0,
                    10.0
                    + sum(
                        max((record.timeout for record in records), default=0.0)
                        for records in self._handlers.values()
                    ),
                ),
            )
        remaining_global_budget = shutdown_remaining_budget_seconds()
        if remaining_global_budget is not None:
            cross_loop_timeout = min(
                cross_loop_timeout,
                max(0.05, remaining_global_budget),
            )
        completed = await run_sync_shutdown_callable(
            lambda: self._completion.wait(cross_loop_timeout),
            timeout_s=cross_loop_timeout + 0.1,
            name="coordinator-cross-loop-wait",
        )
        if not completed:
            raise TimeoutError(
                f"shutdown coordinator owner loop did not finish within {cross_loop_timeout:.1f}s"
            )
        with self._lock:
            if self._report is None:
                raise RuntimeError("shutdown coordinator completed without a report")
            replay = self._report.clone()
            replay.repeated_call_count = self._repeated_call_count
            return replay

    async def _execute_shutdown(
        self,
        *,
        timeout_per_phase: float | None,
    ) -> ShutdownReport:
        with self._lock:
            report = self._working_report
        if report is None:  # pragma: no cover - protected by shutdown()
            report = ShutdownReport(started_at_unix=time.time())

        started_monotonic = time.monotonic()
        self._lifecycle_transition("DRAINING")
        try:
            for phase in self._phases:
                phase_started = time.monotonic()
                probe_target = f"coordinator:{phase}"
                probe_hold_seconds = shutdown_probe_hold_seconds(probe_target)
                boundary = self._LIFECYCLE_BOUNDARIES.get(phase)
                if boundary:
                    self._lifecycle_transition(boundary)
                with self._lock:
                    report.current_phase = phase
                    handlers = list(self._handlers.get(phase, []))
                    if timeout_per_phase is not None:
                        configured_timeout = float(timeout_per_phase)
                    else:
                        longest_handler_timeout = max(
                            (handler.timeout for handler in handlers),
                            default=0.0,
                        )
                        # Give per-handler wait_for() enough scheduling margin to
                        # publish the real blocker before the phase-level fuse.
                        configured_timeout = longest_handler_timeout + min(
                            1.0,
                            max(0.1, longest_handler_timeout * 0.1),
                        )
                        if not handlers:
                            configured_timeout = 0.0
                    global_remaining = shutdown_remaining_budget_seconds()
                    effective_timeout = configured_timeout
                    if global_remaining is not None:
                        effective_timeout = min(
                            configured_timeout,
                            max(0.0, global_remaining - probe_hold_seconds),
                        )
                    self._current_phase_started_monotonic = phase_started
                    self._current_phase_timeout_seconds = effective_timeout
                    phase_deadline = phase_started + probe_hold_seconds + effective_timeout
                    global_deadline = shutdown_deadline_monotonic()
                    if global_deadline is not None:
                        phase_deadline = min(phase_deadline, global_deadline)
                    self._current_phase_deadline_monotonic = phase_deadline
                    self._active_handlers.clear()
                logger.info(
                    "ShutdownCoordinator: phase started "
                    "(phase=%s handlers=%d timeout_seconds=%.3f)",
                    phase,
                    len(handlers),
                    effective_timeout,
                )
                probe_budget = shutdown_remaining_budget_seconds()
                if probe_budget is not None and probe_budget <= 0.0:
                    self._record_global_deadline_exhausted(report, phase=phase)
                    break
                if probe_budget is None:
                    await hold_shutdown_probe_async(probe_target)
                else:
                    try:
                        await asyncio.wait_for(
                            hold_shutdown_probe_async(probe_target),
                            timeout=max(0.05, probe_budget),
                        )
                    except TimeoutError:
                        self._record_global_deadline_exhausted(report, phase=phase)
                        break
                if not handlers:
                    report.completed_phases.append(phase)
                    report.phase_durations_seconds[phase] = round(
                        time.monotonic() - phase_started, 6
                    )
                    with self._lock:
                        self._current_phase_started_monotonic = None
                        self._current_phase_timeout_seconds = None
                        self._current_phase_deadline_monotonic = None
                    continue
                remaining_after_probe = shutdown_remaining_budget_seconds()
                if remaining_after_probe is not None:
                    effective_timeout = min(effective_timeout, remaining_after_probe)
                if effective_timeout <= 0.0:
                    self._record_global_deadline_exhausted(report, phase=phase)
                    break
                phase_failed = False
                coros: list[asyncio.Future[Any]] = []
                for record in handlers:
                    coros.append(
                        get_task_tracker().track(
                            self._invoke(record, report),
                            name=f"shutdown:{phase}:{record.name}",
                            allow_during_shutdown=True,
                        )
                    )
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*coros, return_exceptions=True),
                        timeout=effective_timeout,
                    )
                except TimeoutError:
                    for fut in coros:
                        if not fut.done():
                            fut.cancel()
                    if phase not in report.failed_phases:
                        report.failed_phases.append(phase)
                    report.handler_failures[phase] = "phase timed out"
                    active_handlers = sorted(
                        key
                        for key, status in report.handler_statuses.items()
                        if status in {"running", "cancelled"}
                    )
                    global_deadline_exhausted = (
                        shutdown_remaining_budget_seconds() is not None
                        and shutdown_remaining_budget_seconds() <= 0.0
                    )
                    report.escalations.append(
                        {
                            "kind": (
                                "global_deadline_exhausted"
                                if global_deadline_exhausted
                                else "phase_timeout"
                            ),
                            "phase": phase,
                            "at_unix": time.time(),
                            "timeout_seconds": effective_timeout,
                            "active_handlers": active_handlers,
                        }
                    )
                    logger.error("Shutdown phase '%s' timed out", phase)
                    report.phase_durations_seconds[phase] = round(
                        time.monotonic() - phase_started, 6
                    )
                    with self._lock:
                        self._active_handlers.clear()
                        self._current_phase_started_monotonic = None
                        self._current_phase_timeout_seconds = None
                        self._current_phase_deadline_monotonic = None
                    continue
                for record, result in zip(handlers, results, strict=True):
                    if isinstance(result, BaseException):
                        phase_failed = True
                        msg = repr(result)
                        report.handler_failures[f"{phase}:{record.name}"] = msg
                        logger.error(
                            "Shutdown handler '%s' in phase '%s' failed: %s",
                            record.name,
                            phase,
                            msg,
                        )
                if phase_failed:
                    if phase not in report.failed_phases:
                        report.failed_phases.append(phase)
                else:
                    report.completed_phases.append(phase)
                report.phase_durations_seconds[phase] = round(
                    time.monotonic() - phase_started, 6
                )
                with self._lock:
                    self._active_handlers.clear()
                    self._current_phase_started_monotonic = None
                    self._current_phase_timeout_seconds = None
                    self._current_phase_deadline_monotonic = None
        except asyncio.CancelledError as exc:
            # Teardown is process-critical. Record cancellation as a failure but
            # still publish a final report so concurrent waiters never hang.
            report.failed_phases.append("coordinator")
            report.handler_failures["coordinator"] = repr(exc)
            logger.error("Shutdown coordinator task was cancelled before all phases completed")
        except Exception as exc:  # noqa: BLE001 - final process teardown boundary
            report.failed_phases.append("coordinator")
            report.handler_failures["coordinator"] = repr(exc)
            logger.error("Shutdown coordinator failed internally: %s", exc, exc_info=True)
        finally:
            self._lifecycle_transition("TERMINATED")
            report.current_phase = None
            report.completed_at_unix = time.time()
            report.duration_seconds = round(time.monotonic() - started_monotonic, 6)
            try:
                # On a thread: the verdict is two atomic writes with their
                # fsyncs, and this is the loop, still running the handlers
                # that have not finished (LIVE 2026-09-21).
                artifact = await asyncio.to_thread(
                    publish_shutdown_verdict,
                    coordinator_report=report,
                    stage="coordinator",
                    final=False,
                )
                report.artifact_path = str(artifact["artifact_path"])
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                if "coordinator" not in report.failed_phases:
                    report.failed_phases.append("coordinator")
                report.handler_failures["coordinator:durable_report"] = repr(exc)
                report.escalations.append(
                    {
                        "kind": "durable_report_failure",
                        "phase": "coordinator",
                        "at_unix": time.time(),
                        "error": repr(exc),
                    }
                )
                logger.error("Failed to persist shutdown coordinator report: %s", exc)
            with self._lock:
                self._running = False
                self._active_handlers.clear()
                self._current_phase_started_monotonic = None
                self._current_phase_timeout_seconds = None
                self._current_phase_deadline_monotonic = None
                self._report = report.clone()
                self._working_report = None
                self._completion.set()

        if not report.clean and os.environ.get("AURA_STRICT_RUNTIME") == "1":
            logger.error(
                "ShutdownCoordinator: strict mode shutdown failures: phases=%s handlers=%s",
                report.failed_phases,
                report.handler_failures,
            )
        else:
            logger.info(
                "ShutdownCoordinator: shutdown complete (clean=%s phases=%s)",
                report.clean,
                report.completed_phases,
            )
        return report

    def _record_global_deadline_exhausted(
        self,
        report: ShutdownReport,
        *,
        phase: str,
    ) -> None:
        if phase not in report.failed_phases:
            report.failed_phases.append(phase)
        report.handler_failures[phase] = "process-wide shutdown deadline exhausted"
        report.escalations.append(
            {
                "kind": "global_deadline_exhausted",
                "phase": phase,
                "at_unix": time.time(),
                "active_handlers": sorted(self._active_handlers),
                "request": shutdown_request_snapshot(),
            }
        )
        logger.error(
            "Shutdown process-wide deadline exhausted (phase=%s active_handlers=%s)",
            phase,
            sorted(self._active_handlers),
        )

    async def _invoke(
        self,
        record: _RegisteredHandler,
        report: ShutdownReport,
    ) -> None:
        key = f"{record.phase}:{record.name}"
        started = time.monotonic()
        with self._lock:
            self._active_handlers.add(key)
            self._handler_progress[key] = {
                "phase": record.phase,
                "name": record.name,
                "status": "running",
                "started_at_unix": time.time(),
                "timeout_seconds": record.timeout,
            }
            report.handler_statuses[key] = "running"
        try:
            effective_timeout = record.timeout
            remaining_global_budget = shutdown_remaining_budget_seconds()
            if remaining_global_budget is not None:
                effective_timeout = min(effective_timeout, remaining_global_budget)
            if effective_timeout <= 0.0:
                raise TimeoutError("process-wide shutdown deadline exhausted")
            deadline = time.monotonic() + effective_timeout
            if inspect.iscoroutinefunction(record.handler):
                result = record.handler()
            else:
                result = await run_sync_shutdown_callable(
                    cast(Callable[[], Any], record.handler),
                    timeout_s=effective_timeout,
                    name=f"{record.phase}:{record.name}",
                )
            if inspect.isawaitable(result):
                await asyncio.wait_for(
                    result,
                    timeout=max(0.05, deadline - time.monotonic()),
                )
        except asyncio.CancelledError:
            duration = round(time.monotonic() - started, 6)
            with self._lock:
                self._active_handlers.discard(key)
                self._handler_progress[key].update(
                    status="cancelled",
                    duration_seconds=duration,
                )
                report.handler_statuses[key] = "cancelled"
                report.handler_durations_seconds[key] = duration
            raise
        except Exception as exc:  # noqa: BLE001 - final teardown boundary: a failing handler must be recorded, never re-raised into shutdown
            duration = round(time.monotonic() - started, 6)
            with self._lock:
                self._active_handlers.discard(key)
                self._handler_progress[key].update(
                    status="failed",
                    duration_seconds=duration,
                    error=repr(exc),
                )
                report.handler_statuses[key] = "failed"
                report.handler_durations_seconds[key] = duration
            if isinstance(exc, TimeoutError):
                report.escalations.append(
                    {
                        "kind": "handler_timeout",
                        "phase": record.phase,
                        "handler": record.name,
                        "at_unix": time.time(),
                        "timeout_seconds": record.timeout,
                    }
                )
            raise
        else:
            duration = round(time.monotonic() - started, 6)
            with self._lock:
                self._active_handlers.discard(key)
                self._handler_progress[key].update(
                    status="completed",
                    duration_seconds=duration,
                )
                report.handler_statuses[key] = "completed"
                report.handler_durations_seconds[key] = duration


# Singleton accessor ---------------------------------------------------------

_shutdown_coordinator: ShutdownCoordinator | None = None
_singleton_lock = threading.RLock()
_shutdown_requested = threading.Event()
_shutdown_state_lock = threading.RLock()
_shutdown_first_reason = ""
_shutdown_last_reason = ""
_shutdown_first_requested_at_unix: float | None = None
_shutdown_first_requested_at_monotonic: float | None = None
_shutdown_deadline_at_unix: float | None = None
_shutdown_deadline_monotonic: float | None = None
_shutdown_deadline_source = ""
_shutdown_deadline_tighten_count = 0
_shutdown_initial_budget_seconds: float | None = None
_shutdown_request_count = 0
_shutdown_admission_counts: dict[str, int] = {}
_shutdown_admission_events: deque[dict[str, object]] = deque(maxlen=64)

_ADMISSION_OUTCOMES = frozenset(
    {
        "suppressed",
        "crossed",
        "reaped",
        "survived",
        "allowed_read_only",
        "allowed_teardown",
    }
)


def record_shutdown_admission_event(
    operation: str,
    *,
    resource_kind: str,
    outcome: str,
    detail: str = "",
) -> None:
    """Record bounded evidence for work attempted at the shutdown boundary."""

    normalized_outcome = str(outcome or "").strip().lower()
    if normalized_outcome not in _ADMISSION_OUTCOMES:
        raise ValueError(f"unknown shutdown admission outcome: {outcome}")

    def _bounded(value: object, limit: int) -> str:
        return str(value or "").replace("\r", " ").replace("\n", " ")[:limit]

    event = {
        "at_unix": time.time(),
        "operation": _bounded(operation, 240),
        "resource_kind": _bounded(resource_kind, 80),
        "outcome": normalized_outcome,
        "detail": _bounded(detail, 400),
        "thread_id": threading.get_ident(),
    }
    with _shutdown_state_lock:
        _shutdown_admission_counts[normalized_outcome] = (
            _shutdown_admission_counts.get(normalized_outcome, 0) + 1
        )
        _shutdown_admission_events.append(event)


def shutdown_admission_snapshot() -> dict[str, object]:
    with _shutdown_state_lock:
        counts = {
            outcome: int(_shutdown_admission_counts.get(outcome, 0))
            for outcome in sorted(_ADMISSION_OUTCOMES)
        }
        return {
            "total": sum(counts.values()),
            "counts": counts,
            "recent_events": [dict(item) for item in _shutdown_admission_events],
        }


def _declared_string(name: str, default: str, description: str) -> str:
    """Read a string knob through the typed flag layer (C1 discipline)."""
    try:
        from core.runtime.flags import FlagKind, declare

        return str(
            declare(
                name,
                kind=FlagKind.STRING,
                default=default,
                description=description,
                owner="core.runtime.shutdown_coordinator",
            ).value()
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return default


def _declared_bool(name: str, default: bool, description: str) -> bool:
    try:
        from core.runtime.flags import FlagKind, declare

        return bool(
            declare(
                name,
                kind=FlagKind.BOOL,
                default=default,
                description=description,
                owner="core.runtime.shutdown_coordinator",
            ).value()
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return default


def _declared_float(name: str, default: float, description: str) -> float:
    try:
        from core.runtime.flags import FlagKind, declare

        return float(
            declare(
                name,
                kind=FlagKind.FLOAT,
                default=default,
                description=description,
                owner="core.runtime.shutdown_coordinator",
            ).value()
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return default


def _shutdown_budget_seconds(*, first_request: bool) -> float:
    if first_request:
        value = _declared_float(
            "AURA_SHUTDOWN_GLOBAL_TIMEOUT_SECONDS",
            180.0,
            "Process-wide graceful shutdown deadline",
        )
    else:
        value = _declared_float(
            "AURA_SHUTDOWN_REPEAT_TIMEOUT_SECONDS",
            120.0,
            "Maximum remaining shutdown time after a repeated stop request",
        )
    return min(3600.0, max(0.05, float(value)))


def shutdown_verdict_path() -> Path:
    configured = str(_declared_string(
        "AURA_SHUTDOWN_REPORT_PATH",
        "",
        "Override path for the durable shutdown verdict report",
    ) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return Path(tempfile.gettempdir()) / f"aura-shutdown-report-pytest-{os.getpid()}.json"
    return state_root() / "run" / "shutdown_report.json"


def _shutdown_verdict_history_path(target: Path) -> Path:
    request = shutdown_request_snapshot()
    requested_at = request.get("first_requested_at_unix")
    try:
        timestamp_value = (
            requested_at
            if isinstance(requested_at, (int, float, str))
            else time.time()
        )
        timestamp_ms = int(float(timestamp_value or time.time()) * 1000)
    except (TypeError, ValueError):
        timestamp_ms = int(time.time() * 1000)
    return target.parent / "shutdown_history" / f"{timestamp_ms}-{os.getpid()}.json"


def _prune_shutdown_verdict_history(history_dir: Path, *, keep: int = 128) -> None:
    try:
        artifacts = sorted(
            history_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for artifact in artifacts[max(1, keep):]:
        try:
            delete_shutdown_artifact(artifact)
        except OSError:
            continue


def publish_shutdown_verdict(
    *,
    coordinator_report: ShutdownReport | dict[str, object] | None,
    container_report: dict[str, object] | None = None,
    runtime_hygiene_report: dict[str, object] | None = None,
    stage: str,
    final: bool,
) -> dict[str, object]:
    """Atomically publish machine-readable shutdown evidence for operators."""

    if isinstance(coordinator_report, ShutdownReport):
        coordinator_payload = coordinator_report.as_dict()
    elif isinstance(coordinator_report, dict):
        coordinator_payload = dict(coordinator_report)
    else:
        coordinator_payload = None

    blockers: list[str] = []
    if coordinator_payload is None:
        blockers.append("coordinator_report_missing")
    elif coordinator_payload.get("clean") is not True:
        blockers.append("coordinator_degraded")
    if final and container_report is None:
        blockers.append("container_report_missing")
    elif container_report is not None and container_report.get("clean") is not True:
        blockers.append("container_degraded")
    if final and runtime_hygiene_report is None:
        blockers.append("runtime_hygiene_report_missing")
    elif runtime_hygiene_report is not None and runtime_hygiene_report.get("clean") is not True:
        blockers.append("runtime_hygiene_residuals")

    admission = shutdown_admission_snapshot()
    admission_counts = admission.get("counts")
    if isinstance(admission_counts, dict) and int(admission_counts.get("survived", 0) or 0) > 0:
        blockers.append("shutdown_resurrection_survived")

    final_tasks: dict[str, object] | None = None
    if final:
        try:
            final_tasks = get_task_tracker().get_active_task_snapshot(
                exclude_current=True,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            final_tasks = {"count": -1, "error": repr(exc), "tasks": []}
            blockers.append("final_task_snapshot_unavailable")
        else:
            final_task_count = final_tasks.get("count", 0)
            # A final verdict must be CONSERVATIVE: the tracker's view is one
            # witness, the loop's own task table is another. Any pending task
            # the tracker cannot see (lost visibility, foreign creation) must
            # still block — a clean verdict over a live task is a lie. The
            # 2026-07-12 register carried exactly that shape: clean=True
            # while a lingering task existed, tracker census empty.
            loop_pending = 0
            try:
                current = asyncio.current_task()
                loop_pending = sum(
                    1
                    for task in asyncio.all_tasks()
                    if not task.done() and task is not current
                )
            except RuntimeError:
                loop_pending = 0  # no running loop: sync teardown context
            if not isinstance(final_task_count, int):
                blockers.append("final_task_snapshot_unavailable")
            else:
                effective = max(final_task_count, loop_pending)
                if isinstance(final_tasks, dict):
                    final_tasks["loop_pending"] = loop_pending
                    final_tasks["effective_count"] = effective
                if effective > 0:
                    blockers.append("tasks_remaining_after_final_sweep")
                if loop_pending > final_task_count >= 0:
                    blockers.append("untracked_pending_tasks")

    target = shutdown_verdict_path()
    history_target = _shutdown_verdict_history_path(target)
    payload: dict[str, object] = {
        "schema": "aura.shutdown_verdict.v1",
        "pid": os.getpid(),
        "generated_at_unix": time.time(),
        "stage": str(stage or "unknown"),
        "final": bool(final),
        "artifact_path": str(target),
        "history_artifact_path": str(history_target),
        "request": shutdown_request_snapshot(),
        "admission": admission,
        "components": {
            "coordinator": coordinator_payload,
            "container": dict(container_report) if container_report is not None else None,
            "runtime_hygiene": (
                dict(runtime_hygiene_report) if runtime_hygiene_report is not None else None
            ),
            "final_tasks": final_tasks,
        },
        "verdict": {
            "clean": bool(final and not blockers),
            "blockers": blockers,
        },
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    write_shutdown_artifact(history_target, serialized, encoding="utf-8")
    write_shutdown_artifact(target, serialized, encoding="utf-8")
    _prune_shutdown_verdict_history(history_target.parent)
    return payload


def publish_root_exit_verdict(
    *,
    lock_released: bool,
    finalizers_completed: bool,
    logging_shutdown_completed: bool,
    root_resource_report: dict[str, object],
    exit_code: int,
) -> dict[str, object]:
    """Publish the terminal receipt after root-owned process cleanup.

    ``graceful_shutdown_complete`` proves service/container teardown. The root
    launcher still owns the singleton lock and multiprocessing finalizers at
    that point, so it is not the terminal process receipt. This update reuses
    the same history identity and atomically advances it exactly once to the
    root-exit boundary.
    """

    target = shutdown_verdict_path()
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        loaded = {}
    payload = dict(loaded) if isinstance(loaded, dict) else {}
    if payload.get("stage") == "root_process_exit":
        raise RuntimeError("terminal root-exit receipt was already published")

    blockers: list[str] = []
    prior_verdict = payload.get("verdict")
    if not isinstance(prior_verdict, dict) or prior_verdict.get("clean") is not True:
        blockers.append("prior_shutdown_verdict_not_clean")
        if isinstance(prior_verdict, dict):
            prior_blockers = prior_verdict.get("blockers")
            if isinstance(prior_blockers, list):
                blockers.extend(str(item) for item in prior_blockers if str(item))
    if payload.get("final") is not True:
        blockers.append("prior_final_shutdown_receipt_missing")
    if not lock_released:
        blockers.append("singleton_lock_release_failed")
    if not finalizers_completed:
        blockers.append("multiprocessing_finalizers_incomplete")
    if not logging_shutdown_completed:
        blockers.append("logging_shutdown_incomplete")
    if root_resource_report.get("clean") is not True:
        blockers.append("root_process_resources_remaining")
        resource_blockers = root_resource_report.get("blockers")
        if isinstance(resource_blockers, list):
            blockers.extend(str(item) for item in resource_blockers if str(item))
    if int(exit_code) != 0:
        blockers.append(f"nonzero_exit_code:{int(exit_code)}")

    request = shutdown_request_snapshot()
    admission = shutdown_admission_snapshot()
    admission_counts = admission.get("counts")
    if isinstance(admission_counts, dict) and int(admission_counts.get("survived", 0) or 0) > 0:
        blockers.append("shutdown_resurrection_survived")

    unique_blockers = list(dict.fromkeys(blockers))
    history_target = _shutdown_verdict_history_path(target)
    first_requested = request.get("first_requested_at_unix")
    try:
        timestamp_value = (
            first_requested
            if isinstance(first_requested, (int, float, str))
            else time.time()
        )
        receipt_timestamp_ms = int(float(timestamp_value or time.time()) * 1000)
    except (TypeError, ValueError, OverflowError):
        receipt_timestamp_ms = int(time.time() * 1000)

    payload.update(
        {
            "schema": "aura.shutdown_verdict.v1",
            "pid": os.getpid(),
            "generated_at_unix": time.time(),
            "stage": "root_process_exit",
            "final": True,
            "artifact_path": str(target),
            "history_artifact_path": str(history_target),
            "terminal_receipt_id": f"{receipt_timestamp_ms}-{os.getpid()}",
            "terminal_receipt_sequence": 1,
            "request": request,
            "admission": admission,
            "root_exit": {
                "lock_released": bool(lock_released),
                "multiprocessing_finalizers_completed": bool(finalizers_completed),
                "logging_shutdown_completed": bool(logging_shutdown_completed),
                "exit_code": int(exit_code),
                "completed_at_unix": time.time(),
                "resources": dict(root_resource_report),
            },
            "verdict": {
                "clean": not unique_blockers,
                "blockers": unique_blockers,
            },
        }
    )
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    write_shutdown_artifact(history_target, serialized, encoding="utf-8")
    write_shutdown_artifact(target, serialized, encoding="utf-8")
    _prune_shutdown_verdict_history(history_target.parent)
    return payload


def shutdown_request_snapshot() -> dict[str, object]:
    """Return process-wide latch metadata for health and stop diagnostics."""

    with _shutdown_state_lock:
        requested = _shutdown_requested.is_set()
        now_monotonic = time.monotonic()
        elapsed = None
        if requested and _shutdown_first_requested_at_monotonic is not None:
            elapsed = max(0.0, now_monotonic - _shutdown_first_requested_at_monotonic)
        remaining = None
        if requested and _shutdown_deadline_monotonic is not None:
            remaining = max(0.0, _shutdown_deadline_monotonic - now_monotonic)
        return {
            "requested": requested,
            "first_reason": _shutdown_first_reason,
            "last_reason": _shutdown_last_reason,
            "first_requested_at_unix": _shutdown_first_requested_at_unix,
            "elapsed_seconds": round(elapsed, 6) if elapsed is not None else None,
            "deadline_at_unix": _shutdown_deadline_at_unix,
            "deadline_source": _shutdown_deadline_source,
            "initial_budget_seconds": _shutdown_initial_budget_seconds,
            "remaining_budget_seconds": (
                round(remaining, 6) if remaining is not None else None
            ),
            "deadline_exhausted": bool(remaining is not None and remaining <= 0.0),
            "deadline_tighten_count": _shutdown_deadline_tighten_count,
            "request_count": _shutdown_request_count,
        }


def shutdown_deadline_monotonic() -> float | None:
    """Return the immutable-or-tightened process deadline for internal fuses."""

    with _shutdown_state_lock:
        return _shutdown_deadline_monotonic


def shutdown_remaining_budget_seconds() -> float | None:
    """Return remaining process-wide teardown budget, or ``None`` before stop."""

    with _shutdown_state_lock:
        if not _shutdown_requested.is_set() or _shutdown_deadline_monotonic is None:
            return None
        return max(0.0, _shutdown_deadline_monotonic - time.monotonic())


def _write_grace_flag(*, reason: str, created_at_unix: float) -> None:
    from pathlib import Path

    grace_file = state_root() / "run" / "grace_exit.flag"

    # A test process must never drop a shutdown-grace flag into the REAL
    # ~/.aura/run: the live desktop runtime reads that directory, and a leaked
    # flag (observed 2026-07-25: reason="unit_test", pid long dead) becomes a
    # confusing artifact in the user's runtime state. Tests that genuinely
    # exercise this path redirect HOME to a tmp dir; if HOME still points at the
    # real home while pytest is running, refuse rather than pollute.
    if "PYTEST_CURRENT_TEST" in os.environ:
        # The REAL home comes from the passwd entry, which $HOME cannot spoof —
        # a test that correctly redirects HOME to a tmp dir writes outside it and
        # is allowed through; only a test writing into the actual user's
        # ~/.aura is refused.
        try:
            import pwd

            real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
        except (ImportError, KeyError, OSError):
            real_home = None
        try:
            writing_into_real_home = bool(
                real_home is not None
                and grace_file.resolve().is_relative_to(real_home / ".aura")
            )
        except (OSError, ValueError):
            writing_into_real_home = False
        if writing_into_real_home and os.environ.get("AURA_ALLOW_LIVE_RUNTIME_WRITES") != "1":
            logger.warning(
                "Refusing to write a shutdown-grace flag into the live runtime "
                "directory from a test process (reason=%s).", reason,
            )
            return
    write_shutdown_artifact(
        grace_file,
        json.dumps(
            {
                "schema": "aura.shutdown_grace.v1",
                "pid": os.getpid(),
                "reason": reason,
                "created_at_unix": created_at_unix,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _requesting_frames(depth: int = 6) -> str:
    """The nearest callers outside this module, innermost first."""
    import traceback

    names: list[str] = []
    for frame in reversed(traceback.extract_stack()[:-2]):
        if frame.filename.endswith("shutdown_coordinator.py"):
            continue
        names.append(f"{os.path.basename(frame.filename)}:{frame.lineno} {frame.name}")
        if len(names) >= depth:
            break
    return " <- ".join(names) or "unknown"


def request_shutdown(
    reason: str = "",
    *,
    deadline_seconds: float | None = None,
) -> dict[str, object]:
    """Latch process shutdown before logging or filesystem side effects.

    The event is the mechanical no-new-work fence. Metadata and the grace flag
    are useful evidence, but neither is allowed to delay the fence becoming
    visible to task, subprocess, or model admission paths.
    """

    global _shutdown_first_reason
    global _shutdown_last_reason
    global _shutdown_first_requested_at_unix
    global _shutdown_first_requested_at_monotonic
    global _shutdown_deadline_at_unix
    global _shutdown_deadline_monotonic
    global _shutdown_deadline_source
    global _shutdown_deadline_tighten_count
    global _shutdown_initial_budget_seconds
    global _shutdown_request_count

    normalized_reason = str(reason or "")
    now_unix = time.time()
    now_monotonic = time.monotonic()
    with _shutdown_state_lock:
        first_request = not _shutdown_requested.is_set()
        requested_budget = (
            _shutdown_budget_seconds(first_request=first_request)
            if deadline_seconds is None
            else min(3600.0, max(0.05, float(deadline_seconds)))
        )
        candidate_deadline_monotonic = now_monotonic + requested_budget
        candidate_deadline_at_unix = now_unix + requested_budget
        if first_request:
            _shutdown_requested.set()
            _shutdown_first_reason = normalized_reason
            _shutdown_first_requested_at_unix = now_unix
            _shutdown_first_requested_at_monotonic = now_monotonic
            _shutdown_deadline_monotonic = candidate_deadline_monotonic
            _shutdown_deadline_at_unix = candidate_deadline_at_unix
            _shutdown_deadline_source = normalized_reason
            _shutdown_initial_budget_seconds = requested_budget
            _shutdown_deadline_tighten_count = 0
            _shutdown_request_count = 1
        else:
            _shutdown_request_count += 1
            if (
                _shutdown_deadline_monotonic is None
                or candidate_deadline_monotonic < _shutdown_deadline_monotonic
            ):
                _shutdown_deadline_monotonic = candidate_deadline_monotonic
                _shutdown_deadline_at_unix = candidate_deadline_at_unix
                _shutdown_deadline_source = normalized_reason
                _shutdown_deadline_tighten_count += 1
        _shutdown_last_reason = normalized_reason
        snapshot = shutdown_request_snapshot()

    if first_request:
        # The first request ends the process, and the reason string names a
        # path, not a caller: "orchestrator_shutdown" ended a boot 89s in on
        # 2026-09-16 and nothing said who called stop(). The caller chain is
        # part of the record.
        logger.info(
            "Shutdown requested%s. Requested from: %s",
            f": {normalized_reason}" if normalized_reason else "",
            _requesting_frames(),
        )
        try:
            # The flag is evidence, written with an fsync; the fence above
            # is already visible. Behind the loop when a coroutine asked
            # (LIVE 2026-09-19: named from the GUI reaper), inline elsewhere.
            from core.runtime.executors import behind_the_loop

            behind_the_loop(
                "shutdown_coordinator.grace_flag",
                lambda: _write_grace_flag(reason=normalized_reason, created_at_unix=now_unix),
            )
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            logger.debug(
                "Suppressed %s in core.runtime.shutdown_coordinator: %s",
                type(exc).__name__,
                exc,
            )
    return snapshot


def is_shutdown_requested() -> bool:
    return _shutdown_requested.is_set()


def _shutdown_reset_allowed() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return bool(
        _declared_bool(
            "AURA_ALLOW_IN_PROCESS_SHUTDOWN_RESET",
            False,
            "Permit clearing the monotonic shutdown latch in-process (tests/operators only)",
        )
    )


def clear_shutdown_request() -> None:
    """Clear test state; production shutdown is monotonic within one process."""
    global _shutdown_first_reason
    global _shutdown_last_reason
    global _shutdown_first_requested_at_unix
    global _shutdown_first_requested_at_monotonic
    global _shutdown_deadline_at_unix
    global _shutdown_deadline_monotonic
    global _shutdown_deadline_source
    global _shutdown_deadline_tighten_count
    global _shutdown_initial_budget_seconds
    global _shutdown_request_count

    if _shutdown_requested.is_set() and not _shutdown_reset_allowed():
        raise RuntimeError(
            "production shutdown latch is monotonic; restart Aura in a new process"
        )
    with _shutdown_state_lock:
        _shutdown_requested.clear()
        _shutdown_first_reason = ""
        _shutdown_last_reason = ""
        _shutdown_first_requested_at_unix = None
        _shutdown_first_requested_at_monotonic = None
        _shutdown_deadline_at_unix = None
        _shutdown_deadline_monotonic = None
        _shutdown_deadline_source = ""
        _shutdown_deadline_tighten_count = 0
        _shutdown_initial_budget_seconds = None
        _shutdown_request_count = 0
        _shutdown_admission_counts.clear()
        _shutdown_admission_events.clear()


def get_shutdown_coordinator() -> ShutdownCoordinator:
    global _shutdown_coordinator
    with _singleton_lock:
        if _shutdown_coordinator is None:
            _shutdown_coordinator = ShutdownCoordinator()
        return _shutdown_coordinator


def reset_shutdown_coordinator() -> None:
    """Test helper. Drops the singleton so a fresh instance is created."""
    global _shutdown_coordinator
    if not _shutdown_reset_allowed():
        raise RuntimeError("shutdown coordinator reset is test-only")
    with _singleton_lock:
        _shutdown_coordinator = None
    clear_shutdown_request()
