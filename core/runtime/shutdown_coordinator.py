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

from core.utils.task_tracker import get_task_tracker

import asyncio
import inspect
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("Aura.ShutdownCoordinator")

ShutdownHandler = Callable[[], Union[None, Awaitable[None]]]


# Canonical phases. Order matters.
SHUTDOWN_PHASES: Tuple[str, ...] = (
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
    completed_phases: List[str] = field(default_factory=list)
    failed_phases: List[str] = field(default_factory=list)
    handler_failures: Dict[str, str] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.failed_phases and not self.handler_failures


class ShutdownCoordinator:
    """Single owner of teardown ordering."""

    def __init__(self, phases: Tuple[str, ...] = SHUTDOWN_PHASES):
        self._phases = phases
        self._handlers: Dict[str, List[_RegisteredHandler]] = {p: [] for p in phases}
        self._lock = threading.RLock()
        self._running = False

    # --- Registration ---------------------------------------------------

    def register(
        self,
        handler: ShutdownHandler,
        *,
        phase: str,
        name: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        if phase not in self._handlers:
            raise ValueError(
                f"unknown shutdown phase '{phase}'; expected one of {self._phases}"
            )
        if not callable(handler):
            raise TypeError("shutdown handler must be callable")
        record = _RegisteredHandler(
            name=name or getattr(handler, "__name__", "anonymous"),
            handler=handler,
            phase=phase,
            timeout=timeout,
        )
        with self._lock:
            self._handlers[phase].append(record)

    def clear(self) -> None:
        with self._lock:
            for phase in self._handlers:
                self._handlers[phase] = []

    def phases(self) -> Tuple[str, ...]:
        return self._phases

    def handler_names(self, phase: str) -> List[str]:
        with self._lock:
            return [h.name for h in self._handlers.get(phase, [])]

    # --- Execution ------------------------------------------------------

    async def shutdown(self, *, timeout_per_phase: Optional[float] = None) -> ShutdownReport:
        report = ShutdownReport()
        if self._running:
            logger.warning("ShutdownCoordinator.shutdown() invoked re-entrantly")
        self._running = True
        try:
            for phase in self._phases:
                with self._lock:
                    handlers = list(self._handlers.get(phase, []))
                if not handlers:
                    report.completed_phases.append(phase)
                    continue
                phase_failed = False
                coros: List[asyncio.Future] = []
                for record in handlers:
                    coros.append(
                        get_task_tracker().track(self._invoke(record))
                    )
                effective_timeout = timeout_per_phase or max(
                    (h.timeout for h in handlers), default=15.0
                )
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*coros, return_exceptions=True),
                        timeout=effective_timeout,
                    )
                except asyncio.TimeoutError:
                    for fut in coros:
                        if not fut.done():
                            fut.cancel()
                    report.failed_phases.append(phase)
                    report.handler_failures[phase] = "phase timed out"
                    logger.error("Shutdown phase '%s' timed out", phase)
                    continue
                for record, result in zip(handlers, results):
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
                    report.failed_phases.append(phase)
                else:
                    report.completed_phases.append(phase)
        finally:
            self._running = False

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

    async def _invoke(self, record: _RegisteredHandler) -> None:
        try:
            result = record.handler()
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=record.timeout)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise


# Singleton accessor ---------------------------------------------------------

_shutdown_coordinator: Optional[ShutdownCoordinator] = None
_singleton_lock = threading.RLock()


def get_shutdown_coordinator() -> ShutdownCoordinator:
    global _shutdown_coordinator
    with _singleton_lock:
        if _shutdown_coordinator is None:
            _shutdown_coordinator = ShutdownCoordinator()
        return _shutdown_coordinator


def reset_shutdown_coordinator() -> None:
    """Test helper. Drops the singleton so a fresh instance is created."""
    global _shutdown_coordinator
    with _singleton_lock:
        _shutdown_coordinator = None
