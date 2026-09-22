"""core/fsw/health_checker.py — active liveness pings.

Clean-room adoption of F Prime's `Svc::Health` component.

Every health mechanism Aura has is *passive*: it reads a value the
component published, or a heartbeat the component remembered to send. That
works right up until the component stops running, at which point the last
published value sits there looking fine, and the heartbeat's absence is
only noticed by something that thought to look for absence.

F Prime's health component is active. On a rate group it *pings* every
registered component and requires an answer within a declared timeout. A
component that does not answer is not "quiet" — it is unresponsive, which
is a fact the system establishes rather than infers. The distinction
matters because the two have completely different causes: a quiet
component may be idle by design, while an unresponsive one is wedged.

The mechanism also separates two failure shapes that Aura's incident
history keeps conflating:

* **Slow** — answered, but past its budget. Something is contended.
* **Unresponsive** — did not answer at all, N times running. Something is
  wedged, and the mind_tick false-death incidents are exactly the cost of
  not distinguishing these two.

Escalation is declared per component and happens after a declared number
of consecutive misses, not on the first one — because a single missed ping
under load is normal and treating it as death is what caused a duplicate
32B to be spawned beside a wedged one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.HealthChecker")

#: A ping that takes longer than its timeout counts as a miss.
DEFAULT_PING_TIMEOUT_S = 2.0
#: Consecutive misses before a component is declared unresponsive. One
#: missed ping under load is normal; treating it as death is how a
#: duplicate runtime gets spawned beside a wedged one.
DEFAULT_MISS_THRESHOLD = 3


class Liveness(StrEnum):
    RESPONSIVE = "responsive"
    SLOW = "slow"
    UNRESPONSIVE = "unresponsive"
    #: Registered but never pinged yet.
    UNKNOWN = "unknown"


@dataclass
class Watched:
    name: str
    ping: Callable[[], Any]
    timeout_s: float = DEFAULT_PING_TIMEOUT_S
    miss_threshold: int = DEFAULT_MISS_THRESHOLD
    #: Called once when the component is declared unresponsive.
    on_unresponsive: Callable[[str], Any] | None = None
    #: Called once when it answers again after being unresponsive.
    on_recovered: Callable[[str], Any] | None = None
    critical: bool = False

    state: Liveness = Liveness.UNKNOWN
    consecutive_misses: int = 0
    pings: int = 0
    misses: int = 0
    last_answer_at: float = 0.0
    last_latency_s: float = 0.0
    max_latency_s: float = 0.0
    declared_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": str(self.state),
            "critical": self.critical,
            "timeout_ms": round(self.timeout_s * 1000, 1),
            "miss_threshold": self.miss_threshold,
            "consecutive_misses": self.consecutive_misses,
            "pings": self.pings,
            "misses": self.misses,
            "miss_rate": round(self.misses / self.pings, 4) if self.pings else 0.0,
            "last_latency_ms": round(self.last_latency_s * 1000, 2),
            "max_latency_ms": round(self.max_latency_s * 1000, 2),
            "silent_for_s": (
                round(time.time() - self.last_answer_at, 1) if self.last_answer_at else None
            ),
        }


class HealthChecker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._watched: dict[str, Watched] = {}
        self.rounds = 0

    def watch(
        self,
        name: str,
        ping: Callable[[], Any],
        *,
        timeout_s: float = DEFAULT_PING_TIMEOUT_S,
        miss_threshold: int = DEFAULT_MISS_THRESHOLD,
        on_unresponsive: Callable[[str], Any] | None = None,
        on_recovered: Callable[[str], Any] | None = None,
        critical: bool = False,
    ) -> Watched:
        """Register a component to be actively pinged.

        The ping should be cheap and should touch something that only
        works if the component is genuinely alive — reading a cached
        field proves nothing.
        """
        with self._lock:
            entry = Watched(
                name=name,
                ping=ping,
                timeout_s=timeout_s,
                miss_threshold=miss_threshold,
                on_unresponsive=on_unresponsive,
                on_recovered=on_recovered,
                critical=critical,
            )
            self._watched[name] = entry
            return entry

    def unwatch(self, name: str) -> None:
        with self._lock:
            self._watched.pop(name, None)

    def watched(self) -> list[Watched]:
        with self._lock:
            return list(self._watched.values())

    async def _ping_one(self, entry: Watched) -> None:
        started = time.monotonic()
        answered = False
        try:
            outcome = entry.ping()
            if asyncio.iscoroutine(outcome):
                await asyncio.wait_for(outcome, timeout=entry.timeout_s)
            answered = outcome is not False
        # not a failure: the wait ran out, which is what the timeout was set to decide.
        except TimeoutError:
            answered = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — an exception is a failed ping
            logger.debug("health ping %s raised: %s", entry.name, exc)
            answered = False

        latency = time.monotonic() - started
        entry.pings += 1
        entry.last_latency_s = latency
        entry.max_latency_s = max(entry.max_latency_s, latency)

        if answered and latency <= entry.timeout_s:
            entry.last_answer_at = time.time()
            previous = entry.state
            entry.consecutive_misses = 0
            entry.state = Liveness.RESPONSIVE
            if previous is Liveness.UNRESPONSIVE:
                self._announce_recovery(entry)
            return

        if answered:
            # Answered, but late. Contended, not wedged — and the
            # difference is the whole point of measuring both.
            entry.last_answer_at = time.time()
            entry.consecutive_misses = 0
            entry.state = Liveness.SLOW
            logger.info(
                "health: %s answered in %.0fms, past its %.0fms budget",
                entry.name,
                latency * 1000,
                entry.timeout_s * 1000,
            )
            return

        entry.misses += 1
        entry.consecutive_misses += 1
        if entry.consecutive_misses >= entry.miss_threshold and entry.state is not Liveness.UNRESPONSIVE:
            entry.state = Liveness.UNRESPONSIVE
            entry.declared_at = time.time()
            self._announce_unresponsive(entry)

    def _announce_unresponsive(self, entry: Watched) -> None:
        logger.error(
            "💀 health: %s is UNRESPONSIVE — %d consecutive missed pings "
            "(last answered %s)",
            entry.name,
            entry.consecutive_misses,
            f"{time.time() - entry.last_answer_at:.0f}s ago" if entry.last_answer_at else "never",
        )
        try:
            from core.fsw.telemetry_dictionary import EventSeverity, emit_event

            emit_event(
                "component_unresponsive",
                severity=EventSeverity.FATAL if entry.critical else EventSeverity.WARNING_HI,
                component=entry.name,
                misses=entry.consecutive_misses,
                critical=entry.critical,
            )
        except Exception:  # noqa: BLE001
            logger.debug("unresponsive telemetry failed", exc_info=True)
        if entry.critical:
            from core.runtime.taint import TaintFlag, taint

            taint(
                TaintFlag.CRASHED_ORGAN,
                f"critical component {entry.name} stopped answering health pings",
                subsystem="health_checker",
            )
        if entry.on_unresponsive is not None:
            with contextlib.suppress(Exception):
                entry.on_unresponsive(entry.name)

    def _announce_recovery(self, entry: Watched) -> None:
        logger.info(
            "🫀 health: %s is answering again after %.0fs unresponsive",
            entry.name,
            time.time() - entry.declared_at if entry.declared_at else 0.0,
        )
        try:
            from core.fsw.telemetry_dictionary import EventSeverity, emit_event

            emit_event(
                "component_recovered",
                severity=EventSeverity.ACTIVITY_HI,
                component=entry.name,
                unresponsive_for_s=round(time.time() - entry.declared_at, 1)
                if entry.declared_at
                else 0.0,
            )
        except Exception:  # noqa: BLE001
            logger.debug("recovery telemetry failed", exc_info=True)
        if entry.on_recovered is not None:
            with contextlib.suppress(Exception):
                entry.on_recovered(entry.name)

    async def run_round(self) -> dict[str, Any]:
        """One ping to every watched component. Suitable as a rate-group member."""
        entries = self.watched()
        for entry in entries:
            await self._ping_one(entry)
        self.rounds += 1
        return self.report()

    def report(self) -> dict[str, Any]:
        entries = self.watched()
        return {
            "rounds": self.rounds,
            "watched": len(entries),
            "responsive": [e.name for e in entries if e.state is Liveness.RESPONSIVE],
            "slow": [e.name for e in entries if e.state is Liveness.SLOW],
            "unresponsive": [e.name for e in entries if e.state is Liveness.UNRESPONSIVE],
            "unknown": [e.name for e in entries if e.state is Liveness.UNKNOWN],
            "critical_unresponsive": [
                e.name for e in entries if e.critical and e.state is Liveness.UNRESPONSIVE
            ],
            "components": [e.to_dict() for e in entries],
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._watched.clear()
            self.rounds = 0


_CHECKER = HealthChecker()


def get_health_checker() -> HealthChecker:
    return _CHECKER


def watch(
    name: str,
    ping: Callable[[], Any],
    *,
    timeout_s: float = DEFAULT_PING_TIMEOUT_S,
    miss_threshold: int = DEFAULT_MISS_THRESHOLD,
    critical: bool = False,
    on_unresponsive: Callable[[str], Any] | None = None,
    on_recovered: Callable[[str], Any] | None = None,
) -> Watched:
    return _CHECKER.watch(
        name,
        ping,
        timeout_s=timeout_s,
        miss_threshold=miss_threshold,
        critical=critical,
        on_unresponsive=on_unresponsive,
        on_recovered=on_recovered,
    )


def install_runtime_pings() -> list[str]:
    """Ping the disciplines that must keep answering to be worth anything.

    Each ping does real work — reading a report that requires the
    subsystem's locks and state — rather than checking a cached flag,
    because a cached flag proves only that something once set it.
    """

    def ping_event_bus() -> bool:
        from core.event_bus import get_event_bus

        return bool(get_event_bus().is_alive())

    def ping_telemetry() -> bool:
        from core.fsw.telemetry_dictionary import get_telemetry

        return get_telemetry().report() is not None

    def ping_lockdep() -> bool:
        from core.runtime.lockdep import lockdep_report

        return lockdep_report() is not None

    def ping_pressure() -> bool:
        from core.runtime.pressure_stall import psi_report

        return psi_report() is not None

    def ping_controllers() -> bool:
        from core.runtime.reconcile import reconcile_report

        return reconcile_report() is not None

    def ping_diagnostics() -> bool:
        from core.health.diagnostics_aggregator import get_aggregator

        return get_aggregator().aggregate() is not None

    watch("event_bus", ping_event_bus, critical=True)
    watch("telemetry", ping_telemetry)
    watch("lockdep", ping_lockdep)
    watch("pressure_stall", ping_pressure)
    watch("controllers", ping_controllers)
    watch("diagnostics", ping_diagnostics)
    return [e.name for e in _CHECKER.watched()]


def health_checker_report() -> dict[str, Any]:
    return _CHECKER.report()


def reset_health_checker_for_test() -> None:
    _CHECKER.reset_for_test()


__all__ = [
    "DEFAULT_MISS_THRESHOLD",
    "DEFAULT_PING_TIMEOUT_S",
    "HealthChecker",
    "Liveness",
    "Watched",
    "get_health_checker",
    "health_checker_report",
    "install_runtime_pings",
    "reset_health_checker_for_test",
    "watch",
]
