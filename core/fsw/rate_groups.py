"""core/fsw/rate_groups.py — deterministic periodic execution.

Clean-room adoption of F Prime's rate groups and cycle-slip detection.

Flight software does not schedule periodic work with a pile of
independent timers. It has rate groups — 1Hz, 5Hz, 10Hz — each driven by
one tick, each running its members in a declared order, each *measuring*
whether it finished before the next tick was due. That last part is the
one that matters: a rate group knows when it slipped, by how much, and
which member ate the budget.

Aura's periodic work is currently a set of independent `asyncio` loops
each doing `await asyncio.sleep(interval)`. That pattern has three
properties nobody chose:

* The interval is between *finishing* and *starting again*, so a loop that
  takes 4s with a 5s sleep actually runs every 9s and nothing says so.
* Nothing knows the loop is late. It just is.
* Every loop is its own task, so ten periodic jobs are ten wakeups with
  no shared budget, and under load they all slip together with no
  ordering.

A rate group fixes all three. It ticks on a fixed schedule (so the period
is a period), it measures overrun and reports cycle slips, and members run
in declared order under one budget, so the slow one is identifiable rather
than merely suspected.

Members declare a budget. Exceeding it is reported per-member, which turns
"the 1Hz group is slipping" into "the 1Hz group is slipping because
vector_index took 800ms of its 1000ms".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.RateGroups")

#: A cycle that runs this far past its period is a slip, not jitter.
SLIP_TOLERANCE = 0.10
#: While a group keeps slipping, one line per this many cycles.
_SAY_STREAK_EVERY = 30


@dataclass
class Member:
    name: str
    fn: Callable[[], Any]
    #: Fraction of the group's period this member may use.
    budget_fraction: float = 0.25
    order: int = 100
    runs: int = 0
    failures: int = 0
    overruns: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    last_error: str = ""
    enabled: bool = True

    def budget_s(self, period_s: float) -> float:
        return period_s * self.budget_fraction

    def to_dict(self, period_s: float) -> dict[str, Any]:
        return {
            "name": self.name,
            "order": self.order,
            "budget_ms": round(self.budget_s(period_s) * 1000, 1),
            "runs": self.runs,
            "failures": self.failures,
            "overruns": self.overruns,
            "mean_ms": round((self.total_s / self.runs) * 1000, 2) if self.runs else 0.0,
            "max_ms": round(self.max_s * 1000, 2),
            "last_error": self.last_error,
            "enabled": self.enabled,
        }


@dataclass
class CycleRecord:
    cycle: int
    started_at: float
    duration_s: float
    slipped: bool
    late_by_s: float
    slowest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_s * 1000, 2),
            "slipped": self.slipped,
            "late_by_ms": round(self.late_by_s * 1000, 2),
            "slowest": self.slowest,
        }


class RateGroup:
    """One period, one tick, members in declared order, measured."""

    def __init__(self, name: str, period_s: float) -> None:
        if period_s <= 0:
            raise ValueError("a rate group needs a positive period")
        self.name = name
        self.period_s = period_s
        self._lock = threading.Lock()
        self._members: dict[str, Member] = {}
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._history: list[CycleRecord] = []
        self.cycles = 0
        self.slips = 0
        self.consecutive_slips = 0
        self._next_due = 0.0

    def add(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        budget_fraction: float = 0.25,
        order: int = 100,
    ) -> Member:
        with self._lock:
            member = Member(name=name, fn=fn, budget_fraction=budget_fraction, order=order)
            self._members[name] = member
            return member

    def remove(self, name: str) -> None:
        with self._lock:
            self._members.pop(name, None)

    def members(self) -> list[Member]:
        with self._lock:
            return sorted(self._members.values(), key=lambda m: (m.order, m.name))

    # ── the cycle ─────────────────────────────────────────────────────
    async def run_cycle(self) -> CycleRecord:
        """Run every member once, in order, measuring each."""
        started_monotonic = time.monotonic()
        started_wall = time.time()
        slowest = ""
        slowest_s = 0.0

        for member in self.members():
            if not member.enabled:
                continue
            member_started = time.monotonic()
            try:
                outcome = member.fn()
                if asyncio.iscoroutine(outcome):
                    await outcome
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — one member must not stop the group
                member.failures += 1
                member.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "rate group %s: member %s failed: %s", self.name, member.name, exc
                )
                from core.runtime.errors import record_degradation

                with contextlib.suppress(Exception):
                    record_degradation(
                        f"rate_group.{self.name}",
                        exc,
                        severity="warning",
                        action=f"member {member.name} skipped this cycle",
                        enforce_failure_policy=False,
                    )
            elapsed = time.monotonic() - member_started
            member.runs += 1
            member.total_s += elapsed
            member.max_s = max(member.max_s, elapsed)
            if elapsed > member.budget_s(self.period_s):
                member.overruns += 1
            if elapsed > slowest_s:
                slowest_s = elapsed
                slowest = member.name

        duration = time.monotonic() - started_monotonic
        late_by = max(0.0, duration - self.period_s)
        slipped = duration > self.period_s * (1.0 + SLIP_TOLERANCE)

        self.cycles += 1
        streak_ended = 0
        if slipped:
            self.slips += 1
            self.consecutive_slips += 1
        else:
            streak_ended = self.consecutive_slips
            self.consecutive_slips = 0

        record = CycleRecord(
            cycle=self.cycles,
            started_at=started_wall,
            duration_s=duration,
            slipped=slipped,
            late_by_s=late_by,
            slowest=slowest,
        )
        with self._lock:
            self._history.append(record)
            if len(self._history) > 128:
                del self._history[:-128]

        if slipped:
            # A slip is said when it begins and while it lasts it is said
            # every _SAY_STREAK_EVERY cycles, not every cycle: on a loaded
            # host the 1Hz group slipped 99 cycles in half an hour and each
            # was a warning line (2026-09-16). The streak's end is said too.
            first = self.consecutive_slips == 1
            say = logger.warning if first or self.consecutive_slips % _SAY_STREAK_EVERY == 0 else logger.info
            say(
                "⏱️ rate group %s slipped: cycle took %.0fms of a %.0fms period "
                "(slowest member: %s at %.0fms)%s",
                self.name,
                duration * 1000,
                self.period_s * 1000,
                slowest or "unknown",
                slowest_s * 1000,
                "" if first else f"; {self.consecutive_slips} in a row",
            )
            self._announce_slip(record, slowest_s)
        elif streak_ended:
            logger.info(
                "⏱️ rate group %s back on period after %d slipped cycle(s)",
                self.name,
                streak_ended,
            )
        return record

    def _announce_slip(self, record: CycleRecord, slowest_s: float) -> None:
        try:
            from core.fsw.telemetry_dictionary import EventSeverity, emit_event

            emit_event(
                "rate_group_slip",
                severity=(
                    EventSeverity.WARNING_HI
                    if self.consecutive_slips >= 3
                    else EventSeverity.WARNING_LO
                ),
                # The event record carries every slip; the feed line does not.
                quiet=self.consecutive_slips != 1 and self.consecutive_slips % _SAY_STREAK_EVERY != 0,
                group=self.name,
                period_ms=round(self.period_s * 1000, 1),
                duration_ms=round(record.duration_s * 1000, 1),
                slowest=record.slowest,
                slowest_ms=round(slowest_s * 1000, 1),
                consecutive=self.consecutive_slips,
            )
        except Exception:  # noqa: BLE001
            logger.debug("slip telemetry failed", exc_info=True)

        # Sustained slipping is an overload, and overload has a declared
        # response rather than an implicit one.
        if self.consecutive_slips == 5:
            try:
                from core.fsw.restart_protection import get_restart_protection

                get_restart_protection().overload(
                    reason=(
                        f"rate group {self.name} has slipped {self.consecutive_slips} "
                        f"cycles in a row; {record.slowest} is over budget"
                    )
                )
            except Exception:  # noqa: BLE001
                logger.debug("overload escalation failed", exc_info=True)

    # ── lifecycle ─────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping.clear()
        self._next_due = time.monotonic()
        self._task = get_task_tracker().create_task(
            self._run(),
            name=f"rate_group.{self.name}",
        )

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001 — the group never dies quietly
                logger.exception("rate group %s cycle raised", self.name)

            # Schedule from the DUE time, not from now: the period is a
            # period, not a gap between finishing and starting again.
            self._next_due += self.period_s
            delay = self._next_due - time.monotonic()
            if delay < 0:
                # Already late. Do not try to catch up by running back to
                # back — that turns a slip into a stampede. Skip ahead.
                missed = int(-delay // self.period_s) + 1
                self._next_due += missed * self.period_s
                delay = max(0.0, self._next_due - time.monotonic())
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)
            except TimeoutError:
                continue

    async def stop(self) -> None:
        self._stopping.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def report(self) -> dict[str, Any]:
        with self._lock:
            history = list(self._history)
        durations = sorted(r.duration_s for r in history)
        return {
            "name": self.name,
            "period_ms": round(self.period_s * 1000, 1),
            "running": self._task is not None and not self._task.done(),
            "cycles": self.cycles,
            "slips": self.slips,
            "slip_rate": round(self.slips / self.cycles, 4) if self.cycles else 0.0,
            "consecutive_slips": self.consecutive_slips,
            "p50_ms": round(durations[len(durations) // 2] * 1000, 2) if durations else 0.0,
            "max_ms": round(max(durations) * 1000, 2) if durations else 0.0,
            "members": [m.to_dict(self.period_s) for m in self.members()],
            "over_budget": [
                m.name for m in self.members() if m.overruns and m.runs and m.overruns / m.runs > 0.1
            ],
            "recent_slips": [r.to_dict() for r in history if r.slipped][-4:],
        }


class RateGroupScheduler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._groups: dict[str, RateGroup] = {}

    def group(self, name: str, period_s: float) -> RateGroup:
        with self._lock:
            existing = self._groups.get(name)
            if existing is not None:
                return existing
            group = RateGroup(name, period_s)
            self._groups[name] = group
            return group

    def groups(self) -> list[RateGroup]:
        with self._lock:
            return sorted(self._groups.values(), key=lambda g: g.period_s)

    async def start_all(self) -> list[str]:
        started: list[str] = []
        for group in self.groups():
            with contextlib.suppress(Exception):
                await group.start()
                started.append(group.name)
        return started

    async def stop_all(self) -> None:
        for group in self.groups():
            with contextlib.suppress(Exception):
                await group.stop()

    def report(self) -> dict[str, Any]:
        groups = [g.report() for g in self.groups()]
        return {
            "count": len(groups),
            "groups": groups,
            "slipping": [g["name"] for g in groups if g["consecutive_slips"] > 0],
            "total_cycles": sum(g["cycles"] for g in groups),
            "total_slips": sum(g["slips"] for g in groups),
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._groups.clear()


_SCHEDULER = RateGroupScheduler()


def get_scheduler() -> RateGroupScheduler:
    return _SCHEDULER


def rate_group(name: str, period_s: float) -> RateGroup:
    return _SCHEDULER.group(name, period_s)


def rate_group_report() -> dict[str, Any]:
    return _SCHEDULER.report()


def reset_rate_groups_for_test() -> None:
    _SCHEDULER.reset_for_test()


__all__ = [
    "SLIP_TOLERANCE",
    "CycleRecord",
    "Member",
    "RateGroup",
    "RateGroupScheduler",
    "get_scheduler",
    "rate_group",
    "rate_group_report",
    "reset_rate_groups_for_test",
]
