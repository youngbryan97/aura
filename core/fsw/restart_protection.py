"""core/fsw/restart_protection.py — restart groups and overload response.

Clean-room adoption of the Apollo Guidance Computer's restart protection
and its 1202/1201 overload response.

The AGC is worth studying here for one specific reason. During the Apollo
11 descent it became overloaded — more work arriving than it could
complete in a cycle. A computer that handled this the ordinary way would
have fallen behind, missed deadlines silently, and eventually stopped
guiding a spacecraft toward the ground. Instead it:

1. **noticed** that it was out of core sets before anything was missed,
2. **announced** the condition with a specific alarm code rather than
   degrading quietly,
3. **restarted**, discarding low-priority work,
4. **resumed the critical jobs at declared restart points**, not from the
   beginning, and
5. **kept guiding**, because guidance had been declared essential and the
   discarded work had been declared not.

Every one of those five is a design decision made in advance, and the
landing happened because of them. The pattern is exactly what Aura needs
and has been missing: its recorded failures are an endurance run that grew
to 35GB and was killed whole, a false-death that respawned a duplicate,
and a cognitive tick loop that slows under load with no declared notion of
what may be dropped.

This module provides the machinery:

* **Restart groups** — work is declared as a group with a priority and a
  restart point. A restart discards everything below a threshold and
  resumes the rest from its declared point, not from the start.
* **Phase tables** — a long operation declares its phases, so "where was
  it" survives a restart. Resuming a ten-minute consolidation from phase 6
  is not the same as running it again.
* **Overload response** — a bounded pool of "core sets". Running out is
  the 1202 condition: announce it with a code, shed the lowest-priority
  groups, keep the essential ones, and record what was shed. Silence and
  slow degradation are the failure this prevents.
* **Essential work cannot be shed.** The declaration is made in advance,
  in code, where it can be reviewed — not during the overload, where it
  cannot.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger("Aura.RestartProtection")

#: The AGC had a fixed pool of core sets; running out was the 1202 alarm.
#: Aura's analogue is concurrent significant work items.
DEFAULT_CORE_SETS = 24

#: Alarm codes, kept numeric and specific for the same reason the AGC's
#: were: "1202" is searchable, "system busy" is not.
ALARM_NO_CORE_SETS = 1202
ALARM_NO_VAC_AREAS = 1201
ALARM_RESTART_LOOP = 1204
ALARM_PHASE_LOST = 1206


class Priority(IntEnum):
    """Declared in advance, where it can be reviewed."""

    #: Discarded first, without ceremony. Speculation, prefetch, idle work.
    BACKGROUND = 0
    #: Useful, deferrable. Consolidation, indexing, research.
    ROUTINE = 1
    #: The user is waiting. Response generation, tool execution.
    INTERACTIVE = 2
    #: Never shed. The tick loop, health, the Will, shutdown.
    ESSENTIAL = 3


@dataclass
class Phase:
    """One checkpoint in a long operation."""

    name: str
    index: int
    entered_at: float = field(default_factory=lambda: time.time())
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "index": self.index,
            "entered_at": self.entered_at,
            "age_s": round(time.time() - self.entered_at, 2),
            "detail": dict(self.detail),
        }


@dataclass
class RestartGroup:
    """A unit of work with a declared priority and a place to resume from."""

    name: str
    priority: Priority
    #: Called on restart to resume from ``phase``. Returning False means
    #: the group could not resume and must be treated as lost.
    resume: Callable[[Phase | None], Any] | None = None
    #: The phase table this group declares, in order.
    phases: tuple[str, ...] = ()
    current_phase: Phase | None = None
    active: bool = False
    core_sets: int = 1
    started_at: float = 0.0
    restarts: int = 0
    sheds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "priority": self.priority.name,
            "active": self.active,
            "core_sets": self.core_sets,
            "phases": list(self.phases),
            "current_phase": self.current_phase.to_dict() if self.current_phase else None,
            "restarts": self.restarts,
            "sheds": self.sheds,
            "running_for_s": round(time.time() - self.started_at, 2) if self.started_at else None,
        }


@dataclass(frozen=True)
class Alarm:
    code: int
    at: float
    detail: str
    shed: tuple[str, ...] = ()
    kept: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "at": self.at,
            "detail": self.detail,
            "shed": list(self.shed),
            "kept": list(self.kept),
        }


class RestartProtection:
    def __init__(self, *, core_sets: int = DEFAULT_CORE_SETS) -> None:
        self._lock = threading.RLock()
        self._groups: dict[str, RestartGroup] = {}
        self._alarms: list[Alarm] = []
        self.total_core_sets = core_sets
        self.restarts = 0
        self._restart_times: list[float] = []

    # ── declaration ───────────────────────────────────────────────────
    def declare(
        self,
        name: str,
        *,
        priority: Priority,
        phases: tuple[str, ...] = (),
        resume: Callable[[Phase | None], Any] | None = None,
        core_sets: int = 1,
    ) -> RestartGroup:
        """Declare a restart group. Done in advance, not during overload."""
        with self._lock:
            existing = self._groups.get(name)
            if existing is not None:
                existing.priority = priority
                existing.phases = phases or existing.phases
                existing.resume = resume or existing.resume
                existing.core_sets = max(1, core_sets)
                return existing
            group = RestartGroup(
                name=name,
                priority=priority,
                resume=resume,
                phases=phases,
                core_sets=max(1, core_sets),
            )
            self._groups[name] = group
            return group

    def get(self, name: str) -> RestartGroup | None:
        with self._lock:
            return self._groups.get(name)

    # ── core-set accounting ───────────────────────────────────────────
    def used_core_sets(self) -> int:
        with self._lock:
            return sum(g.core_sets for g in self._groups.values() if g.active)

    def available_core_sets(self) -> int:
        return max(0, self.total_core_sets - self.used_core_sets())

    def begin(self, name: str) -> bool:
        """Claim core sets for a group. False means overload — see :meth:`overload`.

        The AGC's contribution is that this check happens *before* the work
        starts, so the overload is announced rather than discovered as a
        missed deadline.
        """
        with self._lock:
            group = self._groups.get(name)
            if group is None:
                return False
            if group.active:
                return True
            if group.core_sets > self.available_core_sets():
                if group.priority is Priority.ESSENTIAL:
                    # Essential work runs regardless; the shed happens
                    # around it. Overcommitting here is correct — the
                    # alternative is not guiding the spacecraft.
                    group.active = True
                    group.started_at = time.time()
                    return True
                return False
            group.active = True
            group.started_at = time.time()
            group.current_phase = None
            return True

    def end(self, name: str) -> None:
        with self._lock:
            group = self._groups.get(name)
            if group is None:
                return
            group.active = False
            group.current_phase = None
            group.started_at = 0.0

    # ── phase tables ──────────────────────────────────────────────────
    def enter_phase(self, name: str, phase: str, **detail: Any) -> Phase | None:
        """Mark where a long operation has got to.

        This is the difference between resuming a ten-minute consolidation
        at phase 6 and running the whole thing again.
        """
        with self._lock:
            group = self._groups.get(name)
            if group is None:
                return None
            try:
                index = group.phases.index(phase)
            except ValueError:
                index = -1
                if group.phases:
                    logger.debug(
                        "restart group %s entered undeclared phase %r", name, phase
                    )
            group.current_phase = Phase(name=phase, index=index, detail=dict(detail))
            return group.current_phase

    # ── the 1202 ──────────────────────────────────────────────────────
    def overload(self, *, needed: int = 1, reason: str = "") -> Alarm:
        """Announce and act on an overload.

        Announce first, then shed from the bottom, then say what was kept.
        Degrading quietly is the failure this prevents.
        """
        shed: list[str] = []
        with self._lock:
            groups = sorted(
                (g for g in self._groups.values() if g.active),
                key=lambda g: (g.priority, -g.core_sets, g.name),
            )
            freed = 0
            for group in groups:
                if group.priority >= Priority.INTERACTIVE:
                    break
                if freed >= needed:
                    break
                group.active = False
                group.sheds += 1
                freed += group.core_sets
                shed.append(group.name)
            kept = [g.name for g in self._groups.values() if g.active]

        alarm = Alarm(
            code=ALARM_NO_CORE_SETS,
            at=time.time(),
            detail=(
                reason
                or f"needed {needed} core set(s), {self.available_core_sets()} available "
                f"of {self.total_core_sets}"
            ),
            shed=tuple(shed),
            kept=tuple(kept),
        )
        self._record_alarm(alarm)
        return alarm

    def _record_alarm(self, alarm: Alarm) -> None:
        with self._lock:
            self._alarms.append(alarm)
            if len(self._alarms) > 128:
                del self._alarms[:-128]
        logger.warning(
            "🚨 ALARM %d: %s — shed %s, kept %s",
            alarm.code,
            alarm.detail,
            list(alarm.shed) or "nothing",
            list(alarm.kept) or "nothing",
        )
        try:
            from core.fsw.telemetry_dictionary import EventSeverity, emit_event

            emit_event(
                "program_alarm",
                severity=EventSeverity.WARNING_HI,
                code=alarm.code,
                detail=alarm.detail,
                shed=list(alarm.shed),
                kept=list(alarm.kept),
            )
        except Exception:  # noqa: BLE001 — the alarm must be recorded regardless
            logger.debug("alarm telemetry failed", exc_info=True)

    # ── restart ───────────────────────────────────────────────────────
    def restart(self, *, keep_above: Priority = Priority.ROUTINE, reason: str = "") -> dict[str, Any]:
        """Discard low-priority work; resume the rest from its declared phase.

        A restart that resumes from the beginning is a restart that loses
        the work. The phase table is what makes the difference.
        """
        now = time.time()
        with self._lock:
            self.restarts += 1
            self._restart_times.append(now)
            self._restart_times = [t for t in self._restart_times if now - t < 60.0]
            looping = len(self._restart_times) >= 5
            groups = list(self._groups.values())

        discarded: list[str] = []
        resumed: list[str] = []
        lost: list[str] = []

        for group in groups:
            if not group.active:
                continue
            if group.priority < keep_above:
                group.active = False
                group.current_phase = None
                group.sheds += 1
                discarded.append(group.name)
                continue
            group.restarts += 1
            if group.resume is None:
                # No resume hook: it survives, but from wherever it was.
                resumed.append(group.name)
                continue
            try:
                outcome = group.resume(group.current_phase)
                if outcome is False:
                    lost.append(group.name)
                    group.active = False
                else:
                    resumed.append(group.name)
            except Exception as exc:  # noqa: BLE001
                logger.error("restart group %s failed to resume: %s", group.name, exc)
                lost.append(group.name)
                group.active = False

        if lost:
            self._record_alarm(
                Alarm(
                    code=ALARM_PHASE_LOST,
                    at=now,
                    detail=f"{len(lost)} group(s) could not resume: {', '.join(lost)}",
                    shed=tuple(lost),
                    kept=tuple(resumed),
                )
            )
        if looping:
            self._record_alarm(
                Alarm(
                    code=ALARM_RESTART_LOOP,
                    at=now,
                    detail=(
                        f"{len(self._restart_times)} restarts in 60s — restarting is "
                        "no longer recovering anything"
                    ),
                )
            )
            from core.runtime.taint import TaintFlag, taint

            taint(
                TaintFlag.CRASHED_ORGAN,
                f"restart loop: {len(self._restart_times)} restarts in 60s",
                subsystem="restart_protection",
            )

        report = {
            "reason": reason,
            "at": now,
            "discarded": discarded,
            "resumed": resumed,
            "lost": lost,
            "restart_loop": looping,
        }
        logger.warning(
            "🔄 restart (%s): discarded %d, resumed %d, lost %d",
            reason or "unspecified",
            len(discarded),
            len(resumed),
            len(lost),
        )
        return report

    # ── reporting ─────────────────────────────────────────────────────
    def report(self) -> dict[str, Any]:
        with self._lock:
            groups = [g.to_dict() for g in self._groups.values()]
            alarms = [a.to_dict() for a in self._alarms[-8:]]
        used = self.used_core_sets()
        return {
            "core_sets": {
                "total": self.total_core_sets,
                "used": used,
                "available": self.total_core_sets - used,
                "utilization": round(used / self.total_core_sets, 3)
                if self.total_core_sets
                else 0.0,
            },
            "groups": sorted(groups, key=lambda g: (-Priority[g["priority"]].value, g["name"])),
            "active": [g["name"] for g in groups if g["active"]],
            "essential": [g["name"] for g in groups if g["priority"] == "ESSENTIAL"],
            "sheddable_now": [
                g["name"]
                for g in groups
                if g["active"] and Priority[g["priority"]] < Priority.INTERACTIVE
            ],
            "restarts": self.restarts,
            "recent_alarms": alarms,
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._groups.clear()
            self._alarms.clear()
            self._restart_times.clear()
            self.restarts = 0


_PROTECTION = RestartProtection()


def get_restart_protection() -> RestartProtection:
    return _PROTECTION


def declare_group(
    name: str,
    *,
    priority: Priority,
    phases: tuple[str, ...] = (),
    resume: Callable[[Phase | None], Any] | None = None,
    core_sets: int = 1,
) -> RestartGroup:
    return _PROTECTION.declare(
        name, priority=priority, phases=phases, resume=resume, core_sets=core_sets
    )


def install_standard_groups() -> list[str]:
    """Declare Aura's work in advance, where the priorities can be reviewed.

    This is the list that decides what survives an overload. It belongs in
    source, reviewed, not in a heuristic evaluated during the emergency.
    """
    declarations = (
        ("kernel_tick", Priority.ESSENTIAL, ()),
        ("unified_will", Priority.ESSENTIAL, ()),
        ("health_surface", Priority.ESSENTIAL, ()),
        ("shutdown", Priority.ESSENTIAL, ()),
        ("memory_sentinel", Priority.ESSENTIAL, ()),
        (
            "response_generation",
            Priority.INTERACTIVE,
            ("retrieve", "assemble", "generate", "verify", "deliver"),
        ),
        ("tool_execution", Priority.INTERACTIVE, ("admit", "execute", "validate")),
        (
            "memory_consolidation",
            Priority.ROUTINE,
            ("select", "summarize", "embed", "write", "index"),
        ),
        ("vector_indexing", Priority.ROUTINE, ("scan", "embed", "commit")),
        ("research_cycle", Priority.ROUTINE, ("plan", "gather", "verify", "record")),
        ("curiosity", Priority.BACKGROUND, ()),
        ("speculative_prefetch", Priority.BACKGROUND, ()),
        ("idle_reflection", Priority.BACKGROUND, ()),
    )
    for name, priority, phases in declarations:
        declare_group(name, priority=priority, phases=phases)
    return [name for name, _, _ in declarations]


def restart_report() -> dict[str, Any]:
    return _PROTECTION.report()


def reset_restart_protection_for_test() -> None:
    _PROTECTION.reset_for_test()


__all__ = [
    "ALARM_NO_CORE_SETS",
    "ALARM_NO_VAC_AREAS",
    "ALARM_PHASE_LOST",
    "ALARM_RESTART_LOOP",
    "DEFAULT_CORE_SETS",
    "Alarm",
    "Phase",
    "Priority",
    "RestartGroup",
    "RestartProtection",
    "declare_group",
    "get_restart_protection",
    "install_standard_groups",
    "reset_restart_protection_for_test",
    "restart_report",
]
