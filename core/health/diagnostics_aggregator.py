"""core/health/diagnostics_aggregator.py — hierarchical diagnostics.

Clean-room adoption of ROS 2's `diagnostic_updater` / `diagnostic_aggregator`
pair, and specifically of the one level that Aura's health surface does not
have: **STALE**.

Aura's health contract answers "is each required service present and
passing". That is the right question and it has a blind spot that every
robot stack learns about the hard way: a component that *stops reporting*
looks different from a component that reports a problem, and it must not
look the same as a component that was never expected. Today a subsystem
whose probe thread died contributes nothing, and contributing nothing is
indistinguishable from contributing "fine". The aggregate stays green
while a limb goes numb.

ROS 2's model:

* Each component owns an **updater** with named **tasks**. A task returns
  a status: OK, WARN, ERROR — plus key/value pairs, which is what makes a
  diagnostic actionable rather than a mood ring.
* The **aggregator** groups tasks into a tree of analyzers, and rolls the
  worst level upward. A subsystem is exactly as healthy as its unhealthiest
  part, and the path to that part is in the name.
* An analyzer knows how many items it **expects**. An expected item that
  has not reported within its timeout is STALE — worse than WARN, because
  a component that is lying by omission is less trustworthy than one
  reporting a known problem.

This composes with the health contract rather than replacing it: the
aggregate tree becomes one more input, and its STALE set is the thing the
contract could not see on its own.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger("Aura.Diagnostics")

#: A task that has not refreshed within this is STALE.
DEFAULT_STALE_AFTER_S = 30.0


class Level(IntEnum):
    """Ordered so aggregation is a max()."""

    OK = 0
    WARN = 1
    ERROR = 2
    #: Expected and silent. Deliberately the worst: a component lying by
    #: omission is less trustworthy than one reporting a known problem.
    STALE = 3

    @property
    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class DiagnosticStatus:
    name: str
    level: Level
    message: str
    values: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=lambda: time.time())
    hardware_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level.label,
            "message": self.message,
            "values": dict(self.values),
            "at": self.at,
            "age_s": round(max(0.0, time.time() - self.at), 2),
            "hardware_id": self.hardware_id,
        }


TaskFn = Callable[[], "DiagnosticStatus | tuple[Level, str] | Level | None"]


@dataclass
class _Task:
    name: str
    fn: TaskFn
    stale_after_s: float
    last: DiagnosticStatus | None = None
    runs: int = 0
    failures: int = 0


class DiagnosticUpdater:
    """One component's diagnostic tasks."""

    def __init__(self, component: str, *, stale_after_s: float = DEFAULT_STALE_AFTER_S) -> None:
        self.component = component
        self.stale_after_s = stale_after_s
        self._lock = threading.Lock()
        self._tasks: dict[str, _Task] = {}

    def add(self, name: str, fn: TaskFn, *, stale_after_s: float | None = None) -> None:
        """Register a diagnostic task. Name is relative to the component."""
        with self._lock:
            self._tasks[name] = _Task(
                name=name,
                fn=fn,
                stale_after_s=stale_after_s
                if stale_after_s is not None
                else self.stale_after_s,
            )

    def remove(self, name: str) -> None:
        with self._lock:
            self._tasks.pop(name, None)

    def task_names(self) -> list[str]:
        with self._lock:
            return sorted(self._tasks)

    def update(self) -> list[DiagnosticStatus]:
        """Run every task. A task that raises reports ERROR, never nothing."""
        with self._lock:
            tasks = list(self._tasks.values())
        results: list[DiagnosticStatus] = []
        for task in tasks:
            full_name = f"{self.component}/{task.name}"
            try:
                status = _coerce_status(full_name, task.fn())
                task.runs += 1
            except Exception as exc:  # noqa: BLE001 — silence is the failure mode we fix
                task.failures += 1
                status = DiagnosticStatus(
                    name=full_name,
                    level=Level.ERROR,
                    message=f"diagnostic task raised {type(exc).__name__}: {exc}",
                )
            task.last = status
            results.append(status)
        return results

    def statuses(self) -> list[DiagnosticStatus]:
        """Last known statuses, with staleness applied."""
        now = time.time()
        with self._lock:
            tasks = list(self._tasks.values())
        out: list[DiagnosticStatus] = []
        for task in tasks:
            full_name = f"{self.component}/{task.name}"
            if task.last is None:
                out.append(
                    DiagnosticStatus(
                        name=full_name,
                        level=Level.STALE,
                        message="expected but has never reported",
                    )
                )
                continue
            age = now - task.last.at
            if age > task.stale_after_s:
                out.append(
                    DiagnosticStatus(
                        name=full_name,
                        level=Level.STALE,
                        message=(
                            f"last reported {age:.0f}s ago (stale after "
                            f"{task.stale_after_s:.0f}s); its previous word was "
                            f"{task.last.level.label}: {task.last.message}"
                        ),
                        values=dict(task.last.values),
                        at=task.last.at,
                    )
                )
                continue
            out.append(task.last)
        return out


def _coerce_status(name: str, outcome: Any) -> DiagnosticStatus:
    if isinstance(outcome, DiagnosticStatus):
        return outcome if outcome.name else DiagnosticStatus(
            name=name,
            level=outcome.level,
            message=outcome.message,
            values=outcome.values,
            at=outcome.at,
            hardware_id=outcome.hardware_id,
        )
    if isinstance(outcome, Level):
        return DiagnosticStatus(name=name, level=outcome, message=outcome.label)
    if isinstance(outcome, tuple) and len(outcome) == 2:
        level, message = outcome
        return DiagnosticStatus(name=name, level=Level(level), message=str(message))
    if outcome is None or outcome is True:
        return DiagnosticStatus(name=name, level=Level.OK, message="ok")
    if outcome is False:
        return DiagnosticStatus(name=name, level=Level.ERROR, message="task returned False")
    return DiagnosticStatus(name=name, level=Level.OK, message=str(outcome))


@dataclass
class Analyzer:
    """Groups statuses under a path and knows how many to expect."""

    path: str
    #: Statuses whose name starts with any of these prefixes belong here.
    prefixes: tuple[str, ...]
    #: Names that must be present. A missing expected item is STALE.
    expected: tuple[str, ...] = ()
    #: An analyzer marked non-critical never worsens the top-level roll-up
    #: beyond WARN — for genuinely optional subsystems.
    critical: bool = True

    def matches(self, status: DiagnosticStatus) -> bool:
        return any(status.name.startswith(prefix) for prefix in self.prefixes)


@dataclass
class AggregateNode:
    path: str
    level: Level
    message: str
    children: list[DiagnosticStatus] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    critical: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "level": self.level.label,
            "message": self.message,
            "critical": self.critical,
            "missing": list(self.missing),
            "children": [c.to_dict() for c in self.children],
        }


class DiagnosticsAggregator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._updaters: dict[str, DiagnosticUpdater] = {}
        self._analyzers: list[Analyzer] = []

    def updater(self, component: str, *, stale_after_s: float = DEFAULT_STALE_AFTER_S) -> DiagnosticUpdater:
        with self._lock:
            existing = self._updaters.get(component)
            if existing is not None:
                return existing
            updater = DiagnosticUpdater(component, stale_after_s=stale_after_s)
            self._updaters[component] = updater
            return updater

    def add_analyzer(self, analyzer: Analyzer) -> Analyzer:
        with self._lock:
            self._analyzers = [a for a in self._analyzers if a.path != analyzer.path]
            self._analyzers.append(analyzer)
            self._analyzers.sort(key=lambda a: a.path)
            return analyzer

    def update_all(self) -> list[DiagnosticStatus]:
        with self._lock:
            updaters = list(self._updaters.values())
        results: list[DiagnosticStatus] = []
        for updater in updaters:
            results.extend(updater.update())
        return results

    def statuses(self) -> list[DiagnosticStatus]:
        with self._lock:
            updaters = list(self._updaters.values())
        out: list[DiagnosticStatus] = []
        for updater in updaters:
            out.extend(updater.statuses())
        return out

    def aggregate(self) -> dict[str, Any]:
        """Roll the worst level upward. The path names the sick part."""
        statuses = self.statuses()
        with self._lock:
            analyzers = list(self._analyzers)

        nodes: list[AggregateNode] = []
        claimed: set[str] = set()
        for analyzer in analyzers:
            children = [s for s in statuses if analyzer.matches(s)]
            claimed.update(s.name for s in children)
            present = {s.name for s in children}
            missing = [
                name
                for name in analyzer.expected
                if not any(p == name or p.endswith(f"/{name}") for p in present)
            ]
            level = max((s.level for s in children), default=Level.OK)
            if missing:
                level = max(level, Level.STALE)
            worst = max(children, key=lambda s: s.level, default=None) if children else None
            if missing:
                message = f"{len(missing)} expected diagnostic(s) never reported: {', '.join(missing)}"
            elif worst is not None and worst.level is not Level.OK:
                message = f"{worst.name}: {worst.message}"
            else:
                message = f"{len(children)} diagnostic(s) ok"
            nodes.append(
                AggregateNode(
                    path=analyzer.path,
                    level=level,
                    message=message,
                    children=children,
                    missing=missing,
                    critical=analyzer.critical,
                )
            )

        unclaimed = [s for s in statuses if s.name not in claimed]
        if unclaimed:
            level = max(s.level for s in unclaimed)
            nodes.append(
                AggregateNode(
                    path="/other",
                    level=level,
                    message=f"{len(unclaimed)} diagnostic(s) match no analyzer",
                    children=unclaimed,
                    critical=False,
                )
            )

        top = Level.OK
        for node in nodes:
            contribution = node.level if node.critical else min(node.level, Level.WARN)
            top = max(top, contribution)

        return {
            "level": top.label,
            "ok": top is Level.OK,
            "nodes": [n.to_dict() for n in nodes],
            "stale": [s.name for s in statuses if s.level is Level.STALE],
            "errors": [s.name for s in statuses if s.level is Level.ERROR],
            "warnings": [s.name for s in statuses if s.level is Level.WARN],
            "count": len(statuses),
            "summary": _summary(top, nodes),
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._updaters.clear()
            self._analyzers.clear()


def _summary(top: Level, nodes: list[AggregateNode]) -> str:
    if top is Level.OK:
        return f"all {sum(len(n.children) for n in nodes)} diagnostics ok"
    worst = [n for n in nodes if n.level is top]
    return f"{top.label}: " + "; ".join(f"{n.path} — {n.message}" for n in worst[:3])


_AGGREGATOR = DiagnosticsAggregator()


def get_aggregator() -> DiagnosticsAggregator:
    return _AGGREGATOR


def updater(component: str, *, stale_after_s: float = DEFAULT_STALE_AFTER_S) -> DiagnosticUpdater:
    return _AGGREGATOR.updater(component, stale_after_s=stale_after_s)


def diagnostics_report() -> dict[str, Any]:
    _AGGREGATOR.update_all()
    return _AGGREGATOR.aggregate()


def install_default_analyzers() -> list[str]:
    """The standing tree. Adding an organ means adding a task, not a node."""
    analyzers = (
        Analyzer(
            path="/runtime",
            prefixes=("runtime/", "lockdep/", "memory/", "pressure/"),
            expected=("taint", "lockdep", "pressure", "oom_ladder"),
        ),
        Analyzer(path="/cognition", prefixes=("cognition/", "cortex/", "kernel/")),
        Analyzer(path="/memory", prefixes=("memory_facade/", "episodic/", "vector/")),
        Analyzer(path="/senses", prefixes=("voice/", "vision/", "soma/"), critical=False),
        Analyzer(path="/autonomy", prefixes=("autonomy/", "goals/", "research/"), critical=False),
        Analyzer(path="/orchestration", prefixes=("controller/", "lease/", "eviction/")),
    )
    for analyzer in analyzers:
        _AGGREGATOR.add_analyzer(analyzer)
    return [a.path for a in analyzers]


def install_runtime_diagnostics() -> list[str]:
    """Diagnostic tasks over the disciplines this codebase already has.

    They are cheap reads of reports that already exist — the value is
    that they now go STALE if they stop being produced, instead of
    quietly vanishing from the aggregate.
    """
    runtime = updater("runtime")

    def taint_task() -> DiagnosticStatus:
        from core.runtime.taint import taint_report

        report = taint_report()
        credibility = report["credibility_affecting"]
        return DiagnosticStatus(
            name="runtime/taint",
            level=Level.WARN if credibility else Level.OK,
            message=(
                f"tainted: {report['compact']}" if report["tainted"] else "untainted"
            ),
            values={"flags": report["compact"], "credibility": credibility},
        )

    def lockdep_task() -> DiagnosticStatus:
        from core.runtime.lockdep import lockdep_report

        report = lockdep_report()
        return DiagnosticStatus(
            name="runtime/lockdep",
            level=Level.OK if report["clean"] else Level.ERROR,
            message=(
                "no lock-order violations"
                if report["clean"]
                else f"{len(report['splats'])} splat(s): {report['splats'][0]['kind']}"
            ),
            values={"acquires_checked": report["acquires_checked"]},
        )

    def pressure_task() -> DiagnosticStatus:
        from core.runtime.pressure_stall import psi_narrative, saturated_resources

        saturated = saturated_resources()
        return DiagnosticStatus(
            name="runtime/pressure",
            level=Level.WARN if saturated else Level.OK,
            message=psi_narrative(),
            values={"saturated": saturated},
        )

    def oom_task() -> DiagnosticStatus:
        from core.runtime.oom_policy import oom_report

        report = oom_report()
        has_rungs = report["sheddable_organs"] > 0
        return DiagnosticStatus(
            name="runtime/oom_ladder",
            level=Level.OK if has_rungs else Level.WARN,
            message=(
                f"next victim: {report['next_victim']}"
                if has_rungs
                else "no organ can be shed; restart is the only response to pressure"
            ),
            values={
                "sheddable": report["sheddable_organs"],
                "immune": len(report["immune_organs"]),
            },
        )

    runtime.add("taint", taint_task)
    runtime.add("lockdep", lockdep_task)
    runtime.add("pressure", pressure_task)
    runtime.add("oom_ladder", oom_task)

    orchestration = updater("controller")

    def controllers_task() -> DiagnosticStatus:
        from core.runtime.reconcile import reconcile_report

        report = reconcile_report()
        unconverged = report["unconverged"]
        return DiagnosticStatus(
            name="controller/queues",
            level=Level.WARN if unconverged else Level.OK,
            message=(
                f"{len(unconverged)} controller(s) not converged: {', '.join(unconverged)}"
                if unconverged
                else f"{report['count']} controller(s) converged"
            ),
            values={"total_queue_depth": report["total_queue_depth"]},
        )

    orchestration.add("queues", controllers_task)
    return runtime.task_names() + orchestration.task_names()


def reset_diagnostics_for_test() -> None:
    _AGGREGATOR.reset_for_test()


__all__ = [
    "DEFAULT_STALE_AFTER_S",
    "AggregateNode",
    "Analyzer",
    "DiagnosticStatus",
    "DiagnosticUpdater",
    "DiagnosticsAggregator",
    "Level",
    "diagnostics_report",
    "get_aggregator",
    "install_default_analyzers",
    "install_runtime_diagnostics",
    "reset_diagnostics_for_test",
    "updater",
]
