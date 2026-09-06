"""Order declared by the parts, not by the list that happens to hold them.

Aura's phases and background jobs run in the order they appear in a list. That
order is real — a phase that reads memory has to run after the one that loads
it — but nothing anywhere says so. Move a line and the dependency breaks
silently, because the thing that needed it does not know it needed it.

AutoGPT has every component declare ``requires``, ``before`` and ``after``,
compiles a graph, and refuses a cycle at construction rather than at the
deadlock. This is that.

Three parts:

* **requires** is the hard one — this cannot run without that.
* **before/after** are preferences with no dependency: two phases that could
  run in either order but read better in one.
* **a cycle is refused where it is declared**, with the loop named. A cycle
  found at runtime is a hang; a cycle found at declaration is a sentence.

The order that comes out is deterministic: ready nodes go in declared order,
not set order, so two runs of the same declarations give the same sequence.
Ordering that changes per process is how a test passes on one machine.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhatRunsBeforeWhat")

__all__ = [
    "AStep",
    "ACycle",
    "AMissingRequirement",
    "TheOrder",
    "an_order_named",
    "how_the_orders_compiled",
]


class ACycle(ValueError):
    """A declared order that loops. Named at declaration, not at the hang."""


class AMissingRequirement(ValueError):
    """A step requires something nothing declared."""


@dataclass(frozen=True, slots=True)
class AStep:
    """One phase or job, and what it says about where it goes."""

    name: str
    #: Cannot run at all without these. A missing one is an error.
    requires: tuple[str, ...] = ()
    #: Prefers to run before/after these, and does not need them to exist.
    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    owner: str = ""
    why: str = ""


@dataclass
class TheOrder:
    """A set of steps and the sequence they compile to."""

    name: str
    _steps: dict[str, AStep] = field(default_factory=dict)
    _declared_in: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, step: AStep) -> "TheOrder":
        with self._lock:
            if step.name in self._steps:
                raise ValueError(f"{self.name}: {step.name} declared twice")
            self._steps[step.name] = step
            self._declared_in.append(step.name)
        return self

    def edges(self) -> dict[str, set[str]]:
        """name -> the steps that must run before it."""
        with self._lock:
            steps = dict(self._steps)
        known = set(steps)
        need: dict[str, set[str]] = {name: set() for name in steps}
        for name, step in steps.items():
            for other in step.requires:
                if other not in known:
                    raise AMissingRequirement(
                        f"{self.name}: {name} requires {other}, which nothing declared"
                    )
                need[name].add(other)
            for other in step.after:
                if other in known:
                    need[name].add(other)
            for other in step.before:
                if other in known:
                    need[other].add(name)
        return need

    def sequence(self) -> tuple[str, ...]:
        """The order to run in. Deterministic, and refuses a cycle by name."""
        need = self.edges()
        with self._lock:
            declared = list(self._declared_in)
        left = {name: set(before) for name, before in need.items()}
        done: list[str] = []
        while left:
            ready = [name for name in declared if name in left and not left[name]]
            if not ready:
                raise ACycle(f"{self.name}: {self._name_the_loop(left)}")
            for name in ready:
                del left[name]
                done.append(name)
            for waiting in left.values():
                waiting.difference_update(ready)
        return tuple(done)

    def _name_the_loop(self, left: dict[str, set[str]]) -> str:
        """Walk the remainder until a node repeats, and print that walk."""
        start = sorted(left)[0]
        seen: list[str] = []
        at = start
        while at not in seen:
            seen.append(at)
            nxt = sorted(left.get(at, set()) & set(left))
            if not nxt:
                break
            at = nxt[0]
        if at in seen:
            loop = seen[seen.index(at):] + [at]
            return " -> ".join(loop) + " (a cycle)"
        return f"cannot order {sorted(left)}"

    def groups(self) -> tuple[tuple[str, ...], ...]:
        """The same order, grouped into what may run in parallel.

        Everything in one group has no dependency on anything else in it, so a
        group is exactly what a superstep may hold.
        """
        need = self.edges()
        with self._lock:
            declared = list(self._declared_in)
        left = {name: set(before) for name, before in need.items()}
        out: list[tuple[str, ...]] = []
        while left:
            ready = tuple(name for name in declared if name in left and not left[name])
            if not ready:
                raise ACycle(f"{self.name}: {self._name_the_loop(left)}")
            out.append(ready)
            for name in ready:
                del left[name]
            for waiting in left.values():
                waiting.difference_update(ready)
        return tuple(out)

    def steps_with_no_declared_order(self) -> tuple[str, ...]:
        """Steps that neither require anything nor are required. Free-floating.

        Not an error. It is the list worth reading, because a step that truly
        has no ordering constraint is rare, and one that has one nobody wrote
        down looks exactly the same from here.
        """
        need = self.edges()
        needed_by_someone = {other for befores in need.values() for other in befores}
        return tuple(
            sorted(
                name
                for name, before in need.items()
                if not before and name not in needed_by_someone
            )
        )

    def report(self) -> dict[str, Any]:
        with self._lock:
            count = len(self._steps)
            without_a_reason = sorted(
                name for name, s in self._steps.items() if not s.why.strip()
            )
        try:
            sequence = list(self.sequence())
            groups = [list(g) for g in self.groups()]
            trouble = ""
        except (ACycle, AMissingRequirement) as exc:
            sequence, groups, trouble = [], [], str(exc)
        return {
            "name": self.name,
            "steps": count,
            "sequence": sequence,
            "groups": groups,
            "widest_group": max((len(g) for g in groups), default=0),
            "unordered": list(self.steps_with_no_declared_order()) if not trouble else [],
            "declared_without_a_reason": without_a_reason,
            "trouble": trouble,
        }


_ORDERS: dict[str, TheOrder] = {}
_ORDERS_LOCK = threading.Lock()


def an_order_named(name: str) -> TheOrder:
    """The order by that name, made on first use."""
    with _ORDERS_LOCK:
        return _ORDERS.setdefault(name, TheOrder(name=name))


def how_the_orders_compiled() -> dict[str, Any]:
    """For the health report: every declared order and whether it compiles."""
    with _ORDERS_LOCK:
        orders = list(_ORDERS.values())
    reports = [o.report() for o in orders]
    return {
        "orders": len(reports),
        "that_do_not_compile": [r["name"] for r in reports if r["trouble"]],
        "steps": sum(r["steps"] for r in reports),
        "each": reports,
    }
