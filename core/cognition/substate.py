"""core/cognition/substate.py — every deadlock becomes a smaller problem.

:mod:`core.cognition.impasse` classifies a failure to decide and records it.
:mod:`core.cognition.preference_semantics` resolves preferences properly and
raises a typed impasse when it cannot. Both are correct, and between them they
have one caller. Everywhere else in the architecture a deadlock still resolves
the way it always did: the workspace sorts candidates and takes ``[0]``, a
planner returns its first option, a tool chooser falls through to a default.
Nothing records that the decision was arbitrary, so nothing learns from it.

Soar's insight is that the deadlock is the *occasion*. When the architecture
cannot decide, it makes the decision itself the problem, works on it in a
subordinate state, and compiles the answer. This module is the two halves that
were missing: one route every decision system reports through, and a substate
that is the same object whichever organ opened it.

The substate carries four things and refuses without them
---------------------------------------------------------
* **inherited context** — what the parent knew. A substate reasoning about a
  world its parent could not see is solving a different problem.
* **a local goal** — what would resolve this. A substate with no resolution
  condition cannot terminate and cannot be scored.
* **a budget** — depth, wall clock and work. Recursion without one is the
  failure mode Soar spent years on, and Aura runs for weeks.
* **a parent event** — the causal id from :mod:`core.cognition.cognitive_event`,
  so the resolution's support can be traced back through the deadlock that
  caused it.

Nesting is generic. A substate that itself deadlocks opens another, under the
remaining budget, with no organ-specific code — which is card 017's bar:
three unrelated domains produce nested subgoals and nobody wrote recursion for
any of them.

What resolves a substate
------------------------
Whatever the handler returns, typed as a :class:`Resolution`: a chosen
candidate, a new preference that breaks the tie, a refusal, or an admission
that the budget ran out. Every one of the four is a learning occasion, and
running out is the most informative — it says the problem was harder than the
architecture assumed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.cognition.cognitive_event import Phase, cycle, get_event_graph
from core.cognition.impasse import Impasse, ImpasseType

__all__ = [
    "SubstateOutcome",
    "SubstateBudget",
    "Resolution",
    "CognitiveSubstate",
    "ImpasseBus",
    "get_impasse_bus",
    "reset_impasse_bus_for_test",
]


class SubstateOutcome(StrEnum):
    #: The substate produced a choice.
    RESOLVED = "resolved"
    #: The substate produced a preference that breaks the deadlock next time.
    LEARNED_PREFERENCE = "learned_preference"
    #: The substate concluded that none of the candidates should be taken.
    REFUSED = "refused"
    #: Depth, clock or work ran out. The most informative failure.
    EXHAUSTED = "exhausted"
    #: No handler was registered for this kind of deadlock.
    UNHANDLED = "unhandled"


@dataclass(frozen=True, slots=True)
class SubstateBudget:
    """What a substate is allowed to spend. Inherited and always shrinking."""

    depth: int = 3
    seconds: float = 5.0
    work: int = 200

    def child(self, *, spent_seconds: float = 0.0, spent_work: int = 0) -> "SubstateBudget":
        """The budget a nested substate gets: strictly less than this one."""
        return SubstateBudget(
            depth=self.depth - 1,
            seconds=max(0.0, self.seconds - spent_seconds),
            work=max(0, self.work - spent_work),
        )

    @property
    def exhausted(self) -> bool:
        return self.depth <= 0 or self.seconds <= 0.0 or self.work <= 0


@dataclass(frozen=True, slots=True)
class Resolution:
    """What a substate concluded."""

    outcome: SubstateOutcome
    choice: str = ""
    preference: Any = None
    detail: str = ""
    work_done: int = 0

    @property
    def decided(self) -> bool:
        return self.outcome in (SubstateOutcome.RESOLVED, SubstateOutcome.LEARNED_PREFERENCE)


@dataclass
class CognitiveSubstate:
    """A deadlock, promoted to a problem in its own right."""

    substate_id: str
    impasse: Impasse
    organ: str
    goal: str
    context: Mapping[str, Any]
    budget: SubstateBudget
    parent_event: int = 0
    parent_substate: str = ""
    opened_at: float = field(default_factory=time.time)
    resolution: Resolution | None = None
    nested: list[str] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return 0 if not self.parent_substate else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "substate_id": self.substate_id,
            "impasse_type": self.impasse.type.value,
            "organ": self.organ,
            "goal": self.goal,
            "budget": {"depth": self.budget.depth, "seconds": self.budget.seconds, "work": self.budget.work},
            "parent_event": self.parent_event,
            "parent_substate": self.parent_substate,
            "nested": list(self.nested),
            "outcome": self.resolution.outcome.value if self.resolution else "open",
            "detail": self.resolution.detail if self.resolution else "",
        }


Handler = Callable[[CognitiveSubstate], Resolution]


class ImpasseBus:
    """The one route every decision system reports a deadlock through.

    Handlers register per impasse type, not per organ, which is what makes the
    mechanism architecture-wide: a tie in the planner and a tie in tool
    selection reach the same code because they are the same kind of failure.
    """

    def __init__(self, *, default_budget: SubstateBudget | None = None, clock=time.monotonic) -> None:
        self._lock = threading.RLock()
        self._handlers: dict[ImpasseType, list[tuple[str, Handler]]] = {}
        self._substates: dict[str, CognitiveSubstate] = {}
        self._counter = 0
        self._default = default_budget or SubstateBudget()
        self._clock = clock
        self._by_organ: dict[str, int] = {}
        self._by_outcome: dict[str, int] = {}
        self._max_history = 2048

    def register(self, impasse_type: ImpasseType, name: str, handler: Handler) -> None:
        with self._lock:
            self._handlers.setdefault(impasse_type, []).append((name, handler))

    def unregister(self, name: str) -> None:
        with self._lock:
            for handlers in self._handlers.values():
                handlers[:] = [(n, h) for n, h in handlers if n != name]

    def raise_impasse(
        self,
        impasse: Impasse,
        *,
        organ: str,
        goal: str,
        context: Mapping[str, Any] | None = None,
        budget: SubstateBudget | None = None,
        parent_event: int = 0,
        parent_substate: str = "",
    ) -> CognitiveSubstate:
        """Open a substate for this deadlock and run it to a resolution.

        Returns the substate with its :attr:`~CognitiveSubstate.resolution`
        filled in. An organ that ignores the return value has still recorded
        the deadlock, which is the minimum this bus exists to guarantee.
        """
        with self._lock:
            self._counter += 1
            substate_id = f"ss{self._counter}"
            parent = self._substates.get(parent_substate)
            effective = budget or (parent.budget.child() if parent else self._default)
            substate = CognitiveSubstate(
                substate_id=substate_id,
                impasse=impasse,
                organ=organ,
                goal=goal,
                context=dict(context or {}),
                budget=effective,
                parent_event=parent_event,
                parent_substate=parent_substate,
            )
            self._substates[substate_id] = substate
            if parent is not None:
                parent.nested.append(substate_id)
            handlers = list(self._handlers.get(impasse.type, ()))
            self._by_organ[organ] = self._by_organ.get(organ, 0) + 1
            self._trim_locked()

        graph = get_event_graph()
        if effective.exhausted:
            substate.resolution = Resolution(
                SubstateOutcome.EXHAUSTED,
                detail="budget was already spent when the substate opened",
            )
            self._finish(substate, graph, parent_event)
            return substate

        started = self._clock()
        with cycle(f"substate:{substate_id}"):
            event = graph.record(
                Phase.IMPASSE,
                organ,
                f"{impasse.type.value} on {goal}",
                parents=[parent_event] if parent_event else (),
                detail={"substate": substate_id, "candidates": list(impasse.candidates)},
            )
            resolution = Resolution(SubstateOutcome.UNHANDLED, detail="no handler registered")
            for name, handler in handlers:
                if self._clock() - started > effective.seconds:
                    resolution = Resolution(
                        SubstateOutcome.EXHAUSTED, detail=f"clock ran out in {name}"
                    )
                    break
                try:
                    candidate = handler(substate)
                except Exception as exc:  # noqa: BLE001 - a broken handler is a datum
                    resolution = Resolution(
                        SubstateOutcome.UNHANDLED, detail=f"{name}: {type(exc).__name__}: {exc}"
                    )
                    continue
                if candidate is not None and candidate.outcome is not SubstateOutcome.UNHANDLED:
                    resolution = candidate
                    break
            substate.resolution = resolution
            self._finish(substate, graph, event.seq)
        return substate

    def _finish(self, substate: CognitiveSubstate, graph, parent_event: int) -> None:
        resolution = substate.resolution or Resolution(SubstateOutcome.UNHANDLED)
        with self._lock:
            key = resolution.outcome.value
            self._by_outcome[key] = self._by_outcome.get(key, 0) + 1
        graph.record(
            Phase.LEARN,
            substate.organ,
            f"substate {resolution.outcome.value}",
            parents=[parent_event] if parent_event else (),
            outcome=resolution.outcome.value,
            detail={"substate": substate.substate_id, "choice": resolution.choice},
        )

    def _trim_locked(self) -> None:
        if len(self._substates) <= self._max_history:
            return
        for key in list(self._substates)[: len(self._substates) - self._max_history]:
            del self._substates[key]

    def get(self, substate_id: str) -> CognitiveSubstate | None:
        with self._lock:
            return self._substates.get(substate_id)

    def report(self) -> dict[str, Any]:
        """Which organs deadlock, and what happens when they do.

        ``organs_reporting`` is card 175's number: an impasse bus with one
        caller is a module, and with eight is a mechanism.
        """
        with self._lock:
            resolved = sum(
                self._by_outcome.get(k, 0)
                for k in (SubstateOutcome.RESOLVED.value, SubstateOutcome.LEARNED_PREFERENCE.value)
            )
            total = sum(self._by_outcome.values())
            return {
                "substates": len(self._substates),
                "organs_reporting": len(self._by_organ),
                "by_organ": dict(sorted(self._by_organ.items())),
                "by_outcome": dict(sorted(self._by_outcome.items())),
                "handlers": {k.value: [n for n, _ in v] for k, v in self._handlers.items()},
                "resolved_fraction": (resolved / total) if total else None,
                "nested": sum(1 for s in self._substates.values() if s.parent_substate),
            }


_bus_lock = threading.Lock()
_bus: ImpasseBus | None = None


def get_impasse_bus() -> ImpasseBus:
    global _bus
    with _bus_lock:
        if _bus is None:
            _bus = ImpasseBus()
        return _bus


def reset_impasse_bus_for_test(**kwargs: Any) -> ImpasseBus:
    global _bus
    with _bus_lock:
        _bus = ImpasseBus(**kwargs)
        return _bus
