"""core/cognition/automaticity.py — practice should get cheaper, and be seen to.

The claim a developmental architecture makes is that doing something twice is
cheaper than doing it once. Aura has the mechanisms that would make that true —
chunks, generalized rules, macros, distillation — and no number that says
whether it is. Each learner reports its own hit rate, which answers "is my
cache working" rather than "is she getting better at this".

The measure here is **executive cost**: what the expensive machinery spent on
one occurrence of a recurring task.

    executive_cost = cortex_tokens·w_t + planner_expansions·w_p
                     + workspace_competitions·w_w + wall_seconds·w_s

Procedure hits are deliberately NOT in it. A procedure firing is the mechanism
under test, and putting it in the score would let the score improve by firing
more procedures rather than by spending less. What a procedure hit does is
appear beside the cost, so a fall in cost that comes with no hits is visible as
what it is: something else got faster.

Two properties are checked rather than assumed:

* **Monotone decrease under practice.** :meth:`Automaticity.trend` fits the
  slope of executive cost against occurrence index. A negative slope is skill;
  a flat one with rising procedure hits means the procedures are firing and
  not saving anything.
* **De-automatisation on novelty.** When a variant of a practised task fails,
  cost should rise again. A system that stays cheap through a failure is not
  automatic, it is stuck, and :meth:`Automaticity.rigidity` is the number that
  separates them.

The weights are a caller's choice and are recorded with every reading, because
a cost function nobody can see is a cost function nobody can argue with.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CostWeights",
    "Occurrence",
    "TaskCurve",
    "Automaticity",
    "get_automaticity",
    "reset_automaticity_for_test",
]


@dataclass(frozen=True, slots=True)
class CostWeights:
    """What each unit of expensive machinery counts for.

    Defaults put a cortex token and a planner expansion on the same footing as
    each other and a second of wall clock well above both, which is the right
    ordering for an agent whose scarcest resource is the user's time. A caller
    measuring something else should say so and pass its own.
    """

    per_cortex_token: float = 0.001
    per_planner_expansion: float = 0.01
    per_workspace_competition: float = 0.005
    per_second: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "per_cortex_token": self.per_cortex_token,
            "per_planner_expansion": self.per_planner_expansion,
            "per_workspace_competition": self.per_workspace_competition,
            "per_second": self.per_second,
        }


@dataclass(frozen=True, slots=True)
class Occurrence:
    """One time a recurring task was performed."""

    task: str
    index: int
    cortex_tokens: int = 0
    planner_expansions: int = 0
    workspace_competitions: int = 0
    seconds: float = 0.0
    procedure_hits: int = 0
    succeeded: bool = True
    variant: str = ""
    at: float = field(default_factory=time.time)

    def executive_cost(self, weights: CostWeights) -> float:
        return (
            self.cortex_tokens * weights.per_cortex_token
            + self.planner_expansions * weights.per_planner_expansion
            + self.workspace_competitions * weights.per_workspace_competition
            + self.seconds * weights.per_second
        )


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Ordinary least squares slope. Zero when it cannot be computed."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator


@dataclass
class TaskCurve:
    """Every occurrence of one recurring task, in order."""

    task: str
    occurrences: list[Occurrence] = field(default_factory=list)

    def costs(self, weights: CostWeights) -> list[float]:
        return [o.executive_cost(weights) for o in self.occurrences]

    def to_dict(self, weights: CostWeights) -> dict[str, Any]:
        costs = self.costs(weights)
        return {
            "task": self.task,
            "occurrences": len(self.occurrences),
            "first_cost": costs[0] if costs else None,
            "last_cost": costs[-1] if costs else None,
            "procedure_hits": sum(o.procedure_hits for o in self.occurrences),
            "failures": sum(1 for o in self.occurrences if not o.succeeded),
        }


class Automaticity:
    """Executive cost per recurring task, and what it does under practice."""

    def __init__(self, *, weights: CostWeights | None = None, max_tasks: int = 2048) -> None:
        self._lock = checked_lock("core.cognition.automaticity.Automaticity", reentrant=True)
        self._weights = weights or CostWeights()
        self._curves: dict[str, TaskCurve] = {}
        self._max_tasks = int(max_tasks)

    def observe(
        self,
        task: str,
        *,
        cortex_tokens: int = 0,
        planner_expansions: int = 0,
        workspace_competitions: int = 0,
        seconds: float = 0.0,
        procedure_hits: int = 0,
        succeeded: bool = True,
        variant: str = "",
    ) -> Occurrence:
        """Record one performance of a task."""
        with self._lock:
            curve = self._curves.get(task)
            if curve is None:
                if len(self._curves) >= self._max_tasks:
                    self._curves.pop(next(iter(self._curves)))
                curve = TaskCurve(task=task)
                self._curves[task] = curve
            occurrence = Occurrence(
                task=task,
                index=len(curve.occurrences),
                cortex_tokens=cortex_tokens,
                planner_expansions=planner_expansions,
                workspace_competitions=workspace_competitions,
                seconds=seconds,
                procedure_hits=procedure_hits,
                succeeded=succeeded,
                variant=variant,
            )
            curve.occurrences.append(occurrence)
            return occurrence

    def trend(self, task: str) -> dict[str, Any]:
        """Does this task get cheaper with practice, and is it the procedures?

        ``cost_slope`` negative is skill. ``hits_without_saving`` is the
        uncomfortable case: procedures fire more and cost does not fall, which
        means they are being matched and are not replacing anything.
        """
        with self._lock:
            curve = self._curves.get(task)
            if curve is None or len(curve.occurrences) < 2:
                return {"task": task, "measurable": False, "occurrences": 0}
            indices = [float(o.index) for o in curve.occurrences]
            costs = curve.costs(self._weights)
            hits = [float(o.procedure_hits) for o in curve.occurrences]
            cost_slope = _slope(indices, costs)
            hit_slope = _slope(indices, hits)
            return {
                "task": task,
                "measurable": True,
                "occurrences": len(curve.occurrences),
                "cost_slope": cost_slope,
                "procedure_hit_slope": hit_slope,
                "first_cost": costs[0],
                "last_cost": costs[-1],
                "reduction": (costs[0] - costs[-1]) / costs[0] if costs[0] else 0.0,
                "automatic": cost_slope < 0,
                "hits_without_saving": hit_slope > 0 and cost_slope >= 0,
                "weights": self._weights.to_dict(),
            }

    def rigidity(self, task: str) -> dict[str, Any]:
        """What happened to cost when a practised task started failing.

        Automatic and adaptive means cost rises again on novelty. Cheap through
        failure is not skill; it is a procedure firing where it does not apply
        and nothing noticing.
        """
        with self._lock:
            curve = self._curves.get(task)
            if curve is None:
                return {"task": task, "measurable": False}
            successes = [o for o in curve.occurrences if o.succeeded]
            failures = [o for o in curve.occurrences if not o.succeeded]
            if not failures or len(successes) < 2:
                return {"task": task, "measurable": False, "failures": len(failures)}
            settled = sum(o.executive_cost(self._weights) for o in successes[-3:]) / len(successes[-3:])
            after = [
                o for o in curve.occurrences
                if o.index > min(f.index for f in failures)
            ]
            if not after:
                return {"task": task, "measurable": False, "failures": len(failures)}
            recovered = sum(o.executive_cost(self._weights) for o in after) / len(after)
            return {
                "task": task,
                "measurable": True,
                "settled_cost": settled,
                "cost_after_failure": recovered,
                "de_automatised": recovered > settled,
                "rigid": recovered <= settled,
            }

    def report(self) -> dict[str, Any]:
        with self._lock:
            trends = [self.trend(task) for task in self._curves]
            measurable = [t for t in trends if t.get("measurable")]
            return {
                "tasks": len(self._curves),
                "measurable_tasks": len(measurable),
                "tasks_getting_cheaper": sum(1 for t in measurable if t["automatic"]),
                "tasks_with_hits_but_no_saving": sum(
                    1 for t in measurable if t["hits_without_saving"]
                ),
                "mean_reduction": (
                    sum(t["reduction"] for t in measurable) / len(measurable)
                ) if measurable else None,
                "weights": self._weights.to_dict(),
                "by_task": {t["task"]: t for t in measurable},
            }


_lock = checked_lock("core.cognition.automaticity.singleton")
_index: Automaticity | None = None


def get_automaticity() -> Automaticity:
    global _index
    with _lock:
        if _index is None:
            _index = Automaticity()
        return _index


def reset_automaticity_for_test(**kwargs: Any) -> Automaticity:
    global _index
    with _lock:
        _index = Automaticity(**kwargs)
        return _index
