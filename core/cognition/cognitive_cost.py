"""core/cognition/cognitive_cost.py — what thinking costs, in one unit.

Aura budgets in at least six currencies. There is a wall-clock deadline on the
turn, a token ceiling on the cortex, an expected value on a chunk in seconds, a
memory budget in bytes, a metabolic budget in the body model, and a disk budget
in megabytes. Each is right for its own resource and none of them can answer
the question that actually comes up: *is it worth thinking about this for
another four seconds?*

A :class:`CognitiveCost` is that question's unit. It is not a conversion table
between the six; it is one number they all reduce into, deliberately in the
same unit as benefit so that subtraction means something.

    cost = seconds·w_s + tokens·w_t + memory_pressure·w_m + tier_penalty + risk

The weights are per-context and travel with every reading, because the cost of
a second is not a property of the universe: it is high while someone is waiting
and low at three in the morning. :meth:`CostWeights.while_waiting` and
:meth:`CostWeights.idle` are the two contexts Aura actually has.

The budget
----------
A :class:`CognitiveBudget` is what NARS calls priority, durability and quality,
and Aura had as six unrelated classes. Priority is what to do first. Durability
is how fast it decays when nothing refreshes it — a task with high durability
survives being ignored, which is what stops "urgent" and "important" from being
the same number. Quality is how good the result has to be, which is what lets a
scheduler spend less on something that only needs to be roughly right.

Children inherit and shrink, which is card 039: a subgoal's budget derives from
its parent's value, the uncertainty that motivated it, and the information it
could return, rather than from a fixed depth limit.

The controller
--------------
:class:`ValueOfComputation` decides whether more thinking pays. It learns, per
method, the marginal gain per unit cost from what actually happened, and its
default when it has no history is to say so rather than to guess — an untried
method is offered once at a bounded cost, and after that it competes on
measurements.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "ModelTier",
    "CostWeights",
    "CognitiveCost",
    "CognitiveBudget",
    "MethodStats",
    "ValueOfComputation",
    "expected_information_gain",
    "get_controller",
    "reset_controller_for_test",
]


class ModelTier(StrEnum):
    """Which brain answered. The penalty is for the resource, not the quality."""

    REFLEX = "reflex"        # no model at all: a cached procedure
    SMALL = "small"
    RESIDENT = "resident"    # the 27B that is already warm
    DEEP = "deep"            # a second pass, a bigger context, more sampling

    @property
    def penalty(self) -> float:
        return {ModelTier.REFLEX: 0.0, ModelTier.SMALL: 0.05,
                ModelTier.RESIDENT: 0.2, ModelTier.DEEP: 1.0}[self]


@dataclass(frozen=True, slots=True)
class CostWeights:
    """What a second, a token and a byte are worth right now."""

    per_second: float = 1.0
    per_thousand_tokens: float = 0.05
    per_memory_pressure: float = 0.5
    context: str = "default"

    @classmethod
    def while_waiting(cls) -> CostWeights:
        """Someone is at the keyboard. Time dominates everything else."""
        return cls(per_second=4.0, per_thousand_tokens=0.05, per_memory_pressure=0.5,
                   context="while_waiting")

    @classmethod
    def idle(cls) -> CostWeights:
        """Nobody is waiting. Memory pressure matters more than the clock."""
        return cls(per_second=0.1, per_thousand_tokens=0.05, per_memory_pressure=2.0,
                   context="idle")

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_second": self.per_second,
            "per_thousand_tokens": self.per_thousand_tokens,
            "per_memory_pressure": self.per_memory_pressure,
            "context": self.context,
        }


@dataclass(frozen=True, slots=True)
class CognitiveCost:
    """One act of thinking, priced."""

    seconds: float = 0.0
    tokens: int = 0
    memory_pressure: float = 0.0
    tier: ModelTier = ModelTier.RESIDENT
    risk: float = 0.0
    weights: CostWeights = field(default_factory=CostWeights)

    @property
    def total(self) -> float:
        return (
            self.seconds * self.weights.per_second
            + (self.tokens / 1000.0) * self.weights.per_thousand_tokens
            + self.memory_pressure * self.weights.per_memory_pressure
            + self.tier.penalty
            + self.risk
        )

    def under(self, weights: CostWeights) -> CognitiveCost:
        """The same act, priced in a different context."""
        return replace(self, weights=weights)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "tokens": self.tokens,
            "memory_pressure": self.memory_pressure,
            "tier": self.tier.value,
            "risk": self.risk,
            "total": self.total,
            "weights": self.weights.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CognitiveBudget:
    """Priority, durability and quality, for anything that competes for compute."""

    priority: float = 0.5
    #: Half-life in seconds. High durability survives being ignored, which is
    #: what stops "urgent" and "important" collapsing into one number.
    durability: float = 60.0
    #: How good the answer has to be, 0 to 1. A scheduler may spend less on
    #: something that only needs to be roughly right.
    quality: float = 0.5
    expected_cost: float = 0.0
    deadline: float | None = None
    created_at: float = field(default_factory=time.time)

    def priority_at(self, now: float) -> float:
        """Priority after decay. A deadline overrides decay as it approaches."""
        age = max(0.0, now - self.created_at)
        decayed = self.priority * math.exp(-age / max(1e-6, self.durability))
        if self.deadline is None:
            return decayed
        remaining = self.deadline - now
        if remaining <= 0:
            return 0.0
        urgency = min(1.0, 1.0 / max(1e-6, remaining))
        return max(decayed, self.priority * urgency)

    def child(
        self, *, uncertainty: float, information_gain: float, complexity: float = 1.0
    ) -> CognitiveBudget:
        """A subgoal's budget, derived rather than fixed.

        A subgoal is worth what its parent is worth, times how uncertain the
        parent is about the thing it is asking, times how much the answer would
        tell it, divided by how hard it looks. A fixed depth limit gets all four
        of those wrong at once.
        """
        share = max(0.0, min(1.0, uncertainty)) * max(0.0, min(1.0, information_gain))
        return CognitiveBudget(
            priority=self.priority * share / max(1e-6, complexity),
            durability=self.durability * 0.5,
            quality=self.quality,
            expected_cost=self.expected_cost * 0.5,
            deadline=self.deadline,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority,
            "durability": self.durability,
            "quality": self.quality,
            "expected_cost": self.expected_cost,
            "deadline": self.deadline,
        }


def expected_information_gain(
    prior: Sequence[float], outcomes: Sequence[Sequence[float]]
) -> float:
    """Expected reduction in entropy from an act that could return any outcome.

    ``prior`` is the current belief over hypotheses. ``outcomes`` is, for each
    possible observation, the posterior it would produce. The return is the
    prior entropy minus the expected posterior entropy, in bits, and it is what
    makes "go and look" an action a planner can price against "answer now".
    """

    def entropy(distribution: Sequence[float]) -> float:
        total = sum(distribution) or 1.0
        return -sum((p / total) * math.log2(p / total) for p in distribution if p > 0)

    if not outcomes:
        return 0.0
    prior_entropy = entropy(prior)
    expected = 0.0
    for posterior in outcomes:
        weight = sum(posterior)
        expected += weight * entropy(posterior)
    scale = sum(sum(p) for p in outcomes) or 1.0
    return max(0.0, prior_entropy - expected / scale)


@dataclass
class MethodStats:
    """What one way of thinking has actually returned per unit of cost."""

    method: str
    attempts: int = 0
    total_cost: float = 0.0
    total_gain: float = 0.0

    @property
    def gain_per_cost(self) -> float | None:
        if self.attempts == 0 or self.total_cost <= 0:
            return None
        return self.total_gain / self.total_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "attempts": self.attempts,
            "total_cost": self.total_cost,
            "total_gain": self.total_gain,
            "gain_per_cost": self.gain_per_cost,
        }


class ValueOfComputation:
    """Whether thinking longer is worth it, learned from what happened.

    The default for an untried method is not a guess. It is offered once, at a
    bounded cost, and after that it competes on its measurements — which is the
    only honest treatment of a method nobody has run, and it means the
    controller explores exactly as much as it must.
    """

    #: What an untried method is allowed to spend on its one trial.
    TRIAL_COST = 1.0

    def __init__(self) -> None:
        self._lock = checked_lock("core.cognition.cognitive_cost.ValueOfComputation", reentrant=True)
        self._stats: dict[str, MethodStats] = {}

    def observe(self, method: str, *, cost: float, gain: float) -> None:
        with self._lock:
            stats = self._stats.setdefault(method, MethodStats(method))
            stats.attempts += 1
            stats.total_cost += max(0.0, cost)
            stats.total_gain += gain

    def should_continue(
        self, method: str, *, remaining_budget: float, expected_gain: float
    ) -> dict[str, Any]:
        """Is another unit of this method worth its cost.

        Returns the decision and the reason. The reason is the useful half: a
        controller that says no without saying whether it was the budget, the
        history or the expected gain is a controller nobody can debug.
        """
        with self._lock:
            stats = self._stats.get(method)
        if remaining_budget <= 0:
            return {"method": method, "continue": False, "reason": "budget spent"}
        if stats is None or stats.gain_per_cost is None:
            return {
                "method": method,
                "continue": remaining_budget >= self.TRIAL_COST,
                "reason": "untried; offered one bounded trial",
                "cost_allowed": min(self.TRIAL_COST, remaining_budget),
            }
        projected = stats.gain_per_cost * remaining_budget
        return {
            "method": method,
            "continue": projected > 0 and projected >= expected_gain * 0.0,
            "reason": f"measured {stats.gain_per_cost:.4g} gain per unit cost",
            "gain_per_cost": stats.gain_per_cost,
            "projected_gain": projected,
            "cost_allowed": remaining_budget,
        }

    def rank(self, methods: Sequence[str]) -> list[dict[str, Any]]:
        """Order methods by measured return, untried ones last but not excluded."""
        with self._lock:
            rows = []
            for method in methods:
                stats = self._stats.get(method)
                rows.append(
                    {
                        "method": method,
                        "gain_per_cost": stats.gain_per_cost if stats else None,
                        "attempts": stats.attempts if stats else 0,
                    }
                )
        return sorted(
            rows,
            key=lambda r: (r["gain_per_cost"] is None, -(r["gain_per_cost"] or 0.0)),
        )

    def report(self) -> dict[str, Any]:
        with self._lock:
            stats = list(self._stats.values())
        measured = [s for s in stats if s.gain_per_cost is not None]
        return {
            "methods": len(stats),
            "measured": len(measured),
            "by_method": {s.method: s.to_dict() for s in stats},
            "best": max(measured, key=lambda s: s.gain_per_cost).method if measured else None,
        }


_lock = checked_lock("core.cognition.cognitive_cost.singleton")
_controller: ValueOfComputation | None = None


def get_controller() -> ValueOfComputation:
    global _controller
    with _lock:
        if _controller is None:
            _controller = ValueOfComputation()
        return _controller


def reset_controller_for_test() -> ValueOfComputation:
    global _controller
    with _lock:
        _controller = ValueOfComputation()
        return _controller
