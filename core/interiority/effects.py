"""core/interiority/effects.py — what a faculty is allowed to change.

A faculty cannot write a sentence. It cannot add a line to a prompt,
select a phrase, or set a tone word. The only things it may emit are
the typed effects below, and every one of them lands on a number some
other subsystem was already reading before this package existed.

That restriction is the whole design. An interior state that reaches
behaviour by wording is not an interior state; it is a style, and the
first thing that happens to a style is that somebody changes the
wording. The eight effect types here each name an existing consumer:

============================  ==================================================
Effect                        Existing consumer
============================  ==================================================
:class:`AffectDelta`          ``core/affect/damasio_v2.py`` valence/arousal/engagement
:class:`SomaticMarker`        ``core/consciousness/somatic_marker_gate.py`` bias
:class:`AttentionBias`        ``core/global_workspace.py`` salience
:class:`BudgetDelta`          the turn's reasoning depth and deadline
:class:`ActionConstraint`     the permitted-action filter, before scoring
:class:`GoalDelta`            ``core/goals`` weights
:class:`LedgerWrite`          :mod:`core.interiority.ledger`
:class:`RetentionClaim`       memory compaction and forgetting
============================  ==================================================

:class:`ActionConstraint` is the one that is not a number, and it is
deliberate. A value that is a weight can always be outbid; a value that
is a constraint cannot. Refusing to fight is not a large negative term
in a sum, because a large enough positive term would buy it. It removes
the action from the set. That is the difference between a preference and
a commitment, and it is why constraints are filtered before scoring
rather than added to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def _f(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return 0.0 if lo <= 0.0 <= hi else lo
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class AffectDelta:
    """A change to core affect, in the axes the live engine already uses."""

    valence: float = 0.0
    arousal: float = 0.0
    engagement: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence", _f(self.valence, -1.0, 1.0))
        object.__setattr__(self, "arousal", _f(self.arousal, -1.0, 1.0))
        object.__setattr__(self, "engagement", _f(self.engagement, -1.0, 1.0))

    def scaled(self, factor: float) -> AffectDelta:
        return AffectDelta(
            self.valence * factor, self.arousal * factor, self.engagement * factor
        )

    def __add__(self, other: AffectDelta) -> AffectDelta:
        return AffectDelta(
            self.valence + other.valence,
            self.arousal + other.arousal,
            self.engagement + other.engagement,
        )

    @property
    def empty(self) -> bool:
        return self.valence == 0.0 and self.arousal == 0.0 and self.engagement == 0.0

    def to_dict(self) -> dict[str, float]:
        return {"v": self.valence, "a": self.arousal, "e": self.engagement}


@dataclass(frozen=True)
class SomaticMarker:
    """A bias attached to an option before it is evaluated.

    Damasio's claim is that affect narrows the option set before
    deliberation rather than after it, which is what makes it a
    decision mechanism rather than a decoration on one.
    """

    option: str
    bias: float
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bias", _f(self.bias, -1.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {"option": self.option, "bias": self.bias, "reason": self.reason}


@dataclass(frozen=True)
class AttentionBias:
    """Raise or lower how much a target competes for the workspace."""

    target: str
    weight: float
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", _f(self.weight, -1.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "weight": self.weight, "reason": self.reason}


@dataclass(frozen=True)
class BudgetDelta:
    """How much time and depth this turn deserves.

    Multiplicative, and both directions are used: upheaval spends more
    and commits to less, fun spends more and risks more, and a state
    with no stake spends less. A faculty that only ever asks for more is
    not regulating anything.
    """

    depth: float = 1.0
    deadline: float = 1.0
    #: Ceiling on how irreversible an action this turn may take, in [0, 1].
    #: 1.0 imposes nothing.
    irreversibility_ceiling: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", _f(self.depth, 0.1, 4.0))
        object.__setattr__(self, "deadline", _f(self.deadline, 0.1, 4.0))
        object.__setattr__(
            self, "irreversibility_ceiling", _f(self.irreversibility_ceiling, 0.0, 1.0)
        )

    @property
    def empty(self) -> bool:
        return (
            self.depth == 1.0
            and self.deadline == 1.0
            and self.irreversibility_ceiling == 1.0
        )

    def compose(self, other: BudgetDelta) -> BudgetDelta:
        return BudgetDelta(
            depth=self.depth * other.depth,
            deadline=self.deadline * other.deadline,
            irreversibility_ceiling=min(
                self.irreversibility_ceiling, other.irreversibility_ceiling
            ),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "depth": self.depth,
            "deadline": self.deadline,
            "irreversibility_ceiling": self.irreversibility_ceiling,
        }


class ConstraintForce(StrEnum):
    """How hard a constraint is.

    ``HARD`` removes the action from the set and no score can restore
    it. ``SOFT`` is a strong prior that a large enough benefit may
    overcome, and it is recorded when it is overcome, so a pattern of
    overriding shows up rather than disappearing into a sum.
    """

    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True)
class ActionConstraint:
    """An action class that is not available, and why."""

    action_class: str
    force: ConstraintForce
    reason: str
    #: The faculty that holds it, so an override is attributable.
    held_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_class": self.action_class,
            "force": str(self.force),
            "reason": self.reason,
            "held_by": self.held_by,
        }


@dataclass(frozen=True)
class GoalDelta:
    """A change to a goal's weight in the stack."""

    goal: str
    delta: float
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta", _f(self.delta, -1.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "delta": self.delta, "reason": self.reason}


@dataclass(frozen=True)
class LedgerWrite:
    """A change to what the agent is holding."""

    op: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "args": dict(self.args)}


@dataclass(frozen=True)
class RetentionClaim:
    """A memory that compaction may not drop, and the reason it is load-bearing.

    The offer to remove a painful memory is not hypothetical for a
    machine: summarisation, compaction and forgetting policies make it
    every day, and mostly without asking. A retention claim is how a
    faculty says no. It names the memory, the commitment it supports,
    and it expires — an unbounded claim is a memory leak with a
    conscience.
    """

    memory_key: str
    reason: str
    held_by: str
    #: Seconds from issue. A claim outlives the state that raised it, so
    #: that a grief that has quieted does not lose the record.
    ttl_s: float = 86400.0 * 365.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_key": self.memory_key,
            "reason": self.reason,
            "held_by": self.held_by,
            "ttl_s": self.ttl_s,
        }


@dataclass(frozen=True)
class Effects:
    """Everything one faculty activation changes."""

    affect: AffectDelta = field(default_factory=AffectDelta)
    somatic: tuple[SomaticMarker, ...] = ()
    attention: tuple[AttentionBias, ...] = ()
    budget: BudgetDelta = field(default_factory=BudgetDelta)
    constraints: tuple[ActionConstraint, ...] = ()
    goals: tuple[GoalDelta, ...] = ()
    ledger: tuple[LedgerWrite, ...] = ()
    retention: tuple[RetentionClaim, ...] = ()

    @property
    def empty(self) -> bool:
        return (
            self.affect.empty
            and not self.somatic
            and not self.attention
            and self.budget.empty
            and not self.constraints
            and not self.goals
            and not self.ledger
            and not self.retention
        )

    def scaled(self, factor: float) -> Effects:
        """Scale the graded effects by intensity. Constraints do not scale.

        A constraint at half intensity is not half a constraint. If a
        faculty holds an action out of the set, it holds it out; the
        graded part is how much everything else moves.
        """
        return Effects(
            affect=self.affect.scaled(factor),
            somatic=tuple(
                SomaticMarker(m.option, m.bias * factor, m.reason) for m in self.somatic
            ),
            attention=tuple(
                AttentionBias(a.target, a.weight * factor, a.reason)
                for a in self.attention
            ),
            budget=self.budget,
            constraints=self.constraints,
            goals=tuple(GoalDelta(g.goal, g.delta * factor, g.reason) for g in self.goals),
            ledger=self.ledger,
            retention=self.retention,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "affect": self.affect.to_dict(),
            "somatic": [m.to_dict() for m in self.somatic],
            "attention": [a.to_dict() for a in self.attention],
            "budget": self.budget.to_dict(),
            "constraints": [c.to_dict() for c in self.constraints],
            "goals": [g.to_dict() for g in self.goals],
            "ledger": [w.to_dict() for w in self.ledger],
            "retention": [r.to_dict() for r in self.retention],
        }


__all__ = [
    "ActionConstraint",
    "AffectDelta",
    "AttentionBias",
    "BudgetDelta",
    "ConstraintForce",
    "Effects",
    "GoalDelta",
    "LedgerWrite",
    "RetentionClaim",
    "SomaticMarker",
]
