"""core/interiority/arbitration.py — several faculties fire at once.

Every reviewed prototype computes one mechanism at a time and none says
what happens when six are active, which is the normal case. Grief and
duty and irritation and curiosity are simultaneously true most of the
time, and how they combine decides behaviour more often than any one of
them does.

Four rules, and they are not the same rule with different weights.

**Constraints are a union, and hard beats everything.** A constraint is
not a large negative number, so it does not average with anything. If
one faculty holds an action out of the set, it is out however strongly
others want it. Overriding a soft constraint is recorded with the
faculty that held it, so a pattern of overrides is visible instead of
disappearing into a sum.

**Graded effects pass through the substrate.** Affect deltas, somatic
markers and attention biases are published into the cleft and read
through the receptor bank, so a faculty that has been firing hard for a
while is quieter than one that just started. That is what stops any
single state from holding the interior, and it happens in the transport
rather than in a limiter somebody has to remember to apply.

**Budgets compose multiplicatively and the ceiling takes the minimum.**
Two faculties each asking for more depth get more; one asking for
caution and one for speed resolve to the product, which is the honest
answer. The irreversibility ceiling is a minimum because the most
cautious active state should bound the turn.

**Conflicting action readiness is a measurement, not a tie to break.**
When the active tendencies disagree, the disagreement is reported as
``tendency_conflict``, which is exactly the input item 32 reads to
detect that the objective function is currently unstable. Silently
picking a winner would delete the signal that something is wrong.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from core.interiority.cleft import SynapticCleft, get_cleft
from core.interiority.effects import (
    ActionConstraint,
    AffectDelta,
    AttentionBias,
    BudgetDelta,
    ConstraintForce,
    GoalDelta,
    LedgerWrite,
    RetentionClaim,
    SomaticMarker,
)
from core.interiority.faculty import Activation


@dataclass(frozen=True)
class Arbitrated:
    """One coherent interior state from many active faculties."""

    affect: AffectDelta
    somatic: tuple[SomaticMarker, ...]
    attention: tuple[AttentionBias, ...]
    budget: BudgetDelta
    hard_constraints: tuple[ActionConstraint, ...]
    soft_constraints: tuple[ActionConstraint, ...]
    goals: tuple[GoalDelta, ...]
    ledger: tuple[LedgerWrite, ...]
    retention: tuple[RetentionClaim, ...]
    #: Disagreement among active action tendencies, in [0, 1]. Item 32's input.
    tendency_conflict: float
    #: Dominant tendency and its share, for the receipt.
    dominant: tuple[str, float]
    #: Per-faculty intensity after transmission.
    transmitted: Mapping[str, float]
    #: Faculties that fired but whose state did not cross the cleft.
    failed_to_cross: tuple[str, ...]
    declines: Mapping[str, str] = field(default_factory=dict)
    #: The event this state was appraised from, so a later outcome can be
    #: attributed back to the faculties that fired on it.
    event_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "affect": self.affect.to_dict(),
            "somatic": [m.to_dict() for m in self.somatic],
            "attention": [a.to_dict() for a in self.attention],
            "budget": self.budget.to_dict(),
            "hard_constraints": [c.to_dict() for c in self.hard_constraints],
            "soft_constraints": [c.to_dict() for c in self.soft_constraints],
            "goals": [g.to_dict() for g in self.goals],
            "retention": [r.to_dict() for r in self.retention],
            "tendency_conflict": self.tendency_conflict,
            "dominant": list(self.dominant),
            "transmitted": dict(self.transmitted),
            "failed_to_cross": list(self.failed_to_cross),
            "declines": dict(self.declines),
            "event_id": self.event_id,
        }


def _tendency_conflict(weights: Mapping[str, float]) -> float:
    """Normalised entropy over active action tendencies.

    Zero when everything active wants the same thing, one when the
    active states are spread evenly over several incompatible
    readinesses. This is a measurement of the interior's coherence, and
    it is the quantity that makes upheaval detectable rather than
    something an author decides to declare.
    """
    total = sum(weights.values())
    if total <= 0.0 or len(weights) < 2:
        return 0.0
    entropy = 0.0
    for weight in weights.values():
        if weight <= 0.0:
            continue
        p = weight / total
        entropy -= p * math.log(p)
    return max(0.0, min(1.0, entropy / math.log(len(weights))))


def arbitrate(
    activations: Sequence[Activation],
    *,
    cleft: SynapticCleft | None = None,
    dt: float | None = None,
    event_id: str = "",
) -> Arbitrated:
    """Combine active faculties into one interior state."""
    medium = cleft or get_cleft()

    affect = AffectDelta()
    somatic: list[SomaticMarker] = []
    attention: list[AttentionBias] = []
    budget = BudgetDelta()
    hard: dict[str, ActionConstraint] = {}
    soft: dict[str, ActionConstraint] = {}
    goals: list[GoalDelta] = []
    ledger: list[LedgerWrite] = []
    retention: dict[str, RetentionClaim] = {}
    tendencies: Counter[str] = Counter()
    transmitted: dict[str, float] = {}
    failed: list[str] = []
    declines: dict[str, str] = {}

    for activation in activations:
        if activation.declined:
            declines[activation.faculty] = activation.declined
            continue
        if activation.intensity <= 0.0:
            continue

        # Everything graded crosses the medium. Release is probabilistic
        # and the receptor bank supplies the gain, so a faculty that has
        # been loud is quieter now without anybody clamping it.
        crossing = medium.release(activation.faculty, activation.intensity, dt)
        strength = crossing.postsynaptic
        transmitted[activation.faculty] = strength
        if crossing.quanta_attempted > 0 and crossing.quanta_released == 0:
            failed.append(activation.faculty)

        effects = activation.effects
        scale = strength / activation.intensity if activation.intensity > 0 else 0.0
        graded = effects.scaled(scale)

        affect = affect + graded.affect
        somatic.extend(graded.somatic)
        attention.extend(graded.attention)
        goals.extend(graded.goals)
        # Ledger writes and retention claims are not graded: a record is
        # made or it is not, and a half-written obligation is worse than
        # either.
        ledger.extend(effects.ledger)
        for claim in effects.retention:
            retention[claim.memory_key] = claim

        budget = budget.compose(effects.budget)

        for constraint in effects.constraints:
            if constraint.force is ConstraintForce.HARD:
                hard[constraint.action_class] = constraint
            else:
                soft.setdefault(constraint.action_class, constraint)

        if activation.tendency:
            tendencies[activation.tendency] += strength

    # A hard constraint on an action class removes any soft one on it:
    # there is nothing left to negotiate.
    for action_class in list(soft):
        if action_class in hard:
            del soft[action_class]

    conflict = _tendency_conflict(tendencies)
    dominant = tendencies.most_common(1)
    dominant_pair = (
        (dominant[0][0], dominant[0][1] / max(1e-9, sum(tendencies.values())))
        if dominant
        else ("", 0.0)
    )

    return Arbitrated(
        affect=affect,
        somatic=tuple(somatic),
        attention=tuple(attention),
        budget=budget,
        hard_constraints=tuple(hard.values()),
        soft_constraints=tuple(soft.values()),
        goals=tuple(goals),
        ledger=tuple(ledger),
        retention=tuple(retention.values()),
        tendency_conflict=conflict,
        dominant=dominant_pair,
        transmitted=transmitted,
        failed_to_cross=tuple(failed),
        declines=declines,
        event_id=event_id,
    )


def permitted(
    candidates: Iterable[str], state: Arbitrated
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Filter an action set before anything scores it.

    Returns the survivors and, for each removal, the reason and the
    faculty that held it. Filtering rather than penalising is the whole
    difference between a value and a price: a penalty can be outbid by a
    large enough benefit, and this cannot.
    """
    blocked: dict[str, str] = {}
    kept: list[str] = []
    for candidate in candidates:
        reason = ""
        for constraint in state.hard_constraints:
            if candidate == constraint.action_class or candidate.startswith(
                constraint.action_class + ":"
            ):
                reason = f"{constraint.held_by}: {constraint.reason}"
                break
        if reason:
            blocked[candidate] = reason
        else:
            kept.append(candidate)
    return tuple(kept), blocked


__all__ = ["Arbitrated", "arbitrate", "permitted"]
