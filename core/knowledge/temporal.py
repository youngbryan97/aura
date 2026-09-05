"""core/knowledge/temporal.py — when, in the same language as what.

The AtomSpace can say that rain implies wet. It cannot say that rain precedes
wet, that the gap is about a minute, or that the pattern recurs. Time lives in
the world model, in event timestamps and in memory recency, none of which the
inference rules can read, so a regularity Aura has lived through a hundred
times cannot become a rule she can reason with.

Six relations, chosen because they are what a lived event stream actually
supports and no more:

* ``before`` / ``after`` — ordering, with a measured lag distribution.
* ``during`` — one interval inside another.
* ``overlaps`` — two intervals that share time without containment.
* ``recurs`` — a period, with the variance that says how regular it is.
* ``within`` — a bounded delay, which is what a precondition needs.

Temporal induction
------------------
:func:`induce_temporal_rules` reads an event stream and proposes relations from
it, and the interesting part is what it refuses. A pair that co-occurs is not a
pair that is ordered: if B precedes A as often as A precedes B, the ordering is
noise and no rule is proposed. And a lag whose spread is as large as its mean
is not a lag; it is two events that happen sometimes.

The rules land as ordinary atoms with ordinary truth values, so PLN deduction
composes them with everything else, and evidence lineage applies - a regularity
observed once in a hundred sessions and re-derived a hundred times is still
one observation.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.knowledge.atomspace import Atom, AtomSpace, Link, Node, TruthValue, concept, predicate

__all__ = [
    "TemporalRelation",
    "Interval",
    "TimedEvent",
    "temporal_link",
    "induce_temporal_rules",
    "assert_temporal_rules",
    "InducedRule",
    "MIN_OBSERVATIONS",
    "ORDER_CONSISTENCY",
]

#: Pairs needed before an ordering is proposed at all.
MIN_OBSERVATIONS = 5

#: Fraction of observations that must agree on the direction. Below this, A and
#: B co-occur and nothing is known about which comes first.
ORDER_CONSISTENCY = 0.8


class TemporalRelation(StrEnum):
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    RECURS = "recurs"
    WITHIN = "within"


@dataclass(frozen=True, slots=True)
class Interval:
    """A stretch of time. A point is an interval with zero width."""

    start: float
    end: float | None = None

    @property
    def finish(self) -> float:
        return self.start if self.end is None else self.end

    def before(self, other: "Interval") -> bool:
        return self.finish < other.start

    def during(self, other: "Interval") -> bool:
        return other.start <= self.start and self.finish <= other.finish

    def overlaps(self, other: "Interval") -> bool:
        return (
            self.start < other.finish
            and other.start < self.finish
            and not self.during(other)
            and not other.during(self)
        )


@dataclass(frozen=True, slots=True)
class TimedEvent:
    """One thing that happened, and when."""

    name: str
    interval: Interval
    detail: Mapping[str, Any] = field(default_factory=dict)


def temporal_link(
    relation: TemporalRelation, a: Atom | str, b: Atom | str, *, lag: float | None = None
) -> Link:
    """A temporal relation as an ordinary atom, so PLN can compose it.

    ``lag`` becomes part of the link's identity when given, because "A precedes
    B by a second" and "A precedes B by an hour" are different claims and
    merging their evidence would blend two regularities into one that describes
    neither.
    """
    left = concept(a) if isinstance(a, str) else a
    right = concept(b) if isinstance(b, str) else b
    outgoing: tuple[Atom, ...] = (predicate(relation.value), left, right)
    if lag is not None:
        outgoing = (*outgoing, concept(f"lag:{lag:.3g}"))
    return Link("Temporal", outgoing)


@dataclass(frozen=True, slots=True)
class InducedRule:
    """A temporal regularity read off an event stream, with what it rests on."""

    relation: TemporalRelation
    antecedent: str
    consequent: str
    observations: int
    consistency: float
    mean_lag: float | None = None
    lag_spread: float | None = None
    period: float | None = None

    @property
    def lag_is_meaningful(self) -> bool:
        """Whether the delay is a delay or two things that happen sometimes."""
        if self.mean_lag is None or self.lag_spread is None:
            return False
        return self.lag_spread < abs(self.mean_lag)

    def truth(self) -> TruthValue:
        return TruthValue(self.consistency, float(self.observations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation.value,
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "observations": self.observations,
            "consistency": self.consistency,
            "mean_lag": self.mean_lag,
            "lag_spread": self.lag_spread,
            "period": self.period,
            "lag_is_meaningful": self.lag_is_meaningful,
        }


def induce_temporal_rules(
    events: Sequence[TimedEvent],
    *,
    window: float = 60.0,
    min_observations: int = MIN_OBSERVATIONS,
) -> list[InducedRule]:
    """Read ordering and recurrence off a stream, refusing what it cannot support.

    ``window`` bounds how far apart two events may be and still be considered
    related. Without it every event in a long session relates to every other,
    and the induction reports the session rather than the structure.
    """
    ordered = sorted(events, key=lambda e: e.interval.start)
    pairs: dict[tuple[str, str], list[float]] = {}
    for i, first in enumerate(ordered):
        for second in ordered[i + 1:]:
            gap = second.interval.start - first.interval.finish
            if gap > window:
                break
            if first.name == second.name:
                continue
            pairs.setdefault((first.name, second.name), []).append(gap)

    rules: list[InducedRule] = []
    seen: set[tuple[str, str]] = set()
    for (a, b), forward in pairs.items():
        if (b, a) in seen or (a, b) in seen:
            continue
        seen.add((a, b))
        backward = pairs.get((b, a), [])
        total = len(forward) + len(backward)
        if total < min_observations:
            continue
        consistency = len(forward) / total
        if consistency < ORDER_CONSISTENCY:
            # A and B co-occur. Which comes first is not known, and proposing
            # the majority direction would turn a coin flip into a rule.
            continue
        spread = statistics.pstdev(forward) if len(forward) > 1 else 0.0
        rules.append(
            InducedRule(
                relation=TemporalRelation.BEFORE,
                antecedent=a, consequent=b,
                observations=len(forward), consistency=consistency,
                mean_lag=statistics.fmean(forward), lag_spread=spread,
            )
        )

    # Recurrence: one event whose gaps to itself are regular.
    by_name: dict[str, list[float]] = {}
    for event in ordered:
        by_name.setdefault(event.name, []).append(event.interval.start)
    for name, starts in by_name.items():
        if len(starts) < min_observations:
            continue
        gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
        if not gaps:
            continue
        mean = statistics.fmean(gaps)
        spread = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
        if mean <= 0 or spread >= mean:
            continue
        rules.append(
            InducedRule(
                relation=TemporalRelation.RECURS,
                antecedent=name, consequent=name,
                observations=len(gaps), consistency=1.0 - min(1.0, spread / mean),
                period=mean, mean_lag=mean, lag_spread=spread,
            )
        )
    return rules


def assert_temporal_rules(
    space: AtomSpace, rules: Iterable[InducedRule], *, source: str
) -> list[Link]:
    """Put induced rules into the space under one source identity.

    One induction pass is one observation of the stream, however many rules it
    produced, so re-running it on the same stream cannot inflate confidence.
    """
    landed = []
    for rule in rules:
        link = temporal_link(
            rule.relation, rule.antecedent, rule.consequent,
            lag=rule.mean_lag if rule.lag_is_meaningful else None,
        )
        space.add(link, rule.truth(), source=source)
        landed.append(link)
    return landed
