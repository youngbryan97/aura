"""core/social/conventions.py — markers that mean something only because we agree.

Some preferences have a reason inside them. Preferring a shorter proof, a
cooler room, a route with fewer turns: take the context away and the reason
still stands.

Others have nothing inside them at all. Which side of the road, which greeting,
whether a trailing comma is correct, which colour belongs to which kind of
person. The dress historian Jo Paoletti found American trade journals in 1918
recommending pink for boys on the grounds that it was the stronger colour, and
blue for girls as the daintier one. The assignment we have now is the opposite
one, it settled around the 1940s, and no property of the wavelength changed in
between. That is what an arbitrary marker is: the meaning is real, the
coordination is real, and the link between the marker and the meaning is not.

Confusing the two kinds of preference is a live failure. A system that stores
"pink means feminine" as a fact cannot represent 1918, cannot represent a
population that is changing its mind, and cannot tell the difference between a
convention and a law. A system that dismisses conventions as arbitrary and
therefore unimportant is worse, because coordination is worth a great deal and
somebody following a convention is usually getting exactly what it is for.

What this module holds is the third option: the meaning, the population it
holds in, and the fact that it could have been otherwise, all at once.

## The dynamics

Coordination on a marker is a game where matching pays and the particular
choice does not. Replicator dynamics on such a game have two stable rests, one
at each convention, and an unstable one between them. Where a population ends
up depends on where it started, which is what historical contingency means
stated as a fact about a dynamical system rather than as a hedge.

The unstable point in the middle is the useful part. It is the size a
committed minority has to reach before the population falls the other way, and
``tipping_point`` computes it. Deliberately taking up a marker whose current
meaning you reject — the pink hat at a march, a slur taken back — is exactly
this move, and it is the one case where the arbitrariness of a marker is the
whole reason the act can work.

## Adopting one knowingly

``adopt`` prices using a marker: what the coordination is worth, what
expressing this particular thing is worth to the user, and what it costs. A
marker can be worth using while its meaning is understood to be conventional,
and holding both of those at once is the thing the module exists to make
representable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Social.Conventions")

#: Steps of the replicator flow taken when projecting where a population is
#: heading. Enough for the flow to reach a rest from anywhere in the interior.
PROJECTION_STEPS = 2000

#: Step size for that flow, in units of the payoff scale.
PROJECTION_DT = 0.01

#: How close to zero or one counts as settled.
SETTLED = 1e-4


@dataclass
class Marker:
    """Something with no payoff of its own that a population uses to coordinate.

    ``intrinsic_payoff`` is here to be zero. A marker with a real payoff of its
    own is not a convention and the tipping-point arithmetic does not apply to
    it, so the field exists to be checked rather than to be filled in.
    """

    key: str
    meanings: tuple[str, str]
    """The two readings the population could settle on."""

    frequency: float = 0.5
    """Share of the population currently on ``meanings[0]``."""

    intrinsic_payoff: float = 0.0
    coordination_payoff: float = 1.0
    """What matching whoever you meet is worth."""

    observations: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def arbitrary(self) -> bool:
        """Whether nothing but agreement is holding the meaning in place."""
        return abs(self.intrinsic_payoff) < 1e-12

    def current_meaning(self) -> str | None:
        """What the marker reads as now, or nothing while the population is split."""
        if self.frequency > 0.5 + SETTLED:
            return self.meanings[0]
        if self.frequency < 0.5 - SETTLED:
            return self.meanings[1]
        return None

    def counterfactual_meaning(self) -> str | None:
        """What it would read as under the other rest.

        Present so that the alternative is a value the system can return
        rather than a caveat in a comment. A model that cannot produce this
        has stored the convention as a fact.
        """
        now = self.current_meaning()
        if now is None:
            return None
        return self.meanings[1] if now == self.meanings[0] else self.meanings[0]

    def observe(self, chose_first: bool, *, at: float | None = None,
                weight: float = 1.0) -> None:
        """Record one use of the marker, and move the estimated frequency.

        A running mean rather than a filter with a rate, so that the estimate
        is what was seen and the number of observations behind it is on the
        record next to it.
        """
        self.observations += 1
        w = max(1e-9, float(weight))
        share = w / (self.observations * w)
        self.frequency += share * ((1.0 if chose_first else 0.0) - self.frequency)
        self.frequency = min(max(self.frequency, 0.0), 1.0)
        self.history.append((at if at is not None else time.time(), self.frequency))
        if len(self.history) > 512:
            del self.history[: len(self.history) - 512]


def replicator_step(x: float, *, coordination: float, bias: float = 0.0,
                    dt: float = PROJECTION_DT) -> float:
    """One step of the replicator flow for a two-convention coordination game.

    Payoff to using the first marker against a population share ``x`` is
    ``coordination * x + bias``; to the second it is ``coordination * (1 - x)``.
    The replicator equation gives

        dx/dt = x * (1 - x) * (payoff_first - payoff_second)

    which for zero bias is ``coordination * x(1-x)(2x - 1)`` — rests at zero
    and one, both stable, and an unstable rest at a half. The bias term is
    what a marker with some intrinsic pull would have, and it moves the
    unstable rest rather than removing it.
    """
    x = min(max(x, 0.0), 1.0)
    advantage = coordination * (2.0 * x - 1.0) + bias
    return min(max(x + dt * x * (1.0 - x) * advantage, 0.0), 1.0)


def project(x: float, *, coordination: float, bias: float = 0.0,
            steps: int = PROJECTION_STEPS) -> float:
    """Run the flow to where it rests."""
    for _ in range(steps):
        nxt = replicator_step(x, coordination=coordination, bias=bias)
        if abs(nxt - x) < 1e-12:
            return nxt
        x = nxt
    return x


def tipping_point(*, coordination: float, bias: float = 0.0) -> float | None:
    """Share that has to hold the minority view before the population flips.

    The interior rest of the flow. With no intrinsic pull it sits at a half;
    a marker the population has some independent reason to prefer moves it,
    and past a large enough pull the interior rest leaves the unit interval
    and the minority cannot win at any size, which is returned as nothing
    rather than as a number outside the range.
    """
    if coordination <= 0:
        return None
    x = 0.5 - bias / (2.0 * coordination)
    if not (0.0 < x < 1.0):
        return None
    return x


@dataclass(frozen=True)
class Adoption:
    """Whether to use a marker, priced."""

    marker: str
    use: bool
    coordination_value: float
    expressive_value: float
    cost: float
    net: float
    meaning: str | None
    counterfactual: str | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "use": self.use,
            "coordination": round(self.coordination_value, 4),
            "expressive": round(self.expressive_value, 4),
            "cost": round(self.cost, 4),
            "net": round(self.net, 4),
            "meaning": self.meaning,
            "would_have_meant": self.counterfactual,
            "reason": self.reason,
        }


class ConventionRegistry:
    """Markers, the populations they hold in, and what they would mean otherwise."""

    def __init__(self) -> None:
        self._markers: dict[str, Marker] = {}

    def declare(self, key: str, meanings: tuple[str, str], *,
                frequency: float = 0.5, coordination_payoff: float = 1.0,
                intrinsic_payoff: float = 0.0) -> Marker:
        marker = self._markers.get(key)
        if marker is None:
            marker = Marker(
                key=key, meanings=meanings, frequency=float(frequency),
                coordination_payoff=float(coordination_payoff),
                intrinsic_payoff=float(intrinsic_payoff),
            )
            self._markers[key] = marker
        return marker

    def get(self, key: str) -> Marker | None:
        return self._markers.get(key)

    def observe(self, key: str, chose_first: bool, **kw: Any) -> None:
        marker = self._markers.get(key)
        if marker is not None:
            marker.observe(chose_first, **kw)

    def settles_at(self, key: str) -> float | None:
        """Where this marker's population is heading if nothing intervenes."""
        marker = self._markers.get(key)
        if marker is None:
            return None
        return project(
            marker.frequency,
            coordination=marker.coordination_payoff,
            bias=marker.intrinsic_payoff,
        )

    def flip_cost(self, key: str) -> dict[str, Any] | None:
        """What it would take to move this marker to the other convention."""
        marker = self._markers.get(key)
        if marker is None:
            return None
        threshold = tipping_point(
            coordination=marker.coordination_payoff, bias=marker.intrinsic_payoff
        )
        if threshold is None:
            return {"reachable": False, "threshold": None, "shortfall": None}
        current = marker.frequency
        heading = self.settles_at(key)
        # Whichever rest the population is not heading toward is the one a
        # minority would be trying to reach.
        target_is_first = (heading is not None and heading < 0.5)
        held = current if target_is_first else 1.0 - current
        needed = threshold if target_is_first else 1.0 - threshold
        return {
            "reachable": True,
            "threshold": round(needed, 4),
            "held": round(held, 4),
            "shortfall": round(max(0.0, needed - held), 4),
            "target": marker.meanings[0] if target_is_first else marker.meanings[1],
        }

    def adopt(self, key: str, *, expressive_value: float = 0.0,
              cost: float = 0.0, audience_share: float | None = None) -> Adoption:
        """Price using a marker, with its arbitrariness on the record.

        Coordination is worth the payoff times the share of the audience on
        the same convention. Expression is what the user gets from it
        regardless of anyone matching, which is the term that lets a marker be
        worth using into a population that reads it the other way.
        """
        marker = self._markers.get(key)
        if marker is None:
            return Adoption(
                marker=key, use=False, coordination_value=0.0,
                expressive_value=float(expressive_value), cost=float(cost),
                net=float(expressive_value) - float(cost), meaning=None,
                counterfactual=None, reason="marker not declared",
            )
        share = marker.frequency if audience_share is None else float(audience_share)
        coordination = marker.coordination_payoff * share
        net = coordination + float(expressive_value) - float(cost)
        if net > 0 and coordination >= float(expressive_value):
            reason = "read the same way by most of the audience"
        elif net > 0:
            reason = "worth using for its own sake against how it is read"
        else:
            reason = "not worth what it costs here"
        return Adoption(
            marker=key, use=net > 0, coordination_value=coordination,
            expressive_value=float(expressive_value), cost=float(cost), net=net,
            meaning=marker.current_meaning(),
            counterfactual=marker.counterfactual_meaning(),
            reason=reason,
        )

    def status(self) -> dict[str, Any]:
        return {
            key: {
                "meanings": list(m.meanings),
                "frequency": round(m.frequency, 4),
                "means_now": m.current_meaning(),
                "would_have_meant": m.counterfactual_meaning(),
                "arbitrary": m.arbitrary,
                "observations": m.observations,
                "settles_at": round(self.settles_at(key) or 0.0, 4),
                "flip": self.flip_cost(key),
            }
            for key, m in sorted(self._markers.items())
        }


_REGISTRY: ConventionRegistry | None = None


def get_convention_registry() -> ConventionRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ConventionRegistry()
    return _REGISTRY


def reset_convention_registry_for_test() -> None:
    global _REGISTRY
    _REGISTRY = None
