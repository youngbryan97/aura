"""What arrived meeting what she recalled: how much her world model's error counts.

Her world model learned at one step size whatever she perceived, and reported
every error at the same weight. The conjunctive codes then set the step size
from perception and memory through a direction drawn from a name, which has no
reason to make the two together count for more than either alone, and on 21
October setting both moved her world model by 0.069 whitened units in no
particular direction.

The hippocampus is a comparator. What arrives from the senses meets what recall
predicts from the same context, and a mismatch drives the dopamine that gates
learning there (Lisman and Grace, Neuron 46, 2005). Novelty in that sense is
associative: something new arriving in a context memory has called up
(Kumaran and Maguire, Hippocampus 17, 2007). A new thing in a context nothing
calls up is just new, and a recalled context with nothing new in it is a
match. Under predictive coding the same signal is a precision: it sets how much
a prediction error counts, both in what is passed on and in how far it moves
the model (Yu and Dayan, Neuron 46, 2005; Feldman and Friston, Frontiers in
Human Neuroscience 4, 2010).

So the precision here is a product: how far more arrived this cycle than
usually does, times how far more is recalled than usually is, each in its own
spread. It weights the surprise the model reports and the step it learns by,
on the codes' scale, exp(m / 2) with m in [0, 1).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SEEN", "WINDOW", "Comparator", "Spread", "arrived", "recalled"]

#: The window her ledgers share: a rate of 1 / min(n, WINDOW).
WINDOW: int = 256

#: The mark a percept carries once the comparator has counted it.
SEEN: str = "compared"


@dataclass
class Spread:
    """A running level and its spread, at the ledgers' window."""

    n: int = 0
    level: float = 0.0
    spread: float = 0.0

    def beyond(self, value: float) -> float:
        """How far ``value`` is above the usual, in its own spread, in [0, 1).

        With no spread yet, any excess is as far beyond as an excess can be.
        """
        over = float(value) - self.level
        if over <= 0.0:
            return 0.0
        if self.spread <= 0.0:
            return 1.0
        return math.tanh(over / self.spread)

    def note(self, value: float) -> None:
        self.n += 1
        rate = 1.0 / min(self.n, WINDOW)
        gap = float(value) - self.level
        self.level += rate * gap
        self.spread += rate * (abs(gap) - self.spread)


def arrived(state: Any) -> float:
    """The salience of what arrived since the comparator last looked. Marks it seen."""
    from core.state.percepts import read_percept

    world = getattr(state, "world", None)
    percepts = getattr(world, "recent_percepts", None) if world is not None else None
    if not isinstance(percepts, list):
        return 0.0
    total = 0.0
    for item in percepts:
        if not isinstance(item, dict) or item.get(SEEN):
            continue
        total += read_percept(item).salience
        item[SEEN] = True
    return total


def recalled(state: Any) -> float:
    """How much is recalled: the match scores of what recall has in mind."""
    cognition = getattr(state, "cognition", None)
    scores = getattr(cognition, "memory_scores", None) if cognition is not None else None
    total = 0.0
    for score in scores or ():
        try:
            value = float(score)
        # not a failure: a score that is not a number recalled nothing.
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            total += max(0.0, min(1.0, value))
    return total


@dataclass
class Comparator:
    """The two running levels and the last precision they set."""

    arrivals: Spread = field(default_factory=Spread)
    recall: Spread = field(default_factory=Spread)
    last: Mapping[str, float] = field(default_factory=dict)

    def precision(self, state: Any) -> float:
        """This cycle's precision, from the state that just closed a turn."""
        new, held = arrived(state), recalled(state)
        mismatch = self.arrivals.beyond(new) * self.recall.beyond(held)
        self.arrivals.note(new)
        self.recall.note(held)
        weight = math.exp(0.5 * mismatch)
        self.last = {"arrived": new, "recalled": held, "mismatch": mismatch, "precision": weight}
        return weight
