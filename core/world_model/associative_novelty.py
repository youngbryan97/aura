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
Human Neuroscience 4, 2010). And novelty opens a window rather than an instant:
what is learned for a while after it is encoded more strongly (behavioural
tagging, Moncada and Viola, Journal of Neuroscience 27, 2007; synaptic tagging,
Frey and Morris, Nature 385, 1997).

The first version read totals: everything that arrived and the sum of recall's
scores, each against its usual. Inside her organism both read nothing of a
single new thing (23 October). Some two and a half salience units arrive in an
ordinary turn, so one new percept was a twentieth of them; and recall rebuilds
what is in mind every turn, so a recollection put there was gone before the
world model looked.

So the two sides are now:

- new to her senses: what arrived of a kind she has not met within the
  ledgers' window, by salience, held as a trace that fades at her affect's
  momentum, the clock her feelings settle on, so the window novelty opens
  stays open for a while;
- in mind: memory replayed this turn, by salience, and whatever is in working
  memory that is not the conversation itself.

The precision is the product of the two on the codes' scale, exp(m / 2) with m
in [0, 1). It weights the error the world model reports and the step it learns
by.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DIALOGUE", "REPLAY", "SEEN", "WINDOW", "Comparator", "Spread", "in_mind", "new_to_her"]

#: The window her ledgers share: a rate of 1 / min(n, WINDOW).
WINDOW: int = 256

#: The mark a percept or a working-memory entry carries once the comparator has counted it.
SEEN: str = "compared"

#: The percept recall raises when it replays a memory (core/phases/memory_retrieval.py).
REPLAY: str = "memory_replay"

#: Working-memory entries that are the conversation itself rather than something brought to mind.
DIALOGUE: frozenset[str] = frozenset({"user", "assistant"})


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


def new_to_her(state: Any, familiar: dict[str, int], look: int) -> tuple[float, float]:
    """What arrived since the comparator last looked: of a kind she has not met lately, and replayed memory.

    Both by salience. ``familiar`` is when each kind was last met, in looks,
    and ``look`` is this one; a kind met within the window is familiar. Marks
    everything it read, so nothing is counted twice.
    """
    from core.state.percepts import read_percept

    world = getattr(state, "world", None)
    percepts = getattr(world, "recent_percepts", None) if world is not None else None
    if not isinstance(percepts, list):
        return 0.0, 0.0
    novel = replayed = 0.0
    for item in percepts:
        if not isinstance(item, dict) or item.get(SEEN):
            continue
        item[SEEN] = True
        percept = read_percept(item)
        if percept.kind == REPLAY:
            replayed += percept.salience
            continue
        last = familiar.get(percept.kind)
        if last is None or look - last > WINDOW:
            novel += percept.salience
        familiar[percept.kind] = look
    return novel, replayed


def in_mind(state: Any) -> float:
    """How many entries in working memory are not the conversation itself."""
    cognition = getattr(state, "cognition", None)
    working = getattr(cognition, "working_memory", None) if cognition is not None else None
    if not isinstance(working, list):
        return 0.0
    return float(sum(1 for entry in working if isinstance(entry, dict) and str(entry.get("role", "")) not in DIALOGUE))


def _momentum(state: Any) -> float:
    affect = getattr(state, "affect", None)
    try:
        value = float(getattr(affect, "momentum", 0.85))
    # not a failure: a momentum that is not a number is her default.
    except (TypeError, ValueError):
        value = 0.85
    return max(0.0, min(1.0, value))


@dataclass
class Comparator:
    """The two traces and the last precision they set."""

    novelty: float = 0.0
    looks: int = 0
    familiar: dict[str, int] = field(default_factory=dict)
    last: Mapping[str, float] = field(default_factory=dict)

    def precision(self, state: Any) -> float:
        """This cycle's precision, from the state that just closed a turn."""
        self.looks += 1
        novel, replayed = new_to_her(state, self.familiar, self.looks)
        held = in_mind(state)
        self.novelty = _momentum(state) * self.novelty + novel
        recall = replayed + held
        mismatch = math.tanh(self.novelty * recall)
        weight = math.exp(0.5 * mismatch)
        self.last = {
            "novel": novel,
            "novelty": self.novelty,
            "replayed": replayed,
            "in_mind": held,
            "mismatch": mismatch,
            "precision": weight,
        }
        return weight
