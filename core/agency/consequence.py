"""What an initiative is worth to her in the world as it stands, not only on average.

Her arbiter valued an initiative by what acts of its kind have been worth to
her, the same in any world. The world entered deliberation only as a gain on
the final score, set by units where the world and her feelings meet; a choice
did not depend on what she expected to happen if she acted now. Value is a
belief about what will happen weighed by how it would feel:

    Q(a) = sum over outcomes o of P(o | W, a) U(o | A, S) - C(a)

Her worth for an act, learned from how turns in which she did it felt against
what she expected (core/conation/what_she_wants.py), is U averaged over the
worlds she has acted in. Conditioning it on this world is Bayes: the expected
worth given W is the worth times P(good | W, a) / P(good | a), the likelihood
ratio of this world among the worlds where acts of that kind went well. "Good"
is the measure her receipts already close on: the turn's worth against what
she had come to expect (core/affect/what_it_was_worth.py) above zero.

P(good | W, a) is a kernel estimate over the worlds she acted in (Nadaraya and
Watson), with one imagined success and one imagined failure as its prior, so
with no record it is one half, the ratio is one, and nothing changes. The
kernel is Gaussian on standardised readings, its bandwidth the dimension of
the reading, the typical squared distance between two standardised readings
being twice that, so nothing about its width is chosen here. A world unlike
any she acted in carries no weight and reads as the prior.

The world reading is the one her conjunctive codes read for W: the partner's
register, how together they are, the cadence, prediction error, and how long
since the person spoke.
"""
from __future__ import annotations

import itertools
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Consequence")

__all__ = ["ConsequenceModel", "act_kind", "get_consequence_model", "worth_in_this_world", "world_of"]


def act_kind(goal: str) -> str:
    """The act an initiative is, keyed as her worth for acts is keyed: by its first word."""
    words = str(goal or "").split()
    return f"act:{words[0].lower()}" if words else "act:"

#: How many outcomes of one kind of act the estimate keeps, most recent last.
_WINDOW = 256
#: How many open choices wait for their outcome before the oldest is dropped.
_OPEN = 256
_DIRECT = itertools.count()


def world_of(state: Any) -> np.ndarray:
    """The world as her W domain reads it, in a fixed order."""
    try:
        from core.consciousness.conjunctive_codes import readings

        return np.asarray(readings(state).get("W", ()), dtype=np.float64)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("the world could not be read for a choice: %s", exc)
        return np.zeros(0)


@dataclass
class _Kind:
    worlds: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    good: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))


class ConsequenceModel:
    """P(acting goes well | the world, the kind of act), from what followed her acts."""

    def __init__(self) -> None:
        self._lock = checked_lock("consequence_model")
        self._kinds: dict[str, _Kind] = {}
        self._open: dict[str, tuple[str, np.ndarray]] = {}
        self._count = 0
        self._mean: np.ndarray | None = None
        self._square: np.ndarray | None = None

    # -- reading the world on its own scale --------------------------------

    def _note_world(self, world: np.ndarray) -> None:
        self._count += 1
        rate = 1.0 / min(self._count, _WINDOW)
        if self._mean is None or self._mean.shape != world.shape:
            self._mean = world.copy()
            self._square = np.zeros_like(world)
            return
        delta = world - self._mean
        self._mean = self._mean + rate * delta
        self._square = (1.0 - rate) * (self._square + rate * delta * delta)

    def _standard(self, world: np.ndarray) -> np.ndarray:
        if self._mean is None or self._mean.shape != world.shape:
            return np.zeros_like(world)
        spread = np.sqrt(np.maximum(self._square, 0.0))
        return np.where(spread > 1e-12, (world - self._mean) / np.where(spread > 1e-12, spread, 1.0), 0.0)

    # -- what she expects --------------------------------------------------

    def chance(self, kind: str, world: np.ndarray) -> tuple[float, float]:
        """P(good | this world, kind) and P(good | kind), each with one success and one failure as prior."""
        world = np.asarray(world, dtype=np.float64)
        with self._lock:
            held = self._kinds.get(kind)
            if held is None or not held.good:
                return 0.5, 0.5
            goods = np.asarray(held.good)
            overall = (float(goods.sum()) + 1.0) / (goods.size + 2.0)
            here = self._standard(world)
            if here.size == 0:
                return overall, overall
            seen = np.stack([self._standard(w) for w in held.worlds])
            distance = np.sum((seen - here) ** 2, axis=1)
            weights = np.exp(-distance / (2.0 * here.size))
            local = (float(weights @ goods) + 1.0) / (float(weights.sum()) + 2.0)
            return local, overall

    def worth_here(self, worth: float, kind: str, world: np.ndarray) -> float:
        """Her worth for an act of this kind, conditioned on this world, in [0, 1]."""
        local, overall = self.chance(kind, world)
        if overall <= 0.0:
            return worth
        return max(0.0, min(1.0, float(worth) * local / overall))

    # -- learning from what followed ---------------------------------------

    def opened(self, decision_id: str, kind: str, world: np.ndarray) -> None:
        """A choice was made in this world; its outcome will come back under its id."""
        if not decision_id:
            return
        world = np.asarray(world, dtype=np.float64)
        with self._lock:
            self._note_world(world)
            self._open[decision_id] = (kind, world)
            while len(self._open) > _OPEN:
                self._open.pop(next(iter(self._open)))

    def resolved(self, decision_id: str, amount: float) -> bool:
        """What followed a choice: the turn's worth against what she expected, its sign the outcome."""
        with self._lock:
            opened = self._open.pop(decision_id, None)
            if opened is None or not math.isfinite(float(amount)):
                return False
            kind, world = opened
            held = self._kinds.setdefault(kind, _Kind())
            held.worlds.append(world)
            held.good.append(1.0 if float(amount) > 0.0 else 0.0)
            return True

    def learn(self, kind: str, world: np.ndarray, good: bool) -> None:
        """Record one outcome directly, for a caller that holds both ends of it."""
        identifier = f"direct:{kind}:{next(_DIRECT)}"
        self.opened(identifier, kind, world)
        self.resolved(identifier, 1.0 if good else -1.0)


_MODEL: ConsequenceModel | None = None


def get_consequence_model() -> ConsequenceModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = ConsequenceModel()
    return _MODEL


def worth_in_this_world(worth: float, kind: str, state: Any) -> float:
    """`ConsequenceModel.worth_here` on the process's model and the world in her state."""
    return get_consequence_model().worth_here(worth, kind, world_of(state))
