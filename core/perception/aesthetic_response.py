"""core/perception/aesthetic_response.py — finding something beautiful, and stopping.

Liking how something looks is not a property of the thing. The same object is
striking the first time, comfortable the tenth, and invisible the hundredth,
and nothing about it changed. Whatever is being measured is happening in the
observer, and the useful consequence is that it can be computed from the
observer's history rather than guessed at from the object's features.

Three strands, and this module keeps them apart because they disagree with
each other and the disagreement is real.

**Order against complexity.** Birkhoff proposed a measure in 1933: order
divided by complexity. Taking complexity as the share of an object that does
not compress, and order as the share that does, his ratio has a property that
is easy to see and was the standing objection to it — a blank page has almost
no complexity and scores enormously, and nobody thinks a blank page is the
most beautiful object available. Eysenck's correction, from the factor
analyses, makes the measure a product instead, which peaks at half-ordered and
falls to zero at both ends. Both are computed here and both are returned. The
argument was never settled, and the blank page is the case where they disagree
most, so a caller that wants one number has to say which.

**Fluency.** Something easy to take in is pleasant, and things get easier with
exposure. Measured here as the conditional encoding cost against everything
this observer has seen, which is the same measurement
:mod:`core.conation.aesthetic` uses for a different purpose one layer down.

**Arousal potential, and Berlyne's curve.** Preference against novelty is an
inverted U: too obvious is dull, too strange is not enjoyed. Berlyne's own
account of why is two opposed systems, a reward response and an aversion
response, each rising with arousal potential, the aversion one later and
steeper. Their difference is the curve, and doing it this way rather than
fitting a parabola means the peak's position comes from the two thresholds
instead of from a coefficient someone chose.

## Habituation is the mechanism, not an afterthought

Exposure raises fluency and lowers novelty. Both move the response, in
opposite directions at first and then the same way, which is why liking often
rises before it falls. ``forget`` and the frozen-history mode exist so this
can be ablated: hold the history still and the decay disappears entirely, and
if it does not, the decay was coming from somewhere else.

## What it cannot tell you

Whether the thing is any good. This measures the shape of an encounter between
one observer and one object, and there is no step in it where an object with
no observer has a value.
"""

from __future__ import annotations

import logging
import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Perception.Aesthetic")

LEVEL = 9

#: Bytes of history retained. The window bounds what familiarity can mean.
MAX_HISTORY_BYTES = 262_144

#: Where the two opposed responses turn on, in units of arousal potential.
#: The reward response comes first and the aversion response later, which is
#: the whole reason the difference is a hump rather than a slope. Berlyne's
#: shape; the values place the peak in the middle of the range, which is where
#: the preference data put it.
REWARD_MIDPOINT = 0.35
AVERSION_MIDPOINT = 0.70

#: Steepness of each. The aversion response is the sharper one, which is what
#: makes the fall on the far side quicker than the rise on the near side.
REWARD_SLOPE = 7.0
AVERSION_SLOPE = 9.0


def _size(payload: bytes) -> int:
    return len(zlib.compress(payload, LEVEL))


def _sigmoid(x: float, midpoint: float, slope: float) -> float:
    return 1.0 / (1.0 + math.exp(-slope * (x - midpoint)))


@dataclass(frozen=True)
class Response:
    """One encounter between an observer and an object."""

    key: str
    order: float
    """Share of the object accounted for by regularity, in [0, 1]."""

    complexity: float
    """Share of the object that does not compress, in [0, 1]."""

    birkhoff: float
    """Order over complexity. Unbounded, and largest for a blank page."""

    eysenck: float
    """Order times complexity, scaled to [0, 1]. Largest at half-ordered."""

    fluency: float
    novelty: float
    arousal_potential: float
    pleasure: float
    """Reward response less aversion response. Berlyne's hedonic value."""

    exposures: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "order": round(self.order, 4),
            "complexity": round(self.complexity, 4),
            "birkhoff": round(self.birkhoff, 4),
            "eysenck": round(self.eysenck, 4),
            "fluency": round(self.fluency, 4),
            "novelty": round(self.novelty, 4),
            "arousal": round(self.arousal_potential, 4),
            "pleasure": round(self.pleasure, 4),
            "exposures": self.exposures,
        }


class AestheticObserver:
    """One observer's history, and what things look like to them because of it.

    Two observers with different histories give different answers about the
    same object, and that is the design rather than a limitation. An object's
    entry here has no value until somebody has looked at it.
    """

    def __init__(self, *, frozen: bool = False) -> None:
        #: When set, exposure is not recorded. The ablation: with the history
        #: held still, nothing habituates, and any decay still observed is
        #: coming from somewhere this module does not control.
        self.frozen = bool(frozen)
        self._history: bytes = b""
        self._exposures: dict[str, int] = {}
        self._responses: list[Response] = []

    # ------------------------------------------------------------- measures

    def complexity(self, payload: bytes) -> float:
        """Share of the object that resists description, in [0, 1].

        Birkhoff's complexity, operationalised. A blank page is near zero, a
        random string near one, and anything with structure in between. Length
        is deliberately not part of it: a long regular thing is not complex,
        it is long, and a measure that confuses the two rates wallpaper above
        a fugue.
        """
        if not payload:
            return 0.0
        return max(0.0, min(1.0, _size(payload) / len(payload)))

    def order(self, payload: bytes) -> float:
        """The complement: what regularity accounts for."""
        return 1.0 - self.complexity(payload)

    def fluency(self, payload: bytes) -> float:
        """How cheaply this goes down, given everything already seen."""
        if not payload:
            return 0.0
        alone = _size(payload)
        if alone <= 0:
            return 0.0
        if not self._history:
            return 0.0
        conditional = _size(self._history + payload) - _size(self._history)
        return max(0.0, min(1.0, 1.0 - conditional / alone))

    def novelty(self, payload: bytes) -> float:
        return 1.0 - self.fluency(payload)

    # -------------------------------------------------------------- respond

    def look(self, key: str, payload: bytes) -> Response:
        """Take it in. Records the exposure unless the observer is frozen."""
        complexity = self.complexity(payload)
        order = 1.0 - complexity
        fluency = self.fluency(payload)
        novelty = 1.0 - fluency
        # Berlyne's collative variables together. Novelty and complexity both
        # raise arousal potential, and they are averaged rather than summed so
        # the result stays on the scale the two response curves are placed on.
        arousal = (novelty + complexity) / 2.0
        pleasure = (
            _sigmoid(arousal, REWARD_MIDPOINT, REWARD_SLOPE)
            - _sigmoid(arousal, AVERSION_MIDPOINT, AVERSION_SLOPE)
        )
        response = Response(
            key=key, order=order, complexity=complexity,
            birkhoff=order / max(complexity, 1e-9),
            # Scaled by four so the product's own maximum, at half-ordered,
            # reads as one and the two measures are on comparable ranges.
            eysenck=4.0 * order * complexity,
            fluency=fluency, novelty=novelty,
            arousal_potential=arousal, pleasure=pleasure,
            exposures=self._exposures.get(key, 0),
        )
        if not self.frozen:
            self._exposures[key] = self._exposures.get(key, 0) + 1
            self._history = (self._history + payload)[-MAX_HISTORY_BYTES:]
        self._responses.append(response)
        if len(self._responses) > 512:
            del self._responses[: len(self._responses) - 512]
        return response

    def consider(self, key: str, payload: bytes) -> Response:
        """Score it without looking at it. No exposure recorded either way."""
        was = self.frozen
        self.frozen = True
        try:
            return self.look(key, payload)
        finally:
            self.frozen = was

    def forget(self) -> None:
        """Clear the history. Everything becomes new again, which is the point."""
        self._history = b""
        self._exposures.clear()

    def curve(self, key: str, payload: bytes, *, times: int) -> list[Response]:
        """Look at the same thing repeatedly and return what happened.

        The measurement that shows habituation rather than asserting it. With
        a frozen observer the returned pleasures are constant, and that
        difference is the ablation.
        """
        return [self.look(key, payload) for _ in range(times)]

    def status(self) -> dict[str, Any]:
        last = self._responses[-1] if self._responses else None
        return {
            "frozen": self.frozen,
            "history_bytes": len(self._history),
            "objects_seen": len(self._exposures),
            "looks": len(self._responses),
            "last": last.as_dict() if last else None,
        }


def berlyne_curve(samples: Sequence[float]) -> list[tuple[float, float]]:
    """The hedonic response at given arousal potentials.

    Exposed on its own so the shape can be checked without a stimulus. If this
    is not a hump, the two midpoints are in the wrong order and every response
    above is wrong in the same way.
    """
    return [
        (
            a,
            _sigmoid(a, REWARD_MIDPOINT, REWARD_SLOPE)
            - _sigmoid(a, AVERSION_MIDPOINT, AVERSION_SLOPE),
        )
        for a in samples
    ]


_OBSERVER: AestheticObserver | None = None


def get_aesthetic_observer() -> AestheticObserver:
    global _OBSERVER
    if _OBSERVER is None:
        _OBSERVER = AestheticObserver()
    return _OBSERVER


def reset_aesthetic_observer_for_test() -> None:
    global _OBSERVER
    _OBSERVER = None
