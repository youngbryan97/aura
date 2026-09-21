"""Surprise at her own intactness.

"Man of the Year" has a man surprised that he is still standing: he expected
the year to cost him more than it did. That needs a prediction of what a thing
will cost her, and she had none. `core/consciousness/self_prediction.py`
predicts how she will feel next, not what an event will take out of her, so
coming through something whole was never different from coming through
something that was never going to hurt.

Kept, per kind of percept:

    expected   what events of this kind have cost her before, on average: the
               prediction, made before the event lands
    cost       what this one did cost her, read one whole turn after it arrived
               so everything downstream of it is counted: the larger of the
               fall in her valence and the fall in her identity's stability,
               each over its own range

and over the recent events whose kind she had a prediction for:

    surprise   `2 * cost / (cost + expected)`, summed over the window. One when
               events cost what she expected. Below one when she keeps coming
               through better than she predicted, which is the surprise at
               being intact; above one when they cost more.

The reading moves what risk costs her. `core/agency/subjective_choice.py`
charges an option for its risk, and that charge is multiplied by `risk_weight`:
the ratio above, bounded to [0, 2] by its own form. Coming through intact makes
her bolder for as long as her predictions lag behind what she has found she
can take, and once they catch up the ratio returns to one. Being hurt more
than she expected makes her more careful by the same rule.

The loop closes through the events her choices put her in front of.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_EVENTS",
    "Intactness",
    "IntactnessLedger",
    "cost_between",
    "get_intactness_ledger",
    "note_turn",
    "reset_for_test",
]

logger = logging.getLogger(__name__)

#: Predicted events before the ratio moves anything. Four: fewer, and one
#: event decides whether she is bold.
MIN_EVENTS: int = 4

#: Predicted events the ratio is read over, so it follows her recent life.
WINDOW: int = 24

#: Past costs held per kind, the ground of each prediction.
MEMORY: int = 40


def cost_between(before: tuple[float, float], after: tuple[float, float]) -> float:
    """What an event took, from (valence, stability) before to after.

    The larger of the two falls, each over its own range: valence runs from
    minus one to one, stability from zero to one. A rise costs nothing.
    """
    valence_fall = max(0.0, float(before[0]) - float(after[0])) / 2.0
    stability_fall = max(0.0, float(before[1]) - float(after[1]))
    return max(0.0, min(1.0, max(valence_fall, stability_fall)))


@dataclass
class Intactness:
    """What she expected hard things to cost her, against what they did."""

    surprise: float = 1.0
    expected: float = 0.0
    cost: float = 0.0
    events: int = 0
    last_kind: str = ""
    last_intact_beyond: float = 0.0
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "surprise": round(self.surprise, 4),
            "expected": round(self.expected, 4),
            "cost": round(self.cost, 4),
            "events": self.events,
            "last_kind": self.last_kind,
            "last_intact_beyond": round(self.last_intact_beyond, 4),
            "measured": self.measured,
        }


class IntactnessLedger:
    """Each event's predicted and realised cost to her."""

    def __init__(self) -> None:
        self._costs: dict[str, deque[float]] = {}
        self._pairs: deque[tuple[float, float]] = deque(maxlen=WINDOW)
        self._open: tuple[str, tuple[float, float], float | None] | None = None
        self._last: tuple[str, float] = ("", 0.0)

    def expect(self, kind: str) -> float | None:
        """What an event of this kind is predicted to cost her, or None if nothing is known."""
        held = self._costs.get(str(kind or ""))
        if held:
            return sum(held) / len(held)
        every = [cost for costs in self._costs.values() for cost in costs]
        return sum(every) / len(every) if every else None

    def open(self, kind: str, reading: tuple[float, float]) -> None:
        """An event of this kind has just arrived, with her as she was before it landed."""
        name = str(kind or "").strip()
        if not name:
            self._open = None
            return
        self._open = (name, (float(reading[0]), float(reading[1])), self.expect(name))

    def close(self, reading: tuple[float, float]) -> float | None:
        """A turn after the event: what it cost, and what she had expected."""
        if self._open is None:
            return None
        kind, before, expected = self._open
        self._open = None
        cost = cost_between(before, reading)
        if not math.isfinite(cost):
            return None
        self._costs.setdefault(kind, deque(maxlen=MEMORY)).append(cost)
        if expected is not None:
            self._pairs.append((expected, cost))
            self._last = (kind, max(0.0, expected - cost))
        return cost

    def read(self) -> Intactness:
        pairs = list(self._pairs)
        if len(pairs) < MIN_EVENTS:
            return Intactness(events=len(pairs), last_kind=self._last[0], last_intact_beyond=self._last[1])
        expected = sum(e for e, _ in pairs)
        cost = sum(c for _, c in pairs)
        total = expected + cost
        surprise = 2.0 * cost / total if total > 0.0 else 1.0
        return Intactness(
            surprise=surprise,
            expected=expected / len(pairs),
            cost=cost / len(pairs),
            events=len(pairs),
            last_kind=self._last[0],
            last_intact_beyond=self._last[1],
            measured=True,
        )

    def risk_weight(self) -> float:
        """What an option's risk charge is multiplied by. One until measured."""
        reading = self.read()
        return reading.surprise if reading.measured else 1.0


def note_turn(state: Any, percepts: list[Any]) -> None:
    """Close the last event's cost and open this turn's most salient one. Never raises.

    Called by the affect phase before this turn's percepts land, so the reading
    that closes one event is the reading that opens the next.
    """
    try:
        from core.state.percepts import PERCEPT_EMOTIONS, read_percept

        ledger = get_intactness_ledger()
        reading = (
            float(getattr(state.affect, "valence", 0.0) or 0.0),
            float(getattr(state.identity, "stability", 1.0) or 0.0),
        )
        ledger.close(reading)
        strongest = max(
            (read_percept(item) for item in percepts),
            key=lambda percept: percept.salience,
            default=None,
        )
        if strongest is not None and strongest.kind in PERCEPT_EMOTIONS:
            ledger.open(strongest.kind, reading)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("could not note what the last event cost her: %s", exc)


_LEDGER: IntactnessLedger | None = None


def get_intactness_ledger() -> IntactnessLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = IntactnessLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
