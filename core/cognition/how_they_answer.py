"""Whether whoever answers her is choosing, and whether for her or against her.

After she acts, something else has its turn: a board deals a tile, a person replies, an opponent moves, a page
reloads. Her look ahead used to take that turn as luck, averaging over every way it could go. Luck and an opponent
can offer the same answers in the same proportions. What tells them apart is which answer comes. Put each answer
that could have come in order by her own judgement of where it leaves her: an answer chosen against her comes from
the worst end, one chosen for her from the best end, and luck from anywhere.

So she keeps where each answer that came fell among those that could have, 0 the worst for her and 1 the best. Under
luck that place averages a half. Where an answerer picks the worst for her some share of the time and answers by
luck otherwise, its place averages a half less half that share, so the record gives the share directly: twice the
distance from a half. Her look ahead then weighs their turn by it: the worst for her by that share, the average by the rest. "If I go
here and they come there" is then asked as it should be: of somebody choosing, as far as they have been seen to
choose, and of luck only as far as it has been luck.

What she has seen of it is counted against how little that may be: a lean within the record's own spread is no lean.
Nothing here knows who answers or what about. A judgement of where each answer leaves her is all it needs.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["KEPT_AT_MOST", "SPREADS", "HowTheyAnswer", "where_it_fell"]

#: The answers kept: enough that the record's spread is small, few enough that an answerer who changes is seen to.
KEPT_AT_MOST = 200
#: How many of luck's own spreads a lean must lie beyond to be one: two, the usual bar for telling a difference from
#: chance.
SPREADS = 2.0


def where_it_fell(could: Sequence[float], came: float) -> float | None:
    """Where an answer fell among the answers that could have come, by her judgement of each: 0 the worst of them for
    her, 1 the best, an answer level with others in the middle of their places. None where nothing could have gone
    differently."""
    values = sorted(float(v) for v in could)
    if len(values) < 2 or values[0] == values[-1]:
        return None
    below = sum(1 for v in values if v < came)
    level = sum(1 for v in values if v == came)
    place = below + (level - 1) / 2.0 if level else below - 0.5
    return min(1.0, max(0.0, place / (len(values) - 1)))


def _lucks_spread(ways: int) -> float:
    """How widely an answer's place spreads when it comes by luck, from among ``ways`` evenly spaced places."""
    return math.sqrt((ways + 1) / (12.0 * max(1, ways - 1)))


@dataclass
class HowTheyAnswer:
    """Where the answers that came fell among those that could have, each with how many could have."""

    fell: list[float] = field(default_factory=list)
    ways: list[int] = field(default_factory=list)

    def answered(self, could: Sequence[float], came: float) -> None:
        """One answer: her judgement of where each answer that could have come leaves her, and of the one that came."""
        place = where_it_fell(could, came)
        if place is None:
            return
        self.fell.append(place)
        self.ways.append(len(could))
        del self.fell[:-KEPT_AT_MOST]
        del self.ways[:-KEPT_AT_MOST]

    def leaning(self) -> float:
        """How far they choose, and which way: -1 always the worst for her, 1 always the best, 0 luck. Only what lies
        beyond luck's own spread, twice over, counts."""
        if not self.fell:
            return 0.0
        seen = len(self.fell)
        off = sum(self.fell) / seen - 0.5
        ways = self.ways if len(self.ways) == seen else [2] * seen
        spread = math.sqrt(sum(_lucks_spread(n) ** 2 for n in ways)) / seen
        beyond = max(0.0, abs(off) - SPREADS * spread)
        return max(-1.0, min(1.0, math.copysign(2.0 * beyond, off)))

    def weigh(self, values: Sequence[float], shares: Sequence[float]) -> float:
        """What their turn comes to for her: the average of the ways it can go, moved toward the worst for her (or the
        best) by as much as they have been seen to choose."""
        values = [float(v) for v in values]
        if not values:
            return 0.0
        total = sum(float(s) for s in shares)
        averaged = (sum(float(s) * v for s, v in zip(shares, values, strict=False)) / total if total > 0
                    else sum(values) / len(values))
        lean = self.leaning()
        if not lean:
            return averaged
        extreme = min(values) if lean < 0 else max(values)
        return averaged + abs(lean) * (extreme - averaged)

    def says(self) -> str:
        """In words, for whoever has to answer for it."""
        lean = self.leaning()
        if not self.fell:
            return "I haven't seen how this answers me yet"
        if not lean:
            return f"what answers me here looks like luck ({len(self.fell)} answers seen)"
        way = "against me" if lean < 0 else "my way"
        return f"what answers me here chooses {way} about {abs(lean):.0%} of the time ({len(self.fell)} answers seen)"

    def as_memory(self) -> list[list[float]]:
        return [[round(place, 4), n] for place, n in zip(self.fell, self.ways, strict=False)]

    @classmethod
    def from_memory(cls, held: Any) -> HowTheyAnswer:
        if not isinstance(held, list):
            return cls()
        kept = cls()
        for one in held[-KEPT_AT_MOST:]:
            try:
                place, n = one
                kept.fell.append(min(1.0, max(0.0, float(place))))
                kept.ways.append(max(2, int(n)))
            # not a failure: a place that is not a number is not one she kept.
            except (TypeError, ValueError):
                continue
        return kept
