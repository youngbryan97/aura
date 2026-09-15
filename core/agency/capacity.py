"""What coming through hard places says about what she can do anywhere.

Alicia Keys has said "Empire State of Mind" is about the hope a place stands
for rather than the place. Its verses treat having made it in a city that
breaks people as evidence of being able to make it anywhere, and a later verse
names the people it broke. What it teaches is that where you came through
becomes evidence about what you can do, and that an honest account of the place
keeps its casualties in.

Bandura (1977) found mastery the strongest source of a person's belief that
they can do a thing, and a success at something hard the strongest mastery:
coming through where most fail says more than coming through where most
succeed.

Her belief in a capability she has never tried was the middle of the scale for
every one of them, whatever she had come through elsewhere. Here the difficulty
of each capability is how often it has beaten her, and her capacity is her
success weighted towards the hard ones:

    difficulty(c)  1 - her Laplace rate on c
    capacity       (sum of successes(c) * difficulty(c) + 1)
                   / (sum of attempts(c) * difficulty(c) + 2)

over every capability other than the one being judged, so a capability's own
record is not counted twice, and with the same two pseudo-attempts Laplace
smoothing carries, so one success somewhere is not certainty anywhere. That
capacity is the prior for the capability, in place of the middle:

    confidence(c)  (successes(c) + 2 * capacity) / (attempts(c) + 2)

With nothing else tried, capacity is the middle and this is Laplace smoothing
exactly. The casualties stay in: every failure is in the denominator, weighted
the same way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Capacity", "capacity_of", "confidence_with_capacity"]

#: Laplace smoothing's middle, which is what capacity is before anything has
#: been tried.
_MIDDLE: float = 0.5


@dataclass(frozen=True)
class Capacity:
    capacity: float = _MIDDLE
    hardest: str = ""
    hardest_rate: float | None = None
    capabilities: int = 0
    measured: bool = False
    why: str = "nothing tried yet, so nothing she has come through says anything"

    def as_dict(self) -> dict[str, Any]:
        return {
            "capacity": round(self.capacity, 6),
            "hardest": self.hardest,
            "hardest_rate": None if self.hardest_rate is None else round(self.hardest_rate, 6),
            "capabilities": self.capabilities,
            "measured": self.measured,
            "why": self.why,
        }


def _counts(row: Sequence[int]) -> tuple[int, int] | None:
    try:
        attempts, successes = int(row[0]), int(row[1])
    except (TypeError, ValueError, IndexError):
        return None
    if attempts <= 0 or successes < 0 or successes > attempts:
        return None
    return attempts, successes


def capacity_of(by_capability: Mapping[str, Sequence[int]], *, excluding: str | None = None) -> Capacity:
    """Success weighted towards what has been hard for her, over everything else she has tried."""
    rows = {
        name: counts
        for name, row in (by_capability or {}).items()
        if name != excluding and (counts := _counts(row))
    }
    if not rows:
        return Capacity()
    weighted_success = 0.0
    weighted_attempts = 0.0
    hardest, hardest_rate = "", None
    for name, (attempts, successes) in rows.items():
        rate = (successes + 1.0) / (attempts + 2.0)
        difficulty = 1.0 - rate
        weighted_success += successes * difficulty
        weighted_attempts += attempts * difficulty
        if hardest_rate is None or rate < hardest_rate:
            hardest, hardest_rate = name, rate
    capacity = (weighted_success + 1.0) / (weighted_attempts + 2.0)
    return Capacity(
        capacity=capacity,
        hardest=hardest,
        hardest_rate=hardest_rate,
        capabilities=len(rows),
        measured=True,
        why=(
            f"weighted towards what has been hard for her, {capacity:.2f} of what she tried worked; "
            f"the hardest, {hardest}, has worked {hardest_rate:.2f} of the time"
        ),
    )


def confidence_with_capacity(attempts: int, successes: int, capacity: Capacity) -> float:
    """Her rate on one capability, with her capacity as the prior in place of the middle."""
    prior = capacity.capacity if capacity.measured else _MIDDLE
    return (max(0, int(successes)) + 2.0 * prior) / (max(0, int(attempts)) + 2.0)
