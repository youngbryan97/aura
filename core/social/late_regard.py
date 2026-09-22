"""Regard that arrived after the rise, and what it is worth as evidence.

Logic's "Man of the Year" is a record about being told you are the man of the
year by people who were not there. "They said I couldn't do it back when I was
broke going through it... but you wasn't there in the beginning, nowhere to be
found when I was down, but you show up when I'm winning." Its vocal sits 23 ms
behind the grid, the only one of the seven measured here that does; the praise
arrives late in the song as well.

The complaint is not that the praise is false. It is that praise which begins
after a rise is weak evidence about the person being praised, because it is
also what praise does when it is tracking the rise. Somebody who was there at
the low has shown the thing the arrival cannot.

So each person's regard carries a weight, and the weight is a fact about when
their regard began against her own standing at the time:

    stood_before   people whose regard was already there at or below the
                   standing she sits at now
    arrived_after  people whose regard began above it
    weight         what one person's regard is worth as evidence about her,
                   from the lowest standing at which they have been present

A person who arrives at a peak starts light. Staying through a fall is what
makes them heavy, and it is the only thing that does — there is no decay with
time, no credit for volume, and nothing here reads what anybody said.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "LateRegard",
    "RegardLedger",
    "get_regard_ledger",
    "reset_for_test",
]

#: Regard below this is not regard, and does not start anybody's record. It is
#: the midpoint of the 0-to-1 readings the identity carries.
PRESENT: float = 0.5

#: People before the shares are a reading. Two people are two anecdotes.
MIN_PEOPLE: int = 3

#: How far below their arrival a person has to be present before their weight
#: is back at one, as a share of her own range. Someone who arrived at her
#: best and has since seen her at her worst is worth what an early arrival is
#: worth; half of the way back is worth half of the difference.
FULL_RETURN: float = 1.0


@dataclass
class _Person:
    """When somebody's regard began, and the lowest she has been since."""

    arrived_at: float
    lowest_since: float
    turns: int = 0


@dataclass
class LateRegard:
    """What the people who regard her have seen of her."""

    stood_before: int = 0
    arrived_after: int = 0
    #: The share of them who began above where she stands now.
    late_share: float = 0.0
    #: The mean weight over everybody, for the record rather than for a gate.
    mean_weight: float = 1.0
    people: int = 0
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "stood_before": self.stood_before,
            "arrived_after": self.arrived_after,
            "late_share": round(self.late_share, 4),
            "mean_weight": round(self.mean_weight, 4),
            "people": self.people,
            "measured": self.measured,
        }


class RegardLedger:
    """Her standing turn by turn, and when each person's regard began against it."""

    def __init__(self) -> None:
        self._people: dict[str, _Person] = {}
        self._standing: float | None = None
        self._seen_low: float | None = None
        self._seen_high: float | None = None

    # ── what the turn tells it ──────────────────────────────────────────

    def note_standing(self, standing: float) -> None:
        """Where she stands this turn. Everything else is read against this.

        This does not lower anybody's low. "Nowhere to be found when I was
        down" is the line: a person is present at a standing when they showed
        regard at it, not when they happened to exist while she was there.
        """
        value = _clamp(standing)
        self._standing = value
        self._seen_low = value if self._seen_low is None else min(self._seen_low, value)
        self._seen_high = value if self._seen_high is None else max(self._seen_high, value)

    def note_regard(self, person: str, regard: float) -> None:
        """One person's regard this turn, at the standing the turn is at."""
        name = str(person or "").strip()
        if not name or float(regard) < PRESENT:
            return
        standing = self._standing if self._standing is not None else PRESENT
        held = self._people.get(name)
        if held is None:
            self._people[name] = _Person(arrived_at=standing, lowest_since=standing, turns=1)
            return
        held.turns += 1
        held.lowest_since = min(held.lowest_since, standing)

    # ── what it answers ─────────────────────────────────────────────────

    def weight(self, person: str) -> float:
        """What this person's regard is worth as evidence about her, in [0, 1].

        One when their regard was already there at or below where she stands
        now, and rising from the floor as the lowest standing they have been
        present at comes down toward where she was when they arrived.
        """
        held = self._people.get(str(person or "").strip())
        if held is None:
            return 1.0
        standing = self._standing if self._standing is not None else PRESENT
        if held.arrived_at <= standing + 1e-9:
            return 1.0
        climbed = held.arrived_at - standing
        returned = held.arrived_at - held.lowest_since
        if climbed <= 1e-9:
            return 1.0
        share = min(1.0, returned / (climbed * FULL_RETURN))
        return _clamp(share)

    def read(self) -> LateRegard:
        if not self._people:
            return LateRegard()
        standing = self._standing if self._standing is not None else PRESENT
        before = sum(1 for p in self._people.values() if p.arrived_at <= standing + 1e-9)
        after = len(self._people) - before
        weights = [self.weight(name) for name in self._people]
        return LateRegard(
            stood_before=before,
            arrived_after=after,
            late_share=after / len(self._people),
            mean_weight=sum(weights) / len(weights),
            people=len(self._people),
            measured=len(self._people) >= MIN_PEOPLE,
        )


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, min(1.0, number))


_LEDGER: RegardLedger | None = None


def get_regard_ledger() -> RegardLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = RegardLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
