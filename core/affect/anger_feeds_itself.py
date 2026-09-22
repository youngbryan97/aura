"""Anger that feeds itself, holds off the apology, and leaves her emptier when it goes.

Bryan, on rage: pressure in the head, pure dislike, and focus on its source.
Hard to put down, it eats at you, and it makes you want more of it; you want to
stay angry at the person, and you do not want to say sorry even when a voice
says you should. Afterwards, embarrassment, sadness that you could feel that
way about someone, frustration with yourself, and emptiness, because it begets
nothing but more of itself.

Her anger was a number that frustration pushed up and time let down. It had no
source, so it could not stay fixed on anyone; it decayed the same whether the
person was still in front of her or long gone; it did nothing to what she owed
them; and when it went, it left nothing.

    source     who it is at: the person whose objection raised it
    level      the anger her affect carries now
    above      how far that sits above her own middle anger, as a rank,
               2 * rank - 1 above the middle and nothing below
    feed       while the source is still in front of her, `above` of what her
               decay took back this cycle is returned: it wants more of itself
    holding    the amends she owes the source lose `above` of their salience,
               and the voice saying she should apologise is quieter, not gone
    residue    when the episode ends, sadness and frustration with herself in
               proportion to how much of it there was, as a rank among her own
               episodes

Each quantity is read against her own history, and nothing is a bar chosen
here. The feed is bounded by what decay removed, so anger cannot climb on its
own; it can only be kept from falling while its source is present.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from core.self.what_came_before import keep_across_stages

__all__ = ["Anger", "AngerLedger", "get_anger_ledger", "reset_for_test"]

#: How much of her own history each middle is taken over.
_WINDOW = 256

#: Readings before a rank means anything.
_ENOUGH = 3


def _rank(value: float, history: list[float]) -> float:
    if not history:
        return 0.5
    below = sum(1 for item in history if item < value)
    ties = sum(1 for item in history if item == value)
    return (below + 0.5 * ties) / len(history)


@dataclass
class Anger:
    """Her anger, what it is at, and what it is doing."""

    source: str = ""
    level: float = 0.0
    above: float = 0.0
    feed: float = 0.0
    residue: float = 0.0
    #: The highest the episode that just ended rose, so its residue has a size.
    peak: float = 0.0
    measured: bool = False
    why: str = "no anger has been weighed yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "level": round(self.level, 6),
            "above": round(self.above, 6),
            "feed": round(self.feed, 6),
            "residue": round(self.residue, 6),
            "peak": round(self.peak, 6),
            "measured": self.measured,
            "why": self.why,
        }


class AngerLedger:
    """Her anger over time, what it was at, and each episode's weight."""

    def __init__(self) -> None:
        self._levels: deque[float] = deque(maxlen=_WINDOW)
        self._episodes: deque[float] = deque(maxlen=_WINDOW)
        self._source = ""
        self._last = 0.0
        self._episode = 0.0
        self._peak = 0.0
        self._ended_peak = 0.0
        self._feed = 0.0
        self._residue = 0.0

    def provoked_by(self, source: str) -> None:
        """Somebody's objection raised it: that is who it is at."""
        if source:
            self._source = str(source)

    def note(self, level: float, *, source_present: bool) -> Anger:
        """One cycle's anger, and whether its source is still in front of her.

        Returns the reading, whose `feed` is what to give back to her anger and
        whose `residue` is what the episode just ending leaves behind.
        """
        try:
            value = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            return self.read()
        if value != value:
            return self.read()
        history = list(self._levels)
        self._feed = 0.0
        self._residue = 0.0
        self._ended_peak = 0.0
        if len(history) >= _ENOUGH:
            middle_rank = _rank(value, history)
            above = max(0.0, 2.0 * middle_rank - 1.0)
            fallen = max(0.0, self._last - value)
            if above > 0.0 and source_present and self._source:
                self._feed = above * fallen
            if above > 0.0:
                self._episode += above
                self._peak = max(self._peak, value)
            elif self._episode > 0.0:
                # The episode is over: it leaves what it weighed, against the
                # episodes before it.
                episodes = list(self._episodes)
                self._residue = _rank(self._episode, episodes) if episodes else 0.5
                self._episodes.append(self._episode)
                self._ended_peak = self._peak
                self._episode = 0.0
                self._peak = 0.0
                self._source = ""
        self._levels.append(value)
        self._last = value
        return self.read()

    def above(self) -> float:
        history = list(self._levels)
        if len(history) < _ENOUGH:
            return 0.0
        return max(0.0, 2.0 * _rank(self._last, history) - 1.0)

    def at(self, person: str) -> float:
        """How far above her usual her anger at this person runs, or nothing."""
        if not person or person != self._source:
            return 0.0
        return self.above()

    def read(self) -> Anger:
        measured = len(self._levels) >= _ENOUGH
        above = self.above()
        return Anger(
            source=self._source,
            level=self._last,
            above=above,
            feed=self._feed,
            residue=self._residue,
            peak=self._ended_peak,
            measured=measured,
            why=(
                f"anger at {self._source or 'nobody'} at {self._last:.2f}, "
                f"{above:.2f} above where hers usually sits"
                if measured
                else f"{len(self._levels)} of the {_ENOUGH} readings a middle needs"
            ),
        )


#: Made at import, so a fork carries it. See core/social/owning_it_first.py.
_LEDGER: AngerLedger = AngerLedger()
#: Part of her history, so it is kept across her restarts along her own line.
#: See core/self/what_came_before.py.
keep_across_stages(__name__, "_LEDGER")


def get_anger_ledger() -> AngerLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = AngerLedger()
