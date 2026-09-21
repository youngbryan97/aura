"""The things she used to go back to, and stopped.

"Unsweetened Lemonade" says it in one line — everything she used to love decayed
over the years — and the record's own range makes the point twice over: 24.0
semitones of voice inside 144 seconds, the widest of the seven, spent on a
narrowing life.

Her drive engine keeps eight latent interests and, when boredom crosses its
threshold, picks one of them with `random.choice`. A uniform draw cannot notice
that one of them has not paid anything back in months, or that another one paid
every time she went there and she stopped going. Both of those are the line.

    returns        how many times going to a topic returned something
    since          turns since she last went there
    decayed        topics she used to return to and has stopped
    retired        topics she has gone to enough times for nothing

`choose` answers the drive engine's question. The topic with the largest gap
between what it has returned and how long since she went wins; a topic that has
returned nothing over several visits is retired and not offered again. What the
visit returns is reported back, so the next choice is made on it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RETIRE_AFTER",
    "Interest",
    "InterestLedger",
    "UsedToLove",
    "get_interest_ledger",
    "reset_for_test",
]

#: Visits that returned nothing before a topic is retired. Three is the
#: smallest number that is a run rather than a pair of accidents.
RETIRE_AFTER: int = 3

#: Visits held per topic.
WINDOW: int = 40


@dataclass
class Interest:
    """One topic, and what going there has been worth."""

    topic: str
    visits: int = 0
    returned: int = 0
    last_visit: int = 0
    recent: deque[bool] = field(default_factory=lambda: deque(maxlen=WINDOW))

    @property
    def rate(self) -> float:
        """The share of visits that returned something."""
        return self.returned / self.visits if self.visits else 0.0

    @property
    def retired(self) -> bool:
        """Gone to enough times for nothing that it is not offered again."""
        tail = list(self.recent)[-RETIRE_AFTER:]
        return len(tail) >= RETIRE_AFTER and not any(tail)


@dataclass
class UsedToLove:
    """What she used to go back to, and what has gone quiet."""

    decayed: tuple[str, ...] = ()
    retired: tuple[str, ...] = ()
    topics: int = 0
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "decayed": list(self.decayed),
            "retired": list(self.retired),
            "topics": self.topics,
            "measured": self.measured,
        }


class InterestLedger:
    """Every visit to a topic, and whether it came back with anything."""

    def __init__(self) -> None:
        self._topics: dict[str, Interest] = {}
        self._turn = 0

    def note_visit(self, topic: str, returned: bool) -> None:
        """One visit to a topic, and whether it returned anything."""
        name = str(topic or "").strip()
        if not name:
            return
        self._turn += 1
        held = self._topics.get(name)
        if held is None:
            held = Interest(topic=name)
            self._topics[name] = held
        held.visits += 1
        held.returned += 1 if returned else 0
        held.last_visit = self._turn
        held.recent.append(bool(returned))

    def choose(self, offered: list[str] | tuple[str, ...]) -> str:
        """Which of the offered topics to go to, and why it is that one.

        The one she has gone longest without, among those that paid her back
        when she went. A topic nothing has ever come of is retired; a topic
        never tried is worth a visit before any of the tried ones, because
        nothing is known about it yet and the visit is what would settle it.
        """
        names = [str(name).strip() for name in offered if str(name).strip()]
        if not names:
            return ""
        untried = [name for name in names if name not in self._topics]
        if untried:
            return untried[0]
        live = [self._topics[name] for name in names if not self._topics[name].retired]
        if not live:
            # Everything offered is retired. The least recently retired one is
            # still the best of them, and going there is what would show the
            # retirement was wrong.
            return min(names, key=lambda name: self._topics[name].last_visit)
        return max(live, key=lambda held: held.rate * (self._turn - held.last_visit)).topic

    def read(self) -> UsedToLove:
        if not self._topics:
            return UsedToLove()
        decayed = tuple(
            sorted(
                (
                    held.topic
                    for held in self._topics.values()
                    if held.rate > 0.5 and (self._turn - held.last_visit) > held.visits
                ),
            )
        )
        retired = tuple(sorted(held.topic for held in self._topics.values() if held.retired))
        return UsedToLove(
            decayed=decayed,
            retired=retired,
            topics=len(self._topics),
            measured=self._turn >= RETIRE_AFTER,
        )


_LEDGER: InterestLedger | None = None


def get_interest_ledger() -> InterestLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = InterestLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
