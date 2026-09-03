"""Choices that are not choices, and lines that change what a thing is.

Two small rules of draughts, both of which turn out to be about everything.

The first is that a capture is compulsory. Where one is available you must take
it, so there is nothing to weigh — and a player who stops to think there has
spent thought on a decision that was already made. Recognising that costs
nothing and saves all of it, and the same shape is everywhere: one option left,
a required field, a lock only one process can hold, a step with a single
successor. Deliberating over a settled thing is the most common way effort is
wasted, because it looks exactly like diligence.

The second is that a man reaching the far row becomes a king. Nothing about it
grew; it crossed a line, and on the other side it is a different kind of thing
with different moves. A file over a size stops being a file and becomes a
problem. A queue over a length stops being a queue and becomes an outage. A
number of retries stops being caution and becomes a loop.

She had neither. Every situation was weighed even when it had one answer, and
every quantity was a quantity all the way up — so a thing that had crossed
into being something else was still being reasoned about as more of what it
had been.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ALine", "WhatItBecomes", "nothing_to_decide"]


def nothing_to_decide(acts: Sequence[Any], *, compelled: Callable[[Any], bool] | None = None) -> Any:
    """The act, where there is only one, or nothing where there is a choice.

    Two ways a decision can already be made: only one thing is available, or
    the situation compels one of several. Both are worth spotting before
    anything is weighed, because weighing a settled thing is effort spent
    looking like diligence.
    """
    if compelled is not None:
        forced = [one for one in acts if compelled(one)]
        if len(forced) == 1:
            return forced[0]
        if forced:
            # Several compelled and she must take one of them — still a
            # choice, but only among these.
            return None
    got = list(acts)
    return got[0] if len(got) == 1 else None


@dataclass(frozen=True)
class ALine:
    """A place where a quantity stops being a quantity."""

    name: str
    at: float
    becomes: str

    def crossed_by(self, value: float) -> bool:
        return float(value) >= self.at

    def describe(self, value: float) -> str:
        if self.crossed_by(value):
            return f"{self.name} at {value:g} is past {self.at:g} — it is {self.becomes} now"
        return f"{self.name} at {value:g}, {self.at - float(value):g} short of {self.becomes}"


@dataclass
class WhatItBecomes:
    """Lines she has learned, and what is on the other side of them."""

    lines: list[ALine] = field(default_factory=list)
    #: name -> value -> whether it behaved differently, seen.
    _seen: dict[str, list[tuple[float, bool]]] = field(default_factory=dict)

    def a_line(self, name: str, *, at: float, becomes: str) -> None:
        self.lines.append(ALine(name=str(name), at=float(at), becomes=str(becomes)))

    def it_behaved_differently(self, name: str, *, at: float, differently: bool) -> None:
        """One sighting: this much of it, and whether it acted like the other thing."""
        self._seen.setdefault(str(name), []).append((float(at), bool(differently)))

    def where_the_line_is(self, name: str) -> float:
        """The value the behaviour changed at, found rather than declared.

        Between the largest that behaved normally and the smallest that did
        not. Nothing where they overlap — a quantity whose behaviour does not
        change cleanly has no line in it, and inventing one would be worse
        than saying so.
        """
        seen = self._seen.get(str(name)) or []
        normal = [one for one, odd in seen if not odd]
        odd = [one for one, was in seen if was]
        if not normal or not odd:
            return 0.0
        if max(normal) >= min(odd):
            return 0.0
        return (max(normal) + min(odd)) / 2.0

    def what_it_is_now(self, name: str, value: float) -> str:
        """What this counts as, given how much of it there is."""
        for line in self.lines:
            if line.name == str(name) and line.crossed_by(value):
                return line.becomes
        found = self.where_the_line_is(name)
        if found and float(value) >= found:
            return f"something else (past {found:g})"
        return str(name)

    def has_crossed(self, name: str, value: float) -> bool:
        return self.what_it_is_now(name, value) != str(name)
