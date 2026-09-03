"""Working out what is hidden from what nobody was able to produce.

Four people playing Cluedo. The whole of the reasoning is a grid on a
notepad — every card against every player, marked with a tick or a cross — and
almost every mark on it is a cross. A cross is not something anybody said. It
is what follows from somebody being unable to answer: asked about three
things and showing none of them, they hold none of the three, and that is now
known about all three at once.

Nothing about that is a board game. It is how anybody finds a thing that is
not in front of them. Which component could not have produced this output.
Which of the six services was not called. Which file cannot be the one, given
that grep found nothing in four of them. The evidence is a failure to produce,
and reasoning from it is not weaker than reasoning from a sighting — it is
usually stronger, because it says something about several candidates at once
while a sighting says something about one.

She could not do this at all. Everything she had reasons from what happened;
nothing reasoned from what did not. So a thing she had ruled out stayed a
thing she merely had no evidence for, and the answer that was the only one
left was never noticed to be the only one left.

Two kinds of evidence and one closure. Somebody asked about several things and
produced none of them holds none of them. Somebody who produced one of them,
without saying which, holds at least one — worth less, and worth keeping,
because it becomes worth a great deal once the others are crossed off. And a
candidate nobody can hold is the hidden one. The answer is arrived at by
elimination and never by being seen, which is what the grid is for.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhatIsHidden"]


@dataclass
class WhatIsHidden:
    """What each party can still be holding, and what follows from that."""

    #: Everything the hidden thing might be.
    candidates: tuple[str, ...] = ()
    #: Who might be holding things.
    parties: tuple[str, ...] = ()
    #: (party, candidate) pairs known to be impossible.
    ruled_out: set[tuple[str, str]] = field(default_factory=set)
    #: Parties known to hold something, without it being known which.
    holds_one_of: list[tuple[str, frozenset[str]]] = field(default_factory=list)

    def could_not_produce(self, party: str, asked: Iterable[str]) -> None:
        """Somebody asked about several things produced none of them.

        The strongest kind of evidence there is here, because it settles
        several candidates at once where a sighting settles one.
        """
        for one in asked:
            if one in self.candidates:
                self.ruled_out.add((party, one))

    def produced_one_of(self, party: str, asked: Iterable[str]) -> None:
        """Somebody produced one of these, without it being known which.

        Worth little on its own and a great deal later: once all but one of
        them is crossed off for that party, it says which.
        """
        among = frozenset(one for one in asked if one in self.candidates)
        if among:
            self.holds_one_of.append((str(party), among))

    def someone_holds(self, party: str, one: str) -> None:
        """Seen outright: this party holds this, so nobody else does."""
        if one not in self.candidates:
            return
        for other in self.parties:
            if other != party:
                self.ruled_out.add((other, one))
        self.holds_one_of.append((str(party), frozenset({one})))

    def _settle(self) -> None:
        """Follow what is known until nothing more follows.

        A party known to hold one of a group, all but one of which is crossed
        off for them, holds that one — and then nobody else does. That in turn
        crosses things off, which can settle another group. So it runs to a
        standstill rather than once.
        """
        again = True
        while again:
            again = False
            for party, among in list(self.holds_one_of):
                left = [
                    one for one in among if (party, one) not in self.ruled_out
                ]
                if len(left) != 1:
                    continue
                only = left[0]
                for other in self.parties:
                    if other != party and (other, only) not in self.ruled_out:
                        self.ruled_out.add((other, only))
                        again = True

    def who_could_hold(self, one: str) -> tuple[str, ...]:
        """The parties not yet ruled out for this."""
        self._settle()
        return tuple(
            party for party in self.parties if (party, one) not in self.ruled_out
        )

    def what_it_must_be(self) -> tuple[str, ...]:
        """Candidates nobody can be holding, which is what is hidden.

        Several is not a failure. It means the evidence so far narrows it to
        these and no further, and saying which of them it is would be making
        something up.
        """
        self._settle()
        return tuple(one for one in self.candidates if not self.who_could_hold(one))

    def what_is_still_open(self) -> tuple[str, ...]:
        """Candidates that are neither settled nor eliminated."""
        settled = set(self.what_it_must_be())
        return tuple(
            one
            for one in self.candidates
            if one not in settled and self.who_could_hold(one)
        )

    def what_would_settle_it(self, asking: Sequence[str]) -> list[tuple[str, int]]:
        """Which of these to ask about, by how many crosses an answer could add.

        Asking about something already settled learns nothing however
        interesting it is. What is worth asking about is what is still open
        and held by the most parties, because that is where a failure to
        produce says the most.
        """
        open_now = set(self.what_is_still_open())
        return sorted(
            (
                (one, len(self.who_could_hold(one)))
                for one in asking
                if one in open_now
            ),
            key=lambda one: (-one[1], one[0]),
        )

    def describe(self) -> str:
        must = self.what_it_must_be()
        if len(must) == 1:
            return f"it can only be {must[0]}"
        if must:
            return f"it is one of {', '.join(must)} and nothing says which"
        return f"{len(self.what_is_still_open())} still open"

    def as_memory(self) -> dict[str, Any]:
        return {
            "candidates": list(self.candidates),
            "parties": list(self.parties),
            "ruled_out": [list(one) for one in sorted(self.ruled_out)],
            "holds_one_of": [
                [party, sorted(among)] for party, among in self.holds_one_of
            ],
        }
