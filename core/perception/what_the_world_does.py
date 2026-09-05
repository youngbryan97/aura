"""What the world does on its own, between her acts.

She works out what her own acts do and then plans as though that were the
whole story. It is not. A board deals a tile, a page gains a row, a queue takes
another customer — and a future worked out as if none of that happens is a
future that cannot happen. She already tolerates arrivals when scoring a rule,
because a dealt tile is not a rule's mistake. Tolerating a thing is not the
same as knowing it, and the information was being thrown away every move.

It was there all along. The difference between what a rule said would happen
and what she actually saw IS what the world did, separated out for free by the
same comparison that scores the rule. This keeps that difference.

What it learns is what arrives and how often, not where — where is read off
the position each time, because the places something can arrive in are wherever
there is room, and that changes every move. Nothing here knows what a tile is.
It knows that things turn up she did not put there, what they tend to say, and
how often that happens.

Measured 2026-08-27, five games each, run to a dead board, the world model as
its own axis so what it is worth can be read off rather than inferred:

    random                    best 128    total  288    185 moves
    her scoring               best 1024   total 1648    822 moves
    her scoring + this        best 1024   total 1994    996 moves
    her line                  best 512    total 1074    535 moves
    her line + this           best 1024   total 2972   1484 moves, max 2048

The last row is the first time anything here has reached a 2048 tile. Held
against a plan, knowing what the world does between her acts nearly triples
what a game comes to and doubles how long she stays in it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.perception.what_is_there import Arrangement, Cell

__all__ = ["MOST_WAYS", "ENOUGH_TO_EXPECT", "WhatTheWorldDoes"]

logger = logging.getLogger("Aura.WhatTheWorldDoes")

#: How many of the ways the world might answer she works through.
#:
#: Every empty place is one, so a wide-open board has a dozen. Measured
#: 2026-08-27 rather than guessed, six games each, run to a dead board, with
#: the depth of the search left free to adapt to what each level costs:
#:
#:     2 ways    median best  512    total 1219
#:     4 ways    median best  768    total 1726
#:     8 ways    median best 1024    total 1966
#:
#: Four was the guess and it was too few. Each doubling buys less than the
#: last while costing twice as much, and by eight the search is choosing a
#: shallower depth to pay for the breadth — which is where the two stop
#: trading in her favour.
MOST_WAYS = 8

#: How many acts have to have been watched before what turned up in them is
#: worth planning around. Below this, one unlucky look decides everything.
ENOUGH_TO_EXPECT = 6


@dataclass
class WhatTheWorldDoes:
    """What turns up that she did not put there, and how often."""

    #: What arrived, and how many times each thing did.
    arrives: dict[str, int] = field(default_factory=dict)
    #: Acts watched, and acts after which something turned up.
    acts: int = 0
    acts_with_arrivals: int = 0

    # ── learning ─────────────────────────────────────────────────────────

    def forget_what_was_read_differently(self) -> None:
        """Drop what turned up, because it was read out of a different thing.

        What the world does on its own is worked out by comparing what a rule
        expected against what was really there — two readings, both laid into
        a grid. Read through the wrong one, what "turned up on its own" is
        whatever the grid was wrongly covering. One world file on this machine
        has her believing that History, Help, File and Edit arrive by
        themselves, which is a browser's menu bar read as though it were the
        game.
        """
        if not self.arrives and not self.acts:
            return
        self.arrives.clear()
        self.acts = 0
        self.acts_with_arrivals = 0

    def watched(self, expected: Arrangement | None, seen: Arrangement) -> None:
        """One act, what a rule said it would do, and what really happened.

        Whatever is in the second and not the first is the world's doing. This
        is the same comparison that scores the rule, read the other way round.
        """
        if expected is None:
            return
        self.acts += 1
        was = {(cell.row, cell.column): cell.says for cell in expected.cells}
        turned_up = [
            cell.says
            for cell in seen.cells
            if was.get((cell.row, cell.column)) != cell.says
        ]
        if not turned_up:
            return
        self.acts_with_arrivals += 1
        for said in turned_up:
            self.arrives[said] = self.arrives.get(said, 0) + 1

    # ── using it ─────────────────────────────────────────────────────────

    def how_often(self) -> float:
        """The share of her acts after which something turned up."""
        return self.acts_with_arrivals / self.acts if self.acts else 0.0

    def worth_expecting(self) -> bool:
        """Whether she has watched enough for this to be worth planning around."""
        return self.acts >= ENOUGH_TO_EXPECT and bool(self.arrives)

    def what_arrives(self) -> tuple[tuple[str, float], ...]:
        """The things that turn up, commonest first, with how often each does."""
        total = sum(self.arrives.values())
        if not total:
            return ()
        ordered = sorted(self.arrives.items(), key=lambda thing: -thing[1])
        return tuple((said, times / total) for said, times in ordered)

    def might_do(self, state: Arrangement) -> tuple[tuple[Arrangement, float], ...]:
        """The ways this position might look once the world has had its turn.

        Each way carries the share of the time it happens, so a caller can
        take the average rather than the best. Bounded, because averaging over
        four of the ways and over twelve of them give nearly the same number
        and one costs three times as much.
        """
        if not self.worth_expecting():
            return ()
        room = [
            (row, column)
            for row in range(state.rows)
            for column in range(state.columns)
            if state.at(row, column) is None
        ]
        if not room:
            return ()
        arrivals = self.what_arrives()
        chance = self.how_often()
        # Spread the sample across the room there is rather than taking the
        # first few places, which on a board that fills from one end would
        # only ever look at one corner of it.
        step = max(1, len(room) // MOST_WAYS)
        places = room[::step][:MOST_WAYS]
        ways: list[tuple[Arrangement, float]] = []
        for place in places:
            for said, share in arrivals[:2]:
                landed = Arrangement(
                    state.rows,
                    state.columns,
                    state.cells + (Cell(place[0], place[1], said, (0.0, 0.0)),),
                    state.down_at,
                    state.across_at,
                )
                ways.append((landed, chance * share / len(places)))
        # And the chance it does nothing at all, which is a way things go too.
        stayed = 1.0 - sum(share for _way, share in ways)
        if stayed > 0.0:
            ways.append((state, stayed))
        return tuple(ways)

    def says(self) -> str:
        """What the world does without her, for whoever has to answer for it."""
        if not self.worth_expecting():
            return "what this does on its own is not worked out yet"
        arrivals = ", ".join(f"{said} ({share:.0%})" for said, share in self.what_arrives()[:3])
        return f"something turns up after {self.how_often():.0%} of my acts: {arrivals}"

    # ── keeping it ───────────────────────────────────────────────────────

    def as_memory(self) -> dict[str, Any]:
        return {
            "arrives": dict(self.arrives),
            "acts": self.acts,
            "acts_with_arrivals": self.acts_with_arrivals,
        }

    @classmethod
    def from_memory(cls, held: Any, trust: float = 1.0) -> WhatTheWorldDoes:
        """What the world did last time, discounted like anything carried over."""
        if not isinstance(held, dict):
            return cls()
        share = max(0.0, min(1.0, float(trust)))
        counted = held.get("arrives")
        arrives = (
            {
                str(said): int(round(float(times) * share))
                for said, times in counted.items()
                if isinstance(times, (int, float)) and round(float(times) * share) > 0
            }
            if isinstance(counted, dict)
            else {}
        )
        acts = int(round(float(held.get("acts") or 0) * share))
        with_arrivals = int(round(float(held.get("acts_with_arrivals") or 0) * share))
        return cls(arrives=arrives, acts=acts, acts_with_arrivals=min(acts, with_arrivals))
