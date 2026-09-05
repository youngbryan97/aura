"""What worked the last time she was in a position like this one.

Working a world out is not the same as getting good at it. She can know
exactly how a thing moves and still pay the full price of deciding every
single time, which is what she was doing: read, model, look ahead, deliberate,
act — the same deliberation for the hundredth position of a kind as for the
first. People do not do that. A player who has met a shape a few times stops
deciding and starts recognising, and the deciding machinery is freed for the
positions that are actually new.

That is all this is. A position has a kind — where the largest thing is, how
full it is, what is against which edge — and a kind that has come up before
with a move that kept working is a move she can make without thinking about
it again. What she has is experience; what this turns it into is skill.

Nothing here is a cache of answers. An entry earns its way in by working
several times, and loses its place the moment it stops, so a world that
changes under her costs her a few moves rather than a fixed habit. And the
kinds are read off the position itself, so nothing about this knows what any
particular world is: a board, a form, a map and a queue all have position
kinds, and all of them repeat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "KNOWN_WELL_ENOUGH",
    "REMEMBERED_POSITIONS",
    "WORKED_OFTEN_ENOUGH",
    "WhatWorkedBefore",
]

logger = logging.getLogger("Aura.WhatWorkedBefore")

#: How many times a move has to have worked from a kind of position before it
#: is one she can make without deciding again. Twice is a coincidence; a third
#: time is the beginning of a habit worth having.
KNOWN_WELL_ENOUGH = 3

#: The share of those times it has to have left her better off. A move that
#: helps two times in three is not a skill, it is a gamble she keeps taking.
WORKED_OFTEN_ENOUGH = 0.7

#: How many kinds of position she keeps. Enough that a long run in one world
#: builds real fluency, bounded so a run in a world with endless kinds does
#: not grow without limit.
REMEMBERED_POSITIONS = 512


@dataclass
class WhatWorkedBefore:
    """Kinds of position she has met, and what she did that helped."""

    #: kind of position → move → [times tried, times it left her better off]
    known: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    #: Kinds in the order they were last touched, so the oldest goes first.
    _order: list[str] = field(default_factory=list)
    #: Moves taken on recognition rather than by deciding.
    recognised: int = 0

    # ── building it ──────────────────────────────────────────────────────

    def forget_what_was_read_differently(self) -> None:
        """Drop it all, because it is filed under names that mean nothing now.

        What worked before is looked up by the SHAPE of the situation, and a
        shape is a description of a grid. Written down while she was reading a
        board four by seven that is really four by four, every name here
        describes a thing that does not exist — so none of it will ever be
        recognised, and any of it that is will be recognised wrongly.
        """
        if not self.known:
            return
        self.known.clear()
        self._order.clear()
        self.recognised = 0

    def learned(self, kind: str, move: str, better: bool) -> None:
        """One position of a kind, one move, and whether it helped."""
        key = str(kind or "").strip()
        name = str(move or "").strip().lower()
        if not key or not name:
            return
        seen = self.known.setdefault(key, {})
        counts = seen.setdefault(name, [0, 0])
        counts[0] += 1
        if better:
            counts[1] += 1
        self._touch(key)

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        while len(self._order) > REMEMBERED_POSITIONS:
            self.known.pop(self._order.pop(0), None)

    # ── using it ─────────────────────────────────────────────────────────

    def suggests(self, kind: str, among: tuple[str, ...] = ()) -> str:
        """The move this kind of position has taught her, if it has taught one.

        ``among`` is what is actually available now. A move that worked a
        hundred times is worth nothing if it is not one she can make here, and
        offering it anyway would be reciting rather than recognising.
        """
        seen = self.known.get(str(kind or "").strip())
        if not seen:
            return ""
        allowed = {str(name).strip().lower() for name in among}
        best = ""
        best_share = 0.0
        for name, (tried, worked) in seen.items():
            if allowed and name not in allowed:
                continue
            if tried < KNOWN_WELL_ENOUGH:
                continue
            share = worked / tried
            if share < WORKED_OFTEN_ENOUGH or share <= best_share:
                continue
            best, best_share = name, share
        return best

    def took(self, kind: str) -> None:
        """Note that a move was made on recognition rather than by deciding."""
        self.recognised += 1
        self._touch(str(kind or "").strip())

    def fluency(self) -> float:
        """The share of the kinds she has met that she now answers on sight."""
        if not self.known:
            return 0.0
        fluent = sum(1 for kind in self.known if self.suggests(kind))
        return fluent / len(self.known)

    def says(self) -> str:
        """What she has got good at, for whoever has to answer for it."""
        if not self.known:
            return "no kind of position here has taught her anything yet"
        fluent = sum(1 for kind in self.known if self.suggests(kind))
        if not fluent:
            return f"{len(self.known)} kind(s) of position met, none of them settled yet"
        return (
            f"{fluent} of {len(self.known)} kind(s) of position answered on sight, "
            f"{self.recognised} move(s) made that way"
        )

    # ── keeping it ───────────────────────────────────────────────────────

    def as_memory(self) -> dict[str, Any]:
        return {
            "known": {kind: {m: list(c) for m, c in moves.items()} for kind, moves in self.known.items()},
            "order": list(self._order),
            "recognised": self.recognised,
        }

    @classmethod
    def from_memory(cls, held: Any, trust: float = 1.0) -> "WhatWorkedBefore":
        """What she got good at last time, discounted like anything carried over.

        A habit from yesterday is evidence about today rather than a fact about
        it, and a handful of positions that go badly should be able to overturn
        one — which they can only do if what came back is not overwhelming.
        """
        if not isinstance(held, dict):
            return cls()
        share = max(0.0, min(1.0, float(trust)))
        known: dict[str, dict[str, list[int]]] = {}
        for kind, moves in (held.get("known") or {}).items():
            if not isinstance(moves, dict):
                continue
            kept: dict[str, list[int]] = {}
            for move, counts in moves.items():
                if not isinstance(counts, (list, tuple)) or len(counts) != 2:
                    continue
                tried = int(round(float(counts[0]) * share))
                worked = int(round(float(counts[1]) * share))
                if tried > 0:
                    kept[str(move)] = [tried, min(tried, worked)]
            if kept:
                known[str(kind)] = kept
        order = [str(kind) for kind in (held.get("order") or ()) if str(kind) in known]
        order += [kind for kind in known if kind not in order]
        return cls(known=known, _order=order[-REMEMBERED_POSITIONS:])
