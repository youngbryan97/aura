"""Thinking on their time: while the other side thinks, she searches the position she expects to face next.

Her search keeps what it learned of each position between searches (the compiled search's kept table). So a search of
the position after the answer she expects, made while the other side is still choosing it, is not lost when the answer
comes: where they answer as she expected, her own search begins with that position already looked into, and looks
deeper in the same time. Where they answer otherwise, she has lost nothing she would have used.

Which answer she expects is what they answered before in the same position, where she has met it, in this game or an
earlier one (WhatTheyAnsweredWhere, kept with what she learned of the board): a side that chose a move in a position
is likely to choose it again there. Where she has not met the position, it is the answer her own search expected.

Each look on their time lasts about as long as they usually take to answer, measured in this game, and never longer
than a turn of hers, so a look ends about when their move comes and her own search does not wait on it. Nothing here
knows a game.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Pondering", "WhatTheyAnsweredWhere"]

#: Positions remembered at most, the latest met: a game meets about forty of its own.
KEPT_AT_MOST = 20_000


def _key(place: str) -> str:
    """A position as a short name: positions are written out square by square, and thousands are kept."""
    return hashlib.sha1(str(place).encode()).hexdigest()[:16]


@dataclass
class WhatTheyAnsweredWhere:
    """What the other side answered in each position she has met, and how often: for a world whose positions come back
    as the same state, in this game or a later one."""

    answered: dict[str, dict[str, int]] = field(default_factory=dict)

    def they_answered(self, place: str, move: str) -> None:
        key = _key(place)
        counts = self.answered.pop(key, {})
        counts[str(move)] = counts.get(str(move), 0) + 1
        # Met again, a position is the latest met.
        self.answered[key] = counts
        while len(self.answered) > KEPT_AT_MOST:
            del self.answered[next(iter(self.answered))]

    def most_often(self, place: str) -> str | None:
        counts = self.answered.get(_key(place))
        if not counts:
            return None
        return max(sorted(counts), key=lambda move: counts[move])

    def as_bytes(self) -> bytes:
        return json.dumps(self.answered, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> WhatTheyAnsweredWhere:
        try:
            held = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return cls()
        if not isinstance(held, dict):
            return cls()
        return cls(answered={str(key): {str(m): int(n) for m, n in counts.items()}
                             for key, counts in held.items() if isinstance(counts, dict)})


@dataclass
class Pondering:
    """Her look on their time: the one under way, if any, and how many she made."""

    running: Any = None
    looks: int = 0

    def busy(self) -> bool:
        return self.running is not None and not self.running.done()

    def look_for(self, their_answer_times: list[float], her_share: float) -> float:
        """How long one look on their time lasts: about as long as they take to answer, never longer than her turn."""
        if not their_answer_times:
            return her_share
        return max(0.05, min(her_share, statistics.median(their_answer_times)))

    def start(self, pool: Any, family: Any, position: Any, *, seconds: float, weights: Any, earlier: list[Any],
              threads: int, also: Any = None) -> None:
        """A look at ``position`` in the background, where none is under way; and, where ``also`` is another
        position (the one after the answer that would hurt her most, where it is not the one she expects), a look at
        that too, the time shared between them. What they tend to answer schedules where she looks first; what they
        could answer is looked at as well (suggestions of 23 November 2026)."""
        from core.agency.pieces_by_their_kind import FIRST
        from core.agency.searching_by_kind import search

        if self.busy() or pool is None:
            return
        self.looks += 1
        positions = [position] if also is None else [position, also]

        def look() -> Any:
            found = None
            for one in positions:
                # As wary of her own unsure ways as the search of her turn is: the two share what they find.
                found = search(family, one, seconds=seconds / len(positions), weights=weights, earlier=earlier,
                               threads=threads, wary=FIRST)
            return found

        self.running = pool.submit(look)

    def settle(self) -> None:
        """What a finished look found is in her search's table already; only its failure is worth a word."""
        import logging

        if self.running is not None and self.running.done():
            try:
                self.running.result()
            except Exception as exc:  # noqa: BLE001 - a look that failed costs her only the look
                logging.getLogger("Aura.ThinkingOnTheirTime").info("a look on their time did not finish (%s: %s)",
                                                                   type(exc).__name__, exc)
            self.running = None
