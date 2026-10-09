"""Which thing goes where the pointer goes: the pointer's half of finding which thing is hers.

A thing that goes where she takes the pointer is hers, the way a thing a key
moves is. Moved out of core/agency/which_one_answers_to_her.py, whose
`WhichIsHers` takes this as a base: the class had grown past the number of
methods a class may have. The state these read (the pointer's trail, what has
followed it, the belief and since when it was held) is set up by WhichIsHers.
"""
from __future__ import annotations

import bisect
import logging
import math
from typing import Any

logger = logging.getLogger("core.agency.which_one_answers_to_her")

__all__ = ["FollowsThePointer"]


class FollowsThePointer:
    """Where the pointer has been, and the thing that goes with it."""

    def pointed(self, x: float, y: float, at: float) -> None:
        """The pointer was taken to (x, y), in working pixels."""
        self._pointer.append((at, x, y))
        del self._pointer[:-400]

    def _pointer_at(self, at: float) -> tuple[float, float] | None:
        index = bisect.bisect_right(self._pointer, (at, math.inf, math.inf))
        return self._pointer[index - 1][1:] if index else None

    def _with_the_pointer(self, seen: list[tuple[float, float, float]], extent: tuple[float, float] | None) -> tuple[bool | None, bool | None]:
        """Along which axes places seen go with where the pointer was, at whichever delay fits best.

        How long a page takes to answer the pointer is the page's own: one
        follows within a picture, another a few later. Matched at the wrong
        delay, a paddle that goes exactly where the mouse goes lags its
        pointer at every turn and falls short of the test (offline
        2026-10-04, bounce). Each delay is a separate try; the best stands.
        """
        from core.agency.which_one_answers_to_her import POINTER_LAGS_S, _measured_along

        best: list[bool | None] = [None, None]
        for lag in POINTER_LAGS_S:
            pairs = [(p[0], p[1], x, y) for at, x, y in seen if (p := self._pointer_at(at - lag)) is not None]
            if len(pairs) < 20:
                continue
            for axis, value in enumerate(_measured_along(pairs, extent)):
                if value or (value is False and best[axis] is None):
                    best[axis] = value
        return best[0], best[1]

    def _what_follows_the_pointer(self, moves: Any, at: float) -> None:
        """A thing that goes where the pointer went is hers, the way a key's thing is."""
        from core.agency.which_one_answers_to_her import _extent

        if not self._pointer or self._pointer_at(at) is None:
            return
        if self.follows_pointer:
            self._still_follows(moves, at)
            return
        pointer = self._pointer_at(at)
        # By kind, the nearest of each to the pointer: a thing under a pointer
        # that moved too far in one picture comes back as a new thing. Two of
        # a kind take turns being nearest a pointer swept across between
        # them, and together keep pace with it (offline 2026-10-04, two
        # paddles); neither is under it, which is what _follows asks.
        nearest: dict[int, Any] = {}
        for thing in moves.things.values():
            # Only what moves can follow anything. Of a column of identical
            # still things (the dashes of a net), the one nearest a pointer
            # swept up and down rises and falls with it (LIVE 2026-10-04).
            if thing.seen != at or not thing.moved:
                continue
            best = nearest.get(thing.kind)
            if best is None or math.dist((thing.x, thing.y), pointer) < math.dist((best.x, best.y), pointer):
                nearest[thing.kind] = thing
        for kind, thing in nearest.items():
            self._followed[kind].append((at, thing.x, thing.y))
            del self._followed[kind][:-60]
        self._looked += 1
        if self._looked % 4:
            return
        for kind, seen in self._followed.items():
            if kind not in nearest or len(seen) < 20:
                continue
            along = tuple(bool(v) for v in self._with_the_pointer(seen, _extent(moves)))
            if not any(along):
                continue
            # Once is a sighting; the same again on pictures taken since is a
            # finding. Tested every few pictures at a few delays, chance
            # agreement turns up now and then (offline 2026-10-04, the other
            # paddle during a long Pong game), and seldom twice running.
            if self._sighted != kind:
                self._sighted = kind
                seen.clear()
                continue
            self.number, self.kind, self.follows_pointer = nearest[kind].number, kind, True
            self.follows_along = along
            thing = nearest[kind]
            logger.info(
                "follows the pointer: thing %s of kind %s at (%.0f, %.0f), %dx%d, along %s, over %d pictures",
                thing.number, kind, thing.x, thing.y, thing.w, thing.h, along, len(seen),
            )
            return

    def _still_follows(self, moves: Any, at: float) -> None:
        """Whether her thing still goes where she points, measured as she plays; if not, it was never hers.

        A belief that a thing follows the pointer is tested by every move she
        makes with it. Where she has since taken the pointer both ways along an
        axis and her thing did not go with it, the belief is dropped and which
        thing is hers is found again.
        """
        from core.agency.which_one_answers_to_her import _extent

        mine = moves.things.get(self.number) if self.number is not None else None
        if mine is None or mine.seen != at:
            return
        pairs = self._since_believed
        pairs.append((at, mine.x, mine.y))
        del pairs[:-60]
        self._looked += 1
        if len(pairs) < 30 or self._looked % 4:
            return
        measured = self._with_the_pointer(pairs, _extent(moves))
        believed = [m for m, b in zip(measured, self.follows_along, strict=True) if b and m is not None]
        if believed and not any(believed):
            self.follows_pointer, self.follows_along = False, (False, False)
            self._followed.clear()
            pairs.clear()
            self.number = None

    def _under_the_pointer(self, moves: Any, at: float) -> int | None:
        """The thing of her kind nearest the pointer: a thing that follows it is re-made when it jumps."""
        pointer = self._pointer_at(at)
        mine = [t for t in moves.things.values() if t.kind == self.kind]
        if not mine or pointer is None:
            return None
        return min(mine, key=lambda t: math.dist((t.x, t.y), pointer)).number
