"""Her own questions beside her work, asked one at a time, the one that matters most first.

She has one model. While she plays, her thinking at each move holds it, and beside her work she wonders about several
things at once: what the place's rules ask, what the place is, what she knows of it, what the things in it are, what an
odd look suggests. Sent together, they queued in the order they were thought of. LIVE 2026-10-10 a game's lesson was
never read in five minutes of play: the question of what its rules ask sat behind questions about what a dog and a
cabinet are, and she skipped the lesson, tested an empty trap and called it won.

A person reading a new place reads what it asks of them before they wonder about its furniture. So a question waits
for its turn, and the turn goes to the waiting question that matters most (``RULES`` before ``THE_PLACE`` before
``THINGS`` before ``A_LOOK``); among equals, the one asked first. One of her questions is in her model at a time, which
also leaves room for her thinking at each move.

Nothing here knows a place.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = ["A_LOOK", "RULES", "THE_PLACE", "THINGS", "in_turn"]

#: What a question is about, in the order she takes them: what the place asks of her; what the place is and what she
#: knows of it; what its things are; what an odd look suggests.
RULES, THE_PLACE, THINGS, A_LOOK = 3, 2, 1, 0

_ORDER = itertools.count()


@dataclass
class _Turns:
    busy: bool = False
    waiting: list[tuple[int, int, Any]] = field(default_factory=list)

    def hand_on(self) -> None:
        """The turn passed to the waiting question that matters most, or let go where none waits."""
        while self.waiting:
            _matters, _order, future = heapq.heappop(self.waiting)
            if not future.done():
                future.set_result(None)
                return
        self.busy = False


_BY_LOOP: weakref.WeakKeyDictionary[Any, _Turns] = weakref.WeakKeyDictionary()


@asynccontextmanager
async def in_turn(matters: int) -> AsyncIterator[None]:
    """Hold her model for one question, once it is this question's turn."""
    loop = asyncio.get_running_loop()
    turns = _BY_LOOP.setdefault(loop, _Turns())
    if turns.busy:
        future = loop.create_future()
        entry = (-int(matters), next(_ORDER), future)
        heapq.heappush(turns.waiting, entry)
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                turns.hand_on()                      # the turn had come: passed on, not kept
            elif entry in turns.waiting:
                turns.waiting.remove(entry)
                heapq.heapify(turns.waiting)
            raise
    else:
        turns.busy = True
    try:
        yield
    finally:
        turns.hand_on()
