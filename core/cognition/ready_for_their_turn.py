"""Ready for their turn: what she gets ready while someone else has the turn, for the turn they are likely to take.

At a board she thinks on the other side's time about the answer she expects (core/agency/thinking_on_their_time.py).
A conversation has the same shape. After she has answered somebody, the turn is theirs, and the turn people take
after an answer most often goes on with what was being talked about. So while they read and write, she does the slow
part of answering them that does not need their words yet: recalling what she knows that bears on the exchange as it
stands. When their turn comes, it is there.

What she got ready stands in only for what she could not get in time. Their own words decide what is recalled for
them; where that recall comes back empty, as it does when the store is slow and the turn cannot wait for it, what she
got ready for the exchange they are carrying on is used instead. A turn of theirs that comes while she is still
getting ready stops that work, so it never takes time from the turn itself; and what was got ready is used once, for
the turn that came next, and only within the sitting it was got ready in.

Suggestions of 23 November 2026, item 21: thinking on the person's time. Nothing here knows a conversation: a party is
a name, getting ready is any work that can wait, and its result is kept as it came.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.ReadyForTheirTurn")

__all__ = ["ReadyForTheirTurn"]


@dataclass
class _Getting:
    task: asyncio.Task | None
    began: float
    result: Any = None


class ReadyForTheirTurn:
    """For each party whose turn it is, the work she is doing, or has done, toward the turn they are likely to take."""

    def __init__(self) -> None:
        self._for: dict[str, _Getting] = {}

    def get_ready(self, who: str, work: Callable[[], Awaitable[Any]]) -> None:
        """Begin the work for ``who``'s next turn, in place of any begun before."""
        self.their_turn_came(who)
        getting = _Getting(task=None, began=time.time())

        async def doing() -> None:
            try:
                getting.result = await work()
            except asyncio.CancelledError:
                raise
            # not a failure: what she could not get ready, she does not have, and their turn goes on as before.
            except Exception as why:  # noqa: BLE001 - getting ready is never the turn itself
                logger.info("what she was getting ready for %s's turn was not got ready (%s)", who, type(why).__name__)

        getting.task = asyncio.get_running_loop().create_task(doing(), name="ready_for_their_turn")
        self._for[str(who)] = getting

    def their_turn_came(self, who: str) -> None:
        """Their turn has come: work still under way for it stops; what is done is kept for it to use."""
        getting = self._for.get(str(who))
        if getting is not None and getting.task is not None and not getting.task.done():
            getting.task.cancel()
            self._for.pop(str(who), None)

    def take(self, who: str, *, since: float = 0.0) -> Any:
        """What was got ready for ``who``'s turn, got ready no earlier than ``since``, once; None where nothing was."""
        getting = self._for.get(str(who))
        if getting is None or getting.task is None or not getting.task.done() or getting.began < since:
            return None
        self._for.pop(str(who), None)
        return getting.result
