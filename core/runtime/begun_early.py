"""What a turn will need, begun as soon as it is known, and waited for where it is used.

A turn took its slow readings one after another, each where its result was wanted. LIVE 25 November 2026, 02:46, from the
reading of what was asked to her model being handed the turn: 2.7 s recalling, then 0.6 s more, then 1.5 s for what the words
ask of her, then 1.3 s for her senses and the screen. None of them waits on another, and her model's worker stood idle for all
but the third.

``in_a_turn`` opens a turn, ``begin`` starts a reading in it, and ``begun`` gives that reading where it is used, or takes it
there when nobody began it. A reading is held by what it is about, so words that changed on the way are read again. What nobody
waited for is stopped with the turn.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

__all__ = ["begin", "begun", "in_a_turn"]

logger = logging.getLogger("Aura.BegunEarly")

Make = Callable[[], Awaitable[Any]]
_TURN: ContextVar[dict[tuple[str, Hashable], asyncio.Task[Any]] | None] = ContextVar("begun_early", default=None)


@contextmanager
def in_a_turn() -> Iterator[None]:
    """The turn its readings belong to; those still under way when it ends are stopped."""
    held: dict[tuple[str, Hashable], asyncio.Task[Any]] = {}
    token = _TURN.set(held)
    try:
        yield
    finally:
        _TURN.reset(token)
        for task in held.values():
            if not task.done():
                task.cancel()


def _seen(task: asyncio.Task[Any]) -> None:
    """What failed early says so once; whoever waits for it is given the failure itself."""
    if not task.cancelled() and task.exception() is not None:
        logger.info("%s failed before it was waited for: %r", task.get_name(), task.exception())


def begin(name: str, about: Hashable, make: Make) -> bool:
    """Start ``make`` now, in the turn under way; whether it was started."""
    held = _TURN.get()
    if held is None or (name, about) in held:
        return False
    try:
        task = asyncio.get_running_loop().create_task(_made(make), name=f"begun_early:{name}")
    # not a failure: there is no loop to begin it on, so it is taken where it is used.
    except RuntimeError:
        return False
    task.add_done_callback(_seen)
    held[(name, about)] = task
    return True


async def _made(make: Make) -> Any:
    return await make()


async def begun(name: str, about: Hashable, make: Make) -> Any:
    """What ``make`` gives for ``about``: from the one begun earlier in this turn, or taken now."""
    held = _TURN.get()
    task = held.get((name, about)) if held is not None else None
    if task is None or task.get_loop() is not asyncio.get_running_loop() or task.cancelled():
        return await make()
    try:
        # Shielded: whoever waits with a limit of their own gives up the wait, and the reading stays for the next.
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if not task.cancelled():
            raise
        return await make()
