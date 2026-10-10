"""Whether a picture goes on by itself, and whether a key held keeps something going.

Two things she settles before she plays a picture as it happens: whether
anything in it goes somewhere while she does nothing (a world that moves on its
own, as against a menu that sways or a board that waits), and whether a key
held keeps something going for as long as it is down (a thing she steers, as
against a highlight that jumps once). Moving in place is not going anywhere: a
thing goes somewhere when it travels further than half its own size, mostly one
way, or faster than two of its sizes a second.

Measured on the pictures' own clock. Nothing here knows a game.
"""
from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from typing import Any

from core.perception.what_moves_in_the_picture import WhatMoves

__all__ = ["HELD_TO_SEE_S", "THE_PRESS_S", "goes_somewhere", "it_goes_while_held", "the_world_moves_on_its_own"]


async def the_world_moves_on_its_own(look: Callable[[], Awaitable[Any]], *, seconds: float = 0.8, longest: float = 3.0) -> bool:
    """Whether things go somewhere in the picture while she does nothing.

    Moving in place is not going anywhere. LIVE 2026-10-06 a checkers game's
    figures swayed beside a board that waited for her move, and she played it
    with the arrow keys for five minutes as if it were an action game. A thing
    goes somewhere when it travels further than half its own size, mostly one
    way, or faster than two of its sizes a second; where things move and none
    has gone anywhere yet, she watches a while longer before saying the world
    waits for her. Measured on the pictures' own clock.
    """
    moves = WhatMoves()
    began: float | None = None
    while True:
        seen = await look()
        if seen is None:
            return False
        picture, at = seen
        began = at if began is None else began
        moves.see(picture, at)
        if moves.pictures <= 4:
            continue
        moving = moves.moving(faster_than=8.0)
        if any(goes_somewhere(thing) for thing in moving):
            return True
        if at - began >= (longest if moving else seconds) + 0.6:
            return False


#: How long she holds a key down to see what it does, and how long after it goes down the jump at the press is over.
HELD_TO_SEE_S = 0.45


THE_PRESS_S = 0.1


async def it_goes_while_held(look: Callable[[], Awaitable[Any]], hands: Any, key: str, *, held: float = HELD_TO_SEE_S) -> bool:
    """``key`` pressed the way a person presses one to see what it does, held a moment while she watches: whether something goes on going while it is down.

    A world that stands still until she moves in it is played as it happens all
    the same: LIVE 2026-10-07 a top-down shooter's room held still, it was
    taken for a screen to be stepped through, and each tap of an arrow moved
    its hero a few pixels, which she took for nothing; for three hours she
    clicked the words on its scoreboard. A menu's highlight jumps once as the
    key goes down and is still for the rest of the hold; a thing she steers
    keeps going for as long as the key is down. So what moved in the first
    tenth of a second is not counted, and a thing goes on going when it is seen
    in three pictures after that and has travelled half its own size between
    them. She looks before she presses, so what is there already is known.
    Measured on the pictures' own clock. The key goes down once, so to whatever
    counts presses it is one press.
    """
    moves = WhatMoves()
    began: float | None = None
    while not moves.has_looked:
        seen = await look()
        if seen is None:
            return False
        began = seen[1] if began is None else began
        moves.see(*seen)
        if seen[1] - began > 3 * held:
            break
    await hands.down(key)
    first: float | None = None
    try:
        while True:
            seen = await look()
            if seen is None:
                return False
            picture, at = seen
            moves.see(picture, at)
            first = at if first is None else first
            if at - first >= held:
                break
    finally:
        await hands.up(key)
    since = first + THE_PRESS_S
    return any(_went_on(thing, since) for thing in moves.things.values())


def _went_on(thing: Any, since: float) -> bool:
    points = [(at, x, y) for at, x, y in thing.path if at >= since]
    if len(points) < 3:
        return False
    size = max(float(thing.w), float(thing.h), 1.0)
    return math.dist(points[0][1:], points[-1][1:]) > 0.5 * size


def goes_somewhere(thing: Any) -> bool:
    """Whether a moving thing travels rather than moving in place, by its path and its speed against its own size."""
    size = max(float(thing.w), float(thing.h), 1.0)
    if math.hypot(thing.vx, thing.vy) > 2.0 * size:
        return True
    points = [(x, y) for _at, x, y in thing.path]
    if len(points) < 3:
        return False
    xs, ys = [x for x, _y in points], [y for _x, y in points]
    extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return extent > 0.5 * size and math.dist(points[0], points[-1]) > 0.5 * extent
