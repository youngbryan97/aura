"""Building up: dropping a piece so it lands square on what is already built.

A piece swings or slides above a pile; a press lets it go and it falls
straight down onto whatever is under it. A tower is as good as its worst
piece: one landed half over the edge leans, and what is put on it leans more.
So a person drops when the piece is over the top of the pile, not when it
happens to be anywhere, and the top is the highest of the things that have
stopped, under the way the piece goes.

This is a press timed to a place (core/agency/when_a_press_pays.py), with the
place known before any press has paid: square over the top. Where play shows
otherwise (a press that paid elsewhere), timing by what paid takes over.

Nothing here knows a piece, a tower or a game.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["drop_now", "the_top"]

#: How near square over the top a piece must be when let go, as a share of the narrower of the two.
SQUARE_WITHIN = 0.2

#: Slower than this, in working pixels a second, a thing has stopped: it is part of what is built.
STOPPED_BELOW = 6.0


def the_top(things: Iterable[Any], piece: Any) -> Any | None:
    """The top of what is built under the way ``piece`` goes: the highest of the still things below it whose span the
    piece's way crosses; None where nothing is built under it yet."""
    below = [t for t in things if t is not piece and abs(float(t.vx)) < STOPPED_BELOW and abs(float(t.vy)) < STOPPED_BELOW
             and float(t.y) > float(piece.y) + float(piece.h) / 2]
    return min(below, key=lambda t: float(t.y) - float(t.h) / 2, default=None)


def drop_now(things: Iterable[Any], ahead_s: float = 0.0) -> bool:
    """Whether to let the piece go now: the thing going across above the others will be square over the top of what
    is built when the press lands, ``ahead_s`` from now."""
    things = list(things)
    moving = [t for t in things if abs(float(t.vx)) >= STOPPED_BELOW and abs(float(t.vy)) < abs(float(t.vx))]
    piece = min(moving, key=lambda t: float(t.y), default=None)
    if piece is None:
        return False
    top = the_top(things, piece)
    if top is None:
        return False
    x = float(piece.x) + float(piece.vx) * ahead_s
    return abs(x - float(top.x)) <= SQUARE_WITHIN * min(float(piece.w), float(top.w))
