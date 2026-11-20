"""A board of two sides of pieces, met in a window she is working in, played by search with rules she learns from it.

Her pursuit of a goal on screen reads words and presses keys and clicks labels. A board of pieces facing each other is
not met that way: it is a game whose rules she does not know, played a move at a time against someone. When the window
she is working in shows one at its start, the board is handed to core/agency/playing_pieces_by_kind.py, which learns
how the pieces move by picking them up and watching, and searches its moves. Her eyes there are the window's own
pixels and her hands are clicks at shares of the window, as everywhere else in her pursuit. A board that will not let
her pick anything up (a picture of a board on a screen that is still being set up) is handed back, and looked at again
a little later.

Nothing here knows a game or an application.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("Aura.ScreenPursuit.BoardOfPieces")

__all__ = ["played_a_board_of_pieces"]

#: How often the window is looked at for a board of pieces, in seconds.
LOOKED_FOR_EVERY_S = 4.0
#: How long a board that would not take a move is left before it is tried again.
LEFT_UNTIL_S = 8.0


async def played_a_board_of_pieces(target_app: str, ends_at: float, keep: dict[str, Any]) -> dict[str, Any] | None:
    """Where ``target_app``'s window shows two sides of pieces at their start, play the game and say what came of it;
    None where it shows none, or it was looked at a moment ago."""
    from core.agency.playing_pieces_by_kind import (
        a_board_of_two_sides_at_its_start,
        play_pieces_by_kind,
    )
    from core.capabilities import window_server
    from core.perception.what_the_pixels_show import _the_pixels_of

    from .screen_pursuit import _tell
    from .screen_pursuit_surface import click_normalized

    now = time.monotonic()
    if not target_app or now < float(keep.get("board_left_until", 0.0)):
        return None
    if now - float(keep.get("board_looked_for_at", -1e9)) < LOOKED_FOR_EVERY_S:
        return None
    keep["board_looked_for_at"] = now
    import asyncio

    window = await asyncio.to_thread(window_server.window_of, target_app)
    if window is None:
        return None
    picture = await _the_pixels_of(window)
    if picture is None:
        return None
    seat = await asyncio.to_thread(a_board_of_two_sides_at_its_start, picture)
    if seat is None:
        return None
    logger.info("two sides of pieces at their start in %r: %dx%d; playing it by search", target_app, seat.rows,
                seat.columns)

    async def look() -> Any:
        current = await asyncio.to_thread(window_server.window_of, target_app) or window
        return await _the_pixels_of(current)

    async def click(x: float, y: float) -> bool:
        current = await asyncio.to_thread(window_server.window_of, target_app) or window
        return await click_normalized(x, y, expect_app=target_app, bounds=list(current.bounds))

    came = await play_pieces_by_kind(seat, look, click, seconds=max(5.0, ends_at - time.monotonic()), say=_tell,
                                     learned=keep.get("pieces_learned"))
    logger.info("a board of pieces played: %s", came)
    if came.get("learned") is not None:
        keep["pieces_learned"] = came.pop("learned")
    if came.get("ended") == "not taking moves":
        keep["board_left_until"] = time.monotonic() + LEFT_UNTIL_S
        return None
    if came.get("ended") in GAME_OVER:
        # Finished, she closes what she opened for it, as a person puts a game away.
        from core.capabilities.host_automation import get_host_automation

        receipt = await get_host_automation().close_app(target_app)
        came["closed"] = bool(getattr(receipt, "success", False))
        _tell(f"That's the game. I've closed {target_app}." if came["closed"] else "That's the game.")
    return came


#: How a game handed over can end, as opposed to running out of time or losing sight of the board.
GAME_OVER = ("won", "lost", "drawn", "over")
