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
import re
import time
from typing import Any

logger = logging.getLogger("Aura.ScreenPursuit.BoardOfPieces")

__all__ = ["played_a_board_of_pieces"]

#: How often the window is looked at for a board of pieces, in seconds.
LOOKED_FOR_EVERY_S = 4.0
#: How long a board that would not take a move is left before it is tried again.
LEFT_UNTIL_S = 8.0


async def played_a_board_of_pieces(target_app: str, ends_at: float, keep: dict[str, Any],
                                   seen: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Where ``target_app``'s window shows two sides of pieces at their start, play the game and say what came of it;
    None where it shows none, or it was looked at a moment ago."""
    from core.agency.playing_pieces_by_kind import (
        a_board_of_two_sides_at_its_start,
        play_pieces_by_kind,
    )
    from core.capabilities import window_server
    from core.perception.what_the_pixels_show import _the_pixels_of

    from .screen_pursuit import _tell
    from .screen_pursuit_on_a_page import on_her_page
    from .screen_pursuit_surface import click_normalized

    now = time.monotonic()
    if not target_app or now < float(keep.get("board_left_until", 0.0)):
        return None
    if seen is not None and not seen.get("window_number"):
        # Only behind a look at a window: a reading given any other way is all there is to see, and the desktop is not
        # looked at behind it (a test's reading named an application that was really running).
        return None
    if on_her_page() is not None:
        # Her own page is played by her page's own player (core/skills/screen_pursuit_as_it_happens.py), which has a
        # board of its own to hand over to; the desktop is not touched for it.
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
    kept = _what_she_kept(target_app)
    seat = await asyncio.to_thread(a_board_of_two_sides_at_its_start, picture, kept.get("looks"))
    if seat is None:
        return None
    learned = keep.get("pieces_learned") or kept.get("rules")
    if learned is not None:
        learned.start = dict(seat.start)
    logger.info("two sides of pieces at their start in %r: %dx%d; playing it by search", target_app, seat.rows,
                seat.columns)

    async def look() -> Any:
        current = await asyncio.to_thread(window_server.window_of, target_app) or window
        return await _the_pixels_of(current)

    async def click(x: float, y: float) -> bool:
        current = await asyncio.to_thread(window_server.window_of, target_app) or window
        return await click_normalized(x, y, expect_app=target_app, bounds=list(current.bounds))

    async def clock() -> float | None:
        return await _her_clock(target_app)

    def tell(line: str) -> None:
        # What happens in the game, as it happens: each move is news, not a remark to be paced.
        _tell(line, news=True)

    came = await play_pieces_by_kind(seat, look, click, seconds=max(5.0, ends_at - time.monotonic()), say=tell,
                                     learned=learned, rehearsal=keep.get("pieces_rehearsal") or kept.get("rehearsal"),
                                     clock=clock)
    learned_now, rehearsal_now = came.pop("learned", None), came.pop("rehearsal", None)
    logger.info("a board of pieces played: %s", came)
    if learned_now is not None:
        keep["pieces_learned"] = learned_now
    if rehearsal_now is not None:
        keep["pieces_rehearsal"] = rehearsal_now
    _keep_what_she_learned(target_app, seat.looks, learned_now, rehearsal_now)
    if came.get("ended") in ("not taking moves", "not answered"):
        keep["board_left_until"] = time.monotonic() + LEFT_UNTIL_S
        return None
    if came.get("ended") == "over":
        # The game stopped letting her move and her own model could not say why: the window says how it ended, as a
        # person looks at the screen to see. LIVE 21 November 2026 she said "it's over" four times against Stockfish,
        # with En Croissant reading "0-1 Black wins" beside the board.
        ended, words = await what_the_window_says_of_the_end(target_app)
        if ended:
            came["ended"], came["read_from_the_window"] = ended, words[:300]
            _tell({"won": "The window says I won.", "lost": "The window says I lost.",
                   "drawn": "The window says it was drawn."}[ended], news=True)
    if came.get("ended") in GAME_OVER:
        # Finished, she closes what she opened for it, as a person puts a game away.
        from core.capabilities.host_automation import get_host_automation

        receipt = await get_host_automation().close_app(target_app)
        came["closed"] = bool(getattr(receipt, "success", False))
        _tell(f"That's the game. I've closed {target_app}." if came["closed"] else "That's the game.", news=True)
    return came


#: What she may take a window's words to say of how a game ended, and what each comes to.
THE_ENDS = {"I won": "won", "I lost": "lost", "it was drawn": "drawn", "the window does not say": ""}


async def what_the_window_says_of_the_end(target_app: str) -> tuple[str, str]:
    """How a game ended, as the words in its window say it and her own model reads them: "won", "lost", "drawn", or ""
    where they do not say; with the words read. The answers her model may give are those, and no others."""
    from typing import Literal

    from pydantic import create_model

    from core.cognition.asking_in_turn import THE_PLACE
    from core.cognition.what_things_are import asked_patiently
    from core.rebuilding.her_model import ask_her_model

    from .screen_pursuit_a_way_on import _its_own_pixels_read

    try:
        reading = await _its_own_pixels_read(target_app)
    except (OSError, RuntimeError, ValueError) as why:
        logger.info("the window was not read for how the game ended (%s)", why)
        return "", ""
    words = " / ".join(" ".join(str(region.get("text") or "").split())
                       for region in (reading or {}).get("layout") or [] if str(region.get("text") or "").strip())
    if not words:
        return "", ""
    schema = create_model("_TheEnd", ended=(Literal[tuple(THE_ENDS)], ...))  # type: ignore[valid-type]
    prompt = (f"She has just played a game of pieces in the application “{target_app}”, as the side nearest the "
              f"bottom of its board, and the game has stopped letting her move. The words in its window now: "
              f"{words[:1500]}\nWhich of these do the words say happened?")
    try:
        got = await asked_patiently(ask_her_model, prompt, schema, 60, matters=THE_PLACE)
    except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
        logger.info("how the game ended was not asked of her model (%s)", why)
        return "", words
    return THE_ENDS.get(str(getattr(got, "ended", "") or ""), ""), words


#: How a game handed over can end, as opposed to running out of time or losing sight of the board.
GAME_OVER = ("won", "lost", "drawn", "over")


#: A clock as a game shows one: minutes and seconds, with tenths or hours where it shows them.
_A_CLOCK = re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:\.(\d))?$")


def _seconds_on(text: str) -> float | None:
    found = _A_CLOCK.match(str(text or "").strip())
    if found is None:
        return None
    hours, minutes, seconds, tenths = found.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(tenths or 0) / 10


async def _her_clock(target_app: str) -> float | None:
    """The time she has left, where the window shows two clocks: the one nearer the bottom, her side of the board."""
    from .screen_pursuit import read_screen

    reading = await read_screen(target_app)
    clocks = []
    for region in reading.get("layout") or []:
        seconds = _seconds_on(region.get("text", ""))
        if seconds is not None:
            clocks.append((float(region.get("center_y", region.get("y", 0.0)) or 0.0), seconds))
    if len(clocks) < 2:
        return None
    return max(clocks)[1]


def _her_record(target_app: str) -> str:
    """Where what she learned of this board is kept: apart from the rest of what she worked out about the app, which
    her pursuit writes back whole."""
    return f"{target_app} board of pieces"


def _what_she_kept(target_app: str) -> dict[str, Any]:
    """The looks, rules and worth she learned at this board before, where she has played it."""
    from core.agency.how_the_pieces_move import HowThePiecesMove
    from core.agency.searching_by_kind import Weights
    from core.agency.what_a_kind_is_worth import Rehearsal
    from core.perception.pieces_by_their_look import Looks
    from core.runtime.what_she_learned import recall

    held = recall(_her_record(target_app))
    kept: dict[str, Any] = {}
    if held.get("looks"):
        kept["looks"] = Looks.from_memory(held["looks"])
    if held.get("rules"):
        kept["rules"] = HowThePiecesMove.from_memory(held["rules"])
    worth = held.get("worth") or {}
    if worth.get("weights"):
        try:
            kept["rehearsal"] = Rehearsal(weights=Weights(**worth["weights"]), earned=list(worth.get("earned", [])),
                                          royal=(list(worth["royal"]) if isinstance(worth.get("royal"), list) else None),
                                          tries={str(k): int(v) for k, v in (worth.get("tries") or {}).items()},
                                          lengths=[int(n) for n in worth.get("lengths", [])])
        except (TypeError, ValueError) as why:
            logger.info("what she measured at this board could not be read back (%s)", why)
    return kept


def _keep_what_she_learned(target_app: str, looks: Any, learned: Any, rehearsal: Any) -> None:
    from dataclasses import asdict

    from core.runtime.what_she_learned import remember

    record: dict[str, Any] = {"looks": looks.as_memory()}
    if learned is not None:
        record["rules"] = learned.as_memory()
    if rehearsal is not None and getattr(rehearsal, "earned", None):
        record["worth"] = {"weights": asdict(rehearsal.weights), "earned": list(rehearsal.earned),
                           "royal": rehearsal.royal, "tries": dict(rehearsal.tries),
                           "lengths": list(rehearsal.lengths)[-400:]}
    remember(_her_record(target_app), record)
