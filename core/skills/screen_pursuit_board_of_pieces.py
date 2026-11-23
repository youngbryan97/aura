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

import asyncio
import json
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
    # The rehearsal she has in hand from a game this sitting, or the one kept with her record, whichever has run more
    # rounds: practice between games (core/agency/practising_between_games.py) keeps its rounds with the record.
    in_hand, recorded = keep.get("pieces_rehearsal"), kept.get("rehearsal")
    rehearsal = max((r for r in (in_hand, recorded) if r is not None), key=lambda r: r.rounds, default=None)
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

    def borrow(start: dict, first_is_hers: bool) -> Any:
        return _borrowed_from_a_board_like_it(target_app, start, seat.rows, seat.columns, first_is_hers)

    came = await play_pieces_by_kind(seat, look, click, seconds=max(5.0, ends_at - time.monotonic()), say=tell,
                                     learned=learned, rehearsal=rehearsal, clock=clock, borrow=borrow,
                                     went_wrong=keep.get("pieces_went_wrong") or kept.get("went_wrong"))
    learned_now, rehearsal_now = came.pop("learned", None), came.pop("rehearsal", None)
    went_wrong_now = came.pop("went_wrong", None)
    if went_wrong_now is not None:
        keep["pieces_went_wrong"] = went_wrong_now
    logger.info("a board of pieces played: %s", came)
    if learned_now is not None:
        keep["pieces_learned"] = learned_now
    if rehearsal_now is not None:
        keep["pieces_rehearsal"] = rehearsal_now
    # Off the loop: what she gathered beside the record runs to megabytes.
    await asyncio.to_thread(_keep_what_she_learned, target_app, seat.looks, learned_now, rehearsal_now,
                            went_wrong_now)
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
    """The time she has left, where the window shows two clocks: the one nearer the bottom, her side of the board.
    Read from one picture: a running clock is what keeps a window from standing still, and LIVE 21 November 2026, game
    12, every late turn spent 1.6 s waiting for her window to stand still before reading the time it showed."""
    from core.perception.what_the_pixels_show import look_at_window

    from .screen_pursuit import read_screen

    reading = await look_at_window(target_app, wait_for_stillness=False) or await read_screen(target_app)
    clocks = []
    for region in reading.get("layout") or []:
        seconds = _seconds_on(region.get("text", ""))
        if seconds is not None:
            clocks.append((float(region.get("center_y", region.get("y", 0.0)) or 0.0), seconds))
    if len(clocks) < 2:
        return None
    return max(clocks)[1]


#: What is kept beside her record of a board: how the pieces stand together, and the positions it is learned from.
_TOGETHER = "how-the-pieces-stand-together"
_LOOKED_AT_LONGER = "positions-looked-at-longer"
_CAME_TO = "what-her-games-came-to"


def _her_record(target_app: str) -> str:
    """Where what she learned of this board is kept: apart from the rest of what she worked out about the app, which
    her pursuit writes back whole."""
    return f"{target_app} board of pieces"


def _what_she_kept(target_app: str) -> dict[str, Any]:
    """The looks, rules and worth she learned at this board before, where she has played it."""
    return _kept_under(_her_record(target_app))


def _kept_under(record: str) -> dict[str, Any]:
    from core.agency.how_the_pieces_move import HowThePiecesMove
    from core.agency.searching_by_kind import Weights
    from core.agency.what_a_kind_is_worth import RESULTS, LookedAtLonger, Rehearsal
    from core.perception.pieces_by_their_look import Looks
    from core.runtime.what_she_learned import recall, recall_beside

    held = recall(record)
    kept: dict[str, Any] = {"app": str(held.get("app") or "")}
    if held.get("looks"):
        kept["looks"] = Looks.from_memory(held["looks"])
    if held.get("rules"):
        kept["rules"] = HowThePiecesMove.from_memory(held["rules"])
    if held.get("went_wrong"):
        from core.cognition.the_furthest_she_has_got import WhatFailedWhere

        kept["went_wrong"] = WhatFailedWhere.from_memory(held["went_wrong"])
    worth = held.get("worth") or {}
    if worth.get("weights"):
        try:
            weights = dict(worth["weights"])
            together = recall_beside(record, _TOGETHER)
            weights["learned"] = json.loads(together) if together else None
            rehearsal = Rehearsal(weights=Weights(**weights), earned=list(worth.get("earned", [])),
                                  royal=(list(worth["royal"]) if isinstance(worth.get("royal"), list) else None),
                                  tries={str(k): int(v) for k, v in (worth.get("tries") or {}).items()},
                                  lengths=[int(n) for n in worth.get("lengths", [])],
                                  teacher=str(worth.get("teacher") or RESULTS), rounds=int(worth.get("rounds") or 0),
                                  together_tried_at=float(worth.get("together_tried_at") or 1.0))
            positions = recall_beside(record, _LOOKED_AT_LONGER)
            if positions:
                rehearsal.looked_at_longer = LookedAtLonger.from_bytes(positions)
            came_to = recall_beside(record, _CAME_TO)
            if came_to:
                from core.agency.what_lies_beyond_the_horizon import WhatGamesCameTo

                rehearsal.came_to = WhatGamesCameTo.from_bytes(came_to)
            kept["rehearsal"] = rehearsal
        except (TypeError, ValueError, KeyError, OSError) as why:
            logger.info("what she measured at this board could not be read back (%s)", why)
    return kept


def _keep_what_she_learned(target_app: str, looks: Any, learned: Any, rehearsal: Any, went_wrong: Any = None) -> None:
    from dataclasses import asdict

    from core.agency.what_a_kind_is_worth import RESULTS
    from core.runtime.what_she_learned import remember, remember_beside

    record: dict[str, Any] = {"app": target_app, "looks": looks.as_memory()}
    if learned is not None:
        record["rules"] = learned.as_memory()
    if went_wrong is not None and went_wrong.ruled_out:
        record["went_wrong"] = went_wrong.as_memory()
    if rehearsal is not None and getattr(rehearsal, "earned", None):
        weights = asdict(rehearsal.weights)
        # How the pieces stand together, and the positions it is learned from, are kept beside the record: thousands of
        # numbers where a record holds sixty thousand characters.
        together = weights.pop("learned", None)
        if together:
            remember_beside(_her_record(target_app), _TOGETHER, json.dumps(together).encode())
        looked = getattr(rehearsal, "looked_at_longer", None)
        if looked is not None and looked.parts:
            remember_beside(_her_record(target_app), _LOOKED_AT_LONGER, looked.as_bytes())
        came_to = getattr(rehearsal, "came_to", None)
        if came_to is not None and came_to.result:
            remember_beside(_her_record(target_app), _CAME_TO, came_to.as_bytes())
        record["worth"] = {"weights": weights, "earned": list(rehearsal.earned),
                           "royal": rehearsal.royal, "tries": dict(rehearsal.tries),
                           "lengths": list(rehearsal.lengths)[-400:],
                           "teacher": getattr(rehearsal, "teacher", RESULTS),
                           "rounds": int(getattr(rehearsal, "rounds", 0)),
                           "together_tried_at": float(getattr(rehearsal, "together_tried_at", 1.0))}
    remember(_her_record(target_app), record)


def _borrowed_from_a_board_like_it(target_app: str, start: dict, rows: int, columns: int,
                                   first_is_hers: bool) -> tuple[Any, Any, str] | None:
    """What she learned at another application's board that begins as this one does, under this board's names: its
    rules, what she measured each kind to be worth, and the application she learned them in. None where no board she
    has kept begins so (core/cognition/the_same_start_under_other_names.py). The looks stay where they were learned:
    every application draws its pieces its own way.

    Her rows count from her own back row, so a board she sat at on the other side is compared turned half round. A
    record that does not say which side she sat at is compared both ways, and lends only where the two ways do not
    disagree: a chess board turned round begins as it does unturned with its king and queen called by each other's
    names. Where two boards she has kept begin so, the one she has played more lends what it learned."""
    from dataclasses import replace

    from core.agency.how_the_pieces_move import turned_round_from
    from core.cognition.the_same_start_under_other_names import the_same_start
    from core.runtime.what_she_learned import kept_worlds, named

    ours, boards = named(_her_record(target_app)), named("board of pieces")
    best: tuple[int, Any, Any, str] | None = None
    for world in kept_worlds():
        if world == ours or not world.endswith(boards):
            continue
        kept = _kept_under(world)
        rules = kept.get("rules")
        if rules is None or (rules.rows, rules.columns) != (rows, columns):
            continue
        ways = [False, True] if rules.first_was_hers is None else [rules.first_was_hers != first_is_hers]
        lined_up = {turned: names for turned in ways
                    if (names := the_same_start(start, turned_round_from(rules.start, rows, columns) if turned
                                                else rules.start)) is not None}
        if not lined_up or len({tuple(sorted(names.items())) for names in lined_up.values()}) > 1:
            continue
        turned, names = next(iter(lined_up.items()))
        here = {theirs: mine for mine, theirs in names.items()}
        rehearsal = kept.get("rehearsal")
        if rehearsal is not None:
            rehearsal = replace(rehearsal, weights=rehearsal.weights.under_other_names(here, turned_round=turned),
                                royal=None if rehearsal.royal is None else [here[k] for k in rehearsal.royal
                                                                             if k in here],
                                samples_x=[], samples_y=[], last={})
        played = len(rehearsal.lengths) if rehearsal is not None else 0
        played += sum(way.took + way.moved_to_empty for ways_of in rules.seen.values() for way in ways_of.values())
        where = kept.get("app") or world.removesuffix(boards).strip("-").replace("-", " ")
        if best is None or played > best[0]:
            best = (played, rules.under_other_names(here, turned_round=turned), rehearsal, where)
    if best is None:
        return None
    logger.info("%r begins as %r did: borrowing what she learned there", target_app, best[3])
    return best[1], best[2], best[3]
