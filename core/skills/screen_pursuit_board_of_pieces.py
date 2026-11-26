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

from core.runtime.what_she_learned_still_holds import replays

logger = logging.getLogger("Aura.ScreenPursuit.BoardOfPieces")

__all__ = ["played_a_board_of_pieces"]

#: How often the window is looked at for a board of pieces, in seconds.
LOOKED_FOR_EVERY_S = 4.0
#: How long a board that would not take a move is left before it is tried again.
LEFT_UNTIL_S = 8.0


def _fewer_than_it_begins_with(start: dict, learned: Any) -> bool:
    """Whether a board read as two sides at their start holds fewer pieces than her record says this board's game
    begins with."""
    kept = getattr(learned, "start", None) or {}
    return bool(kept) and len(start) < len(kept)


async def played_a_board_of_pieces(target_app: str, ends_at: float, keep: dict[str, Any],
                                   seen: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Where ``target_app``'s window shows two sides of pieces at their start, play the game and say what came of it;
    None where it shows none, or it was looked at a moment ago."""
    from core.agency.playing_pieces_by_kind import (
        a_board_of_two_sides_at_its_start,
        play_pieces_by_kind,
    )
    from core.capabilities import window_server
    from core.capabilities.host_automation import get_host_automation
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
    if _fewer_than_it_begins_with(seat.start, learned):
        # A game begins with every piece it will have and loses them as it goes: a board of two sides with fewer is a
        # game under way or over. LIVE 25 November 2026 a finished game's board, its sides still apart, was taken for
        # one at its start; read as one, it taught her a look of a glow round a piece, and became her record's start.
        logger.info("two sides in %r with fewer pieces than its game begins with: a game under way or over",
                    target_app)
        return None
    # The rehearsal she has in hand from a game this sitting, or the one kept with her record, whichever has run more
    # rounds: practice between games (core/agency/practising_between_games.py) keeps its rounds with the record.
    in_hand, recorded = keep.get("pieces_rehearsal"), kept.get("rehearsal")
    rehearsal = max((r for r in (in_hand, recorded) if r is not None), key=lambda r: r.rounds, default=None)
    if learned is not None:
        learned.start = dict(seat.start)
    logger.info("two sides of pieces at their start in %r: %dx%d; playing it by search", target_app, seat.rows,
                seat.columns)

    async def look() -> Any:
        current = await asyncio.to_thread(window_server.window_of, target_app)
        return None if current is None or current.number != window.number else await _the_pixels_of(current)

    async def click(x: float, y: float) -> bool:
        current = await asyncio.to_thread(window_server.window_of, target_app)
        if current is None or current.number != window.number:
            return False
        refusal = await get_host_automation()._ensure_input_target(target_app, "click_at")
        if refusal is not None:
            logger.info("could not restore the game window's input focus in %r: %s", target_app, refusal.error)
            return False
        focused = await asyncio.to_thread(window_server.window_of, target_app)
        if focused is None or focused.number != current.number or focused.bounds[2:] != current.bounds[2:]:
            return False
        return await click_normalized(x, y, expect_app=target_app, bounds=list(focused.bounds))

    async def clock() -> float | None:
        return await _her_clock(target_app)

    def tell(line: str) -> None:
        # What happens in the game, as it happens: each move is news, not a remark to be paced.
        _tell(line, news=True)

    def borrow(start: dict, first_is_hers: bool) -> Any:
        return _borrowed_from_a_board_like_it(target_app, start, seat.rows, seat.columns, first_is_hers)

    # What the window says of the game as it begins (who plays, how strong, on what clock), kept with her record of
    # it: what she played against is part of what the game was. Read beside her first moves, not before them.
    async def read_beginning() -> str:
        try:
            return await the_words_in(target_app)
        except (OSError, RuntimeError, ValueError, TypeError) as why:
            logger.info("the window was not read as the game began (%s)", why)
            return ""

    began = asyncio.create_task(read_beginning())

    async def read_end() -> tuple[str, str]:
        return await what_the_window_says_of_the_end(target_app, before=await began)

    came = await play_pieces_by_kind(seat, look, click, seconds=max(5.0, ends_at - time.monotonic()), say=tell,
                                     learned=learned, rehearsal=rehearsal, clock=clock, borrow=borrow,
                                     went_wrong=keep.get("pieces_went_wrong") or kept.get("went_wrong"),
                                     they_answered=keep.get("pieces_they_answered") or kept.get("they_answered"),
                                     prepared=kept.get("prepared"), held=keep.get("pieces_held") or kept.get("held"),
                                     endings=kept.get("endings"), read_end=read_end)
    came["the_window_when_it_began"] = (await began)[:WORDS_KEPT]
    learned_now, rehearsal_now = came.pop("learned", None), came.pop("rehearsal", None)
    went_wrong_now = came.pop("went_wrong", None)
    if went_wrong_now is not None:
        keep["pieces_went_wrong"] = went_wrong_now
    they_answered_now = came.pop("they_answered", None)
    if they_answered_now is not None:
        keep["pieces_they_answered"] = they_answered_now
    held_now = came.pop("held", None)
    if held_now is not None:
        keep["pieces_held"] = held_now
    # The final position and move are too long for the log.
    came.pop("final", None)
    came.pop("last_move", None)
    earlier = came.pop("positions", None)
    logger.info("a board of pieces played: %s", came)
    if learned_now is not None:
        keep["pieces_learned"] = learned_now
        # However it ended, the game went on from every position before its last: what a draw is read against.
        learned_now.went_on_from(earlier or [])
    if rehearsal_now is not None:
        keep["pieces_rehearsal"] = rehearsal_now
    # Off the loop: what she gathered beside the record runs to megabytes.
    await asyncio.to_thread(_keep_what_she_learned, target_app, seat.looks, learned_now, rehearsal_now,
                            went_wrong_now, they_answered_now, held_now)
    if came.get("ended") in ("not taking moves", "not answered"):
        keep["board_left_until"] = time.monotonic() + LEFT_UNTIL_S
        # Until the board takes a move, it has not answered her (screen_pursuit.nothing_has_answered).
        keep["board_would_not_take"] = True
        return None
    keep.pop("board_would_not_take", None)
    if came.get("ended") in GAME_OVER:
        # Finished, she closes what she opened for it, as a person puts a game away.
        from core.capabilities.host_automation import get_host_automation

        receipt = await get_host_automation().close_app(target_app)
        came["closed"] = bool(getattr(receipt, "success", False))
        _tell(f"That's the game. I've closed {target_app}." if came["closed"] else "That's the game.", news=True)
    return came


#: What she may take a window's words to say of how a game ended, and what each comes to.
THE_ENDS = {"I won": "won", "I lost": "lost", "it was drawn": "drawn", "the window does not say": ""}
#: Characters of a window's words kept with her record of a game: a panel of names, ratings and a clock.
WORDS_KEPT = 600


async def the_words_in(target_app: str) -> str:
    """The words a window shows, each run of them as read, joined: what a person reads in it at a glance."""
    from .screen_pursuit_a_way_on import _its_own_pixels_read

    reading = await _its_own_pixels_read(target_app)
    return " / ".join(" ".join(str(region.get("text") or "").split())
                      for region in (reading or {}).get("layout") or [] if str(region.get("text") or "").strip())


async def what_the_window_says_of_the_end(target_app: str, *, before: str = "") -> tuple[str, str]:
    """How a game ended, as the words in its window say it and her own model reads them: "won", "lost", "drawn", or ""
    where they do not say; with the exact newly visible text that states the result."""
    from typing import Literal

    from pydantic import create_model

    from core.cognition.asking_in_turn import THE_PLACE
    from core.cognition.what_things_are import asked_patiently
    from core.rebuilding.her_model import ask_her_model

    try:
        words = await the_words_in(target_app)
    except (OSError, RuntimeError, ValueError, TypeError) as why:
        logger.info("the window was not read for how the game ended (%s)", why)
        return "", ""
    initial = {span.casefold() for span in before.split(" / ")}
    new = list(dict.fromkeys(span for span in words.split(" / ") if span and span.casefold() not in initial))
    visible = " / ".join(new)[:1500].split(" / ")
    if not new:
        return "", ""
    schema = create_model("_TheEnd", ended=(Literal[tuple(THE_ENDS)], ...), source=(str, ""))  # type: ignore[valid-type]
    prompt = (f"Her side is nearest the bottom in {target_app!r}. Does the window explicitly state a final result "
              "for the current game? Player names, controls, clocks and scores during play are insufficient. "
              "If there is a result, copy the entire text span that states it into source. If the result or "
              "who it applies to is uncertain, answer 'the window does not say'.\n"
              f"Full window context: {words[:1500]}\nNew text spans: {' / '.join(visible)}")
    try:
        got = await asked_patiently(ask_her_model, prompt, schema, 60, matters=THE_PLACE)
    except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
        logger.info("how the game ended was not asked of her model (%s)", why)
        return "", ""
    ended = THE_ENDS.get(str(getattr(got, "ended", "") or ""), "")
    source = " ".join(str(getattr(got, "source", "") or "").split())
    return (ended, source) if ended and source in new and source in visible else ("", "")


#: How a game handed over can end, as opposed to running out of time or losing sight of the board.
GAME_OVER = ("won", "lost", "drawn")


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
_THEY_ANSWERED = "what-they-answered-where"
#: Her answers, worked out ahead, to the situations she expects to meet (core/agency/thinking_before_it_comes.py).
_PREPARED = "worked-out-ahead"
#: Every answer the game gave about what may be done (core/agency/what_the_game_showed.py).
_SHOWN = "what-the-game-showed"
#: What her judgements of this board have gone on to come to (core/cognition/how_far_her_judgement_holds.py).
_HELD_SO = "how-her-judgement-held"
#: The small endings of this board she has worked out to the end (core/agency/what_a_small_ending_comes_to.py).
_ENDINGS = "worked-to-the-end"


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
        if kept["rules"] is not None:
            from core.agency.what_the_game_showed import WhatTheGameShowed

            kept["rules"].shown = WhatTheGameShowed.from_bytes(recall_beside(record, _SHOWN))
            kept["rules"].withheld_takes_read_again()
            kept["rules"].withheld_moves_read_again()
            kept["rules"].refusals_read_again()
    if held.get("went_wrong"):
        from core.cognition.the_furthest_she_has_got import WhatFailedWhere

        kept["went_wrong"] = WhatFailedWhere.from_memory(held["went_wrong"])
    answered = recall_beside(record, _THEY_ANSWERED)
    if answered:
        from core.agency.thinking_on_their_time import WhatTheyAnsweredWhere

        kept["they_answered"] = WhatTheyAnsweredWhere.from_bytes(answered)
    ahead = recall_beside(record, _PREPARED)
    if ahead:
        from core.agency.thinking_before_it_comes import Prepared

        kept["prepared"] = Prepared.from_bytes(ahead)
    ended = recall_beside(record, _ENDINGS)
    if ended:
        from core.agency.what_a_small_ending_comes_to import endings_from_bytes

        kept["endings"] = endings_from_bytes(ended)
    judged = recall_beside(record, _HELD_SO)
    if judged:
        from core.cognition.how_far_her_judgement_holds import HowFarItHolds

        try:
            kept["held"] = HowFarItHolds.from_memory(json.loads(judged))
        # not a failure: what cannot be read back is gathered afresh.
        except (TypeError, ValueError):
            kept["held"] = None
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


@replays("rules")
def _what_no_longer_holds_of_her_rules(world: str, _record: dict[str, Any]) -> list[str]:
    """Her rules of a board as this code reads them, against the board as it began and the answers it gave: at the
    state it began in, the side that acted first has an act, is not already pressed (the kind the game keeps safe
    attacked) and is not already over; and every act the board offered there is one her rules allow."""
    from core.agency.pieces_by_their_kind import FIRST, SECOND, Position, in_check, moves, outcome

    rules = _kept_under(world).get("rules")
    if rules is None:
        return ["her rules could not be read"]
    family = rules.family(unseen_go_every_way=True)
    start = dict(rules.start)
    wrong: list[str] = []
    for side in {True: (FIRST,), False: (SECOND,)}.get(rules.first_was_hers, (FIRST, SECOND)):
        began = Position(cells=start, turn=side, unmoved=frozenset(start))
        legal = moves(family, began)
        if not legal:
            wrong.append(f"side {side} has no act at the state the board began in")
        elif outcome(family, began, legal):
            wrong.append(f"side {side} begins with the board already over: {outcome(family, began, legal)}")
        if in_check(family, began, side):
            wrong.append(f"side {side} begins pressed: its kind the game keeps safe stands attacked")
    shown = getattr(rules, "shown", None)
    if shown is not None:
        logger.info("her rules of %s read again: %s", world, shown.checked(rules.family()).says())
        from core.agency.what_the_game_showed import offered_where_it_began

        for square, missing in offered_where_it_began(shown, start, rules.family()):
            wrong.append(f"the board offered {sorted(missing)} from {square} where it began, and her rules forbid it")
    return wrong


def _keep_what_she_learned(target_app: str, looks: Any, learned: Any, rehearsal: Any, went_wrong: Any = None,
                           they_answered: Any = None, held: Any = None) -> None:
    from dataclasses import asdict

    from core.agency.what_a_kind_is_worth import RESULTS
    from core.cognition.when_more_thought_changes_the_answer import settled
    from core.runtime.what_she_learned import recall, remember, remember_beside

    # How often another depth changed her answer in the searches of this game: the curve she spends a soft time by is
    # fitted again on them, and kept.
    settled()
    # What this run has nothing new of stays as it was kept: a record is written whole, and a run with no rehearsal in
    # hand (a board not yet taking moves) wrote hers without what she had measured each kind to be worth. LIVE 25
    # November 2026 her record had lost the worth 136 rounds had earned her, and she played on the first guess, which
    # holds a leaper worth less than a forward stepper: she gave a leaper for a forward stepper on her third move.
    record: dict[str, Any] = {key: value for key, value in (recall(_her_record(target_app)) or {}).items()
                              if key in ("rules", "went_wrong", "worth")}
    record.update({"app": target_app, "looks": looks.as_memory()})
    if learned is not None:
        record["rules"] = learned.as_memory()
        shown = getattr(learned, "shown", None)
        if shown is not None and (shown.offers or shown.refusals):
            remember_beside(_her_record(target_app), _SHOWN, shown.as_bytes())
    if went_wrong is not None and (went_wrong.ruled_out or went_wrong.suspected or went_wrong.cleared):
        record["went_wrong"] = went_wrong.as_memory()
    if held is not None and held.pairs:
        # What her judgements have come to, kept beside the record: thousands of judgements.
        remember_beside(_her_record(target_app), _HELD_SO, json.dumps(held.as_memory()).encode())
    if they_answered is not None and they_answered.answered:
        # Kept beside the record: thousands of positions where a record holds sixty thousand characters.
        remember_beside(_her_record(target_app), _THEY_ANSWERED, they_answered.as_bytes())
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
