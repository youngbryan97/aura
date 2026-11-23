"""Practice between games: when she has nothing else to do, she rehearses the game she last played.

A player gets better between games as much as in them. Her rehearsal of a board game runs while the other side thinks,
and a game gives it a round or two; what she learns of how the pieces stand together needs tens of thousands of
positions her longer look has searched, which at that rate is dozens of games. So when she is idle (no one has spoken
to her for a while, nothing of hers is at a board, the machine has room and runs off the mains), she plays a round of
the last board she played in her model of it, against herself, and keeps what the round earns with what she learned
there. Each round is the same rehearsal a game runs (core/agency/what_a_kind_is_worth.py): a group is kept only where
it wins its match.

Nothing here knows a game: the board is whichever one she kept a record of last.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger("Aura.PractisingBetweenGames")

__all__ = ["at_a_board", "practise_when_idle"]

#: The suffix her records of boards of pieces carry (core/skills/screen_pursuit_board_of_pieces.py).
_A_BOARD = "-board-of-pieces"

#: How many of her games are being played now: none, and she may practise.
_PLAYING = 0
#: The round under way, if any, and the rehearsal it carries on, with which record it is of and when that was written.
_RUNNING: asyncio.Task | None = None
_HELD: dict[str, Any] = {}
#: Why she last did not practise, said once each time it changes.
_WAITING = ""
#: Situations worked out ahead in one round of preparing, and how many turns from the start she walks at most: a
#: round of preparing takes about as long as a round of rehearsal (two minutes), and the opening is where situations
#: come back.
PREPARED_A_ROUND = 4
PREPARES_TURNS = 20
#: Whether the next round prepares rather than rehearses: they take turns.
_NEXT: dict[str, bool] = {"prepares": False}


def _waits(why: str) -> bool:
    global _WAITING
    if why != _WAITING:
        _WAITING = why
        if why:
            logger.info("practice between games waits: %s", why)
    return False


class at_a_board:  # noqa: N801 - used as a statement: `with at_a_board():`
    """While a game is played, she does not practise."""

    def __enter__(self) -> at_a_board:
        global _PLAYING
        _PLAYING += 1
        return self

    def __exit__(self, *exc: object) -> None:
        global _PLAYING
        _PLAYING = max(0, _PLAYING - 1)


def _on_mains() -> bool:
    """Whether the machine runs off the mains: practice is not worth its battery. LIVE 14 November 2026 a night of
    work on battery left it flat."""
    from core.runtime.resource_observation import get_resource_observer

    try:
        return bool(get_resource_observer().power().plugged)
    # not a failure: a machine that cannot say how it is powered is taken to be on battery, and she does not practise.
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _the_board_she_last_played() -> tuple[str, tuple[int, int]] | None:
    from core.runtime.what_she_learned import when_each_was_written

    boards = [(written, name) for name, written in when_each_was_written().items() if name.endswith(_A_BOARD)]
    if not boards:
        return None
    written, name = max(boards)
    return name, written


def _prepare_some(name: str) -> int:
    """A round of thinking before it comes (core/agency/thinking_before_it_comes.py): the situations she is likeliest
    to meet at the board kept as ``name``, worked out ahead under the rules and judgement she plays by, kept beside its
    record. How many were worked out. Blocks; run it in a thread."""
    from core.agency.pieces_by_their_kind import FIRST, SECOND, Position
    from core.agency.playing_pieces_by_kind import _cores_free, _Rehearsing
    from core.agency.thinking_before_it_comes import Prepared
    from core.runtime.what_she_learned import remember_beside
    from core.skills.screen_pursuit_board_of_pieces import _PREPARED, _kept_under

    kept = _kept_under(name)
    rules = kept.get("rules")
    if rules is None or not rules.start:
        return 0
    family = rules.family(unseen_go_every_way=True)
    weights = _Rehearsing(_HELD.get("rehearsal") or kept.get("rehearsal")).weights_for(family)
    prepared = kept.get("prepared") or Prepared()
    start = Position(cells=dict(rules.start), turn=SECOND if rules.first_was_hers is False else FIRST,
                     unmoved=frozenset(rules.start))
    began = time.monotonic()
    done = prepared.prepare(family, start, weights, they_answered=kept.get("they_answered"),
                            went_wrong=_HELD.get("went_wrong") or kept.get("went_wrong"),
                            positions=PREPARED_A_ROUND, plies=PREPARES_TURNS, threads=_cores_free())
    if done and not _PLAYING:
        remember_beside(name, _PREPARED, prepared.as_bytes())
    logger.info("prepared %s between games: %d situation(s) worked out ahead in %.0f s, %d held", name, done,
                time.monotonic() - began, len(prepared.answers))
    return done


def _one_round(name: str, written: tuple[int, int]) -> dict[str, Any] | None:
    """One round of practice at the board kept as ``name``, kept with it: a round of rehearsal, or, every other round
    while anything is left to work out, a round of preparing. Blocks; run it in a thread."""
    if _NEXT["prepares"]:
        _NEXT["prepares"] = False
        if _prepare_some(name):
            return {"prepared": True}
    else:
        _NEXT["prepares"] = True
    return _one_rehearsal(name, written)


def _one_rehearsal(name: str, written: tuple[int, int]) -> dict[str, Any] | None:
    """One round of rehearsal of the board kept as ``name``, kept with it. Blocks; run it in a thread."""
    from core.agency.what_a_kind_is_worth import rehearse
    from core.skills.screen_pursuit_board_of_pieces import _keep_what_she_learned, _kept_under

    if _HELD.get("name") != name or _HELD.get("written") != written:
        # A record written since (a game she played) is what she carries on from; otherwise the rehearsal she has in
        # hand, with the positions it gathered this sitting, is.
        kept = _kept_under(name)
        if kept.get("rehearsal") is None or kept.get("rules") is None or kept.get("looks") is None:
            return None
        _HELD.clear()
        _HELD.update(name=name, kept=kept, rehearsal=kept["rehearsal"])
    kept, rehearsal = _HELD["kept"], _HELD["rehearsal"]
    family = kept["rules"].family()
    if not any(kind.steps for kind in family.kinds.values()):
        return None
    began = time.monotonic()
    rehearsal = rehearse(family, family.the_start(), rehearsal, seed=int(time.time()) % 100_000)
    _HELD["rehearsal"] = rehearsal
    if _PLAYING:
        # A game begun while the round ran owns her record of the board, and writes it when it ends: the round is not
        # written over it. LIVE 22 November 2026 a round begun before game 18 was asked for finished while she was
        # getting to the board.
        logger.info("practised %s between games, a round not kept: a game began while it ran", name)
        return rehearsal.last
    _keep_what_she_learned(kept.get("app") or name, kept["looks"], kept["rules"], rehearsal, kept.get("went_wrong"))
    # What she just wrote is what she holds: it is not read back as a game's.
    held = _the_board_she_last_played()
    if held is not None and held[0] == name:
        _HELD["written"] = held[1]
    logger.info("practised %s between games, a round in %.0f s: %s", name, time.monotonic() - began, rehearsal.last)
    return rehearsal.last


def practise_when_idle(orch: Any, track: Any = None) -> bool:
    """A round of practice, where she is idle, no game of hers is being played, and none is under way. Whether one
    was begun."""
    global _RUNNING
    if _PLAYING or (_RUNNING is not None and not _RUNNING.done()):
        return False
    from core.runtime.background_policy import (
        IDLE_COGNITION_BACKGROUND_POLICY,
        background_activity_reason,
    )

    # Her own work, not something said to anyone: it does not wait for someone to have spoken to her since she woke.
    # LIVE 22 November 2026 her first hour on 225d54fcd ran no round, and the reason was "no_user_anchor".
    why = background_activity_reason(orch, profile=IDLE_COGNITION_BACKGROUND_POLICY, needs_the_model=False,
                                     allow_no_user_anchor=True)
    if why:
        return _waits(str(why))
    if not _on_mains():
        return _waits("the machine is on its battery")
    board = _the_board_she_last_played()
    if board is None:
        return _waits("she has played no board yet")
    try:
        loop = asyncio.get_running_loop()
    # not a failure: outside a running loop there is nothing to practise beside.
    except RuntimeError:
        return False
    _waits("")
    work = asyncio.to_thread(_one_round, *board)
    _RUNNING = track("metabolic.practice_between_games", work) if track is not None else loop.create_task(work)
    if not isinstance(_RUNNING, asyncio.Task):
        _RUNNING = None
    return True
