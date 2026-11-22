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


def _one_round(name: str, written: tuple[int, int]) -> dict[str, Any] | None:
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

    if background_activity_reason(orch, profile=IDLE_COGNITION_BACKGROUND_POLICY, needs_the_model=False):
        return False
    if not _on_mains():
        return False
    board = _the_board_she_last_played()
    if board is None:
        return False
    try:
        loop = asyncio.get_running_loop()
    # not a failure: outside a running loop there is nothing to practise beside.
    except RuntimeError:
        return False
    work = asyncio.to_thread(_one_round, *board)
    _RUNNING = track("metabolic.practice_between_games", work) if track is not None else loop.create_task(work)
    if not isinstance(_RUNNING, asyncio.Task):
        _RUNNING = None
    return True
