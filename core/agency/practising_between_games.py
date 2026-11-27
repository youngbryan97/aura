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
#: The kinds of round: rehearsing, preparing, looking into where a lost game went wrong, and working a small ending out
#: to the end. Which comes next is whichever has lately paid best, with a margin for how little each has been tried
#: (core/cognition/where_practice_pays.py); a kind with nothing to do gives its turn to the next.
_ROUNDS = ("rehearses", "prepares", "investigates", "looks back", "looks again", "works out an ending")
_PAYS: dict[str, Any] = {}


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
    to meet at the board kept as ``name``, worked out ahead among the choices each offers her, kept beside its
    record. How many were worked out. Blocks; run it in a thread."""
    from core.agency.pieces_by_their_kind import FIRST, SECOND, Position
    from core.agency.playing_pieces_by_kind import SEEN_TO_BELIEVE, _cores_free, _Rehearsing
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
                            positions=PREPARED_A_ROUND, plies=PREPARES_TURNS, threads=_cores_free(),
                            doubtful=lambda position, move: rules.seen_going_there(position, move) < SEEN_TO_BELIEVE)
    if done and not _PLAYING:
        remember_beside(name, _PREPARED, prepared.as_bytes())
    logger.info("prepared %s between games: %d situation(s) worked out ahead in %.0f s, %d held", name, done,
                time.monotonic() - began, len(prepared.answers))
    return done


def _one_round(name: str, written: tuple[int, int]) -> dict[str, Any] | None:
    """One round of practice at the board kept as ``name``, kept with it: of rehearsal, preparing, looking into where a
    lost game went wrong and working an ending out, the kind that has lately paid best, a kind with nothing left to do
    giving its turn to the next. A round pays where it changes what she will do: a group of weights kept, a suspect
    ruled out, an ending worked out, situations worked out ahead. Blocks; run it in a thread."""
    from core.runtime.interpreter_turns import short_turns

    with short_turns():
        return _one_round_of(name, written)


def _one_round_of(name: str, written: tuple[int, int]) -> dict[str, Any] | None:
    from core.cognition.where_practice_pays import WherePracticePays

    pays = _PAYS.setdefault(name, WherePracticePays())
    for kind in pays.ranked(_ROUNDS):
        if kind == "prepares":
            done = _prepare_some(name)
            if done:
                pays.came_to(kind, True)
                return {"prepared": True}
        elif kind == "investigates":
            settled = _investigate_some(name, written)
            if settled:
                pays.came_to(kind, settled == 2)
                return {"investigated": True, "ruled out": settled == 2}
        elif kind == "looks back":
            looked, kept = _look_back_some(name, written)
            if looked:
                # A position kept from looking back changes what she learns how the pieces stand together from.
                pays.came_to(kind, kept > 0)
                return {"looked back": looked, "kept": kept > 0}
        elif kind == "looks again":
            looked = _look_again_some(name, written)
            if looked:
                # A longer look taken again changes what she learns how the pieces stand together from.
                pays.came_to(kind, True)
                return {"looked again": looked}
        elif kind == "works out an ending":
            if _work_out_an_ending(name, written):
                pays.came_to(kind, True)
                return {"worked out an ending": True}
        else:
            last = _one_rehearsal(name, written)
            if last is not None:
                pays.came_to(kind, bool(last.get("kept")))
            return last
    return None


def _look_back_some(name: str, written: tuple[int, int]) -> tuple[int, int]:
    """A round of looking again, for far longer than a turn could take, at places games in the world showed her wrong
    (core/agency/where_she_was_shown_wrong.py), for about as long as her last round of rehearsal took. How many places
    were looked at, and how many positions were kept from them. Blocks; run it in a thread."""
    from core.agency.playing_pieces_by_kind import _cores_free, _Rehearsing
    from core.agency.where_she_was_shown_wrong import LOOK_BACK_S, look_back_on_some
    from core.skills.screen_pursuit_board_of_pieces import _keep_what_she_learned

    if _held(name, written) is None:
        return 0, 0
    kept, rehearsal = _HELD["kept"], _HELD["rehearsal"]
    shown = getattr(rehearsal, "shown_wrong", None)
    if shown is None or not shown.waiting:
        return 0, 0
    rules = kept["rules"]
    family = rules.family(unseen_go_every_way=True)
    weights = _Rehearsing(rehearsal).weights_for(family)
    before = len(shown.parts)
    looked = look_back_on_some(shown, family, weights, dict(rules.start),
                               seconds=max(LOOK_BACK_S, float(_HELD.get("round_s") or 0.0)), threads=_cores_free())
    if looked and not _PLAYING:
        _keep_what_she_learned(kept.get("app") or name, kept["looks"], rules, rehearsal, kept.get("went_wrong"))
        held = _the_board_she_last_played()
        if held is not None and held[0] == name:
            _HELD["written"] = held[1]
    return looked, len(shown.parts) - before


def _look_again_some(name: str, written: tuple[int, int]) -> int:
    """A round of looking again, with the weights she plays by now, at the kept positions whose longer look is oldest
    (core/agency/looking_again_at_what_she_kept.py), for about as long as her last round of rehearsal took. How many
    were looked at. Blocks; run it in a thread."""
    from core.agency.looking_again_at_what_she_kept import looked_at_again
    from core.agency.playing_pieces_by_kind import _cores_free, _Rehearsing
    from core.agency.what_a_kind_is_worth import LONGER_LOOK_NODES
    from core.skills.screen_pursuit_board_of_pieces import _keep_what_she_learned

    if _held(name, written) is None:
        return 0
    kept, rehearsal = _HELD["kept"], _HELD["rehearsal"]
    store = getattr(rehearsal, "looked_at_longer", None)
    if store is None or not store.parts:
        return 0
    # As long as a round of rehearsal takes, once one has been measured this sitting.
    if not _HELD.get("round_s"):
        return 0
    rules = kept["rules"]
    family = rules.family()
    weights = _Rehearsing(rehearsal).weights_for(family)
    looked = looked_at_again(store, family, weights, dict(rules.start), now=int(rehearsal.rounds),
                             seconds=float(_HELD["round_s"]), nodes=LONGER_LOOK_NODES, threads=_cores_free())
    if looked and not _PLAYING:
        _keep_what_she_learned(kept.get("app") or name, kept["looks"], rules, rehearsal, kept.get("went_wrong"))
        held = _the_board_she_last_played()
        if held is not None and held[0] == name:
            _HELD["written"] = held[1]
    if looked:
        logger.info("looked again at %d kept position(s) of %s with the weights she plays by now", looked, name)
    return looked


def _work_out_an_ending(name: str, written: tuple[int, int]) -> int:
    """A round of working one small ending of the board kept as ``name`` out to the end, under the rules she holds
    for its kinds (core/agency/what_a_small_ending_comes_to.py): the first of those worth working out that she has not
    worked out under those rules. How many were worked out. Blocks; run it in a thread."""
    from core.agency.what_a_small_ending_comes_to import (
        an_ending_worked_out,
        endings_as_bytes,
        endings_worth_working_out,
        rules_of_the_ending,
    )
    from core.runtime.what_she_learned import remember_beside
    from core.skills.screen_pursuit_board_of_pieces import _ENDINGS

    kept = _held(name, written)
    if kept is None:
        return 0
    rules = kept["rules"]
    family = rules.family(unseen_go_every_way=True)
    endings = kept.setdefault("endings", {})
    for pieces in endings_worth_working_out(rules):
        held = endings.get(pieces)
        if held is not None and held.rules == rules_of_the_ending(family, pieces):
            continue
        began = time.monotonic()
        endings[pieces] = an_ending_worked_out(family, pieces)
        if not _PLAYING:
            remember_beside(name, _ENDINGS, endings_as_bytes(endings))
        logger.info("worked out the ending %s of %s between games in %.0f s", pieces, name, time.monotonic() - began)
        return 1
    return 0


#: How long each look at a suspected move takes: the look measured to choose better than her game pace
#: (core/agency/thinking_before_it_comes.py, LOOKS_FOR_S), split between finding what else was open and weighing the two.
INVESTIGATES_FOR_S = 30.0


def _held(name: str, written: tuple[int, int]) -> dict[str, Any] | None:
    """Her record of the board kept as ``name``, as practice holds it: one copy, so that what one round changes the
    next writes back. A record written since (a game she played) is what she carries on from."""
    from core.skills.screen_pursuit_board_of_pieces import _kept_under

    if _HELD.get("name") != name or _HELD.get("written") != written:
        kept = _kept_under(name)
        if kept.get("rehearsal") is None or kept.get("rules") is None or kept.get("looks") is None:
            return None
        _HELD.clear()
        _HELD.update(name=name, kept=kept, rehearsal=kept["rehearsal"])
    return _HELD["kept"]


def _investigate_some(name: str, written: tuple[int, int]) -> int:
    """A round of looking into where a lost game went wrong: the suspect after which her judgement fell furthest is
    weighed against the best of what else was open there, each by a longer look from the situation it leaves. Worse by
    more than the looks' own unsteadiness, it is ruled out there; otherwise it is cleared, and the next suspect, which
    may come earlier in the game, is looked into next round. Nought where there was nothing to look into, 1 where the
    suspect was cleared, 2 where it was ruled out. Blocks; run it in a thread."""
    from core.agency.pieces_by_their_kind import FIRST, after, moves
    from core.agency.playing_pieces_by_kind import (
        _cores_free,
        _named,
        _position_from_place,
        _Rehearsing,
    )
    from core.agency.searching_by_kind import search
    from core.skills.screen_pursuit_board_of_pieces import _keep_what_she_learned

    kept = _held(name, written)
    went_wrong = kept.get("went_wrong") if kept is not None else None
    if kept is None or went_wrong is None:
        return 0
    suspect = went_wrong.next_suspect()
    if suspect is None:
        return 0
    place, act = suspect
    rules = kept["rules"]
    family = rules.family(unseen_go_every_way=True)
    position = _position_from_place(place, dict(rules.start))
    legal = moves(family, position) if position is not None else []
    mine = [m for m in legal if _named(m) == act]
    others = [m for m in legal if _named(m) != act]
    began = time.monotonic()
    worse = False
    if mine and others:
        weights = _Rehearsing(_HELD.get("rehearsal") or kept.get("rehearsal")).weights_for(family)
        look = INVESTIGATES_FOR_S / 3
        threads = _cores_free()
        best_else = search(family, position, seconds=look, weights=weights, set_aside=mine, threads=threads,
                           wary=FIRST).move
        if best_else is not None:
            # Each weighed by a look of the same length from the situation it leaves, from her side.
            it = search(family, after(family, position, mine[0]), seconds=look, weights=weights, threads=threads,
                        wary=FIRST)
            instead = search(family, after(family, position, best_else), seconds=look, weights=weights,
                             threads=threads, wary=FIRST)
            noise = max(int(it.unsteady), int(instead.unsteady), 1)
            worse = (-instead.score) - (-it.score) > noise
            logger.info("looked into %s where a lost game went wrong: %s; %s instead judged %+d against it, by looks "
                        "unsteady by %d", act, "ruled out" if worse else "cleared", _named(best_else),
                        (-instead.score) - (-it.score), noise)
    went_wrong.settled(place, act, worse=worse)
    if not _PLAYING:
        _keep_what_she_learned(kept.get("app") or name, kept["looks"], rules,
                               _HELD.get("rehearsal") or kept.get("rehearsal"), went_wrong)
        # What she just wrote is what she holds: it is not read back as a game's.
        held = _the_board_she_last_played()
        if held is not None and held[0] == name:
            _HELD["written"] = held[1]
    logger.info("investigated %s between games in %.0f s", name, time.monotonic() - began)
    return 2 if worse else 1


def _one_rehearsal(name: str, written: tuple[int, int]) -> dict[str, Any] | None:
    """One round of rehearsal of the board kept as ``name``, kept with it. Blocks; run it in a thread."""
    from core.agency.what_a_kind_is_worth import rehearse
    from core.skills.screen_pursuit_board_of_pieces import _keep_what_she_learned

    # A record written since (a game she played) is what she carries on from; otherwise the rehearsal she has in hand,
    # with the positions it gathered this sitting, is.
    if _held(name, written) is None:
        return None
    kept, rehearsal = _HELD["kept"], _HELD["rehearsal"]
    family = kept["rules"].family()
    if not any(kind.steps for kind in family.kinds.values()):
        return None
    began = time.monotonic()
    from core.agency.how_she_looks_set_by_trying import a_round_of_trying, wants_a_round

    if _HELD.get("round_s") and wants_a_round(rehearsal):
        # Rounds of trying how she looks take turns with rounds of rehearsal, each as long as the last of those took.
        rehearsal.last = a_round_of_trying(rehearsal, family, seconds=float(_HELD["round_s"]))
    else:
        rehearsal = rehearse(family, family.the_start(), rehearsal, seed=int(time.time()) % 100_000)
        _HELD["round_s"] = time.monotonic() - began
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


#: How long nobody must have touched the machine before she takes its screen for a practice game: longer than one of
#: her games against the world's other player takes, so a person who stepped away does not come back to find one
#: under way, nor is met by one started as they sit down.
AWAY_FOR_S = 900.0
#: The longest a practice game is given, start to end: within the lease Bryan's standing yes to it gives a call
#: (core/executive/standing_authority.py, aura.practice-games, 900 s), with time to spare for its end.
A_PRACTICE_GAME_S = 840.0
#: The round of rehearsal after which the last practice game was begun: a round of practice comes between two games.
_GAME_AFTER_ROUND: list[int] = [-1]


def nobody_at_the_machine() -> bool:
    """Whether nobody has touched the machine for AWAY_FOR_S: what a practice game waits for, here and in the
    standing yes that lets it take the desktop (core/executive/her_own_passes.py)."""
    from core.capabilities import window_server

    return window_server.seconds_since_someone_touched_it() >= AWAY_FOR_S


def _a_practice_game_due(board: tuple[str, tuple[int, int]]) -> tuple[str, str, int] | None:
    """The board, its application and the setting of the other player's strength a practice game is to be played at,
    where one is due: nobody has touched the machine for AWAY_FOR_S, and a round of rehearsal has run since the last
    (core/agency/where_her_results_vary.py)."""
    from core.agency.where_her_results_vary import the_setting_to_play
    from core.runtime.proof_policy import proof_run_active

    # A run of tests or proofs never takes the screen of the machine it runs on.
    if proof_run_active() or not nobody_at_the_machine():
        return None
    name, written = board
    if _held(name, written) is None:
        return None
    kept, rehearsal = _HELD["kept"], _HELD["rehearsal"]
    app = str(kept.get("app") or "")
    if not app or int(getattr(rehearsal, "rounds", 0)) <= _GAME_AFTER_ROUND[0]:
        return None
    return name, app, the_setting_to_play(rehearsal)


async def _a_practice_game(name: str, app: str, setting: int) -> dict[str, Any] | None:
    """A game against the world's other player at ``setting`` of its strength, its points kept where her results vary
    and against any keep of her practice on trial there. What came of it."""
    import json

    from core.agency.where_her_results_vary import came_from_a_game
    from core.container import ServiceContainer
    from core.skills.screen_pursuit_a_setting import FOR_HER_PRACTICE
    from core.skills.screen_pursuit_board_of_pieces import _keep_what_she_learned

    rehearsal = _HELD.get("rehearsal")
    _GAME_AFTER_ROUND[0] = int(getattr(rehearsal, "rounds", 0))
    goal = f"Play a game against the other player, with the other player's strength set to {setting}."
    # Her hands on the desktop act through her capability engine, where her will decides on them and they run under
    # its lease, as every other act of hers on the desktop does. LIVE 27 November 2026 the first practice game called
    # the pursuit itself, and its first click was refused as "called outside governed context".
    engine = ServiceContainer.get("capability_engine", default=None)
    if engine is None:
        logger.info("a practice game in %s not begun: no capability engine to act through", app)
        return None
    logger.info("a practice game in %s: %s", app, goal)
    target = {"goal": goal, "success_when": "the game has ended", "target_app": app,
              "max_seconds": A_PRACTICE_GAME_S, "narrate": False}
    context = {"origin": "her_practice", "source": "her_practice",
               "objective": "a practice game against the world's other player where her results vary"}
    # What this game changes in the application is put back for the next game she is asked to play there.
    marked = FOR_HER_PRACTICE.set(True)
    try:
        result = await engine.execute("computer_use", {"action": "pursue_on_screen", "target": json.dumps(target)},
                                      context)
    finally:
        FOR_HER_PRACTICE.reset(marked)
    result = dict(result or {}) if isinstance(result, dict) else {}
    if "handed_over" not in result and isinstance(result.get("result"), dict):
        result = dict(result["result"])
    played = result.get("handed_over") or {}
    points = {"won": 1.0, "lost": 0.0, "drawn": 0.5}.get(str(played.get("ended") or ""))
    settings = dict(played.get("settings_set") or {})
    if points is None or not settings:
        logger.info("a practice game in %s came to nothing kept: ended %r, settings set %s", app, played.get("ended"),
                    settings)
        return None
    # The game wrote her record as it ended: what it wrote is carried on from.
    held = _written_now(name)
    if held is None or _held(name, held) is None:
        return None
    kept, rehearsal = _HELD["kept"], _HELD["rehearsal"]
    at = float(next(iter(settings.values())))
    said = came_from_a_game(rehearsal, at, points)
    _keep_what_she_learned(kept.get("app") or name, kept["looks"], kept["rules"], rehearsal, kept.get("went_wrong"))
    last = _the_board_she_last_played()
    if last is not None and last[0] == name:
        _HELD["written"] = last[1]
    logger.info("a practice game in %s at %s: %s%s", app, at, points, f"; the keep on trial: {said}" if said else "")
    return {"practice game": at, "points": points, "trial": said}


def _written_now(name: str) -> tuple[int, int] | None:
    held = _the_board_she_last_played()
    return held[1] if held is not None and held[0] == name else None


def practise_when_idle(orch: Any, track: Any = None) -> bool:
    """A round of practice, where she is idle, no game of hers is being played, and none is under way. Whether one
    was begun."""
    global _RUNNING
    if _PLAYING or (_RUNNING is not None and not _RUNNING.done()):
        return False
    from core.runtime.background_policy import (
        COMPUTE_ONLY_BACKGROUND_POLICY,
        background_activity_reason,
    )

    # Her own work, not something said to anyone: it does not wait for someone to have spoken to her since she woke.
    # LIVE 22 November 2026 her first hour on 225d54fcd ran no round, and the reason was "no_user_anchor".
    why = background_activity_reason(orch, profile=COMPUTE_ONLY_BACKGROUND_POLICY, needs_the_model=False,
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
    game = _a_practice_game_due(board)
    if game is not None:
        work = _a_practice_game(*game)
        _RUNNING = track("metabolic.practice_between_games", work) if track is not None else loop.create_task(work)
        if not isinstance(_RUNNING, asyncio.Task):
            _RUNNING = None
        return True
    work = asyncio.to_thread(_one_round, *board)
    _RUNNING = track("metabolic.practice_between_games", work) if track is not None else loop.create_task(work)
    if not isinstance(_RUNNING, asyncio.Task):
        _RUNNING = None
    return True
