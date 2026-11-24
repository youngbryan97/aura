"""What can still change is proved by reachability, not by looking ahead.

Suggestions of 23 November 2026, item 18: a search misjudges a position where the change it waits for lies past every
look; a fortress stands a side ahead and is drawn. Which things can never move again, where the rest could ever go, and
whether the two sides could ever meet are found over the moves alone (core/cognition/what_can_still_change.py), on a
board under the rules she holds (core/agency/what_cannot_change_on_a_board.py).
"""
from __future__ import annotations

import time

from core.agency.what_cannot_change_on_a_board import what_can_change
from core.cognition.what_can_still_change import could_reach, staying
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

#: Pawns locked across every file, each king on its own side of them.
LOCKED = "4k3/8/1p1p1p1p/pPpPpPpP/P1P1P1P1/8/8/4K3 w - -"
#: The same with the a and b files opened: a king can walk round.
OPENED = "4k3/8/3p1p1p/2pPpPpP/2P1P1P1/8/8/4K3 w - -"


def test_in_any_world_what_stays_is_what_has_nowhere_to_step_but_onto_what_stays() -> None:
    # Three counters on a line of five places, each stepping one place right: the two at the end are stuck behind
    # the wall of the line's end and each other, the one at the start has somewhere to go.
    at = {"a": 0, "b": 3, "c": 4}

    def has_somewhere(thing, held):
        ahead = at[thing] + 1
        return ahead < 5 and ahead not in {at[other] for other in held if other != thing}

    assert staying(at, has_somewhere) == frozenset({"b", "c"})
    assert could_reach(0, lambda place: [place + 1] if place + 1 < 3 else []) == frozenset({0, 1, 2})


def test_pawns_locked_across_every_file_leave_nothing_that_can_change() -> None:
    found = what_can_change(CHESS, from_fen(LOCKED))
    assert not found.can, found.why
    assert len(found.staying) == 16, "every pawn stays; the kings move"


def test_an_open_file_lets_a_king_walk_round_and_take() -> None:
    found = what_can_change(CHESS, from_fen(OPENED))
    assert found.can and found.why == "a piece may be taken"


def test_a_pawn_free_to_run_can_change_everything() -> None:
    free = what_can_change(CHESS, from_fen("4k3/8/8/8/8/8/P7/4K3 w - -"))
    assert free.can


def test_where_a_side_left_no_move_loses_a_locked_board_can_still_be_decided() -> None:
    from dataclasses import replace

    found = what_can_change(replace(CHESS, no_move_unattacked="loses"), from_fen(LOCKED))
    assert found.can


def test_an_ordinary_position_is_told_able_to_change_at_once() -> None:
    began = time.perf_counter()
    for _ in range(20):
        assert what_can_change(CHESS, CHESS.the_start()).can
    assert (time.perf_counter() - began) / 20 < 0.05


#: As LOCKED, but her pawn on g3: pushed to g4, it locks the board.
ONE_FROM_LOCKED = "4k3/8/1p1p1p1p/pPpPpPpP/P1P1P3/6P1/8/4K3 w - -"


def _a_game_at(fen: str):
    from core.agency.how_the_pieces_move import HowThePiecesMove
    from core.agency.playing_pieces_by_kind import _Game

    position = from_fen(fen)
    return _Game(learned=HowThePiecesMove.from_the_start(8, 8, dict(position.cells)), believed=position), position


def test_behind_she_takes_the_move_after_which_nothing_can_change() -> None:
    from core.agency.pieces_by_their_kind import Move, moves
    from core.agency.playing_pieces_by_kind import _held_to_what_can_change
    from core.agency.searching_by_kind import Found

    game, position = _a_game_at(ONE_FROM_LOCKED)
    assert what_can_change(CHESS, position).can
    king_aside = Move((0, 4), (0, 3))
    behind = Found(king_aside, -300, 12, 1000, [king_aside])
    taken = _held_to_what_can_change(game, CHESS, moves(CHESS, position), [], behind, lambda aside: behind)
    assert (taken.move.start, taken.move.end) == ((2, 6), (3, 6)) and taken.by == "nothing can change after it"


def test_ahead_she_keeps_clear_of_a_move_after_which_nothing_can_change() -> None:
    from core.agency.pieces_by_their_kind import Move, moves
    from core.agency.playing_pieces_by_kind import _held_to_what_can_change
    from core.agency.searching_by_kind import Found

    game, position = _a_game_at(ONE_FROM_LOCKED)
    push, king_aside = Move((2, 6), (3, 6)), Move((0, 4), (0, 3))
    asked = []

    def search_only(aside):
        asked.append(list(aside))
        return Found(king_aside, 250, 10, 1000, [king_aside])

    kept = _held_to_what_can_change(game, CHESS, moves(CHESS, position), [], Found(push, 300, 12, 1000, [push]),
                                    search_only)
    assert kept.move == king_aside and asked == [[push]]
