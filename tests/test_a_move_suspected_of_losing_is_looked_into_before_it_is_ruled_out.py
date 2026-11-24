"""A move after which her judgement fell is suspected, looked into, and only then ruled out or cleared.

From the suggestions of 23 November 2026: after a lost game, the move after which her judgement fell furthest was
ruled out at that place for good. A fall after a move is a reason to look, not proof: the other side may have found
something against a good move, her judgement before may have been wrong, or the loss began earlier. Now the moves
after the largest falls are suspects; between games a longer look weighs each against the best of what else was open
there, and rules it out only where something else is better by more than the looks' own unsteadiness.
"""
from __future__ import annotations

import core.agency.practising_between_games as practice
import core.agency.searching_by_kind as searching
from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import Position, moves
from core.agency.playing_pieces_by_kind import (
    _a_place,
    _Game,
    _named,
    _position_from_place,
    _where_it_went_wrong,
)
from core.agency.searching_by_kind import Found
from core.cognition.the_furthest_she_has_got import WhatFailedWhere
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

START = dict(CHESS.start)


def test_a_lost_game_leaves_suspects_and_rules_nothing_out() -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, START)
    game = _Game(learned=learned, believed=CHESS.the_start())
    game.judged = [("a", "1,4-3,4", 40), ("b", "0,6-2,5", 10), ("c", "0,3-4,7", -300), ("d", "0,5-3,2", -320)]
    _where_it_went_wrong(game)
    assert not game.went_wrong.ruled_out
    assert game.went_wrong.next_suspect() == ("b", "0,6-2,5")
    assert set(game.went_wrong.suspected) == {"a", "b", "c"}


def test_a_suspect_settles_one_way_or_the_other_and_is_kept() -> None:
    failed = WhatFailedWhere()
    failed.suspect("here", "e2-e4", 200)
    failed.settled("here", "e2-e4", worse=True)
    assert failed.already_failed_at("here") == {"e2-e4"} and failed.next_suspect() is None
    failed.suspect("there", "g1-f3", 90)
    failed.settled("there", "g1-f3", worse=False)
    failed.suspect("there", "g1-f3", 120)  # cleared once: not suspected again for the same fall
    assert failed.next_suspect() is None and not failed.already_failed_at("there")
    kept = WhatFailedWhere.from_memory(failed.as_memory())
    assert kept.ruled_out == failed.ruled_out and kept.cleared == failed.cleared


def test_a_situation_reads_back_from_its_name() -> None:
    begun = CHESS.the_start()
    assert _position_from_place(_a_place(begun, START), START) == Position(begun.cells, begun.turn, begun.unmoved)
    back = Position(cells=begun.cells, turn=begun.turn, unmoved=begun.unmoved - {(0, 4)})
    read = _position_from_place(_a_place(back, START), START)
    assert read is not None and read.unmoved == back.unmoved
    passing = from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6")
    read = _position_from_place(_a_place(passing, START, CHESS), START)
    assert read is not None and read.passable == passing.passable


def _investigated(monkeypatch, *, it: int, instead: int) -> WhatFailedWhere:
    begun = CHESS.the_start()
    legal = sorted(moves(CHESS, begun), key=_named)
    suspect, better = legal[0], legal[1]
    failed = WhatFailedWhere()
    failed.suspect(_a_place(begun, START), _named(suspect), 150)
    from types import SimpleNamespace

    rules = SimpleNamespace(family=lambda **_: CHESS, start=START)  # rules that are the game's
    kept = {"rules": rules, "went_wrong": failed, "looks": object(), "rehearsal": None, "app": "A Board"}
    monkeypatch.setattr(practice, "_held", lambda name, written: kept)
    monkeypatch.setattr(practice, "_keep_what_she_learned", lambda *a, **k: None, raising=False)
    import core.skills.screen_pursuit_board_of_pieces as board

    monkeypatch.setattr(board, "_keep_what_she_learned", lambda *a, **k: None)
    monkeypatch.setattr(practice, "_HELD", {})
    looks = iter([Found(better, 0, 9, 1, [better]), Found(None, it, 9, 1, []), Found(None, instead, 9, 1, [])])
    monkeypatch.setattr(searching, "search", lambda *a, **k: next(looks))
    # 2 where the suspect was ruled out, 1 where it was cleared: what practice credits the round with.
    assert practice._investigate_some("A Board", (0, 0)) == (2 if instead - it < 0 else 1)
    return failed


def test_a_suspect_with_something_clearly_better_instead_is_ruled_out(monkeypatch) -> None:
    # Their judgement after her move +300, after the other -100: the other is 400 better for her.
    failed = _investigated(monkeypatch, it=300, instead=-100)
    assert failed.ruled_out and failed.next_suspect() is None


def test_a_suspect_no_worse_than_the_rest_is_cleared(monkeypatch) -> None:
    failed = _investigated(monkeypatch, it=-50, instead=-40)
    assert not failed.ruled_out and failed.cleared and failed.next_suspect() is None
