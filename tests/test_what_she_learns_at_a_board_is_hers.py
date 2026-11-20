"""What she learns at a board is hers to keep, and what a piece is worth is measured by her own rehearsal.

The looks, the rules she learned and the weights she earned are written to her per-world record and read back the same;
rehearsal in her model keeps new weights only where they beat the ones she plays by.
"""
from __future__ import annotations

import random

import pytest

from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, after, moves, outcome
from core.agency.searching_by_kind import first_weights
from core.perception.pieces_by_their_look import Looks, squares_of_one_ground
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, NOBODYS_GAME


def _learned(hidden, plies: int) -> HowThePiecesMove:
    rng = random.Random(3)
    position = hidden.the_start()
    learned = HowThePiecesMove.from_the_start(hidden.rows, hidden.columns, hidden.start)
    for _ in range(plies):
        legal = moves(hidden, position)
        if not legal or outcome(hidden, position, legal):
            position = hidden.the_start()
            continue
        if position.turn == FIRST:
            family = learned.family()
            for square, (side, _kind) in list(position.cells.items()):
                if side == FIRST:
                    learned.offered(position, square, {m.end for m in legal if m.start == square}, family=family)
        move = rng.choice(legal)
        learned.watched(position, move)
        position = after(hidden, position, move)
    return learned


@pytest.mark.parametrize("hidden", [CHESS, NOBODYS_GAME], ids=["chess", "nobodys_game"])
def test_the_rules_she_learned_come_back_the_same(hidden) -> None:
    learned = _learned(hidden, 200)
    again = HowThePiecesMove.from_memory(learned.as_memory())
    assert again is not None
    position = hidden.the_start()
    for _ in range(30):
        legal = moves(learned.family(), position)
        assert {(m.start, m.end, m.becomes) for m in legal} == {
            (m.start, m.end, m.becomes) for m in moves(again.family(), position)}
        if not legal:
            break
        position = after(hidden, position, legal[0]) if legal[0] in moves(hidden, position) else position


def test_the_looks_she_knows_come_back_and_keep_their_names() -> None:
    app = BoardApplication(CHESS, seed=2)
    picture = app.draw()
    lattice = squares_of_one_ground(picture)
    looks = Looks()
    first = looks.read(picture, lattice)
    again = Looks.from_memory(looks.as_memory()).read(picture, lattice)
    assert first.says() == again.says()


def test_rehearsal_keeps_only_weights_that_win() -> None:
    pytest.importorskip("aura_board_search")
    from core.agency.what_a_kind_is_worth import a_first_rehearsal, rehearse

    rehearsal = a_first_rehearsal(NOBODYS_GAME)
    before = dict(rehearsal.weights.worth)
    rehearsal = rehearse(NOBODYS_GAME, NOBODYS_GAME.the_start(), rehearsal, games=16, nodes=400, seed=5)
    assert rehearsal.rounds == 1 and rehearsal.lengths
    if not rehearsal.last.get("kept"):
        assert rehearsal.weights.worth == before
    else:
        assert rehearsal.earned and rehearsal.last["scored"] > 0.5


def test_a_kind_she_has_not_seen_move_is_not_worth_nothing() -> None:
    learned = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    weights = first_weights(learned.family())
    assert all(value > 0 for letter, value in weights.worth.items())
