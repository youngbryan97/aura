"""What one misread look taught her rules is weighed against what came after, not kept for good
(core/agency/how_the_pieces_move.py, core/agency/pieces_by_their_kind.py).

LIVE 25 November 2026, measured on her record of the board she plays Stockfish on: her rules held three kinds that
become others on the far row, from looks read once; a change of places four rows up for her royal kind, from one look;
her straight-runner held to stop five squares along its row, from squares withheld while a piece of hers was kept safe;
and asking her rules for a look they hold no kind of ended a game's account with a KeyError.
"""
from __future__ import annotations

import pytest

from core.agency.how_the_pieces_move import READING, SEEN_TO_BELIEVE, HowThePiecesMove
from core.agency.pieces_by_their_kind import Move
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

pytestmark = pytest.mark.unit

FAR = 7


def _rules() -> HowThePiecesMove:
    return HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)


def test_a_becoming_seen_once_and_contradicted_by_landings_that_stayed_is_not_held() -> None:
    rules = _rules()
    rules._becomes("N", Move((5, 1), (FAR, 2), becomes="R"), FAR)
    assert rules.becomings("N") == ["R"], "seen once and nothing against it: held until something says otherwise"
    for _ in range(2):
        rules._becomes("N", Move((5, 1), (FAR, 2)), FAR)  # landed on that row and stayed a knight
    assert rules.becomings("N") == []
    assert rules.family().kinds["N"].becomes == ()


def test_a_becoming_seen_as_often_as_a_way_is_believed_on_is_held_against_a_few_landings_that_stayed() -> None:
    rules = _rules()
    for _ in range(SEEN_TO_BELIEVE):
        rules._becomes("P", Move((6, 0), (FAR, 0), becomes="Q"), FAR)
    for _ in range(SEEN_TO_BELIEVE + 1):
        rules._becomes("P", Move((6, 1), (FAR, 1)), FAR)
    assert rules.becomings("P") == ["Q"]


def test_a_landing_read_as_staying_and_then_seen_to_have_become_counts_once_as_becoming() -> None:
    rules = _rules()
    rules._becomes("P", Move((6, 0), (FAR, 0), becomes="Q"), FAR)
    rules._becomes("P", Move((6, 1), (FAR, 1)), FAR)  # first read as staying a pawn
    rules.watched_becoming("P", "Q", FAR)  # the second look shows the queen
    assert rules.stayed.get("P", 0) == 0 and rules.became["P"]["Q"] == 2


def test_a_change_of_places_seen_once_and_not_offered_there_with_its_own_piece_standing_is_not_held() -> None:
    rules = _rules()
    alone = from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ -")
    for _ in range(4):
        rules.offered(alone, (0, 4), {(1, 3), (1, 4), (1, 5), (0, 3), (0, 5)}, family=CHESS)
    rules.seen["K"][(1, 0)].swapped.add(2)
    rules._saw_special("K", "swaps")
    held = [s for s in rules.family().kinds["K"].steps if s.may == "swaps"]
    assert held and not held[0].sure, "seen once: held, not relied on"
    with_its_own = from_fen("4k3/8/8/8/8/4B3/8/4K3 w - -")
    rules.offered(with_its_own, (0, 4), {(1, 3), (1, 4), (1, 5), (0, 3), (0, 5)}, family=CHESS)
    assert 2 in rules.seen["K"][(1, 0)].not_swapped
    assert not [s for s in rules.family().kinds["K"].steps if s.may == "swaps"]


def test_a_square_withheld_while_her_royal_piece_is_kept_safe_says_nothing_of_how_far_a_piece_goes() -> None:
    """Her straight-runner stands between her royal piece and theirs on the same file: the game offers it the file and
    withholds its row, because going along the row would leave her royal piece where it can be taken."""
    rules = _rules()
    rules.royal = lambda kind: kind == "K"  # what her rules have learned the game keeps safe
    open_board = from_fen("6k1/8/8/8/8/8/4R3/K7 w - -")
    row_and_file = {(1, c) for c in range(8) if c != 4} | {(r, 4) for r in range(8) if r != 1}
    for _ in range(4):
        rules.offered(open_board, (1, 4), row_and_file, family=CHESS)
    before = {d: set(rules.seen["R"][d].withheld) for d in ((0, 1), (0, -1))}
    pinned = from_fen("4r1k1/8/8/8/8/8/4R3/4K3 w - -")
    rules.offered(pinned, (1, 4), {(r, 4) for r in range(2, 8)}, family=CHESS)
    assert {d: set(rules.seen["R"][d].withheld) for d in ((0, 1), (0, -1))} == before


def test_what_an_earlier_reading_withheld_stands_only_as_far_as_the_answers_kept_bear_it_out() -> None:
    from core.agency.what_the_game_showed import WhatTheGameShowed

    rules = _rules()
    open_row = from_fen("4k3/8/8/8/8/8/8/R3K3 w - -")
    for _ in range(3):
        rules.offered(open_row, (0, 0), {(0, 1), (0, 2), (0, 3)} | {(r, 0) for r in range(1, 8)}, family=CHESS)
    rules.seen["R"][(0, 1)].withheld |= {2, 3}  # kept by an earlier reading, from answers no longer kept
    kept = rules.as_memory()
    kept["read_by"] = 4
    read = HowThePiecesMove.from_memory(kept)
    read.shown = WhatTheGameShowed()
    read.shown.offered(open_row, (0, 0), {(0, 1), (0, 2), (0, 3)} | {(r, 0) for r in range(1, 8)})
    read.withheld_moves_read_again()
    assert 2 not in read.seen["R"][(0, 1)].withheld and 3 not in read.seen["R"][(0, 1)].withheld
    assert read.read_by == READING


def test_a_look_her_rules_hold_no_kind_of_is_a_kind_of_which_nothing_is_known() -> None:
    from core.agency.pieces_by_their_kind import attacked

    unknown = CHESS.kind("6")
    assert unknown.steps == () and not unknown.royal
    board = from_fen("4k3/8/8/8/8/8/8/4K3 w - -")
    board = type(board)(cells={**board.cells, (3, 3): (1, "6")}, turn=0, unmoved=board.unmoved)
    assert not attacked(CHESS, board, (0, 4), 1)
