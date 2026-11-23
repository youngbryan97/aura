"""Where a situation comes back, the thinking it needs is done before it comes, with time no game gives.

Every game at a board begins at the same place, so its first decisions come back in every game, with the answers the
other side tends to give. On the opening positions of her live games of 23 November 2026, a 30 s look chose moves
costing 202 hundredths of a piece by Stockfish, where her 2.5 s game pace chose moves costing 346. Between games she
works out the situations she is likeliest to meet; in a game she answers those from what she worked out.
"""
from __future__ import annotations

import core.agency.practising_between_games as practice
from core.agency.pieces_by_their_kind import after, moves
from core.agency.playing_pieces_by_kind import _a_place, _Game, _named, _worked_out_ahead
from core.agency.searching_by_kind import Found, first_weights
from core.agency.thinking_before_it_comes import Prepared, under
from core.agency.thinking_on_their_time import WhatTheyAnsweredWhere
from tests.fixtures.a_game_of_pieces_to_learn import CHESS


def _a_quick_look(searched: list) -> object:
    def search(family, position, *, seconds, weights=None, earlier=None, threads=1, **_):
        searched.append(_a_place(position))
        legal = sorted(moves(family, position), key=_named)
        mine = legal[0]
        replies = sorted(moves(family, after(family, position, mine)), key=_named)
        return Found(mine, 12, 9, 1000, [mine, replies[0]] if replies else [mine])
    return search


def test_she_walks_her_answers_and_theirs_from_the_start() -> None:
    start, weights = CHESS.the_start(), first_weights(CHESS)
    searched: list = []
    prepared = Prepared()
    # They have answered her first move two ways before, one more often.
    first = sorted(moves(CHESS, start), key=_named)[0]
    there = after(CHESS, start, first)
    replies = sorted(moves(CHESS, there), key=_named)
    seen = WhatTheyAnsweredWhere()
    for _ in range(3):
        seen.they_answered(_a_place(there), _named(replies[5]))
    seen.they_answered(_a_place(there), _named(replies[7]))
    done = prepared.prepare(CHESS, start, weights, they_answered=seen, positions=3, plies=6,
                            search=_a_quick_look(searched))
    assert done == 3 and len(searched) == 3
    # Her first move, then her move after each of the two answers they have given, most often first.
    assert searched[0] == _a_place(start)
    assert searched[1] == _a_place(after(CHESS, there, replies[5]))
    assert searched[2] == _a_place(after(CHESS, there, replies[7]))
    # Worked out once: a second walk finds them done and goes further.
    more: list = []
    prepared.prepare(CHESS, start, weights, they_answered=seen, positions=1, plies=6, search=_a_quick_look(more))
    assert more and more[0] not in searched


def test_where_they_have_not_answered_she_follows_the_answer_she_expects() -> None:
    start, weights = CHESS.the_start(), first_weights(CHESS)
    searched: list = []
    Prepared().prepare(CHESS, start, weights, positions=2, plies=4, search=_a_quick_look(searched))
    first = sorted(moves(CHESS, start), key=_named)[0]
    there = after(CHESS, start, first)
    expected = sorted(moves(CHESS, there), key=_named)[0]
    assert searched == [_a_place(start), _a_place(after(CHESS, there, expected))]


def test_an_answer_counts_only_under_the_rules_and_judgement_it_was_worked_out_under() -> None:
    start, weights = CHESS.the_start(), first_weights(CHESS)
    prepared = Prepared()
    prepared.prepare(CHESS, start, weights, positions=1, plies=2, search=_a_quick_look([]))
    assert prepared.answer(_a_place(start), under(CHESS, weights)) is not None
    other = first_weights(CHESS)
    other.tempo += 7
    assert prepared.answer(_a_place(start), under(CHESS, other)) is None
    again = Prepared.from_bytes(prepared.as_bytes())
    assert again.answer(_a_place(start), under(CHESS, weights)) == prepared.answer(_a_place(start),
                                                                                  under(CHESS, weights))


def test_in_a_game_a_situation_worked_out_ahead_is_answered_from_it() -> None:
    from core.agency.how_the_pieces_move import HowThePiecesMove

    start, weights = CHESS.the_start(), first_weights(CHESS)
    prepared = Prepared()
    prepared.prepare(CHESS, start, weights, positions=1, plies=2, search=_a_quick_look([]))
    game = _Game(learned=HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, dict(start.cells)),
                 believed=start)
    found = _worked_out_ahead(prepared, game, CHESS, weights, moves(CHESS, start))
    assert found is not None and found.by == "worked out ahead" and found.depth == 9
    assert _named(found.move) == _named(sorted(moves(CHESS, start), key=_named)[0])
    assert _worked_out_ahead(None, game, CHESS, weights, moves(CHESS, start)) is None


def test_practice_takes_turns_between_rehearsing_and_preparing(monkeypatch) -> None:
    ran: list[str] = []
    monkeypatch.setattr(practice, "_NEXT", {"prepares": False})
    monkeypatch.setattr(practice, "_one_rehearsal", lambda name, written: ran.append("rehearsed") or {})
    monkeypatch.setattr(practice, "_prepare_some", lambda name: ran.append("prepared") or 2)
    for _ in range(4):
        practice._one_round("A Board", (0, 0))
    assert ran == ["rehearsed", "prepared", "rehearsed", "prepared"]
    # Where nothing is left to work out, the round rehearses instead.
    ran.clear()
    monkeypatch.setattr(practice, "_prepare_some", lambda name: ran.append("nothing to prepare") and 0)
    practice._one_round("A Board", (0, 0))
    practice._one_round("A Board", (0, 0))
    assert ran == ["rehearsed", "nothing to prepare", "rehearsed"]
