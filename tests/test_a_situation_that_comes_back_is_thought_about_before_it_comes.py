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
from core.agency.thinking_before_it_comes import Prepared, among
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
    done = prepared.prepare(CHESS, start, weights, they_answered=seen, positions=4, plies=6,
                            search=_a_quick_look(searched))
    assert done == 4 and len(searched) == 4
    # Her first move; then her move after the answer they have given most; then the next situation on that line, met
    # in as many games (three in four), before the answer they gave once; then that one.
    assert searched[0] == _a_place(start)
    main = after(CHESS, there, replies[5])
    assert searched[1] == _a_place(main)
    mine = sorted(moves(CHESS, main), key=_named)[0]
    expected = sorted(moves(CHESS, after(CHESS, main, mine)), key=_named)[0]
    assert searched[2] == _a_place(after(CHESS, after(CHESS, main, mine), expected))
    assert len(set(searched)) == 4, "each look is at a situation of its own"
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


def test_an_answer_counts_while_the_choices_it_was_made_among_stand() -> None:
    from dataclasses import replace

    start, weights = CHESS.the_start(), first_weights(CHESS)
    prepared = Prepared()
    prepared.prepare(CHESS, start, weights, positions=1, plies=2, search=_a_quick_look([]))
    choices = moves(CHESS, start)
    assert prepared.answer(_a_place(start), among(choices)) is not None
    # Her rules changed and offer her the same choices here: what she worked out among them still stands.
    other_rules = replace(CHESS, count_to_draw=CHESS.count_to_draw + 1)
    assert prepared.answer(_a_place(start), among(moves(other_rules, start))) is not None
    # A choice her rules no longer give her, or one more: the look never weighed these choices.
    assert prepared.answer(_a_place(start), among(choices[1:])) is None
    again = Prepared.from_bytes(prepared.as_bytes())
    assert again.answer(_a_place(start), among(choices)) == prepared.answer(_a_place(start), among(choices))


def test_a_walk_goes_on_through_answers_made_among_the_same_choices_under_other_rules() -> None:
    from dataclasses import replace

    start, weights = CHESS.the_start(), first_weights(CHESS)
    prepared = Prepared()
    prepared.prepare(CHESS, start, weights, positions=1, plies=4, search=_a_quick_look([]))
    # Her rules grew in a way that leaves the start's choices as they were: the walk does not work the start out
    # again, and goes on to the next situation.
    searched: list = []
    other_rules = replace(CHESS, count_to_draw=CHESS.count_to_draw + 1)
    prepared.prepare(other_rules, start, weights, positions=1, plies=4, search=_a_quick_look(searched))
    assert searched and searched[0] != _a_place(start)


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


def test_a_situation_worked_out_is_not_looked_at_again_for_longer() -> None:
    # Longer looks chose worse opening moves than 30 s ones, by Stockfish: what she has worked out stays worked out.
    start, weights = CHESS.the_start(), first_weights(CHESS)
    looks: list = []

    def search(family, position, *, seconds, weights=None, earlier=None, threads=1, **_):
        looks.append((_a_place(position), seconds))
        return _a_quick_look([])(family, position, seconds=seconds)

    prepared = Prepared()
    done = prepared.prepare(CHESS, start, weights, positions=3, plies=2, search=search)
    assert done == 1 and looks == [(_a_place(start), 30.0)]


def test_a_move_a_lost_game_went_through_is_worked_out_again_without_it() -> None:
    from core.cognition.the_furthest_she_has_got import WhatFailedWhere

    start, weights = CHESS.the_start(), first_weights(CHESS)
    prepared = Prepared()
    prepared.prepare(CHESS, start, weights, positions=1, plies=2, search=_a_quick_look([]))
    first = prepared.answer(_a_place(start), among(moves(CHESS, start)))["move"]
    lost = WhatFailedWhere()
    lost.failed_at(_a_place(start), first)
    asked: list = []

    def search(family, position, *, seconds, weights=None, earlier=None, threads=1, set_aside=(), **_):
        asked.append([_named(m) for m in set_aside])
        legal = [m for m in sorted(moves(family, position), key=_named) if m not in set_aside]
        return Found(legal[0], 0, 9, 1, [legal[0]])

    prepared.prepare(CHESS, start, weights, went_wrong=lost, positions=1, plies=2, search=search)
    assert asked and first in asked[0]
    assert prepared.answer(_a_place(start), among(moves(CHESS, start)))["move"] != first
