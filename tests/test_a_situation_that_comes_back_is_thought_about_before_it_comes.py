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


def test_practice_goes_where_it_has_paid_without_leaving_the_rest_untried(monkeypatch) -> None:
    """Each kind of round is tried first once; then a round that changed what she will do (situations worked out
    ahead) is chosen over one that did not (a rehearsal that kept nothing), and the other is still tried now and then
    (core/cognition/where_practice_pays.py)."""
    ran: list[str] = []
    monkeypatch.setattr(practice, "_PAYS", {})
    monkeypatch.setattr(practice, "_ROUNDS", ("rehearses", "prepares"))
    monkeypatch.setattr(practice, "_one_rehearsal", lambda name, written: ran.append("rehearsed") or {"kept": False})
    monkeypatch.setattr(practice, "_prepare_some", lambda name: ran.append("prepared") or 2)
    for _ in range(16):
        practice._one_round("A Board", (0, 0))
    assert ran[:2] == ["rehearsed", "prepared"]
    assert ran.count("prepared") > 2 * ran.count("rehearsed") and ran.count("rehearsed") >= 2
    # Where nothing is left to work out ahead, the round goes to the next kind.
    ran.clear()
    monkeypatch.setattr(practice, "_prepare_some", lambda name: ran.append("nothing to prepare") and 0)
    practice._one_round("A Board", (0, 0))
    assert ran == ["nothing to prepare", "rehearsed"]


def test_a_situation_worked_out_is_not_looked_at_again_for_longer() -> None:
    # Longer looks chose worse opening moves than 30 s ones, by Stockfish: what she has worked out stays worked out.
    start, weights = CHESS.the_start(), first_weights(CHESS)
    looks: list = []

    def search(family, position, *, seconds, weights=None, earlier=None, threads=1, **_):
        looks.append((_a_place(position), seconds))
        return _a_quick_look([])(family, position, seconds=seconds)

    prepared = Prepared()
    prepared.prepare(CHESS, start, weights, positions=3, plies=2, search=search)
    assert [look for look in looks if look[0] == _a_place(start)] == [(_a_place(start), 30.0)]
    assert {seconds for _place, seconds in looks} == {30.0}


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


def _an_honest_look(searched: list) -> object:
    def search(family, position, *, seconds, weights=None, earlier=None, threads=1, set_aside=(), **_):
        searched.append(_a_place(position))
        legal = [m for m in sorted(moves(family, position), key=_named) if m not in set_aside]
        mine = legal[0]
        replies = sorted(moves(family, after(family, position, mine)), key=_named)
        return Found(mine, 12, 9, 1000, [mine, replies[0]] if replies else [mine])
    return search


def test_with_every_likely_situation_worked_out_she_works_out_what_else_they_may_answer() -> None:
    import json

    start, weights = CHESS.the_start(), first_weights(CHESS)
    searched: list = []
    prepared = Prepared()
    done = prepared.prepare(CHESS, start, weights, positions=4, plies=3, search=_an_honest_look(searched))
    first = sorted(moves(CHESS, start), key=_named)[0]
    there = after(CHESS, start, first)
    replies = sorted(moves(CHESS, there), key=_named)
    # Her first move and her move after the answer she expects; then, with nothing likelier left, what else they
    # might answer where they have given nothing she has seen, and her move after that.
    assert done == 4
    assert searched == [_a_place(start), _a_place(after(CHESS, there, replies[0])), _a_place(there),
                        _a_place(after(CHESS, there, replies[1]))]
    again = Prepared.from_bytes(prepared.as_bytes())
    assert again.answers == prepared.answers and again.otherwise == prepared.otherwise
    # Kept before what else they may answer was kept: the answers alone, still read.
    assert Prepared.from_bytes(json.dumps(prepared.answers).encode()).answers == prepared.answers


def test_where_they_have_answered_she_does_not_look_for_what_else_they_may() -> None:
    start, weights = CHESS.the_start(), first_weights(CHESS)
    first = sorted(moves(CHESS, start), key=_named)[0]
    there = after(CHESS, start, first)
    replies = sorted(moves(CHESS, there), key=_named)
    seen = WhatTheyAnsweredWhere()
    seen.they_answered(_a_place(there), _named(replies[3]))
    searched: list = []
    Prepared().prepare(CHESS, start, weights, they_answered=seen, positions=4, plies=3,
                       search=_an_honest_look(searched))
    assert _a_place(there) not in searched


def test_a_move_her_rules_allow_on_too_little_is_set_aside_and_an_answer_that_is_one_worked_out_again() -> None:
    start, weights = CHESS.the_start(), first_weights(CHESS)
    first = sorted(moves(CHESS, start), key=_named)[0]
    prepared = Prepared()
    prepared.prepare(CHESS, start, weights, positions=1, plies=2, search=_an_honest_look([]))
    assert prepared.answer(_a_place(start), among(moves(CHESS, start)))["move"] == _named(first)
    # Her rules came to hold that move on too little: the answer is worked out again without it.
    searched: list = []
    prepared.prepare(CHESS, start, weights, positions=1, plies=2, search=_an_honest_look(searched),
                     doubtful=lambda position, move: move == first)
    assert searched == [_a_place(start)]
    assert prepared.answer(_a_place(start), among(moves(CHESS, start)))["move"] != _named(first)


def test_an_answer_worked_out_under_the_rules_she_plays_by_is_played_and_one_under_other_rules_is_a_hint() -> None:
    from dataclasses import replace

    from core.agency.how_the_pieces_move import HowThePiecesMove

    start, weights = CHESS.the_start(), first_weights(CHESS)
    prepared = Prepared()
    prepared.prepare(CHESS, start, weights, positions=1, plies=2, search=_a_quick_look([]))
    game = _Game(learned=HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, dict(start.cells)),
                 believed=start)
    assert _worked_out_ahead(prepared, game, CHESS, weights, moves(CHESS, start)).by == "worked out ahead"
    # Her rules grew elsewhere, and the start offers the same choices: what she worked out there is a hint.
    grown = replace(CHESS, count_to_draw=CHESS.count_to_draw + 1)
    hint = _worked_out_ahead(prepared, game, grown, weights, moves(grown, start))
    assert hint is not None and hint.by == "worked out ahead, to check"
    held = prepared.answer(_a_place(start), among(moves(CHESS, start)))
    assert held["line"] and held["line"][0] == held["move"]


def test_a_hint_stands_unless_a_look_now_finds_it_worse_than_its_own_judgement_moved() -> None:
    from core.agency.playing_pieces_by_kind import _checked

    legal = sorted(moves(CHESS, CHESS.the_start()), key=_named)
    hint = Found(legal[0], 30, 16, 0, [legal[0]], by="worked out ahead, to check")
    now = Found(legal[1], 40, 9, 1000, [legal[1]], unsteady=15)
    # Searched alone, the hint comes within what the look's own judgement moved: it stands.
    kept = _checked(hint, now, legal, lambda aside: Found(legal[0], 30, 10, 500, [legal[0]]))
    assert kept.move == legal[0] and kept.by == "worked out ahead, checked"
    # Its worth and depth are the look's made now, not the ones worked out under rules since changed.
    assert (kept.score, kept.depth) == (30, 10)
    # Further off than that: the look made now is played.
    assert _checked(hint, now, legal, lambda aside: Found(legal[0], 10, 10, 500, [legal[0]])).move == legal[1]
    # A look that finds the same move needs no second look.
    agreeing = Found(legal[0], 25, 9, 1000, [legal[0]])
    checked = _checked(hint, agreeing, legal, lambda aside: 1 / 0)
    assert checked.by == "worked out ahead, checked" and (checked.score, checked.depth) == (25, 9)


def test_an_answer_stands_at_another_count_toward_a_draw_only_where_neither_could_reach_the_limit() -> None:
    from dataclasses import replace

    start = CHESS.the_start()
    prepared = Prepared()
    prepared.prepare(CHESS, start, first_weights(CHESS), positions=1, plies=2, search=_a_quick_look([]))
    place, choices = _a_place(start), among(moves(CHESS, start))
    limit = CHESS.count_to_draw
    assert limit and prepared.answer(place, choices, count=0, limit=limit) is not None
    # Alike but for the count, both far from the limit: the look could not have reached it, and the answer stands.
    assert prepared.answer(place, choices, count=3, limit=limit) is not None
    # Near the limit, the count can change what happens next: it is another situation.
    assert prepared.answer(place, choices, count=limit - 1, limit=limit) is None
    # Worked out near the limit, it stands at that count alone.
    near = replace(start, count=limit - 1)
    other = Prepared()
    other.prepare(CHESS, near, first_weights(CHESS), positions=1, plies=2, search=_a_quick_look([]))
    assert other.answer(_a_place(near), choices, count=limit - 1, limit=limit) is not None
    assert other.answer(_a_place(near), choices, count=limit - 2, limit=limit) is None


def test_an_answer_worked_out_under_another_judgement_or_one_that_repeats_the_game_is_a_hint() -> None:
    """Suggestions of 23 November 2026: a certificate named the rules a look was made under and not the judgement, and
    a key leaves out what a repeat can end."""
    from dataclasses import replace

    from core.agency.how_the_pieces_move import HowThePiecesMove

    start, weights = CHESS.the_start(), first_weights(CHESS)
    prepared = Prepared()
    prepared.prepare(CHESS, start, weights, positions=1, plies=2, search=_a_quick_look([]))
    game = _Game(learned=HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, dict(start.cells)),
                 believed=start)
    assert _worked_out_ahead(prepared, game, CHESS, weights, moves(CHESS, start)).by == "worked out ahead"
    other = replace(weights, tempo=weights.tempo + 7)
    assert _worked_out_ahead(prepared, game, CHESS, other, moves(CHESS, start)).by == "worked out ahead, to check"
    held = prepared.answer(_a_place(start), among(moves(CHESS, start)))
    move = next(m for m in moves(CHESS, start) if _named(m) == held["move"])
    # The situation her move would leave stood earlier in this game.
    came_back = replace(start, seen=(after(CHESS, start, move).key(),))
    game.believed = came_back
    assert _worked_out_ahead(prepared, game, CHESS, weights, moves(CHESS, came_back)).by == \
        "worked out ahead, to check"


def test_among_moves_her_looks_cannot_tell_apart_the_one_leading_where_she_has_worked_out_is_kept() -> None:
    """Item 6, built and measured and not called by her practice: asked for, a second look finds the best move
    besides, and where the two are alike the one leading to a situation she has worked out is kept."""
    start, weights = CHESS.the_start(), first_weights(CHESS)
    legal = sorted(moves(CHESS, start), key=_named)
    first, second = legal[0], legal[1]

    def search(family, position, *, seconds, weights=None, earlier=None, threads=1, set_aside=(), **_):
        mine = second if first in set_aside else first
        replies = sorted(moves(family, after(family, position, mine)), key=_named)
        return Found(mine, 10, 9, 1000, [mine, replies[0]], unsteady=30)

    prepared = Prepared()
    there = after(CHESS, after(CHESS, start, second), sorted(moves(CHESS, after(CHESS, start, second)),
                                                               key=_named)[0])
    from core.agency.thinking_before_it_comes import _key

    prepared.answers[_key(_a_place(there))] = {"move": "x", "among": among(moves(CHESS, there)), "depth": 1}
    prepared.prepare(CHESS, start, weights, positions=1, plies=1, search=search, among_equals=True)
    assert prepared.answer(_a_place(start), among(legal))["move"] == _named(second)
