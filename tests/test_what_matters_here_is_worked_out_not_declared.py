"""What each thing about a situation is worth is a fact about the world.

A situation has several things to like about it: how near it is to what she
was asked for, whether her line still holds, how much room is left, how much
of it runs in order. What each is worth was set by three numbers somebody
picked, once, for every world she will ever be in.

They cannot all be right. Room matters enormously in a world that fills up and
not at all in one that does not. Order is the whole game where things combine
by neighbour and meaningless where they do not. No amount of care in choosing
fixes that, because the right answer is about the world rather than about her.

So she works it out the way she works everything else out: each move gives a
situation, and whether it left her better off. Where a thing was higher in the
moves that went well than in the ones that went badly, it is worth more than
she thought.
"""

from __future__ import annotations

import pytest

from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY, how_good, terms
from core.agency.what_makes_it_good_here import (
    ENOUGH_TO_REWEIGH,
    LEAST_A_THING_CAN_BE_WORTH,
    MOST_A_THING_CAN_BE_WORTH,
    WhatMakesItGoodHere,
)
from core.perception.what_is_there import arranged


def watch(matters: WhatMakesItGoodHere, *, moves: int, room_decides: bool) -> None:
    """Moves where one thing, and only one, tells the good from the bad."""
    for turn in range(moves):
        went_well = turn % 3 != 0
        matters.watched(
            {
                "nearness": 0.5,
                "line": 0.0,
                "room": (0.9 if went_well else 0.1) if room_decides else 0.5,
                "order": 0.5,
            },
            went_well,
        )


# ── the terms exist apart from the sum of them ───────────────────────────

def test_a_situation_can_be_asked_what_there_is_to_like_about_it():
    board = arranged([(0.2, 0.2, "128"), (0.2, 0.35, "64"), (0.35, 0.2, "4")])
    each = terms(board, toward="2048")
    assert set(each) == set(AS_GOOD_A_GUESS_AS_ANY)
    assert each["nearness"] > 0.0


def test_and_the_sum_is_the_terms_under_the_standing_weights():
    board = arranged([(0.2, 0.2, "128"), (0.2, 0.35, "64")])
    each = terms(board, toward="2048")
    by_hand = sum(each[name] * AS_GOOD_A_GUESS_AS_ANY[name] for name in each)
    assert how_good(board, toward="2048") == pytest.approx(by_hand)


def test_weighing_it_differently_scores_it_differently():
    board = arranged([(0.2, 0.2, "128"), (0.2, 0.35, "64")])
    standing = how_good(board, toward="2048")
    otherwise = how_good(board, toward="2048", weights={"nearness": 0.1, "room": 2.0})
    assert standing != otherwise


# ── working out what matters ─────────────────────────────────────────────

def test_before_she_has_watched_enough_the_weights_are_not_hers():
    matters = WhatMakesItGoodHere()
    watch(matters, moves=ENOUGH_TO_REWEIGH - 1, room_decides=True)
    assert matters.weights() is None
    assert "not worked out yet" in matters.says()


def test_the_thing_that_told_the_good_moves_from_the_bad_is_worth_more():
    matters = WhatMakesItGoodHere()
    watch(matters, moves=ENOUGH_TO_REWEIGH + 20, room_decides=True)
    worth = matters.weights()
    assert worth is not None
    assert worth["room"] > AS_GOOD_A_GUESS_AS_ANY["room"]


def test_and_the_things_that_told_her_nothing_are_left_alone():
    matters = WhatMakesItGoodHere()
    watch(matters, moves=ENOUGH_TO_REWEIGH + 20, room_decides=True)
    worth = matters.weights()
    for name in ("nearness", "line", "order"):
        assert worth[name] == pytest.approx(AS_GOOD_A_GUESS_AS_ANY[name])


def test_where_nothing_tells_them_apart_nothing_moves():
    matters = WhatMakesItGoodHere()
    watch(matters, moves=ENOUGH_TO_REWEIGH + 20, room_decides=False)
    assert matters.weights() == pytest.approx(AS_GOOD_A_GUESS_AS_ANY)


def test_she_can_say_what_matters_here():
    matters = WhatMakesItGoodHere()
    watch(matters, moves=ENOUGH_TO_REWEIGH + 20, room_decides=True)
    assert "what matters here" in matters.says()
    assert "room" in matters.says()


# ── and no weight runs away ──────────────────────────────────────────────

def test_a_thing_never_becomes_worthless():
    """Pinned at zero it could never earn its way back, and a world changes."""
    matters = WhatMakesItGoodHere()
    for _ in range(4000):
        matters.watched({"nearness": 0.0, "line": 0.0, "room": 0.0, "order": 1.0}, False)
        matters.watched({"nearness": 1.0, "line": 1.0, "room": 1.0, "order": 0.0}, True)
    assert all(v >= LEAST_A_THING_CAN_BE_WORTH for v in matters.worth.values())


def test_nor_swamps_everything_else():
    matters = WhatMakesItGoodHere()
    for _ in range(4000):
        matters.watched({"nearness": 1.0, "line": 0.0, "room": 0.0, "order": 0.0}, True)
        matters.watched({"nearness": 0.0, "line": 0.0, "room": 0.0, "order": 0.0}, False)
    assert all(v <= MOST_A_THING_CAN_BE_WORTH for v in matters.worth.values())


def test_a_move_with_nothing_in_it_teaches_nothing():
    matters = WhatMakesItGoodHere()
    matters.watched({}, True)
    assert matters.seen == 0


# ── keeping it ───────────────────────────────────────────────────────────

def test_what_mattered_here_survives_the_process():
    matters = WhatMakesItGoodHere()
    watch(matters, moves=ENOUGH_TO_REWEIGH + 20, room_decides=True)
    back = WhatMakesItGoodHere.from_memory(matters.as_memory())
    assert back.worth["room"] == pytest.approx(matters.worth["room"])


def test_but_comes_back_part_of_the_way_toward_the_standing_guess():
    """A weighting from yesterday is a starting point, not a verdict."""
    matters = WhatMakesItGoodHere()
    watch(matters, moves=ENOUGH_TO_REWEIGH + 60, room_decides=True)
    back = WhatMakesItGoodHere.from_memory(matters.as_memory(), trust=0.5)
    assert AS_GOOD_A_GUESS_AS_ANY["room"] < back.worth["room"] < matters.worth["room"]


def test_rubbish_is_not_a_memory():
    assert WhatMakesItGoodHere.from_memory("not a memory").weights() is None
    assert WhatMakesItGoodHere.from_memory({"worth": "bad"}).worth == AS_GOOD_A_GUESS_AS_ANY
