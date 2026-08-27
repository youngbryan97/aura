"""A future worked out as if the world sits still is a future that cannot happen.

She works out what her own acts do and then plans as though that were the
whole story. A board deals a tile, a page gains a row, a queue takes another
customer. She already tolerates arrivals when scoring a rule, because a dealt
tile is not a rule's mistake — and tolerating a thing is not the same as
knowing it. The information was there every move and thrown away every move:
the difference between what a rule said would happen and what she actually saw
IS what the world did.

What she does with it is stop taking the best continuation at every level. Her
own move is hers to pick. What the world does is not, so it is averaged over
rather than chosen, which is the difference between a plan and a wish.
"""

from __future__ import annotations

import pytest

from core.agency.looking_ahead import look_ahead
from core.perception.how_it_moves import HowItMoves, shifted_and_combined
from core.perception.what_is_there import Arrangement, Cell, arranged
from core.perception.what_the_world_does import ENOUGH_TO_EXPECT, WhatTheWorldDoes

ROOM = [["2", "4", "", ""], ["", "", "", ""], ["", "", "", ""], ["", "", "8", ""]]


def board(rows=ROOM) -> Arrangement:
    return arranged([
        (0.20 + r * 0.15, 0.20 + c * 0.15, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ])


def with_one_more(state: Arrangement, row: int, column: int, says: str) -> Arrangement:
    return Arrangement(
        state.rows, state.columns,
        state.cells + (Cell(row, column, says, (0.0, 0.0)),),
        state.down_at, state.across_at,
    )


def watched(times: int = ENOUGH_TO_EXPECT, says: str = "2") -> WhatTheWorldDoes:
    world = WhatTheWorldDoes()
    here = board()
    for turn in range(times):
        world.watched(here, with_one_more(here, 1, turn % 4, says))
    return world


# ── the difference is the world's doing ──────────────────────────────────

def test_what_a_rule_did_not_claim_is_what_the_world_put_there():
    assert watched().what_arrives() == (("2", 1.0),)


def test_two_things_arrive_in_the_share_they_arrive_in():
    world = WhatTheWorldDoes()
    here = board()
    for turn in range(8):
        world.watched(here, with_one_more(here, 1, turn % 4, "4" if turn < 2 else "2"))
    arrivals = dict(world.what_arrives())
    assert arrivals["2"] == pytest.approx(0.75)
    assert arrivals["4"] == pytest.approx(0.25)


def test_an_act_the_world_did_not_answer_still_counts():
    world = WhatTheWorldDoes()
    here = board()
    for _ in range(4):
        world.watched(here, with_one_more(here, 1, 1, "2"))
    for _ in range(4):
        world.watched(here, here)
    assert world.how_often() == pytest.approx(0.5)


def test_a_rule_that_said_nothing_teaches_nothing():
    world = WhatTheWorldDoes()
    world.watched(None, board())
    assert world.acts == 0


# ── it has to be watched before it is worth planning around ──────────────

def test_one_look_is_not_a_model_of_anything():
    assert watched(times=1).worth_expecting() is False
    assert watched(times=1).might_do(board()) == ()


def test_and_she_says_so_plainly():
    assert "not worked out yet" in watched(times=1).says()


def test_enough_looks_and_it_is():
    assert watched().worth_expecting() is True
    assert "after 100% of my acts" in watched().says()


# ── the ways it might go ─────────────────────────────────────────────────

def test_the_ways_it_might_go_are_weighted_and_sum_to_one():
    ways = watched().might_do(board())
    assert ways
    assert sum(share for _way, share in ways) == pytest.approx(1.0)


def test_each_way_puts_something_where_there_was_room():
    here = board()
    for way, _share in watched().might_do(here):
        assert way.occupied() >= here.occupied()


def test_a_thing_with_no_room_left_can_answer_in_no_way_at_all():
    full = board([["2", "4", "8", "16"]] * 4)
    assert watched().might_do(full) == ()


def test_the_ways_are_spread_across_the_room_rather_than_bunched():
    """A board that fills from one end would otherwise only ever be sampled
    in one corner of itself."""
    ways = watched().might_do(board())
    landed = {
        (cell.row, cell.column)
        for way, _share in ways
        for cell in way.cells
    }
    assert len(landed) > board().occupied()


# ── what it is for ───────────────────────────────────────────────────────

def test_looking_ahead_works_without_it():
    knows = HowItMoves()
    here = board()
    for turn in range(8):
        move = ("left", "right", "up", "down")[turn % 4]
        knows.watched(here, move, shifted_and_combined(here, move))
    assert look_ahead(knows, here, ["left", "right", "up", "down"], toward="2048")


def test_and_averages_over_the_world_when_she_has_worked_it_out():
    knows = HowItMoves()
    here = board()
    for turn in range(8):
        move = ("left", "right", "up", "down")[turn % 4]
        knows.watched(here, move, shifted_and_combined(here, move))
    moves = ["left", "right", "up", "down"]
    hoping = look_ahead(knows, here, moves, toward="2048")
    planning = look_ahead(knows, here, moves, toward="2048", world=watched())
    assert planning
    # The point is that it is a DIFFERENT number. Hoping takes the best
    # continuation at every level; planning takes the average over what the
    # world might do. Which is larger depends on whether what the world does
    # helps — here a dealt tile adds something, so it can come out above.
    # What matters is that the world is in the arithmetic at all.
    assert {move: round(score, 6) for move, (score, _w) in planning.items()} != {
        move: round(score, 6) for move, (score, _w) in hoping.items()
    }


def test_a_world_model_it_cannot_read_is_ignored_rather_than_fatal():
    knows = HowItMoves()
    here = board()
    for turn in range(8):
        move = ("left", "right", "up", "down")[turn % 4]
        knows.watched(here, move, shifted_and_combined(here, move))
    assert look_ahead(knows, here, ["left", "right"], toward="2048", world="not a model")


# ── keeping it ───────────────────────────────────────────────────────────

def test_what_the_world_did_survives_the_process():
    back = WhatTheWorldDoes.from_memory(watched().as_memory())
    assert back.what_arrives() == (("2", 1.0),)


def test_but_comes_back_light_enough_to_be_overturned():
    back = WhatTheWorldDoes.from_memory(watched(times=20).as_memory(), trust=0.5)
    assert back.acts == 10


def test_nothing_carried_back_claims_more_than_it_was():
    back = WhatTheWorldDoes.from_memory(watched(times=20).as_memory(), trust=0.5)
    assert back.acts_with_arrivals <= back.acts


def test_rubbish_is_not_a_memory():
    assert WhatTheWorldDoes.from_memory("not a memory").acts == 0
    assert WhatTheWorldDoes.from_memory({"arrives": "bad"}).arrives == {}
