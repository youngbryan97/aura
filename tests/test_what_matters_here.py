"""What a situation is worth, in the world she is actually in.

`how_good_is_this` scores a situation on a handful of properties and adds
them up with weights. The file that holds them says what they are: "a good
enough one to start from and no more than that". Nothing moved them, so
entering an unfamiliar world she judged it by what mattered in the last one.

Measured on sealed worlds whose only signal is a number nobody explained:
freedom weighted at one and nearness at one, and the result was a policy that
kept its options open forever and never went anywhere — nought arrivals in
twelve, with a perfectly correct model of what every act did. She knew
exactly what she was doing and had no reason to do anything.
"""

from __future__ import annotations

import pytest

from core.agency.what_matters_here import (
    AS_GOOD_AS_THE_GUESS,
    ENOUGH_RUNS,
    WhatMattersHere,
    for_this_world,
    forget_what_mattered,
)

A_GUESS = {"nearness": 1.0, "newness": 1.0, "room": 0.15, "freedom": 1.0}


@pytest.fixture(autouse=True)
def nothing_remembered():
    forget_what_mattered()
    yield
    forget_what_mattered()


def _a_run(mattered: WhatMattersHere, *, freedom: float, nearness: float, well: bool):
    mattered.watched(
        [{"freedom": freedom, "nearness": nearness, "room": 0.5}] * 4,
        went_well=well,
    )


def test_before_the_first_success_she_steers_by_what_is_in_front_of_her():
    """Not by a better guess about what matters. By the refusal to guess.

    The standing weights are not neutral: they weight keeping your options
    open as highly as getting anywhere. In a world whose good places are in a
    corner among places you cannot leave, that is a policy that never
    arrives.
    """

    weights = WhatMattersHere(world="somewhere new").weights(A_GUESS)
    assert weights["nearness"] == 1.0
    assert weights["newness"] == 1.0
    assert weights["freedom"] == 0.0
    assert weights["room"] == 0.0


def test_newness_is_in_that_pair_and_leaving_it_out_cost_everything():
    """Steering by the reading alone, she climbed to a ridge and paced along
    it for the rest of the budget: a step back and a step forward read the
    same and nothing else was allowed to speak."""

    weights = WhatMattersHere(world="somewhere new").weights(A_GUESS)
    assert weights["newness"] > 0.0


def test_one_run_moves_nothing():
    """A weight learned from two runs is two runs written as a number."""

    mattered = for_this_world("a world")
    _a_run(mattered, freedom=0.9, nearness=0.2, well=True)
    _a_run(mattered, freedom=0.2, nearness=0.9, well=False)
    assert mattered.runs == 2
    assert mattered.weights(A_GUESS) == A_GUESS
    assert not mattered.what_it_learned()["moved"]


def test_a_term_that_ran_higher_on_the_runs_that_went_well_is_worth_more():
    mattered = for_this_world("a world")
    for _ in range(ENOUGH_RUNS):
        _a_run(mattered, freedom=0.1, nearness=0.9, well=True)
        _a_run(mattered, freedom=0.9, nearness=0.1, well=False)
    weights = mattered.weights(A_GUESS)
    assert weights["nearness"] > A_GUESS["nearness"]
    assert weights["freedom"] < A_GUESS["freedom"]


def test_a_term_worth_avoiding_can_go_negative():
    """Some things are worth avoiding, and a floor at zero cannot say so."""

    mattered = for_this_world("a world")
    for _ in range(30):
        _a_run(mattered, freedom=0.0, nearness=1.0, well=True)
        _a_run(mattered, freedom=1.0, nearness=0.0, well=False)
    assert mattered.weights({"freedom": 0.2, "nearness": 1.0})["freedom"] < 0.0


def test_it_is_shrunk_towards_the_guess_by_how_much_evidence_there_is():
    """Nothing chooses the rate. It is the count."""

    a_few, many = for_this_world("few"), for_this_world("many")
    for _ in range(ENOUGH_RUNS):
        _a_run(a_few, freedom=0.1, nearness=0.9, well=True)
        _a_run(a_few, freedom=0.9, nearness=0.1, well=False)
    for _ in range(200):
        _a_run(many, freedom=0.1, nearness=0.9, well=True)
        _a_run(many, freedom=0.9, nearness=0.1, well=False)
    assert many.what_it_learned()["shrunk_by"] > a_few.what_it_learned()["shrunk_by"]
    moved_a_little = abs(a_few.weights(A_GUESS)["nearness"] - 1.0)
    moved_a_lot = abs(many.weights(A_GUESS)["nearness"] - 1.0)
    assert moved_a_lot > moved_a_little
    assert AS_GOOD_AS_THE_GUESS > 0


def test_all_wins_or_all_losses_leaves_the_guess_alone():
    """A difference of means needs both means."""

    mattered = for_this_world("a world")
    for _ in range(20):
        _a_run(mattered, freedom=0.5, nearness=0.5, well=True)
    assert mattered.weights(A_GUESS) == A_GUESS


def test_it_reads_the_run_rather_than_the_state_it_ended_in():
    """The last state of a run that went badly is the state it went badly in,
    and every term reads the same there whatever the run was like."""

    mattered = for_this_world("a world")
    mattered.watched(
        [{"nearness": 0.9}, {"nearness": 0.9}, {"nearness": 0.0}], went_well=True
    )
    assert mattered.well["nearness"] == pytest.approx(0.6)


def test_each_world_is_learned_about_separately():
    here, there = for_this_world("here"), for_this_world("there")
    _a_run(here, freedom=0.1, nearness=0.9, well=True)
    assert there.runs == 0
    assert for_this_world("here") is here
