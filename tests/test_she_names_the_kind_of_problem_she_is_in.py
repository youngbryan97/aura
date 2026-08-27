"""Recognising the shape of a problem is the general part.

Reasoning through every low-level step is not what makes a mind general — a
general mind recognises what it is facing and recruits something suited to it.
Asked for the shortest route through a hundred thousand cities, contemplating
routes one by one in language is the stupid reading of generality; naming it a
graph problem is the intelligent one.

Everything read here is something she established rather than something
declared: which of her acts do anything, what they do, whether the world adds
things of its own, and whether what she wants can be counted.
"""

from __future__ import annotations

import random

import pytest

from core.agency.what_kind_of_problem import (
    FEW_ENOUGH_ACTS,
    SMALL_ENOUGH_TO_SEARCH,
    recognise,
)
from core.perception.how_it_moves import HowItMoves, shifted_and_combined
from core.perception.what_is_there import Arrangement, Cell, arranged

MOVES = ["up", "down", "left", "right"]


def board(rows):
    return arranged([
        (0.2 + r * 0.15, 0.2 + c * 0.15, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ])


START = board([["2", "2", "4", ""], ["", "4", "", "8"], ["8", "", "", "8"], ["", "", "2", "2"]])


def spawn(state, rng):
    free = [
        (r, c) for r in range(state.rows) for c in range(state.columns) if state.at(r, c) is None
    ]
    if not free:
        return state
    r, c = rng.choice(free)
    return Arrangement(
        state.rows, state.columns, state.cells + (Cell(r, c, "2", (0.0, 0.0)),),
        state.down_at, state.across_at,
    )


def watched(*, adds_things: bool, moves: int = 20, seed: int = 4):
    rng = random.Random(seed)
    knows = HowItMoves()
    state = START
    for _ in range(moves):
        move = rng.choice(MOVES)
        after = shifted_and_combined(state, move)
        if adds_things:
            after = spawn(after, rng)
        knows.watched(state, move, after)
        state = after
    return knows, state


# ── what she has to have worked out first ────────────────────────────────

def test_before_she_knows_what_her_acts_do_the_answer_is_to_do_one():
    suits = recognise(acts=MOVES, knows_how_it_moves=HowItMoves(), state=START, toward="256")
    assert not suits.shape.transition_known
    assert "act and look" in suits.shape.named()
    assert suits.have_it


def test_with_nothing_to_do_there_is_nothing_to_choose_between():
    suits = recognise(acts=[], knows_how_it_moves=HowItMoves(), state=START, toward="256")
    assert suits.process == "nothing"


# ── the shape, from what she established ─────────────────────────────────

def test_she_finds_out_whether_the_world_moves_too():
    noisy, _ = watched(adds_things=True)
    quiet, _ = watched(adds_things=False)
    assert "stochastic" in recognise(
        acts=MOVES, knows_how_it_moves=noisy, state=START, toward="256"
    ).shape.named()
    assert "deterministic" in recognise(
        acts=MOVES, knows_how_it_moves=quiet, state=START, toward="256"
    ).shape.named()


def test_a_small_world_with_a_few_acts_and_a_countable_goal_wants_looking_ahead():
    knows, state = watched(adds_things=True)
    suits = recognise(acts=MOVES, knows_how_it_moves=knows, state=state, toward="256")
    assert "looking ahead" in suits.process
    assert suits.have_it


def test_the_same_world_with_nothing_to_prefer_a_result_by_says_so():
    knows, state = watched(adds_things=True)
    suits = recognise(
        acts=MOVES, knows_how_it_moves=knows, state=state, toward="open the settings"
    )
    assert not suits.have_it
    assert "prefer" in suits.process


def test_too_many_acts_to_try_all_of_them_wants_something_else():
    knows, state = watched(adds_things=True)
    many = [f"key{n}" for n in range(FEW_ENOUGH_ACTS + 4)]
    suits = recognise(acts=many, knows_how_it_moves=knows, state=state, toward="256")
    assert not suits.have_it
    assert "specialist" in suits.process


def test_a_world_too_big_to_search_is_not_called_small():
    knows, _ = watched(adds_things=True)
    huge = board([[str(c) for c in range(20)] for _ in range(10)])
    assert huge.places() > SMALL_ENOUGH_TO_SEARCH
    suits = recognise(acts=MOVES, knows_how_it_moves=knows, state=huge, toward="256")
    assert "open" in suits.shape.named()


# ── what it is for ───────────────────────────────────────────────────────

def test_it_always_says_whether_she_has_the_process_it_names():
    knows, state = watched(adds_things=True)
    for toward in ("256", "open the settings"):
        suits = recognise(acts=MOVES, knows_how_it_moves=knows, state=state, toward=toward)
        assert isinstance(suits.have_it, bool)
        assert suits.because


def test_nothing_it_says_is_about_any_particular_subject():
    knows, state = watched(adds_things=True)
    said = recognise(acts=MOVES, knows_how_it_moves=knows, state=state, toward="256").says()
    for subject in ("2048", "tile", "board", "game", "score"):
        assert subject not in said.lower()
