"""Which way an act pushes is learned from what it does, not read off its name.

Her rules used to take an act called "left" to push left. A game played with
w, a, s and d had no act her rules could apply, and one whose keys run the
other way had every prediction wrong: in both she worked out nothing, searched
nothing and played at random (2026-09-25, `tools/measure_getting_there.py`,
furthest 128 against a goal of 1024).
"""

from __future__ import annotations

import random

from core.agency.a_world_compiled import compiled
from core.perception.how_it_moves import HowItMoves
from tools.measure_getting_there import WORLDS, AWorld

REVERSED = WORLDS["keys the other way"]
RENAMED = WORLDS["keys that say nothing"]


def _lived(world: AWorld, moves: int = 60, seed: int = 3) -> tuple[HowItMoves, object]:
    """Some moves in a world, watched the way the pursuit watches them."""
    roll = random.Random(seed)
    knows = HowItMoves()
    state = world.start(roll)
    names = list(world.acts)
    for _ in range(moves):
        if world.over(state):
            break
        move = roll.choice(names)
        after = world.act(state, move)
        knows.watched(state, move, after)
        if after.as_text() != state.as_text():
            after = world.something_turns_up(after, roll)
        state = after
    return knows, state


def test_before_any_move_a_name_that_says_a_way_is_taken_at_its_word():
    knows = HowItMoves()
    assert knows.way_of("left") == "left"
    assert knows.way_of("Up") == "up"


def test_a_name_that_says_nothing_pushes_no_way_until_it_has_moved_something():
    assert HowItMoves().way_of("w") == ""


def test_keys_that_run_the_other_way_are_learned():
    knows, _state = _lived(REVERSED)
    assert {act: knows.way_of(act) for act in REVERSED.acts} == REVERSED.acts
    assert knows.rule() is not None and knows.rule().name == "slides and combines"


def test_keys_named_for_nothing_are_learned():
    knows, _state = _lived(RENAMED)
    assert {act: knows.way_of(act) for act in RENAMED.acts} == RENAMED.acts
    assert knows.rule() is not None


def test_and_what_she_expects_of_them_is_what_they_do():
    knows, state = _lived(REVERSED)
    for act in REVERSED.acts:
        foreseen = knows.expect(state, act)
        assert foreseen is not None
        assert foreseen.as_text() == REVERSED.act(state, act).as_text()


def test_what_she_learned_about_her_keys_is_remembered():
    knows, _state = _lived(RENAMED)
    back = HowItMoves.from_memory(knows.as_memory(), 1.0)
    assert {act: back.way_of(act) for act in RENAMED.acts} == RENAMED.acts


def test_her_compiled_world_pushes_the_way_her_keys_do():
    knows, state = _lived(REVERSED)
    made = compiled(knows, None, state, list(REVERSED.acts))
    assert made is not None
    board = made.board(state)
    for act in REVERSED.acts:
        assert made.arrangement(made.act(board, act)).as_text() == REVERSED.act(state, act).as_text()


def test_an_act_that_never_moves_anything_is_given_no_way():
    knows, state = _lived(RENAMED)
    knows.watched(state, "n", state)
    assert knows.way_of("n") == ""
    assert knows.expect(state, "n") is None
