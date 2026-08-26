"""How a thing responds, worked out from watching herself act in it.

Every move yields a triple — what was there, what she did, what was there
afterwards — and all three were thrown away after one glance. So she could
never try a move without making it, and every plan was a bet placed blind.

Nothing here is told what it is looking at. The rules are ways a set of things
in rows and columns can answer to being pushed, and which one holds is decided
by whether it keeps predicting.
"""

from __future__ import annotations

import random

import pytest

from core.perception.how_it_moves import (
    ENOUGH_TO_TRUST,
    HowItMoves,
    shifted,
    shifted_and_combined,
    unchanged,
)
from core.perception.what_is_there import Arrangement, Cell, arranged


def board(rows: list[list[str]]) -> Arrangement:
    cells = [
        (0.2 + r * 0.15, 0.2 + c * 0.15, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ]
    return arranged(cells)


START = board([
    ["2", "2", "4", ""],
    ["", "4", "", "8"],
    ["8", "", "", "8"],
    ["", "", "2", "2"],
])


def spawn(state: Arrangement, seed: random.Random) -> Arrangement:
    """What a world adds on its own, which no rule about her acts claims to know."""
    empty = [
        (r, c)
        for r in range(state.rows)
        for c in range(state.columns)
        if state.at(r, c) is None
    ]
    if not empty:
        return state
    r, c = seed.choice(empty)
    return Arrangement(state.rows, state.columns, state.cells + (Cell(r, c, "2", (0.0, 0.0)),))


def watch(world, moves: int = 8, seed: int = 7) -> tuple[HowItMoves, Arrangement]:
    """Let her act in a world she was told nothing about."""
    rng = random.Random(seed)
    knows = HowItMoves()
    state = START
    for _ in range(moves):
        move = rng.choice(["left", "right", "up", "down"])
        after = spawn(world(state, move), rng)
        knows.watched(state, move, after)
        state = after
    return knows, state


# ── the rules themselves ─────────────────────────────────────────────────

def test_sliding_puts_everything_as_far_that_way_as_it_goes():
    assert shifted(START, "left").as_text().splitlines()[0] == "2 2 4 ."
    assert shifted(START, "left").as_text().splitlines()[2] == "8 8 . ."


def test_sliding_and_combining_joins_equal_neighbours():
    assert shifted_and_combined(START, "left").as_text().splitlines() == [
        "4 4 . .",
        "4 8 . .",
        "16 . . .",
        "4 . . .",
    ]


def test_a_push_the_other_way_runs_the_other_way():
    assert shifted_and_combined(START, "down").as_text().splitlines()[3] == "8 4 2 2"


def test_what_is_not_a_number_still_stacks():
    """Two of the same thing become one of it, which is what stacking looks like."""
    same = board([["a", "a", "b", "c"], ["d", "e", "f", "g"]])
    assert shifted_and_combined(same, "left").as_text().splitlines()[0] == "a b c ."


def test_a_direction_it_does_not_know_is_not_answered():
    assert shifted(START, "diagonally") is None
    assert unchanged(START, "") is None


# ── working out which one holds ──────────────────────────────────────────

def test_she_does_not_guess_before_she_has_watched_enough():
    knows, _ = watch(shifted_and_combined, moves=ENOUGH_TO_TRUST - 1)
    assert knows.rule() is None
    assert knows.expect(START, "left") is None
    assert "not worked out yet" in knows.says()


def test_she_works_out_a_world_that_slides_and_combines():
    knows, _ = watch(shifted_and_combined)
    assert knows.rule().name == "slides and combines"
    assert knows.confidence() == pytest.approx(1.0)


def test_she_works_out_a_different_world_as_different():
    knows, _ = watch(shifted)
    assert knows.rule().name == "slides"


def test_she_works_out_a_thing_that_does_not_answer_at_all():
    knows, _ = watch(lambda state, _move: state)
    assert knows.rule().name == "does not move"


def test_a_world_that_does_none_of_these_leaves_her_saying_so():
    def scrambles(state: Arrangement, _move: str) -> Arrangement:
        return board([["7", "", "", ""], ["", "7", "", ""], ["", "", "7", ""], ["", "", "", "7"]])

    knows, _ = watch(scrambles, moves=10)
    assert knows.rule() is None
    assert knows.expect(START, "left") is None


# ── using it ─────────────────────────────────────────────────────────────

def test_once_she_knows_she_can_try_a_move_without_making_it():
    knows, state = watch(shifted_and_combined)
    imagined = knows.expect(state, "left")
    assert imagined is not None
    assert imagined.as_text() == shifted_and_combined(state, "left").as_text()


def test_she_can_lay_out_every_way_it_could_go():
    knows, state = watch(shifted_and_combined)
    futures = knows.expect_all(state, ["up", "down", "left", "right"])
    assert set(futures) == {"up", "down", "left", "right"}
    assert all(isinstance(future, Arrangement) for future in futures.values())


def test_with_no_rule_there_are_no_futures_to_weigh():
    assert HowItMoves().expect_all(START, ["left", "right"]) == {}


def test_what_the_world_adds_on_its_own_does_not_falsify_her_rule():
    """A dealt tile is not a mistake by a rule about what her own act moves."""
    knows, _ = watch(shifted_and_combined, moves=12)
    assert knows.confidence() == pytest.approx(1.0)


def test_a_rule_that_stops_working_stops_being_trusted():
    knows, state = watch(shifted_and_combined)
    assert knows.rule() is not None
    rng = random.Random(3)
    for _ in range(20):
        after = spawn(state, rng)
        knows.watched(state, "left", after)
        state = after
    assert knows.rule() is None or knows.confidence() < 1.0


def test_watching_nothing_happen_in_an_empty_world_teaches_nothing():
    knows = HowItMoves()
    knows.watched(arranged([]), "left", arranged([]))
    assert knows.seen == 0
