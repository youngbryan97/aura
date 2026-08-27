"""A score beside a board answers to her, and is not part of what she moves.

The part of the screen that responds to her acts includes the score, the
clock, the count of moves — they all change when she acts. No rule about what
her act MOVES can ever predict one, so every hypothesis was wrong every time.
Measured live 2026-08-26: seventeen readings watched, nothing worked out, on a
board she was reading correctly.

Nothing here knows what a score is. What it knows is that a thing which never
goes anywhere and keeps saying something different is not a thing her moves
move.
"""

from __future__ import annotations

import random

import pytest

from core.perception.how_it_moves import (
    ALWAYS_THERE,
    STILL_ENOUGH_TO_JUDGE,
    HowItMoves,
    shifted_and_combined,
)
from core.perception.what_is_there import Arrangement, Cell, arranged

BOARD = [["2", "2", "4", ""], ["", "4", "", "8"], ["8", "", "", "8"], ["", "", "2", "2"]]


def board(rows) -> Arrangement:
    return arranged([
        (0.20 + r * 0.15, 0.20 + c * 0.15, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ])


def with_readout(inner: Arrangement, score: int, like: Arrangement | None) -> Arrangement:
    """The thing, with a number stuck above it that changes on every act."""
    cells = [(0.05, 0.20, str(score))] + [
        (0.20 + cell.row * 0.15, 0.20 + cell.column * 0.15, cell.says) for cell in inner.cells
    ]
    return arranged(cells, like=like)


def spawn(state: Arrangement, rng: random.Random) -> Arrangement:
    free = [
        (r, c)
        for r in range(state.rows)
        for c in range(state.columns)
        if state.at(r, c) is None
    ]
    if not free:
        return state
    r, c = rng.choice(free)
    return Arrangement(
        state.rows,
        state.columns,
        state.cells + (Cell(r, c, "2", (0.0, 0.0)),),
        state.down_at,
        state.across_at,
    )


def watch(moves: int = 60, seed: int = 5) -> tuple[HowItMoves, Arrangement]:
    rng = random.Random(seed)
    knows = HowItMoves()
    plain = board(BOARD)
    score = 0
    state = with_readout(plain, score, None)
    for _ in range(moves):
        move = rng.choice(["left", "right", "up", "down"])
        plain = spawn(shifted_and_combined(plain, move), rng)
        score += 4
        after = with_readout(plain, score, state)
        knows.watched(state, move, after)
        state = after
    return knows, state


# ── telling a readout from the thing ─────────────────────────────────────

def test_she_finds_the_readout_and_only_the_readout():
    knows, _ = watch()
    assert knows.counters == {(0, 0)}


def test_and_then_works_out_how_the_thing_moves():
    knows, _ = watch()
    assert knows.rule() is not None
    assert knows.rule().name == "slides and combines"


def test_and_can_try_a_move_without_making_it():
    knows, state = watch()
    assert knows.expect(state, "left") is not None


def test_what_she_imagines_is_the_thing_without_its_furniture():
    knows, state = watch()
    imagined = knows.expect(state, "left")
    assert (imagined.rows, imagined.columns) == (4, 4)


# ── what makes something furniture ───────────────────────────────────────

def test_a_place_that_is_often_empty_is_a_place_not_a_readout():
    """A board cell changes often while it is occupied, and is empty between."""
    knows, _ = watch()
    assert all(where == (0, 0) for where in knows.counters)


def test_nothing_is_judged_before_it_has_been_watched_enough():
    knows, _ = watch(moves=STILL_ENOUGH_TO_JUDGE - 2)
    assert not knows.counters


def test_a_place_that_never_changes_is_not_a_readout():
    knows = HowItMoves()
    still = board([["7", "7", "7"], ["7", "7", "7"], ["7", "7", "7"]])
    for _ in range(STILL_ENOUGH_TO_JUDGE + 4):
        knows.watched(still, "left", still)
    assert not knows.counters


def test_somewhere_seen_empty_once_is_a_place_for_good():
    knows, _ = watch()
    assert (2, 2) in knows._a_place
    assert (2, 2) not in knows.counters


def test_being_never_empty_is_what_makes_it_furniture():
    assert 0.0 < ALWAYS_THERE <= 1.0


# ── and what she learned about the wrong thing is dropped ────────────────

def test_learning_what_the_thing_is_starts_the_learning_again():
    """Everything until then compared a board with a score stuck to it."""
    knows, _ = watch()
    tried = max(knows.tried.values()) if knows.tried else 0
    assert 0 < tried < 60


def test_a_reading_with_the_furniture_cropped_out_is_what_is_compared():
    knows, state = watch()
    assert knows.the_thing(state).rows == 4
    assert state.rows == 5


# ── furniture as it really appears ───────────────────────────────────────

def dressed(inner: Arrangement, score: int, like: Arrangement | None) -> Arrangement:
    """The thing as a page really presents it: a title, a label, a score."""
    cells = [(0.02, 0.20, "2048"), (0.05, 0.50, "SCORE"), (0.05, 0.65, str(score))] + [
        (0.20 + cell.row * 0.15, 0.20 + cell.column * 0.15, cell.says) for cell in inner.cells
    ]
    return arranged(cells, like=like)


def watch_a_page(moves: int = 60, seed: int = 5) -> tuple[HowItMoves, Arrangement]:
    rng = random.Random(seed)
    knows = HowItMoves()
    plain = board(BOARD)
    score = 0
    state = dressed(plain, score, None)
    for _ in range(moves):
        move = rng.choice(["left", "right", "up", "down"])
        moved = shifted_and_combined(plain, move)
        if moved.occupied() < plain.occupied():
            score += 8  # a score changes on a merge, not on every act
        plain = spawn(moved, rng)
        after = dressed(plain, score, state)
        knows.watched(state, move, after)
        state = after
    return knows, state


def test_a_title_a_label_and_a_score_are_all_furniture():
    knows, _ = watch_a_page()
    assert len(knows.counters) == 3


def test_a_readout_that_only_changes_sometimes_is_still_a_readout():
    """A score changes on a merge, not on every act. Never being empty is
    what tells it apart, not how often it changes."""
    knows, _ = watch_a_page()
    assert knows.rule() is not None


def test_and_the_thing_underneath_is_worked_out():
    knows, state = watch_a_page()
    assert knows.rule().name == "slides and combines"
    assert knows.confidence() == pytest.approx(1.0)
    assert knows.expect(state, "left") is not None
