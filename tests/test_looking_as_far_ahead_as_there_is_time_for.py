"""Several moves ahead, as deep as the time this decision is worth allows.

Once she can try a move without making it and say which result is better, the
two compose. How deep is not a setting: each level costs branching times the
last, the cost of a level is measured rather than assumed, and the depth is
whatever fits.
"""

from __future__ import annotations

import pytest

from core.agency.looking_ahead import DEEPEST, how_deep_to_look, look_ahead
from core.perception.how_it_moves import HowItMoves, shifted_and_combined
from core.perception.what_is_there import arranged


def board(rows: list[list[str]]):
    return arranged([
        (0.2 + r * 0.15, 0.2 + c * 0.15, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ])


CORNER = board([
    ["2", "4", "", "8"],
    ["", "2", "4", ""],
    ["4", "", "", "2"],
    ["64", "2", "", "4"],
])
LINE = "keep the 64 in the bottom-left corner"
MOVES = ["up", "down", "left", "right"]


@pytest.fixture
def knows() -> HowItMoves:
    model = HowItMoves()
    state = CORNER
    for move in ("left", "up", "right", "down", "left", "up"):
        after = shifted_and_combined(state, move)
        model.watched(state, move, after)
        state = after
    return model


# ── how deep ─────────────────────────────────────────────────────────────

def test_one_level_is_always_affordable():
    assert how_deep_to_look(4, 0.0000001) == 1


def test_more_time_buys_more_depth():
    assert how_deep_to_look(4, 30.0) > how_deep_to_look(4, 0.001)


def test_it_never_looks_deeper_than_a_world_that_moves_under_her_allows():
    assert how_deep_to_look(4, 10_000.0) == DEEPEST


def test_nothing_available_is_not_a_search():
    assert how_deep_to_look(0, 1.0) == 1


# ── looking ──────────────────────────────────────────────────────────────

def test_with_no_worked_out_rule_there_is_nothing_to_see(knows):
    assert look_ahead(HowItMoves(), CORNER, MOVES, toward="256") == {}


def test_with_nothing_to_prefer_by_it_still_answers_from_room(knows):
    seen = look_ahead(knows, CORNER, MOVES, toward="256")
    assert set(seen) == set(MOVES)


def test_the_line_she_is_holding_wins_the_search(knows):
    seen = look_ahead(knows, CORNER, MOVES, toward="256", approach=LINE)
    best = max(seen.items(), key=lambda row: row[1][0])[0]
    assert "bottom-left" in shifted_and_combined(CORNER, best).places_of("64")


def test_every_move_comes_back_with_a_reason_she_could_say(knows):
    seen = look_ahead(knows, CORNER, MOVES, toward="256", approach=LINE)
    assert all(said for _score, said in seen.values())


def test_looking_deeper_scores_higher_than_looking_once(knows):
    shallow = look_ahead(knows, CORNER, MOVES, toward="256", approach=LINE, budget_s=0.0000001)
    deep = look_ahead(knows, CORNER, MOVES, toward="256", approach=LINE, budget_s=5.0)
    assert max(s for s, _ in deep.values()) > max(s for s, _ in shallow.values())


def test_a_search_over_nothing_is_nothing(knows):
    assert look_ahead(knows, CORNER, [], toward="256") == {}
    assert look_ahead(knows, None, MOVES, toward="256") == {}
    assert look_ahead(None, CORNER, MOVES, toward="256") == {}


def test_a_model_she_does_not_trust_is_not_searched():
    """Confidence gates it: a rule that has not been right is not a future."""
    class _Unsure:
        def confidence(self):
            return 0.0

        def expect(self, state, action):
            return state

    assert look_ahead(_Unsure(), CORNER, MOVES, toward="256") == {}
