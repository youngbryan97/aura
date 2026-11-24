"""She thinks on while another pass is likely to change her answer, as her own passes have shown it.

Her compiled search, given a time it means to take, decided when to stop by a rule set by hand: so much more time for
each change of its best move, so much for a falling score. Whether another depth changes the answer can be learned
from her own searches instead: each completed depth is a case, with how often the answer has changed lately, how long
it has held, how far the judgement just fell, how deep it is, and whether the next depth changed it
(core/cognition/when_more_thought_changes_the_answer.py). The fitted curve reaches the search as habits, and a soft
time is spent by it: steady decisions give their time to unsteady ones.
"""
from __future__ import annotations

import random

import pytest

from core.cognition.when_more_thought_changes_the_answer import (
    ENOUGH_CASES,
    WhenMoreThoughtChanges,
    cases_from,
)


def test_each_pass_but_the_first_and_last_is_a_case_with_whether_the_next_changed_the_answer() -> None:
    passes = [(1, 30, False, 1), (2, 40, True, 2), (3, -60, False, 4), (4, -50, True, 9), (5, -50, False, 20)]
    cases = cases_from(passes, margin=50)
    assert [c[4] for c in cases] == [0, 1, 0]
    changes, held, fell, depth, _ = cases[1]
    # Depth 3: the change at depth 2 has faded by half, the answer has held one pass, the judgement fell two margins.
    assert (changes, held, fell, depth) == (0.5, 1.0, 2.0, 3.0)


def test_a_curve_is_fitted_only_on_enough_cases_and_reads_unsteadiness_as_it_was() -> None:
    rng = random.Random(5)
    record = WhenMoreThoughtChanges()
    record.cases = [(1.0, 0.0, 0.0, 8.0, 1)] * 10
    assert not record.fit()
    cases = []
    for _ in range(ENOUGH_CASES * 4):
        changes, held = rng.random() * 2, float(rng.randint(0, 6))
        chance = 0.6 if changes > 1 else 0.1 if held >= 3 else 0.25
        cases.append((changes, held, rng.uniform(-1, 1), float(rng.randint(3, 14)), int(rng.random() < chance)))
    record.cases = cases
    assert record.fit()
    assert record.chance(1.8, 0, 0, 8) > record.mean > record.chance(0.0, 6, 0, 8)
    assert set(record.as_habits()) == {"flip_mean", "flip_level", "flip_changes", "flip_held", "flip_fell",
                                       "flip_depth"}


def _searched_for(habits: dict) -> int:
    from dataclasses import replace

    from core.agency.searching_by_kind import first_weights, search
    from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

    weights = replace(first_weights(CHESS), habits=habits)
    quiet = from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -")
    return search(CHESS, quiet, seconds=2.0, soft=0.5, weights=weights).ms


def test_the_compiled_search_spends_its_soft_time_by_the_chance_another_depth_changes_its_answer() -> None:
    pytest.importorskip("aura_board_search")
    steady = {"flip_mean": 0.2, "flip_level": -12.0, "flip_changes": 0.0, "flip_held": 0.0, "flip_fell": 0.0,
              "flip_depth": 0.0}
    unsteady = {**steady, "flip_level": 12.0}
    assert _searched_for(steady) < 250
    assert _searched_for(unsteady) > 1500


def test_every_board_search_adds_its_passes_to_her_record() -> None:
    pytest.importorskip("aura_board_search")
    from core.agency.searching_by_kind import search
    from core.cognition.when_more_thought_changes_the_answer import kept
    from tests.fixtures.a_game_of_pieces_to_learn import CHESS

    before = len(kept().cases)
    search(CHESS, CHESS.the_start(), seconds=5.0, nodes=200_000)
    assert len(kept().cases) > before


def test_a_look_ahead_in_a_world_whose_answer_has_held_gives_its_time_back(monkeypatch) -> None:
    """The same curve, kept for her looks ahead in a world she acts in: where another level is unlikely to change what
    she would do, she stops looking and acts; where nothing has been learned yet, she looks as deep as the clock
    lets her finish."""
    from core.agency.looking_ahead import look_ahead
    from core.cognition import when_more_thought_changes_the_answer as thought
    from core.perception.how_it_moves import HowItMoves, shifted_and_combined
    from tests.test_looking_as_far_ahead_as_there_is_time_for import (  # noqa: F401
        CORNER,
        MOVES,
        knows,
    )

    model = HowItMoves()
    state = CORNER
    for move in ("left", "up", "right", "down", "left", "up"):
        after = shifted_and_combined(state, move)
        model.watched(state, move, after)
        state = after
    looked: list[list] = []
    monkeypatch.setattr(thought, "noted", lambda passes, margin, what=thought.BOARD_SEARCHES: looked.append(passes))
    steady = WhenMoreThoughtChanges(weights=(-12.0, 0.0, 0.0, 0.0, 0.0), mean=0.2)
    monkeypatch.setitem(thought._KEPT, thought.LOOKS_AHEAD, steady)
    look_ahead(model, CORNER, MOVES, toward="256", budget_s=2.0)
    monkeypatch.setitem(thought._KEPT, thought.LOOKS_AHEAD, WhenMoreThoughtChanges())
    look_ahead(model, CORNER, MOVES, toward="256", budget_s=2.0)
    held, unlearned = looked
    assert len(held) == 2
    assert len(unlearned) > 2
