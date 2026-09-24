"""A search deeper than her model is right is that many levels of fiction.

She searched as deep as the clock allowed, which is the wrong bound: three
levels into a model that is only right about the next move is arrived at
expensively, looks surer the deeper it goes, and is acted on.

The right bound is a measurement she is already producing. Every move carries
a prediction of what the world will look like after it, and every move of a
plan she commits to carries one at its own distance. Graded against what
happened, those say how far her model carries here.
"""

from __future__ import annotations

from core.agency.how_far_her_model_carries import (
    ENOUGH_AT_A_DISTANCE,
    HowFarHerModelCarries,
)


def _grading(carried: HowFarHerModelCarries, distance: int, *, right: int, wrong: int,
             baseline_right: int = 0, confidence: float = 0.8) -> None:
    for _ in range(right):
        carried.it_predicted(distance=distance, confidence=confidence, was_right=True)
    for _ in range(wrong):
        carried.it_predicted(distance=distance, confidence=confidence, was_right=False)
    for _ in range(baseline_right):
        carried.it_predicted(
            distance=distance, confidence=confidence, was_right=False,
            would_no_change_have_been_right=True,
        )


def test_a_distance_with_too_little_graded_says_nothing():
    carried = HowFarHerModelCarries()
    _grading(carried, 1, right=ENOUGH_AT_A_DISTANCE - 1, wrong=0)
    assert carried.carries_to() == 0


def test_it_carries_as_far_as_it_beats_saying_nothing_changed():
    carried = HowFarHerModelCarries()
    _grading(carried, 1, right=20, wrong=1)
    _grading(carried, 2, right=16, wrong=4)
    # At three acts out she is wrong as often as saying nothing changed.
    _grading(carried, 3, right=2, wrong=6, baseline_right=6)
    assert carried.carries_to() == 2


def test_a_model_that_only_matches_the_baseline_carries_nowhere():
    carried = HowFarHerModelCarries()
    # Right every time, and so is saying nothing changed: a world that does
    # nothing is not a world she has a model of.
    for _ in range(20):
        carried.it_predicted(
            distance=1, confidence=0.9, was_right=True, would_no_change_have_been_right=True
        )
    assert carried.carries_to() == 0


def test_her_confidence_is_corrected_by_her_own_overclaiming():
    carried = HowFarHerModelCarries()
    # She says 0.9 and is right half the time.
    for i in range(20):
        carried.it_predicted(distance=1, confidence=0.9, was_right=bool(i % 2))
    assert carried.calibration.overconfidence is not None
    assert carried.how_sure_she_should_be(0.9) < 0.6
    # And says so.
    assert "overconfident" in carried.says()


def test_with_nothing_measured_a_claim_stands_as_it_is():
    carried = HowFarHerModelCarries()
    assert carried.how_sure_she_should_be(0.7) == 0.7
    assert carried.says() == ""


def test_what_it_measured_survives_the_run():
    carried = HowFarHerModelCarries()
    _grading(carried, 1, right=20, wrong=2)
    _grading(carried, 2, right=12, wrong=1)
    again = HowFarHerModelCarries.from_memory(carried.as_memory())
    assert again.carries_to() == carried.carries_to()
    assert again.calibration.n == carried.calibration.n
    assert again.graded == carried.graded


def test_a_record_of_nothing_reads_back_as_nothing():
    assert HowFarHerModelCarries.from_memory(None).carries_to() == 0
    assert HowFarHerModelCarries.from_memory({"calibration": "rubbish"}).calibration.n == 0


def test_the_search_stops_where_her_model_stops_carrying():
    """Measured, not assumed: the bound is her own graded predictions."""
    from core.agency.a_world_compiled import compiled, search
    from core.perception.how_it_moves import HowItMoves, shifted_and_combined
    from core.perception.what_is_there import Arrangement, Cell
    from core.perception.what_the_world_does import WhatTheWorldDoes

    def board(values):
        return Arrangement(
            4, 4, tuple(Cell(i // 4, i % 4, str(v), (0.0, 0.0)) for i, v in enumerate(values) if v)
        )

    knows, world = HowItMoves(), WhatTheWorldDoes()
    state = board([2, 4, 0, 8, 0, 2, 4, 0, 4, 0, 0, 2, 64, 2, 0, 4])
    for move in ("left", "up", "right", "down", "left", "up"):
        after = shifted_and_combined(state, move)
        knows.watched(state, move, after)
        world.watched(knows.expect(state, move), after)
        state = after
    made = compiled(knows, world, state, ("up", "down", "left", "right"))
    assert made is not None

    def worth(_board):
        return 0.0

    _scored, deep = search(
        made, state, ("up", "down", "left", "right"),
        budget_s=2.0, worth=worth, dead=-1.0,
    )
    _scored, bounded = search(
        made, state, ("up", "down", "left", "right"),
        budget_s=2.0, worth=worth, dead=-1.0, no_deeper_than=2,
    )
    assert bounded <= 2
    assert deep >= bounded


def test_the_run_measures_it_and_the_search_is_given_it():
    from screen_pursuit_support import pursuit_source

    source = pursuit_source()
    assert "HowFarHerModelCarries.from_memory(" in source
    assert "carries=carries," in source
    assert '"carries": carries.as_memory()' in source
    at = source.index("carries.it_predicted(")
    graded = source[at : at + 500]
    assert 'distance=expected["took"]' in graded
    assert 'confidence=expected["confidence"]' in graded
    assert "was_right=same" in graded
    assert "would_no_change_have_been_right" in graded
    assert "no_deeper_than=as_far_as_it_carries" in source


def test_a_distance_nobody_measured_does_not_bound_the_search():
    """Distances past one are graded only when she commits to several moves.

    On a fast board she rarely does, so they never gathered enough to count,
    and reading the furthest distance MEASURED to carry as the bound held every
    search to one move. LIVE 2026-09-23 on 2048: "she thought for 0.00s" on
    every move, and the board filled at 128.
    """
    carried = HowFarHerModelCarries()
    _grading(carried, 1, right=120, wrong=80)
    _grading(carried, 2, right=2, wrong=1)
    assert carried.carries_to() == 0
    assert carried.measured_to() == 1
    assert "at least 1" in carried.says()


def test_a_distance_measured_to_fail_still_bounds_it():
    carried = HowFarHerModelCarries()
    _grading(carried, 1, right=20, wrong=1)
    _grading(carried, 2, right=0, wrong=8, baseline_right=8)
    _grading(carried, 3, right=16, wrong=4)
    assert carried.carries_to() == 1
