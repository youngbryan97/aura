"""A goal she can only pass or fail tells her nothing while she is working on it.

What somebody working on something actually reasons about is the middle: that
two steps ago took thirty moves each and this one has taken ninety, that the
approach adopted for this step has not moved her since, that the next step is
twice the last one because every step here has been. That is what says when
to change what she is doing rather than keep pushing, and it is what a second
attempt needs to know from the first.

Nothing here knows what a world is: a rung is a number she reached, a cost is
moves, and the pattern is whichever of "times as much" or "more than" fits
what she has actually passed.
"""

from __future__ import annotations

from core.agency.how_it_is_going import HowItIsGoing


def _passing(*rungs: tuple[float, int]) -> HowItIsGoing:
    """A record of rungs reached at those moves."""
    going = HowItIsGoing(toward=2048.0)
    for reached, at_move in rungs:
        going.noticed(reached, at_move)
    return going


def test_what_she_reached_is_a_rung_and_only_a_better_one_is():
    going = HowItIsGoing(toward=2048.0)
    assert going.noticed(8, at_move=4) is True
    assert going.noticed(8, at_move=9) is False
    assert going.noticed(4, at_move=11) is False
    assert going.noticed(16, at_move=14) is True
    assert [rung.reached for rung in going.rungs] == [8.0, 16.0]
    assert [rung.took for rung in going.rungs] == [4, 10]


def test_the_next_one_is_projected_from_the_ones_behind_it():
    doubling = _passing((8, 5), (16, 15), (32, 40))
    assert doubling.next_rung() == 64.0
    by_tens = HowItIsGoing(toward=100.0)
    for reached, at_move in ((10, 3), (20, 9), (30, 20)):
        by_tens.noticed(reached, at_move)
    assert by_tens.next_rung() == 40.0


def test_a_projection_past_the_finish_is_the_finish():
    nearly = _passing((512, 100), (1024, 300))
    assert nearly.next_rung() == 2048.0
    assert nearly.toward == 2048.0


def test_with_no_pattern_the_finish_is_the_next_thing():
    scattered = HowItIsGoing(toward=500.0)
    for reached, at_move in ((7, 2), (113, 9), (120, 30)):
        scattered.noticed(reached, at_move)
    assert scattered.next_rung() == 500.0


def test_what_a_rung_costs_is_read_off_what_they_cost():
    steady = _passing((8, 10), (16, 20), (32, 30))
    assert steady.usually_takes() == 10.0


def test_a_cost_that_grows_is_projected_by_its_growth():
    """A rung worth twice the last one costs about twice the last one."""
    growing = _passing((8, 10), (16, 30), (32, 70), (64, 150))
    # 10, 20, 40, 80 — the next is about 160, not the middle of what is behind.
    assert 140.0 <= growing.usually_takes() <= 180.0


def test_behind_is_her_own_history_not_a_clock():
    going = _passing((8, 10), (16, 20), (32, 30))
    # Ten moves is usual here, and fifteen is not yet behind.
    assert going.behind(at_move=45) is False
    assert going.behind(at_move=90) is True
    assert "without reaching" in going.why_reassess(at_move=90)
    assert going.why_reassess(at_move=45) == ""


def test_nothing_reached_yet_is_not_behind():
    nothing = HowItIsGoing(toward=2048.0)
    assert nothing.behind(at_move=500) is False
    assert nothing.where_it_stands(at_move=500) == "nothing reached here yet"


def test_it_says_where_it_stands():
    going = _passing((8, 10), (16, 20), (32, 30))
    said = going.where_it_stands(at_move=44)
    assert "32 so far" in said
    assert "working on 64" in said
    assert "toward 2048" in said
    assert "14 move(s) on this one" in said


def test_it_survives_the_attempt_that_made_it():
    first = _passing((8, 10), (16, 20), (32, 30))
    first.holding = "keep the largest in a corner"
    again = HowItIsGoing.from_memory(first.as_memory(), toward=2048.0)
    assert again.best == 32.0
    assert [rung.reached for rung in again.rungs] == [8.0, 16.0, 32.0]
    assert again.next_rung() == 64.0
    # A new attempt starts its own clock and keeps what the rungs cost.
    assert again.last_rung_at == 0
    assert again.usually_takes() == 10.0
    assert again.holding == "keep the largest in a corner"


def test_a_record_of_nothing_is_read_without_complaint():
    assert HowItIsGoing.from_memory(None, toward=64.0).toward == 64.0
    assert HowItIsGoing.from_memory({"rungs": [{"reached": "bad"}]}).rungs == []


def test_the_run_carries_it_and_the_approach_is_reconsidered_on_it():
    """The pieces have to talk: the loop records rungs and reads them back."""
    from screen_pursuit_support import pursuit_source

    source = pursuit_source()
    # Loaded with what she knew, carried into the decision, kept at the end.
    assert "HowItIsGoing.from_memory(" in source
    assert "going=going," in source
    assert '"how_it_is_going": going.as_memory()' in source
    # A rung is recorded where she notices she got further.
    at = source.index("going.noticed(")
    assert "holding=plan[\"held\"].approach" in source[at : at + 300]
    # And the approach is reconsidered because it stopped moving her.
    at = source.index("going.why_reassess(len(moves))")
    nearby = source[at - 200 : at + 400]
    assert "still_holds" in source[: at]
    assert "it has not moved me on" in nearby
    assert "going.it_was_reassessed()" in nearby
