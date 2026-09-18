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
    assert "holding=plan[\"held\"].approach" in source[at : at + 400]
    # And the approach is reconsidered because it stopped moving her.
    at = source.index("going.why_reassess(len(moves))")
    nearby = source[at - 200 : at + 800]
    assert "still_holds" in source[: at]
    assert "it has not moved me on" in nearby
    assert "going.it_was_reassessed()" in nearby


def test_a_line_is_graded_by_whether_it_moved_her_up_a_rung():
    """An approach judged per move is judged on whether the last keystroke came out.

    That is the move's business. What an approach is for is getting her from
    one rung to the next, so that is what it is credited and debited on.
    """
    from screen_pursuit_support import pursuit_source

    from source_contract import in_order

    source = pursuit_source()
    in_order(source, "going.noticed(", 'lines.learned(A_LINE_HERE, plan["held"].approach, True)')
    in_order(
        source,
        "going.why_reassess(len(moves))",
        'lines.learned(A_LINE_HERE, plan["held"].approach, False)',
    )


def test_what_she_is_doing_says_how_far_along_it_is():
    """Asked what she is doing, "playing 2048" is half the answer.

    A run of five hundred moves and a run of five read the same from outside
    when all that is carried is what she set out to do and how she is going
    about it.
    """
    from core.agency import what_she_is_doing as doing

    doing.taking_on("play until the 2048 tile", where="2048 Game")
    doing.going_about_it("keep the largest in a corner", because="it worked here before")
    assert not any("How far along" in line for line in doing.as_lines())
    doing.getting_somewhere("128 so far, working on 256, 5 rung(s) toward 2048", reached=128)
    lines = doing.as_lines()
    assert any("How far along: 128 so far, working on 256" in line for line in lines)
    assert doing.right_now().reached == 128.0
    # Said once per thing reached, not per move.
    before = doing.right_now().changed_at
    doing.getting_somewhere("128 so far, working on 256, 5 rung(s) toward 2048")
    assert doing.right_now().changed_at == before


def test_a_rung_is_told_to_the_rest_of_her():
    from screen_pursuit_support import pursuit_source

    source = pursuit_source()
    at = source.index("doing.getting_somewhere(")
    assert "going.where_it_stands(len(moves))" in source[at : at + 200]


def test_what_was_already_there_is_where_she_starts_not_a_rung_she_climbed():
    """A board inherited with a 256 on it made "256 in three moves" her idea of usual."""
    going = HowItIsGoing(toward=2048.0)
    going.starting_from(256)
    assert going.best == 256.0
    assert going.rungs == []
    assert going.usually_takes() == 0.0
    assert going.behind(at_move=400) is False
    # And what she climbs from there is a rung, costing what it cost.
    going.noticed(512, at_move=120)
    assert [rung.took for rung in going.rungs] == [120]


def test_the_first_reading_of_a_run_is_where_she_starts():
    from screen_pursuit_support import pursuit_source

    source = pursuit_source()
    at = source.index("going.starting_from(made)")
    assert "not moves" in source[at - 300 : at]


def test_a_line_says_what_it_is_for_and_is_reported_against_it():
    """An approach adopted without saying what it is for cannot be wrong about anything."""
    going = _passing((8, 10), (16, 20), (32, 30))
    said = going.expecting("keep the largest in a corner", at_move=30)
    assert "should get me to 64" in said
    assert "about 10 move(s)" in said
    assert going.holding == "keep the largest in a corner"
    going.noticed(64, at_move=44)
    assert "took 14" in going.how_it_turned_out(at_move=44)
    # With no rung behind her she names what it is for and claims no cost.
    fresh = HowItIsGoing(toward=100.0)
    first = fresh.expecting("try the edges", at_move=0)
    assert first == "this should get me to 100"
    # And with nothing to aim at at all, nothing is claimed.
    assert HowItIsGoing().expecting("try the edges", at_move=0) == ""


def test_the_loop_says_what_it_expects_and_reports_back():
    from screen_pursuit_support import pursuit_source

    source = pursuit_source()
    at = source.index("going.expecting(fresh.approach, len(moves))")
    assert "said = f\"{said} — {wants}\"" in source[at - 400 : at + 200]
    assert "going.how_it_turned_out(len(moves))" in source
