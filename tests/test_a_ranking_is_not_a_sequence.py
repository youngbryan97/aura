"""The moves after this one have to be moves about the situation after it.

A ranking is a judgement about alternatives at one moment. Sent as a sequence
it is four different answers to the same question, executed in order.

LIVE 2026-09-04, driving the real 2048 app: every cycle committed to four
moves — down, up, left, right, in ranking order — because every prediction
"the view will be different" held and the commitment length is read off how
often predictions hold. Each key changed the whole board, so the second, third
and fourth were answers to a board that no longer existed. Sixty moves a game,
no rule ever formed, and the pair the learner was handed spanned four acts
while being labelled as one.
"""

from __future__ import annotations

from core.skills.screen_pursuit import a_run_she_can_carry


def test_a_world_whose_acts_change_it_gets_one_act_at_a_time():
    assert a_run_she_can_carry(["up", "left", "right"], False, moved=12) == []


def test_a_model_that_can_say_what_comes_next_carries_the_whole_run():
    assert a_run_she_can_carry(["up", "left"], True, moved=12) == ["up", "left"]


def test_a_page_where_nothing_has_moved_yet_still_tries_the_next_thing():
    """A key that may simply do nothing: the ranking IS the list to try."""
    assert a_run_she_can_carry(["tab", "return"], False, moved=0) == ["tab", "return"]


def test_no_run_named_is_no_run_taken():
    assert a_run_she_can_carry([], True, moved=0) == []
    assert a_run_she_can_carry([], False, moved=99) == []


def test_the_pursuit_asks_before_committing_to_a_named_sequence():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    named = source.index('getattr(made, "then", ())')
    assert "a_run_she_can_carry(" in source[named : named + 700]


def test_every_act_of_a_run_is_counted_however_the_run_was_made():
    """The guard that refuses to learn from several acts and one reading
    reads this number, and it was only ever set for the run she built
    herself."""
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index('expected["took"] = len(follow_on) + 1')
    before = source[:at]
    # Not inside the branch that builds a run from the model: at the same
    # indentation as the branch itself.
    line_start = before.rindex("\n") + 1
    assert source[line_start:at] == " " * 8
