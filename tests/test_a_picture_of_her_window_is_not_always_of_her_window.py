"""A rectangle is not a window.

The capture takes whatever is drawn in that part of the display. A window
sitting over hers comes back scoped to her application, the right shape, the
right size, and full of somebody else's words — and ``scoped_to``, which
exists to catch a reading of the whole desktop, says it is hers, because the
bounds were found and the picture was taken there.

LIVE 2026-09-04, driving the 2048 app: arrangements laid into her four by four
board reading ". The White House 12h Walker Kessler | . ARCADE.GOV . Show
more", learned from as though they were the game, and every move compared
across one of them recorded as having changed nothing.

She already refuses to type into a window that is not in front. This is the
same rule for looking.
"""

from __future__ import annotations

from core.skills.screen_pursuit import _both_of_the_thing, _was_of_that_window


def _reading(**over: object) -> dict:
    base = {
        "scoped_to": "2048 Game",
        "in_front_then": "2048 Game",
        "her_window_showing": True,
        "layout": [],
    }
    base.update(over)
    return base


def test_a_reading_taken_while_it_was_in_front_is_of_that_window():
    assert _was_of_that_window(_reading(), "2048 Game")


def test_a_reading_taken_while_something_else_was_in_front_is_not():
    assert not _was_of_that_window(_reading(in_front_then="X"), "2048 Game")


def test_a_reading_that_never_said_who_was_in_front_says_nothing_either_way():
    assert _was_of_that_window(_reading(in_front_then=""), "2048 Game")
    assert _was_of_that_window({"scoped_to": "2048 Game"}, "2048 Game")


def test_the_name_need_only_match_the_way_the_host_reports_it():
    assert _was_of_that_window(_reading(in_front_then="2048 Game.app"), "2048 Game")
    assert _was_of_that_window(_reading(in_front_then="2048 game"), "2048 Game")


def test_a_pair_read_through_another_window_is_not_a_pair_of_the_thing():
    mine, theirs = _reading(), _reading(in_front_then="X")
    assert _both_of_the_thing("2048 Game", mine, mine)
    assert not _both_of_the_thing("2048 Game", mine, theirs)
    assert not _both_of_the_thing("2048 Game", theirs, mine)


def test_an_unscoped_reading_is_still_refused():
    assert not _both_of_the_thing(
        "2048 Game", _reading(), _reading(scoped_to="", in_front_then="2048 Game")
    )


def test_a_run_driving_nothing_in_particular_is_unaffected():
    assert _both_of_the_thing("", _reading(in_front_then="X"), _reading())


def test_a_window_on_another_space_is_not_on_the_screen_photographed():
    """Its bounds come back exactly as before and none of its pixels are there."""
    assert not _was_of_that_window(_reading(her_window_showing=False), "2048 Game")
    assert not _both_of_the_thing(
        "2048 Game", _reading(), _reading(her_window_showing=False)
    )


def test_the_reading_records_who_was_in_front_and_whether_hers_was_drawn():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.read_screen)
    assert '"in_front_then": in_front' in source
    assert '"her_window_showing": showing' in source
    assert "_who_the_screen_belongs_to(app_name) if app_name else" in source


def test_the_window_server_is_asked_about_the_application_being_driven():
    from core.skills.screen_pursuit import _who_the_screen_belongs_to

    front, drawing = _who_the_screen_belongs_to("a name no application has")
    assert drawing is False
    assert isinstance(front, str)
