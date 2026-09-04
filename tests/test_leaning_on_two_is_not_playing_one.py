"""Leaning on a pair means considering both of them.

Two of the four directions, almost exclusively, is how somebody clearing 2048
plays, and leaning on a pair was measured to beat leaning on any single act:
down with right keeps the same corner, and neither of them says so alone.

LIVE 2026-09-04 on the real board: the habit returned one act, the choice was
narrowed to that act before anything looked ahead, and every move came back
"down is the only thing available". The pair's whole advantage is a fact about
the two together, and only one of them was ever played.
"""

from __future__ import annotations

from core.cognition.the_ones_she_reaches_for import TheOnesSheReachesFor

ACTS = ["down", "left", "right", "up"]


def _leaning(*on: str) -> TheOnesSheReachesFor:
    habit = TheOnesSheReachesFor()
    habit.leaning_on = tuple(on)
    return habit


def test_both_of_a_pair_are_considered():
    assert _leaning("down", "right").the_ones_to_consider(ACTS) == ("down", "right")


def test_the_order_is_what_is_on_offer_not_what_she_leans_on():
    assert _leaning("right", "down").the_ones_to_consider(ACTS) == ("down", "right")


def test_one_of_the_pair_missing_leaves_the_other():
    assert _leaning("down", "right").the_ones_to_consider(["right", "up"]) == ("right",)


def test_none_of_them_available_says_so():
    assert _leaning("down", "right").the_ones_to_consider(["up", "left"]) == ()


def test_leaning_on_nothing_narrows_nothing():
    assert TheOnesSheReachesFor().the_ones_to_consider(ACTS) == ()


def test_the_pursuit_narrows_to_the_set_and_lets_the_model_choose():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    assert "reaches.the_ones_to_consider(foreseeable)" in source
    at = source.index("reaches.the_ones_to_consider(foreseeable)")
    assert "one.name in wants" in source[at : at + 1400]


def test_a_habit_that_covers_everything_narrows_nothing():
    """Leaning on all four is not a preference and must not look like one."""
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index("reaches.the_ones_to_consider(foreseeable)")
    assert "len(wants) < len(foreseeable)" in source[at : at + 900]


def test_a_leaning_of_one_act_is_not_applied_as_a_filter():
    """A choice of one is not a choice, and her search is worth more.

    Offline, looking ahead reaches a median of 512 where taking any legal move
    reaches 128. A habit that narrows the options to a single act takes that
    search off the board: the deliberation arrives with one option in it and
    reports "the only thing available".
    """
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index("reaches.the_ones_to_consider(foreseeable)")
    nearby = source[at : at + 900]
    assert "if len(wants) < 2:" in nearby
    assert "wants = ()" in nearby
