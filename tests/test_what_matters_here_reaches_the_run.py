"""What a thing about a situation is worth here has to reach the deciding.

The organ that works it out was written and tested and called by nothing. She
judged every world by the same five numbers somebody picked as a starting
guess, including worlds she had played for hours — and the weights parameter
that carries the answer runs all the way from the pursuit through look_ahead
into the measure, with nothing ever filling it in.
"""

from __future__ import annotations

import inspect

from core.skills import screen_pursuit

SOURCE = inspect.getsource(screen_pursuit.pursue_on_screen)


def test_it_is_carried_in_from_what_she_learned_here():
    assert "WhatMakesItGoodHere.from_memory(" in SOURCE
    assert 'knew.get("matters")' in SOURCE


def test_every_move_teaches_it_against_the_world_s_own_count():
    at = SOURCE.index("matters.what_came_of_it(")
    nearby = SOURCE[at - 500 : at + 400]
    assert "rose" in nearby, "graded by how far the tally moved"
    assert "terms(" in nearby


def test_what_it_worked_out_is_what_she_looks_ahead_with():
    at = SOURCE.index("ahead = look_ahead(")
    assert "weights=matters.weights()" in SOURCE[at : at + 700]


def test_the_run_she_builds_on_the_model_uses_them_too():
    at = SOURCE.index("ahead_now = look_ahead(")
    assert "weights=matters.weights()" in SOURCE[at : at + 400]


def test_it_is_kept_for_the_next_time_she_is_here():
    assert '"matters": matters.as_memory(),' in SOURCE


def test_she_says_it_when_she_has_worked_it_out():
    at = SOURCE.index("matters.what_came_of_it(")
    assert "matters.says()" in SOURCE[at : at + 700]


def test_nothing_is_weighed_by_it_until_it_is_worked_out():
    """The organ answers None until it has watched enough, and None is the
    standing guess — so a fresh world is unaffected."""
    from core.agency.what_makes_it_good_here import WhatMakesItGoodHere

    assert WhatMakesItGoodHere().weights() is None
