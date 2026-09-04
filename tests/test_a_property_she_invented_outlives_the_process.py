"""A measure she composed lived only in the process that composed it.

Inventing one takes a run's worth of pairs her measure could not account for,
and proving one takes sixty observations under trial — more than a run. So
every restart threw away the only part of her judgement that was hers rather
than authored, and the next run started composing it again from nothing.

A measure is four words and a weight. Putting it back is looking its name up
in the space it came from.
"""

from __future__ import annotations

import inspect

from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY, INVENTED, forget, promote
from core.agency.inventing_a_measure import Measure, every_measure, measure_named
from core.skills import screen_pursuit

SOURCE = inspect.getsource(screen_pursuit.pursue_on_screen)


def test_a_measure_can_be_found_again_by_its_name():
    one = next(iter(every_measure()))
    assert measure_named(one.name) == one


def test_a_promoted_measure_is_a_name_and_a_weight():
    one = Measure(at="rows", of="how full", summed="the middle one")
    try:
        assert promote(one, 0.42) == one.name
        assert one.name in INVENTED
        assert AS_GOOD_A_GUESS_AS_ANY[one.name] == 0.42
    finally:
        forget(one.name)
    assert one.name not in INVENTED


def test_the_run_puts_back_what_this_world_proved():
    at = SOURCE.index('knew.get("judging")')
    nearby = SOURCE[at : at + 500]
    assert "measure_named(str(name))" in nearby
    assert "promote(found, float(worth))" in nearby


def test_it_keeps_them_by_name_and_weight():
    at = SOURCE.index('"judging": {')
    nearby = SOURCE[at : at + 300]
    assert "AS_GOOD_A_GUESS_AS_ANY.get(str(name), 0.0)" in nearby
    assert "for name in INVENTED" in nearby


def test_a_name_that_is_no_longer_in_the_space_is_skipped():
    """The space can change; a record of it must not crash a run."""
    assert measure_named("a property nobody composed") is None
