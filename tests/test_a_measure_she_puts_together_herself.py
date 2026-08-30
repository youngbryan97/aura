"""Everything she judges a situation by was written down by somebody.

Nearness to what was asked for, room left to act in, how much runs in order,
whether her stated line still holds — and, added by hand on 2026-08-29 after a
weight sweep, how near neighbouring things are in value. That last one doubled
her play, and a person found it, wrote its arithmetic, ran the experiment and
admitted it to her evaluator.

The loop underneath could learn weights, rules, models and strategies. It could
not look at what it fails to explain and propose a new property.

This pins the space it would search: a small algebra whose closure contains the
authored measures, so that a measure is composed rather than chosen — and
neither of the two a person wrote is in it by name.
"""

from __future__ import annotations

import pytest

from core.agency.inventing_a_measure import (
    AT,
    OF,
    SUMMED,
    Measure,
    every_measure,
    measure_named,
)
from core.perception.what_is_there import arranged


def board(rows):
    return arranged([
        (0.2 + r * 0.15, 0.2 + c * 0.15, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ])


TIDY = board([["2", "4", "8", "16"], ["4", "8", "16", "32"],
              ["8", "16", "32", "64"], ["16", "32", "64", "128"]])
SCATTERED = board([["128", "2", "32", "4"], ["8", "64", "4", "16"],
                   ["16", "8", "64", "2"], ["32", "4", "16", "8"]])


# ── the authored measures are points in the space ────────────────────────

def test_the_property_a_person_added_is_a_composition():
    """Neighbouring pairs, the gap between them, averaged, read upside down."""
    smoothness = Measure("neighbours", "the gap between them", "on average", True)
    assert smoothness in set(every_measure())
    assert smoothness.read(TIDY) > smoothness.read(SCATTERED)


def test_and_so_is_the_one_that_was_there_before_it():
    order = Measure("along a line", "whether it is in order", "how many hold")
    assert order in set(every_measure())
    assert order.read(TIDY) > order.read(SCATTERED)


def test_neither_is_in_the_space_by_name():
    """The space is an algebra, not a list somebody remembered to extend."""
    import inspect

    from core.agency import inventing_a_measure

    source = inspect.getsource(inventing_a_measure)
    code = [line for line in source.splitlines() if not line.strip().startswith("#")]
    assert not any("smoothness" in line for line in code)


def test_the_space_is_the_product_of_its_parts():
    assert len(list(every_measure())) == len(AT) * len(OF) * len(SUMMED) * 2


# ── every measure is well behaved ────────────────────────────────────────

@pytest.mark.parametrize("measure", list(every_measure()))
def test_a_measure_reads_between_nought_and_one(measure):
    assert 0.0 <= measure.read(TIDY) <= 1.0
    assert 0.0 <= measure.read(SCATTERED) <= 1.0


@pytest.mark.parametrize("measure", list(every_measure())[:20])
def test_nothing_is_read_without_falling_over(measure):
    assert measure.read(arranged([])) == 0.0
    assert measure.read(None) == 0.0


def test_reading_a_thing_of_words_is_not_an_error():
    words = arranged([(0.2, 0.2, "Mon"), (0.2, 0.35, "Tue")])
    for measure in list(every_measure())[:12]:
        assert 0.0 <= measure.read(words) <= 1.0


# ── and each one can be named and found again ────────────────────────────

def test_a_measure_can_be_found_by_the_name_it_gives_itself():
    smoothness = Measure("neighbours", "the gap between them", "on average", True)
    assert measure_named(smoothness.name) == smoothness


def test_a_name_nobody_composed_finds_nothing():
    assert measure_named("the vibe of it") is None


def test_the_other_way_up_is_the_same_fact_read_upside_down():
    gap = Measure("neighbours", "the gap between them", "on average")
    close = Measure("neighbours", "the gap between them", "on average", True)
    assert gap.read(TIDY) + close.read(TIDY) == pytest.approx(1.0)
