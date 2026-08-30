"""She may change the language she thinks in. Not the ruler she is measured with.

Both of those are description length, which is the whole difficulty. A word she
made has a short NAME — so counting names, a way of building that hands every
long thing a one-word label collapses every length in the system and every
promotion looks like a triumph. Nothing improved: the ruler moved.

That failure is available to any system that defines both its representations
and the measure of how complicated they are.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition.one_algebra import Term, as_a_maker
from core.cognition.the_ruler_she_cannot_move import (
    what_it_costs_to_be,
    what_the_language_costs_to_be,
)
from core.cognition.what_it_costs_to_say import _symbols

TWICE = Term(
    "through",
    (Term("hole", value=0), Term("through", (Term("hole", value=0), Term("where")))),
)


@pytest.fixture(autouse=True)
def _ways_left_as_found():
    was = dict(kinds.WAYS_TO_BUILD)
    kinds.WAYS_TO_BUILD.clear()
    try:
        yield
    finally:
        kinds.WAYS_TO_BUILD.clear()
        kinds.WAYS_TO_BUILD.update(was)


def test_a_word_she_was_given_costs_one():
    assert what_it_costs_to_be(kinds.WHERE_FROM["the far end"]) == 1


def test_a_word_a_maker_produced_costs_the_maker_plus_its_parts():
    kinds.WAYS_TO_BUILD["a way she wrote: twice"] = as_a_maker(TWICE)
    made = kinds.addressings()
    name = next(one for one in made if "a word (#0)" in one)
    assert what_it_costs_to_be(made[name]) > 1


def test_the_short_name_does_not_make_it_cheap():
    """The gaming move, stated as a test: one symbol to say, six to be."""
    kinds.WAYS_TO_BUILD["a way she wrote: twice"] = as_a_maker(TWICE)
    made = kinds.addressings()
    name = next(one for one in made if "a word (#0)" in one)
    assert _symbols(name) == 1
    assert what_it_costs_to_be(made[name]) > _symbols(name)


def test_admitting_a_maker_never_lowers_what_the_language_costs_to_be():
    given = what_the_language_costs_to_be()
    kinds.WAYS_TO_BUILD["a way she wrote: twice"] = as_a_maker(TWICE)
    assert what_the_language_costs_to_be() >= given


def test_the_certificate_pays_for_a_maker_once_and_uses_it_many_times():
    """Which is what compression actually is, and it needs the reuse to be real."""
    import inspect

    from core.cognition import one_algebra

    source = inspect.getsource(one_algebra)
    at = source.index("What it saves is paid once and used many times")
    nearby = source[at : at + 1200]
    assert "made - 1" in nearby
    assert "term.how_long()" in nearby
