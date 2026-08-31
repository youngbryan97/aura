"""A maker that fits what it was made from has passed the test a table passes.

What it has to earn is a place in the language she thinks in from now on, and
every word it makes is another branch at every step of every search. So it is
weighed on evidence the synthesis did not see, and the terms are all counted in
one unit — expressions she would otherwise have to walk — so the weights that
usually carry the argument are forced to one.
"""

from __future__ import annotations

import pytest

from core.cognition.is_it_worth_keeping import what_it_is_worth


def _weigh(**over):
    settings = dict(
        now_sayable=lambda family: True,
        held_out=[("a family",)],
        was_sayable=(False,),
        vocabulary_before=5,
        vocabulary_after=25,
        longest=4,
        shorter_by=2,
        used=3,
    )
    settings.update(over)
    return what_it_is_worth(**settings)


def test_reaching_a_family_it_was_not_shown_is_worth_the_search_it_saves():
    worth = _weigh()
    assert worth.reaches > 0
    assert worth.keep_it


def test_a_family_that_was_already_sayable_buys_nothing():
    worth = _weigh(was_sayable=(True,))
    assert worth.reaches == 0


def test_a_maker_that_reaches_nothing_new_and_costs_a_lot_is_refused():
    worth = _weigh(
        now_sayable=lambda family: False,
        shorter_by=0,
        used=0,
        vocabulary_after=200,
    )
    assert worth.reaches == 0
    assert worth.shortens == 0
    assert not worth.keep_it


def test_every_term_is_in_the_same_unit_so_nothing_is_weighted():
    """The sum is reaches + shortens - costs, with no coefficient anywhere."""
    worth = _weigh()
    assert worth.worth == worth.reaches + worth.shortens - worth.costs


def test_it_says_how_many_it_was_weighed_on_so_a_zero_can_be_read():
    assert _weigh(held_out=[]).tried == 0
    assert _weigh(held_out=[("one",), ("two",)], was_sayable=(False, False)).tried == 2


def test_the_maker_path_weighs_before_it_keeps():
    import inspect

    from core.cognition import one_algebra

    source = inspect.getsource(one_algebra)
    assert "_earns_its_place(term, transitions" in source
    assert "WAYS_TO_BUILD.pop(name, None)" in source


def test_a_family_too_small_to_hold_anything_back_is_not_refused_for_it():
    """Nothing held back is not evidence against."""
    from core.cognition.one_algebra import Term, _earns_its_place

    assert _earns_its_place(Term("where"), [((1, 2), (2, 1))], 5)


def test_the_budget_covers_admitting_a_candidate_and_not_only_finding_one():
    """It was checked once per term, which bounds the search but not one
    candidate's admission.

    Putting a maker into the language rebuilds every word it makes and weighs
    the result, and a maker that makes hundreds takes minutes on its own. A
    thirty-second budget ran for over ten minutes, all of it inside a single
    iteration the check at the top had already passed.
    """
    import time

    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.one_algebra import a_maker_she_wrote

    shown = [
        (one, tuple(reversed(one)))
        for one in [(1, 2, 3, 4), (5, 6, 7, 8), (1, 2, 3, 4, 5), (6, 7, 8, 9, 1)]
    ]
    assert WHERE_FROM  # the language she was given is there
    began = time.monotonic()
    a_maker_she_wrote(shown, now_sayable=lambda: False, holes=2, within=5.0)
    assert time.monotonic() - began < 60.0
