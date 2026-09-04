"""The whole loop, run end to end, in the order an external review named it.

    detect that a developmental concept is missing
    construct a new developmental operator
    construct the installer that puts it where it goes
    try it transactionally
    judge it on problems it was not chosen for
    promote it, or put the state back exactly
    make it available to the next developmental search

Aura had pieces of nearly every arrow and not the whole loop. What was
missing was the first (nothing said "the set of operators has a hole"), part
of the third (three of seven destinations had no installer), the sixth (no
promotion ever recorded what it replaced), and the arrow from the first to
the second — nothing turned a named gap into a new developmental action.

These tests run the loop rather than assert that its parts exist.
"""

from __future__ import annotations

import pytest

from core.cognition.an_action_she_writes_for_a_gap import (
    a_gap_she_could_fill,
    offer_writing_an_action_for_a_gap,
)
from core.cognition.how_a_change_is_promoted import (
    a_ledger_of_its_own,
    put_it_back,
    what_it_replaced,
)
from core.cognition.the_record_of_her_own_work import (
    forget_the_record,
    note_an_episode,
)
from core.cognition.what_she_can_take_back import as_it_stands, only_if_it_pays
from core.cognition.what_she_could_do_next import (
    WHAT_SHE_COULD_DO,
    the_actions_she_has,
    what_she_could_do,
)
from core.cognition.what_she_notices_about_herself import nothing_she_has

_THE_WRITER = "write an action for what nothing she has can do"


@pytest.fixture
def a_family_that_beat_everything():
    """Three actions, and a family that defeated all three ten times."""
    was = as_it_stands()
    held = dict(WHAT_SHE_COULD_DO)
    WHAT_SHE_COULD_DO.clear()
    forget_the_record()
    for at in range(3):
        what_she_could_do(
            f"a probe called {at}",
            over="the words",
            kind="a probe",
            do_it=lambda situation=None: None,
            needs_a_case=False,
        )
    for at in range(10):
        note_an_episode(
            "a_family_that_beats_her",
            route=None,
            walked=900,
            tried=f"a probe called {at % 3}",
        )
    offer_writing_an_action_for_a_gap()
    yield
    forget_the_record()
    was.restore()
    WHAT_SHE_COULD_DO.clear()
    WHAT_SHE_COULD_DO.update(held)


def test_the_gap_is_detected(a_family_that_beat_everything):
    assert {one.about for one in nothing_she_has()} == {"a_family_that_beats_her"}


def test_the_gap_names_a_family_and_a_place(a_family_that_beat_everything):
    found = a_gap_she_could_fill()
    assert found is not None
    family, where = found
    assert family == "a_family_that_beats_her"
    assert where


def test_a_new_action_is_written_and_the_next_search_can_take_it(
    a_family_that_beat_everything,
):
    """The arrow that closes the loop over itself."""
    before = {one.name for one in the_actions_she_has()}
    said = WHAT_SHE_COULD_DO[_THE_WRITER].do_it(None)
    assert said is not None, "she wrote nothing for a family that beat everything"
    after = {one.name for one in the_actions_she_has()}
    written = after - before
    assert written, "the new action is not in the set the next search reads"
    assert WHAT_SHE_COULD_DO[next(iter(written))].do_it(None) is not None


def test_a_written_action_is_taken_back_with_the_trial_that_wrote_it(
    a_family_that_beat_everything,
):
    """Enlarging what she can do is a change, and an unkept change leaves nothing."""
    before = len(WHAT_SHE_COULD_DO)
    with only_if_it_pays("a trial that is not kept"):
        WHAT_SHE_COULD_DO[_THE_WRITER].do_it(None)
    assert len(WHAT_SHE_COULD_DO) == before


def test_writing_stops_when_every_place_already_has_one(
    a_family_that_beat_everything,
):
    """Nothing left to add is not failure; it is the ranking's turn."""
    for _ in range(10):
        if WHAT_SHE_COULD_DO[_THE_WRITER].do_it(None) is None:
            break
    assert a_gap_she_could_fill() is None


def test_with_no_gap_she_writes_nothing():
    """It must not fire on a family nobody tried, or on one something worked on."""
    was = as_it_stands()
    held = dict(WHAT_SHE_COULD_DO)
    WHAT_SHE_COULD_DO.clear()
    forget_the_record()
    try:
        what_she_could_do(
            "a probe",
            over="the words",
            kind="a probe",
            do_it=lambda situation=None: None,
            needs_a_case=False,
        )
        for _ in range(10):
            note_an_episode("worked", route="a probe", walked=900, tried="a probe")
            note_an_episode("never_tried", route=None, walked=900)
        offer_writing_an_action_for_a_gap()
        assert a_gap_she_could_fill() is None
        assert WHAT_SHE_COULD_DO[_THE_WRITER].do_it(None) is None
    finally:
        forget_the_record()
        was.restore()
        WHAT_SHE_COULD_DO.clear()
        WHAT_SHE_COULD_DO.update(held)


def test_the_written_action_can_be_promoted_and_put_back_exactly(
    a_family_that_beat_everything,
):
    """The last arrow: promote, or restore the state exactly as it stood."""
    from core.cognition.how_a_change_is_promoted import promote

    stood = as_it_stands()
    WHAT_SHE_COULD_DO[_THE_WRITER].do_it(None)
    written = [one for one in the_actions_she_has() if "one she wrote" in one.name]
    assert written
    at = f"{written[0].over}/{written[0].name}"
    with a_ledger_of_its_own():
        promote(
            at,
            became="shadow",
            started_by="she",
            evidence="a test",
            replaced=stood,
        )
        assert what_it_replaced(at) is not None
        put_it_back(at)
    assert not [one for one in the_actions_she_has() if "one she wrote" in one.name]


def test_the_writer_is_registered_where_production_registers_the_others():
    """An action nobody offers is a function."""
    import inspect

    from core.cognition import sequence_induction

    source = inspect.getsource(sequence_induction._register_what_she_could_do)
    assert "offer_writing_an_action_for_a_gap()" in source
