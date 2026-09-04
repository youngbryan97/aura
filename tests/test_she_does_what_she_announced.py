"""One decision per episode, and the one she announced is the one she does.

The idle loop asked what was worth developing, told the user "I decided to
X", and then called ``she_develops_herself()`` with no argument — which asked
again. The draw is a draw from what the counts support, so the second answer
differed from the announced one in about four episodes out of five, and the
first decision was discarded into a variable named ``_again``.

Both entry points also carried out their own copy of the episode, and the
copies had drifted: only one applied the action's own success test.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.cognition import she_decides_to_develop as S
from core.cognition.the_record_of_her_own_work import note_an_episode
from core.cognition.what_she_could_do_next import (
    WHAT_SHE_COULD_DO,
    what_she_could_do,
)


@pytest.fixture
def a_choice_worth_making():
    """Several unpriced actions and a costly family, so the draw has room."""
    from core.cognition.the_record_of_her_own_work import forget_the_record
    from core.cognition.what_she_could_do_next import WHAT_THEY_HAVE_DONE

    # The record too. Each test adds thirty episodes to the same family, so by
    # the second one the totals have moved far enough that nothing clears the
    # ceiling, both draws come back None, and the null below reports that
    # asking twice always agrees — which would leave the test above it
    # passing while proving nothing.
    forget_the_record()
    held = dict(WHAT_SHE_COULD_DO)
    # The counts too. Without this the previous test's episodes price these
    # actions, the draw stops being a draw, and the null below reports that
    # asking twice always agrees — which would make the test above it vacuous
    # while still passing.
    counted = dict(WHAT_THEY_HAVE_DONE)
    WHAT_THEY_HAVE_DONE.clear()
    WHAT_SHE_COULD_DO.clear()
    for at in range(5):
        what_she_could_do(
            f"a probe called {at}",
            over="the words",
            kind="a probe",
            do_it=lambda situation=None: "it did something",
            needs_a_case=False,
        )
    for _ in range(30):
        note_an_episode("a_costly_family", route=None, walked=9_000, admitted=None)
    yield
    WHAT_SHE_COULD_DO.clear()
    WHAT_SHE_COULD_DO.update(held)
    WHAT_THEY_HAVE_DONE.clear()
    WHAT_THEY_HAVE_DONE.update(counted)
    forget_the_record()


def test_a_decision_handed_in_is_the_decision_carried_out(a_choice_worth_making):
    """The whole defect: she said one thing and did another."""
    for _ in range(40):
        announced = S.what_is_worth_doing_now()
        if announced.action is None:
            continue
        performed, _came = S.she_develops_herself(announced)
        assert performed.action is announced.action, (
            f"announced {announced.action.name}, performed "
            f"{getattr(performed.action, 'name', None)}"
        )


def test_asking_twice_really_does_give_different_answers(a_choice_worth_making):
    """The null for the test above: if the draw never moved, it proves nothing."""
    differed = 0
    for _ in range(60):
        first = getattr(S.what_is_worth_doing_now().action, "name", None)
        second = getattr(S.what_is_worth_doing_now().action, "name", None)
        differed += first != second
    assert differed > 0, (
        "two draws never disagreed, so this fixture cannot detect the defect"
    )


def test_an_action_that_declares_itself_failed_is_not_recorded_as_kept():
    """Only one of the two episode runners applied the success test.

    The decision is handed in rather than drawn, so this asserts every run
    instead of skipping whenever the draw refused.
    """
    held = dict(WHAT_SHE_COULD_DO)
    WHAT_SHE_COULD_DO.clear()
    try:
        action = what_she_could_do(
            "one that says it worked and did not",
            over="the words",
            kind="a probe",
            do_it=lambda situation=None: "looks like it worked",
            succeeded=lambda came: False,
            needs_a_case=False,
        )
        decided = S.Decision(
            action=action,
            worth=None,
            because="chosen",
            grounds="handed in by the test",
        )
        _again, came_of_it = S.she_develops_herself(decided)
        assert came_of_it is None, "a failed action was recorded as kept"

        from core.cognition.what_she_could_do_next import what_it_has_done

        record = what_it_has_done(action.name)
        assert record.taken == 1
        assert record.kept == 0, "a failed action was counted as one that paid"
    finally:
        WHAT_SHE_COULD_DO.clear()
        WHAT_SHE_COULD_DO.update(held)


def test_both_entry_points_run_the_same_episode():
    """One episode, one place — the copies are what drifted."""
    import inspect

    for name in ("she_decides_to_develop", "she_develops_herself"):
        source = inspect.getsource(getattr(S, name))
        assert "_carry_it_out(" in source, f"{name} runs its own copy again"
