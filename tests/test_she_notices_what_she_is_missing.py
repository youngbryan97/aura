"""She notices when the gap is in what she CAN do, not in what she did.

Twelve of the thirteen readings are about what happened — what recurs, what
costs, what sits idle. None said "the set of operators has a hole in it",
which is the only reading whose answer is "write a new kind of action". An
external review named that as the first arrow of the closure loop, and it
was the one genuinely missing.

The distinction it turns on is one ``silence`` cannot make: silence counts
occasions where nothing could be said, and that happens both when she never
tried and when she tried everything. Only the second is a gap. Telling them
apart needed the record to say what was TRIED, which it did not — ``route``
names the action only when the change was kept.
"""

from __future__ import annotations

import pytest

from core.cognition.the_record_of_her_own_work import (
    forget_the_record,
    note_an_episode,
)
from core.cognition.what_she_could_do_next import (
    WHAT_SHE_COULD_DO,
    what_she_could_do,
)
from core.cognition.what_she_notices_about_herself import (
    THE_READINGS,
    nothing_she_has,
    what_she_notices,
)


@pytest.fixture
def three_actions_and_three_families():
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
    for at in range(8):
        note_an_episode(
            "everything_failed", route=None, walked=900, tried=f"a probe called {at % 3}"
        )
        note_an_episode("never_tried", route=None, walked=900)
        note_an_episode(
            "something_worked",
            route="a probe called 0",
            walked=900,
            tried="a probe called 0",
        )
    yield
    forget_the_record()
    WHAT_SHE_COULD_DO.clear()
    WHAT_SHE_COULD_DO.update(held)


def test_it_names_only_the_family_that_beat_everything(
    three_actions_and_three_families,
):
    assert {one.about for one in nothing_she_has()} == {"everything_failed"}


def test_a_family_she_never_tried_is_not_a_gap_in_what_she_can_do(
    three_actions_and_three_families,
):
    """Never trying calls for the action she has, not for a new one."""
    assert "never_tried" not in {one.about for one in nothing_she_has()}


def test_a_family_something_worked_on_is_not_a_gap(
    three_actions_and_three_families,
):
    assert "something_worked" not in {one.about for one in nothing_she_has()}


def test_it_says_what_it_wants(three_actions_and_three_families):
    """A reading that cannot say what would answer it is a complaint."""
    found = nothing_she_has()[0]
    assert found.evidence["what it wants"] == "an action of a kind she does not have"
    assert found.evidence["of"] == 3
    assert len(found.evidence["tried"]) == 3


def test_with_nothing_registered_it_says_nothing(monkeypatch):
    """No actions is not a gap in the actions; it is a system before boot."""
    held = dict(WHAT_SHE_COULD_DO)
    WHAT_SHE_COULD_DO.clear()
    try:
        assert nothing_she_has() == []
    finally:
        WHAT_SHE_COULD_DO.update(held)


def test_it_is_one_of_the_readings(three_actions_and_three_families):
    """A reading nobody runs is a function."""
    assert THE_READINGS["nothing she has"] is nothing_she_has
    assert any(one.noticed_by == "nothing she has" for one in what_she_notices())


def test_the_record_says_what_was_tried_and_not_only_what_worked():
    """The field the reading rests on, held on its own."""
    forget_the_record()
    made = note_an_episode("f", route=None, walked=1, tried="an action that failed")
    assert made.tried == "an action that failed"
    assert made.route is None
    kept = note_an_episode("f", route="an action that worked", walked=1)
    assert kept.tried == "an action that worked", (
        "an episode that named a route must count as having tried it"
    )
    forget_the_record()


def _a_family(name: str, *, over: int, tried: int, has: int) -> float:
    """Strength for a family met `over` times that defeated `tried` of `has`."""
    from core.cognition.the_record_of_her_own_work import (
        forget_the_record,
        note_an_episode,
    )
    from core.cognition.what_she_could_do_next import (
        WHAT_SHE_COULD_DO,
        what_she_could_do,
    )
    from core.cognition.what_she_notices_about_herself import nothing_she_has

    held = dict(WHAT_SHE_COULD_DO)
    WHAT_SHE_COULD_DO.clear()
    forget_the_record()
    try:
        for at in range(has):
            what_she_could_do(
                f"an action called {at}",
                over="the words",
                kind="a probe",
                do_it=lambda situation=None: None,
                needs_a_case=False,
            )
        for at in range(over):
            note_an_episode(
                name, route=None, walked=900, tried=f"an action called {at % tried}"
            )
        found = [one for one in nothing_she_has() if one.about == name]
        return found[0].strength if found else 0.0
    finally:
        forget_the_record()
        WHAT_SHE_COULD_DO.clear()
        WHAT_SHE_COULD_DO.update(held)


def test_more_episodes_is_stronger_evidence():
    """The claim the docstring makes, held against the arithmetic.

    The first version multiplied the share by the episode count and divided
    by it again inside ``_share``, so the count cancelled and a family met
    twice scored exactly what one met forty times scored.
    """
    twice = _a_family("met_twice", over=2, tried=2, has=2)
    forty = _a_family("met_forty", over=40, tried=2, has=2)
    assert forty > twice, f"forty episodes ({forty}) is not stronger than two ({twice})"


def test_defeating_more_of_the_toolkit_is_stronger_evidence():
    one_of_four = _a_family("beat_one", over=20, tried=1, has=4)
    four_of_four = _a_family("beat_four", over=20, tried=4, has=4)
    assert four_of_four > one_of_four


def test_strength_stays_inside_its_range():
    assert 0.0 <= _a_family("f", over=200, tried=3, has=3) <= 1.0
    assert 0.0 <= _a_family("g", over=1, tried=1, has=9) <= 1.0


def test_the_evidence_says_the_share_and_the_width():
    """A reading that cannot be checked is a reading that has to be trusted."""
    from core.cognition.the_record_of_her_own_work import (
        forget_the_record,
        note_an_episode,
    )
    from core.cognition.what_she_could_do_next import (
        WHAT_SHE_COULD_DO,
        what_she_could_do,
    )
    from core.cognition.what_she_notices_about_herself import nothing_she_has

    held = dict(WHAT_SHE_COULD_DO)
    WHAT_SHE_COULD_DO.clear()
    forget_the_record()
    try:
        for at in range(2):
            what_she_could_do(
                f"an action called {at}",
                over="the words",
                kind="a probe",
                do_it=lambda situation=None: None,
                needs_a_case=False,
            )
        for at in range(30):
            note_an_episode(
                "a_family", route=None, walked=900, tried=f"an action called {at % 2}"
            )
        found = nothing_she_has()[0]
        assert found.evidence["share"] == 1.0
        assert found.evidence["how_far_out"] > 0
        assert found.evidence["over"] == 30
    finally:
        forget_the_record()
        WHAT_SHE_COULD_DO.clear()
        WHAT_SHE_COULD_DO.update(held)
