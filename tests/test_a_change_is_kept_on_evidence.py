"""Keeping a self-authored change requires evidence, not a returned sentence.

An external review put the missing invariant precisely: promotion should
require E[capability | candidate] > E[capability | incumbent] under an
evaluation the candidate was not optimised against. Individual developmental
mechanisms did something close to that. The generic layer every action passes
through did not — it kept anything that returned a non-None result, so "it
reported doing something" was the whole of the evidence.

Three outcomes now, and the third is the honest one. A change that pays on
held-out families is kept. A change that does not is put back exactly. A
change nobody can build a probe for is kept and COUNTED as unmeasured, rather
than refused (which would stop development on any faculty without a probe) or
called evidence (which would be a lie).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.cognition.an_invented_kind import WHERE_FROM
from core.cognition.what_she_can_take_back import as_it_stands
from core.cognition.what_she_could_do_next import (
    HOW_CHANGES_WERE_JUDGED,
    WHAT_SHE_COULD_DO,
    how_changes_were_judged,
    what_she_could_do,
)


@pytest.fixture(autouse=True)
def _leave_it_as_found():
    was = as_it_stands()
    held = dict(WHAT_SHE_COULD_DO)
    counts = dict(HOW_CHANGES_WERE_JUDGED)
    yield
    was.restore()
    WHAT_SHE_COULD_DO.clear()
    WHAT_SHE_COULD_DO.update(held)
    HOW_CHANGES_WERE_JUDGED.clear()
    HOW_CHANGES_WERE_JUDGED.update(counts)


def _an_action(name: str, *, judges_itself: bool = False):
    def writes(situation=None):
        WHERE_FROM[f"a_word_from_{name}"] = lambda a, b: a
        return f"{name} did something"

    return what_she_could_do(
        name,
        over="the words",
        kind="a test",
        do_it=writes,
        needs_a_case=False,
        judges_itself=judges_itself,
    )


def _probe_says(paid: bool | None):
    """Force the held-out verdict, so the gate is tested and not the probe."""
    return patch(
        "core.cognition.what_she_could_do_next._held_out_says_it_paid",
        lambda name, before, probe: paid,
    )


def test_a_change_that_does_not_pay_is_put_back():
    """The whole defect: returning a sentence used to be enough."""
    action = _an_action("one that does not pay")
    with _probe_says(False):
        said = action.do_it(None)
    assert said is None, "a change that did not pay still reported success"
    assert "a_word_from_one that does not pay" not in WHERE_FROM


def test_a_change_that_pays_is_kept():
    """The gate must not be a way of never keeping anything."""
    action = _an_action("one that pays")
    with _probe_says(True):
        said = action.do_it(None)
    assert said == "one that pays did something"
    assert "a_word_from_one that pays" in WHERE_FROM


def test_a_change_nobody_can_measure_is_kept_and_counted():
    """Not a gate, and not pretending to be one. The number is the point."""
    before = HOW_CHANGES_WERE_JUDGED["unmeasured"]
    action = _an_action("one nobody can measure")
    with _probe_says(None):
        said = action.do_it(None)
    assert said is not None
    assert HOW_CHANGES_WERE_JUDGED["unmeasured"] == before + 1


def test_an_action_that_judges_itself_is_not_judged_twice():
    """Its own held-out test already returned None where it did not pay."""
    action = _an_action("one that judges itself", judges_itself=True)
    calls: list[int] = []
    with patch(
        "core.cognition.what_she_could_do_next._held_out_says_it_paid",
        lambda *a: calls.append(1) or False,
    ):
        said = action.do_it(None)
    assert said is not None, "a self-judging action was overruled by the gate"
    assert not calls, "the generic gate ran a probe the action had already run"


def test_declining_is_counted_apart_from_failing_the_gate():
    """Two ways to leave nothing behind, and they mean different things."""
    action = what_she_could_do(
        "one that declines",
        over="the words",
        kind="a test",
        do_it=lambda situation=None: None,
        needs_a_case=False,
    )
    before = dict(HOW_CHANGES_WERE_JUDGED)
    assert action.do_it(None) is None
    assert HOW_CHANGES_WERE_JUDGED["declined"] == before["declined"] + 1
    assert HOW_CHANGES_WERE_JUDGED["did not pay"] == before["did not pay"]


def test_the_report_says_what_share_of_kept_changes_had_evidence():
    HOW_CHANGES_WERE_JUDGED.update(
        {"held out": 3, "unmeasured": 1, "judged itself": 0,
         "did not pay": 5, "declined": 9}
    )
    said = how_changes_were_judged()
    assert said["kept"] == 4
    assert said["kept_without_evidence"] == 1
    assert said["share_with_evidence"] == 0.75


def test_the_report_is_zero_rather_than_dividing_by_it():
    HOW_CHANGES_WERE_JUDGED.update(
        {"held out": 0, "unmeasured": 0, "judged itself": 0,
         "did not pay": 0, "declined": 0}
    )
    assert how_changes_were_judged()["share_with_evidence"] == 0.0


def test_a_probe_that_raises_is_unmeasured_and_not_a_refusal():
    """A broken probe must not silently revert every change she makes."""
    from core.cognition import what_she_could_do_next as W

    def boom(before, probe):
        raise RuntimeError("the probe is broken")

    with patch(
        "core.cognition.what_she_does_about_herself.worth_keeping", boom
    ):
        assert W._held_out_says_it_paid("x", {}, [("fam", ())]) is None
