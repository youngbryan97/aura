"""The three sentences hold, and each check can actually fail.

A check that cannot fire reports green forever, which is the way an
invariant becomes decoration. Every test here breaks the mechanism the
invariant protects and asserts the invariant notices.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.cognition import what_a_change_means as M
from core.cognition.an_invented_kind import WHERE_FROM
from core.cognition.what_she_can_take_back import as_it_stands


@pytest.fixture(autouse=True)
def _leave_the_registries_as_found():
    was = as_it_stands()
    yield
    was.restore()


def test_all_three_hold_right_now():
    held = M.what_a_change_means()["holds"]
    assert held == {
        "a refusal changes nothing": True,
        "a promotion can go back": True,
        "a refused proposal does not run": True,
    }, held


def test_the_refusal_check_fires_when_a_trial_stops_taking_itself_back():
    """Break only_if_it_pays, and the invariant must say so."""
    from contextlib import contextmanager

    @contextmanager
    def keeps_everything(what=""):
        class _Kept:
            kept = True

            def keep(self, why=""):
                return None

        yield _Kept()

    with patch(
        "core.cognition.what_she_can_take_back.only_if_it_pays", keeps_everything
    ):
        broken = list(M._a_refusal_changes_nothing())
    assert broken, "the invariant did not notice a trial that keeps everything"
    assert "rejected development" in broken[0].message


def test_the_rollback_check_fires_when_nothing_is_recorded_to_go_back_to():
    """Drop `replaced`, which is exactly how this was broken for its whole life."""
    from core.cognition import how_a_change_is_promoted as P

    real = P.promote

    def forgets_what_it_replaced(at, **kw):
        kw.pop("replaced", None)
        return real(at, **kw)

    with patch.object(P, "promote", forgets_what_it_replaced):
        broken = list(M._a_promotion_can_go_back())
    assert broken, "the invariant did not notice a promotion with nothing to undo"
    assert "nothing to go back to" in broken[0].message


def test_the_rollback_check_fires_when_the_undo_does_not_undo():
    """put_it_back used to hand the value back and leave the state alone."""
    from core.cognition import how_a_change_is_promoted as P

    with patch.object(P, "put_it_back", lambda at: "handed back, not applied"):
        broken = list(M._a_promotion_can_go_back())
    assert broken, "the invariant did not notice an undo that undid nothing"
    assert "cannot be undone" in broken[0].message


def test_the_consent_check_fires_when_a_refusal_reads_as_truthy():
    """The original defect: every object is truthy."""
    from core.self_modification.growth_ladder import ModificationProposal

    from core.self_modification.consent_invariant import (
        _a_refused_proposal_does_not_run,
    )

    with patch.object(ModificationProposal, "__bool__", lambda self: True):
        broken = list(_a_refused_proposal_does_not_run())
    assert len(broken) == 2, [one.message for one in broken]
    assert any("proceeds over a refusal" in one.message for one in broken)
    assert any("the default answer" in one.message for one in broken)


def test_every_check_is_registered_with_the_verifier():
    """An invariant nobody runs is a comment.

    The consent check registers when the ladder module loads, which both boot
    paths do when they construct a GrowthLadder. Importing it here is what a
    boot does, not a convenience.
    """
    import core.self_modification.growth_ladder  # noqa: F401

    from core.verify.invariants import get_registry

    names = {one.name for one in get_registry().specs()}
    for said in (
        "development.a_refusal_changes_nothing",
        "development.a_promotion_can_go_back",
        "development.a_refused_proposal_does_not_run",
    ):
        assert said in names, f"{said} is not registered; {sorted(names)[:5]}"


def test_reading_the_report_does_not_write_into_the_record():
    """A check that reads the evidence chain must not write into it.

    The rollback check has to perform a promotion, and performing one put two
    lines into the receipt chain every time anybody read the health report.
    The chain is what makes "this decision was hers" checkable; filling it
    with the checks that read it is how that gets lost.
    """
    import core.self_modification.growth_ladder  # noqa: F401 - registers the third

    from core.cognition.how_a_change_is_promoted import the_receipts

    before = len(the_receipts())
    for _ in range(5):
        M.what_a_change_means()
    assert len(the_receipts()) == before


def test_the_scoped_ledger_still_chains():
    """The check must exercise the real mechanism, not a stub that always passes."""
    from core.cognition.how_a_change_is_promoted import (
        a_ledger_of_its_own,
        promote,
        the_chain_holds,
        the_receipts,
    )

    held = len(the_receipts())
    with a_ledger_of_its_own():
        promote("a/b", became="shadow", started_by="she", evidence="one")
        promote("a/c", became="shadow", started_by="she", evidence="two")
        assert len(the_receipts()) == 2
        assert the_chain_holds()
    assert len(the_receipts()) == held


def test_a_sentence_whose_check_is_missing_does_not_read_as_holding():
    """"Not registered" and "holds" are different readings, and only one is true.

    If a boot ever stops loading the module that owns a check, the report has
    to say the check is not running rather than say the rule holds.
    """
    with patch.dict(
        M.WHAT_A_CHANGE_MEANS,
        {"a rule nobody registered": "development.no_such_invariant"},
    ):
        held = M.what_a_change_means()["holds"]
    assert held["a rule nobody registered"] == ["not registered, so nobody runs it"]
