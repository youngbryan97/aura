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

    with patch.object(ModificationProposal, "__bool__", lambda self: True):
        broken = list(M._a_refused_proposal_does_not_run())
    assert len(broken) == 2, [one.message for one in broken]
    assert any("proceeds over a refusal" in one.message for one in broken)
    assert any("the default answer" in one.message for one in broken)


def test_every_check_is_registered_with_the_verifier():
    """An invariant nobody runs is a comment."""
    from core.verify.invariants import get_registry

    names = {one.name for one in get_registry().specs()}
    for said in (
        "development.a_refusal_changes_nothing",
        "development.a_promotion_can_go_back",
        "development.a_refused_proposal_does_not_run",
    ):
        assert said in names, f"{said} is not registered; {sorted(names)[:5]}"
