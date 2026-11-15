"""A check that yields something other than a violation is a broken check, said as one.

Its yield used to be read as a violation's fields outside the check's own try, so a string or a tuple from one
invariant raised AttributeError out of the whole pass: no report, no last results, and every other invariant
unchecked. A verifier that cannot check reports nothing, which is worse than reporting a breach.
"""
from __future__ import annotations

import pytest

from core.verify import invariants
from core.verify.invariants import InvariantRegistry, Severity, Violation

pytestmark = pytest.mark.unit


def test_one_bad_yield_is_its_own_violation_and_the_rest_are_still_checked(monkeypatch):
    registry = InvariantRegistry()
    monkeypatch.setattr(invariants, "_REGISTRY", registry)
    ran: list[str] = []

    @invariants.invariant("a.yields_a_string", scope="test", owner="tests/here")
    def _bad():
        ran.append("bad")
        yield "this is not a violation"

    @invariants.invariant("b.yields_a_real_one", scope="test", owner="tests/here")
    def _good():
        ran.append("good")
        yield Violation(subject="x", message="a real breach")

    report = invariants.verify("test", record=False)
    assert ran == ["bad", "good"] and report.checked == 2
    by_name = {violation.invariant: violation for violation in report.violations}
    assert by_name["a.yields_a_string"].severity is Severity.ERROR
    assert "yielded a str" in by_name["a.yields_a_string"].message
    assert by_name["b.yields_a_real_one"].message == "a real breach"
