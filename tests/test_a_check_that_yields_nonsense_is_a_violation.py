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


@pytest.mark.parametrize("module, name", [
    ("core.learning.semantic_program_feature_materialization", "learning.stop_contrast_selection_keeps_both_sides"),
    ("core.learning.semantic_native_fit_sampling", "learning.native_pilot_donors_stay_in_fit"),
    ("core.learning.semantic_candidate_union", "learning.mixed_candidate_agreement_is_not_truth"),
])
def test_the_three_that_returned_a_dict_are_clean_by_yielding_nothing(module, name):
    """Each returned a dict when it held, which verify read as its keys: one breach per key, and before that change a
    crash of the whole pass (test_the_ledger_hears_from_all_of_her, after test_train_semantic_native_program)."""
    import importlib

    importlib.import_module(module)
    spec = next(one for one in invariants.get_registry().specs() if one.name == name)
    assert list(spec.check() or ()) == []
    assert not [v for v in invariants.verify(spec.scope, record=False).violations if v.invariant == name]


def test_the_mixed_agreement_claim_still_reads_its_receipt():
    from core.learning.semantic_candidate_union import mixed_agreement_receipt
    from core.organism.claims_computational_knowledge import _mixed_program_agreement_is_not_truth

    assert mixed_agreement_receipt()["serving_authority"] is False
    assert _mixed_program_agreement_is_not_truth()
