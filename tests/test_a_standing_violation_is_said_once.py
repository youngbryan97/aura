"""A verifier finding is logged when it appears, not on every pass it stands.

LIVE 2026-09-19: one claim with no evidence this run was logged 130 times in
a morning, once per verification pass, and the neural stream's lines about
it pushed out the ones that were new.
"""
from __future__ import annotations

import logging
import uuid

from core.verify import invariants
from core.verify.invariants import Severity, Violation, invariant, verify


def test_the_same_finding_is_logged_once_and_a_new_one_is_logged(caplog):
    scope = f"test_scope_{uuid.uuid4().hex[:8]}"
    subjects = ["a"]

    @invariant(f"{scope}.standing", scope=scope, severity=Severity.WARNING, owner="test")
    def _standing():
        for subject in subjects:
            yield Violation(subject=subject, message="no evidence this run")

    invariants.reset_verifier_for_test()
    try:
        with caplog.at_level(logging.WARNING, logger=invariants.logger.name):
            verify(scope)
            verify(scope)
            verify(scope)
            said = [r for r in caplog.records if "@ a:" in r.getMessage()]
            assert len(said) == 1

            subjects.append("b")
            report = verify(scope)
            assert len(report.warnings) == 2, "the report still carries every standing finding"
            assert len([r for r in caplog.records if "@ b:" in r.getMessage()]) == 1
            assert len([r for r in caplog.records if "@ a:" in r.getMessage()]) == 1

            subjects.remove("a")
            verify(scope)
            subjects.append("a")
            verify(scope)
            assert len([r for r in caplog.records if "@ a:" in r.getMessage()]) == 2, (
                "a finding that cleared and came back is new again"
            )
    finally:
        registry = invariants.get_registry()
        with registry._lock:
            registry._specs.pop(f"{scope}.standing", None)
        invariants.reset_verifier_for_test()


def test_a_check_that_needs_arguments_is_refused_when_it_is_declared():
    """LIVE 2026-09-19: a proof function taking four arguments was declared an
    invariant, and every verification pass failed it with a TypeError."""
    import pytest

    with pytest.raises(TypeError, match="requires manifest"):

        @invariant("test.needs_arguments", scope="test_needs_arguments", owner="test")
        def _needs(*, manifest):
            return ()
