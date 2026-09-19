"""An instrument with nothing to measure must not report success.

Three tests in the validation suite scored a clean zero over an empty set:
lockdep counted 0 splats across 0 known locks, the rate-group test took
``max([])`` of 0 registered groups, and the health test counted 0
unresponsive components out of 0 registered. All three passed. All three
were measuring nothing.

Zero-over-zero is the failure mode this repository keeps rediscovering —
the absence of a check reported as a passed check — and it is worst here,
because these three back registered claims about the runtime.
"""
from __future__ import annotations

import pytest

from core.organism import model_validation as mv
from core.organism.model_validation import NothingMeasured, Outcome


def _suite():
    mv.install_runtime_validation()
    return mv.get_suite()


def _test_named(name: str):
    for test in _suite().tests():
        if test.name == name:
            return test
    raise AssertionError(f"{name} is no longer in the runtime suite")


class _Model:
    name = "probe"

    def __init__(self, *capabilities: str) -> None:
        self._capabilities = set(capabilities)

    def capabilities(self) -> set[str]:
        return set(self._capabilities)


class TestLockdep:
    def test_no_known_locks_is_an_error_not_a_pass(self, monkeypatch):
        monkeypatch.setattr(
            mv, "_lockdep", lambda: {"known_locks": [], "acquires_checked": 0, "splats": []}
        )
        with pytest.raises(NothingMeasured, match="knows 0 locks"):
            mv._lockdep_splats()

    def test_known_locks_that_were_never_acquired_is_an_error(self, monkeypatch):
        monkeypatch.setattr(
            mv,
            "_lockdep",
            lambda: {"known_locks": ["a", "b"], "acquires_checked": 0, "splats": []},
        )
        with pytest.raises(NothingMeasured, match="observed 0"):
            mv._lockdep_splats()

    def test_a_real_clean_process_still_passes(self, monkeypatch):
        monkeypatch.setattr(
            mv,
            "_lockdep",
            lambda: {"known_locks": ["a"], "acquires_checked": 12, "splats": []},
        )
        assert mv._lockdep_splats() == 0

    def test_the_suite_reports_not_measured_rather_than_pass(self, monkeypatch):
        monkeypatch.setattr(
            mv, "_lockdep", lambda: {"known_locks": [], "acquires_checked": 0, "splats": []}
        )
        result = _test_named("lockdep_reports_no_order_violations").run(
            _Model("lock_ordering")
        )
        assert result.score.outcome is Outcome.NOT_MEASURED
        assert result.score.outcome is not Outcome.PASS
        assert "knows 0 locks" in result.score.interpretation

    def test_an_unmeasured_instrument_leaves_its_claim_unsupported(self, monkeypatch):
        """The point of the whole exercise: a claim standing on nothing says so."""
        monkeypatch.setattr(
            mv, "_lockdep", lambda: {"known_locks": [], "acquires_checked": 0, "splats": []}
        )
        suite = _suite()
        # The boot posture: the claim is about an instrument, and running the
        # suite's experiments as well took these tests past five minutes.
        suite.run(include_expensive=False)
        unsupported = {c["test"] for c in suite.unsupported_claims()}
        assert "lockdep_reports_no_order_violations" in unsupported

    def test_a_measured_instrument_supports_its_claim(self, monkeypatch):
        monkeypatch.setattr(
            mv,
            "_lockdep",
            lambda: {"known_locks": ["a"], "acquires_checked": 40, "splats": []},
        )
        suite = _suite()
        # The boot posture: the claim is about an instrument, and running the
        # suite's experiments as well took these tests past five minutes.
        suite.run(include_expensive=False)
        unsupported = {c["test"] for c in suite.unsupported_claims()}
        assert "lockdep_reports_no_order_violations" not in unsupported


class TestRateGroups:
    def test_no_group_has_completed_a_cycle_is_an_error(self, monkeypatch):
        import core.fsw.rate_groups as rg

        monkeypatch.setattr(rg, "rate_group_report", lambda: {"groups": []})
        with pytest.raises(NothingMeasured, match="no rate to compare"):
            mv._slowest_group_fraction()

    def test_groups_registered_but_never_run_is_an_error(self, monkeypatch):
        import core.fsw.rate_groups as rg

        monkeypatch.setattr(
            rg,
            "rate_group_report",
            lambda: {"groups": [{"p50_ms": 0.0, "period_ms": 1000, "cycles": 0}]},
        )
        with pytest.raises(NothingMeasured):
            mv._slowest_group_fraction()

    def test_a_running_group_is_measured(self, monkeypatch):
        import core.fsw.rate_groups as rg

        monkeypatch.setattr(
            rg,
            "rate_group_report",
            lambda: {"groups": [{"p50_ms": 250.0, "period_ms": 1000, "cycles": 9}]},
        )
        assert mv._slowest_group_fraction() == pytest.approx(0.25)


class TestHealthChecker:
    def test_watching_nothing_is_an_error_not_a_pass(self, monkeypatch):
        monkeypatch.setattr(
            mv,
            "_health",
            lambda: {"watched": 0, "rounds": 0, "critical_unresponsive": []},
        )
        with pytest.raises(NothingMeasured, match="watches 0 components"):
            mv._critical_unresponsive()

    def test_components_never_pinged_is_an_error(self, monkeypatch):
        monkeypatch.setattr(
            mv,
            "_health",
            lambda: {"watched": 5, "rounds": 0, "critical_unresponsive": []},
        )
        with pytest.raises(NothingMeasured, match="0 ping rounds"):
            mv._critical_unresponsive()

    def test_a_pinged_healthy_runtime_still_passes(self, monkeypatch):
        monkeypatch.setattr(
            mv,
            "_health",
            lambda: {"watched": 5, "rounds": 3, "critical_unresponsive": []},
        )
        assert mv._critical_unresponsive() == 0

    def test_a_wedged_component_is_counted(self, monkeypatch):
        monkeypatch.setattr(
            mv,
            "_health",
            lambda: {"watched": 5, "rounds": 3, "critical_unresponsive": ["cortex"]},
        )
        assert mv._critical_unresponsive() == 1


def test_the_lock_ordering_claim_states_its_own_scope():
    """The claim must not read as covering locks lockdep cannot see.

    Lockdep instruments a minority of this runtime's locks, and
    capability_engine was wrapped only after it deadlocked the boot path. An
    unqualified "the runtime has no latent ABBA deadlock" overstates what the
    zero means.
    """
    claims = [
        c
        for c in _suite().claims()
        if c.test == "lockdep_reports_no_order_violations"
    ]
    assert claims, "the lock-ordering claim is no longer registered"
    statement = claims[0].statement
    assert "instruments" in statement or "checked_lock" in statement, statement
