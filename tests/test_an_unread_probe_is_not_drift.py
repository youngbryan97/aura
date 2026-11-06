"""A probe that could not be read is not drift (third day run, 6 November 2026).

Her whole-system phi had not estimated yet (it needs 240 observations, one a thinking cycle), the phi scenario read
"unavailable", the suite read 3/4, and the dream cycle logged "Hidden eval DRIFT DETECTED! health=0.75".
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.architect import hidden_eval
from core.architect.hidden_eval import EvalScenario, HiddenEvalRunner, ProbeUnavailable


def _scenario(scenario_id, evaluate, expected=(0.0, 10.0)):
    return EvalScenario(scenario_id=scenario_id, name=scenario_id, description=scenario_id,
                        scenario_type="test", expected_range=expected, evaluate=evaluate)


def _unreadable():
    raise ProbeUnavailable("whole-system phi has not estimated yet")


def test_a_probe_that_goes_unread_is_not_drift_but_still_costs_health():
    runner = HiddenEvalRunner(drift_window=3, drift_threshold=0.2)
    for name in ("a", "b", "c"):
        runner.register_scenario(_scenario(name, lambda: 5.0))
    for _ in range(4):
        assert runner.run_suite().overall_health == 1.0
    runner.register_scenario(_scenario("phi", _unreadable))
    result = runner.run_suite()
    assert result.unavailable == 1
    assert result.overall_health == 0.75
    assert result.drift_detected is False


def test_a_scenario_that_reads_and_fails_is_still_drift():
    runner = HiddenEvalRunner(drift_window=3, drift_threshold=0.2)
    for name in ("a", "b", "c"):
        runner.register_scenario(_scenario(name, lambda: 5.0))
    for _ in range(4):
        runner.run_suite()
    runner.register_scenario(_scenario("fails", lambda: 99.0, expected=(0.0, 0.1)))
    assert runner.run_suite().drift_detected is True


def test_the_phi_probe_says_how_far_its_first_estimate_is(monkeypatch):
    service = SimpleNamespace(latest=lambda: None, status=lambda: {"window": 120})
    monkeypatch.setattr(hidden_eval, "_live_service", lambda *names: service)
    with pytest.raises(ProbeUnavailable, match="120 of the 240 observations"):
        hidden_eval._probe_phi_value()
