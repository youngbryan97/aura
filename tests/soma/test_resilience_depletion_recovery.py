"""Depletion must have a working way down.

LIVE DEFECT, 2026-08-10, found by using the desktop for one working session.
Resilience depletion climbed 0.42 -> 0.97 in about forty minutes of ordinary
use, the engine logged "Execution suppressed due to depletion/exhaustion in
ResilienceEngine", and her replies fell to the degraded composer. Nothing was
broken; she had simply been running.

Depletion had no working way down at all:

  * a routine degradation (severity 0.55, stakes 0.60) adds 0.0495;
  * passive decay is a 4-hour half-life — 0.29% of the current value per
    minute, roughly 0.0014 at d=0.5, some 35x slower than one failure adds;
  * record_rest, the only strong reducer, had no production caller;
  * record_success, whose docstring says it "reduces frustration more than it
    reduces depletion", reduced depletion by exactly zero.
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def engine():
    from core.soma.resilience_engine import ResilienceEngine

    return ResilienceEngine()


def _routine_failures(engine, count: int) -> None:
    """The exact severity/stakes the live runtime recorded, on distinct faults.

    Forty minutes of ordinary use is many different things going wrong, not
    one going wrong forty times. The distinction became load-bearing on
    2026-09-07, when one morphogenesis fault repeating once a second drove
    both axes to 1.00 in under a minute — see
    `test_a_repeating_fault_is_one_standing_fact.py`. The uphill half of the
    ratchet this file guards is about the first case, and it is still real.
    """
    for index in range(count):
        engine.record_failure(
            "continuous_cognition", 0.55, 0.60, signature=f"routine-{index}"
        )


def test_the_same_fault_repeating_is_not_ordinary_use(engine) -> None:
    """The other half of the premise, so neither can drift into the other."""
    for _ in range(20):
        engine.record_failure(
            "continuous_cognition", 0.55, 0.60, signature="one-standing-fault"
        )
    assert engine.profile.depletion < engine.DEPLETION_THRESHOLD


def test_routine_failures_still_deplete(engine) -> None:
    """Guard the premise — the ratchet's uphill half must remain real."""
    from core.soma.resilience_engine import ResilienceState

    _routine_failures(engine, 20)

    assert engine.profile.depletion > engine.DEPLETION_THRESHOLD
    assert engine.profile.state is ResilienceState.DEPLETION


def test_success_relieves_depletion(engine) -> None:
    """The docstring's promise, which the code did not keep."""
    _routine_failures(engine, 20)
    before = engine.profile.depletion

    engine.record_success("planning", stakes=0.6)

    assert engine.profile.depletion < before


def test_success_relieves_frustration_more_than_depletion(engine) -> None:
    """Literally what record_success documents."""
    _routine_failures(engine, 10)
    frustration_before = engine.profile.frustration
    depletion_before = engine.profile.depletion

    engine.record_success("planning", stakes=0.6)

    frustration_relief = frustration_before - engine.profile.frustration
    depletion_relief = depletion_before - engine.profile.depletion

    assert depletion_relief > 0.0
    assert frustration_relief > depletion_relief


def test_recovery_uses_the_engines_own_exchange_rate(engine) -> None:
    """No second, invented constant: relief rides the failure-loading ratio."""
    _routine_failures(engine, 10)
    frustration_before = engine.profile.frustration
    depletion_before = engine.profile.depletion

    engine.record_success("planning", stakes=0.6)

    frustration_relief = frustration_before - engine.profile.frustration
    depletion_relief = depletion_before - engine.profile.depletion
    expected_ratio = engine.DEPLETION_GAIN / engine.FRUSTRATION_GAIN

    assert depletion_relief == pytest.approx(
        frustration_relief * expected_ratio, rel=1e-6
    )


def test_sustained_success_escapes_the_depletion_state(engine) -> None:
    """A working session must be able to recover, not only a restart."""
    from core.soma.resilience_engine import ResilienceState

    _routine_failures(engine, 20)
    assert engine.profile.state is ResilienceState.DEPLETION

    for _ in range(20):
        engine.record_success("planning", stakes=0.6)

    assert engine.profile.state is not ResilienceState.DEPLETION


def test_quiet_time_is_credited_as_rest(engine) -> None:
    """record_rest had no caller anywhere in production."""
    _routine_failures(engine, 20)
    before = engine.profile.depletion

    engine.profile.failure_history[-1].timestamp = time.time() - 1800
    engine.profile.last_rest = time.time() - 3600
    engine._credit_quiet_interval_as_rest()

    assert engine.profile.depletion < before


def test_rest_credit_is_not_paid_twice(engine) -> None:
    """The same quiet seconds must not be banked on every loop tick."""
    _routine_failures(engine, 20)
    engine.profile.failure_history[-1].timestamp = time.time() - 1800
    engine.profile.last_rest = time.time() - 3600

    engine._credit_quiet_interval_as_rest()
    once = engine.profile.depletion
    engine._credit_quiet_interval_as_rest()

    assert engine.profile.depletion == pytest.approx(once)


def test_a_failure_inside_the_window_still_accumulates(engine) -> None:
    """Rest crediting must not quietly cancel live failures."""
    _routine_failures(engine, 20)
    engine.profile.failure_history[-1].timestamp = time.time() - 1800
    engine.profile.last_rest = time.time() - 3600
    engine._credit_quiet_interval_as_rest()
    rested = engine.profile.depletion

    engine.record_failure("continuous_cognition", 0.55, 0.60)

    assert engine.profile.depletion > rested


def test_decay_loop_credits_rest() -> None:
    """The wiring itself — an orphaned reducer is the whole defect."""
    import inspect

    from core.soma.resilience_engine import ResilienceEngine

    source = inspect.getsource(ResilienceEngine._decay_loop)

    assert "_credit_quiet_interval_as_rest" in source
