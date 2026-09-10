"""One fault repeating is not a world getting steadily worse.

Observed live on 2026-09-07: a morphogenesis population sync proposed
attachments that its own degree budget refused, so the identical transaction
failed once a second for the life of the process. Each failure was recorded as
a fresh one, and forty-three of them in half a minute drove frustration and
depletion to 1.00. Every reply after that was written from saturation caused by
an internal bookkeeping error nobody had told her about.

Two defects, and both are held here: the proposer must respect the budget that
judges it, and a repeat of a fact already held must not move the state as far
as a new one.
"""

from __future__ import annotations

import pytest

from core.soma.resilience_engine import ResilienceEngine


@pytest.fixture
def engine() -> ResilienceEngine:
    return ResilienceEngine()


def test_forty_repeats_of_one_fault_do_not_saturate(engine: ResilienceEngine) -> None:
    for _ in range(43):
        engine.record_failure(
            domain="degradation:morphogenesis.runtime",
            severity=0.55,
            stakes=0.60,
            signature="degradation:morphogenesis.runtime:out-degree budget exceeded",
        )
    assert engine.profile.frustration < 1.0, "one repeating fault saturated frustration"
    assert engine.profile.depletion < 1.0, "one repeating fault saturated depletion"


def test_the_same_count_of_distinct_faults_moves_the_state_further(
    engine: ResilienceEngine,
) -> None:
    """Habituation must not become deafness: novelty still lands."""
    repeated = ResilienceEngine()
    for _ in range(20):
        repeated.record_failure("d", 0.55, 0.60, signature="one-standing-fault")
    for index in range(20):
        engine.record_failure("d", 0.55, 0.60, signature=f"distinct-{index}")
    assert engine.profile.frustration > repeated.profile.frustration


def test_the_first_occurrence_is_undamped(engine: ResilienceEngine) -> None:
    baseline = ResilienceEngine()
    baseline.record_failure("d", 0.5, 0.5)
    engine.record_failure("d", 0.5, 0.5, signature="named")
    assert engine.profile.frustration == pytest.approx(baseline.profile.frustration)


def test_success_makes_the_next_failure_land_in_full(engine: ResilienceEngine) -> None:
    for _ in range(10):
        engine.record_failure("planning", 0.5, 0.5, signature="planning:same")
    engine.record_success("planning")
    before = engine.profile.frustration
    engine.record_failure("planning", 0.5, 0.5, signature="planning:same")
    fresh = ResilienceEngine()
    fresh.record_failure("planning", 0.5, 0.5, signature="planning:same")
    assert engine.profile.frustration - before == pytest.approx(
        fresh.profile.frustration
    )


def test_what_is_repeating_is_reportable(engine: ResilienceEngine) -> None:
    for _ in range(5):
        engine.record_failure("d", 0.5, 0.5, signature="loud")
    engine.record_failure("d", 0.5, 0.5, signature="quiet")
    state = engine.repetition_state()
    assert state.get("loud") == 5
    assert "quiet" not in state


def test_the_signature_table_stays_bounded(engine: ResilienceEngine) -> None:
    for index in range(ResilienceEngine.MAX_TRACKED_SIGNATURES * 2):
        engine.record_failure("d", 0.1, 0.1, signature=f"s{index}")
    assert len(engine._repeats) <= ResilienceEngine.MAX_TRACKED_SIGNATURES


def test_an_unsigned_caller_habituates_on_domain_and_magnitude(
    engine: ResilienceEngine,
) -> None:
    """A caller that names nothing is read as narrowly as it can be."""
    for _ in range(40):
        engine.record_failure("tool_execution", 0.5, 0.5)
    assert engine.profile.frustration < 1.0


def test_two_magnitudes_in_one_domain_are_two_facts() -> None:
    """Habituation may not merge failures the caller reported differently."""
    engine = ResilienceEngine()
    engine.record_failure("planning", 0.5, 0.5)
    first = engine.profile.frustration
    engine.record_failure("planning", 0.9, 1.0)
    second = engine.profile.frustration - first
    fresh = ResilienceEngine()
    fresh.record_failure("planning", 0.9, 1.0)
    assert second == pytest.approx(fresh.profile.frustration)
