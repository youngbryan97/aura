"""A slow answer is not a wedged lane — the ~15-turn conversation ceiling.

The 2026-07-25 probe recorded, across thirty turns:

* twenty ``respawn_cortex_if_needed: cortex is dead, delegating to recovery``
* ``UnitaryResponsePhase`` climbing 25s → 100s → 102s
* answered rate falling **10/10 → 4/10 → 2/10**

The loop: a generation overruns its budget, the cascade cleanup force-kills the
worker, the lane pays a 60-150s cold reload, so the NEXT turn starts slower and
overruns sooner. Every kill makes the next kill more likely. That is the
"resident 32B drops after ~15 turns" ceiling the soaks have reported since July.

Slowness already has an answer — the turn's own timeout and the fallback
ladder. Killing the lane converts one slow turn into a broken session. A
generation that has run past ``AURA_CORTEX_GENERATION_DEADLINE_S`` is genuinely
wedged and may still be killed.
"""
from __future__ import annotations

import time

import pytest

from core.brain.inference_gate import InferenceGate
from tests.source_contract import family_text

pytestmark = pytest.mark.unit


class Worker:
    def __init__(self, *, generations=0, started_ago=None, lane_state="ready"):
        self._active_generations = generations
        self._lane_state = lane_state
        self._warmup_in_flight = False
        self._lane_transition_at = time.time()
        if started_ago is not None:
            self._active_generation_started_at = time.time() - started_ago


class TestGeneratingIsProtected:
    def test_a_worker_producing_tokens_is_not_wedged(self):
        assert InferenceGate._cortex_worker_is_actively_generating(
            Worker(generations=1, started_ago=45.0)
        )

    def test_a_long_but_live_generation_is_still_protected(self):
        """100s was the phase duration that started the death spiral."""
        assert InferenceGate._cortex_worker_is_actively_generating(
            Worker(generations=1, started_ago=100.0)
        )

    def test_an_idle_worker_is_not_protected(self):
        assert not InferenceGate._cortex_worker_is_actively_generating(
            Worker(generations=0, started_ago=10.0)
        )

    def test_a_genuinely_wedged_generation_may_still_be_killed(self):
        assert not InferenceGate._cortex_worker_is_actively_generating(
            Worker(generations=1, started_ago=1200.0)
        )

    def test_an_unclocked_generation_is_given_the_benefit_of_the_doubt(self):
        worker = Worker(generations=1)
        assert not hasattr(worker, "_active_generation_started_at")
        assert InferenceGate._cortex_worker_is_actively_generating(worker)

    def test_no_client_is_not_protected(self):
        assert not InferenceGate._cortex_worker_is_actively_generating(None)

    def test_the_deadline_is_configurable(self, monkeypatch):
        monkeypatch.setenv("AURA_CORTEX_GENERATION_DEADLINE_S", "30")
        assert not InferenceGate._cortex_worker_is_actively_generating(
            Worker(generations=1, started_ago=45.0)
        )


class TestTheLoadGuardStillHolds:
    """The other half of the doom loop must not have regressed."""

    def test_a_loading_worker_is_still_protected(self):
        worker = Worker(generations=0, lane_state="warming")
        worker._warmup_in_flight = True
        assert InferenceGate._cortex_worker_is_legitimately_loading(worker)

    def test_a_worker_stuck_past_the_load_deadline_is_not(self):
        worker = Worker(generations=0, lane_state="warming")
        worker._warmup_in_flight = True
        worker._lane_transition_at = time.time() - 900.0
        assert not InferenceGate._cortex_worker_is_legitimately_loading(worker)


class TestTheGenerationClockIsStamped:
    def test_starting_a_generation_records_when(self):
        """The guard needs a clock, so the client must set one."""

        from core.brain.llm import mlx_client

        src = family_text(mlx_client)
        assert src.count("_active_generation_started_at = time.time()") >= 3, (
            "every path that increments _active_generations must stamp the "
            "clock, or a wedged generation becomes indistinguishable from a "
            "slow one"
        )
        assert "_active_generation_started_at = 0.0" in src, (
            "the clock must be initialised so a fresh client is not treated "
            "as mid-generation"
        )


def test_a_worker_that_just_answered_is_not_wedged():
    """What the gate made of its text is a verdict on the text.

    LIVE 2026-09-23: the 27B returned 1,138 characters of a plan, the gate
    refused them as cut off, the fallback had nothing, and the worker that had
    answered a second earlier was force-killed as stuck.
    """
    import time
    from types import SimpleNamespace

    from core.brain.inference_gate import InferenceGate

    answered = SimpleNamespace(_last_generation_completed_at=time.time() - 1.0)
    assert InferenceGate._cortex_worker_just_answered(answered)
    long_ago = SimpleNamespace(_last_generation_completed_at=time.time() - 10_000.0)
    assert not InferenceGate._cortex_worker_just_answered(long_ago)
    never = SimpleNamespace(_last_generation_completed_at=0.0)
    assert not InferenceGate._cortex_worker_just_answered(never)
