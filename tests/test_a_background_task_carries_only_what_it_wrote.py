"""The ALife background lane must carry its own writes and nothing else.

Each background task gets a derived state, and a derived state holds a copy of
every response modifier the turn had at that moment. The queue drains one tick
later. When the whole copy was merged back, every modifier a phase had written
in between was reverted to the value it held a tick ago, and a value that only
one phase ever writes was pinned at its default forever.

The defect was live: ``ontogenetic_novelty`` read 0.5 on every turn of a
300-round recording while the affect phase was writing 0.22, 0.29, 0.45.
"""

import asyncio

import numpy as np

from core.phases.cognitive_integration_phase import (
    CognitiveIntegrationPhase,
    _modifier_delta,
    _modifier_snapshot,
)
from core.state.aura_state import AuraState


class DummyKernel:
    cycle_count = 1


async def _drain_background_tasks() -> None:
    """Let the phase's background tasks finish before reading the queue."""
    for _ in range(50):
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if not pending:
            return
        await asyncio.wait(pending, timeout=2.0)


def _phase_with_background_write(**writes):
    phase = CognitiveIntegrationPhase(DummyKernel())

    async def _write(state):
        state.response_modifiers.update(writes)

    phase._run_criticality = _write
    return phase


# ── the helpers ──────────────────────────────────────────────────────────


def test_an_untouched_scalar_is_not_in_the_delta():
    before = _modifier_snapshot({"ontogenetic_novelty": 0.5, "energy": 0.8})
    after = {"ontogenetic_novelty": 0.5, "energy": 0.8}
    assert _modifier_delta(before, after) == {}


def test_a_changed_scalar_is_in_the_delta():
    before = _modifier_snapshot({"criticality_score": 0.1})
    after = {"criticality_score": 0.9}
    assert _modifier_delta(before, after) == {"criticality_score": 0.9}


def test_a_new_key_is_in_the_delta():
    before = _modifier_snapshot({})
    after = {"entropy": 1.25}
    assert _modifier_delta(before, after) == {"entropy": 1.25}


def test_an_append_to_a_list_under_a_key_is_seen():
    modifiers = {"cognitive_integration_degraded": [{"method": "a"}]}
    before = _modifier_snapshot(modifiers)
    modifiers["cognitive_integration_degraded"].append({"method": "b"})
    delta = _modifier_delta(before, modifiers)
    assert list(delta) == ["cognitive_integration_degraded"]
    assert len(delta["cognitive_integration_degraded"]) == 2


def test_a_write_into_a_nested_dict_is_seen():
    modifiers = {"scope_binding": {"scope": "reasoning_only"}}
    before = _modifier_snapshot(modifiers)
    modifiers["scope_binding"]["scope"] = "governed_actions"
    assert _modifier_delta(before, modifiers) == {
        "scope_binding": {"scope": "governed_actions"}
    }


def test_an_array_whose_equality_is_not_one_answer_is_carried():
    modifiers = {"coupling": np.zeros(4)}
    before = _modifier_snapshot(modifiers)
    delta = _modifier_delta(before, modifiers)
    assert list(delta) == ["coupling"]


def test_a_value_no_copy_can_be_made_of_is_carried():
    lock = asyncio.Lock()
    modifiers = {"handle": lock}
    before = _modifier_snapshot(modifiers)
    assert _modifier_delta(before, modifiers) == {"handle": lock}


def test_the_snapshot_does_not_share_a_container_with_its_source():
    modifiers = {"log": ["one"]}
    snapshot = _modifier_snapshot(modifiers)
    assert snapshot["log"] is not modifiers["log"]
    assert snapshot["log"] == ["one"]


# ── the lane ─────────────────────────────────────────────────────────────


def test_the_queue_holds_only_what_the_task_wrote():
    async def run():
        phase = _phase_with_background_write(criticality_score=0.9)
        state = AuraState()
        state.response_modifiers["ontogenetic_novelty"] = 0.5
        await phase.execute(state)
        await _drain_background_tasks()
        return phase._pending_deltas

    deltas = asyncio.run(run())
    assert deltas, "the background task wrote a modifier and queued nothing"
    for delta in deltas:
        assert "ontogenetic_novelty" not in delta
    assert {"criticality_score": 0.9} in deltas


def test_a_task_that_wrote_nothing_queues_nothing():
    async def run():
        phase = _phase_with_background_write()
        state = AuraState()
        state.response_modifiers["ontogenetic_novelty"] = 0.5
        await phase.execute(state)
        await _drain_background_tasks()
        return phase._pending_deltas

    assert asyncio.run(run()) == []


def test_the_drain_does_not_revert_what_an_earlier_phase_wrote():
    """The regression itself: a per-turn value must survive the next tick."""

    async def run():
        phase = _phase_with_background_write(criticality_score=0.9)
        state = AuraState()

        # Turn one: the value is at its default when the background task
        # takes its copy of the turn.
        state.response_modifiers["ontogenetic_novelty"] = 0.5
        state = await phase.execute(state)
        await _drain_background_tasks()

        # Turn two: an earlier phase measures the real value.
        state.response_modifiers["ontogenetic_novelty"] = 0.4546
        state = await phase.execute(state)
        await _drain_background_tasks()
        return state.response_modifiers

    modifiers = asyncio.run(run())
    assert modifiers["ontogenetic_novelty"] == 0.4546
    assert modifiers["criticality_score"] == 0.9


def test_the_background_write_still_reaches_the_next_turn():
    async def run():
        phase = _phase_with_background_write(criticality_score=0.9, entropy=1.25)
        state = AuraState()
        state = await phase.execute(state)
        await _drain_background_tasks()
        assert "criticality_score" not in state.response_modifiers
        state = await phase.execute(state)
        return state.response_modifiers

    modifiers = asyncio.run(run())
    assert modifiers["criticality_score"] == 0.9
    assert modifiers["entropy"] == 1.25
