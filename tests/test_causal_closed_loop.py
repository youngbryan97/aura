import numpy as np
import pytest

from core.consciousness.closed_loop import (
    ClosedCausalLoop,
    OutputReceptor,
    PhiWitness,
    SelfPredictiveCore,
)
from core.container import ServiceContainer


class MockSubstrate:
    def __init__(self):
        self.x = np.zeros(64, dtype=np.float32)

    def inject_stimulus(self, delta, weight):
        self.x += delta * weight
        return None


def test_output_receptor_action_parsing_and_simulation():
    # Setup mock substrate in ServiceContainer
    sub = MockSubstrate()
    ServiceContainer.register("conscious_substrate", sub)

    receptor = OutputReceptor(neuron_count=64)

    # 1. Verbal text with action call
    text = "Aura suggests we: reroute_vessel(Vessel_Alpha, 270.0, 22.0)"
    res = receptor.receive_output(text)

    assert res is not None
    delta, magnitude = res
    assert magnitude > 0.0
    # Positive valence/arousal should have been set
    assert delta[0] > 0.0  # Valence
    assert delta[1] > 0.0  # Arousal
    assert delta[3] < 0.0  # Frustration should be reduced


def test_output_receptor_survives_container_lookup_failure(monkeypatch):
    events = []

    def boom(*args, **kwargs):
        raise RuntimeError("container unavailable")

    monkeypatch.setattr(ServiceContainer, "get", boom)
    monkeypatch.setattr(
        "core.consciousness.closed_loop._emit_closed_loop_fault",
        lambda exc, **kwargs: events.append((exc, kwargs)),
    )

    receptor = OutputReceptor(neuron_count=64)
    res = receptor.receive_output("reroute_vessel(Vessel_Alpha, 270.0, 20.0)")

    assert res is not None
    assert any(event[1].get("stage") == "output_receptor_substrate_lookup" for event in events)
    assert any(event[1].get("stage") == "output_receptor_loop_lookup" for event in events)
    assert any(event[1].get("stage") == "output_receptor_no_substrate" for event in events)


def test_output_receptor_rejects_malformed_numeric_action_text():
    receptor = OutputReceptor(neuron_count=64)
    assert receptor.receive_output("reroute_vessel(Vessel_Alpha, 1.2.3, 20.0)") is None


@pytest.mark.asyncio
async def test_output_receptor_observes_async_injection_failure(monkeypatch):
    events = []

    class FailingAsyncSubstrate:
        def __init__(self):
            self.x = np.zeros(64, dtype=np.float32)

        async def inject_stimulus(self, delta, weight):
            raise RuntimeError("async injection failed")

    ServiceContainer.register("conscious_substrate", FailingAsyncSubstrate())
    monkeypatch.setattr(
        "core.consciousness.closed_loop._emit_closed_loop_fault",
        lambda exc, **kwargs: events.append((exc, kwargs)),
    )

    receptor = OutputReceptor(neuron_count=64)
    assert receptor.receive_output("reroute_vessel(Vessel_Alpha, 270.0, 20.0)") is not None

    await asyncio_sleep()
    assert any(event[1].get("stage") == "output_receptor_injection_task" for event in events)


async def asyncio_sleep():
    import asyncio

    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_closed_loop_start_fails_closed_when_task_scheduling_fails(monkeypatch):
    class BrokenTracker:
        def create_task(self, *args, **kwargs):
            raise RuntimeError("scheduler offline")

    events = []
    monkeypatch.setattr("core.consciousness.closed_loop.get_task_tracker", lambda: BrokenTracker())
    monkeypatch.setattr(
        "core.consciousness.closed_loop._emit_closed_loop_fault",
        lambda exc, **kwargs: events.append((exc, kwargs)),
    )

    loop = ClosedCausalLoop()

    with pytest.raises(RuntimeError):
        await loop.start()

    assert loop.is_running is False
    assert any(event[1].get("stage") == "closed_loop_start_task" for event in events)


def test_self_predictive_core_physical_free_energy():
    core = SelfPredictiveCore(neuron_count=64)
    core.predict(np.zeros(64, dtype=np.float32))

    # Expected vs actual physical sensors matches
    expectations = {
        "port_east_load": 800.0,
        "port_west_load": 400.0,
        "vessel_alpha_speed": 15.0,
        "warehouse_load": 200.0,
        "system_cpu_usage": 35.0,
    }

    # Actual state
    actual_x = np.zeros(64, dtype=np.float32)

    cycle = core.observe_and_update(actual_x, simulated_expectations=expectations)
    assert cycle is not None
    # Since simulated matches actual physical values (we synced the real sensors), physical FE should be low/zero
    assert cycle.free_energy < 0.2


def test_phi_witness_pads_short_substrate_vectors():
    witness = PhiWitness()
    witness.record_substrate_state(np.array([0.1], dtype=np.float32))
    assert witness.get_diagnostics()["substrate_history_len"] == 1
