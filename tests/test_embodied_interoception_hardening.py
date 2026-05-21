import asyncio
import math

import numpy as np
import pytest

from core.consciousness import embodied_interoception as embodied
from core.consciousness.embodied_interoception import (
    EmbodiedInteroception,
    InteroceptiveChannel,
)


def test_channel_derivatives_use_immediate_previous_sample_and_reject_nan():
    channel = InteroceptiveChannel("metabolic_load", alpha=1.0)

    channel.update(0.2)
    assert channel.smoothed == pytest.approx(0.2)
    assert channel.velocity == pytest.approx(0.2)

    channel.update(0.8)
    assert channel.smoothed == pytest.approx(0.8)
    assert channel.velocity == pytest.approx(0.6)
    assert channel.acceleration == pytest.approx(0.4)

    channel.update(float("nan"))
    assert channel._failed is True
    assert math.isfinite(channel.smoothed)
    assert 0.0 <= channel.smoothed <= 1.0


def test_sensory_vector_and_budget_clamp_invalid_channel_values():
    interoception = EmbodiedInteroception()
    interoception.channels["metabolic_load"].smoothed = float("nan")
    interoception.channels["process_load"].smoothed = float("inf")

    vector = interoception.get_sensory_vector()
    budget = interoception.get_body_budget()
    state = interoception.get_interoceptive_state()

    assert vector.shape == (1024,)
    assert np.isfinite(vector).all()
    assert all(math.isfinite(value) for value in budget.values())
    assert state["metabolic_load"] == 0.5
    assert state["process_load"] == 0.5


def test_mesh_failure_is_recorded_without_dropping_body_state(monkeypatch):
    receipts = []

    def record_degradation(*args, **kwargs):
        receipts.append((args, kwargs))

    class BrokenMesh:
        def inject_sensory(self, vector):
            assert vector.shape == (1024,)
            raise RuntimeError("mesh offline")

    monkeypatch.setattr(embodied, "record_degradation", record_degradation)

    interoception = EmbodiedInteroception()
    interoception.channels["metabolic_load"].update(0.6)
    before = interoception.get_interoceptive_state()
    interoception._mesh_ref = BrokenMesh()

    interoception._push_to_mesh()

    assert interoception.get_interoceptive_state() == before
    assert receipts
    assert receipts[0][1]["receipt_required"] is True
    assert "preserving interoceptive state" in receipts[0][1]["action"]


def test_neurochemical_event_failures_are_isolated(monkeypatch):
    receipts = []

    def record_degradation(*args, **kwargs):
        receipts.append((args, kwargs))

    class PartiallyBrokenNeurochemistry:
        def __init__(self):
            self.threat_attempts = []
            self.prediction_errors = []

        def on_threat(self, **kwargs):
            self.threat_attempts.append(kwargs)
            raise RuntimeError("threat path offline")

        def on_prediction_error(self, **kwargs):
            self.prediction_errors.append(kwargs)

    monkeypatch.setattr(embodied, "record_degradation", record_degradation)

    interoception = EmbodiedInteroception()
    interoception._neurochemical_ref = PartiallyBrokenNeurochemistry()
    interoception.channels["metabolic_load"].smoothed = 0.9
    interoception.channels["metabolic_load"].velocity = 0.2

    interoception._trigger_neurochemical_events()

    assert interoception._neurochemical_ref.prediction_errors == [{"error": 0.4}]
    assert receipts
    assert "isolated failed neurochemical event on_threat" in receipts[0][1]["action"]


@pytest.mark.asyncio
async def test_sampling_loop_survives_one_failed_tick(monkeypatch):
    receipts = []

    def record_degradation(*args, **kwargs):
        receipts.append((args, kwargs))

    monkeypatch.setattr(embodied, "record_degradation", record_degradation)

    interoception = EmbodiedInteroception()
    interoception._SAMPLE_HZ = 500.0
    calls = 0

    def sample_hardware():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("sensor bus offline")
        interoception._tick_count += 1
        interoception._running = False

    monkeypatch.setattr(interoception, "_sample_hardware", sample_hardware)
    monkeypatch.setattr(interoception, "_push_to_mesh", lambda: None)
    monkeypatch.setattr(interoception, "_trigger_neurochemical_events", lambda: None)

    interoception._running = True
    await asyncio.wait_for(interoception._run_loop(), timeout=1.0)

    assert calls == 2
    assert interoception._tick_count == 1
    assert interoception._consecutive_tick_failures == 0
    assert receipts
    assert "kept sampling loop alive" in receipts[0][1]["action"]
