import asyncio
import math
import sys
from types import SimpleNamespace

import pytest

from core.agency import compute_orchestrator as co


@pytest.mark.asyncio
async def test_resource_sampling_keeps_cpu_ram_when_thermal_sensor_is_unsupported(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        co,
        "_record_compute_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    fake_psutil = SimpleNamespace(
        cpu_percent=lambda _interval: 87.5,
        virtual_memory=lambda: SimpleNamespace(percent=91.0),
        sensors_temperatures=lambda: (_ for _ in ()).throw(
            NotImplementedError("thermal unavailable")
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    orchestrator = co.ComputeOrchestrator()

    cpu, ram, temp = await orchestrator._sample_resources()

    assert cpu == 87.5
    assert ram == 91.0
    assert temp is None
    assert orchestrator._last_resource_sample_error is None
    assert recorded[0][1]["action"] == "continued resource allocation without thermal sensor data"
    assert recorded[0][1]["severity"] == "debug"


def test_hedonic_allocation_clamps_invalid_substrate_values(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        co,
        "_record_compute_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    from core.consciousness import hedonic_gradient

    monkeypatch.setattr(
        hedonic_gradient,
        "get_hedonic_gradient",
        lambda: SimpleNamespace(
            score=2.5,
            allocation=SimpleNamespace(hedonic_score=-4.0, token_multiplier=math.nan),
        ),
    )

    orchestrator = co.ComputeOrchestrator()
    hedonic, token_multiplier = orchestrator._read_hedonic_allocation()

    assert hedonic == 1.0
    assert token_multiplier == 1.0
    assert recorded[-1][1]["action"] == (
        "replaced non-finite hedonic token multiplier with neutral default"
    )


@pytest.mark.asyncio
async def test_affect_anxiety_delivery_failures_are_owned_by_compute_orchestrator(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        co,
        "_record_compute_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    class BrokenAffect:
        async def apply_stimulus(self, _stimulus_type, _intensity):
            raise RuntimeError("affect loop unavailable")

    class CapturingTracker:
        def __init__(self):
            self.tasks = []

        def track(self, coro, name=None):
            task = asyncio.create_task(coro, name=name)
            self.tasks.append((task, name))
            return task

    tracker = CapturingTracker()
    monkeypatch.setattr(co, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: BrokenAffect() if name == "affect_engine" else default),
    )

    orchestrator = co.ComputeOrchestrator()

    assert orchestrator._push_anxiety_to_affect(0.8) is True
    assert tracker.tasks[0][1] == "ComputeOrchestrator.affect_anxiety"
    await tracker.tasks[0][0]

    assert orchestrator._affect_delivery_failures == 1
    assert orchestrator._last_affect_delivery_error == "RuntimeError: affect loop unavailable"
    assert recorded[0][1]["action"] == (
        "captured asynchronous affect anxiety delivery failure and left compute throttles active"
    )


def test_missing_affect_path_is_reported_without_crashing_compute(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        co,
        "_record_compute_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )

    orchestrator = co.ComputeOrchestrator()

    assert orchestrator._push_anxiety_to_affect(0.9) is False
    assert recorded[0][1]["action"] == (
        "kept compute throttles active without affect feedback because affect stimulus path is unavailable"
    )
