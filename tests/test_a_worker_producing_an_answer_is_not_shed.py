"""A worker mid-generation is not a worker to shed.

LIVE, 2026-09-21 boot. The brainstem was loaded to serve a foreground turn
and four seconds into its 144-second first-token ceiling the memory-pressure
shed rebooted it: "Generation cancelled for Ternary-Bonsai-2-27B-mlx-2bit
during expected reboot (background_memory_pressure_shed)". The turn ended
"I'm here. My response was cut short."

Every rule in that shed is about which LANE to preserve — the ladder, the
primary, the cortex reserve — and none of them asked whether the worker was
busy. Shedding it does not free the memory sooner either: the weights stay
resident until the reboot completes. The only thing the shed gained was the
destroyed answer.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.brain.inference_gate import InferenceGate


class _Worker:
    def __init__(self, *, busy: bool) -> None:
        self._busy = busy
        self.rebooted_with: list[str] = []

    def is_alive(self) -> bool:
        return True

    def generation_in_flight(self) -> bool:
        return self._busy

    async def reboot_worker(self, *, reason: str = "", **_kwargs: Any) -> bool:
        self.rebooted_with.append(reason)
        return True


def _gate_that_will_shed(monkeypatch: pytest.MonkeyPatch, registry: dict[str, Any]) -> Any:
    gate = InferenceGate.__new__(InferenceGate)
    gate._last_background_memory_shed_at = 0.0
    gate._mlx_client = None
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: type("S", (), {"available_gb": 1.0})(),
        raising=False,
    )
    monkeypatch.setattr(
        InferenceGate, "_primary_load_required_gb", lambda self: 20.0, raising=False
    )
    monkeypatch.setattr(InferenceGate, "_env_float", lambda self, *a: 8.0, raising=False)
    monkeypatch.setattr(
        InferenceGate, "_fallback_ladder_paths", lambda self: frozenset(), raising=False
    )
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.clients_snapshot", lambda: registry, raising=False
    )
    return gate


def test_a_busy_worker_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _Worker(busy=True)
    gate = _gate_that_will_shed(monkeypatch, {"/models/brainstem": worker})

    asyncio.run(gate._shed_background_workers_for_memory_pressure(force=True))

    assert worker.rebooted_with == [], "the answer it was producing was destroyed"


def test_an_idle_worker_is_still_shed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The null. A shed that spares everything frees nothing."""
    worker = _Worker(busy=False)
    gate = _gate_that_will_shed(monkeypatch, {"/models/brainstem": worker})

    asyncio.run(gate._shed_background_workers_for_memory_pressure(force=True))

    assert worker.rebooted_with == ["background_memory_pressure_shed"]


def test_the_busy_one_is_spared_and_the_idle_one_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    busy = _Worker(busy=True)
    idle = _Worker(busy=False)
    gate = _gate_that_will_shed(
        monkeypatch, {"/models/busy": busy, "/models/idle": idle}
    )

    asyncio.run(gate._shed_background_workers_for_memory_pressure(force=True))

    assert busy.rebooted_with == []
    assert idle.rebooted_with == ["background_memory_pressure_shed"]


def test_a_worker_that_cannot_say_is_shed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An older client with no such method keeps the previous behaviour."""

    class _Old:
        def __init__(self) -> None:
            self.rebooted_with: list[str] = []

        def is_alive(self) -> bool:
            return True

        async def reboot_worker(self, *, reason: str = "", **_kwargs: Any) -> bool:
            self.rebooted_with.append(reason)
            return True

    worker = _Old()
    gate = _gate_that_will_shed(monkeypatch, {"/models/old": worker})

    asyncio.run(gate._shed_background_workers_for_memory_pressure(force=True))

    assert worker.rebooted_with == ["background_memory_pressure_shed"]
