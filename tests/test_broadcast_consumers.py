"""A broadcast that no process consumes has been logged, not broadcast.

The workspace ran a real competition and handed the winner to a processor list
nothing ever registered anything on. These tests pin the wiring: that the
consumers exist, that each one does something a specialised process would do
with the content, and that a failing consumer cannot take the tick down with
it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from core.consciousness.broadcast_consumers import register_broadcast_consumers


class _Workspace:
    def __init__(self) -> None:
        self.processors = []
        self.ignition_level = 0.8

    def register_processor(self, fn):
        self.processors.append(fn)

    async def broadcast(self, winner):
        event = SimpleNamespace(winners=[winner], timestamp=0.0)
        for processor in self.processors:
            await processor(event)


class _Substrate:
    def __init__(self) -> None:
        self.injected: list[tuple[int, float]] = []

    def encode_text_to_stimulus(self, text: str) -> np.ndarray:
        return np.ones(8) * (len(text) / 100.0)

    async def inject_stimulus(self, vector, weight: float = 1.0) -> None:
        self.injected.append((len(vector), float(weight)))


def _winner(source: str = "memory", priority: float = 0.7):
    return SimpleNamespace(source=source, content="something worth attending to", effective_priority=priority)


def test_every_consumer_is_registered():
    workspace = _Workspace()
    names = register_broadcast_consumers(workspace, substrate=_Substrate())
    assert set(names) >= {"recurrent_cognition", "self_model", "affect", "memory"}
    assert len(workspace.processors) == len(names)


def test_the_winner_reaches_the_substrate_with_its_own_priority_as_the_weight():
    workspace = _Workspace()
    substrate = _Substrate()
    register_broadcast_consumers(workspace, substrate=substrate)
    asyncio.run(workspace.broadcast(_winner(priority=0.42)))
    assert substrate.injected == [(8, pytest.approx(0.42))]


def test_the_winner_lands_in_a_bounded_memory_trace():
    workspace = _Workspace()
    register_broadcast_consumers(workspace, substrate=_Substrate())

    async def many():
        for index in range(50):
            await workspace.broadcast(_winner(source=f"s{index}"))

    asyncio.run(many())
    trace = workspace.broadcast_trace
    assert len(trace) == 32
    assert trace[-1]["source"] == "s49"


def test_a_consumer_that_cannot_do_its_job_does_not_take_the_others_down():
    workspace = _Workspace()

    class _Broken(_Substrate):
        def encode_text_to_stimulus(self, text: str):
            raise RuntimeError("no encoder here")

    register_broadcast_consumers(workspace, substrate=_Broken())
    asyncio.run(workspace.broadcast(_winner()))
    assert len(workspace.broadcast_trace) == 1


def test_no_workspace_means_no_consumers_rather_than_an_error():
    assert register_broadcast_consumers(None) == []
    assert register_broadcast_consumers(object()) == []


def test_the_live_consciousness_system_registers_them():
    """The wiring that matters: the real system, not a stand-in."""
    from core.consciousness.system import ConsciousnessSystem

    system = ConsciousnessSystem(SimpleNamespace(affect_engine=None, substrate=None, state=None))
    assert getattr(system, "broadcast_consumers", []) != []
    assert system.global_workspace._processors


def test_the_broadcast_arousal_is_left_where_affect_reads_it():
    """The first version wrote the blend onto the affect engine, which nothing
    reads back into `AuraState.affect` — a channel of exactly the kind this
    file was written to fix."""
    workspace = _Workspace()
    register_broadcast_consumers(workspace, substrate=_Substrate())
    asyncio.run(workspace.broadcast(_winner(source="memory", priority=0.9)))

    reading = workspace.last_broadcast_arousal
    assert reading["ignition"] == pytest.approx(0.8)
    assert reading["priority"] == pytest.approx(0.9)

    import asyncio as _asyncio

    from core.phases.affect_update import AffectUpdatePhase
    from core.runtime.service_registry import register_runtime_service
    from core.state.aura_state import AuraState

    register_runtime_service("global_workspace", workspace, required=False)
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "user", "content": "hello"})
    state.affect.arousal = 0.0
    _asyncio.run(AffectUpdatePhase(None).execute(state))
    assert state.affect.arousal > 0.5
    assert state.response_modifiers.get("broadcast_ignition") == pytest.approx(0.8)
