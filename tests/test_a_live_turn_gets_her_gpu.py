"""A live turn, from a person's first word to her last played: her background generation stands aside for it.

Her voice is made on the GPU her cortex works on. LIVE 11 November 2026, one patch of her voice took 122 ms with the
GPU quiet and 415 ms beside her goal pass, against the 160 ms it plays for (core/brain/llm_health_router_gate_owners.py).
"""
import asyncio
import threading

import pytest

from core.brain import llm_health_router as router_module
from core.brain import llm_health_router_gate_owners as owners


@pytest.fixture(autouse=True)
def _no_live_turns(monkeypatch):
    monkeypatch.setattr(owners, "_LIVE_TURNS", set())


def test_a_live_turn_is_a_person_waiting_for_her_model_until_it_ends(monkeypatch):
    monkeypatch.setattr(owners, "_background_stands_aside", lambda reason: None)
    assert not owners.a_person_is_waiting_for_her_model()
    owners.a_live_turn("voice-1", True)
    assert owners.a_person_is_waiting_for_her_model() and owners.a_live_turn_is_under_way()
    owners.a_live_turn("voice-1", True)  # thinking, then speaking: still the one turn
    owners.a_live_turn("voice-1", False)
    assert not owners.a_person_is_waiting_for_her_model() and not owners.a_live_turn_is_under_way()


def test_the_turn_beginning_stops_her_background_work_and_never_a_persons_reply(monkeypatch):
    import core.brain.llm.mlx_client as mlx_client

    asked = []
    monkeypatch.setattr(mlx_client, "soft_cancel_active_generations", lambda *, reason, owner_label=None: asked.append(reason))
    monkeypatch.setattr(router_module, "_oldest_generation_gate_lease", lambda: (3, 0.0, "her_goal_at_work:pass"))
    monkeypatch.setattr(owners, "_lease_is_user_foreground", lambda lease_id, owner: False)
    owners.a_live_turn("voice-1", True)
    owners.a_live_turn("voice-1", True)  # the same turn going on asks nothing more
    assert len(asked) == 1
    owners.a_live_turn("voice-1", False)
    monkeypatch.setattr(owners, "_lease_is_user_foreground", lambda lease_id, owner: True)
    owners.a_live_turn("voice-1", True)
    assert len(asked) == 1


def test_background_generation_is_refused_while_a_live_turn_is_under_way(monkeypatch):
    from core.brain.llm_health_router import HealthAwareLLMRouter

    monkeypatch.setattr(owners, "_background_stands_aside", lambda reason: None)
    gate = threading.BoundedSemaphore(1)
    monkeypatch.setattr(router_module, "_GENERATION_GATE", gate)
    monkeypatch.setattr(router_module, "_GENERATION_GATE_ACTIVE_LEASES", {})
    monkeypatch.setattr(router_module, "_GENERATION_GATE_LEASE_DEADLINES", {})
    owners.a_live_turn("voice-1", True)
    result = asyncio.run(HealthAwareLLMRouter().generate_with_metadata(
        "a background thought", origin="parallel_thought", purpose="background_probe", foreground_request=False))
    assert result["ok"] is False and result["deferred"] is True and "a live conversation turn" in result["error"]
    assert gate.acquire(False) is True  # nothing took the gate


def _a_held_lease(monkeypatch, *, foreground=False):
    gate = threading.BoundedSemaphore(1)
    monkeypatch.setattr(router_module, "_GENERATION_GATE", gate)
    monkeypatch.setattr(router_module, "_GENERATION_GATE_ACTIVE_LEASES", {})
    monkeypatch.setattr(router_module, "_GENERATION_GATE_LEASE_DEADLINES", {})
    monkeypatch.setattr(router_module, "_GENERATION_GATE_FOREGROUND_LEASES", set())
    monkeypatch.setattr(router_module, "_GENERATION_GATE_FORCED_LEASES", set())
    assert gate.acquire(False)
    return gate, router_module._mark_generation_gate_acquired("stream_narrative", foreground=foreground)


class _LazyClient:
    def __init__(self, device):
        self.kwargs = {"device": device}


def test_background_work_that_reaches_a_cpu_model_gives_the_gate_back(monkeypatch):
    """LIVE 11 November 2026: her narrative held her GPU's gate for 80 s from her CPU model, and a voice reply waited."""
    gate, lease = _a_held_lease(monkeypatch)
    with owners.holding_the_gate(lease):
        assert owners.a_cpu_client_gives_back_the_gate(_LazyClient("cpu")) is True
        assert gate.acquire(False)  # her reply can have it now
        gate.release()
        assert owners.a_cpu_client_gives_back_the_gate(_LazyClient("cpu")) is False  # given back once
    router_module._release_generation_gate_after_call(lease)  # the router's own release after the call is harmless
    assert gate.acquire(False) and not gate.acquire(False)


def test_a_gpu_model_or_a_persons_turn_keeps_the_gate(monkeypatch):
    gate, lease = _a_held_lease(monkeypatch)
    with owners.holding_the_gate(lease):
        assert owners.a_cpu_client_gives_back_the_gate(_LazyClient("gpu")) is False
        assert owners.a_cpu_client_gives_back_the_gate(object()) is False
    assert not gate.acquire(False)
    router_module._release_generation_gate_after_call(lease)
    gate, lease = _a_held_lease(monkeypatch, foreground=True)
    with owners.holding_the_gate(lease):
        assert owners.a_cpu_client_gives_back_the_gate(_LazyClient("cpu")) is False
    assert not gate.acquire(False)
    router_module._release_generation_gate_after_call(lease)
    assert owners.a_cpu_client_gives_back_the_gate(_LazyClient("cpu")) is False  # outside any held lease
