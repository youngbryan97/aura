from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.belief_challenger import BeliefChallenger


@pytest.mark.asyncio
async def test_start_is_idempotent_when_event_bus_registration_fails(monkeypatch):
    class BrokenBus:
        async def publish(self, event: str, payload: dict):
            self.last_event = event
            self.last_payload = payload
            if event:
                raise RuntimeError("event bus offline")

    def _service_get(name, default=None):
        services = {
            "belief_revision_engine": object(),
            "epistemic_tracker": object(),
            "api_adapter": object(),
        }
        return services.get(name, default)

    monkeypatch.setattr("core.container.ServiceContainer.get", _service_get)
    monkeypatch.setattr("core.event_bus.get_event_bus", lambda: BrokenBus())

    challenger = BeliefChallenger()
    first = await challenger.start()
    first_task = challenger._challenge_task
    second = await challenger.start()

    assert first["ok"] is True
    assert first["event_registered"] is False
    assert second["already_running"] is True
    assert challenger._challenge_task is first_task
    assert getattr(first_task, "_aura_supervised", False) is True

    await challenger.stop()
    assert challenger.running is False


@pytest.mark.asyncio
async def test_dialectical_pass_times_out_without_mutating_beliefs():
    class SlowApi:
        async def generate(self, _prompt, _options):
            await asyncio.sleep(0.2)
            return "late answer"

    beliefs = SimpleNamespace(process_new_claim=pytest.fail)
    challenger = BeliefChallenger(challenge_timeout_s=0.01)
    challenger._api = SlowApi()
    challenger._beliefs = beliefs

    result = await challenger._perform_dialectical_pass("AI systems need external checks.")

    assert result["ok"] is False
    assert result["reason"] == "TimeoutError"


@pytest.mark.asyncio
async def test_revision_updates_beliefs_and_persists_learning(monkeypatch):
    class Api:
        def __init__(self):
            self.calls = []

        async def generate(self, prompt, options):
            self.calls.append((prompt, options))
            if options["purpose"] == "belief_challenge":
                return "This belief ignores deployment failure modes."
            return "Valid point: I should revise this belief to include operational evidence."

    class Beliefs:
        def __init__(self):
            self.claims = []

        async def process_new_claim(self, **kwargs):
            self.claims.append(kwargs)

    class Learner:
        def __init__(self):
            self.examples = []

        def record_example(self, **kwargs):
            self.examples.append(kwargs)

    class Memory:
        def __init__(self):
            self.items = []

        async def store(self, **kwargs):
            self.items.append(kwargs)

    learner = Learner()
    memory = Memory()

    def _service_get(name, default=None):
        if name == "live_learner":
            return learner
        if name == "vector_memory_engine":
            return memory
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", _service_get)

    challenger = BeliefChallenger()
    challenger._api = Api()
    challenger._beliefs = Beliefs()

    result = await challenger._perform_dialectical_pass("Shipping is mostly about ideas.")

    assert result == {"ok": True, "revised": True, "belief": "Shipping is mostly about ideas."}
    assert challenger._beliefs.claims[0]["domain"] == "revision"
    assert learner.examples
    assert memory.items[0]["source"] == "belief_challenger"
