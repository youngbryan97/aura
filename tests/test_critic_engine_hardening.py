from __future__ import annotations

import asyncio
import types

from core import critic_engine as critic_module
from core.critic_engine import CriticEngine


def test_critic_engine_start_records_mycelium_registration_failure(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    class Bus:
        async def publish(self, _topic, _payload):
            self.attempted = True
            raise RuntimeError("event bus offline")

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    monkeypatch.setattr(
        critic_module.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(critic_module, "get_event_bus", lambda: Bus())
    monkeypatch.setattr(critic_module, "record_degradation", record_degradation)

    engine = CriticEngine()
    asyncio.run(engine.start())

    assert engine.running is True
    assert engine._critic_task is None
    assert recorded
    assert recorded[0][0] == "critic_engine"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True
    assert "mycelium registration" in str(recorded[0][2]["action"])


def test_critic_response_parse_clamps_and_sanitizes(monkeypatch):
    monkeypatch.setattr(
        critic_module,
        "record_degradation",
        lambda *_args, **_kwargs: None,
    )

    judgment = asyncio.run(
        CriticEngine()._parse_critic_response(
            {
                "goal_progress": "2.7",
                "evidence": "done\x00 enough",
                "contradictions": ["missing proof", 123],
                "recommendation": "REPLAN",
                "first_person_thought": "I need to rethink this.",
            },
            current_step=4,
        )
    )

    assert judgment.step_number == 4
    assert judgment.goal_progress == 1.0
    assert judgment.evidence == "done enough"
    assert judgment.contradictions == ["missing proof", "123"]
    assert judgment.recommendation == "replan"


def test_critical_shard_spawn_failure_returns_false_with_receipt(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    class Swarm:
        async def spawn_shard(self, _goal, _context):
            self.attempted = True
            raise RuntimeError("swarm unavailable")

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    orch = types.SimpleNamespace(sovereign_swarm=Swarm())
    monkeypatch.setattr(
        critic_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: orch if name == "orchestrator" else default),
    )
    monkeypatch.setattr(critic_module, "record_degradation", record_degradation)

    result = asyncio.run(CriticEngine().spawn_critical_shard("insight", "context"))

    assert result is False
    assert recorded
    assert recorded[0][0] == "critic_engine"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True
    assert "critical shard spawn" in str(recorded[0][2]["action"])
