from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.belief_revision import BeliefRevisionEngine


@pytest.mark.asyncio
async def test_corrupt_belief_store_is_quarantined_and_reseeded(tmp_path):
    db_path = tmp_path / "belief_system.json"
    db_path.write_text("{not-valid-json", encoding="utf-8")

    engine = BeliefRevisionEngine(db_path=str(db_path))

    assert engine.beliefs
    assert db_path.exists()
    assert list(tmp_path.glob("belief_system.json.corrupt.*"))
    data = json.loads(db_path.read_text(encoding="utf-8"))
    assert len(data["beliefs"]) >= 3


@pytest.mark.asyncio
async def test_conversation_update_persists_even_when_engine_is_not_running(tmp_path):
    db_path = tmp_path / "beliefs.json"
    engine = BeliefRevisionEngine(db_path=str(db_path))

    await engine.update_from_conversation(
        "Aura should keep state carefully.",
        "I will preserve the details.",
    )

    data = json.loads(db_path.read_text(encoding="utf-8"))
    assert any("Aura should keep state carefully" in item["content"] for item in data["beliefs"])


@pytest.mark.asyncio
async def test_process_new_claim_clamps_confidence_and_updates_existing(tmp_path):
    engine = BeliefRevisionEngine(db_path=str(tmp_path / "beliefs.json"))

    created = await engine.process_new_claim(
        "Runtime claims need evidence.",
        domain="world",
        source="tool_result",
        confidence=2.0,
    )
    updated = await engine.process_new_claim(
        "Runtime claims need evidence.",
        domain="world",
        source="conversation",
        confidence=0.5,
    )

    belief = next(
        item for item in engine.beliefs if item.content == "Runtime claims need evidence."
    )
    assert created["ok"] is True
    assert updated == {"ok": True, "updated": True, "belief_id": belief.id}
    assert 0.0 <= belief.confidence <= 1.0
    assert belief.supporting_evidence == ["tool_result", "conversation"]


@pytest.mark.asyncio
async def test_start_is_idempotent_when_event_bus_registration_fails(monkeypatch, tmp_path):
    class BrokenBus:
        async def publish(self, event: str, payload: dict):
            self.event = event
            self.payload = payload
            if event:
                raise RuntimeError("event bus offline")

    monkeypatch.setattr("core.belief_revision.get_event_bus", lambda: BrokenBus())
    monkeypatch.setattr(
        "core.belief_revision.ServiceContainer.get",
        lambda name, default=None: SimpleNamespace() if name == "memory_facade" else default,
    )

    engine = BeliefRevisionEngine(db_path=str(tmp_path / "beliefs.json"))
    first = await engine.start()
    first_task = engine._revision_task
    second = await engine.start()

    assert first["ok"] is True
    assert first["event_registered"] is False
    assert first["dependencies"]["memory_facade"] is True
    assert second["already_running"] is True
    assert engine._revision_task is first_task
    assert getattr(first_task, "_aura_supervised", False) is True

    await engine.stop()
