from __future__ import annotations

import pytest

from core.brain.multimodal_orchestrator import MultimodalOrchestrator


class VoiceEngine:
    def __init__(self):
        self.spoken = []

    async def speak(self, content):
        self.spoken.append(content)


class EventBus:
    def __init__(self):
        self.published = []

    async def publish(self, topic, payload):
        self.published.append((topic, payload))


@pytest.mark.asyncio
async def test_render_returns_receipt_and_schedules_only_needed_modalities():
    orchestrator = MultimodalOrchestrator()
    orchestrator._is_setup = True
    orchestrator.voice_engine = VoiceEngine()
    orchestrator.event_bus = EventBus()
    orchestrator.capability_engine = None

    receipt = await orchestrator.render("I am happy.", {"voice": True})

    assert receipt["ok"] is True
    assert receipt["scheduled"] == ["voice", "expression"]
    assert receipt["task_count"] == 2


@pytest.mark.asyncio
async def test_manifest_assets_executes_registered_capability_skill():
    calls = []

    class CapabilityEngine:
        skills = {"local_media_generation": object(), "image_generation": object()}

        async def execute(self, skill_name, payload):
            calls.append((skill_name, payload))
            return {"ok": True, "uri": "artifact://generated/aura.png"}

    orchestrator = MultimodalOrchestrator()
    orchestrator.capability_engine = CapabilityEngine()

    result = await orchestrator._manifest_assets("[Manifesting: silver control room]")

    assert result["ok"] is True
    assert calls[0][0] == "local_media_generation"
    assert calls[0][1]["prompt"] == "silver control room"
    assert result["generated"][0]["result"]["ok"] is True


@pytest.mark.asyncio
async def test_manifest_assets_reports_missing_capability_engine():
    orchestrator = MultimodalOrchestrator()
    orchestrator.capability_engine = None

    result = await orchestrator._manifest_assets("[Drawing: resilient runtime map]")

    assert result == {
        "ok": False,
        "reason": "capability_engine_unavailable",
        "generated": [],
    }


def test_setup_failure_returns_false(monkeypatch):
    def _raise(*_args, **_kwargs):
        reason = "container offline"
        raise RuntimeError(reason)

    monkeypatch.setattr("core.brain.multimodal_orchestrator.ServiceContainer.get", _raise)

    orchestrator = MultimodalOrchestrator()

    assert orchestrator._setup() is False
    assert orchestrator._is_setup is False
