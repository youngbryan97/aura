import time
from types import SimpleNamespace

import pytest

from core.brain import initiative_engine as initiative_module
from core.brain.initiative_engine import ProactiveInitiativeEngine


class _Affect:
    def __init__(self):
        self._raw_state = {"curiosity_metric": 90.0}

    def get_context_injection(self):
        return "curious"

    def get_resonance_string(self):
        return "Aura 100%"


class _Brain:
    async def think(self, _prompt, mode="fast"):
        return SimpleNamespace(content="I noticed a thread worth tugging on.")


class _Voice:
    def __init__(self):
        self.called = False

    async def speak_stream(self, stream):
        self.called = True
        async for _chunk in stream:
            pass


class _Memory:
    def __init__(self):
        self.entries = []

    def append_to_stm(self, *, role, content):
        self.entries.append((role, content))


@pytest.mark.asyncio
async def test_denied_proactive_release_does_not_bypass_authority_to_voice(monkeypatch):
    voice = _Voice()
    memory = _Memory()
    engine = ProactiveInitiativeEngine(_Brain(), voice, _Affect(), memory)

    class DenyingAuthority:
        async def release_expression(self, *_args, **_kwargs):
            return {"ok": False, "reason": "output_gate_missing"}

    monkeypatch.setattr(
        "core.consciousness.executive_authority.get_executive_authority",
        lambda _orchestrator: DenyingAuthority(),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda _name, default=None: default,
    )

    await engine._trigger_autonomous_conversation()

    assert voice.called is False
    assert memory.entries == []
    assert engine._last_delivery_blocked_reason == "output_gate_missing"


@pytest.mark.asyncio
async def test_successful_proactive_release_accepts_sync_memory_append(monkeypatch):
    memory = _Memory()
    affect = _Affect()
    engine = ProactiveInitiativeEngine(_Brain(), _Voice(), affect, memory)

    class ReleasingAuthority:
        async def release_expression(self, *_args, **_kwargs):
            return {"ok": True, "reason": "approved"}

    monkeypatch.setattr(
        "core.consciousness.executive_authority.get_executive_authority",
        lambda _orchestrator: ReleasingAuthority(),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda _name, default=None: default,
    )

    await engine._trigger_autonomous_conversation()

    assert memory.entries == [("assistant", "I noticed a thread worth tugging on.")]
    assert affect._raw_state["curiosity_metric"] == 30.0


@pytest.mark.asyncio
async def test_proactive_loop_records_failure_and_backs_off(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        initiative_module,
        "_record_initiative_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    engine = ProactiveInitiativeEngine(_Brain(), _Voice(), _Affect(), _Memory())
    engine.last_interaction_time = time.time() - 7200
    engine.silence_threshold_seconds = 0
    engine.boredom_trigger_level = 1.0

    async def fail_generation():
        raise RuntimeError("proactive path fault")

    engine._trigger_autonomous_conversation = fail_generation

    async def fake_sleep(delay):
        if delay != 60:
            engine.stop()

    monkeypatch.setattr(initiative_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda _name, default=None: default,
    )

    await engine.start_proactive_loop()

    assert engine._loop_failure_count == 1
    assert engine._last_loop_error == "RuntimeError: proactive path fault"
    assert recorded[0][1]["action"] == (
        "kept proactive loop alive with bounded backoff after recoverable loop failure"
    )
