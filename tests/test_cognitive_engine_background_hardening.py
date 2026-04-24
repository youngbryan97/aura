from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.brain.cognitive_engine import CognitiveEngine
from core.brain.types import ThinkingMode, Thought
from core.state.aura_state import AuraState


def test_cognitive_engine_treats_prefixed_user_origin_as_foreground():
    assert CognitiveEngine._is_background_request("routing_user", False) is False
    assert CognitiveEngine._is_background_request("routing_voice_command", False) is False
    assert CognitiveEngine._is_background_request("autonomous_thought", False) is True


@pytest.mark.asyncio
async def test_cognitive_engine_skips_identity_refresh_for_background_origin(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = SimpleNamespace(get_current=AsyncMock(return_value=state))
    captured = {}

    async def _fake_run(state, objective, mode, origin, context=None, **kwargs):
        captured["objective"] = objective
        return Thought(id="bg-thought", content="ok", mode=mode)

    monitor = SimpleNamespace(needs_context_refresh=lambda *_args, **_kwargs: True)

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: repo if name == "state_repository" else default),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: monitor if name == "drift_monitor" else default,
    )
    monkeypatch.setattr(
        "core.brain.cognitive_engine.ContextAssembler.build_system_prompt",
        staticmethod(lambda _state: "context" * 500),
    )
    monkeypatch.setattr(engine, "_run_thinking_loop", _fake_run)

    thought = await engine.think(
        "Summarize internal maintenance state.",
        mode=ThinkingMode.FAST,
        origin="autonomous",
        is_background=True,
    )

    assert thought.content == "ok"
    assert captured["objective"] == "Summarize internal maintenance state."


@pytest.mark.asyncio
async def test_cognitive_engine_suppresses_background_thoughts_when_background_policy_blocks(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = SimpleNamespace(get_current=AsyncMock(return_value=state))

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: repo if name == "state_repository" else SimpleNamespace()),
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "failure_lockdown_0.20",
    )

    thought = await engine.think(
        "Distill this memory to its essential insight.",
        mode=ThinkingMode.FAST,
        origin="sovereign_pruner",
        is_background=True,
    )

    assert thought.metadata["suppressed"] is True
    assert "background_thought_suppressed" in thought.reasoning[0]


@pytest.mark.asyncio
async def test_cognitive_engine_resolves_missing_origin_from_orchestrator(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = SimpleNamespace(get_current=AsyncMock(return_value=state), _current=state)
    orchestrator = SimpleNamespace(_current_origin="terminal_monitor")
    captured = {}

    async def _fake_run(state, objective, mode, origin, context=None, **kwargs):
        captured["origin"] = origin
        return Thought(id="origin-from-orch", content="ok", mode=mode)

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: (
                orchestrator if name == "orchestrator" else repo if name == "state_repository" else default
            )
        ),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: default,
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(engine, "_run_thinking_loop", _fake_run)

    thought = await engine.think("Investigate the timeout.", mode=ThinkingMode.FAST)

    assert thought.content == "ok"
    assert captured["origin"] == "terminal_monitor"


@pytest.mark.asyncio
async def test_cognitive_engine_defaults_missing_origin_to_system(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = SimpleNamespace(get_current=AsyncMock(return_value=state), _current=state)
    captured = {}

    async def _fake_run(state, objective, mode, origin, context=None, **kwargs):
        captured["origin"] = origin
        return Thought(id="origin-default", content="ok", mode=mode)

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: repo if name == "state_repository" else default),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: default,
    )
    monkeypatch.setattr(engine, "_run_thinking_loop", _fake_run)

    thought = await engine.think("Perform internal maintenance.", mode=ThinkingMode.FAST)

    assert thought.content == "ok"
    assert captured["origin"] == "system"
