from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.autonomous_initiative_loop import AutonomousInitiativeLoop
from core.orchestrator.mixins.output_formatter import OutputFormatterMixin
from core.proactive_presence import ProactivePresence


def test_emit_thought_stream_falls_back_to_thought_emitter(monkeypatch):
    emitter = SimpleNamespace(emit=MagicMock())
    monkeypatch.setattr("core.thought_stream.get_emitter", lambda: emitter)

    formatter = OutputFormatterMixin()
    formatter._emit_thought_stream("Mind wandering through loose threads.")

    emitter.emit.assert_called_once_with(
        "Autonomous Thought",
        "Mind wandering through loose threads.",
        level="info",
        category="Autonomy",
    )


@pytest.mark.asyncio
async def test_self_development_cycle_runs_scan_tests_and_proposal(monkeypatch):
    capability_engine = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                {
                    "ok": True,
                    "issues_found": 1,
                    "top_issues": [
                        {
                            "file": "core/example.py",
                            "message": "Function 'foo' is too long (88 lines).",
                        }
                    ],
                },
                {
                    "ok": False,
                    "error": "1 generated sandbox test failed",
                },
                {
                    "ok": True,
                    "proposal_path": "/tmp/evolution/proposal.md",
                },
            ]
        )
    )
    monkeypatch.setattr(
        "core.autonomous_initiative_loop.optional_service",
        lambda name, default=None: capability_engine if name == "capability_engine" else default,
    )

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace(cognitive_engine=object()))
    emitted: list[tuple[str, str, str]] = []
    loop._emit_feed = lambda title, content, *, category: emitted.append((title, content, category))

    await loop._run_self_development_cycle()

    calls = capability_engine.execute.await_args_list
    assert [call.args[0] for call in calls] == ["auto_refactor", "test_generator", "self_evolution"]
    assert any("sandbox tests" in content.lower() for _, content, _ in emitted)
    assert any("proposal" in content.lower() or "saved to" in content.lower() for _, content, _ in emitted)


@pytest.mark.asyncio
async def test_proactive_presence_prefers_visible_primary(monkeypatch):
    orchestrator = SimpleNamespace(
        emit_spontaneous_message=AsyncMock(
            return_value={
                "ok": True,
                "action": "released",
                "target": "primary",
            }
        ),
        _last_thought_time=0.0,
    )
    emitter = SimpleNamespace(emit=MagicMock())
    terminal = SimpleNamespace(queue_autonomous_message=MagicMock())
    monkeypatch.setattr("core.thought_stream.get_emitter", lambda: emitter)
    monkeypatch.setattr("core.terminal_chat.get_terminal_fallback", lambda: terminal)

    presence = ProactivePresence(orchestrator=orchestrator)
    await presence._emit("still here.")

    orchestrator.emit_spontaneous_message.assert_awaited_once()
    _, kwargs = orchestrator.emit_spontaneous_message.await_args
    assert kwargs["origin"] == "proactive_presence"
    assert kwargs["metadata"]["visible_presence"] is True
    assert kwargs["metadata"]["initiative_activity"] is False
    emitter.emit.assert_not_called()
    terminal.queue_autonomous_message.assert_not_called()
    assert presence._outputs_this_hour == 1
    assert presence._consecutive_unprompted == 1


@pytest.mark.asyncio
async def test_proactive_presence_requeues_visible_update_when_primary_is_temporarily_held(monkeypatch):
    orchestrator = SimpleNamespace(
        emit_spontaneous_message=AsyncMock(
            return_value={
                "ok": True,
                "action": "released",
                "target": "secondary",
                "reason": "user_recently_active",
            }
        ),
        _last_thought_time=0.0,
    )
    emitter = SimpleNamespace(emit=MagicMock())
    terminal = SimpleNamespace(queue_autonomous_message=MagicMock())
    monkeypatch.setattr("core.thought_stream.get_emitter", lambda: emitter)
    monkeypatch.setattr("core.terminal_chat.get_terminal_fallback", lambda: terminal)

    presence = ProactivePresence(orchestrator=orchestrator)
    await presence._emit(
        "I found something worth holding onto for now.",
        source="autonomous_initiative_loop",
        initiative_activity=True,
        allow_during_away=True,
    )

    orchestrator.emit_spontaneous_message.assert_awaited_once()
    emitter.emit.assert_not_called()
    terminal.queue_autonomous_message.assert_not_called()
    assert len(presence._queued_messages) == 1
    queued = presence._queued_messages[0]
    assert queued["content"] == "I found something worth holding onto for now."
    assert queued["initiative_activity"] is True
    assert queued["allow_during_away"] is True
    assert queued["retries"] == 1


def test_proactive_presence_allows_queued_visible_updates_during_away_mode():
    orchestrator = SimpleNamespace(
        _last_user_interaction_time=0.0,
        _last_thought_time=0.0,
    )
    presence = ProactivePresence(orchestrator=orchestrator)
    presence._user_away = True
    presence._user_away_since = time.time()
    assert presence.queue_autonomous_message(
        "I'm still here and actively working.",
        source="autonomous_initiative_loop",
        initiative_activity=True,
        allow_during_away=True,
    )

    queued = presence._next_ready_queued_message()

    assert queued is not None
    assert queued["content"] == "I'm still here and actively working."
