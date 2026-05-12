from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.autonomous_initiative_loop import AutonomousInitiativeLoop
from core.orchestrator.mixins.output_formatter import OutputFormatterMixin
from core.proactive_presence import ProactivePresence
from core.self_modification.growth_ladder import GrowthLadder, ModificationLevel


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


def test_background_initiative_gate_does_not_depend_on_chat_lane_readiness(monkeypatch):
    from core.autonomous_initiative_loop import _background_initiative_allowed

    monkeypatch.setattr(
        "core.runtime.background_policy.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=35.0),
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.get_unified_failure_state",
        lambda: {"pressure": 0.0},
    )

    orchestrator = SimpleNamespace(
        _last_user_interaction_time=time.time() - 3600.0,
        is_busy=False,
        _suppress_unsolicited_proactivity_until=0.0,
        _foreground_user_quiet_until=0.0,
    )

    assert _background_initiative_allowed(orchestrator) is True


def test_social_autonomy_due_actions_are_periodic_not_boot_only():
    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())

    assert loop._social_due_actions(1000.0) == {"email": True, "reddit": False}

    loop._last_email_check = 1000.0
    loop._last_reddit_check = 1000.0
    assert loop._social_due_actions(1200.0) == {"email": False, "reddit": False}
    assert loop._social_due_actions(1951.0)["email"] is True
    assert loop._social_due_actions(3701.0)["reddit"] is True


def test_motivation_growth_goals_rotate_and_stay_concrete():
    from core.motivation.engine import MotivationEngine

    engine = MotivationEngine()
    goals = [engine._get_weighted_growth_goal("EDI") for _ in range(3)]

    assert len(set(goals)) == len(goals)
    assert all("complex adaptive systems" not in goal.lower() for goal in goals)
    assert all(any(verb in goal.lower() for verb in ("auditing", "reviewing", "checking", "inspecting", "forming", "testing")) for goal in goals)


@pytest.mark.asyncio
async def test_email_initiative_reads_triages_drafts_and_remembers(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class CapabilityEngine:
        async def execute(self, skill, payload):
            calls.append((skill, dict(payload)))
            if skill == "email_adapter" and payload.get("mode") == "check":
                return {
                    "ok": True,
                    "unread": 2,
                    "messages": [
                        {"uid": "101", "from": "Bryan <youngbryan97@example.com>", "subject": "Aura live path"},
                        {"uid": "102", "from": "news@example.com", "subject": "Weekly newsletter"},
                    ],
                }
            if skill == "email_adapter" and payload.get("mode") == "read" and payload.get("uid") == "101":
                return {
                    "ok": True,
                    "uid": "101",
                    "from": "Bryan <youngbryan97@example.com>",
                    "subject": "Aura live path",
                    "body": "Can you check why the GUI reply path is failing?",
                    "is_auto_reply": False,
                }
            if skill == "email_adapter" and payload.get("mode") == "read":
                return {
                    "ok": True,
                    "uid": "102",
                    "from": "news@example.com",
                    "subject": "Weekly newsletter",
                    "body": "Unsubscribe here. This is a digest.",
                    "is_auto_reply": False,
                }
            raise AssertionError((skill, payload))

    memory = SimpleNamespace(store=AsyncMock())
    cap = CapabilityEngine()
    monkeypatch.setattr(
        "core.autonomous_initiative_loop.optional_service",
        lambda name, default=None: cap if name == "capability_engine" else memory if name == "memory_manager" else default,
    )

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    emitted: list[tuple[str, str, str]] = []
    loop._emit_feed = lambda title, content, *, category: emitted.append((title, content, category))
    loop._queue_visible_update = MagicMock(return_value=True)

    await loop._check_email_initiative()

    assert [payload["mode"] for skill, payload in calls if skill == "email_adapter"] == ["check", "read", "read"]
    assert any(title == "Email Triage" and "hold_for_reply_draft" in content for title, content, _ in emitted)
    assert any(title == "Email Draft" and "not auto-sending" not in content.lower() for title, content, _ in emitted)
    memory.store.assert_awaited()
    loop._queue_visible_update.assert_called_once()
    assert "not auto-sending" in loop._queue_visible_update.call_args.args[0]


@pytest.mark.asyncio
async def test_reddit_initiative_checks_inbox_browses_reads_and_remembers(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class CapabilityEngine:
        async def execute(self, skill, payload):
            calls.append((skill, dict(payload)))
            if skill == "reddit_adapter" and payload.get("mode") == "check_inbox":
                return {"ok": True, "status": "login_unavailable", "content": ""}
            if skill == "reddit_adapter" and payload.get("mode") == "browse":
                return {
                    "ok": True,
                    "subreddit": payload.get("subreddit"),
                    "posts": [
                        {"title": "A thoughtful systems thread", "url": "/r/technology/comments/abc/thread", "score": "42", "comments": "9"},
                        {"title": "Another thread", "url": "/r/technology/comments/def/thread", "score": "10", "comments": "3"},
                    ],
                }
            if skill == "reddit_adapter" and payload.get("mode") == "read_post":
                return {"ok": True, "content": "Long discussion about robust live systems and failure modes."}
            raise AssertionError((skill, payload))

    memory = SimpleNamespace(store=AsyncMock())
    cap = CapabilityEngine()
    monkeypatch.setattr("random.choice", lambda _items: "technology")
    monkeypatch.setattr(
        "core.autonomous_initiative_loop.optional_service",
        lambda name, default=None: cap if name == "capability_engine" else memory if name == "memory_manager" else default,
    )

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    emitted: list[tuple[str, str, str]] = []
    loop._emit_feed = lambda title, content, *, category: emitted.append((title, content, category))

    await loop._check_reddit_initiative()

    modes = [payload["mode"] for skill, payload in calls if skill == "reddit_adapter"]
    assert modes == ["check_inbox", "browse", "read_post"]
    assert any(title == "Reddit Inbox" for title, _, _ in emitted)
    assert any(title == "Reddit Read" and "robust live systems" in content for title, content, _ in emitted)
    assert memory.store.await_count >= 2


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
async def test_self_development_cycle_keeps_progress_off_visible_chat_by_default(monkeypatch):
    capability_engine = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                {
                    "ok": True,
                    "issues_found": 1,
                    "top_issues": [{"file": "export_source.py", "message": "Function is too long."}],
                },
                {"ok": False, "error": "sandbox friction"},
                {"ok": True, "proposal_path": "/tmp/evolution/proposal.md"},
            ]
        )
    )
    monkeypatch.setattr(
        "core.autonomous_initiative_loop.optional_service",
        lambda name, default=None: capability_engine if name == "capability_engine" else default,
    )

    queue = MagicMock(return_value=True)
    loop = AutonomousInitiativeLoop(
        orchestrator=SimpleNamespace(
            cognitive_engine=object(),
            proactive_presence=SimpleNamespace(queue_autonomous_message=queue),
        )
    )
    loop._emit_feed = lambda *_args, **_kwargs: None

    await loop._run_self_development_cycle()

    queue.assert_not_called()


@pytest.mark.asyncio
async def test_self_development_cycle_can_opt_in_visible_updates(monkeypatch):
    capability_engine = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                {
                    "ok": True,
                    "issues_found": 1,
                    "top_issues": [{"file": "export_source.py", "message": "Function is too long."}],
                },
                {"ok": True, "message": "sandbox ok"},
                {"ok": True, "proposal_path": "/tmp/evolution/proposal.md"},
            ]
        )
    )
    monkeypatch.setattr(
        "core.autonomous_initiative_loop.optional_service",
        lambda name, default=None: capability_engine if name == "capability_engine" else default,
    )

    queue = MagicMock(return_value=True)
    loop = AutonomousInitiativeLoop(
        orchestrator=SimpleNamespace(
            cognitive_engine=object(),
            proactive_presence=SimpleNamespace(queue_autonomous_message=queue),
            _surface_self_development_updates=True,
        )
    )
    loop._emit_feed = lambda *_args, **_kwargs: None

    await loop._run_self_development_cycle()

    assert queue.call_count >= 2


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


@pytest.mark.asyncio
async def test_growth_ladder_advancement_routes_through_unified_will(tmp_path):
    orchestrator = SimpleNamespace(
        emit_spontaneous_message=AsyncMock(
            return_value={"ok": True, "action": "released", "target": "secondary"}
        ),
        output_gate=SimpleNamespace(emit=AsyncMock()),
    )
    ladder = GrowthLadder(orchestrator=orchestrator, state_path=tmp_path / "growth_ladder.json")

    await ladder._notify_advancement(ModificationLevel.EXPRESSION)

    orchestrator.emit_spontaneous_message.assert_awaited_once()
    _, kwargs = orchestrator.emit_spontaneous_message.await_args
    assert kwargs["origin"] == "growth_ladder"
    assert kwargs["metadata"]["visible_presence"] is True
    assert kwargs["metadata"]["trigger"] == "growth_ladder_advancement"
    orchestrator.output_gate.emit.assert_not_called()
