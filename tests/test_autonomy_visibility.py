from __future__ import annotations

import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.autonomy.autonomous_initiative_loop import AutonomousInitiativeLoop
from core.autonomy.proactive_presence import ProactivePresence
from core.orchestrator.mixins.output_formatter import OutputFormatterMixin
from core.self_modification.growth_ladder import GrowthLadder, ModificationLevel

PROPOSAL_PATH = str(Path(tempfile.gettempdir()) / "evolution" / "proposal.md")


class RecordedCall:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self.args
        yield self.kwargs


class AsyncCallRecorder:
    def __init__(self, result=None, *, side_effect=None):
        self.result = result
        self.side_effect = side_effect
        self.await_args_list = []
        self.await_args = None

    @property
    def await_count(self):
        return len(self.await_args_list)

    async def __call__(self, *args, **kwargs):
        call = RecordedCall(args, kwargs)
        self.await_args_list.append(call)
        self.await_args = call
        if isinstance(self.side_effect, list):
            value = self.side_effect.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        if callable(self.side_effect):
            return self.side_effect(*args, **kwargs)
        return self.result

    def assert_awaited(self):
        assert self.await_args_list

    def assert_awaited_once(self):
        assert len(self.await_args_list) == 1

    def assert_not_called(self):
        assert not self.await_args_list


class CallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []
        self.call_args = None

    @property
    def call_count(self):
        return len(self.calls)

    def __call__(self, *args, **kwargs):
        call = RecordedCall(args, kwargs)
        self.calls.append(call)
        self.call_args = call
        return self.result

    def assert_called_once_with(self, *args, **kwargs):
        assert len(self.calls) == 1
        call = self.calls[0]
        assert call.args == args
        assert call.kwargs == kwargs

    def assert_called_once(self):
        assert len(self.calls) == 1

    def assert_not_called(self):
        assert not self.calls


def _simulate_idle_background_runtime(monkeypatch) -> None:
    """Make background-admission tests independent of host boot/foreground state."""
    import core.runtime.background_policy as background_policy

    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "0")
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    monkeypatch.setattr(
        "core.runtime.proof_policy.proof_run_active",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "core.runtime.foreground_guard.foreground_activity_reason",
        lambda: "",
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda _name, default=None: default,
    )
    monkeypatch.setattr(background_policy, "_read_compute_pressure_reason", lambda: "")
    monkeypatch.setattr(
        background_policy,
        "_read_memory_pressure_snapshot",
        lambda: background_policy._MemoryPressureSnapshot(
            pressure_pct=35.0,
            reason="memory_pressure_35.0",
            refuse_heavy_local_generation=False,
        ),
    )
    monkeypatch.setattr(
        background_policy,
        "get_unified_failure_state",
        lambda: {"pressure": 0.0},
    )


def test_emit_thought_stream_falls_back_to_thought_emitter(monkeypatch):
    emitter = SimpleNamespace(emit=CallRecorder())
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
    from core.autonomy.autonomous_initiative_loop import _background_initiative_allowed

    _simulate_idle_background_runtime(monkeypatch)

    orchestrator = SimpleNamespace(
        _last_user_interaction_time=time.time() - 3600.0,
        is_busy=False,
        _suppress_unsolicited_proactivity_until=0.0,
        _foreground_user_quiet_until=0.0,
    )

    assert _background_initiative_allowed(orchestrator) is True


def test_autonomous_initiative_status_exposes_admission_reasons(monkeypatch):
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop._background_initiative_blocker",
        lambda _orchestrator=None: "",
    )
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop._self_development_blocker",
        lambda _orchestrator=None: "recent_user_12",
    )
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop._passive_social_blocker",
        lambda _orchestrator=None: "memory_pressure_83.0",
    )

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    loop.running = True

    status = loop.get_status()

    assert status["admission"]["world_and_knowledge"] == "allowed"
    assert status["admission"]["self_development"] == "recent_user_12"
    assert status["admission"]["social"] == "memory_pressure_83.0"


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
    calls: list[tuple[str, dict, dict]] = []

    class CapabilityEngine:
        async def execute(self, skill, payload, context):
            calls.append((skill, dict(payload), dict(context)))
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

    memory = SimpleNamespace(store=AsyncCallRecorder())
    cap = CapabilityEngine()
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop.optional_service",
        lambda name, default=None: cap if name == "capability_engine" else memory if name == "memory_manager" else default,
    )

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    emitted: list[tuple[str, str, str]] = []
    loop._emit_feed = lambda title, content, *, category: emitted.append((title, content, category))
    loop._queue_visible_update = CallRecorder(result=True)

    await loop._check_email_initiative()

    assert [payload["mode"] for skill, payload, _ in calls if skill == "email_adapter"] == ["check", "read", "read"]
    assert all(
        context["origin"] == "autonomous_initiative_loop"
        and context["intent_source"] == "autonomous_initiative_loop"
        and context["user_facing"] is False
        for skill, _, context in calls
        if skill == "email_adapter"
    )
    assert any(title == "Email Triage" and "hold_for_reply_draft" in content for title, content, _ in emitted)
    assert any(title == "Email Draft" and "not auto-sending" not in content.lower() for title, content, _ in emitted)
    memory.store.assert_awaited()
    loop._queue_visible_update.assert_called_once()
    assert "not auto-sending" in loop._queue_visible_update.call_args.args[0]


@pytest.mark.asyncio
async def test_reddit_initiative_checks_inbox_browses_reads_and_remembers(monkeypatch):
    calls: list[tuple[str, dict, dict]] = []

    class CapabilityEngine:
        async def execute(self, skill, payload, context):
            calls.append((skill, dict(payload), dict(context)))
            if skill == "reddit_adapter" and payload.get("mode") == "check_inbox":
                return {"ok": True, "status": "login_unavailable", "content": ""}
            if skill == "reddit_adapter" and payload.get("mode") == "browse":
                return {
                    "ok": True,
                    "subreddit": payload.get("subreddit"),
                    "provider": {"state": "session_unverified"},
                    "posts": [
                        {"title": "A thoughtful systems thread", "url": "/r/technology/comments/abc/thread", "score": "42", "comments": "9"},
                        {"title": "Another thread", "url": "/r/technology/comments/def/thread", "score": "10", "comments": "3"},
                    ],
                }
            if skill == "reddit_adapter" and payload.get("mode") == "read_post":
                return {"ok": True, "content": "Long discussion about robust live systems and failure modes."}
            raise AssertionError((skill, payload))

    memory = SimpleNamespace(store=AsyncCallRecorder())
    cap = CapabilityEngine()
    monkeypatch.setattr("random.choice", lambda _items: "technology")
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop.optional_service",
        lambda name, default=None: cap if name == "capability_engine" else memory if name == "memory_manager" else default,
    )

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    emitted: list[tuple[str, str, str]] = []
    loop._emit_feed = lambda title, content, *, category: emitted.append((title, content, category))

    await loop._check_reddit_initiative()

    modes = [payload["mode"] for skill, payload, _ in calls if skill == "reddit_adapter"]
    assert modes == ["browse", "read_post", "check_inbox"]
    assert all(
        context["origin"] == "autonomous_initiative_loop"
        and context["intent_source"] == "autonomous_initiative_loop"
        and context["user_facing"] is False
        for skill, _, context in calls
        if skill == "reddit_adapter"
    )
    assert any(title == "Reddit Inbox" for title, _, _ in emitted)
    assert any(title == "Reddit Read" and "robust live systems" in content for title, content, _ in emitted)
    assert memory.store.await_count >= 2


@pytest.mark.asyncio
async def test_self_development_cycle_runs_scan_tests_and_proposal(monkeypatch):
    capability_engine = SimpleNamespace(
        execute=AsyncCallRecorder(
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
                    "proposal_path": PROPOSAL_PATH,
                },
            ]
        )
    )
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop.optional_service",
        lambda name, default=None: capability_engine if name == "capability_engine" else default,
    )

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace(cognitive_engine=object()))
    emitted: list[tuple[str, str, str]] = []
    loop._emit_feed = lambda title, content, *, category: emitted.append((title, content, category))

    await loop._run_self_development_cycle()

    calls = capability_engine.execute.await_args_list
    assert [call.args[0] for call in calls] == ["auto_refactor", "test_generator", "self_evolution"]
    assert calls[1].args[1]["read_only"] is True
    assert calls[2].args[1]["read_only"] is True
    assert any("sandbox tests" in content.lower() for _, content, _ in emitted)
    assert any("proposal" in content.lower() or "saved to" in content.lower() for _, content, _ in emitted)


@pytest.mark.asyncio
async def test_self_development_cycle_keeps_progress_off_visible_chat_by_default(monkeypatch):
    capability_engine = SimpleNamespace(
        execute=AsyncCallRecorder(
            side_effect=[
                {
                    "ok": True,
                    "issues_found": 1,
                    "top_issues": [{"file": "export_source.py", "message": "Function is too long."}],
                },
                {"ok": False, "error": "sandbox friction"},
                {"ok": True, "proposal_path": PROPOSAL_PATH},
            ]
        )
    )
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop.optional_service",
        lambda name, default=None: capability_engine if name == "capability_engine" else default,
    )

    queue = CallRecorder(result=True)
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
        execute=AsyncCallRecorder(
            side_effect=[
                {
                    "ok": True,
                    "issues_found": 1,
                    "top_issues": [{"file": "export_source.py", "message": "Function is too long."}],
                },
                {"ok": True, "message": "sandbox ok"},
                {"ok": True, "proposal_path": PROPOSAL_PATH},
            ]
        )
    )
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop.optional_service",
        lambda name, default=None: capability_engine if name == "capability_engine" else default,
    )

    queue = CallRecorder(result=True)
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
@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("deferral", [
    {"ok": False, "status": "deferred", "reason": "foreground_generation_active"},
    {"ok": False, "status": "deferred_by_executive", "deferred": True, "reason": "retry_later"},
    {"ok": False, "error": "background_deferred:foreground_quiet_window"},
])
async def test_self_development_deferral_stops_dependent_work_without_failure_feedback(
    monkeypatch, stage, deferral,
):
    results = [
        {"ok": True, "top_issues": [{"file": "example.py", "message": "complexity"}]},
        {"ok": True, "message": "tests passed"},
        {"ok": True, "proposal_path": PROPOSAL_PATH},
    ]
    results[stage] = deferral
    capability = SimpleNamespace(execute=AsyncCallRecorder(side_effect=results))
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop.optional_service",
        lambda name, default=None: capability if name == "capability_engine" else default,
    )
    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace(cognitive_engine=object()))
    loop._curriculum_practice_step = AsyncCallRecorder()
    loop._emit_feed = CallRecorder()
    loop._queue_visible_update = CallRecorder()
    await loop._run_self_development_cycle()
    assert capability.execute.await_count == stage + 1
    messages = [call.args[1] for call in loop._emit_feed.calls]
    assert "deferred:" in messages[-1]
    assert not any("unknown error" in text or "friction" in text or "Scan stalled" in text for text in messages)
    assert not any("gate held" in call.args[0] for call in loop._queue_visible_update.calls)


@pytest.mark.asyncio
async def test_proactive_presence_prefers_visible_primary(monkeypatch):
    _simulate_idle_background_runtime(monkeypatch)
    orchestrator = SimpleNamespace(
        emit_spontaneous_message=AsyncCallRecorder(
            {
                "ok": True,
                "action": "released",
                "target": "primary",
            }
        ),
        _last_thought_time=0.0,
    )
    emitter = SimpleNamespace(emit=CallRecorder())
    terminal = SimpleNamespace(queue_autonomous_message=CallRecorder())
    monkeypatch.setattr("core.thought_stream.get_emitter", lambda: emitter)
    monkeypatch.setattr("core.conversation.terminal_chat.get_terminal_fallback", lambda: terminal)

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
async def test_proactive_presence_cannot_publish_during_process_foreground_lease(monkeypatch):
    import core.runtime.foreground_guard as foreground_guard

    monkeypatch.setenv("AURA_BACKGROUND_BOOT_GRACE_S", "0")
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason", lambda *_a, **_k: ""
    )
    orchestrator = SimpleNamespace(
        emit_spontaneous_message=AsyncCallRecorder(
            {"ok": True, "action": "released", "target": "primary"}
        ),
        _last_thought_time=0.0,
    )
    presence = ProactivePresence(orchestrator=orchestrator)
    foreground_guard._reset_for_tests()
    lease = foreground_guard.begin_foreground_turn(owner="test", source="desktop-ui")
    try:
        await presence._emit("I found something relevant to this conversation.")
    finally:
        lease.close()
        foreground_guard._reset_for_tests()

    orchestrator.emit_spontaneous_message.assert_not_called()
    assert presence._outputs_this_hour == 0


@pytest.mark.asyncio
async def test_proactive_presence_rejects_backend_failure_text_from_visible_chat(monkeypatch):
    _simulate_idle_background_runtime(monkeypatch)
    orchestrator = SimpleNamespace(
        emit_spontaneous_message=AsyncCallRecorder(
            {
                "ok": True,
                "action": "released",
                "target": "primary",
            }
        ),
        _last_thought_time=0.0,
    )
    emitter = SimpleNamespace(emit=CallRecorder())
    terminal = SimpleNamespace(queue_autonomous_message=CallRecorder())
    monkeypatch.setattr("core.thought_stream.get_emitter", lambda: emitter)
    monkeypatch.setattr("core.conversation.terminal_chat.get_terminal_fallback", lambda: terminal)

    presence = ProactivePresence(orchestrator=orchestrator)
    await presence._emit(
        "I could not produce a reliable answer because the reasoning backend failed before returning usable text.",
    )

    assert orchestrator.emit_spontaneous_message.await_count == 0
    assert emitter.emit.calls == []
    assert terminal.queue_autonomous_message.calls == []
    assert presence._outputs_this_hour == 0


@pytest.mark.asyncio
async def test_proactive_presence_requeues_visible_update_when_primary_is_temporarily_held(monkeypatch):
    _simulate_idle_background_runtime(monkeypatch)
    orchestrator = SimpleNamespace(
        emit_spontaneous_message=AsyncCallRecorder(
            {
                "ok": True,
                "action": "released",
                "target": "secondary",
                "reason": "user_recently_active",
            }
        ),
        _last_thought_time=0.0,
    )
    emitter = SimpleNamespace(emit=CallRecorder())
    terminal = SimpleNamespace(queue_autonomous_message=CallRecorder())
    monkeypatch.setattr("core.thought_stream.get_emitter", lambda: emitter)
    monkeypatch.setattr("core.conversation.terminal_chat.get_terminal_fallback", lambda: terminal)

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


def test_proactive_presence_allows_queued_visible_updates_during_away_mode(monkeypatch):
    _simulate_idle_background_runtime(monkeypatch)
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
        emit_spontaneous_message=AsyncCallRecorder(
            {"ok": True, "action": "released", "target": "secondary"}
        ),
        output_gate=SimpleNamespace(emit=AsyncCallRecorder()),
    )
    ladder = GrowthLadder(orchestrator=orchestrator, state_path=tmp_path / "growth_ladder.json")

    await ladder._notify_advancement(ModificationLevel.EXPRESSION)

    orchestrator.emit_spontaneous_message.assert_awaited_once()
    _, kwargs = orchestrator.emit_spontaneous_message.await_args
    assert kwargs["origin"] == "growth_ladder"
    assert kwargs["metadata"]["visible_presence"] is True
    assert kwargs["metadata"]["trigger"] == "growth_ladder_advancement"
    orchestrator.output_gate.emit.assert_not_called()
