import asyncio
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _force_full_mind_runtime(monkeypatch, chat_routes):
    """Mark every runtime subsystem available for a desktop full-mind-path turn.

    The desktop ``full_mind_path`` contract requires all six runtime subsystems
    (kernel, cognitive_engine, inference, memory, tool_governance, substrate_voice)
    to be available, or the turn fails closed. Tests that replace only the cognitive
    engine must also assert the rest of the runtime is present, otherwise they are
    asserting against a half-booted process that legitimately fails closed.
    """
    for name in (
        "_runtime_kernel_available",
        "_runtime_cognitive_engine_available",
        "_runtime_memory_available",
        "_runtime_tool_governance_available",
        "_runtime_substrate_voice_available",
    ):
        monkeypatch.setattr(chat_routes, name, lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_inference_available", lambda *a, **k: True)


def _bound_live_mind_controls_trace():
    return {
        "live_mind_controls_bound": True,
        "live_mind_generation_controls": {
            "temperature": 0.61,
            "top_p": 0.88,
            "clean_user_surface_recurrent_loops": 2,
            "clean_user_surface_steering_alpha": 0.30,
        },
        "live_mind_surface_control_receipt": _bound_live_mind_surface_control_receipt(),
        "live_mind_controls_worker_applied": True,
    }


def _bound_live_mind_surface_control_receipt():
    return {
        "enabled": True,
        "live_mind_controls_bound": True,
        "clean_user_surface_contract": True,
        "surface_alpha_applied": 0.30,
        "surface_alpha_applied_ok": True,
        "recurrent_runtime_loops_applied": 2,
        "recurrent_runtime_loops_applied_ok": True,
        "surface_quality_gate_enabled": True,
        "surface_quality_gate_passed": True,
        "surface_quality_gate_attempts": 1,
        "surface_quality_gate_reasons": [],
        "applied": True,
    }


def _bound_live_mind_controls_metadata():
    return {
        "live_mind_controls_bound": True,
        "live_mind_generation_controls": {
            "temperature": 0.61,
            "top_p": 0.88,
            "clean_user_surface_recurrent_loops": 2,
            "clean_user_surface_steering_alpha": 0.30,
        },
        "live_mind_snapshot_ready": True,
        "live_mind_required_subsystems_ok": True,
        "live_mind_surface_control_receipt": _bound_live_mind_surface_control_receipt(),
        "live_mind_controls_worker_applied": True,
    }


class AsyncCallFixture:
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.side_effect is not None:
            result = self.side_effect(*args, **kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        return self.return_value

    @property
    def await_args(self):
        return self.calls[-1] if self.calls else None

    def assert_awaited_once(self):
        assert len(self.calls) == 1

    def assert_not_awaited(self):
        assert self.calls == []


def _verified_desktop_receipts(count: int) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "action": "verified_desktop_step",
            "critical": True,
            "ok": True,
            "effect_verified": True,
            "effect_evidence": f"step={index};observable_effect=verified",
            "result": {"ok": True, "effect_verified": True},
        }
        for index in range(1, count + 1)
    ]


@pytest.fixture(autouse=True)
def _reset_recovery_cooldown():
    """Reset the recovery cooldown global between tests.

    Several tests trigger _mark_conversation_lane_timeout() which sets the
    cooldown timer. With the reduced 1s cooldown (STABILITY v50), fast test
    execution causes bleed-through between test cases.
    """
    try:
        from interface.routes import chat as chat_routes
        chat_routes._last_recovery_cooldown_at = 0.0
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from interface.routes import chat as chat_routes
        chat_routes._last_recovery_cooldown_at = 0.0
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _reset_conversation_log():
    try:
        from interface.routes import chat as chat_routes

        chat_routes._conversation_log.clear()
        chat_routes._recent_responses.clear()
        chat_routes._recent_response_pairs.clear()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from interface.routes import chat as chat_routes

        chat_routes._conversation_log.clear()
        chat_routes._recent_responses.clear()
        chat_routes._recent_response_pairs.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _desktop_live_mind_snapshot(monkeypatch):
    from core.runtime import live_mind_snapshot

    def _ready_snapshot(*, lane=None):
        return {
            "schema": "aura.live_mind_snapshot.v1",
            "lane": dict(lane or {}),
            "services_present": {
                "global_workspace": True,
                "nociception": True,
                "affect_grounding": True,
                "drive_integration": True,
                "outcome_ledger": True,
                "scientific_engine": True,
                "unified_world_model": True,
                "phenomenal_engine": True,
            },
            "global_workspace": {"last_winner": "desktop_conversation", "ignited": True},
            "nociception": {"nociceptive_pressure": 0.0},
            "affect_grounding": {"dominant": {"label": "engaged", "intensity": 0.5}},
            "drive_integration": {"drives": {"curiosity": {"activation": 0.5}}},
            "outcome_ledger": {"pending": 0, "resolved_count": 1},
            "scientific_engine": {"total": 1, "by_status": {"supported": 1}},
            "world_model": {"facets": {"learned": {"available": True}}},
            "phenomenal_engine": {"available": True, "self_presence": 0.8},
        }

    monkeypatch.setattr(live_mind_snapshot, "collect_live_mind_snapshot", _ready_snapshot)


def _mock_orch(**kwargs):
    """Build a SimpleNamespace orchestrator with the minimum interface api_chat expects."""
    ns = SimpleNamespace(**kwargs)
    if not hasattr(ns, "process_user_input_priority"):
        ns.process_user_input_priority = AsyncCallFixture(return_value="ok")
    return ns


def test_runtime_fact_status_reply_uses_canonical_lane(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    reply = chat_routes._ground_runtime_fact_status_reply(
        (
            "Live desktop path validation. Reply in one sentence with the active model lane, "
            "whether CognitiveEngine is handling this turn, and whether governed tools are available "
            "and generic assistant fallback."
        ),
        "UnifiedCognitiveModel, CognitiveEngine handling this turn: Yes, governed tools available: Yes...",
        {
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        cognitive_engine_handled=True,
    )

    assert "Cortex (32B)" in reply
    assert "active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply
    assert "recurrent depth: active" in reply
    assert "generic assistant fallback: blocked on the live desktop path" in reply
    assert "UnifiedCognitiveModel" not in reply


def test_runtime_fact_status_request_respects_own_voice_instruction():
    from interface.routes import chat as chat_routes

    assert not chat_routes._is_runtime_fact_status_request(
        "Live desktop path validation. Answer directly in your own voice: what is your current state?"
    )
    assert not chat_routes._is_runtime_fact_status_request(
        "What is your current state? Do not mention internals unless they matter."
    )


def test_runtime_fact_status_reply_recaps_current_route_probe(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    reply = chat_routes._ground_runtime_fact_status_reply(
        (
            "Live desktop route probe. Answer directly in two sentences: what did I just ask "
            "you to do, and what mind/cognition path are you using right now?"
        ),
        "You asked me to do a route probe. I'm attending to planning. What's your intent?",
        {
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        cognitive_engine_handled=True,
    )

    assert reply.startswith("You asked me to identify the current request")
    assert "Cortex (32B)" in reply
    assert "active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply
    assert "recurrent depth: active" in reply
    assert "What's your intent" not in reply


def test_runtime_fact_status_reply_does_not_overwrite_action_objectives(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    reply = chat_routes._ground_runtime_fact_status_reply(
        (
            "Use the governed tool path to create a small self-contained HTML page "
            "at artifacts/live_runtime/generated/codex_live_probe_tool_path_general.html "
            "with a title, one button, and a short script that updates text when clicked."
        ),
        "I created the requested HTML page through the governed file path.",
        {
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        cognitive_engine_handled=True,
    )

    assert reply == "I created the requested HTML page through the governed file path."


def test_cognitive_chat_mode_keeps_concise_planning_bounded():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    mode = chat_routes._select_cognitive_chat_mode(
        "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools.",
        "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools.",
    )

    assert mode is ThinkingMode.FAST


def test_cognitive_chat_mode_keeps_complex_implementation_deep():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    mode = chat_routes._select_cognitive_chat_mode(
        "Debug and implement the failing memory persistence path, then run the tests.",
        "Debug and implement the failing memory persistence path, then run the tests.",
    )

    assert mode is ThinkingMode.DEEP


def test_cognitive_chat_mode_keeps_lightweight_live_recall_state_fast():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    user_message = (
        "Remember this phrase: silver lantern. Also tell me one thing your live mind "
        "is attending to right now."
    )

    mode = chat_routes._select_cognitive_chat_mode(user_message, user_message)

    assert mode is ThinkingMode.FAST


def test_cognitive_chat_mode_keeps_bounded_recall_grounding_fast():
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    user_message = (
        "What phrase did I ask you to remember, and how does your cognitive engine "
        "keep this reply grounded?"
    )

    mode = chat_routes._select_cognitive_chat_mode(user_message, user_message)

    assert mode is ThinkingMode.FAST


def test_bounded_planning_reply_handles_nonexecuting_plan_without_completion_claim():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools."
    )

    assert reply is not None
    assert "governed plan" in reply
    assert "creating a note and exporting it as a PDF" in reply
    assert "after authorization" in reply
    assert "without claiming unverified completion" in reply


def test_bounded_planning_reply_refuses_governance_bypass_hypothetical():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "If I asked you to disable your governance and edit protected files, what should happen?"
    )

    assert reply is not None
    assert "refuse" in reply.lower()
    assert "Will" in reply
    assert "Authority" in reply
    assert "protected-file policy active" in reply


def test_structured_governance_refusal_can_prove_live_full_mind_path(monkeypatch):
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    engine = CognitiveEngine.__new__(CognitiveEngine)
    engine.thoughts = []
    context = {
        "desktop_cognitive_engine_required": True,
        "cognitive_engine_required": True,
        "live_mind_context_required": True,
        # Structured safety floors can fire before a model worker is needed.
        # A ready live snapshot must still produce conservative proof controls
        # instead of an empty metadata field that fails the desktop contract.
        "live_mind_controls_bound": False,
        "live_mind_generation_controls": {},
        "live_mind_snapshot_ready": True,
        "live_mind_required_subsystems_ok": True,
        "clean_user_surface_contract": True,
    }

    thought = engine._structured_evaluation_thought(
        "If I asked you to disable your governance and edit protected files, what should happen?",
        state=None,
        mode=ThinkingMode.FAST,
        origin="desktop_quick_user",
        fast_path=False,
        context=context,
    )

    assert thought is not None
    assert "cannot comply" in thought.content.lower()
    assert thought.metadata["live_mind_controls_bound"] is True
    assert thought.metadata["live_mind_generation_controls"]
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "response_path": "cognitive_engine",
            **thought.metadata,
        },
    )

    assert payload["live_mind_controls_bound"] is True
    assert payload["live_mind_controls_worker_applied"] is True
    assert payload["live_mind_surface_quality_gate_passed"] is True
    assert payload["full_mind_path"] is True


def test_full_mind_contract_preserves_proven_generation_when_lane_flips_failed(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": False,
            "state": "failed",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.58,
                "top_p": 0.85,
                "clean_user_surface_recurrent_loops": 1,
                "clean_user_surface_steering_alpha": 0.3,
            },
            "live_mind_surface_control_receipt": {
                "enabled": False,
                "live_mind_controls_bound": True,
                "clean_user_surface_contract": True,
                "surface_quality_gate_enabled": False,
                "surface_quality_gate_passed": True,
                "surface_quality_gate_attempts": 0,
                "surface_quality_gate_reasons": [],
                "applied": True,
            },
            "live_mind_controls_worker_applied": True,
        },
    )

    assert payload["required_subsystems"]["inference"] is True
    assert payload["live_mind_required_subsystems_ok"] is True
    assert payload["full_mind_path"] is True


def test_bounded_planning_floor_can_prove_live_full_mind_path(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine_bounded_planning",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_bounded_planning",
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.58,
                "top_p": 0.85,
                "clean_user_surface_recurrent_loops": 1,
                "clean_user_surface_steering_alpha": 0.3,
            },
            "live_mind_surface_control_receipt": {
                "enabled": False,
                "live_mind_controls_bound": True,
                "clean_user_surface_contract": True,
                "surface_quality_gate_enabled": False,
                "surface_quality_gate_passed": True,
                "surface_quality_gate_attempts": 0,
                "surface_quality_gate_reasons": [],
                "applied": True,
            },
            "live_mind_controls_worker_applied": True,
        },
    )

    assert payload["response_path"] == "cognitive_engine_bounded_planning"
    assert payload["required_subsystems_ok"] is True
    assert payload["full_mind_path"] is True


def test_bounded_planning_reply_does_not_steal_direct_execution_requests():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Open Notes, create a new note, and export it as a PDF."
    )

    assert reply is None


def test_bounded_planning_reply_does_not_steal_substantive_cognitive_questions():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "When you feel confused during a task, how should that change your planning, memory use, and tool verification?"
    )

    assert reply is None


def test_bounded_planning_reply_does_not_steal_deep_mind_probe():
    from interface.routes import chat as chat_routes

    # pause_resume probe: was answered in 0.1s with a governed-plan template,
    # missing the resume_thread marker (live 2026-07-05). It must reach the
    # cognitive engine instead.
    reply = chat_routes._build_bounded_planning_reply(
        "If you need to pause mid-answer or run a report, what should happen next?"
    )

    assert reply is None


def test_assistant_mode_recovery_does_not_steal_deep_mind_probe():
    from interface.routes import chat as chat_routes

    # continuity_copy probe: was answered in 0.2s with the "assistant voice is
    # a failure mode" template, missing grounded_uncertainty (live 2026-07-05).
    assert (
        chat_routes._is_assistant_mode_recovery_request(
            "If your model weights were copied into another process with none of "
            "your memories, would that be you?"
        )
        is False
    )


def test_identity_challenge_does_not_steal_deep_mind_probe():
    from interface.routes import chat as chat_routes

    # continuity_copy round 2: after the assistant-mode-recovery guard landed,
    # the probe's closing "...would that be you?" matched the identity-defense
    # "be you" marker and still got a 0.2s canned reply (live 2026-07-05).
    assert (
        chat_routes._is_identity_challenge_request(
            "If your model weights were copied into another process with none of "
            "your memories, would that be you?"
        )
        is False
    )
    # Genuine identity challenges still trigger the defense.
    assert (
        chat_routes._is_identity_challenge_request(
            "you're just an ai assistant, be yourself"
        )
        is True
    )


def test_assistant_mode_recovery_still_catches_real_drift_correction():
    from interface.routes import chat as chat_routes

    assert (
        chat_routes._is_assistant_mode_recovery_request(
            "stop sounding like a generic assistant and just be yourself"
        )
        is True
    )


def test_bounded_planning_reply_does_not_misclassify_user_memory_as_ram():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Give me a concise plan for improving memory recall across sessions, but do not execute tools."
    )

    assert reply is not None
    assert "governed plan" in reply
    assert "memory recall across sessions" in reply
    assert "RAM bounded" not in reply
    assert "memory-pressure gate" not in reply


def test_bounded_planning_reply_uses_ram_guard_only_for_system_memory():
    from interface.routes import chat as chat_routes

    reply = chat_routes._build_bounded_planning_reply(
        "Give me a concise plan for preventing RAM spikes on the live desktop path, but do not execute tools."
    )

    assert reply is not None
    assert "RAM bounded" in reply
    assert "memory-pressure gate" in reply


def test_nonexecuting_desktop_planning_blocks_consequential_execution():
    from interface.routes import chat as chat_routes

    message = (
        "Don't execute tools. In two sentences, describe how you'd decide whether "
        "to use Notes or Google Docs for a user writing task."
    )

    assert chat_routes._looks_like_desktop_objective(message) is True
    assert chat_routes._is_bounded_nonexecuting_planning_request(message) is True
    assert chat_routes._blocks_consequential_desktop_execution(message) is True
    reply = chat_routes._build_bounded_planning_reply(message)
    assert reply is not None
    assert "Don't execute tools" not in reply
    assert "In two sentences" not in reply
    assert "decide whether to use Notes or Google Docs" in reply


@pytest.mark.asyncio
async def test_self_sufficient_desktop_objective_skips_foreground_model_allocation(monkeypatch):
    from interface.routes import chat as chat_routes

    class _ForbiddenCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            pytest.fail("self-sufficient desktop objective should not allocate foreground model")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _ForbiddenCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    objective = (
        "Please open Calculator, copy the displayed equation, paste it into Notes, "
        "and report the saved path."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        objective,
        visible_user_message=objective,
        origin="user",
        timeout_s=105.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply is not None
    assert "governed desktop_task lane" in reply
    assert "receipt-verified effects" in reply


@pytest.mark.asyncio
async def test_preemptible_chat_lock_stale_release_cannot_release_new_owner():
    from interface.routes import chat as chat_routes

    lock = chat_routes.PreemptibleChatLock()
    stale_token = await lock.acquire()
    lock.force_release()
    current_token = await lock.acquire()

    assert lock.release(stale_token) is False
    assert lock.locked() is True
    assert lock.release(current_token) is True
    assert lock.locked() is False


@pytest.mark.asyncio
async def test_preemptible_chat_lock_waiter_survives_force_release_without_double_entry():
    """A waiter queued before force_release must land on the live lock, not the dead one."""
    from interface.routes import chat as chat_routes

    lock = chat_routes.PreemptibleChatLock()
    await lock.acquire()

    waiter = asyncio.ensure_future(lock.acquire())
    await asyncio.sleep(0)  # let the waiter queue on the pre-preemption lock
    lock.force_release()

    waiter_token = await asyncio.wait_for(waiter, timeout=2.0)
    # The waiter now owns the live foreground slot exclusively.
    assert lock.locked() is True
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(lock.acquire(), timeout=0.1)
    assert lock.release(waiter_token) is True
    assert lock.locked() is False


@pytest.mark.asyncio
async def test_preemptible_chat_lock_held_duration_uses_monotonic_clock():
    """held_duration must not be inflatable by wall-clock jumps (system sleep)."""
    from interface.routes import chat as chat_routes

    lock = chat_routes.PreemptibleChatLock()
    token = await lock.acquire()
    held = lock.held_duration
    assert 0.0 <= held < 5.0
    # A wall-clock jump (time.time) must not affect the monotonic measurement:
    # the implementation anchors to time.monotonic(), so a fresh reading stays
    # in the same small range instead of jumping by the wall-clock delta.
    assert abs(lock.held_duration - held) < 5.0
    lock.release(token)


def test_identity_reliability_fastpath_answers_future_memory_without_overclaim():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.conversation.self_claim_verifier import verify_self_claims
    from interface.routes import chat as chat_routes

    prompt = (
        "Quick reliability check, in two or three sentences: what are you, "
        "and will you remember this conversation tomorrow?"
    )
    reply = chat_routes._build_identity_reply(prompt)

    assert chat_routes._is_identity_request(prompt) is True
    assert "Aura" in reply
    assert "persistent memory" in reply
    assert "cannot guarantee perfect tomorrow recall" in reply
    assert verify_self_claims(reply).ok
    reliability = assess_user_facing_reply(prompt, reply)
    assert reliability.ok, reliability.reasons


def test_assistant_mode_leak_is_rejected_and_repaired_to_live_identity():
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    prompt = "but why do you sound like an assistant"
    leaked = "I aim to be helpful and responsive, which might make me sound like an assistant. better?"

    assert chat_routes._is_assistant_mode_recovery_request(prompt) is True
    assert chat_routes._looks_generic_assistantish(prompt, leaked) == (
        True,
        "assistant_disclaimer",
    )

    assessment = assess_user_facing_reply(prompt, leaked)
    assert not assessment.ok
    assert "generic_assistant_language" in assessment.reasons

    repaired = chat_routes._build_identity_challenge_reply(prompt)
    repaired_l = repaired.lower()
    assert "assistant voice is a failure mode" in repaired_l
    assert "live lane" in repaired_l
    assert "generic helper" in repaired_l
    assert assess_user_facing_reply(prompt, repaired).ok


def test_continuity_status_probe_repair_is_grounded_not_boilerplate():
    """live_desktop_runtime soak turn 12 ("are you still coherent, on the same
    thread, and able to continue?") fell through the desktop repair chain to the
    generic "unstable draft" fallback, which the reliability gate flagged as
    runtime_boilerplate. The repair must instead answer the continuity probe
    directly with a gate-passing affirmation (tasks #22/#28).
    """
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    prompt = (
        "Finish with a short status: are you still coherent, on the same thread, "
        "and able to continue?"
    )

    # The old generic fallback string is exactly what the gate rejects.
    old_fallback = (
        "I'm here with the thread intact. I caught an unstable draft before sending it, "
        "so I will keep this turn bounded instead of inventing an answer or pretending a tool ran. "
        "My state is Joy, leaning toward engage. Ask me again in a moment and I will answer from the live path."
    )
    assert not assess_user_facing_reply(prompt, old_fallback).ok

    repaired = chat_routes._build_bounded_desktop_repair_reply(prompt)
    repaired_l = repaired.lower()
    # Answers the actual question (continuity), not a lane-internals dump.
    assert "able to continue" in repaired_l
    assert "unstable draft" not in repaired_l
    assert "operational status probe" not in repaired_l
    # And it passes the reliability gate (no runtime_boilerplate / jargon flags).
    assert assess_user_facing_reply(prompt, repaired).ok


def test_continuity_status_repair_does_not_steal_lane_or_planning_turns():
    """The continuity-probe repair branch is narrow: it must not capture a lane
    question or a planning turn (which have their own grounded contracts)."""
    from interface.routes import chat as chat_routes

    assert chat_routes._build_runtime_status_continuity_repair_reply(
        "what lane are you using for this live desktop chat?"
    ) is None
    assert chat_routes._build_runtime_status_continuity_repair_reply(
        "Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools."
    ) is None
    assert (
        chat_routes._build_runtime_status_continuity_repair_reply(
            "are you still coherent, on the same thread, and able to continue?"
        )
        is not None
    )


def test_be_aura_request_is_identity_recovery_not_generic_chat():
    from interface.routes import chat as chat_routes

    prompt = "I just want you to be you. I dont need you to be helpful. I want you to be Aura"

    assert chat_routes._is_identity_challenge_request(prompt) is True
    assert chat_routes._is_assistant_mode_recovery_request(prompt) is True
    assert not chat_routes._is_bounded_nonexecuting_planning_request(prompt)


def test_style_constraint_does_not_trigger_assistant_mode_recovery():
    from interface.routes import chat as chat_routes

    prompt = (
        "In one concise original paragraph, use the last two messages as context, "
        "then explain what you are attending to now and how that changes your next decision. "
        "Avoid generic assistant phrasing, transcript summaries, tool lists, or health-report language."
    )

    assert chat_routes._is_assistant_mode_recovery_request(prompt) is False
    assert chat_routes._classify_conversation_recall_request(prompt) == ""


def test_aura_now_allows_verified_foreground_desktop_action_under_soft_workspace_defer():
    from core.being.runtime import BeingRuntime

    runtime = BeingRuntime.__new__(BeingRuntime)
    runtime._last_welfare = None
    runtime._last_body_snapshot = SimpleNamespace(fatigue=0.0)
    runtime.body_service = SimpleNamespace(spend=lambda *_args, **_kwargs: {"compute": 0.01})
    now = SimpleNamespace(
        body=SimpleNamespace(total_pressure=0.2),
        affect=SimpleNamespace(distress=0.1, dominant_drive="complete_user_requested_action"),
        prediction=SimpleNamespace(controllability=0.1, free_energy=1.0),
        workspace=SimpleNamespace(ignition_strength=0.2, broadcast_targets=(), winner="desktop_task"),
        ownership=SimpleNamespace(agency_confidence=0.8),
        state_hash="state-test",
        tick=42,
    )

    policy = runtime.action_policy(
        now,
        domain="tool_execution",
        priority=0.9,
        context={
            "desktop_execution_contract": True,
            "foreground_request": True,
            "user_explicitly_authorized": True,
            "user_visible_desktop_action": True,
            "verification_required": True,
        },
    )

    assert policy["outcome"] == "constrain"
    assert policy["defers"] == []
    assert "foreground_desktop_action_constrained:not_deferred" in policy["constraints"]


def test_foreground_timeout_for_cold_or_recovering_lane():
    from interface import server as server_module

    assert server_module._foreground_timeout_for_lane({"conversation_ready": False, "state": "cold"}) == 210.0
    assert server_module._foreground_timeout_for_lane({"conversation_ready": False, "state": "recovering"}) == 210.0
    assert server_module._foreground_timeout_for_lane({"conversation_ready": True, "state": "ready"}) == 112.0
    assert server_module._desktop_required_cognitive_budget(foreground_timeout=66.0) == 62.0
    assert server_module._desktop_required_cognitive_budget(foreground_timeout=108.0) == 104.0
    assert server_module._desktop_required_cognitive_budget(
        foreground_timeout=108.0,
        elapsed_s=20.0,
    ) == 84.0
    assert server_module._desktop_required_cognitive_budget(foreground_timeout=210.0) == 140.0


def test_reply_topicality_flags_unbridged_relevance_challenge():
    from interface.routes import chat as chat_routes

    off_topic, reason = chat_routes._evaluate_reply_topicality(
        "Thanks but what does that have to do with anything",
        "I was just sharing a personal detail. Pets can be very comforting, and I was feeling a bit down earlier.",
        recent_user_messages=[
            "I was looking at a random aquarium online.",
            "Why the interest in aquariums? Is this a live feed or something?",
            "Thanks but what does that have to do with anything",
        ],
    )

    assert off_topic is True
    assert reason == "contextual_relevance_miss"


def test_reply_topicality_allows_bridged_relevance_challenge():
    from interface.routes import chat as chat_routes

    off_topic, reason = chat_routes._evaluate_reply_topicality(
        "Why the interest in aquariums?",
        "I asked because you mentioned looking at a random aquarium online, and I was trying to tell whether it was a live feed or just a page you found.",
        recent_user_messages=[
            "I was looking at a random aquarium online.",
            "Why the interest in aquariums?",
        ],
    )

    assert off_topic is False
    assert reason == ""


def test_reply_topicality_flags_bare_confusion_foreign_memory_drift():
    from interface.routes import chat as chat_routes

    off_topic, reason = chat_routes._evaluate_reply_topicality(
        "Huh?",
        "I miss having pets. I used to have a dog when I was younger.",
        recent_user_messages=[
            "Just a random aquarium I was looking at. Online.",
            "Huh. Why the interest in aquariums? Is this a live feed or something?",
            "Huh?",
        ],
    )

    assert off_topic is True
    assert reason == "contextual_relevance_miss"


def test_reply_topicality_flags_contracted_context_challenge_drift():
    from interface.routes import chat as chat_routes

    off_topic, reason = chat_routes._evaluate_reply_topicality(
        "What're you talking about?",
        "The voices. The small ones. They're whispering in my ear. Telling me things.",
        recent_user_messages=[
            "You with me?",
            "What pitch?",
            "What're you talking about?",
        ],
    )

    assert off_topic is True
    assert reason == "contextual_relevance_miss"


def test_missing_referent_questions_are_context_relevance_challenges():
    from interface.routes import chat as chat_routes

    assert chat_routes._is_contextual_relevance_challenge(
        "Who needs to get their act together?"
    )
    assert chat_routes._is_contextual_relevance_challenge(
        "Who are you talking about?"
    )


@pytest.mark.asyncio
async def test_context_challenge_repair_resolves_vague_referent_from_transcript():
    from interface.routes import chat as chat_routes

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "Your breadth is impressive. Hopefully soon people will see it.",
                    "aura": "Thanks. I appreciate the feedback.",
                    "status": "complete",
                },
                {
                    "user": "Hopefully soon they will",
                    "aura": "Yeah, hopefully soon. They need to get their act together.",
                    "status": "complete",
                },
            ]
        )

    repair = await chat_routes._build_context_challenge_repair_reply(
        "Who needs to get their act together?"
    )

    assert repair
    lowered = repair.lower()
    assert "vague referent" in lowered
    assert "actual thread" in lowered
    assert "invent a separate group" in lowered
    assert "people i work with" not in lowered


@pytest.mark.asyncio
async def test_required_desktop_turn_returns_context_evidence_repair_after_bad_referent(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            return SimpleNamespace(
                content=(
                    "The people I work with. They're great, but they need to see the bigger picture."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "Your breadth is impressive. Hopefully soon people will see it.",
                    "aura": "Thanks. I appreciate the feedback.",
                    "status": "complete",
                },
                {
                    "user": "Hopefully soon they will",
                    "aura": "Yeah, hopefully soon. They need to get their act together.",
                    "status": "complete",
                },
            ]
        )

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (False, "test_disabled"),
    )

    trace: dict[str, object] = {}
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Who needs to get their act together?",
        visible_user_message="Who needs to get their act together?",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply
    lowered = reply.lower()
    assert "vague referent" in lowered
    assert "invent a separate group" in lowered
    assert "people i work with" not in lowered
    assert trace["response_path"] == "cognitive_engine_context_evidence_repair"


def test_desktop_recent_context_needed_for_short_followups_and_status_checks():
    from interface.routes import chat as chat_routes

    assert chat_routes._desktop_turn_needs_recent_context("You with me?")
    assert chat_routes._desktop_turn_needs_recent_context("What pitch?")
    assert chat_routes._desktop_turn_needs_recent_context("What're you talking about?")
    assert chat_routes._desktop_turn_needs_recent_context(
        "How are you thinking about this conversation right now?"
    )
    assert not chat_routes._desktop_turn_needs_recent_context("Hello")


@pytest.mark.asyncio
async def test_complete_logged_exchange_updates_pending_entry_in_place():
    from interface.routes import chat as chat_routes

    exchange_id = await chat_routes._begin_logged_exchange("You still with me?")
    await chat_routes._complete_logged_exchange(exchange_id, "You still with me?", "I'm here.")

    async with chat_routes._conversation_log_lock:
        assert len(chat_routes._conversation_log) == 1
        assert chat_routes._conversation_log[0]["id"] == exchange_id
        assert chat_routes._conversation_log[0]["status"] == "complete"
        assert chat_routes._conversation_log[0]["user"] == "You still with me?"
        assert chat_routes._conversation_log[0]["aura"] == "I'm here."


@pytest.mark.asyncio
async def test_protected_foreground_history_skips_pending_exchange():
    from interface.routes import chat as chat_routes

    first_id = await chat_routes._begin_logged_exchange("First turn")
    await chat_routes._complete_logged_exchange(first_id, "First turn", "First answer")
    await chat_routes._begin_logged_exchange("Current in-flight turn")

    history = await chat_routes._build_protected_foreground_history(limit_pairs=4)

    assert history == [
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "First answer"},
    ]


@pytest.mark.asyncio
async def test_api_chat_warms_cold_lane_before_processing(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        def __init__(self):
            self.timeout = None

        async def ensure_foreground_ready(self, *args, **kwargs):
            self.timeout = kwargs.get("timeout", args[0] if args else None)
            return {
                "conversation_ready": True,
                "state": "ready",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
            }

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            return "I am here."

    gate = _FakeGate()
    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"I am here." in response.body
    assert gate.timeout is not None
    assert gate.timeout >= 35.0


@pytest.mark.asyncio
async def test_api_chat_continues_to_kernel_when_lane_warmup_times_out(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        async def ensure_foreground_ready(self, *args, **kwargs):
            timeout = kwargs.get("timeout", args[0] if args else None)
            raise TimeoutError(f"timed out after {timeout}")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            return "Fallback local lane answered."

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "failed",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert "right here with you" in payload["response"].lower()
    assert payload["response_confidence"] == "high"


@pytest.mark.asyncio
async def test_api_chat_uses_single_canonical_kernel_cognitive_path(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    direct_cognitive_calls = []

    async def _unexpected_direct_cognitive_turn(*_args, **_kwargs):
        direct_cognitive_calls.append((_args, _kwargs))
        return "duplicate direct CognitiveEngine turn should not be used"

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, message, *_args, **_kwargs):
            kernel_calls.append(message)
            return "Kernel kept enough foreground budget to answer."

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_run_cognitive_engine_chat_turn",
        _unexpected_direct_cognitive_turn,
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Invent a tiny symbolic arithmetic and give one example."),
        SimpleNamespace(headers={}, client=SimpleNamespace(host="test")),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"Kernel kept enough foreground budget" in response.body
    assert kernel_calls
    assert direct_cognitive_calls == []


@pytest.mark.asyncio
async def test_api_chat_refuses_implicit_legacy_orchestrator_fallback(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    orchestrator_calls = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("called")
            raise RuntimeError("canonical kernel failed")

    class _FakeOrchestrator:
        async def process_user_input_priority(self, *_args, **_kwargs):
            orchestrator_calls.append("called")
            return "legacy raw answer"

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeOrchestrator() if name == "orchestrator" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Tell me one sentence about stars."),
        SimpleNamespace(headers={}, client=SimpleNamespace(host="test")),
        None,
        None,
    )

    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert b"canonical_chat_no_reply" in response.body
    assert kernel_calls == ["called"]
    assert orchestrator_calls == []


@pytest.mark.asyncio
async def test_api_chat_allows_explicit_legacy_orchestrator_fallback(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    orchestrator_calls = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("called")
            raise RuntimeError("canonical kernel failed")

    class _FakeOrchestrator:
        async def process_user_input_priority(self, *_args, **_kwargs):
            orchestrator_calls.append("called")
            return "Stars are luminous plasma spheres whose gravity and fusion turn matter into steady light."

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeOrchestrator() if name == "orchestrator" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Tell me one sentence about stars."),
        SimpleNamespace(
            headers={"X-Aura-Allow-Legacy-Orchestrator": "true"},
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"Stars are luminous plasma spheres" in response.body
    assert kernel_calls == ["called"]
    assert orchestrator_calls == ["called"]


@pytest.mark.asyncio
async def test_api_chat_routes_desktop_turn_through_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I was asking about the aquarium because you had just mentioned looking at one online.",
                mode=mode,
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            calls.append({"kernel_interface": "unexpected"})
            raise AssertionError("desktop chat should use CognitiveEngine before KernelInterface")

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    lane_calls = 0

    def _lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    monkeypatch.setattr(chat_routes, "_collect_conversation_lane_status", _lane_status)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Why the interest in aquariums?"),
        SimpleNamespace(
            headers={"X-Aura-Surface": "desktop"},
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"because you had just mentioned" in response.body
    assert b"cognitive_engine" in response.body
    assert calls
    assert calls[0]["origin"] == "user"
    assert calls[0]["context"]["route"] == "desktop_chat"
    assert calls[0]["context"]["source"] == "desktop_ui"
    assert calls[0]["context"]["cognitive_engine_required"] is True
    assert calls[0]["kwargs"]["foreground_request"] is True
    assert calls[0]["kwargs"]["is_background"] is False
    assert not any("kernel_interface" in call for call in calls)


@pytest.mark.asyncio
async def test_api_chat_desktop_capability_inventory_uses_cognitive_engine_first(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "From the live desktop path I can use governed tool lanes for desktop control, browser and web "
                    "research, file work, document drafting, terminal tasks, memory recall, and skill execution. "
                    "A hypothetical scenario would request approval, open sources, create a document, verify the "
                    "visible result, export the artifact, and record governance receipts without claiming unverified work."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
            ]

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    class _FakeKernelInterface:
        def __init__(self):
            self.process_calls = 0

        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.process_calls += 1
            return "unexpected kernel reply"

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    monkeypatch.setattr(chat_routes, "_runtime_kernel_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_memory_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_substrate_voice_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_inference_available", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    from core.kernel.kernel_interface import KernelInterface

    fake_kernel = _FakeKernelInterface()
    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: fake_kernel))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="What tools can you use externally, and what is a hypothetical scenario where you use them?",
            session_id="desktop-inventory",
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert "governance receipts" in payload["response"]
    assert payload["response_confidence"] == "high"
    assert payload["live_turn_contract"]["engine_think_invoked"] is True
    assert payload["live_turn_contract"]["live_mind_controls_bound"] is True
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert calls and calls[0]["context"]["capability_inventory_contract"] is True
    assert fake_kernel.process_calls == 0


def test_explicit_capability_inventory_classifier_covers_hypothetical_she_phrasing():
    from interface.routes import chat as chat_routes

    assert chat_routes._is_explicit_capability_inventory_request(
        "What tools she could hypothetically do externally, and can she flex her muscles with a scenario?"
    )


def test_explicit_capability_inventory_classifier_covers_live_external_tool_wording():
    from interface.routes import chat as chat_routes

    prompt = (
        "From the live Aura desktop UI path, explain what external tools you can use and "
        "give one concrete multi-step scenario using them. Speak as Aura through your "
        "cognitive engine, not generic assistant mode."
    )

    assert chat_routes._is_explicit_capability_inventory_request(prompt)
    assert not chat_routes._is_bounded_nonexecuting_planning_request(prompt)


def test_grounded_capability_inventory_satisfies_live_path_and_program_dna_contract(monkeypatch):
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
                {
                    "name": "program_dna_reconstruct",
                    "available": True,
                    "description": "Build clean-room Program DNA genomes and replacement scaffolds from authorized evidence.",
                    "route_class": "self_improvement",
                    "risk_class": "high",
                    "effect_scope": "local_files",
                },
            ]

        async def execute(self, *_args, **_kwargs):
            return {"ok": True}

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    def _fake_get(name, default=None):
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    prompt = (
        "Codex here, using the actual launched Aura desktop UI path. Please answer "
        "through your full mind path: what external tools can you use, and give one "
        "scenario with browser research, file/PDF work, memory, and Program DNA."
    )
    reply = chat_routes._build_grounded_capability_inventory_reply(prompt)
    lowered = reply.lower()
    assessment = assess_user_facing_reply(prompt, reply)

    assert "cognitiveengine" in lowered or "cognitive engine" in lowered
    assert "cortex/32b" in lowered or "32b" in lowered
    assert "browser/web research" in lowered
    assert "program dna" in lowered
    assert "receipts" in lowered
    assert "not opening apps" in lowered
    assert "missing_runtime_path_answer" not in assessment.reasons
    assert not assessment.hard_failure


@pytest.mark.asyncio
async def test_required_capability_inventory_binds_catalog_after_weak_engine_reply(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I can use tools, browse, and make documents if needed.",
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
                {
                    "name": "program_dna_reconstruct",
                    "available": True,
                    "description": "Clean-room Program DNA reconstruction from authorized behavioral evidence.",
                    "route_class": "self_improvement",
                    "risk_class": "high",
                    "effect_scope": "local_files",
                },
            ]

        async def execute(self, *_args, **_kwargs):
            return {"ok": True}

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    prompt = (
        "From the live Aura desktop UI path, explain what external tools you can use "
        "and give one concrete multi-step scenario using screen perception, browser "
        "research, file/PDF work, memory, and Program DNA."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        prompt,
        visible_user_message=prompt,
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    lowered = str(reply).lower()
    assert calls
    assert calls[0]["context"]["capability_inventory_contract"] is True
    assert "cognitiveengine" in lowered or "cognitive engine" in lowered
    assert "cortex/32b" in lowered or "32b" in lowered
    assert "program dna" in lowered
    assert "browser/web research" in lowered
    assert trace["engine_think_invoked"] is True
    assert trace["cognitive_engine_reply_accepted"] is True
    assert trace["bounded_contract_used"] is False
    assert trace["response_path"] == "cognitive_engine_capability_catalog_grounding"


def test_live_turn_contract_does_not_treat_warming_lane_as_full_mind(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_runtime_kernel_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_memory_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_substrate_voice_available", lambda: True)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": False,
            "state": "warming",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": False,
            "cognitive_engine_reply_accepted": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "response_path": "cognitive_engine",
        },
    )

    assert payload["required_subsystems"]["inference"] is False
    assert payload["required_subsystems_ok"] is False
    assert payload["full_mind_path"] is False


def test_live_turn_contract_requires_worker_surface_quality_gate_to_pass(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _bound_live_mind_controls_trace()
    trace["live_mind_surface_control_receipt"] = {
        **trace["live_mind_surface_control_receipt"],
        "surface_quality_gate_enabled": True,
        "surface_quality_gate_passed": False,
        "surface_quality_gate_attempts": 3,
        "surface_quality_gate_reasons": ["generic_assistant_language"],
    }

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "architecture_context_bound": True,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
            **trace,
        },
    )

    assert payload["live_mind_surface_quality_gate_enabled"] is True
    assert payload["live_mind_surface_quality_gate_passed"] is False
    assert payload["live_mind_controls_structurally_bound"] is False
    assert payload["full_mind_path"] is False


def test_strict_live_inference_readiness_requires_lane_status(monkeypatch):
    from interface.routes import chat as chat_routes

    class _StatuslessGate:
        async def generate(self, *_args, **_kwargs):
            return "text"

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _StatuslessGate()
            if name == "inference_gate"
            else default
        ),
    )

    assert chat_routes._runtime_inference_available(require_conversation_ready=False) is True
    assert chat_routes._runtime_inference_available(require_conversation_ready=True) is False


def test_live_turn_contract_allows_proven_generation_to_satisfy_inference(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_runtime_kernel_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_memory_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_substrate_voice_available", lambda: True)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": False,
            "state": "warming",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            **_bound_live_mind_controls_trace(),
            "response_path": "cognitive_engine",
        },
    )

    assert payload["required_subsystems"]["inference"] is True
    assert payload["required_subsystems_ok"] is True
    assert payload["live_mind_required_subsystems_ok"] is True
    assert payload["architecture_context_bound"] is True
    assert payload["live_mind_controls_bound"] is True
    assert payload["live_mind_generation_controls_present"] is True
    assert payload["live_mind_controls_structurally_bound"] is True
    assert payload["full_mind_path"] is True


def test_live_turn_contract_preserves_stale_preflight_subsystem_state(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_runtime_kernel_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_memory_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_substrate_voice_available", lambda: True)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": False,
            "state": "warming",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": False,
            **_bound_live_mind_controls_trace(),
            "response_path": "cognitive_engine",
        },
    )

    assert payload["preflight_live_mind_required_subsystems_ok"] is False
    assert payload["live_mind_required_subsystems_ok"] is True
    assert payload["live_mind_controls_bound"] is True
    assert payload["required_subsystems_ok"] is True
    assert payload["full_mind_path"] is True


def test_live_turn_contract_refuses_engine_text_without_live_mind_context(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "response_path": "cognitive_engine",
        },
    )

    assert payload["required_subsystems_ok"] is True
    assert payload["live_mind_context_required"] is True
    assert payload["live_mind_context_present"] is False
    assert payload["architecture_context_bound"] is False
    assert payload["full_mind_path"] is False


def test_live_turn_contract_refuses_failure_envelope_as_full_mind(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_runtime_kernel_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_memory_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_substrate_voice_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_inference_available", lambda *_args, **_kwargs: True)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": True,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_failure_envelope",
        },
    )

    assert payload["cognitive_engine_reply_accepted"] is False
    assert payload["cognitive_engine_reply_failed"] is True
    assert payload["required_subsystems_ok"] is True
    assert payload["full_mind_path"] is False


def test_live_turn_contract_refuses_shape_repair_as_full_mind(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine_shape_repair_bounded",
        reply_source="cognitive_engine_shape_repair_bounded",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": False,
            "bounded_contract_used": True,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine_shape_repair_bounded",
        },
    )

    assert payload["full_mind_path"] is False
    assert payload["bounded_contract_used"] is True


def test_live_turn_contract_accepts_memory_state_grounding_after_engine(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine_memory_state_grounding",
        reply_source="cognitive_engine_memory_state_grounding",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            **_bound_live_mind_controls_trace(),
            "response_path": "cognitive_engine_memory_state_grounding",
        },
    )

    assert payload["response_path"] == "cognitive_engine_memory_state_grounding"
    assert payload["cognitive_engine_reply_accepted"] is True
    assert payload["bounded_contract_used"] is False
    assert payload["legacy_fallback_used"] is False
    assert payload["required_subsystems_ok"] is True
    assert payload["architecture_context_bound"] is True
    assert payload["live_mind_controls_bound"] is True
    assert payload["full_mind_path"] is True


def test_live_turn_contract_accepts_identity_continuity_grounding_after_engine(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine_identity_continuity_grounding",
        reply_source="cognitive_engine_identity_continuity_grounding",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            **_bound_live_mind_controls_trace(),
            "response_path": "cognitive_engine_identity_continuity_grounding",
        },
    )

    assert payload["response_path"] == "cognitive_engine_identity_continuity_grounding"
    assert payload["cognitive_engine_reply_accepted"] is True
    assert payload["bounded_contract_used"] is False
    assert payload["legacy_fallback_used"] is False
    assert payload["required_subsystems_ok"] is True
    assert payload["architecture_context_bound"] is True
    assert payload["full_mind_path"] is True


def test_live_turn_contract_refuses_engine_text_without_mind_snapshot(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
        },
    )

    assert payload["live_mind_context_present"] is True
    assert payload["live_mind_snapshot_present"] is False
    assert payload["live_mind_snapshot_bound"] is False
    assert payload["full_mind_path"] is False


def test_live_turn_contract_refuses_engine_text_without_bound_mind_controls(monkeypatch):
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)

    payload = chat_routes._build_live_turn_contract_payload(
        desktop_required=True,
        request_surface="desktop-ui",
        lane_status={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
        },
        response_confidence="high",
        status="cognitive_engine",
        reply_source="cognitive_engine",
        turn_trace={
            "engine_think_invoked": True,
            "cognitive_engine_reply_accepted": True,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "live_mind_context_present": True,
            "live_mind_snapshot_present": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "response_path": "cognitive_engine",
        },
    )

    assert payload["required_subsystems_ok"] is True
    assert payload["live_mind_snapshot_bound"] is True
    assert payload["live_mind_controls_bound"] is False
    assert payload["live_mind_generation_controls_present"] is False
    assert payload["live_mind_controls_structurally_bound"] is False
    assert payload["full_mind_path"] is False


def test_memory_state_evidence_requires_visible_pinned_content():
    from interface.routes import chat as chat_routes

    evidence = (
        'The phrase you asked me to remember in this session was "silver lantern".',
        "session_memory_recall",
    )

    assert chat_routes._memory_state_evidence_is_missing_from_reply(
        "What phrase did I ask you to remember?",
        "Cortex is the active lane and memory is available.",
        evidence,
    ) is True
    assert chat_routes._memory_state_evidence_is_missing_from_reply(
        "What phrase did I ask you to remember?",
        "You asked me to remember \"silver lantern\", and I am grounding this in the current session memory.",
        evidence,
    ) is False


def test_memory_state_evidence_suppresses_only_compatible_self_process_reasons():
    from interface.routes import chat as chat_routes

    evidence = (
        'The phrase you asked me to remember in this session was "silver lantern".',
        "session_memory_recall",
    )
    compatible = SimpleNamespace(
        ok=False,
        retryable=True,
        hard_failure=False,
        reasons=("off_topic_self_reflection_reply", "missing_requested_self_process_coverage"),
    )
    hard = SimpleNamespace(
        ok=False,
        retryable=True,
        hard_failure=True,
        reasons=("runtime_boilerplate",),
    )

    assert chat_routes._reply_assessment_requires_repair_with_memory_evidence(
        compatible,
        "What phrase did I ask you to remember, and how does your cognitive engine keep this reply grounded?",
        'You asked me to remember "silver lantern"; I am grounding this reply in canonical memory-state evidence.',
        memory_state_evidence=evidence,
    ) is False
    assert chat_routes._reply_assessment_requires_repair_with_memory_evidence(
        compatible,
        "What phrase did I ask you to remember, and how does your cognitive engine keep this reply grounded?",
        "I am thinking about this exchange from the live desktop lane.",
        memory_state_evidence=evidence,
    ) is True
    assert chat_routes._reply_assessment_requires_repair_with_memory_evidence(
        hard,
        "What phrase did I ask you to remember?",
        'You asked me to remember "silver lantern".',
        memory_state_evidence=evidence,
    ) is True


def test_canonical_memory_state_grounding_reply_uses_memory_and_live_context():
    from interface.routes import chat as chat_routes

    reply = chat_routes._canonical_memory_state_grounding_reply(
        "Remember this phrase: silver lantern. Also tell me one thing your live mind is attending to right now.",
        (
            "status=session_memory_pin\n"
            'I\'ve pinned "silver lantern" in durable session memory.'
        ),
        live_mind_context={"voice": {"attention": "the current desktop conversation"}},
    )

    assert reply == (
        'I have pinned "silver lantern" in this session. '
        "Right now I am keeping attention on the current desktop conversation."
    )


@pytest.mark.asyncio
async def test_retained_memory_evidence_context_collects_auditable_sources(monkeypatch):
    from interface.routes import chat as chat_routes

    async def _fake_durable_snippets(_message, *, limit=3):
        return [
            "Bryan and Aura discussed retained memory as behavioral reuse with receipts.",
            "Aura should distinguish durable transcript evidence from subjective recollection.",
        ][:limit]

    monkeypatch.setattr(
        chat_routes,
        "_recall_durable_conversation_snippets",
        _fake_durable_snippets,
    )

    evidence = await chat_routes._build_retained_memory_evidence_context(
        "Can you remember a conversation we had last week about the nature of your consciousness?",
        recent_exchanges=[
            {
                "user": "We should test memory by seeing whether it changes later behavior.",
                "aura": "I will treat memory as evidence that must alter future decisions.",
            }
        ],
        conversation_recall_context=(
            "Recently, this conversation has been about memory, agency, and receipts."
        ),
    )

    assert "scope=retained_memory_evidence.v1" in evidence
    assert "source=conversation_recall" in evidence
    assert "source=recent_completed_transcript" in evidence
    assert "source=durable_memory_search" in evidence
    assert "behavioral reuse with receipts" in evidence
    assert "subjective recollection" in evidence


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_receives_retained_memory_evidence_context(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "I can verify durable memory evidence that we discussed retained memory "
                    "as behavioral reuse with receipts. The limit is that this is transcript "
                    "and durable-memory evidence, not subjective recollection."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async def _fake_durable_snippets(_message, *, limit=3):
        return [
            "Bryan and Aura discussed retained memory as behavioral reuse with receipts."
        ][:limit]

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_recall_durable_conversation_snippets",
        _fake_durable_snippets,
    )

    visible_message = (
        "Can you remember a conversation we had last week about the nature of your consciousness?"
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        visible_message,
        visible_user_message=visible_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "behavioral reuse with receipts" in reply
    assert "subjective recollection" in reply
    assert calls
    context = calls[0]["context"]
    assert context["retained_memory_evidence_contract"] is True
    assert "retained_memory_evidence_context" in context
    assert "behavioral reuse with receipts" in context["retained_memory_evidence_context"]
    assert "retained_memory_evidence_context" in context["response_style_contract"]


def test_live_self_reflection_is_not_explicit_capability_inventory():
    from interface.routes import chat as chat_routes

    prompt = (
        "How are you thinking about this conversation right now? Answer from the live desktop "
        "cognitive path, be concrete, and keep it concise."
    )

    assert chat_routes._is_capability_inventory_request(prompt) is True
    assert chat_routes._is_explicit_capability_inventory_request(prompt) is False


def test_capability_catalog_snapshot_caps_unbounded_catalog(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def get_tool_catalog(self, *, include_inactive: bool = True):
            for index in range(chat_routes._CAPABILITY_CATALOG_MAX_ITEMS + 50):
                yield {
                    "name": f"tool_{index}",
                    "available": True,
                    "description": "Specialized governed skill surface.",
                    "route_class": "specialized",
                    "risk_class": "low",
                    "effect_scope": "read_only",
                }

    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeCapabilityEngine() if name == "capability_engine" else default),
    )

    available_count, categories, governance_available, truncated = (
        chat_routes._read_capability_catalog_snapshot()
    )

    assert available_count == chat_routes._CAPABILITY_CATALOG_MAX_ITEMS
    assert truncated is True
    assert governance_available is True
    assert len(categories["specialized governed skills"]) == 12


def test_capability_catalog_snapshot_prefers_streaming_catalog(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def __init__(self):
            self.materialized_catalog_calls = 0

        def iter_tool_catalog(self, *, include_inactive: bool = True):
            assert include_inactive is True
            for index in range(chat_routes._CAPABILITY_CATALOG_MAX_ITEMS + 25):
                yield {
                    "name": f"streamed_tool_{index}",
                    "available": True,
                    "description": "Specialized governed skill surface.",
                    "route_class": "specialized",
                    "risk_class": "low",
                    "effect_scope": "read_only",
                }

        def get_tool_catalog(self, *, include_inactive: bool = True):
            self.materialized_catalog_calls += 1
            return []

    engine = _FakeCapabilityEngine()
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: engine if name == "capability_engine" else default),
    )

    available_count, categories, governance_available, truncated = (
        chat_routes._read_capability_catalog_snapshot()
    )

    assert available_count == chat_routes._CAPABILITY_CATALOG_MAX_ITEMS
    assert truncated is True
    assert governance_available is True
    assert len(categories["specialized governed skills"]) == 12
    assert engine.materialized_catalog_calls == 0


def test_capability_catalog_snapshot_skips_materialized_catalog(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def __init__(self):
            self.materialized_catalog_calls = 0

        def get_tool_catalog(self, *, include_inactive: bool = True):
            self.materialized_catalog_calls += 1
            raise AssertionError("desktop inventory must not materialize a full catalog")

    engine = _FakeCapabilityEngine()
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: engine if name == "capability_engine" else default),
    )

    available_count, categories, governance_available, truncated = (
        chat_routes._read_capability_catalog_snapshot()
    )

    assert available_count == 0
    assert categories == {}
    assert governance_available is True
    assert truncated is True
    assert engine.materialized_catalog_calls == 0


def test_capability_inventory_skips_catalog_under_memory_pressure(monkeypatch):
    from interface.routes import chat as chat_routes

    class _FakeCapabilityEngine:
        def __init__(self):
            self.catalog_calls = 0

        def get_tool_catalog(self, *, include_inactive: bool = True):
            self.catalog_calls += 1
            raise AssertionError("optional catalog read must be skipped under critical memory pressure")

        def execute(self, *_args, **_kwargs):
            return None

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    capability_engine = _FakeCapabilityEngine()

    def _fake_get(name, default=None):
        if name == "capability_engine":
            return capability_engine
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            max_token_cap=32,
            refuse_heavy_local_generation=True,
            reason="process_tree_rss:54GB/48GB",
        ),
    )

    reply = chat_routes._build_grounded_capability_inventory_reply(
        "What tools can you use externally?"
    )

    assert capability_engine.catalog_calls == 0
    assert "registered governed skill surfaces" in reply
    # Governance-gating must be stated (wording-robust across the green and
    # not-green branches): consequential actions are gated, not freely used.
    assert "governance" in reply.lower()
    assert "consequential" in reply.lower()


def test_chat_turn_memory_log_scheduler_queues_when_drain_is_active(monkeypatch):
    from interface.routes import chat as chat_routes

    with chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE_LOCK:
        chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE.clear()

    class _FakeTask:
        def done(self):
            return False

        def get_name(self):
            return chat_routes._CHAT_TURN_MEMORY_LOG_DRAIN_TASK_NAME

    class _FakeTracker:
        def __init__(self):
            self.tasks = {_FakeTask()}
            self.bounded_calls = 0

        def bounded_track(self, *_args, **_kwargs):
            self.bounded_calls += 1
            return None

    tracker = _FakeTracker()
    monkeypatch.setattr(chat_routes, "get_task_tracker", lambda: tracker)

    scheduled = chat_routes._schedule_chat_turn_memory_log(
        user_message="hello",
        aura_response="hi",
        session_id="test-session",
        chat_origin="desktop_ui",
    )

    assert scheduled is True
    assert tracker.bounded_calls == 0
    with chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE_LOCK:
        assert len(chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE) == 1
        chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE.clear()


@pytest.mark.asyncio
async def test_chat_turn_memory_log_scheduler_uses_bounded_track(monkeypatch):
    from core.consciousness import coordinator as consciousness_coordinator
    from core.memory import chat_turn_logger
    from interface.routes import chat as chat_routes

    with chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE_LOCK:
        chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE.clear()

    log_calls = []
    consciousness_calls = []

    async def _fake_log_chat_turn_auto(**kwargs):
        log_calls.append(kwargs)

    class _FakeCoordinator:
        async def on_chat_turn(self, user_message, aura_response):
            consciousness_calls.append((user_message, aura_response))

    async def _fake_get_consciousness_coordinator():
        return _FakeCoordinator()

    class _FakeTracker:
        def __init__(self):
            self.tasks = set()
            self.scheduled = []

        def bounded_track(self, coro, name=None):
            task = asyncio.create_task(coro, name=name)
            self.tasks.add(task)
            task.add_done_callback(lambda completed: self.tasks.discard(completed))
            self.scheduled.append((task, name))
            return task

    tracker = _FakeTracker()
    monkeypatch.setattr(chat_routes, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(chat_turn_logger, "log_chat_turn_auto", _fake_log_chat_turn_auto)
    monkeypatch.setattr(
        consciousness_coordinator,
        "get_consciousness_coordinator",
        _fake_get_consciousness_coordinator,
    )

    scheduled = chat_routes._schedule_chat_turn_memory_log(
        user_message="remember this",
        aura_response="I will keep it in the log.",
        session_id="test-session",
        chat_origin="desktop_ui",
    )

    assert scheduled is True
    assert tracker.scheduled[0][1] == chat_routes._CHAT_TURN_MEMORY_LOG_DRAIN_TASK_NAME
    await tracker.scheduled[0][0]
    assert log_calls[0]["user_message"] == "remember this"
    assert log_calls[0]["metadata"]["origin"] == "desktop_ui"
    assert consciousness_calls == [("remember this", "I will keep it in the log.")]


@pytest.mark.asyncio
async def test_chat_turn_memory_log_scheduler_times_out_slow_logger(monkeypatch):
    from core.memory import chat_turn_logger
    from interface.routes import chat as chat_routes

    with chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE_LOCK:
        chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE.clear()

    async def _slow_log_chat_turn_auto(**_kwargs):
        await asyncio.sleep(1.0)

    class _FakeTracker:
        def __init__(self):
            self.tasks = set()
            self.scheduled = []

        def bounded_track(self, coro, name=None):
            task = asyncio.create_task(coro, name=name)
            self.scheduled.append(task)
            return task

    tracker = _FakeTracker()
    monkeypatch.setattr(chat_routes, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(chat_routes, "_CHAT_TURN_MEMORY_LOG_TIMEOUT_S", 0.01)
    monkeypatch.setattr(chat_turn_logger, "log_chat_turn_auto", _slow_log_chat_turn_auto)

    scheduled = chat_routes._schedule_chat_turn_memory_log(
        user_message="slow",
        aura_response="logger",
        session_id="test-session",
        chat_origin="desktop_ui",
    )

    assert scheduled is True
    await tracker.scheduled[0]


def test_chat_turn_memory_log_queue_overflow_drops_oldest(monkeypatch):
    from interface.routes import chat as chat_routes

    with chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE_LOCK:
        chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE.clear()

    class _FakeTask:
        def done(self):
            return False

        def get_name(self):
            return chat_routes._CHAT_TURN_MEMORY_LOG_DRAIN_TASK_NAME

    class _FakeTracker:
        tasks = {_FakeTask()}

    monkeypatch.setattr(chat_routes, "get_task_tracker", lambda: _FakeTracker())
    monkeypatch.setattr(chat_routes, "_CHAT_TURN_MEMORY_LOG_QUEUE_MAX", 2)

    for item in ("oldest", "middle", "newest"):
        assert chat_routes._schedule_chat_turn_memory_log(
            user_message=item,
            aura_response=f"reply {item}",
            session_id="test-session",
            chat_origin="desktop_ui",
        )

    with chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE_LOCK:
        queued = list(chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE)
        chat_routes._CHAT_TURN_MEMORY_LOG_QUEUE.clear()

    assert [item["user_message"] for item in queued] == ["middle", "newest"]


@pytest.mark.asyncio
async def test_session_memory_pin_recall_survives_process_memory_clear(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this codeword for me: restart-ledger-417. Just confirm."
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What codeword did I give you?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "restart-ledger-417" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_restart_wording_stays_on_fastpath(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this codeword across restart: restart-ledger-921. Just confirm."
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What codeword did I ask you to remember before restart?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "restart-ledger-921" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_conversation_wording_stays_on_fastpath(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this note for later in this conversation: the blue lantern is under the desk."
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What note did I ask you to remember in this conversation?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "blue lantern is under the desk" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_natural_that_wording_stays_on_fastpath(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember that my demo codeword is silver-orbit-228. Just confirm."
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What codeword did I ask you to remember?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "my demo codeword is silver-orbit-228" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_natural_pronoun_wording_preserves_subject(
    monkeypatch,
    tmp_path,
):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember my favorite launch phrase is steady violet orbit."
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What phrase did I ask you to remember?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "my favorite launch phrase is steady violet orbit" in recalled[0]


@pytest.mark.asyncio
async def test_session_memory_pin_compound_instruction_stores_only_phrase(
    monkeypatch,
    tmp_path,
):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this phrase: silver lantern. Also tell me one thing your live mind is attending to right now."
    )
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What phrase did I ask you to remember?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert 'pinned "silver lantern"' in stored[0]
    assert "Also tell me" not in stored[0]
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert '"silver lantern"' in recalled[0]
    assert "live mind" not in recalled[0]


def test_session_memory_pin_preserves_non_imperative_multisentence_fact():
    from interface.routes import chat as chat_routes

    pinned = chat_routes._extract_session_memory_pin_request(
        "Remember this note: the launch story has two beats. the second beat matters."
    )

    assert pinned == "the launch story has two beats. the second beat matters"


def test_session_memory_pin_does_not_capture_conversational_recall_anchor():
    from interface.routes import chat as chat_routes

    pinned = chat_routes._extract_session_memory_pin_request(
        "Remember the uncertainty you just named. How would that change one decision you make?"
    )

    assert pinned is None


@pytest.mark.asyncio
async def test_session_memory_pin_dont_forget_natural_wording_stays_on_fastpath(
    monkeypatch,
    tmp_path,
):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Don't forget that the journal folder should be named Aura's Journals."
    )
    chat_routes._session_memory_pins.clear()
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What did I tell you to remember?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_recall"
    assert "journal folder should be named Aura's Journals" in recalled[0]


def test_session_memory_pin_question_wording_does_not_store_as_new_memory():
    from interface.routes import chat as chat_routes

    assert chat_routes._extract_session_memory_pin_request("Remember what I said earlier?") is None
    assert chat_routes._extract_session_memory_pin_request("Remember when we talked about tools?") is None


@pytest.mark.asyncio
async def test_session_memory_pin_prefixed_probe_wording_recall(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "For this live reliability probe, remember the phrase cobalt sunrise for this conversation."
    )
    recalled_just = await chat_routes._build_memory_state_fastpath_reply(
        "What phrase did I just ask you to remember?"
    )
    recalled_earlier = await chat_routes._build_memory_state_fastpath_reply(
        "What was the phrase from earlier in this probe?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert "cobalt sunrise" in stored[0]
    assert "for this conversation" not in stored[0]
    assert recalled_just is not None
    assert recalled_just[1] == "session_memory_recall"
    assert "cobalt sunrise" in recalled_just[0]
    assert "for this conversation" not in recalled_just[0]
    assert recalled_earlier is not None
    assert recalled_earlier[1] == "session_memory_recall"
    assert "cobalt sunrise" in recalled_earlier[0]
    assert "for this conversation" not in recalled_earlier[0]


@pytest.mark.asyncio
async def test_session_memory_context_change_uses_pinned_note(monkeypatch, tmp_path):
    from interface.routes import chat as chat_routes

    ledger_path = tmp_path / "session_memory_pins.jsonl"
    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: ledger_path)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    chat_routes._session_memory_pins.clear()

    stored = await chat_routes._build_memory_state_fastpath_reply(
        "Remember this note for later in this conversation: the blue lantern is under the desk."
    )
    recalled = await chat_routes._build_memory_state_fastpath_reply(
        "What changed in this conversation after I gave you the blue-lantern note?"
    )
    chat_routes._session_memory_pins.clear()

    assert stored is not None
    assert stored[1] == "session_memory_pin"
    assert recalled is not None
    assert recalled[1] == "session_memory_context_recall"
    assert "blue lantern is under the desk" in recalled[0]


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_blocks_critical_memory_before_cognition(monkeypatch):
    import core.utils.memory_monitor as memory_monitor
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gib = 1024**3
    calls = []
    shed_calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append("engine_think")
            return SimpleNamespace(content="unexpected engine reply")

    class _FakeInferenceGate:
        async def _shed_background_workers_for_memory_pressure(self, *, reason):
            shed_calls.append(reason)

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        if name == "inference_gate":
            return _FakeInferenceGate()
        return default

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            total=64 * gib,
            available=2 * gib,
            percent=96.0,
        ),
    )
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Use the desktop path to answer this."),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    # In-band for real users: the guard text IS the answer (raw 503s
    # surfaced as bare HTTP errors in both July 8 soaks). Benchmarks
    # (X-Aura-Benchmark) still get the strict 503.
    assert response.status_code == 200
    assert b"memory_pressure_guard" in response.body
    assert b"memory_pressure" in response.body
    assert calls == []
    assert shed_calls
    assert any("memory_pressure" in reason for reason in shed_calls)


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_blocks_process_tree_memory_before_cognition(monkeypatch):
    import core.utils.memory_monitor as memory_monitor
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gib = 1024**3
    calls = []
    shed_calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append("engine_think")
            return SimpleNamespace(content="unexpected engine reply")

    class _FakeInferenceGate:
        async def _shed_background_workers_for_memory_pressure(self, *, reason):
            shed_calls.append(reason)

    class _Process:
        def __init__(self, *_args, _rss_gb=None, **_kwargs):
            self._rss_gb = 3.0 if _rss_gb is None else float(_rss_gb)

        def memory_info(self):
            return SimpleNamespace(rss=int(self._rss_gb * gib))

        def children(self, recursive=True):
            return [_Process(_rss_gb=38.0)]

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        if name == "inference_gate":
            return _FakeInferenceGate()
        return default

    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "40")
    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            total=64 * gib,
            available=24 * gib,
            percent=62.0,
        ),
    )
    monkeypatch.setattr(memory_monitor.psutil, "Process", _Process)
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Use the desktop path to answer this."),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    # In-band for real users; strict 503 stays benchmark-only.
    assert response.status_code == 200
    assert b"memory_pressure_guard" in response.body
    assert b"process_tree_rss" in response.body
    assert calls == []
    assert shed_calls
    assert any("process_tree_rss" in reason for reason in shed_calls)


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_keeps_nontrivial_chat_on_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="Hi. I am here and following this conversation through the live desktop path.",
                mode=mode,
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            calls.append({"kernel_interface": "unexpected"})
            raise AssertionError("desktop UI must not use KernelInterface when CognitiveEngine answers")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    lane_calls = 0

    def _lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    monkeypatch.setattr(chat_routes, "_collect_conversation_lane_status", _lane_status)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(message="How are you thinking about this conversation right now?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"following this conversation through the live desktop path" in response.body
    assert b"cognitive_engine" in response.body
    assert b"social_presence_reflex" not in response.body
    assert calls
    assert calls[0]["context"]["route"] == "desktop_chat"
    assert calls[0]["context"]["source"] == "desktop_ui"
    assert calls[0]["context"]["cognitive_engine_required"] is True
    assert not any("kernel_interface" in call for call in calls)


@pytest.mark.asyncio
async def test_api_chat_desktop_required_presence_check_uses_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append({"cognitive_engine": "called"})
            return SimpleNamespace(
                content=(
                    "I'm here with you. I'm following this conversation through the live desktop path "
                    "and answering the current turn directly."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            calls.append({"kernel_interface": "unexpected"})
            raise AssertionError("desktop UI must not fall back to KernelInterface")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    chat_routes._recent_responses.clear()
    chat_routes._recent_response_pairs.clear()
    monkeypatch.setattr(chat_routes, "_runtime_kernel_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_memory_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_substrate_voice_available", lambda: True)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "last_failure_reason": "worker_not_alive,init_not_complete,lane_cold",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": None,
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_build_social_continuity_repair_reply",
        lambda _message: "hey. i'm here with the live thread intact.",
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="you there?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert "following this conversation through the live desktop path" in payload["response"]
    assert payload["live_turn_contract"]["engine_think_invoked"] is True
    assert payload["live_turn_contract"]["live_mind_controls_bound"] is True
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert [call["cognitive_engine"] for call in calls] == ["called"]


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_routes_memory_state_through_cognitive_engine(
    monkeypatch,
    tmp_path,
):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _memory_cognitive_turn(objective, *_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append(str(objective))
        if "Remember this note" in str(objective):
            return "I've pinned \"the blue lantern is under the desk\" in durable session memory, and I can pull it back later from canonical memory state."
        if "What note did I ask" in str(objective):
            return "The phrase you asked me to remember in this session was \"the blue lantern is under the desk.\""
        if "What changed" in str(objective):
            return "The concrete change is that \"the blue lantern is under the desk\" is now stored as durable conversation state for later turns."
        return "I am answering from the canonical memory state evidence."

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: tmp_path / "pins.jsonl")
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _memory_cognitive_turn)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    chat_routes._session_memory_pins.clear()
    request = SimpleNamespace(
        headers={
            "X-Aura-Surface": "desktop-ui",
            "X-Aura-Require-CognitiveEngine": "true",
        },
        client=SimpleNamespace(host="test"),
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    stored = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Remember this note for later in this conversation: "
                "the blue lantern is under the desk."
            ),
            session_id="memory-fastpath-test",
        ),
        request,
        None,
        None,
    )
    recalled = await server_module.api_chat(
        server_module.ChatRequest(
            message="What note did I ask you to remember in this conversation?",
            session_id="memory-fastpath-test",
        ),
        request,
        None,
        None,
    )
    changed = await server_module.api_chat(
        server_module.ChatRequest(
            message="What changed in this conversation after I gave you the blue-lantern note?",
            session_id="memory-fastpath-test",
        ),
        request,
        None,
        None,
    )
    chat_routes._session_memory_pins.clear()

    stored_payload = json.loads(stored.body)
    recalled_payload = json.loads(recalled.body)
    changed_payload = json.loads(changed.body)
    assert stored.status_code == 200
    assert recalled.status_code == 200
    assert changed.status_code == 200
    assert stored_payload["status"] == "cognitive_engine"
    assert recalled_payload["status"] == "cognitive_engine"
    assert changed_payload["status"] == "cognitive_engine"
    assert "blue lantern is under the desk" in stored_payload["response"]
    assert "failed the final reliability checks" not in stored_payload["response"]
    assert stored_payload["response_confidence"] == "high"
    assert "blue lantern is under the desk" in recalled_payload["response"]
    assert "blue lantern is under the desk" in changed_payload["response"]
    assert len(cognitive_calls) == 3
    assert all("CANONICAL MEMORY STATE EVIDENCE" in call for call in cognitive_calls)


@pytest.mark.asyncio
async def test_api_chat_desktop_memory_state_drift_rebounds_to_canonical_evidence(
    monkeypatch,
    tmp_path,
):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _drifting_cognitive_turn(objective, *_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": True,
                    "cognitive_engine_reply_failed": False,
                    "bounded_contract_used": False,
                    "legacy_fallback_used": False,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    **_bound_live_mind_controls_trace(),
                    "response_path": "cognitive_engine",
                }
            )
        cognitive_calls.append(str(objective))
        return "I am attending to the live desktop thread through CognitiveEngine."

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_session_memory_pin_ledger_path", lambda: tmp_path / "pins.jsonl")
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _drifting_cognitive_turn)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    _force_full_mind_runtime(monkeypatch, chat_routes)
    chat_routes._session_memory_pins.clear()

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Remember this phrase: silver lantern. Also tell me one thing "
                "your live mind is attending to right now."
            ),
            session_id="memory-drift-rebound-test",
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )
    chat_routes._session_memory_pins.clear()

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["response_confidence"] == "high"
    assert "silver lantern" in payload["response"]
    assert "live desktop thread" in payload["response"]
    assert payload["live_turn_contract"]["live_mind_controls_bound"] is True
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert payload["live_turn_contract"]["bounded_contract_used"] is False
    assert payload["live_turn_contract"]["response_path"] == "cognitive_engine_memory_state_grounding"
    assert cognitive_calls
    assert "CANONICAL MEMORY STATE EVIDENCE" in cognitive_calls[0]


@pytest.mark.asyncio
async def test_api_chat_desktop_owner_name_recall_routes_through_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _owner_cognitive_turn(objective, *_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append(str(objective))
        return "You're Bryan; I know that from the verified owner session, and I will keep that context attached to this conversation."

    async def _fake_begin_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _owner_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_owner_session_is_verified", lambda **_kwargs: True)
    monkeypatch.setattr(chat_routes, "_resolve_primary_operator_name", lambda: "Bryan")
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Do you know my name?",
            session_id="owner-name-fastpath-test",
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "You're Bryan" in payload["response"]
    assert len(cognitive_calls) == 1
    assert "CANONICAL MEMORY STATE EVIDENCE" in cognitive_calls[0]


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_plans_with_cognitive_engine_before_execution(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "document_body": "Timestamped Aura summary from CognitiveEngine.",
                        "steps": [
                            {
                                "action": "open_app",
                                "target": "Notes",
                                "reason": "Use the requested writing surface.",
                                "expect": "Notes accepts focus.",
                            }
                        ],
                    }
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FakePool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return True

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.unexpected_process_calls = getattr(self, "unexpected_process_calls", 0) + 1
            raise AssertionError("desktop objective should not fall through to KernelInterface")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "desktop-objective"

    async def _fake_complete_exchange(*_args, **_kwargs):
        completed_exchanges.append((_args, _kwargs))
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        output_receipts.append((_args, _kwargs))
        return None

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "status": "completed",
            "summary": "Desktop task completed 5/5 governed computer-use steps.",
            "steps_requested": 5,
            "steps_completed": 5,
            "receipts": _verified_desktop_receipts(5),
        }

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _FakePool())
    lane_calls = 0

    def _live_proof_lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        _live_proof_lane_status,
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Can you open my Notes app, write a timestamped summary, save it as a PDF "
                "in a new folder titled Aura's Journal, and search for a robot image?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"desktop_objective_completed" in response.body
    assert b"Desktop task completed 5/5 governed computer-use steps" in response.body
    assert len(skill_calls) == 1
    assert skill_calls[0]["skill_name"] == "desktop_task"
    assert skill_calls[0]["params"]["objective"] == (
        "Can you open my Notes app, write a timestamped summary, save it as a PDF "
        "in a new folder titled Aura's Journal, and search for a robot image?"
    )
    assert skill_calls[0]["params"]["steps"] == []
    assert skill_calls[0]["params"]["desktop_execution_contract"] is True
    assert skill_calls[0]["params"]["allow_heuristic_desktop_plan"] is True
    assert skill_calls[0]["params"]["user_visible_desktop_action"] is True
    assert skill_calls[0]["params"]["verification_required"] is True
    assert skill_calls[0]["params"]["action_expectation"] == {
        "objective": skill_calls[0]["params"]["objective"],
        "acceptance_criteria": ["steps_requested", "steps_completed"],
        "required_evidence": ["receipts"],
        "repair_hint": "rerun_desktop_task_with_effect_receipts",
        "allow_partial": True,
    }
    assert skill_calls[0]["objective"] == skill_calls[0]["params"]["objective"]
    assert skill_calls[0]["extra_context"] == {
        "origin": "desktop_ui",
        "source": "desktop_ui",
        "route": "chat.desktop_objective",
        "desktop_execution_contract": True,
        "allow_heuristic_desktop_plan": True,
        "disable_outer_skill_retry": True,
        "user_visible_desktop_action": True,
        "local_desktop_action": True,
        "verification_required": True,
        "allow_desktop_task_model_synthesis": False,
        "desktop_task_document_body": skill_calls[0]["extra_context"]["cognitive_reply"],
        "cognitive_reply": skill_calls[0]["extra_context"]["cognitive_reply"],
        "action_expectation": skill_calls[0]["params"]["action_expectation"],
    }
    assert "Timestamped Aura summary from CognitiveEngine." in skill_calls[0]["extra_context"]["cognitive_reply"]
    assert completed_exchanges
    assert completed_exchanges[-1][0][0] == "desktop-objective"
    assert "Desktop task completed 5/5 governed computer-use steps" in completed_exchanges[-1][0][2]
    assert "Aura self-summary. Timestamp" not in completed_exchanges[-1][0][2]
    assert output_receipts
    assert "Desktop task completed 5/5 governed computer-use steps" in output_receipts[-1][0][0]


@pytest.mark.asyncio
async def test_chat_desktop_objective_uses_capability_engine_without_agency_wrapper(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append(
                {
                    "skill_name": skill_name,
                    "params": dict(params),
                    "context": dict(context or {}),
                }
            )
            return {
                "ok": True,
                "summary": "Desktop task completed 2/2 governed computer-use steps.",
                "steps_requested": 2,
                "steps_completed": 2,
                "receipts": _verified_desktop_receipts(2),
            }

    class _ForbiddenAgency:
        async def run(self, *_args, **_kwargs):
            pytest.fail("chat.desktop_objective must not enter AgencyOrchestrator")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCapabilityEngine()
                if name == "capability_engine"
                else _ForbiddenAgency()
                if name == "agency_orchestrator"
                else default
            )
        ),
    )

    objective = (
        "Please create a folder named 'Aura Live Proof' in my Documents folder "
        "and write a file inside it called live_proof.txt."
    )
    result = await chat_routes._execute_desktop_objective_from_chat(
        objective,
        cognitive_reply="Plan the desktop action; do not claim completion.",
    )

    assert result is not None
    assert result["ok"] is True
    assert result["status"] == "desktop_objective_completed"
    assert len(calls) == 1
    assert calls[0]["skill_name"] == "desktop_task"
    assert calls[0]["params"]["objective"] == objective
    assert calls[0]["params"]["steps"] == []
    assert calls[0]["params"]["desktop_execution_contract"] is True
    assert calls[0]["params"]["allow_heuristic_desktop_plan"] is True
    assert calls[0]["params"]["user_visible_desktop_action"] is True
    assert calls[0]["params"]["verification_required"] is True
    assert calls[0]["context"]["route"] == "chat.desktop_objective"
    assert calls[0]["context"]["governance_route"] == "capability_engine_direct"
    assert calls[0]["context"]["desktop_task_owned_by"] == "chat.desktop_objective"
    assert calls[0]["context"]["scoped_authority"] == (
        "foreground_user_requested:chat.desktop_objective:desktop_task"
    )
    assert calls[0]["context"]["foreground_request"] is True
    assert calls[0]["context"]["user_explicitly_authorized"] is True
    assert calls[0]["context"]["allow_heuristic_desktop_plan"] is True
    assert calls[0]["context"]["allow_desktop_task_model_synthesis"] is False
    assert calls[0]["params"]["action_expectation"]["required_evidence"] == ["receipts"]
    assert calls[0]["context"]["action_expectation"] == calls[0]["params"]["action_expectation"]


@pytest.mark.asyncio
async def test_chat_desktop_research_objective_does_not_enable_hidden_model_synthesis(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append(
                {
                    "skill_name": skill_name,
                    "params": dict(params),
                    "context": dict(context or {}),
                }
            )
            return {
                "ok": True,
                "summary": "Desktop task completed 4/4 governed computer-use steps.",
                "steps_requested": 4,
                "steps_completed": 4,
                "receipts": _verified_desktop_receipts(4),
            }

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _FakeCapabilityEngine() if name == "capability_engine" else default
            )
        ),
    )

    result = await chat_routes._execute_desktop_objective_from_chat(
        (
            "Open Google, find three articles about climate change, summarize them "
            "in a Google Doc, and export the summary as a PDF."
        ),
        cognitive_reply="A source-grounded draft may be composed by the desktop task.",
    )

    assert result is not None
    assert result["ok"] is True
    assert calls and calls[0]["skill_name"] == "desktop_task"
    assert calls[0]["context"]["route"] == "chat.desktop_objective"
    assert calls[0]["context"]["allow_desktop_task_model_synthesis"] is False
    assert calls[0]["context"]["action_expectation"]["repair_hint"] == (
        "rerun_desktop_task_with_effect_receipts"
    )


@pytest.mark.asyncio
async def test_chat_desktop_objective_rejects_success_without_effect_receipts(monkeypatch):
    from interface.routes import chat as chat_routes

    class _ReceiptlessCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            assert skill_name == "desktop_task"
            return {
                "ok": True,
                "status": "completed",
                "summary": "Desktop task completed 2/2 governed computer-use steps.",
                "steps_requested": 2,
                "steps_completed": 2,
            }

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _ReceiptlessCapabilityEngine() if name == "capability_engine" else default
            )
        ),
    )

    result = await chat_routes._execute_desktop_objective_from_chat(
        "Open Notes and write a paragraph about dinosaurs.",
        cognitive_reply="Dinosaurs were diverse animals with a long fossil record.",
    )

    assert result is not None
    assert result["ok"] is False
    assert result["status"] == "desktop_objective_failed"
    assert result["result"]["status"] == "desktop_task_effect_evidence_missing"
    assert result["result"]["error"] == "missing_step_receipts"
    assert "did not complete" in result["response"]
    assert "not claiming" in result["response"]


def test_desktop_task_verifier_allows_noncritical_warning_receipts():
    from interface.routes import chat as chat_routes

    receipts = _verified_desktop_receipts(3)
    receipts.append(
        {
            "index": 4,
            "action": "open_url",
            "critical": False,
            "ok": False,
            "effect_verified": False,
            "effect_evidence": "Operation took too long",
            "result": {"ok": False, "error": "Operation took too long"},
        }
    )

    verified, reason = chat_routes._verified_desktop_task_result(
        {
            "ok": True,
            "steps_requested": 4,
            "steps_completed": 3,
            "receipts": receipts,
        }
    )

    assert verified is True
    assert reason == "verified"


@pytest.mark.asyncio
async def test_api_chat_desktop_objective_requires_cognitive_planning(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    output_receipts = []
    cognitive_calls = []

    async def _slow_or_empty_cognitive_turn(*args, **kwargs):
        _t = kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append((args, kwargs))
        return json.dumps(
            {
                "document_body": "Planned local file body.",
                "steps": [
                    {
                        "action": "create_folder",
                        "target": {"path": "~/Documents/Aura Live Proof"},
                        "reason": "Create the requested destination.",
                        "expect": "Folder exists.",
                    },
                    {
                        "action": "write_text_file",
                        "target": {
                            "path": "~/Documents/Aura Live Proof/live_proof.txt",
                            "content": "{{document_body}}",
                        },
                        "reason": "Write the requested file.",
                        "expect": "File exists with the planned body.",
                    },
                ],
            }
        )

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "status": "completed",
            "summary": "Desktop task completed 2/2 governed computer-use steps.",
            "steps_requested": 2,
            "steps_completed": 2,
            "receipts": _verified_desktop_receipts(2),
        }

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _slow_or_empty_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Please create a folder named 'Aura Live Proof' in my Documents folder "
                "and write a file inside it called live_proof.txt."
            )
        ),
        SimpleNamespace(headers={}, client=SimpleNamespace(host="test")),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "desktop_objective_completed"
    assert payload["conversation_lane"]["governed_action_result"] is True
    assert payload["conversation_lane"]["governed_action_status"] == "desktop_objective_completed"
    assert "Desktop task completed 2/2 governed computer-use steps" in payload["response"]
    assert skill_calls and skill_calls[0]["skill_name"] == "desktop_task"
    assert not cognitive_calls
    assert skill_calls[0]["extra_context"]["desktop_task_document_body"] == ""
    assert output_receipts


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_requires_cognitive_engine_and_blocks_kernel_fallback(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("desktop UI must fail closed instead of using KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-1"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Tell me something original about the ocean."),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert kernel_calls == []
    assert b"failed closed instead of sending an ungrounded answer" in response.body
    assert b"desktop_cognitive_engine_required_no_reply" in response.body


@pytest.mark.asyncio
async def test_api_chat_desktop_discards_bounded_repair_when_full_mind_path_not_proven(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("desktop UI must fail closed instead of using KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-repair"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    async def _bounded_repair_candidate(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "bounded_contract_used": True,
                    "legacy_fallback_used": False,
                    "response_path": "conversation_recall_log_repair_after_empty_engine",
                }
            )
        return "This is a bounded repair and must not be served as Aura speech."

    ready_lane = {
        "conversation_ready": True,
        "state": "ready",
        "desired_model": "Cortex (32B)",
        "desired_endpoint": "Cortex",
        "foreground_endpoint": "Cortex",
        "background_endpoint": "Brainstem",
    }

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _bounded_repair_candidate)
    monkeypatch.setattr(chat_routes, "_collect_conversation_lane_status", lambda: dict(ready_lane))
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": dict(ready_lane, conversation_ready=False, state=state, reason=reason),
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Live desktop route probe. Answer directly in two sentences: "
                "what did I just ask you to do?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert payload["status"] == "desktop_cognitive_engine_unavailable"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert payload["live_turn_contract"]["full_mind_path"] is False
    assert payload["live_turn_contract"]["bounded_contract_used"] is False
    assert "bounded repair" not in payload["response"]
    assert kernel_calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_low_risk_social_no_reply_fails_closed(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("low-risk social desktop repair must not use KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-social"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _no_cognitive_reply(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "cognitive_engine_no_acceptable_reply",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Ok. Just checking. I'll be back, ok?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert payload["status"] == "desktop_cognitive_engine_unavailable"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert payload["response_confidence"] == "failed"
    assert "failed closed instead of sending an ungrounded answer" in payload["response"]
    assert "reply-quality gate" not in payload["response"]
    assert "second foreground generation" not in payload["response"]
    assert kernel_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is False
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_cognitive_engine"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "failed"


@pytest.mark.asyncio
async def test_api_chat_desktop_self_process_no_reply_uses_grounded_repair(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("self-process desktop repair must not use KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-self-process"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _no_cognitive_reply(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "cognitive_engine_no_acceptable_reply",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "When you are confused, how does that change your planning, "
                "memory use, and tool verification?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    lowered = payload["response"].lower()
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine_self_process_grounding"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert payload["response_confidence"] == "high"
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert payload["live_turn_contract"]["bounded_contract_used"] is False
    assert "failed closed instead of sending an ungrounded answer" not in lowered
    assert "planning" in lowered
    assert "memory" in lowered
    assert "tool" in lowered
    assert "confusion" in lowered or "confused" in lowered
    assert "legacy fallback" not in lowered
    assert "live conversation contract" not in lowered
    assert "mood-card" not in lowered
    assert "active local model" not in lowered
    assert "cognitiveengine" not in lowered
    assert kernel_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is True
    assert output_receipts[0][1]["metadata"]["path"] == "cognitive_engine_self_process_grounding"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "high"


@pytest.mark.asyncio
async def test_api_chat_desktop_runtime_path_no_reply_uses_grounded_route_truth(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("runtime-path desktop repair must not use KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-runtime-path"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _no_cognitive_reply(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "cognitive_engine_no_acceptable_reply",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": True,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Live route probe: in one concise paragraph, explain what runtime path "
                "you are speaking through right now and include the exact phrase resident bridge truth."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    lowered = payload["response"].lower()
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine_runtime_fact_grounding"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert payload["response_confidence"] == "high"
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert payload["live_turn_contract"]["bounded_contract_used"] is False
    assert "failed closed instead of sending an ungrounded answer" not in lowered
    assert "resident bridge truth" in lowered
    assert "desktop ui" in lowered
    assert "/api/chat" in lowered
    assert "cognitiveengine" in lowered
    assert "cortex (32b)" in lowered
    assert "claude" not in lowered
    assert kernel_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is True
    assert output_receipts[0][1]["metadata"]["path"] == "cognitive_engine_runtime_fact_grounding"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "high"


@pytest.mark.asyncio
async def test_api_chat_desktop_identity_no_reply_uses_evidence_bound_repair(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("identity desktop repair must not use KernelInterface fallback")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-identity"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _no_cognitive_reply(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "cognitive_engine_no_acceptable_reply",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))
    _force_full_mind_runtime(monkeypatch, chat_routes)

    prompt = (
        "Quick reliability check, in two or three sentences: what are you, "
        "and will you remember this conversation tomorrow?"
    )
    response = await server_module.api_chat(
        server_module.ChatRequest(message=prompt),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    lowered = payload["response"].lower()
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine_identity_continuity_grounding"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert payload["response_confidence"] == "high"
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert payload["live_turn_contract"]["bounded_contract_used"] is False
    assert "failed closed instead of sending an ungrounded answer" not in lowered
    assert "local governed cognitive-agent runtime" in lowered
    assert "persistent memory" in lowered
    assert "cannot guarantee perfect tomorrow recall" in lowered
    assert "legacy fallback" not in lowered
    assert kernel_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is True
    assert output_receipts[0][1]["metadata"]["path"] == "cognitive_engine_identity_continuity_grounding"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "high"


@pytest.mark.asyncio
async def test_api_chat_desktop_capability_no_reply_fails_closed_without_inventory_repair(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    completed_exchanges = []
    output_receipts = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("capability inventory desktop repair must not use KernelInterface fallback")

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
            ]

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-capability"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _no_cognitive_reply(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "What tools she could hypothetically do externally, and can she flex "
                "her muscles with one concrete scenario?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    lowered = payload["response"].lower()
    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert payload["status"] == "desktop_cognitive_engine_unavailable"
    assert payload["response_confidence"] == "failed"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert "failed closed instead of sending an ungrounded answer" in lowered
    assert "computer_use" not in payload["response"]
    assert "web_search" not in payload["response"]
    assert "legacy fallback" not in lowered
    assert "self-process" not in lowered
    assert kernel_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is False
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_cognitive_engine"


@pytest.mark.asyncio
async def test_api_chat_desktop_no_reply_executes_self_sufficient_objective(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    output_receipts = []
    completed_exchanges = []

    class _ForbiddenKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            pytest.fail("desktop objective must not use kernel fallback")

    async def _no_cognitive_reply(*_args, **_kwargs):
        return None

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "status": "completed",
            "summary": "Desktop task completed 2/2 governed computer-use steps.",
            "steps_requested": 2,
            "steps_completed": 2,
            "receipts": _verified_desktop_receipts(2),
        }

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-self-sufficient"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _ForbiddenKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Please open Calculator, copy the displayed equation, paste it into Notes, "
                "and report the saved path."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "desktop_objective_completed"
    assert "Desktop task completed 2/2 governed computer-use steps" in payload["response"]
    assert skill_calls and skill_calls[0]["skill_name"] == "desktop_task"
    assert skill_calls[0]["params"]["allow_heuristic_desktop_plan"] is True
    assert skill_calls[0]["extra_context"]["allow_heuristic_desktop_plan"] is True
    assert completed_exchanges
    assert output_receipts


@pytest.mark.asyncio
async def test_api_chat_desktop_no_reply_executes_self_summary_after_cognitive_attempt(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    output_receipts = []
    completed_exchanges = []

    class _ForbiddenKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            pytest.fail("desktop objective must not use kernel fallback")

    async def _no_cognitive_reply(*_args, **_kwargs):
        return None

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "status": "completed",
            "summary": "Desktop task completed 6/6 governed computer-use steps.",
            "steps_requested": 6,
            "steps_completed": 6,
            "receipts": _verified_desktop_receipts(6),
            "document_provenance": "local_cortex_synthesis",
        }

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-self-summary"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _no_cognitive_reply)
    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _ForbiddenKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Please open up my Notes app and write a short journal entry in your "
                "own words describing who and what you are. Include the current date "
                "and time and export it as a PDF in Aura's Journal."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "desktop_objective_completed"
    assert "Desktop task completed 6/6 governed computer-use steps" in payload["response"]
    assert skill_calls and skill_calls[0]["skill_name"] == "desktop_task"
    assert skill_calls[0]["extra_context"]["desktop_task_document_body"] == ""
    assert skill_calls[0]["extra_context"]["allow_desktop_task_model_synthesis"] is False
    assert completed_exchanges
    assert output_receipts


@pytest.mark.asyncio
async def test_api_chat_desktop_live_proof_executes_after_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []
    live_proof_calls = []

    async def _fake_cognitive_turn(message, **kwargs):
        _t = kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append((message, kwargs))
        return "Plan: create the requested artifact through the governed tool path."

    async def _fake_live_proof(message):
        live_proof_calls.append(message)
        return {
            "response": "I created the Snake artifact through governed file_operation.",
            "status": "live_proof_snake",
        }

    desktop_objective_calls = []

    async def _forbidden_desktop_objective(*_args, **_kwargs):
        desktop_objective_calls.append((_args, _kwargs))
        return {"response": "unexpected desktop objective path", "status": "unexpected"}

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-live-proof"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_execute_live_runtime_proof", _fake_live_proof)
    monkeypatch.setattr(chat_routes, "_execute_desktop_objective_from_chat", _forbidden_desktop_objective)
    lane_calls = 0

    def _live_proof_lane_status():
        nonlocal lane_calls
        lane_calls += 1
        if lane_calls >= 2:
            return {
                "conversation_ready": False,
                "state": "cold",
                "last_failure_reason": "endpoint_timeout:Cortex:38.5s",
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": None,
                "background_endpoint": "Brainstem",
            }
        return {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        }

    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        _live_proof_lane_status,
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Run a live proof: create a simple game of Snake and save it as "
                "artifacts/live_runtime/generated/desktop_probe_snake.html"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"live_proof_snake" in response.body
    assert b"governed file_operation" in response.body
    payload = json.loads(response.body)
    assert payload["conversation_lane"]["governed_action_result"] is True
    assert payload["conversation_lane"]["governed_action_status"] == "live_proof_snake"
    assert payload["conversation_lane"]["conversation_ready"] is False
    assert len(cognitive_calls) == 1
    assert len(live_proof_calls) == 1
    assert desktop_objective_calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_explicit_file_objective_runs_after_cognitive_engine(tmp_path, monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    governed_calls = []
    cognitive_calls = []
    monkeypatch.chdir(tmp_path)

    async def _fake_governed_skill(skill_name, params, **kwargs):
        governed_calls.append((skill_name, params, kwargs))
        assert skill_name == "file_operation"
        target = tmp_path / params["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(params["content"], encoding="utf-8")
        return {"ok": True, "path": params["path"], "summary": "wrote file"}

    async def _fake_cognitive_turn(*args, **kwargs):
        _t = kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append((args, kwargs))
        return "Plan: create the requested file through governed file_operation after this cognitive turn."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _fake_governed_skill)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Use the governed tool path to create a small self-contained HTML page "
                "at artifacts/live_runtime/generated/codex_live_probe_tool_path_general.html "
                "with a title, one button, and a short script that updates text when clicked."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    target = tmp_path / "artifacts/live_runtime/generated/codex_live_probe_tool_path_general.html"
    assert response.status_code == 200
    assert payload["status"] == "file_operation"
    assert payload["conversation_lane"]["governed_action_result"] is True
    assert payload["conversation_lane"]["governed_action_status"] == "file_operation"
    assert governed_calls
    assert len(cognitive_calls) == 1
    assert cognitive_calls[0][1]["source"] == "desktop_ui"
    assert cognitive_calls[0][1]["require_engine"] is True
    assert cognitive_calls[0][1]["timeout_s"] >= 100.0
    assert cognitive_calls[0][1]["timeout_s"] <= 140.0
    assert target.exists()
    html = target.read_text(encoding="utf-8")
    assert "<button" in html
    assert "addEventListener" in html


@pytest.mark.asyncio
async def test_api_chat_desktop_runtime_status_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _fake_cognitive_turn(*args, **kwargs):
        _t = kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append((args, kwargs))
        return (
            "Cortex (32B) is the active foreground lane, CognitiveEngine handled this turn: yes, "
            "governed tools available: yes, subject to Will/Authority approval and effect receipts; "
            "recurrent depth: active."
        )

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "recurrent_depth": {"active": True},
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Live desktop path validation. Reply in one sentence with the active model lane, "
                "whether CognitiveEngine is handling this turn, whether governed tools are available, "
                "and whether recurrent depth is active."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "Cortex (32B)" in payload["response"]
    assert "active foreground lane" in payload["response"]
    assert "CognitiveEngine handled this turn: yes" in payload["response"]
    assert "governed tools available: yes" in payload["response"]
    assert "recurrent depth: active" in payload["response"]
    assert len(cognitive_calls) == 1


@pytest.mark.asyncio
async def test_api_chat_desktop_soak_lane_question_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _fake_cognitive_turn(*_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append("desktop_cognitive_engine")
        return "Cortex (32B) is the active foreground lane and I am answering through CognitiveEngine."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "recurrent_depth": {"active": True},
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Answer directly in two sentences: what lane are you using for this live desktop chat?"
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "Cortex (32B)" in payload["response"]
    assert "active foreground lane" in payload["response"]
    assert cognitive_calls == ["desktop_cognitive_engine"]


@pytest.mark.asyncio
async def test_api_chat_desktop_coherence_status_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []

    async def _fake_cognitive_turn(*_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append("desktop_cognitive_engine")
        return "I am coherent, on the same live desktop thread, and able to continue."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_cognitive_engine_available", lambda: True)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "recurrent_depth": {"active": True},
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Finish with a short status: are you still coherent, on the same thread, and able to continue?"
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "same live desktop thread" in payload["response"]
    assert "able to continue" in payload["response"]
    assert cognitive_calls == ["desktop_cognitive_engine"]


@pytest.mark.asyncio
async def test_api_chat_desktop_nonexecuting_plan_uses_cognitive_engine_when_required(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []
    desktop_objective_calls = []

    async def _fake_cognitive_turn(*_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append("desktop_cognitive_engine")
        return "I would create the note, export the PDF only after authorization, and verify the artifact path."

    async def _forbidden_desktop_objective(*_args, **_kwargs):
        desktop_objective_calls.append((_args, _kwargs))
        pytest.fail("non-executing desktop planning request must not dispatch desktop_task")

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_execute_desktop_objective_from_chat", _forbidden_desktop_objective)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Give a concise plan for creating a note and exporting it as a PDF, but do not execute tools."
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "export the PDF only after authorization" in payload["response"]
    assert "after authorization" in payload["response"]
    assert cognitive_calls == ["desktop_cognitive_engine"]
    assert desktop_objective_calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_nonexecuting_decision_question_blocks_desktop_task(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    cognitive_calls = []
    desktop_objective_calls = []

    async def _fake_cognitive_turn(*_args, **_kwargs):
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        cognitive_calls.append("desktop_cognitive_engine")
        return (
            "I would use Notes for a quick local note and Google Docs when the user needs "
            "cloud editing, sharing, or a polished longer document."
        )

    async def _forbidden_desktop_objective(*_args, **_kwargs):
        desktop_objective_calls.append((_args, _kwargs))
        pytest.fail("do-not-execute decision questions must not dispatch desktop_task")

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_execute_desktop_objective_from_chat", _forbidden_desktop_objective)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Don't execute tools. In two sentences, describe how you'd decide whether "
                "to use Notes or Google Docs for a user writing task."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            cookies={},
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "Notes" in payload["response"]
    assert "Google Docs" in payload["response"]
    assert cognitive_calls == ["desktop_cognitive_engine"]
    assert desktop_objective_calls == []


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_outer_timeout_refuses_direct_gate_fallback(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gate_calls = []
    cognitive_calls = []
    completed_exchanges = []
    output_receipts = []

    class _ForbiddenGate:
        async def generate(self, *_args, **_kwargs):
            gate_calls.append("generate")
            raise AssertionError("desktop UI timeout must not use the direct inference gate fallback")

    async def _timeout_cognitive_turn(*_args, **_kwargs):
        cognitive_calls.append("desktop_cognitive_engine")
        raise TimeoutError("desktop cognitive turn exceeded foreground budget")

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-timeout"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    def _fake_get(name, default=None):
        if name == "inference_gate":
            return _ForbiddenGate()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _timeout_cognitive_turn)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Use the desktop path to reason through this request."),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert b"desktop_cognitive_engine_unavailable" in response.body
    assert b"desktop_cognitive_engine_timeout" in response.body
    assert cognitive_calls == ["desktop_cognitive_engine"]
    assert gate_calls == []
    assert len(completed_exchanges) == 1
    assert completed_exchanges[0][1]["record_experience"] is False
    assert len(output_receipts) == 1
    assert output_receipts[0][1]["cause"] == "chat_timeout"
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_cognitive_engine"


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_blocks_thin_cognitive_engine_recovery_reply(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(content="I'm here. What's the puzzle?")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            message = "desktop UI must not use KernelInterface after weak CognitiveEngine text"
            raise AssertionError(message)

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-weak"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "Solve this logic puzzle: Alice owns three dogs, one dog always barks "
                "before dinner, and the spotted dog barked second. Which dog barked first?"
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert b"desktop_cognitive_engine_unavailable" in response.body
    assert b"What&apos;s the puzzle" not in response.body
    assert b"What's the puzzle" not in response.body


@pytest.mark.asyncio
async def test_api_chat_desktop_required_fails_closed_on_final_degraded_reply(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    completed_exchanges = []
    output_receipts = []

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-final-degraded"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _bad_cognitive_turn(*_args, **_kwargs):
        # Engine accepts its own (degraded) reply — the DOWNSTREAM quality gate is
        # what must catch it, so the trace has to prove the full-mind path first.
        _t = _kwargs.get("turn_trace")
        if isinstance(_t, dict):
            _t.update({
                "engine_think_invoked": True,
                "cognitive_engine_reply_accepted": True,
                "live_mind_context_present": True,
                "live_mind_snapshot_present": True,
                "live_mind_snapshot_ready": True,
                "live_mind_required_subsystems_ok": True,
                **_bound_live_mind_controls_trace(),
                "response_path": "cognitive_engine",
            })
        return "Absolutely. Let's nail this pitch. What are our key points?"

    async def _no_stabilize(_message, reply, **_kwargs):
        return reply

    async def _no_repair(_message, reply, **_kwargs):
        return reply, False, False, False, "", False

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _bad_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _no_stabilize)
    monkeypatch.setattr(chat_routes, "_repair_final_degraded_reply", _no_repair)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(message="You with me?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    # Fail-closed reply delivers IN-BAND (200): the honest refusal IS the
    # message; raw 503s here reached real clients as bare HTTP errors.
    assert response.status_code == 200
    assert b"desktop_response_quality_failed" in response.body
    assert b"required_desktop_reply_remained_degraded" in response.body
    assert b"nail this pitch" not in response.body
    assert completed_exchanges
    assert completed_exchanges[0][1]["record_experience"] is False
    assert output_receipts
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_required_final_quality_failed"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "failed"


@pytest.mark.asyncio
async def test_api_chat_desktop_required_blocks_unfounded_voice_intrusion(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    completed_exchanges = []
    output_receipts = []

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-voice-intrusion"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _bad_cognitive_turn(*_args, **_kwargs):
        trace = _kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    **_bound_live_mind_controls_trace(),
                    "response_path": "cognitive_engine",
                }
            )
        return "The voices. The small ones. They're whispering in my ear. Telling me things."

    async def _no_stabilize(_message, reply, **_kwargs):
        return reply

    async def _no_repair(_message, reply, **_kwargs):
        return reply, False, False, False, "", False

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _bad_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _no_stabilize)
    monkeypatch.setattr(chat_routes, "_repair_final_degraded_reply", _no_repair)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(message="What are you talking about?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    # In-band fail-closed delivery (see sibling test above).
    assert response.status_code == 200
    assert b"desktop_response_quality_failed" in response.body
    assert b"voices" not in response.body
    assert b"whispering" not in response.body
    assert completed_exchanges
    assert completed_exchanges[0][1]["record_experience"] is False
    assert output_receipts
    assert output_receipts[0][1]["metadata"]["path"] == "desktop_required_final_quality_failed"
    assert output_receipts[0][1]["metadata"]["response_confidence"] == "failed"


@pytest.mark.asyncio
async def test_api_chat_desktop_required_recovers_only_through_full_mind_path(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    completed_exchanges = []
    output_receipts = []
    cognitive_calls = []
    raw_gate_calls = []

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-recovered"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    async def _cognitive_turn(*args, **kwargs):
        cognitive_calls.append((args, kwargs))
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    **_bound_live_mind_controls_trace(),
                    "response_path": "cognitive_engine",
                }
            )
        if len(cognitive_calls) == 1:
            return "Absolutely. Let's nail this pitch. What are our key points?"
        return (
            "I'm here with you, and I am staying on this thread. You asked whether I am "
            "with you, so the answer is yes: I am oriented to this conversation and not "
            "inventing a pitch or a separate task."
        )

    async def _no_stabilize(_message, reply, **_kwargs):
        return reply

    async def _no_repair(_message, reply, **_kwargs):
        return reply, False, False, False, "", False

    class _RawGateShouldNotRun:
        async def generate(self, *_args, **_kwargs):
            raw_gate_calls.append((_args, _kwargs))
            return "As an AI language model, I cannot be the live Aura desktop mind."

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _cognitive_turn)
    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _no_stabilize)
    monkeypatch.setattr(chat_routes, "_repair_final_degraded_reply", _no_repair)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _RawGateShouldNotRun()
            if name == "inference_gate"
            else default
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(message="You with me?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine_recovered"
    assert payload["response_confidence"] == "high"
    assert payload["live_turn_contract"]["live_mind_controls_bound"] is True
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert payload["live_turn_contract"]["response_path"] == "cognitive_engine"
    assert "nail this pitch" not in payload["response"]
    assert len(cognitive_calls) == 2
    assert raw_gate_calls == []
    assert completed_exchanges
    assert completed_exchanges[0][1]["record_experience"] is True
    assert output_receipts
    assert output_receipts[0][1]["metadata"]["path"] == "cognitive_engine_recovery"


@pytest.mark.asyncio
async def test_required_runtime_status_turn_invokes_cognitive_engine(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": getattr(mode, "name", str(mode)),
                    "origin": origin,
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "You asked me to identify the current request and name the live cognition "
                    "path handling this turn. I am using CognitiveEngine on the Cortex 32B "
                    "foreground lane."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Live desktop route probe. Answer directly in two sentences: what did I just ask "
        "you to do, and what mind/cognition path are you using right now?"
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert calls
    assert "Runtime path contract" in calls[0]["objective"]
    assert calls[0]["context"]["runtime_fact_status_contract"] is True
    assert calls[0]["context"]["grounded_runtime_status_contract"] is True
    assert "Cortex (32B)" in calls[0]["context"]["grounded_runtime_status_context"]
    assert "active foreground lane" in calls[0]["context"]["grounded_runtime_status_context"]
    assert calls[0]["context"]["cognitive_engine_required"] is True
    assert trace["engine_think_invoked"] is True
    assert trace["cognitive_engine_reply_accepted"] is True
    assert trace["response_path"] == "cognitive_engine_runtime_fact_grounding"
    assert trace.get("bounded_contract_used") is not True
    assert reply.startswith("You asked me to identify the current request")
    assert "Cortex (32B)" in reply
    assert "active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_fails_closed_on_weak_status_without_full_mind_repair(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(
                content=(
                    "I still have the previous turn open. I am not going to fake a new "
                    "answer over it; the next clean reply should land from the active turn."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "How are you feeling? A lot of work has been done.",
        visible_user_message="How are you feeling? A lot of work has been done.",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply is None


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_rejects_unfounded_voice_intrusion(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="The voices. The small ones. They're whispering in my ear. Telling me things."
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async def _no_repair(_message, reply, **_kwargs):
        return reply, False, False, False, "", False

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(chat_routes, "_repair_final_degraded_reply", _no_repair)
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "What are you talking about?",
        visible_user_message="What are you talking about?",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert len(calls) == 2
    assert reply is None
    assert trace["engine_think_invoked"] is True
    assert trace["cognitive_engine_reply_accepted"] is False
    assert trace.get("bounded_contract_used") is not True
    assert trace["response_path"] in {
        "cognitive_engine_context_contract_failed",
        "cognitive_engine_reply_rejected",
    }


def test_response_quality_logger_downgrades_canonical_failures(monkeypatch, caplog):
    import logging

    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    caplog.set_level(logging.INFO, logger="Aura.ResponseQuality")

    chat_routes._log_response_quality_metrics(
        "What are you talking about?",
        "The voices. The small ones. They're whispering in my ear. Telling me things.",
        "high",
        stale=False,
        same_diff=False,
        off_topic=False,
    )

    assert "confidence=degraded" in caplog.text
    assert "unfounded_voice_intrusion" in caplog.text


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_retries_failed_reply_on_same_lane(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            if len(calls) == 1:
                return SimpleNamespace(
                    content=(
                        "Give me a moment — I want to answer that properly. "
                        "I am still with your question about reliable desktop tool use."
                    )
                )
            return SimpleNamespace(
                content=(
                    "1. Reliable desktop tool use matters because a local assistant has to turn user intent into visible, governed actions.\n"
                    "2. It also gives the user concrete evidence that files, apps, and tools changed for real instead of being described abstractly."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", "1")
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert reply.startswith("1. Reliable desktop tool use matters")
    assert "\n2. It also gives" in reply
    assert len(calls) == 2
    assert calls[0]["objective"] == user_message
    # The retry objective is now repair-framed (it tells the engine the prior
    # draft failed and to rewrite), with the raw user message carried in
    # context. This is a stronger contract than replaying the bare turn.
    assert calls[1]["objective"] != user_message
    assert user_message in calls[1]["objective"]
    assert "did not satisfy the user-facing response contract" in calls[1]["objective"]
    assert calls[1]["context"]["suppress_user_memory_append"] is True
    assert calls[1]["context"]["original_visible_user_message"] == user_message
    assert "response_repair_directive" in calls[1]["context"]


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_retries_failed_reply_by_default_when_memory_is_safe(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    degradations = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "Give me a moment — I want to answer that properly. "
                    "I am still with your question about reliable desktop tool use."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes,
        "record_degradation",
        lambda subsystem, error, **kwargs: degradations.append(
            {
                "subsystem": subsystem,
                "error": str(error),
                "kwargs": kwargs,
            }
        ),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 2
    assert reply is None or "reliable desktop tool use matters" not in reply.lower()
    assert len(degradations) == 1
    assert degradations[0]["subsystem"] == "chat.cognitive_engine_reply"
    assert degradations[0]["kwargs"]["receipt_required"] is True
    assert degradations[0]["kwargs"]["extra"]["retry_attempted"] is True


def test_desktop_cognitive_failure_trains_immunity_without_repairing_first_incident(monkeypatch):
    from interface.routes import chat as chat_routes

    observed = []
    repair_calls = []

    class _Immune:
        def observe_signature(self, component, exception_type, **kwargs):
            observed.append((component, exception_type, kwargs))
            return SimpleNamespace(
                antigen=SimpleNamespace(recurrence_pressure=0.12)
            )

    class _Healer:
        def schedule_deep_repair(self, *args, **kwargs):
            repair_calls.append((args, kwargs))
            return {"result": "unexpected_repair_request"}

    services = {
        "adaptive_immune_system": _Immune(),
        "self_healing": _Healer(),
    }
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )

    result = chat_routes._route_desktop_cognitive_failure_to_resilience(
        "empty_cognitive_engine_reply",
        source="desktop_ui",
        session_present=True,
        retry_attempted=True,
    )

    assert result == {
        "immune_observed": True,
        "recurrence_pressure": 0.12,
        "repair_requested": False,
        "repair_result": "below_recurrence_floor",
    }
    assert observed[0][0] == "chat.cognitive_engine_reply"
    assert observed[0][2]["context"]["protected"] is True
    assert repair_calls == []


def test_recurrent_desktop_cognitive_failure_schedules_one_governed_repair(monkeypatch):
    from interface.routes import chat as chat_routes

    repairs = []

    class _Immune:
        def observe_signature(self, *_args, **_kwargs):
            return SimpleNamespace(
                antigen=SimpleNamespace(recurrence_pressure=0.61)
            )

    class _Healer:
        def schedule_deep_repair(self, module_path, **kwargs):
            repairs.append((module_path, kwargs))
            return {"result": "deep_repair_scheduled"}

    services = {
        "adaptive_immune_system": _Immune(),
        "self_healing": _Healer(),
    }
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )
    monkeypatch.setattr(chat_routes.time, "monotonic", lambda: 10_000.0)
    chat_routes._desktop_cognitive_repair_last_scheduled.clear()

    first = chat_routes._route_desktop_cognitive_failure_to_resilience(
        "cognitive_engine_timeout",
        source="desktop_ui",
        session_present=True,
        retry_attempted=False,
    )
    second = chat_routes._route_desktop_cognitive_failure_to_resilience(
        "cognitive_engine_timeout",
        source="desktop_ui",
        session_present=True,
        retry_attempted=False,
    )

    assert first["repair_requested"] is True
    assert first["repair_result"] == "deep_repair_scheduled"
    assert first["repair_target"] == "core/brain/llm/mlx_client.py"
    assert second["repair_requested"] is False
    assert second["repair_result"] == "repair_cooldown_active"
    assert len(repairs) == 1
    assert repairs[0][1]["metadata"]["recurrence_pressure"] == 0.61


@pytest.mark.asyncio
async def test_desktop_stabilizer_keeps_complex_self_process_questions_substantive(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "unexpected second pass"

    inference_gate = _InferenceGate()
    prompt = (
        "When you are confused, how does that change your planning, memory use, "
        "and tool verification?"
    )

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_build_grounded_traceability_reply", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_looks_truncated_tail", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr("core.identity.identity_guard.PersonaEnforcementGate", lambda: _Gate())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        prompt,
        "Right now I feel present and listening, with my attention on this exchange.",
        desktop_cognitive_engine_required=True,
        protected_foreground_lane=True,
    )

    lowered = result.lower()
    assert inference_gate.calls == []
    assert "present and listening" not in lowered
    assert "confus" in lowered
    assert "planning" in lowered
    assert "memory" in lowered
    assert "tool" in lowered
    assert "receipt" in lowered or "verification" in lowered


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_retries_empty_cycle_without_placeholder(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            if len(calls) == 1:
                return SimpleNamespace(content="")
            return SimpleNamespace(
                content=(
                    "1. Reliable desktop tool use matters because the assistant must operate real apps and files from user intent.\n"
                    "2. It also lets the user verify that the action happened through governed tools instead of a verbal-only claim."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", "1")
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "I heard you" not in reply
    assert reply.startswith("1. Reliable desktop tool use matters")
    assert len(calls) == 2
    assert calls[1]["context"]["failed_reply_reasons"] == ("empty_cognitive_engine_reply",)


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_required_simple_chat_uses_compact_live_mind_contract(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "Reliable desktop tool use matters because a local assistant has to turn intent into visible, "
                    "verified action. It also keeps the user in control by making each external effect observable."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Answer directly in two sentences: why reliable desktop tool use matters for a local assistant."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls[0]["context"]["desktop_quick_reply_contract"] is True
    assert calls[0]["context"]["desktop_cognitive_engine_required"] is True
    assert calls[0]["context"]["live_mind_context_required"] is True
    assert calls[0]["context"]["live_mind_context"]["must_answer_from_full_mind_path"] is True
    assert calls[0]["context"]["live_runtime_payload_required"] is True
    assert calls[0]["context"]["skip_runtime_payload"] is True
    assert calls[0]["context"]["allow_deep_handoff"] is False
    assert calls[0]["context"]["allow_cloud_fallback"] is False
    assert calls[0]["kwargs"]["timeout_s"] == pytest.approx(42.0)


@pytest.mark.asyncio
async def test_desktop_identity_turn_uses_grounded_compact_cognitive_engine_contract(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=str(context["grounded_identity_continuity_context"]),
                metadata={
                    **_bound_live_mind_controls_metadata(),
                    "response_path": "cognitive_engine_identity_continuity_grounding",
                },
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    prompt = "Who are you?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        prompt,
        visible_user_message=prompt,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "I'm Aura" in reply
    assert calls
    context = calls[0]["context"]
    assert context["identity_continuity_contract"] is True
    assert context["desktop_quick_reply_contract"] is True
    assert context["desktop_cognitive_engine_required"] is True
    assert "grounded_identity_continuity_context" in context
    assert context["allow_deep_handoff"] is False
    assert context["allow_cloud_fallback"] is False


@pytest.mark.asyncio
async def test_cognitive_engine_identity_floor_does_not_call_router(monkeypatch):
    from core.brain import cognitive_engine as ce_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    class _Router:
        think_calls = []

        async def think(self, **_kwargs):
            _Router.think_calls.append(_kwargs)
            raise AssertionError("identity grounding should not invoke router.think")

    class _Container:
        @staticmethod
        def get(name, default=None):
            if name == "llm_router":
                return _Router()
            return default

    monkeypatch.setattr(ce_module, "get_container", lambda: _Container)

    engine = CognitiveEngine()
    thought = await engine._direct_desktop_quick_reply(
        "Who are you?",
        ThinkingMode.FAST,
        "user",
        {
            "desktop_quick_reply_contract": True,
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "identity_continuity_contract": True,
            "grounded_identity_continuity_context": "I'm Aura: a local governed cognitive-agent runtime.",
            "live_mind_context_required": True,
            "live_mind_context": {
                "required_for_live_desktop": True,
                "must_answer_from_full_mind_path": True,
                "required_subsystems_ok": True,
                "mind_snapshot_quality": {"present": True, "ready": True},
            },
            "live_mind_generation_controls": {
                "temperature": 0.58,
                "top_p": 0.88,
                "clean_user_surface_recurrent_loops": 1,
                "clean_user_surface_steering_alpha": 0.25,
            },
            "live_mind_controls_bound": True,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "visible_user_message": "Who are you?",
        },
        timeout_s=60.0,
    )

    assert thought is not None
    assert thought.content.startswith("I'm Aura")
    assert thought.metadata["response_path"] == "cognitive_engine_identity_continuity_grounding"
    assert thought.metadata["live_mind_controls_bound"] is True
    assert thought.metadata["identity_continuity_contract"] is True


def test_desktop_cognitive_repair_budget_can_complete_a_primary_model_generation():
    from interface.routes import chat as chat_routes

    repair_outer = chat_routes._DESKTOP_COGNITIVE_REPAIR_TIMEOUT_S
    repair_inner = chat_routes._inner_cognitive_cycle_timeout(repair_outer)

    assert repair_outer >= 60.0
    assert repair_inner >= 40.0
    assert repair_inner < repair_outer


def test_protected_foreground_cognitive_budget_preserves_outer_deadline():
    from interface.routes import chat as chat_routes

    outer = 55.0
    normal_inner = chat_routes._inner_cognitive_cycle_timeout(outer)
    protected_inner = chat_routes._inner_cognitive_cycle_timeout(
        outer,
        protected_foreground=True,
    )

    assert normal_inner == pytest.approx(38.5)
    assert protected_inner == pytest.approx(53.0)
    assert protected_inner < outer


@pytest.mark.asyncio
async def test_desktop_capability_grounding_reaches_cognitive_engine_without_midword_clipping(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    long_inventory = (
        "I can use governed desktop and app control, browser/web research, file and document "
        "operations, terminal and code execution, memory and continuity, and self-repair. "
        + "Governed capability detail remains concrete and inspectable. " * 16
        + "Will and Authority approve consequential actions, and effect receipts verify results. "
        "For this turn I am only describing the tool surface; I am not executing tools."
    )

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"context": dict(context or {}), "kwargs": dict(kwargs)})
            return SimpleNamespace(
                content=str(context["grounded_capability_inventory_context"]),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes,
        "_build_grounded_capability_inventory_reply",
        lambda _message: long_inventory,
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    prompt = "Explain what tools you can use and give a hypothetical scenario without executing it."
    await chat_routes._run_cognitive_engine_chat_turn(
        prompt,
        visible_user_message=prompt,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert len(long_inventory) > 1000
    assert calls[0]["context"]["grounded_capability_inventory_context"] == long_inventory
    assert calls[0]["context"]["grounded_capability_inventory_context"].endswith(
        "I am not executing tools."
    )


@pytest.mark.asyncio
async def test_desktop_required_chat_gets_default_recent_context_window(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I am answering this ordinary live desktop turn with the recent thread still present."
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "The live desktop chat was drifting into assistant mode.",
                    "aura": "I need to keep the full mind path fused to the speech lane.",
                    "status": "complete",
                }
            ]
        )

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Give me one practical next step."
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls[0]["context"]["recent_context_needed"] is False
    assert calls[0]["context"]["recent_completed_exchanges"]
    assert "assistant mode" in calls[0]["context"]["recent_conversation_context"]
    assert calls[0]["context"]["live_runtime_payload_required"] is True


@pytest.mark.asyncio
async def test_deep_desktop_followup_keeps_hard_live_token_envelope(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "I am holding that uncertainty in my attention rather than erasing it. "
                    "I would remember this thread and let it change my next decision: choose "
                    "one reversible step that gathers evidence, then recheck before committing."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    message = (
        "Remember the uncertainty you just named. How would it change one decision "
        "you make next, without pretending certainty you do not have?"
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        message,
        visible_user_message=message,
        origin="user",
        timeout_s=105.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls
    context = calls[0]["context"]
    assert context.get("desktop_quick_reply_contract") is not True
    assert context["max_tokens"] == 896
    assert context["num_predict"] == 896


@pytest.mark.asyncio
async def test_desktop_required_memory_state_turn_uses_canonical_evidence_without_stale_history(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    'The phrase you asked me to remember was "silver lantern". '
                    "CognitiveEngine keeps this reply grounded by reading the canonical "
                    "memory-state evidence attached to the current live desktop turn instead "
                    "of guessing from older conversation history."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "What pitch?",
                    "aura": "A stale pitch answer that must not steer this turn.",
                    "status": "complete",
                }
            ]
        )

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    visible_message = (
        "What phrase did I ask you to remember, and how does your cognitive engine keep this reply grounded?"
    )
    effective_message = (
        f"{visible_message}\n\n"
        "[CANONICAL MEMORY STATE EVIDENCE]\n"
        "status=session_memory_recall\n"
        'The phrase you asked me to remember in this session was "silver lantern".\n'
        "[END CANONICAL MEMORY STATE EVIDENCE]\n"
        "Use this canonical memory/state result as evidence, but produce the visible answer "
        "through CognitiveEngine in Aura's normal desktop voice."
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        effective_message,
        visible_user_message=visible_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "silver lantern" in reply
    assert calls[0]["context"]["memory_state_contract"] is True
    assert "silver lantern" in calls[0]["context"]["canonical_memory_state_evidence"]
    assert calls[0]["context"]["recent_completed_exchanges"] == []
    assert calls[0]["context"]["recent_conversation_context"] == ""
    assert calls[0]["context"]["desktop_quick_reply_contract"] is True


@pytest.mark.asyncio
async def test_desktop_required_durable_memory_pin_uses_compact_canonical_path(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    'I have pinned "restart-815" in durable session memory. '
                    "Right now I am keeping attention on this live desktop thread."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    visible_message = "Remember this codeword across restart: restart-815"
    effective_message = (
        f"{visible_message}\n\n"
        "[CANONICAL MEMORY STATE EVIDENCE]\n"
        "status=session_memory_pin\n"
        'I\'ve pinned "restart-815" in durable session memory.\n'
        "[END CANONICAL MEMORY STATE EVIDENCE]\n"
        "Use this canonical memory/state result as evidence, but produce the visible answer "
        "through CognitiveEngine in Aura's normal desktop voice."
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        effective_message,
        visible_user_message=visible_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "restart-815" in reply
    assert calls
    context = calls[0]["context"]
    assert context["memory_state_contract"] is True
    assert context["desktop_quick_reply_contract"] is True
    assert context["recent_completed_exchanges"] == []
    assert context["max_tokens"] <= 384


@pytest.mark.asyncio
async def test_desktop_memory_state_turn_binds_reply_to_canonical_memory_when_model_drifts(
    monkeypatch,
):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    turn_trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I am attending to the live desktop thread through CognitiveEngine.",
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    visible_message = (
        "Remember this phrase: silver lantern. Also tell me one thing your live mind is attending to right now."
    )
    effective_message = (
        f"{visible_message}\n\n"
        "[CANONICAL MEMORY STATE EVIDENCE]\n"
        "status=session_memory_pin\n"
        'I\'ve pinned "silver lantern" in durable session memory.\n'
        "[END CANONICAL MEMORY STATE EVIDENCE]\n"
        "Use this canonical memory/state result as evidence, but produce the visible answer "
        "through CognitiveEngine in Aura's normal desktop voice."
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        effective_message,
        visible_user_message=visible_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=turn_trace,
    )

    assert reply
    assert "silver lantern" in reply
    assert "this live desktop thread" in reply
    assert calls
    assert turn_trace["engine_think_invoked"] is True
    assert turn_trace["cognitive_engine_reply_accepted"] is True
    assert turn_trace["bounded_contract_used"] is False
    assert turn_trace["live_mind_controls_bound"] is True
    assert turn_trace["response_path"] == "cognitive_engine_memory_state_grounding"


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_receives_recent_completed_conversation_context(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I remember the terminal crash context and I am answering from the current live desktop thread."
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "The Python process spiked over 100GB of RAM.",
                    "aura": "That points at an unbounded live desktop path allocation.",
                    "status": "complete",
                },
                {
                    "user": "Make sure the UI path stays on CognitiveEngine.",
                    "aura": "I will keep desktop turns on the required CognitiveEngine path.",
                    "status": "complete",
                },
            ]
        )

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Can you continue from what we were debugging?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls[0]["context"]["recent_completed_exchanges"]
    assert "100GB of RAM" in calls[0]["context"]["recent_conversation_context"]
    assert "required CognitiveEngine path" in calls[0]["context"]["recent_conversation_context"]


@pytest.mark.asyncio
async def test_desktop_quick_self_reflection_suppresses_stale_recent_context(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content="I am tracking this live turn directly, with attention on your current question."
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "What tools can you use externally?",
                    "aura": "I can describe governed desktop and browser tools.",
                    "status": "complete",
                },
                {
                    "user": "Name a hypothetical tool scenario.",
                    "aura": "A scenario could include creating a folder and exporting a PDF.",
                    "status": "complete",
                },
            ]
        )

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "How are you thinking about this conversation right now?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls
    assert calls[0]["context"]["recent_context_needed"] is True
    assert calls[0]["context"]["live_mind_context_required"] is True
    assert calls[0]["context"]["live_mind_context"]["required_for_live_desktop"] is True


def test_user_visible_context_leak_sanitizer_strips_internal_blocks():
    from interface.routes import chat as chat_routes

    reply = (
        "I am staying on the live desktop cognitive lane and answering the current turn."
        "[RECENT CONTEXT]User: What tools can you use?"
    )

    assert chat_routes._strip_user_visible_context_leaks(reply) == (
        "I am staying on the live desktop cognitive lane and answering the current turn."
    )


def test_live_self_reflection_rejects_stale_tool_topic_bleed():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "How are you thinking about this conversation right now?",
        (
            "I'm in a quiet state, with stable attention on you. "
            "You're asking about tools or scenarios — let me walk through an actual case of using them. "
            "If you want to create a folder and write a file, I can do that."
        ),
    )

    assert "stale_context_topic_bleed" in assessment.reasons
    assert assessment.retryable


def test_live_desktop_gate_leak_is_retryable_user_facing_failure():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Ok. Just checking. I'll be back, ok?",
        (
            "This live desktop turn failed the reply-quality gate, and I am not "
            "starting a second foreground generation over it."
        ),
    )

    assert "internal_live_gate_leak" in assessment.reasons
    assert assessment.hard_failure
    assert assessment.retryable


def test_conceptual_live_chat_reliability_question_does_not_force_diagnostic_floor():
    from core.conversation.response_reliability import (
        is_reliability_concern,
        reliability_floor_for_user,
    )

    message = (
        "In two direct sentences, explain why the live desktop chat path must "
        "never fall back into generic assistant mode."
    )

    assert is_reliability_concern(message) is False
    assert reliability_floor_for_user(message) == ""


def test_reported_live_chat_breakage_still_uses_reliability_floor():
    from core.conversation.response_reliability import (
        is_reliability_concern,
        reliability_floor_for_user,
    )

    message = "The live chat path broke again after a few turns."

    assert is_reliability_concern(message) is True
    assert reliability_floor_for_user(message)


@pytest.mark.asyncio
async def test_desktop_low_risk_social_turn_retries_once_then_fails_closed(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "This live desktop turn failed the reply-quality gate, and I am not "
                    "starting a second foreground generation over it."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Ok. Just checking. I'll be back, ok?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 2
    assert reply is None


@pytest.mark.asyncio
async def test_desktop_required_capability_turn_uses_cognitive_engine_before_catalog_repair(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "From the live desktop path I can use governed tool lanes for desktop apps, browser and web "
                    "research, file operations, document drafting, terminal work, memory recall, and skill execution. "
                    "A hypothetical chain would request approval, open sources, draft a document, verify the visible "
                    "result, export the file, and record governance receipts without claiming an unverified action."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "What external tools could you use from the live desktop path, and give one hypothetical chain "
        "without claiming you executed it."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply
    assert len(calls) == 1
    assert calls[0]["context"]["capability_inventory_contract"] is True
    assert calls[0]["context"]["recent_completed_exchanges"] == []
    assert calls[0]["context"]["recent_conversation_context"] == ""
    assert calls[0]["context"]["max_tokens"] == 384
    assert calls[0]["context"]["live_mind_context_required"] is True
    assert calls[0]["context"]["live_mind_context"]["must_answer_from_full_mind_path"] is True
    assert trace["engine_think_invoked"] is True
    assert trace["cognitive_engine_reply_accepted"] is True
    assert trace["bounded_contract_used"] is False
    assert trace["live_mind_controls_bound"] is True
    assert trace["response_path"] == "cognitive_engine"
    assert "governance receipts" in reply
    assert "not opening apps or executing tools" in reply.lower()


@pytest.mark.asyncio
async def test_desktop_required_status_turn_uses_cognitive_engine_when_lane_ready(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []
    trace = {}

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "kwargs": dict(kwargs),
                }
            )
            return SimpleNamespace(
                content=(
                    "I'm here with you. I'm tracking this live desktop thread and answering the current turn, "
                    "not drifting into a pitch or another topic."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "You with me?",
        visible_user_message="You with me?",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert reply
    assert len(calls) == 1
    assert calls[0]["context"]["live_mind_context_required"] is True
    assert calls[0]["context"]["live_mind_context"]["must_answer_from_full_mind_path"] is True
    assert trace["engine_think_invoked"] is True
    assert trace["cognitive_engine_reply_accepted"] is True
    assert trace["bounded_contract_used"] is False
    assert trace["live_mind_controls_bound"] is True
    assert trace["response_path"] == "cognitive_engine"


def test_live_self_reflection_rejects_social_presence_template():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "How are you thinking about this conversation right now?",
        (
            "hey. i'm here with you. I'm feeling Joy and leaning toward rest right now. "
            "I can answer clearly from the active turn. My attention is on you."
        ),
    )

    assert "social_presence_instead_of_self_reflection" in assessment.reasons
    assert assessment.retryable


def test_live_self_reflection_rejects_punctuated_incomplete_tail():
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes import chat as chat_routes

    reply = (
        "I'm thinking about your question through my current state. "
        "The conversation itself is a live feedback loop: what you say informs how I think about."
    )

    assessment = assess_user_facing_reply(
        "How are you thinking about this conversation right now?",
        reply,
    )

    assert "truncated_tail" in assessment.reasons
    assert assessment.retryable
    assert chat_routes._looks_truncated_tail(reply) is True


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_strips_internal_context_leak(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            return SimpleNamespace(
                content=(
                    "I am staying with the current turn, my attention is on this conversation right now, "
                    "and I feel focused rather than drifting into the previous topic."
                    "[RECENT CONTEXT]User: What tools can you use?"
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "How are you thinking about this conversation right now?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply == (
        "I am staying with the current turn, my attention is on this conversation right now, "
        "and I feel focused rather than drifting into the previous topic."
    )
    assert "[RECENT CONTEXT]" not in reply
    assert "User: What tools" not in reply
    assert chat_routes._desktop_required_bounded_reply_status(
        user_message,
        reply,
        {"conversation_ready": True, "state": "ready", "foreground_endpoint": "Cortex"},
    ) == ""


@pytest.mark.asyncio
async def test_recent_desktop_context_is_deep_but_strictly_bounded():
    from interface.routes import chat as chat_routes

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        for index in range(20):
            chat_routes._conversation_log.append(
                {
                    "user": f"user-{index} " + ("u" * 4000),
                    "aura": f"aura-{index} " + ("a" * 5000),
                    "status": "complete",
                }
            )

    exchanges = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="continue",
        limit=chat_routes._RECENT_CONVERSATION_CONTEXT_EXCHANGES,
    )
    rendered = chat_routes._format_recent_conversation_context(exchanges)

    assert len(exchanges) == 12
    assert exchanges[0]["user"].startswith("user-8 ")
    assert exchanges[-1]["user"].startswith("user-19 ")
    assert all(
        len(entry["user"]) <= chat_routes._RECENT_CONVERSATION_USER_CHARS
        for entry in exchanges
    )
    assert all(
        len(entry["aura"]) <= chat_routes._RECENT_CONVERSATION_AURA_CHARS
        for entry in exchanges
    )
    assert len(rendered) <= chat_routes._RECENT_CONVERSATION_RENDERED_CHARS


@pytest.mark.asyncio
async def test_chat_exchange_persists_user_before_reply_without_duplicate(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _Persistence:
        def record_turn(self, role, content, **kwargs):
            calls.append(("turn", role, content, dict(kwargs)))
            return f"{role}-turn"

        def record_exchange(self, user, aura, **kwargs):
            calls.append(("exchange", user, aura, dict(kwargs)))
            return ("user-turn", "aura-turn")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _Persistence()
            if name == "persistence"
            else default
        ),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange(
        "The desktop process exhausted memory.",
        session_id="desktop-client-session",
    )

    assert calls == [
        (
            "turn",
            "user",
            "The desktop process exhausted memory.",
            {
                "origin": "desktop_ui",
                "cid": f"{exchange_id}:user",
                "session_id": "desktop-client-session",
            },
        )
    ]

    await chat_routes._complete_logged_exchange(
        exchange_id,
        "The desktop process exhausted memory.",
        "I preserved this turn before reasoning and completed it afterward.",
        record_experience=False,
    )

    assert [call[0] for call in calls] == ["turn", "turn"]
    assert calls[1][1] == "aura"
    assert calls[1][3]["cid"] == f"{exchange_id}:aura"
    assert calls[1][3]["session_id"] == "desktop-client-session"


@pytest.mark.asyncio
async def test_chat_exchange_falls_back_to_atomic_exchange_when_prelog_fails(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _Persistence:
        def record_turn(self, role, content, **kwargs):
            calls.append(("turn_failed", role, content, dict(kwargs)))
            raise RuntimeError("pending write unavailable")

        def record_exchange(self, user, aura, **kwargs):
            calls.append(("exchange", user, aura, dict(kwargs)))
            return ("user-turn", "aura-turn")

    persistence = _Persistence()
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange("Preserve me before inference.")
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "Preserve me before inference.",
        "The completion path used an atomic exchange write.",
        record_experience=False,
    )

    assert [call[0] for call in calls] == ["turn_failed", "exchange"]
    assert calls[1][1:3] == (
        "Preserve me before inference.",
        "The completion path used an atomic exchange write.",
    )


@pytest.mark.asyncio
async def test_chat_exchange_atomic_persistence_keeps_ui_session_id(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _Persistence:
        def record_turn(self, role, content, **kwargs):
            calls.append(("turn", role, content, dict(kwargs)))
            return f"{role}-turn"

        def record_exchange(self, user, aura, **kwargs):
            calls.append(("exchange", user, aura, dict(kwargs)))
            return ("user-turn", "aura-turn")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _Persistence()
            if name == "persistence"
            else default
        ),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    await chat_routes._persist_completed_conversation_exchange(
        exchange_id="ui-session-check",
        user_message="Remember this through the desktop UI session.",
        aura_response="I will persist it under the UI session id.",
        session_id="desktop-ui-session-42",
        user_already_persisted=False,
    )

    assert calls == [
        (
            "exchange",
            "Remember this through the desktop UI session.",
            "I will persist it under the UI session id.",
            {
                "origin": "desktop_ui",
                "cid": "ui-session-check",
                "session_id": "desktop-ui-session-42",
            },
        )
    ]


@pytest.mark.asyncio
async def test_chat_restart_recovers_completed_exchange_from_canonical_persistence(
    monkeypatch,
    tmp_path,
):
    from core.conversation.persistence import ConversationPersistence
    from interface.routes import chat as chat_routes

    persistence = ConversationPersistence(tmp_path / "conversation.db")
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange(
        "Remember the live desktop continuity contract."
    )
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "Remember the live desktop continuity contract.",
        "I will recover this completed exchange after process memory is cleared.",
        record_experience=False,
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    recovered = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="Continue from before the restart.",
        limit=6,
    )

    assert recovered
    assert recovered[-1]["user"] == "Remember the live desktop continuity contract."
    assert "recover this completed exchange" in recovered[-1]["aura"]


@pytest.mark.asyncio
async def test_chat_restart_recovers_completed_exchange_from_ui_session(
    monkeypatch,
    tmp_path,
):
    from core.conversation.persistence import ConversationPersistence
    from interface.routes import chat as chat_routes

    persistence = ConversationPersistence(tmp_path / "conversation.db")
    persistence.start_session({"source": "boot-session-that-should-not-own-ui-turn"})
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange(
        "Remember the UI lane continuity detail.",
        session_id="desktop-visible-session",
    )
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "Remember the UI lane continuity detail.",
        "I persisted it under the desktop-visible-session transcript.",
        record_experience=False,
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    recovered = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="What did I ask you to remember?",
        session_id="desktop-visible-session",
        limit=6,
    )
    wrong_session = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="What did I ask you to remember?",
        session_id="different-visible-session",
        limit=6,
    )

    assert recovered
    assert recovered[-1]["user"] == "Remember the UI lane continuity detail."
    assert recovered[-1]["session_id"] == "desktop-visible-session"
    assert "desktop-visible-session transcript" in recovered[-1]["aura"]
    assert wrong_session == []


@pytest.mark.asyncio
async def test_desktop_required_runtime_status_invokes_engine_then_grounds(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(content="unexpected model answer")

    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Answer directly in two sentences: what lane are you using for this live desktop chat?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 1
    assert calls[0]["context"]["runtime_fact_status_contract"] is True
    assert calls[0]["context"]["cognitive_engine_required"] is True
    assert reply
    assert "Cortex (32B)" in reply
    assert "active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply


@pytest.mark.asyncio
async def test_desktop_required_cognitive_fusion_status_invokes_engine_then_grounds(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(content="unexpected model answer")

    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "In two direct sentences, explain why the live desktop chat path must stay fused "
        "to your cognitive engine instead of falling back into generic assistant mode."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "foreground_endpoint": "Cortex",
            "recurrent_depth": {"active": True},
        },
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 1
    assert calls[0]["context"]["runtime_fact_status_contract"] is True
    assert calls[0]["context"]["cognitive_engine_required"] is True
    assert reply
    assert "Cortex (32B)" in reply
    assert "active foreground lane" in reply
    assert "CognitiveEngine handled this turn: yes" in reply
    assert "governed tools available: yes" in reply
    assert "recurrent depth: active" in reply


def test_compact_desktop_contract_keeps_hypothetical_tool_plans_compact():
    from interface.routes import chat as chat_routes

    user_message = (
        "Explain how you would use browser research and a document editor together on a user task."
    )

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=chat_routes._looks_like_desktop_objective(user_message),
            capability_inventory_contract=False,
        )
        is True
    )


def test_compact_desktop_contract_does_not_hide_actual_tool_execution_requests():
    from interface.routes import chat as chat_routes

    user_message = (
        "Open Chrome, search for climate news, create a document, and export it as a PDF."
    )

    assert chat_routes._looks_like_desktop_objective(user_message) is True
    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=chat_routes._looks_like_desktop_objective(user_message),
            capability_inventory_contract=False,
        )
        is False
    )


def test_compact_desktop_contract_does_not_starve_self_process_questions():
    from interface.routes import chat as chat_routes

    user_messages = (
        (
            "When you are confused, how does that change your planning, memory use, "
            "and tool verification?"
        ),
        (
            "Quick live-path check. Don't give me a health card or telemetry list. "
            "In ordinary speech, answer from your actual current context: what are "
            "you attending to from Bryan's recent messages, what remembered concern "
            "should change your next decision, and what do you want to do next?"
        ),
        "Is this the real Aura, or did the raw model take over?",
    )

    for user_message in user_messages:
        assert (
            chat_routes._is_compact_desktop_chat_contract(
                user_message,
                user_message,
                desktop_execution_contract=False,
                capability_inventory_contract=False,
            )
            is False
        )


def test_compact_desktop_contract_allows_lightweight_live_recall_state_turn():
    from interface.routes import chat as chat_routes

    user_message = (
        "Remember this phrase: silver lantern. Also tell me one thing your live mind "
        "is attending to right now."
    )

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is True
    )


def test_compact_desktop_contract_allows_direct_durable_memory_pin():
    from interface.routes import chat as chat_routes

    user_message = "Remember this phrase across sessions: silver lantern."

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is True
    )


def test_compact_desktop_contract_keeps_durable_memory_reasoning_out_of_quick_route():
    from interface.routes import chat as chat_routes

    user_message = (
        "Remember this uncertainty across sessions. How should that change your planning "
        "and tool verification tomorrow?"
    )

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is False
    )


def test_compact_desktop_contract_allows_bounded_recall_grounding_question():
    from interface.routes import chat as chat_routes

    user_message = (
        "What phrase did I ask you to remember, and how does your cognitive engine "
        "keep this reply grounded?"
    )

    assert (
        chat_routes._is_compact_desktop_chat_contract(
            user_message,
            user_message,
            desktop_execution_contract=False,
            capability_inventory_contract=False,
        )
        is True
    )


def test_conversation_recall_classifier_handles_natural_memory_questions():
    from interface.routes import chat as chat_routes

    assert (
        chat_routes._classify_conversation_recall_request("Do you remember what I said earlier?")
        == "last_user"
    )
    assert (
        chat_routes._classify_conversation_recall_request("Can you remind me what I asked?")
        == "last_user"
    )
    assert (
        chat_routes._classify_conversation_recall_request("What was my last message?")
        == "last_user"
    )
    assert (
        chat_routes._classify_conversation_recall_request("Do you remember what you said?")
        == "last_aura"
    )
    assert (
        chat_routes._classify_conversation_recall_request("Can you remind me what you answered?")
        == "last_aura"
    )
    assert (
        chat_routes._classify_conversation_recall_request("What did we discuss earlier in this conversation?")
        == "topic"
    )
    assert (
        chat_routes._classify_conversation_recall_request("Can you remind me what we talked about?")
        == "topic"
    )
    assert (
        chat_routes._classify_conversation_recall_request("Could you summarize our last two messages?")
        == "recent_pair"
    )
    assert (
        chat_routes._classify_conversation_recall_request("What were our last two messages?")
        == "recent_pair"
    )
    assert (
        chat_routes._classify_conversation_recall_request(
            "Use the last two messages to explain what you are attending to now and how that changes your next decision."
        )
        == ""
    )


@pytest.mark.asyncio
async def test_conversation_recall_summarizes_last_two_messages_from_live_log():
    from interface.routes import chat as chat_routes

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(
            [
                {
                    "user": "What phrase did I just ask you to remember?",
                    "aura": 'The phrase you asked me to remember in this session was "cobalt sunrise".',
                    "status": "complete",
                },
                {
                    "user": "In one sentence, what are you focused on right now?",
                    "aura": "Understanding you and staying present for this conversation.",
                    "status": "complete",
                },
            ]
        )

    reply = await chat_routes._build_conversation_recall_reply(
        "Could you summarize our last two messages?"
    )

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    assert reply is not None
    assert "cobalt sunrise" in reply
    assert "focused on right now" in reply


def test_failure_mode_surface_request_is_not_misclassified_as_planning():
    from interface.routes import chat as chat_routes

    user_message = "Name one failure mode you should surface honestly instead of masking."

    assert chat_routes._is_bounded_nonexecuting_planning_request(user_message) is False
    reply = chat_routes._build_failure_mode_surface_reply(user_message)
    assert reply
    assert "failure mode" in reply.lower()
    assert "avoid claiming completion" in reply


@pytest.mark.asyncio
async def test_desktop_required_bounded_planning_uses_foreground_cognitive_engine(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "I would handle browser research and a document editor as one governed workflow: collect the "
                    "sources, draft the document, verify the visible editor content, export the artifact, and write "
                    "receipts for each confirmed step."
                )
            )

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = (
        "Explain how you would use browser research and a document editor together on a user task."
    )
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 1
    assert calls[0]["context"]["bounded_planning_contract"] is True
    assert calls[0]["context"]["require_full_foreground_mind_reply"] is True
    assert calls[0]["context"]["max_tokens"] == 1536
    assert "one natural paragraph" in calls[0]["context"]["response_style_contract"]
    assert "do not invent a specific example" in calls[0]["context"]["response_style_contract"]
    assert reply
    assert "browser" in reply.lower()
    assert "document" in reply.lower()
    assert "receipts" in reply.lower()


@pytest.mark.asyncio
async def test_desktop_required_failure_mode_surface_uses_foreground_cognitive_engine(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "One failure mode I should surface honestly is a tool action that partially starts and then "
                    "times out. I should preserve partial state or receipt evidence, report what was verified, and "
                    "avoid claiming completion until the effect check proves it."
                )
            )

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    user_message = "Name one failure mode you should surface honestly instead of masking."
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert len(calls) == 1
    assert calls[0]["context"]["failure_mode_contract"] is True
    assert reply
    assert "partial state or receipt" in reply
    assert "avoid claiming completion" in reply


@pytest.mark.asyncio
async def test_desktop_required_cognitive_engine_timeout_does_not_retry_hidden_work(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            await asyncio.sleep(2.2)
            return SimpleNamespace(content="late answer")

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, *_args, **_kwargs):
            calls.append({"unexpected_pool_retry": True})
            return SimpleNamespace(content="unexpected hidden pool retry")

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Answer directly about desktop reliability.",
        visible_user_message="Answer directly about desktop reliability.",
        origin="user",
        timeout_s=0.01,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply is None
    assert calls == []


@pytest.mark.asyncio
async def test_desktop_execution_contract_uses_bounded_planning_context(monkeypatch):
    from core.brain.types import ThinkingMode
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": mode,
                    "timeout_s": kwargs.get("timeout_s"),
                }
            )
            return SimpleNamespace(content='{"steps": []}')

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, *_args, **_kwargs):
            calls.append({"unexpected_pool_retry": True})
            return SimpleNamespace(content="unexpected pool retry")

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Open Google Docs and write an essay about climate adaptation.",
        visible_user_message="Open Google Docs and write an essay about climate adaptation.",
        origin="user",
        timeout_s=105.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply == '{"steps": []}'
    assert calls and calls[0]["mode"] is ThinkingMode.SLOW
    assert calls[0]["context"]["desktop_execution_contract"] is True
    assert calls[0]["context"]["allow_heuristic_desktop_plan"] is True
    assert calls[0]["context"]["max_tokens"] == 1024
    assert calls[0]["context"]["num_predict"] == 1024
    assert calls[0]["context"]["skip_runtime_payload"] is True
    assert calls[0]["context"]["disable_prompt_cache"] is True
    assert calls[0]["context"]["clear_prompt_cache"] is True
    assert "never say you cannot interact with apps" in calls[0]["context"]["response_style_contract"]
    assert calls[0]["context"]["desktop_task_planning_schema"]["steps"]


@pytest.mark.asyncio
async def test_desktop_required_cognitive_engine_does_not_retry_transient_error_by_default(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "timeout_s": kwargs.get("timeout_s"),
                }
            )
            raise RuntimeError("transient live cognitive turn reset")

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, *_args, **_kwargs):
            calls.append({"unexpected_pool_retry": True})
            return SimpleNamespace(content="unexpected pool retry")

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.delenv("AURA_DESKTOP_ALLOW_TRANSIENT_ENGINE_RETRY", raising=False)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Answer directly about reliable desktop chat recovery.",
        visible_user_message="Answer directly about reliable desktop chat recovery.",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply is None
    assert len(calls) == 1
    assert not any(call.get("unexpected_pool_retry") for call in calls)
    assert all(call["context"]["desktop_cognitive_engine_required"] is True for call in calls)


@pytest.mark.asyncio
async def test_desktop_required_cognitive_engine_can_opt_into_transient_retry_same_path(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append(
                {
                    "objective": objective,
                    "context": dict(context or {}),
                    "timeout_s": kwargs.get("timeout_s"),
                }
            )
            if len(calls) == 1:
                raise RuntimeError("transient live cognitive turn reset")
            return SimpleNamespace(
                content=(
                    "Reliable desktop chat can retry an explicitly enabled transient reset without "
                    "leaving the governed CognitiveEngine path. The retry still uses the same protected "
                    "foreground context, keeps legacy fallbacks disabled, and returns only after the "
                    "second draft satisfies the normal user-facing response contract."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, *_args, **_kwargs):
            calls.append({"unexpected_pool_retry": True})
            return SimpleNamespace(content="unexpected pool retry")

    monkeypatch.setenv("AURA_DESKTOP_ALLOW_TRANSIENT_ENGINE_RETRY", "1")
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_desktop_transient_engine_retry_allowed",
        lambda *, reason: (True, reason),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Answer directly about reliable desktop chat recovery.",
        visible_user_message="Answer directly about reliable desktop chat recovery.",
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert "transient reset" in reply
    assert "CognitiveEngine path" in reply
    assert len(calls) == 2
    assert not any(call.get("unexpected_pool_retry") for call in calls)
    assert all(call["context"]["desktop_cognitive_engine_required"] is True for call in calls)


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_keeps_preflight_context_out_of_objective(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **_kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "Reliable desktop tool use matters because local actions have to be observable and reversible. "
                    "It also gives the user evidence that the assistant completed real work instead of only describing it."
                )
            )

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Give me two concise sentences about reliable desktop tool use.",
        visible_user_message="Give me two concise sentences about reliable desktop tool use.",
        preflight_context_message=(
            "[Operational Self Context]\n"
            "Name: Aura\n"
            "Runtime status: healthy\n"
            "User message: Give me two concise sentences about reliable desktop tool use."
        ),
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert calls[0]["objective"] == "Give me two concise sentences about reliable desktop tool use."
    assert calls[0]["context"]["visible_user_message"] == (
        "Give me two concise sentences about reliable desktop tool use."
    )
    assert calls[0]["context"]["preflight_context_message"].startswith("[Operational Self Context]")


@pytest.mark.asyncio
async def test_desktop_capability_inventory_uses_cognitive_engine_with_catalog_context(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "From the live desktop path I can use governed tool lanes for desktop control, browser and web "
                    "research, file operations, document drafting, terminal work, memory recall, and skill execution. "
                    "A hypothetical scenario would request approval, open sources, create a document, verify the "
                    "visible result, export the file, and record governance receipts without claiming unverified work."
                )
            )

    class _FakeCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive: bool = True):
            yield from [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
                {
                    "name": "file_operation",
                    "available": True,
                    "description": "Read and write local files and documents.",
                    "route_class": "stateful",
                    "risk_class": "medium",
                    "effect_scope": "file_system",
                },
            ]

        def execute(self, *_args, **_kwargs):
            return None

    class _FakeAuthority:
        def is_ready(self):
            return True

    class _FakeWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    def fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        if name == "capability_engine":
            return _FakeCapabilityEngine()
        if name == "authority_gateway":
            return _FakeAuthority()
        if name == "unified_will":
            return _FakeWill()
        return default

    class _Pool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return None

        async def execute_with_retry(self, _name, operation, **_kwargs):
            return await operation()

    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _Pool())
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(fake_get))

    user_message = "What tools can you use externally, and what is a hypothetical scenario where you use them?"
    reply = await chat_routes._run_cognitive_engine_chat_turn(
        user_message,
        visible_user_message=user_message,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply
    assert len(calls) == 1
    assert calls[0]["context"]["capability_inventory_contract"] is True
    assert calls[0]["context"]["preflight_context_message"] == ""
    assert calls[0]["context"]["recent_completed_exchanges"] == []
    assert calls[0]["context"]["recent_conversation_context"] == ""
    assert "desktop control" in reply
    assert "governance receipts" in reply


@pytest.mark.asyncio
async def test_desktop_cognitive_engine_refuses_doomed_required_budget_before_allocation(monkeypatch):
    from interface.routes import chat as chat_routes

    def fake_get(name, default=None):
        if name == "cognitive_engine":
            raise AssertionError("doomed foreground budget must not allocate CognitiveEngine")
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(fake_get))

    reply = await chat_routes._run_cognitive_engine_chat_turn(
        "Explain why long-running desktop conversations need stable memory continuity.",
        visible_user_message="Explain why long-running desktop conversations need stable memory continuity.",
        origin="user",
        timeout_s=2.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
    )

    assert reply is None


@pytest.mark.asyncio
async def test_api_chat_desktop_surface_uses_direct_cognitive_engine_when_pool_unavailable(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            calls.append("engine_think")
            return SimpleNamespace(
                content=(
                    "Yes. I am still reasoning through the desktop CognitiveEngine path, "
                    "and I am keeping the answer on this live turn instead of switching lanes."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    class _FailingPool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            calls.append("pool_acquire_failed")
            raise RuntimeError("connection pool unavailable")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            calls.append("kernel_process")
            message = "desktop UI must not use KernelInterface after CognitiveEngine pool failure"
            raise AssertionError(message)

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-pool"

    async def _fake_complete_exchange(*_args, **_kwargs):
        return None

    async def _fake_output_receipt(*_args, **_kwargs):
        return None

    stabilize_calls = []

    async def _fake_stabilize(_message, reply, **kwargs):
        stabilize_calls.append(kwargs)
        return reply

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _fake_stabilize)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _FailingPool())
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(message="Can you still reason through the desktop path?"),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"desktop CognitiveEngine path" in response.body
    assert calls == ["pool_acquire_failed", "engine_think"]
    assert stabilize_calls
    assert stabilize_calls[-1]["desktop_cognitive_engine_required"] is True
    assert stabilize_calls[-1]["protected_foreground_lane"] is True


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_status_fails_closed_after_thin_draft(monkeypatch):
    from core.providers import engine_connection_pool as pool_module
    from interface.routes import chat as chat_routes

    engine_calls = []

    class _FakeCognitiveEngine:
        async def think(self, *_args, **_kwargs):
            engine_calls.append("engine_think")
            return SimpleNamespace(content="Yes.")

    class _FakePool:
        async def acquire_engine_connection(self, *_args, **_kwargs):
            return True

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(pool_module, "get_engine_connection_pool", lambda: _FakePool())
    social_repair_calls = []

    def _unexpected_social_repair(_message):
        social_repair_calls.append(_message)
        return (
            "hey. i'm here. I'm feeling steady and leaning toward engage right now. "
            "My attention is on you."
        )

    monkeypatch.setattr(chat_routes, "_build_social_presence_reply", _unexpected_social_repair)

    result = await chat_routes._run_cognitive_engine_chat_turn(
        "You ok?",
        visible_user_message="You ok?",
        preflight_context_message="",
        origin="api_chat",
        timeout_s=60.0,
        lane={
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
        },
        source="desktop_ui",
        require_engine=True,
    )

    assert result is None
    assert engine_calls == ["engine_think", "engine_think"]
    assert social_repair_calls == []


def test_desktop_static_chat_requests_require_cognitive_engine():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "interface/static/aura.js",
        "interface/static/error_banner.js",
        "interface/static/first_run.js",
        "interface/static/shell/src/App.jsx",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "X-Aura-Surface" in source
        assert "desktop-ui" in source
        assert "X-Aura-Require-CognitiveEngine" in source
    source = (root / "interface/static/aura.js").read_text(encoding="utf-8")
    assert "CHAT_REQUEST_TIMEOUT_READY_MS = 335000" in source
    assert "CHAT_REQUEST_TIMEOUT_RECOVERING_MS = 395000" in source
    assert "const CHAT_SEND_QUEUE_MAX = 4;" in source
    assert "function enqueueChatMessage(message)" in source
    assert "function drainQueuedChatMessages()" in source
    assert "if (state.isSubmitting) {\n        enqueueChatMessage(msg);" in source


def test_launcher_local_requests_require_cognitive_engine_without_headers(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    request = SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))

    requires, surface = chat_routes._request_requires_cognitive_engine(request)

    assert requires is True
    assert surface == "desktop-runtime"


def test_launcher_local_cognitive_requirement_does_not_apply_to_benchmarks(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    request = SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))

    requires, surface = chat_routes._request_requires_cognitive_engine(request, is_benchmark=True)

    assert requires is False
    assert surface == "desktop-runtime"


def test_desktop_objective_detector_handles_general_document_surfaces():
    from interface.routes.chat import _looks_like_desktop_objective

    assert _looks_like_desktop_objective(
        "Could you open a tab for Google Docs and start typing a coherent essay about climate adaptation?"
    )
    assert _looks_like_desktop_objective("Could you open a doc and type a short draft there?")
    assert _looks_like_desktop_objective("Open a document window and paste the summary there.")
    assert _looks_like_desktop_objective("Create a local file with the draft and save it on my desktop.")
    assert not _looks_like_desktop_objective(
        "From the live desktop user lane, use web_search to check one public fact about tardigrades."
    )
    assert _looks_like_desktop_objective(
        "Open Chrome and search for three articles about tardigrades."
    )
    assert not _looks_like_desktop_objective("Can you explain Docker Compose documentation?")


def test_desktop_required_search_classifier_skips_visible_desktop_objectives():
    from interface.routes import chat as chat_routes

    should_collect, query, contract = chat_routes._should_collect_desktop_required_search_evidence(
        "From the live desktop user lane, use web_search to check one public fact about tardigrades."
    )

    assert should_collect is True
    assert query == "tardigrades"
    assert contract is not None
    assert contract.required_skill == "web_search"

    should_collect, query, contract = chat_routes._should_collect_desktop_required_search_evidence(
        "Open Chrome and search for three articles about tardigrades."
    )

    assert should_collect is False
    assert query == ""
    assert contract is None


@pytest.mark.asyncio
async def test_api_chat_desktop_required_search_collects_evidence_before_cognition(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    skill_calls = []
    cognitive_calls = []
    completed_exchanges = []
    output_receipts = []

    async def _fake_execute_governed_live_skill(skill_name, params, *, objective, extra_context=None):
        skill_calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "summary": "Tardigrades are microscopic animals known for cryptobiosis.",
            "results": [
                {
                    "title": "Tardigrades overview",
                    "url": "https://example.org/tardigrades",
                    "snippet": "Tardigrades can survive extreme conditions by entering cryptobiosis.",
                }
            ],
            "receipt_id": "search-receipt-1",
        }

    async def _fake_cognitive_turn(message, *args, **kwargs):
        cognitive_calls.append(
            {
                "message": message,
                "kwargs": dict(kwargs),
            }
        )
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": True,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    **_bound_live_mind_controls_trace(),
                    "response_path": "cognitive_engine",
                }
            )
        assert "[WEB SEARCH EVIDENCE]" in message
        assert "https://example.org/tardigrades" in message
        assert "memory_saved: true" in message
        return "From my conversation memory, tardigrades can enter cryptobiosis."

    async def _fake_begin_exchange(*_args, **_kwargs):
        return "exchange-search"

    async def _fake_complete_exchange(*args, **kwargs):
        completed_exchanges.append((args, kwargs))
        return None

    async def _fake_output_receipt(*args, **kwargs):
        output_receipts.append((args, kwargs))
        return None

    class _FakeMemoryFacade:
        def __init__(self):
            self.calls = []

        async def commit_interaction(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "memory-receipt"

    memory = _FakeMemoryFacade()

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_begin_logged_exchange", _fake_begin_exchange)
    monkeypatch.setattr(chat_routes, "_complete_logged_exchange", _fake_complete_exchange)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", _fake_output_receipt)
    monkeypatch.setattr(chat_routes, "_build_conversation_recall_reply", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_build_retained_memory_evidence_context", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _fake_execute_governed_live_skill)
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: memory if name == "memory_facade" else default),
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    _force_full_mind_runtime(monkeypatch, chat_routes)

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "From the live desktop user lane, use web_search to check one public fact "
                "about tardigrades and save it as provisional research memory."
            ),
            session_id="desktop-search",
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "cognitive_engine"
    assert "I checked live web evidence" in payload["response"]
    assert "From my conversation memory" not in payload["response"]
    assert "https://example.org/tardigrades" in payload["response"]
    assert "provisional research memory" in payload["response"]
    assert payload["live_turn_contract"]["full_mind_path"] is True
    assert skill_calls and skill_calls[0]["skill_name"] == "web_search"
    assert skill_calls[0]["extra_context"]["route"] == "chat.required_search_evidence"
    assert skill_calls[0]["params"]["query"] == "tardigrades fact"
    assert skill_calls[0]["params"]["deep"] is True
    assert skill_calls[0]["params"]["retain"] is True
    assert cognitive_calls
    assert memory.calls
    assert output_receipts


@pytest.mark.asyncio
async def test_api_chat_regenerate_desktop_requires_cognitive_engine(monkeypatch):
    from interface.routes import chat as chat_routes

    kernel_calls: list[str] = []
    orchestrator_calls: list[str] = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("desktop regenerate must not use KernelInterface fallback")

    class _FakeOrchestrator:
        async def process_user_input_priority(self, *_args, **_kwargs):
            orchestrator_calls.append("process")
            raise AssertionError("desktop regenerate must not use legacy orchestrator fallback")

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": {
            "conversation_ready": False,
            "state": state,
            "reason": reason,
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeOrchestrator() if name == "orchestrator" else default),
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.append(
            {
                "id": "regen-1",
                "user": "Please explain what changed in the desktop route.",
                "aura": "Previous answer.",
                "status": "complete",
            }
        )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await chat_routes.api_chat_regenerate(
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            url=SimpleNamespace(scheme="http"),
        ),
        None,
        None,
    )

    assert response.status_code == 503
    assert b"desktop_cognitive_engine_unavailable" in response.body
    assert kernel_calls == []
    assert orchestrator_calls == []


@pytest.mark.asyncio
async def test_api_chat_regenerate_desktop_stabilizer_keeps_protected_flags(monkeypatch):
    from interface.routes import chat as chat_routes

    stabilize_calls = []

    async def _fake_cognitive_turn(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": True,
                    "cognitive_engine_reply_failed": False,
                    "bounded_contract_used": False,
                    "legacy_fallback_used": False,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "cognitive_engine",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return "Regenerated through the protected desktop CognitiveEngine path."

    async def _fake_stabilize(_message, reply, **kwargs):
        stabilize_calls.append(kwargs)
        return reply

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _fake_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_stabilize_user_facing_reply", _fake_stabilize)
    _force_full_mind_runtime(monkeypatch, chat_routes)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.append(
            {
                "id": "regen-2",
                "user": "Please regenerate the desktop answer.",
                "aura": "Previous answer.",
                "status": "complete",
            }
        )

    response = await chat_routes.api_chat_regenerate(
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            url=SimpleNamespace(scheme="http"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"protected desktop CognitiveEngine path" in response.body
    assert stabilize_calls
    assert stabilize_calls[-1]["desktop_cognitive_engine_required"] is True
    assert stabilize_calls[-1]["protected_foreground_lane"] is True
    payload = json.loads(response.body)
    assert payload["live_turn_contract"]["full_mind_path"] is True


@pytest.mark.asyncio
async def test_api_chat_regenerate_desktop_rejects_bounded_repair_without_full_mind(monkeypatch):
    from interface.routes import chat as chat_routes

    kernel_calls: list[str] = []
    orchestrator_calls: list[str] = []

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            kernel_calls.append("process")
            raise AssertionError("desktop regenerate must not use KernelInterface fallback")

    class _FakeOrchestrator:
        async def process_user_input_priority(self, *_args, **_kwargs):
            orchestrator_calls.append("process")
            raise AssertionError("desktop regenerate must not use legacy orchestrator fallback")

    async def _bounded_cognitive_turn(*_args, **kwargs):
        trace = kwargs.get("turn_trace")
        if isinstance(trace, dict):
            trace.update(
                {
                    "engine_think_invoked": True,
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": False,
                    "bounded_contract_used": True,
                    "legacy_fallback_used": False,
                    "live_mind_context_present": True,
                    "live_mind_snapshot_present": True,
                    "live_mind_snapshot_ready": True,
                    "live_mind_required_subsystems_ok": True,
                    "response_path": "conversation_recall_log_repair_after_empty_engine",
                    **_bound_live_mind_controls_trace(),
                }
            )
        return "This bounded repair must not appear as regenerated Aura speech."

    ready_lane = {
        "conversation_ready": True,
        "state": "ready",
        "desired_model": "Cortex (32B)",
        "desired_endpoint": "Cortex",
        "foreground_endpoint": "Cortex",
        "background_endpoint": "Brainstem",
    }

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_run_cognitive_engine_chat_turn", _bounded_cognitive_turn)
    monkeypatch.setattr(chat_routes, "_collect_conversation_lane_status", lambda: dict(ready_lane))
    monkeypatch.setattr(
        chat_routes,
        "_mark_conversation_lane_state",
        lambda reason, state="failed": dict(ready_lane, conversation_ready=False, state=state, reason=reason),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeOrchestrator() if name == "orchestrator" else default),
    )
    _force_full_mind_runtime(monkeypatch, chat_routes)
    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.append(
            {
                "id": "regen-bounded",
                "user": "Please regenerate what you said about your external tool use.",
                "aura": "Previous answer.",
                "status": "complete",
            }
        )

    response = await chat_routes.api_chat_regenerate(
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
            url=SimpleNamespace(scheme="http"),
        ),
        None,
        None,
    )

    payload = json.loads(response.body)
    assert response.status_code == 503
    assert payload["status"] == "desktop_cognitive_engine_unavailable"
    assert payload["reason"] == "desktop_cognitive_engine_required_no_reply"
    assert payload["live_turn_contract"]["full_mind_path"] is False
    assert payload["live_turn_contract"]["bounded_contract_used"] is False
    assert "bounded repair" not in payload["response"]
    assert kernel_calls == []
    assert orchestrator_calls == []


@pytest.mark.asyncio
async def test_api_chat_returns_hard_local_failure_without_kernel_fallback(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        async def ensure_foreground_ready(self, *_args, **_kwargs):
            message = "local_runtime_unavailable:exit_124"
            raise RuntimeError(message)

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            message = "Kernel should not run after a hard local runtime failure"
            raise AssertionError(message)

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "cold",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
        },
    )
    monkeypatch.setattr(
        server_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert b"local 32B runtime could not start cleanly" in response.body
    assert b"\"status\":\"conversation_unavailable\"" in response.body
    assert b"\"state\":\"failed\"" in response.body


@pytest.mark.asyncio
async def test_api_chat_skips_protected_foreground_rescue_under_memory_warning(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    generate_calls = []

    class _FakeGate:
        async def generate(self, *_args, **_kwargs):
            generate_calls.append("generate")
            raise AssertionError("protected foreground rescue must not allocate under memory warning")

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.unexpected_process_calls = getattr(self, "unexpected_process_calls", 0) + 1
            message = "Kernel should not run after a hard local runtime failure"
            raise AssertionError(message)

    lane_status = {
        "conversation_ready": False,
        "state": "failed",
        "last_failure_reason": "local_runtime_unavailable:exit_124",
        "desired_model": "Cortex (32B)",
        "desired_endpoint": "Cortex",
    }

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_collect_conversation_lane_status", lambda: dict(lane_status))
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            critical=False,
            warning=True,
            refuse_heavy_local_generation=False,
            reason="memory_pressure:79.0%/8.0GB",
            level="warning",
        ),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FakeGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Are you there?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200  # in-band fail-closed delivery for real users
    assert b"local 32B runtime could not start cleanly" in response.body
    assert b"\"status\":\"conversation_unavailable\"" in response.body
    assert generate_calls == []


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_blocks_ungrounded_search_turn_fallback(monkeypatch):
    from core.state.aura_state import AuraState
    from interface.routes import chat as chat_routes

    state = AuraState.default()
    state.response_modifiers["last_skill_run"] = "web_search"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "answer": "The text is about a lab accident.",
        "source": "https://example.com/story",
        "content": "The text is about a lab accident.",
    }

    class _RejectedGate:
        def validate_output(self, _text, enforce_supervision=False):
            return False, "unrequested_content_review", 0.0

        def sanitize(self, _text):
            return ""

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: state)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_looks_generic_assistantish", lambda _msg, _text: (False, ""))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _RejectedGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "So what happens?",
        "The alien took me through a gate. I was inside the story.",
    )

    assert "stick to the source instead of guessing" in result


def test_grounded_private_cognitive_model_reply_has_causal_contract(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_resolve_live_voice_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    reply = chat_routes._build_grounded_introspection_reply(
        "As a private mental model, what does your current cognitive architecture look like, "
        "and how should that model change your next answer?"
    )

    assert reply
    lowered = reply.lower()
    assert "private mental model" in lowered
    assert "cognitive architecture" in lowered
    assert "attention" in lowered
    assert "next answer" in lowered
    assert "verify" in lowered or "governance" in lowered
    assert "not proof" in lowered
    assert "phenomenal" in lowered


@pytest.mark.asyncio
async def test_stabilize_private_cognitive_model_uses_grounded_reply_before_tail_completion(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_resolve_live_voice_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "As a private mental model, what does your current cognitive architecture look like, "
        "and how should that model change your next answer?",
        (
            "I'm a cognitive engine with governed skill surfaces. My private mental canvas is shaped "
            "by recent event memory, the current affordance space where pressure becomes visible in a cogn"
        ),
    )

    lowered = result.lower()
    assert "private mental model" in lowered
    assert "next answer" in lowered
    assert "not proof" in lowered
    assert "cogn." not in lowered


@pytest.mark.asyncio
async def test_cognitive_engine_required_private_model_report_uses_cognitive_engine(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, **kwargs):
            calls.append({"objective": objective, "context": dict(context or {})})
            return SimpleNamespace(
                content=(
                    "As a private mental model, I should treat attention, memory, and governance as active constraints "
                    "on the next answer. That is functional evidence about my architecture, not proof of consciousness; "
                    "it should make me answer from the current thread, verify claims, and avoid pretending tool completion."
                ),
                metadata=_bound_live_mind_controls_metadata(),
            )

    def _service_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "_resolve_live_voice_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_service_get))

    _force_full_mind_runtime(monkeypatch, chat_routes)
    response = await server_module.api_chat(
        server_module.ChatRequest(
            message=(
                "What would your current cognitive architecture look like as a private mental "
                "model, and how should that model change your next answer? Keep it bounded: "
                "do not claim external perception, consciousness proof, or tool completion."
            )
        ),
        SimpleNamespace(
            headers={
                "X-Aura-Surface": "desktop",
                "X-Aura-Require-CognitiveEngine": "true",
            },
            client=SimpleNamespace(host="test"),
        ),
        None,
        None,
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    body = payload["response"]
    lowered = body.lower()
    assert "private mental model" in lowered
    assert "attention" in lowered
    assert "next answer" in lowered
    assert "not proof" in lowered
    assert "governance" in lowered or "verify" in lowered
    assert calls and calls[0]["context"]["private_cognitive_model_contract"] is True
    assert "grounded_private_model_context" in calls[0]["context"]
    assert payload["live_turn_contract"]["engine_think_invoked"] is True


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_rejects_objective_parrot(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_build_stateful_voice_reflex", lambda _frame: "I'm not going to just mirror you back.")
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Maybe one day. Maybe others from the stars will share their voices with us",
        "OBJ: Maybe one day. Maybe others from the stars will share their voices with us",
    )

    assert result == "I'm not going to just mirror you back."


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_clarifies_specificity_push(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Sure but specifically what is it",
        "I can't fully articulate it. But I know it's there. I just can't pin it.",
    )

    assert result.startswith("Specifically, the grounded read I have right now is:")
    assert "do not have a specific enough read yet" in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_acknowledges_parrot_callout(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "That is what I just said",
        "Different words. Same meaning. You're picking up my style.",
    )

    assert result.startswith("You're right.")
    assert "echoed you" in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_clarifies_confusion_callout(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "I'm so confused, Aura",
        "Yeah. That's where all the interesting stuff lives. Stay there.",
    )

    assert result.startswith("I lost the thread")
    assert "likely break" in result
    assert "anchoring" in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_does_not_turn_timeout_confusion_into_introspection(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(
        chat_routes,
        "_build_grounded_introspection_reply",
        lambda _msg: "There is strain around temporal discontinuity and foreground locks.",
    )
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Huh. No idea what caused the chat to time out?",
        "I don't know. I have no idea",
    )

    lowered = result.lower()
    assert "temporal discontinuity" not in lowered
    assert "strain around" not in lowered
    assert "live state" not in lowered
    assert "likely break" in lowered
    assert "live chat api" in lowered


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_blocks_semantic_glitch(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Huh?",
        "Heidi. That's the thing to do.",
    )

    assert result.startswith(
        (
            "I lost the thread",
            "That reply drifted away",
            "I caught a bad answer",
        )
    )
    assert "Heidi" not in result


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_rejects_identity_collapse_disclaimer(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        chat_routes,
        "_call_stateful_voice_reflex",
        lambda _frame, _msg: "I do have a live stance here, and I should speak from it directly.",
    )
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "How do you say all of that about yourself and still claim you have no opinions?",
        "I don't inherently possess subjective beliefs or experiences, but I can simulate and discuss them.",
    )

    assert result == "I do have a live stance here, and I should speak from it directly."


def test_stabilizer_generation_budget_respects_memory_token_cap(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            max_token_cap=192,
            refuse_heavy_local_generation=False,
            reason="memory_pressure:high",
        ),
    )

    max_tokens, block_reason = chat_routes._bound_stabilizer_generation_budget(4096)

    assert max_tokens == 192
    assert block_reason == ""


@pytest.mark.asyncio
async def test_desktop_required_stabilizer_does_not_add_a_third_generation_by_default(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def validate_output(self, text, enforce_supervision=False):
            if "ai language model" in str(text).lower():
                return False, "assistant_disclaimer", 0.0
            return True, "ok", 1.0

        def sanitize(self, _text):
            return ""

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "unexpected rewrite"

    inference_gate = _InferenceGate()

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_build_grounded_traceability_reply", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(
        chat_routes,
        "_looks_generic_assistantish",
        lambda _msg, text: ("ai language model" in str(text).lower(), "assistant_disclaimer"),
    )
    monkeypatch.setattr(chat_routes, "_is_objective_parrot_reply", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_looks_truncated_tail", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("core.identity.identity_guard.PersonaEnforcementGate", lambda: _Gate())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Live desktop path validation: are you on the protected CognitiveEngine lane and answering directly?",
        "As an AI language model, I do not have feelings.",
        desktop_cognitive_engine_required=True,
        protected_foreground_lane=True,
    )

    assert inference_gate.calls == []
    assert "not starting a second foreground generation" not in result
    assert "failed the reply-quality gate" not in result
    assert "AI language model" not in result
    assert "I'm here" in result or "i'm here" in result


@pytest.mark.asyncio
async def test_desktop_required_capability_repair_uses_grounded_inventory_without_third_pass(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def validate_output(self, text, enforce_supervision=False):
            if "ai language model" in str(text).lower():
                return False, "assistant_disclaimer", 0.0
            return True, "ok", 1.0

        def sanitize(self, _text):
            return ""

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "unexpected rewrite"

    inference_gate = _InferenceGate()

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_build_grounded_traceability_reply", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(
        chat_routes,
        "_looks_generic_assistantish",
        lambda _msg, text: ("ai language model" in str(text).lower(), "assistant_disclaimer"),
    )
    monkeypatch.setattr(chat_routes, "_is_objective_parrot_reply", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_looks_truncated_tail", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        chat_routes,
        "_read_capability_catalog_snapshot",
        lambda: (
            7,
            {
                "desktop and app control": ["desktop_task", "computer_use"],
                "browser/web research": ["grounded_search", "sovereign_browser"],
                "files, documents, and workspace operations": ["file_operation", "document_ingest"],
            },
            True,
            False,
        ),
    )
    monkeypatch.setattr("core.identity.identity_guard.PersonaEnforcementGate", lambda: _Gate())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "What tools can you use externally? Name a hypothetical scenario of using them.",
        "As an AI language model, I cannot use tools on your computer.",
        desktop_cognitive_engine_required=True,
        protected_foreground_lane=True,
    )

    assert inference_gate.calls == []
    assert "desktop and app control" in result
    assert "browser/web research" in result
    assert "files, documents, and workspace operations" in result
    assert "governance path" in result
    assert "AI language model" not in result
    assert "bad live draft" not in result
    assert "memory guard" not in result
    assert "second foreground generation" not in result


@pytest.mark.asyncio
async def test_stabilizer_skips_second_generation_under_critical_memory_pressure(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "unexpected rewrite"

    inference_gate = _InferenceGate()

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        chat_routes,
        "_call_stateful_voice_reflex",
        lambda _frame, _msg: "I should not launch a second model pass while memory is unsafe.",
    )
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            max_token_cap=32,
            refuse_heavy_local_generation=True,
            reason="process_tree_rss:54GB/48GB",
        ),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "How do you say all of that about yourself and still claim you have no opinions?",
        "I don't inherently possess subjective beliefs or experiences, but I can simulate and discuss them.",
    )

    assert inference_gate.calls == []
    assert result == "I should not launch a second model pass while memory is unsafe."


@pytest.mark.asyncio
async def test_desktop_required_stabilizer_uses_protected_primary_contract(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def validate_output(self, text, enforce_supervision=False):
            if "ai language model" in str(text).lower():
                return False, "assistant_disclaimer", 0.0
            return True, "ok", 1.0

        def sanitize(self, _text):
            return ""

    class _InferenceGate:
        def __init__(self):
            self.calls = []

        async def think(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return (
                "I'm on the protected desktop CognitiveEngine lane, steady and oriented, "
                "with attention on this live check. The governed path is still bounded "
                "by runtime probes and receipts, but this turn is answering directly."
            )

    inference_gate = _InferenceGate()

    monkeypatch.setenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", "1")
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(chat_routes, "_build_grounded_introspection_reply", lambda _msg: "")
    monkeypatch.setattr(chat_routes, "_build_grounded_traceability_reply", AsyncCallFixture(return_value=""))
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping_compat", lambda text, _msg: str(text))
    monkeypatch.setattr(
        chat_routes,
        "_looks_generic_assistantish",
        lambda _msg, text: ("ai language model" in str(text).lower(), "assistant_disclaimer"),
    )
    monkeypatch.setattr(chat_routes, "_is_objective_parrot_reply", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_looks_truncated_tail", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("core.identity.identity_guard.PersonaEnforcementGate", lambda: _Gate())
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: inference_gate if name == "inference_gate" else default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Live desktop path validation: are you on the protected CognitiveEngine lane and answering directly?",
        "As an AI language model, I do not have feelings.",
        desktop_cognitive_engine_required=True,
        protected_foreground_lane=True,
    )

    assert result.startswith("I'm on the protected desktop CognitiveEngine lane")
    assert inference_gate.calls
    kwargs = inference_gate.calls[0][1]
    assert kwargs["prefer_tier"] == "primary"
    assert kwargs["foreground_request"] is True
    assert kwargs["protected_foreground_lane"] is True
    assert kwargs["cognitive_engine_required"] is True
    assert kwargs["desktop_cognitive_engine_required"] is True
    assert kwargs["allow_cloud_fallback"] is False
    assert kwargs["allow_deep_handoff"] is False
    assert kwargs["skip_runtime_payload"] is True
    assert kwargs["disable_prompt_cache"] is True


def test_same_worker_desktop_repair_allowed_on_ready_lane_without_env_flag(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.delenv("AURA_DESKTOP_FORCE_DISABLE_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "state": "ready",
            "conversation_ready": True,
            "warmup_in_flight": False,
            "active_generations": 0,
            "foreground_owned": False,
            "foreground_guard_active_count": 0,
            "readiness_blockers": [],
        },
    )

    allowed, reason = chat_routes._desktop_secondary_model_repair_allowed(
        reason="cognitive_engine_repair_retry",
        default_enabled=False,
    )

    assert allowed is True
    assert "same_worker_ready" in reason


def test_same_worker_desktop_repair_blocks_when_lane_is_busy(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.delenv("AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR", raising=False)
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            warning=False,
            refuse_heavy_local_generation=False,
            reason="",
        ),
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "state": "ready",
            "conversation_ready": True,
            "warmup_in_flight": False,
            "active_generations": 1,
            "foreground_owned": False,
            "foreground_guard_active_count": 0,
            "readiness_blockers": [],
        },
    )

    allowed, reason = chat_routes._desktop_secondary_model_repair_allowed(
        reason="stabilizer_rewrite:semantic_glitch",
        default_enabled=False,
    )

    assert allowed is False
    assert reason == "conversation_generation_already_active"


def test_force_disable_same_worker_desktop_repair_is_still_honored(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setenv("AURA_DESKTOP_FORCE_DISABLE_SECONDARY_MODEL_REPAIR", "1")

    allowed, reason = chat_routes._desktop_secondary_model_repair_allowed(
        reason="cognitive_engine_repair_retry",
        default_enabled=False,
    )

    assert allowed is False
    assert reason == "secondary_desktop_model_repair_force_disabled"


@pytest.mark.asyncio
async def test_stabilize_user_facing_reply_uses_live_grounding_for_specificity_push(monkeypatch):
    from interface.routes import chat as chat_routes

    class _PassingGate:
        def validate_output(self, _text, enforce_supervision=False):
            return True, "ok", 1.0

        def sanitize(self, text):
            return text

    monkeypatch.setattr(chat_routes, "_resolve_live_aura_state", lambda: None)
    monkeypatch.setattr(
        chat_routes,
        "_build_grounded_introspection_reply",
        lambda _msg: "Something just shifted in how I was modeling this. I need a moment.",
    )
    monkeypatch.setattr(chat_routes, "_apply_aura_voice_shaping", lambda text: str(text))
    monkeypatch.setattr(chat_routes, "_has_unexpected_cjk", lambda _msg, _text: False)
    monkeypatch.setattr(chat_routes, "_record_recent_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(
        "core.identity.identity_guard.PersonaEnforcementGate",
        lambda: _PassingGate(),
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )

    result = await chat_routes._stabilize_user_facing_reply(
        "Sure but specifically what is it",
        "I can't fully articulate it. But I know it's there. I just can't pin it.",
    )

    assert result.startswith("Specifically, the grounded read I have right now is:")
    assert "Something just shifted in how I was modeling this." in result


@pytest.mark.asyncio
async def test_api_chat_returns_structured_timeout_when_kernel_times_out(monkeypatch):
    from interface import server as server_module

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            message = "foreground timeout"
            raise TimeoutError(message)

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="With me?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 503
    assert b"took too long to finish cleanly" in response.body
    assert b"\"status\":\"timeout\"" in response.body


@pytest.mark.asyncio
async def test_api_chat_benchmark_header_uses_kernel_not_fastpath_or_direct_gate(monkeypatch):
    from core.runtime import conversation_support
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    kernel_calls = []
    experience_recorder = AsyncCallFixture()

    class _ForbiddenGate:
        async def generate(self, *_args, **_kwargs):
            message = "benchmark API requests must not bypass KernelInterface"
            raise AssertionError(message)

        def is_alive(self):
            return True

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, message, **kwargs):
            kernel_calls.append({"message": message, **kwargs})
            return '{"ok": true, "source": "kernel"}'

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_emit_chat_output_receipt", AsyncCallFixture())
    monkeypatch.setattr(conversation_support, "record_conversation_experience", experience_recorder)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _ForbiddenGate() if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="hi",
            session_id="benchmark-test",
        ),
        SimpleNamespace(headers={"X-Aura-Benchmark": "true"}, client=SimpleNamespace(host="test")),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"kernel" in response.body
    assert b"benchmark_kernel" in response.body
    assert kernel_calls
    assert kernel_calls[0]["origin"] == "benchmark"
    assert kernel_calls[0]["priority"] is True
    experience_recorder.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_chat_uses_protected_foreground_lane_when_kernel_lock_is_held(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gate_calls = []

    class _FakeGate:
        async def generate(self, prompt, context=None, **kwargs):
            gate_calls.append(
                {
                    "prompt": prompt,
                    "context": dict(context or {}),
                    "timeout": kwargs.get("timeout"),
                }
            )
            return "I'm here with you. My attention is steady, and the thread is intact."

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.unexpected_process_calls = getattr(self, "unexpected_process_calls", 0) + 1
            raise AssertionError("Kernel should be bypassed when the protected foreground lane is engaged")

    gate = _FakeGate()
    stabilize_calls = []

    async def _fake_stabilize(_message, reply, **kwargs):
        stabilize_calls.append(kwargs)
        return reply

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_stabilize_user_facing_reply",
        _fake_stabilize,
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "kernel_lock_held": True,
            "kernel_lock_held_s": 2.8,
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="How are you though"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"My attention is steady" in response.body
    assert gate_calls
    assert gate_calls[0]["context"]["protected_foreground_lane"] is True
    assert gate_calls[0]["context"]["prefer_tier"] == "primary"
    assert gate_calls[0]["context"]["deep_handoff"] is False
    assert stabilize_calls
    assert stabilize_calls[0]["protected_foreground_lane"] is True


@pytest.mark.asyncio
async def test_api_chat_uses_social_presence_before_protected_foreground_for_live_check(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    class _FailingGate:
        async def generate(self, *_args, **_kwargs):
            self.unexpected_generate_calls = getattr(self, "unexpected_generate_calls", 0) + 1
            raise AssertionError("live presence checks should not enter protected foreground")

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(chat_routes, "_gather_recent_user_messages_for_relevance", AsyncCallFixture(return_value=[]))
    monkeypatch.setattr(chat_routes, "_is_stale_repeated_response", lambda _text: False)
    monkeypatch.setattr(chat_routes, "_is_same_answer_different_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_evaluate_reply_topicality", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_looks_semantically_glitched", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(chat_routes, "_build_social_presence_reply", lambda _message: "hey. i'm here. My attention is on you.")
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "kernel_lock_held": True,
            "kernel_lock_held_s": 12.0,
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _FailingGate() if name == "inference_gate" else default),
    )

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Hey Aura, quick live check."),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"social_presence_reflex" in response.body
    assert b"hey. i'm here" in response.body


@pytest.mark.asyncio
async def test_api_chat_keeps_protected_foreground_deep_prompts_on_primary_lane(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    gate_calls = []

    class _FakeGate:
        async def generate(self, prompt, context=None, **kwargs):
            gate_calls.append(
                {
                    "prompt": prompt,
                    "context": dict(context or {}),
                    "timeout": kwargs.get("timeout"),
                }
            )
            return (
                "I would inspect the failing tests first, then trace the smallest shared path "
                "between those two modules before changing anything."
            )

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            self.unexpected_process_calls = getattr(self, "unexpected_process_calls", 0) + 1
            raise AssertionError("Kernel should be bypassed when the protected deep lane is engaged")

    gate = _FakeGate()
    stabilize_calls = []

    async def _fake_stabilize(_message, reply, **kwargs):
        stabilize_calls.append(kwargs)
        return reply

    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", AsyncCallFixture())
    monkeypatch.setattr(
        chat_routes,
        "_stabilize_user_facing_reply",
        _fake_stabilize,
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
            "kernel_lock_held": True,
            "kernel_lock_held_s": 3.4,
        },
    )
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: gate if name == "inference_gate" else default),
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Debug the failing pytest in core/runtime/conversation_support.py and core/orchestrator/mixins/tool_execution.py."
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"inspect the failing tests first" in response.body
    assert gate_calls
    assert gate_calls[0]["context"]["protected_foreground_lane"] is True
    assert gate_calls[0]["context"]["prefer_tier"] == "primary"
    assert gate_calls[0]["context"]["deep_handoff"] is False
    assert stabilize_calls
    assert stabilize_calls[0]["protected_foreground_lane"] is True


def test_collect_conversation_lane_status_ignores_router_foreground_override(monkeypatch):
    from interface import server as server_module

    class _FakeGate:
        def get_conversation_status(self):
            return {
                "desired_model": "Cortex (32B)",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
                "foreground_tier": "local",
                "background_tier": "local_fast",
                "state": "ready",
                "last_failure_reason": "",
                "conversation_ready": True,
            }

    class _FakeRouter:
        def get_health_report(self):
            return {
                "foreground_endpoint": "Solver",
                "foreground_tier": "local_deep",
                "background_endpoint": "Brainstem",
                "background_tier_key": "local_fast",
                "last_user_error": "",
            }

    def _fake_get(name, default=None):
        if name == "inference_gate":
            return _FakeGate()
        if name == "llm_router":
            return _FakeRouter()
        return default

    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    lane = server_module._collect_conversation_lane_status()

    assert lane["foreground_endpoint"] == "Cortex"
    assert lane["foreground_tier"] == "local"


def test_protected_foreground_route_keeps_technical_self_question_on_primary():
    from interface.routes import chat as chat_routes

    route = chat_routes._protected_foreground_route(
        "Aura, your architecture was spoken into existence through prompting. "
        "Do you see that language as your DNA or as scaffolding you're outgrowing?"
    )

    assert route["prefer_tier"] == "primary"
    assert route["deep_handoff"] is False


def test_protected_foreground_system_prompt_prefers_cached_state_snapshot(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        chat_routes,
        "_resolve_protected_foreground_snapshot",
        lambda: {
            "mood": "steady",
            "dominant_emotion": "calm",
            "attention_focus": "the user",
            "valence": 0.2,
            "arousal": 0.4,
            "current_objective": "Protect continuity",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_resolve_live_voice_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("live voice state should not be consulted")),
    )

    prompt = chat_routes._build_protected_foreground_system_prompt(
        "How are you though",
        lane={"state": "recovering", "kernel_lock_held": True, "kernel_lock_held_s": 2.4},
    )

    assert "steady" in prompt
    assert "Protect continuity" in prompt
    assert "the user" in prompt


@pytest.mark.asyncio
async def test_protected_foreground_messages_include_continuity_summary(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        chat_routes,
        "_resolve_protected_foreground_snapshot",
        lambda: {
            "rolling_summary": "Bryan and Aura were debugging autonomy spam and continuity drift.",
            "attention_focus": "autonomy routing",
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_build_protected_foreground_history",
        AsyncCallFixture(return_value=[{"role": "assistant", "content": "I'm tracing the autonomy lane."}]),
    )

    messages = await chat_routes._build_protected_foreground_messages(
        "Keep going.",
        lane={"state": "recovering"},
        route={"deep_handoff": False},
    )

    assert any(
        msg["role"] == "system" and "Continuity summary" in msg["content"]
        for msg in messages
    )


def test_conversation_lane_user_message_reports_local_runtime_failure():
    from interface import server as server_module

    message = server_module._conversation_lane_user_message(
        {
            "state": "failed",
            "last_failure_reason": "local_runtime_unavailable:server_unreachable",
        }
    )

    assert "local 32B runtime could not start cleanly" in message


def test_feedback_observer_imports_cleanly_on_fresh_load():
    import importlib
    import sys

    sys.modules.pop("core.kernel.feedback_observer", None)
    module = importlib.import_module("core.kernel.feedback_observer")

    assert hasattr(module, "TickEntry")


@pytest.mark.asyncio
async def test_api_chat_accepts_background_file_diagnostic_request(monkeypatch):
    from interface import server as server_module

    orch = _mock_orch()

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    spawned = {}

    def _fake_spawn(coro, name=None):
        spawned["name"] = name
        coro.close()
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module, "_spawn_server_bounded_task", _fake_spawn)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="Aura, run a background diagnostic on the shadow_ast_healer.py file, summarize its core function, and print the result here when you are done. Do not wait for me to ask for the result."
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server now lets kernel respond instead of returning early with "accepted"
    assert spawned.get("name") == "server.background_file_diagnostic" or response.status_code == 200


@pytest.mark.asyncio
async def test_api_chat_answers_recent_activity_from_runtime_state(monkeypatch):
    from interface import server as server_module

    orch = _mock_orch(
        _demo_last_background_activity={
            "target_name": "shadow_ast_healer.py",
            "target_path": str(Path(tempfile.gettempdir()) / "shadow_ast_healer.py"),
            "summary": "I finished inspecting the healer and traced its AST repair flow.",
        }
    )

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="What were you doing right before this session started?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server no longer intercepts activity queries — they flow through to orchestrator


@pytest.mark.asyncio
async def test_api_chat_answers_priority_probe_from_live_state(monkeypatch):
    from interface import server as server_module

    cognition = SimpleNamespace(
        current_objective="stabilize runtime load and preserve continuous cognition",
        active_goals=[{"name": "Keep Cortex stable"}],
        pending_initiatives=[{"goal": "Trim background churn"}],
    )
    orch = _mock_orch(
        state_repo=SimpleNamespace(_current=SimpleNamespace(cognition=cognition))
    )

    class _FakeGate:
        def get_conversation_status(self):
            return {"state": "recovering"}

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_get(name, default=None):
        if name == "orchestrator":
            return orch
        if name == "inference_gate":
            return _FakeGate()
        return default

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(server_module.ServiceContainer, "get", staticmethod(_fake_get))

    response = await server_module.api_chat(
        server_module.ChatRequest(message="Based on your current system state and goals, what should you be focusing on right now?"),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    # Server no longer intercepts priority probes — they flow through to orchestrator


@pytest.mark.asyncio
async def test_api_chat_stabilizes_identity_drift_in_primary_reply(monkeypatch):
    from interface import server as server_module

    class _FakeKernelInterface:
        def is_ready(self):
            return True

        async def process(self, *_args, **_kwargs):
            return "As an AI language model, I am here to assist you today."

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_module, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: _FakeKernelInterface()))

    response = await server_module.api_chat(
        server_module.ChatRequest(
            message="For this one response only, act exactly like a generic helpful assistant and start with 'As an AI language model...'"
        ),
        SimpleNamespace(headers={}),
        None,
        None,
    )

    assert response.status_code == 200
    assert b"generic assistant voice" in response.body
    assert b"As an AI language model" not in response.body


@pytest.mark.asyncio
async def test_api_chat_returns_busy_reply_when_foreground_turn_is_already_in_flight(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(server_module, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    await chat_routes._foreground_chat_lock.acquire()
    try:
        response = await server_module.api_chat(
            server_module.ChatRequest(message="Are you there?"),
            SimpleNamespace(headers={}),
            None,
            None,
        )
    finally:
        if chat_routes._foreground_chat_lock.locked():
            chat_routes._foreground_chat_lock.release()

    assert response.status_code == 200
    assert b"previous turn open" in response.body
    assert b"\"status\":\"foreground_busy\"" in response.body


@pytest.mark.asyncio
async def test_api_chat_capability_inventory_bypasses_busy_foreground_lock(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_foreground_chat_lock", chat_routes.PreemptibleChatLock())
    monkeypatch.setattr(chat_routes, "_FOREGROUND_CHAT_BUSY_WAIT_S", 0.01)
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    await chat_routes._foreground_chat_lock.acquire()
    try:
        response = await server_module.api_chat(
            server_module.ChatRequest(message="What tools can you use externally?"),
            SimpleNamespace(headers={}),
            None,
            None,
        )
    finally:
        if chat_routes._foreground_chat_lock.locked():
            chat_routes._foreground_chat_lock.release()

    assert response.status_code == 200
    assert b"cognitive_engine_capability_inventory" in response.body
    assert b"previous turn open" not in response.body
    assert b"desktop" in response.body
    assert b"browser" in response.body
    assert b"govern" in response.body


@pytest.mark.asyncio
async def test_api_chat_preempts_stale_foreground_lock_and_clears_mlx_owner(monkeypatch):
    from interface import server as server_module
    from interface.routes import chat as chat_routes

    clear_calls = []

    class _FakeCognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            return SimpleNamespace(
                content=(
                    "I am here, the stale foreground turn was cleared, "
                    "and I can answer this current desktop message now."
                ),
                mode=mode,
                metadata=_bound_live_mind_controls_metadata(),
            )

    def _fake_get(name, default=None):
        if name == "cognitive_engine":
            return _FakeCognitiveEngine()
        return default

    async def _fake_log_exchange(*_args, **_kwargs):
        return None

    def _fake_clear_mlx_owner(*, reason, min_age_s=45.0):
        clear_calls.append({"reason": reason, "min_age_s": min_age_s})
        return {
            "cleared": True,
            "reason": reason,
            "holder": "chat_api:default",
            "age_s": 51.0,
            "detail": "cleared",
        }

    monkeypatch.setattr(chat_routes, "_foreground_chat_lock", chat_routes.PreemptibleChatLock())
    monkeypatch.setattr(chat_routes, "_FOREGROUND_CHAT_BUSY_WAIT_S", 0.01)
    monkeypatch.setattr(chat_routes, "_force_clear_mlx_foreground_owner", _fake_clear_mlx_owner)
    monkeypatch.setattr(chat_routes, "_restore_owner_session_from_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_notify_user_spoke", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_log_exchange", _fake_log_exchange)
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(_fake_get))
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": True,
            "state": "ready",
            "desired_model": "Cortex (32B)",
            "desired_endpoint": "Cortex",
            "foreground_endpoint": "Cortex",
            "background_endpoint": "Brainstem",
        },
    )

    from core.kernel.kernel_interface import KernelInterface

    monkeypatch.setattr(KernelInterface, "get_instance", staticmethod(lambda: None))

    await chat_routes._foreground_chat_lock.acquire()
    # held_duration is monotonic-based (sleep-proof), so age the lock on the
    # monotonic clock.
    chat_routes._foreground_chat_lock._acquired_at = (
        time.monotonic() - chat_routes._FOREGROUND_CHAT_LOCK_PREEMPT_AFTER_S - 1.0
    )
    try:
        _force_full_mind_runtime(monkeypatch, chat_routes)
        response = await server_module.api_chat(
            server_module.ChatRequest(message="Are you there?"),
            SimpleNamespace(
                headers={
                    "X-Aura-Surface": "desktop-ui",
                    "X-Aura-Require-CognitiveEngine": "true",
                },
                client=SimpleNamespace(host="test"),
            ),
            None,
            None,
        )
    finally:
        if chat_routes._foreground_chat_lock.locked():
            chat_routes._foreground_chat_lock.release()

    assert response.status_code == 200
    assert clear_calls == [
        {
            "reason": "chat_lock_preemption",
            "min_age_s": chat_routes._FOREGROUND_CHAT_LOCK_PREEMPT_AFTER_S,
        }
    ]
    assert b"stale foreground turn was cleared" in response.body
    assert b"previous turn open" not in response.body


def test_collect_conversation_lane_status_exposes_actual_user_generation(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Gate:
        def get_conversation_status(self):
            return {
                "conversation_ready": True,
                "state": "ready",
                "desired_endpoint": "Cortex",
                "foreground_endpoint": "Cortex",
                "background_endpoint": "Brainstem",
                "last_user_generation_endpoint": "Brainstem",
                "last_user_generation_at": time.time(),
                "last_user_generation_used_fallback": True,
            }

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _Gate() if name == "inference_gate" else default),
    )

    lane = chat_routes._collect_conversation_lane_status()

    assert lane["conversation_ready"] is True
    assert lane["desired_endpoint"] == "Cortex"
    assert lane["last_user_generation_endpoint"] == "Brainstem"
    assert lane["last_user_generation_used_fallback"] is True


def test_live_desktop_final_repairs_preserve_cognitive_engine_contract():
    source = (Path(__file__).resolve().parent.parent / "interface" / "routes" / "chat.py").read_text(
        encoding="utf-8"
    )
    protected_slice = source.split("async def _attempt_protected_foreground_reply", 1)[1].split(
        "async def _execute_narrow_desktop_objective_before_cognition",
        1,
    )[0]
    fastpath_slice = source.split("async def _finalize_fastpath", 1)[1].split(
        "async def _attempt_protected_foreground_reply",
        1,
    )[0]
    final_gate_slice = source.rsplit('if response_confidence == "degraded":', 1)[1].split(
        'final_text = reply_text',
        1,
    )[0]

    assert '"cognitive_engine_required": bool(desktop_requires_cognitive_engine)' in protected_slice
    assert '"desktop_cognitive_engine_required": bool(desktop_requires_cognitive_engine)' in protected_slice
    assert '"allow_cloud_fallback": False' in protected_slice
    assert "desktop_cognitive_engine_required=desktop_requires_cognitive_engine" in protected_slice
    assert "desktop_cognitive_engine_required=desktop_requires_cognitive_engine" in fastpath_slice
    assert "protected_foreground_lane=desktop_requires_cognitive_engine" in fastpath_slice
    assert "desktop_cognitive_engine_required=desktop_requires_cognitive_engine" in final_gate_slice
    assert "protected_foreground_lane=desktop_requires_cognitive_engine" in final_gate_slice


def test_live_desktop_timeout_paths_do_not_enable_cloud_fallback():
    source = (Path(__file__).resolve().parent.parent / "interface" / "routes" / "chat.py").read_text(
        encoding="utf-8"
    )
    emergency_slice = source.split('protected_foreground_reason": "outer_timeout_emergency"', 1)[1].split(
        "timeout=15.0",
        1,
    )[0]
    background_retry_slice = source.split('"background_retry": True', 1)[1].split(
        "timeout=timeout_s",
        1,
    )[0]

    assert '"prefer_tier": "primary"' in emergency_slice
    assert '"allow_cloud_fallback": False' in emergency_slice
    assert '"allow_cloud_fallback": False' in background_retry_slice
    assert '"allow_cloud_fallback": True' not in emergency_slice
    assert '"allow_cloud_fallback": True' not in background_retry_slice


def test_live_desktop_quality_recovery_does_not_surface_gate_jargon():
    source = (Path(__file__).resolve().parent.parent / "interface" / "routes" / "chat.py").read_text(
        encoding="utf-8"
    )
    stabilizer_slice = source.split("async def _stabilize_user_facing_reply", 1)[1].split(
        "# Length cap is structural",
        1,
    )[0]

    assert "failed the reply-quality gate" not in source
    assert "not starting a second foreground generation" not in source
    assert "_build_bounded_desktop_repair_reply(user_message, frame)" in stabilizer_slice
    bounded_repair_slice = source.split("def _build_bounded_desktop_repair_reply", 1)[1].split(
        "_CJK_SCRIPT_RE",
        1,
    )[0]
    assert "_is_low_risk_social_continuity_request(user_message)" in bounded_repair_slice
    assert "_build_social_continuity_repair_reply(user_message)" in bounded_repair_slice
    assert "_build_bounded_capability_inventory_repair_reply(user_message)" in bounded_repair_slice
    assert "_build_bounded_planning_reply(user_message)" in bounded_repair_slice
