from core.state.aura_state import AuraState
from pathlib import Path


def test_background_policy_blocks_during_foreground_guard(monkeypatch):
    from core.runtime import foreground_guard
    from core.runtime.background_policy import background_activity_reason

    monkeypatch.setenv("AURA_FOREGROUND_QUIET_WINDOW_S", "0.25")
    foreground_guard._reset_for_tests()

    lease = foreground_guard.begin_foreground_turn(owner="test-chat", source="chat_api")
    try:
        assert (
            background_activity_reason(None, allow_no_user_anchor=True)
            == "foreground_chat_active"
        )
    finally:
        lease.close()

    assert (
        background_activity_reason(None, allow_no_user_anchor=True)
        == "foreground_quiet_window"
    )
    foreground_guard._reset_for_tests()


def test_live_chat_context_drops_background_tool_noise():
    from core.brain.llm.context_assembler import ContextAssembler

    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "user", "content": "Hey, Aura", "origin": "api"},
        {
            "role": "system",
            "content": "Read top r/technology thread: Princeton scraps honor code",
            "metadata": {
                "type": "skill_result",
                "skill": "reddit_adapter",
                "source": "skills.reddit_adapter",
            },
        },
        {
            "role": "assistant",
            "content": "Browsing r/technology. Found an interesting thread.",
            "metadata": {"source": "mind_tick", "autonomous": True},
        },
        {
            "role": "system",
            "content": "Cognitive baseline tick 540: monitoring internal state",
            "metadata": {"type": "system", "source": "Aura.IntentionLoop"},
        },
        {
            "role": "tool",
            "content": "email_adapter read UID 24 from Bryan",
            "metadata": {"type": "tool_result", "skill": "email_adapter"},
        },
        {"role": "assistant", "content": "I'm with you now.", "origin": "api"},
    ]

    filtered = ContextAssembler._filter_stale_skill_results(
        state,
        "That had nothing to do with what I said.",
        list(state.cognition.working_memory),
    )
    joined = "\n".join(str(message.get("content", "")) for message in filtered)

    assert "Hey, Aura" in joined
    assert "I'm with you now." in joined
    assert "Princeton" not in joined
    assert "r/technology" not in joined
    assert "baseline tick" not in joined
    assert "email_adapter" not in joined

    messages = ContextAssembler.build_messages(
        state,
        "That had nothing to do with what I said.",
        max_tokens=4096,
    )
    prompt_text = "\n".join(message["content"] for message in messages)
    assert "Princeton" not in prompt_text
    assert "email_adapter" not in prompt_text
    assert messages[-1]["content"] == "That had nothing to do with what I said."


def test_context_keeps_targeted_deterministic_tool_evidence_only():
    from core.brain.llm.context_assembler import ContextAssembler

    state = AuraState.default()
    state.cognition.working_memory = [
        {
            "role": "system",
            "content": "[SKILL RESULT: clock] 14:30",
            "metadata": {"type": "skill_result", "skill": "clock", "source": "api"},
        }
    ]

    targeted = ContextAssembler._filter_stale_skill_results(
        state,
        "What time is it right now?",
        list(state.cognition.working_memory),
    )
    untargeted = ContextAssembler._filter_stale_skill_results(
        state,
        "Tell me a quick thought.",
        list(state.cognition.working_memory),
    )

    assert targeted == state.cognition.working_memory
    assert untargeted == []


def test_frontend_treats_screensaver_resume_as_surface_pause():
    js = Path("interface/static/aura.js").read_text(encoding="utf-8")

    assert "surfaceSuspended" in js
    assert "reconnectLiveSurface" in js
    assert "Live surface paused. Aura keeps running." in js
    assert "Resuming live surface..." in js
    assert "visibilitychange" in js
    assert "pageshow" in js
