"""Where the changing part sits decides how much of the prompt can be reused.

Mood, valence, arousal, energy and focus change on every turn. They sat at the
end of the FIRST system message, which precedes the summary, the entire
history and the user's turn — so the KV prefix diverged inside the system
block and everything after it was recomputed.

Measured live 2026-08-18, the same figure over and over:

    prefix diverges at token 132 (16% of 831 reused)
    prefix diverges at token 132 (17% of 773 reused)
    prefix diverges at token 132 (18% of 719 reused)

Five sixths of the prompt re-prefilled every turn on a 32B, which is most of
what a person waits through.

`llm_health_router` already records this rule for the system-state header it
appends ("volatile grounding last"). This applies it across the message list
rather than within a single message.
"""

from __future__ import annotations

import asyncio

import pytest

from interface.routes import chat_protected_prompt as prompt_module


def _messages(monkeypatch, history):
    monkeypatch.setattr(
        prompt_module,
        "_build_protected_foreground_history",
        lambda **_kwargs: _async(history),
    )
    monkeypatch.setattr(
        prompt_module, "_build_protected_foreground_summary_message", lambda: None
    )
    return asyncio.run(
        prompt_module._build_protected_foreground_messages(
            "what's the weather", lane={"state": "ready"}, route={}, session_id=""
        )
    )


async def _async(value):
    return value


HISTORY = [
    {"role": "user", "content": "earlier question"},
    {"role": "assistant", "content": "earlier answer"},
]


def test_the_volatile_block_comes_after_the_history(monkeypatch):
    messages = _messages(monkeypatch, list(HISTORY))
    snapshot_at = [
        i for i, m in enumerate(messages)
        if prompt_module.SNAPSHOT_HEADING in str(m.get("content", ""))
    ]
    if not snapshot_at:
        pytest.skip("no volatile state available in this environment")
    history_at = [
        i for i, m in enumerate(messages) if m.get("content") == "earlier answer"
    ]
    assert history_at, "history was not included"
    assert snapshot_at[0] > history_at[0], (
        "volatile state still precedes the history, so the history cannot be reused"
    )


def test_the_volatile_block_is_immediately_before_the_user_turn(monkeypatch):
    messages = _messages(monkeypatch, list(HISTORY))
    if not any(
        prompt_module.SNAPSHOT_HEADING in str(m.get("content", "")) for m in messages
    ):
        pytest.skip("no volatile state available in this environment")
    assert messages[-1]["role"] == "user"
    assert prompt_module.SNAPSHOT_HEADING in messages[-2]["content"]


def test_the_first_system_message_holds_no_volatile_state(monkeypatch):
    """It is the longest reusable span, so nothing that changes may live in it."""
    messages = _messages(monkeypatch, list(HISTORY))
    assert messages[0]["role"] == "system"
    assert prompt_module.SNAPSHOT_HEADING not in messages[0]["content"]
    # The instructions themselves must survive there.
    assert "protected foreground chat control plane" in messages[0]["content"]


def test_the_single_string_form_still_carries_everything():
    """Callers that want one string — and the tests that assert on it — keep it."""
    whole = prompt_module._build_protected_foreground_system_prompt(
        "what's the weather", lane={"state": "ready"}
    )
    stable, volatile = prompt_module._protected_foreground_prompt_parts(
        "what's the weather", lane={"state": "ready"}
    )
    assert stable in whole
    if volatile:
        assert volatile in whole
        assert prompt_module.SNAPSHOT_HEADING in whole
