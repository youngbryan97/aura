"""The rescue route delegates generation lifetime to the resident gate."""

import asyncio
from types import SimpleNamespace

import pytest

from interface.routes import chat


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_protected_reply_does_not_reimpose_the_initial_budget(monkeypatch, cancelled):
    calls = []

    async def generate(message, *, context, timeout):
        calls.append((context, timeout))
        await asyncio.sleep(0)
        if cancelled:
            raise asyncio.CancelledError
        return ""

    async def messages(*args, **kwargs):
        return []

    async def duplicate_timer(*args, **kwargs):
        args[0].close()
        raise AssertionError("route added a second generation timer")

    monkeypatch.setattr(chat.ServiceContainer, "get", lambda *a, **kw: SimpleNamespace(generate=generate))
    monkeypatch.setattr(chat, "_protected_foreground_generation_block_reason", lambda: "")
    monkeypatch.setattr(chat, "_protected_foreground_route", lambda _: {"prefer_tier": "primary"})
    monkeypatch.setattr(chat._chat_protected_prompt, "_build_protected_foreground_messages", messages)
    monkeypatch.setattr(chat, "asyncio", SimpleNamespace(wait_for=duplicate_timer))
    operation = chat._protected_foreground_reply(
        "canonical_empty_reply", budget_override_s=6,
        _chat_session_id="completion-owner", _live_turn_trace={},
        _remaining_foreground_budget=lambda **kw: 6,
        _semantic_user_message="Explain a counter update.",
        body=SimpleNamespace(message="Explain a counter update."),
        chat_origin="desktop_ui", desktop_requires_cognitive_engine=True,
        is_benchmark=False, lane={},
    )
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        assert await operation is None
    assert len(calls) == 1
    assert calls[0][1] == 6
    assert calls[0][0]["foreground_request"] is True


@pytest.mark.asyncio
async def test_protected_reply_never_runs_for_benchmark():
    assert await chat._protected_foreground_reply(
        "empty", _chat_session_id="benchmark", _live_turn_trace={},
        _remaining_foreground_budget=None, _semantic_user_message="",
        body=None, chat_origin="benchmark", desktop_requires_cognitive_engine=True,
        is_benchmark=True, lane={},
    ) is None
