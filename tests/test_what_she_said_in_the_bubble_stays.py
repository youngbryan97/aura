"""What she says in the bubble stays: in her conversation, in the companion panel, and in her window.

Reported live 16 November 2026: "also the messages dont stick in the panel". The panel showed only the remark queued
when it opened, cleared it, and read nothing else; after a restart her keeper had nothing in memory to keep a remark
with ("said unasked, and no conversation kept it"); and the history both windows read dropped what she said after an
exchange.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPANION_JS = (ROOT / "interface/static/companion_chat.js").read_text(encoding="utf-8")
AURA_JS = (ROOT / "interface/static/aura.js").read_text(encoding="utf-8")


def test_the_panel_opens_on_the_conversation_so_far():
    restore = COMPANION_JS[COMPANION_JS.index("async function restoreConversation"):]
    restore = restore[:restore.index("\n  }\n")]

    assert 'fetch("/api/ui/bootstrap"' in restore
    assert "entry.said_after" in restore and 'bubble(entry.user, "me")' in restore
    opened = COMPANION_JS[COMPANION_JS.index("await restoreConversation()"):]
    assert opened.index("await restoreConversation()") < opened.index('fetch("/api/ambient/state"'), (
        "the conversation comes before the remark that opened the panel")
    assert "!shown.has(" in opened, "a remark already in the conversation is not shown twice"


def test_her_window_draws_what_she_said_after_an_exchange_under_its_own_id():
    convert = AURA_JS[AURA_JS.index("function conversationEntriesToMessages"):]
    convert = convert[:convert.index("\n}\n")]

    assert "saidAfter(entry)" in convert
    assert ":after:${nth + 1}" in convert, "a remark shares no id with the answer it follows"


@pytest.mark.asyncio
async def test_the_history_both_windows_read_carries_her_remarks(monkeypatch):
    from interface.routes import chat_history, chat_memory_state

    async def durable(**_kwargs):
        return [{"exchange_id": "e1", "user": "Hi.", "aura": "Hello.", "timestamp": "1", "session_id": "s",
                 "said_after": ["That form wants one thing you built, five levels down."]}]

    monkeypatch.setattr(chat_memory_state, "_load_durable_conversation_exchanges", durable)
    async with chat_memory_state._get_convo_lock():
        chat_memory_state._conversation_log.clear()

    recent = await chat_history.recent_ui_conversation(None)

    assert recent[-1]["said_after"] == ["That form wants one thing you built, five levels down."]


def test_after_a_restart_a_remark_is_kept_with_the_last_exchange_on_disk(monkeypatch):
    from interface.routes import chat_memory_state

    recorded: list[dict] = []

    class Persistence:
        def record_turn(self, role, content, **kwargs):
            recorded.append({"role": role, "content": content, **kwargs})

    async def durable(**_kwargs):
        assert chat_memory_state._CHAT_REQUEST_SURFACE.get() == "owner", "read as the owner, whose desk it is"
        return [{"exchange_id": "0d62b5bc", "session_id": "s1", "said_after": ["an earlier remark"]}]

    monkeypatch.setattr(chat_memory_state, "_load_durable_conversation_exchanges", durable)
    monkeypatch.setattr(chat_memory_state.ServiceContainer, "get",
                        lambda name, default=None: Persistence() if name == "persistence" else default)

    async def keep():
        async with chat_memory_state._get_convo_lock():
            chat_memory_state._conversation_log.clear()
        return await chat_memory_state._keep_what_she_said_unasked("That page is the submit form.", "ambient_remark",
                                                                    1.0)

    assert asyncio.run(keep()) is True
    assert recorded and recorded[0]["cid"] == "0d62b5bc:after:2" and recorded[0]["role"] == "aura"
    assert recorded[0]["origin"] == "ambient_remark" and recorded[0]["metadata"] == {"unasked": True}


def test_an_open_obligation_does_not_hold_what_she_does_at_the_desk():
    """LIVE 16 November 2026, 08:32: her pointing was deferred by "A quiet moment with nothing asked of you"."""
    from core.executive.executive_core import ActionType, ExecutiveCore, Intent
    from core.executive.standing_authority import DESK_COMPANY_ORIGIN

    def held(origin):
        intent = Intent(goal="execute_tool:point_at_on_screen", action_type=ActionType.TOOL_CALL,
                        payload={"tool_name": "point_at_on_screen", "args": {"words": "Back"}, "origin": origin})
        return ExecutiveCore._temporal_safe_autonomous_tool_constraints(intent) is None

    assert not held(DESK_COMPANY_ORIGIN)
    assert held("autonomous_initiative_loop")
