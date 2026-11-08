"""What she says in the chat without being asked is in the conversation she keeps, under the exchange it followed.

LIVE 7 November 2026, dress rehearsal at 22:44: twenty remarks and three last words were in the chat window and not
in her store, and asked "So what did you think of all three?", she chose Salad Fingers' Spoons a second time.
"""

import asyncio

import pytest

from core.container import ServiceContainer
from core.conversation import said_unasked
from core.conversation.delivered_history import (
    delivered_exchange_messages,
    reached_exchange_messages,
)
from core.conversation.persistence import ConversationPersistence
from core.runtime.shared_read import SharedRead
from core.utils.injected_blocks import stamp_runtime_payload
from interface.routes import chat_memory_state
from interface.routes.chat_common import _CHAT_REQUEST_PRINCIPAL, _CHAT_REQUEST_SURFACE


@pytest.fixture
def store(monkeypatch, tmp_path):
    kept = ConversationPersistence(tmp_path / "history.db")
    original = ServiceContainer.get
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(
        lambda key, default=None: kept if key == "persistence" else original(key, default=default)
    ))
    monkeypatch.setattr(chat_memory_state, "_conversation_log", [])
    monkeypatch.setattr(chat_memory_state, "_durable_history_reads", SharedRead())
    return kept


def _exchange(exchange_id, user, aura, status="complete"):
    return {"id": exchange_id, "user": user, "aura": aura, "status": status, "session_id": "sitting",
            "principal_id": "owner-a", "principal_surface": "owner"}


def test_her_model_reads_what_she_said_after_her_reply_as_part_of_that_exchange():
    exchanges = [
        stamp_runtime_payload({"user": "Now Salad Fingers. Pick whichever episode you want.",
                               "aura": "I'm picking Spoons. I expect it to be quiet.",
                               "said_after": ["Oh, caress this rusty kettle.", "It's over. That door stayed with me."]}),
    ]
    messages = delivered_exchange_messages(exchanges)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["content"] == ("I'm picking Spoons. I expect it to be quiet.\n\nOh, caress this rusty kettle."
                                      "\n\nIt's over. That door stayed with me.")


def test_the_last_exchange_a_request_answers_into_carries_what_followed_it():
    exchanges = [
        stamp_runtime_payload({"user": f"question {n}", "aura": f"answer {n}"}) for n in range(3)
    ] + [stamp_runtime_payload({"user": "Pick one.", "aura": "This one.", "said_after": ["I watched it."]})]
    reached = reached_exchange_messages(exchanges, request="So what did you think of all three?")
    assert reached.messages[-1] == {"role": "assistant", "content": "This one.\n\nI watched it."}


@pytest.mark.asyncio
async def test_a_remark_is_kept_with_the_last_complete_exchange_in_memory_and_on_disk(store):
    chat_memory_state._conversation_log.extend([
        _exchange("e1", "Let's watch this together.", "I think it'll be about luck."),
        _exchange("e2", "Pending question", "", status="pending"),
    ])
    assert await said_unasked.she_said_unasked("  The hockey birthdays.  ", origin="desktop_watching_along")
    assert await said_unasked.she_said_unasked("It's over.", origin="desktop_watching_along")
    said_after = chat_memory_state._conversation_log[0]["said_after"]
    assert [item["words"] for item in said_after] == ["The hockey birthdays.", "It's over."]
    assert "said_after" not in chat_memory_state._conversation_log[1]
    rows = await asyncio.to_thread(store.get_recent_history, limit=10)
    assert [(row["role"], row["cid"], row["content"]) for row in rows] == [
        ("aura", "e1:after:1", "The hockey birthdays."),
        ("aura", "e1:after:2", "It's over."),
    ]


@pytest.mark.asyncio
async def test_recent_exchanges_carry_what_she_said_after_each(store):
    chat_memory_state._conversation_log.append(_exchange("e1", "Let's watch this.", "Before it plays: luck."))
    await said_unasked.she_said_unasked("Remark one.", origin="desktop_watching_along")
    principal = _CHAT_REQUEST_PRINCIPAL.set("owner-a")
    surface = _CHAT_REQUEST_SURFACE.set("owner")
    try:
        exchanges = await chat_memory_state._recent_completed_conversation_exchanges(
            current_user_message="So what did you think?", limit=1)
    finally:
        _CHAT_REQUEST_SURFACE.reset(surface)
        _CHAT_REQUEST_PRINCIPAL.reset(principal)
    assert exchanges[-1]["said_after"] == ["Remark one."]


def test_after_a_restart_what_she_said_after_an_exchange_is_read_back_with_it_in_order(store):
    store.record_exchange("Let's watch this.", "Before it plays: luck.", session_id="before-reboot", cid="e1",
                          principal_id="owner-a", principal_surface="owner")
    for nth, words in ((2, "Second remark."), (1, "First remark.")):
        store.record_turn("aura", words, origin="desktop_watching_along", cid=f"e1:after:{nth}",
                          session_id="before-reboot", metadata={"unasked": True},
                          principal_id="owner-a", principal_surface="owner")
    principal = _CHAT_REQUEST_PRINCIPAL.set("owner-a")
    surface = _CHAT_REQUEST_SURFACE.set("owner")
    try:
        exchanges = chat_memory_state._load_durable_conversation_exchanges_sync(limit=5)
    finally:
        _CHAT_REQUEST_SURFACE.reset(surface)
        _CHAT_REQUEST_PRINCIPAL.reset(principal)
    assert [(e["user"], e["aura"], e["said_after"]) for e in exchanges] == [
        ("Let's watch this.", "Before it plays: luck.", ["First remark.", "Second remark."]),
    ]


@pytest.mark.asyncio
async def test_nothing_said_is_nothing_kept_and_a_failing_keeper_is_recorded(monkeypatch):
    calls = []

    async def failing(words, origin, at):
        calls.append(words)
        raise RuntimeError("store gone")

    monkeypatch.setattr(said_unasked, "_KEEPERS", [failing])
    assert not await said_unasked.she_said_unasked("   ", origin="x")
    assert calls == []
    assert not await said_unasked.she_said_unasked("Said.", origin="x")
    assert calls == ["Said."]
