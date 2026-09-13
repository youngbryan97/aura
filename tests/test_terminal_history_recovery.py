"""A sealed delivery retains its transcript obligation across process death."""

from dataclasses import replace
import hashlib
import sqlite3

import pytest
from fastapi.responses import JSONResponse

from core.conversation.unified_transcript import UnifiedTranscript
from core.conversation.persistence import ConversationPersistence
from core.runtime.chat_delivery_journal import (
    ChatDeliveryFenceLost,
    ChatDeliveryJournal,
    ChatDeliveryJournalCorruption,
    DeliveryState,
)
from interface.routes import chat, chat_delivery, chat_preflight
from tests.test_chat_delivery_journal import _identity, _payload, _request, _request_hash
from tests.test_terminal_exchange_custody import custody  # noqa: F401 - shared fixture
from core.runtime.sqlite_support import connecting


@pytest.fixture
def journal(tmp_path):
    return ChatDeliveryJournal(tmp_path / "delivery.sqlite3", retention_s=60)


async def seal(journal):
    admission = await journal.reserve(_identity(), _request_hash(), wait_timeout_s=0)
    record = await journal.finalize(
        admission, state=DeliveryState.COMPLETED, http_status=200,
        response={"response": "The sealed answer.", "status": "ok"},
        history_capture={"exchange-1": {
            "user": "Hello", "session": "session-1", "principal": "bryan", "surface": "owner",
        }},
    )
    return record


@pytest.mark.asyncio
async def test_reopened_journal_preserves_history_until_matching_commit(journal):
    record = await seal(journal)
    reopened = ChatDeliveryJournal(journal.db_path, retention_s=60)
    pending = await reopened.pending_history()
    assert pending[0][0] == record
    assert pending[0][1]["exchange-1"]["user"] == "Hello"
    assert "history_capture" not in record.public_status(include_result=True)
    await reopened.acknowledge_history(record)
    await reopened.acknowledge_history(record)
    assert await reopened.pending_history() == []


@pytest.mark.asyncio
async def test_unfinished_history_survives_delivery_retention(journal):
    record = await seal(journal)
    journal._compact_sync(record.terminal_at + 1000)
    assert await journal.get(record.identity) is not None
    assert len(await journal.pending_history()) == 1
    await journal.acknowledge_history(record)
    journal._compact_sync(record.terminal_at + 1000)
    assert await journal.get(record.identity) is None


@pytest.mark.asyncio
async def test_history_acknowledgement_cannot_discard_a_different_reply(journal):
    record = await seal(journal)
    with pytest.raises(ChatDeliveryFenceLost, match="response changed"):
        await journal.acknowledge_history(replace(record, response={"response": "Different"}))
    assert len(await journal.pending_history()) == 1


@pytest.mark.asyncio
async def test_history_insert_failure_rolls_back_the_terminal_seal(journal):
    admission = await journal.reserve(_identity(), _request_hash(), wait_timeout_s=0)
    with connecting(sqlite3.connect(journal.db_path)) as conn:
        conn.execute("CREATE TRIGGER refuse_history BEFORE INSERT ON chat_delivery_history "
                     "BEGIN SELECT RAISE(ABORT, 'injected storage failure'); END")
    with pytest.raises(ChatDeliveryJournalCorruption):
        await journal.finalize(
            admission, state=DeliveryState.COMPLETED, http_status=200,
            response={"response": "Answer"}, history_capture={"exchange": {"user": "Hello"}},
        )
    assert not (await journal.get(admission.record.identity)).terminal
    assert await journal.pending_history() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_hash", [False, True])
async def test_recovery_rejects_corrupt_private_capture(journal, valid_hash):
    await seal(journal)
    with connecting(sqlite3.connect(journal.db_path)) as conn:
        conn.execute("UPDATE chat_delivery_history SET capture_json=?, capture_hash=?",
                     ("{", hashlib.sha256(b"{").hexdigest() if valid_hash else "wrong"))
    with pytest.raises(ChatDeliveryJournalCorruption):
        await journal.pending_history()


@pytest.mark.asyncio
async def test_real_transcript_and_learning_outbox_survive_replayed_history_commit(journal, tmp_path, monkeypatch):
    store = ConversationPersistence(tmp_path / "conversations.sqlite3")
    transcript = UnifiedTranscript()
    monkeypatch.setattr(UnifiedTranscript, "get_instance", lambda: transcript)
    original_get = chat_preflight.ServiceContainer.get
    monkeypatch.setattr(chat_preflight.ServiceContainer, "get", staticmethod(
        lambda key, default=None: store if key == "persistence" else original_get(key, default=default)
    ))
    monkeypatch.setattr(chat_preflight, "_conversation_log", [])
    monkeypatch.setattr(chat_preflight, "_DURABLE_CONVERSATION_WRITES", {})
    monkeypatch.setattr(chat_preflight, "_schedule_chat_turn_memory_log", lambda **kwargs: None)
    record = await seal(journal)
    acknowledge = journal.acknowledge_history

    async def lose_ack(_record):
        raise RuntimeError("process stopped after transcript commit")

    monkeypatch.setattr(journal, "acknowledge_history", lose_ack)
    with pytest.raises(RuntimeError, match="after transcript commit"):
        await chat_preflight.reconcile_terminal_history(journal)
    assert len(await journal.pending_history()) == 1
    assert len(store.get_session_history("session-1")) == 2
    assert store.memory_log_outbox_status()["pending"] == 1
    monkeypatch.setattr(journal, "acknowledge_history", acknowledge)
    assert await chat_preflight.reconcile_terminal_history(journal) is False
    assert len(store.get_session_history("session-1")) == 2
    assert store.memory_log_outbox_status()["pending"] == 1
    assert len(transcript.entries_for_conversation("session-1")) == 2
    assert (await journal.get(record.identity)).response == record.response


@pytest.mark.asyncio
async def test_replay_repairs_history_after_process_memory_is_lost(custody, monkeypatch):
    journal, writes = custody
    monkeypatch.setattr(chat_preflight, "_schedule_chat_turn_memory_log", lambda **kwargs: None)
    original = chat_preflight.finalize_terminal_exchanges

    async def crash_before_transcript(*args):
        raise RuntimeError("simulated crash after seal")

    monkeypatch.setattr(chat_preflight, "finalize_terminal_exchanges", crash_before_transcript)
    calls = []

    @chat_delivery._paired_chat_response_boundary
    async def handler(*, body, request):
        calls.append(True)
        await chat_preflight._begin_logged_exchange(body.message, session_id=body.session_id)
        return JSONResponse({"response": "Only the sealed answer.", "status": "ok"})

    body = chat.ChatRequest(message="Remember this", session_id="session-1")
    request = _request("recover-terminal-history")
    first = _payload(await handler(body=body, request=request))
    assert first["response"] == "Only the sealed answer."
    assert writes == []
    assert len(await journal.pending_history()) == 1
    # A history-store fault must not hide the authoritative public reply.
    stalled_replay = _payload(await handler(body=body, request=request))
    assert stalled_replay["response"] == first["response"]
    assert len(calls) == 1
    chat_preflight._conversation_log.clear()
    monkeypatch.setattr(chat_preflight, "finalize_terminal_exchanges", original)
    replay = _payload(await handler(body=body, request=request))
    assert replay["response"] == first["response"]
    assert len(calls) == 1
    assert len(writes) == 1
    assert writes[0]["session_id"] == "session-1"
    assert writes[0]["user_message"] == "Remember this"
    assert writes[0]["aura_response"] == first["response"]
    assert await journal.pending_history() == []
    assert len(chat_preflight._conversation_log) == 1
    await handler(body=body, request=request)
    assert len(writes) == 1


@pytest.mark.asyncio
async def test_uncommitted_transcript_keeps_its_obligation_without_duplicate_context(custody, monkeypatch):
    journal, _writes = custody
    transcript = UnifiedTranscript()
    monkeypatch.setattr(UnifiedTranscript, "get_instance", lambda: transcript)
    monkeypatch.setattr(chat_preflight, "_schedule_chat_turn_memory_log", lambda **kwargs: None)
    states = iter(["pending", "committed"])

    async def complete(**kwargs):
        return next(states)

    monkeypatch.setattr(chat_preflight, "_persist_completed_conversation_exchange", complete)
    # Use the real context writer, replaced by the custody fixture by default.
    def record(user, reply, *, session_id, exchange_id, **kwargs):
        for role, text in (("user", user), ("aura", reply)):
            transcript.add(role, text, metadata={"exchange_id": exchange_id}, conversation_id=session_id)

    monkeypatch.setattr(chat_preflight, "_record_unified_transcript_exchange", record)

    @chat_delivery._paired_chat_response_boundary
    async def handler(*, body, request):
        await chat_preflight._begin_logged_exchange(body.message, session_id=body.session_id)
        return JSONResponse({"response": "Answer", "status": "ok"})

    await handler(body=chat.ChatRequest(message="Question", session_id="session-1"), request=_request("pending-history"))
    assert len(await journal.pending_history()) == 1
    assert await chat_preflight.reconcile_terminal_history(journal) is False
    assert len(transcript.entries_for_conversation("session-1")) == 2


def test_transcript_identity_replay_is_idempotent_but_not_a_correction():
    transcript = UnifiedTranscript()
    kwargs = {"metadata": {"exchange_id": "same"}, "conversation_id": "session-1"}
    first = transcript.add_text_output("Original", **kwargs)
    assert transcript.add_text_output("Original", **kwargs) is first
    with pytest.raises(ValueError, match="changed its content"):
        transcript.add_text_output("Changed", **kwargs)
    assert transcript.replace_aura_reply(exchange_id="same", expected_content="Original",
                                        replacement_content="Corrected", revision=2, conversation_id="session-1")
    transcript.add_text_output("Independent", metadata={"exchange_id": "same"}, conversation_id="session-2")
    assert len(transcript.entries_for_conversation("session-1")) == 1
