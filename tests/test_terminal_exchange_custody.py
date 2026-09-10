"""Every terminal route owns the same transcript and its exact public bytes."""

from __future__ import annotations

import asyncio
import hashlib

import pytest
from fastapi.responses import JSONResponse

from core.runtime.chat_delivery_journal import ChatDeliveryJournal
from interface.routes import chat, chat_delivery, chat_preflight
from tests.test_chat_delivery_journal import _patch_route_identity, _payload, _request


@pytest.fixture
def custody(tmp_path, monkeypatch):
    journal = ChatDeliveryJournal(tmp_path / "deliveries.sqlite3", poll_interval_s=0.01)
    _patch_route_identity(monkeypatch, journal)
    writes = []

    async def pending(**_kwargs):
        return "committed"

    async def complete(**kwargs):
        writes.append(kwargs)
        return "committed"

    monkeypatch.setattr(chat_preflight, "_persist_pending_conversation_user", pending)
    monkeypatch.setattr(chat_preflight, "_persist_completed_conversation_exchange", complete)
    monkeypatch.setattr(chat_preflight, "_conversation_memory_outbox_available", lambda: True)
    monkeypatch.setattr(chat_preflight, "_record_unified_transcript_exchange", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_preflight, "_durable_conversation_write_snapshot", lambda _key: None)
    monkeypatch.setattr(chat_preflight, "_conversation_log", [])
    return journal, writes


@pytest.mark.asyncio
@pytest.mark.parametrize("early_return", [False, True])
async def test_only_sealed_bytes_enter_history_even_after_an_early_return(custody, early_return):
    _journal, writes = custody
    exchange_ids = []

    @chat_delivery._paired_chat_response_boundary
    async def handler(*, body, request):
        exchange_id = await chat_preflight._begin_logged_exchange(body.message, session_id=body.session_id)
        exchange_ids.append(exchange_id)
        if not early_return:
            result = await chat_preflight._complete_logged_exchange(exchange_id, body.message, "The discarded draft.")
            assert result == "pending_terminal_delivery"
        assert writes == []
        return JSONResponse({"response": "The complete public reply.", "status": "full_mind_contract_unproven_served"})

    body = chat.ChatRequest(message="Explain this", session_id="session-1")
    request = _request("terminal-history")
    response = _payload(await handler(body=body, request=request))
    assert len(writes) == 1
    assert writes[0]["aura_response"] == response["response"]
    assert writes[0]["exchange_id"] == exchange_ids[0]
    entry = chat_preflight._conversation_log[0]
    assert entry["status"] == "complete"
    assert entry["aura"] == response["response"]
    assert entry["metadata"]["delivered_response_sha256"] == hashlib.sha256(response["response"].encode()).hexdigest()
    await handler(body=body, request=request)
    assert len(writes) == 1
    assert len(chat_preflight._conversation_log) == 1
    assert chat_preflight._TERMINAL_EXCHANGES.get() is None


@pytest.mark.asyncio
async def test_cancelled_turn_finishes_its_pending_history_without_learning_a_draft(custody):
    journal, writes = custody
    entered = asyncio.Event()

    @chat_delivery._paired_chat_response_boundary
    async def handler(*, body, request):
        await chat_preflight._begin_logged_exchange(body.message, session_id=body.session_id)
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            from core.runtime.what_stops_it import current

            assert current().stopping.stopped
            assert current().stopping.why == chat_delivery.USER_CANCEL_REASON
            raise

    task = asyncio.create_task(handler(body=chat.ChatRequest(message="Think", session_id="session-1"), request=_request("cancel-history")))
    await asyncio.wait_for(entered.wait(), 2)
    identity = next(iter(chat_delivery._executing_deliveries))
    chat_delivery.request_delivery_cancellation(await journal.get(identity))
    response = _payload(await task)
    assert response["status"] == "cancelled_by_user"
    assert len(writes) == 1
    assert writes[0]["aura_response"] == response["response"]
    assert writes[0]["enqueue_memory_log"] is False
    assert chat_preflight._conversation_log[0]["status"] == "complete"
