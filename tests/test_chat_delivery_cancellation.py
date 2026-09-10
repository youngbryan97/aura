"""Explicit cancellation belongs to an authenticated, fenced turn owner."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse

from core.runtime.chat_delivery_journal import ChatDeliveryJournal, DeliveryState
from interface.routes import chat as chat_mod
from interface.routes import chat_delivery as delivery
from tests.test_chat_delivery_journal import _identity, _patch_route_identity, _payload, _request


@pytest.fixture
def journal(tmp_path, monkeypatch):
    journal = ChatDeliveryJournal(tmp_path / "delivery.sqlite3", poll_interval_s=0.01)
    _patch_route_identity(monkeypatch, journal)
    return journal


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_catches_cancel", [False, True])
async def test_cancel_seals_one_terminal_receipt_and_never_reexecutes(journal, handler_catches_cancel):
    entered = asyncio.Event()
    exited = asyncio.Event()
    calls = 0

    @delivery._paired_chat_response_boundary
    async def handler(*, body, request):
        nonlocal calls
        calls += 1
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not handler_catches_cancel:
                raise
            return JSONResponse({"response": "A late draft", "status": "ok"})
        finally:
            exited.set()

    request = _request("user-cancel")
    body = chat_mod.ChatRequest(message="Explain delivery", session_id="session-1")
    task = asyncio.create_task(handler(body=body, request=request))
    await asyncio.wait_for(entered.wait(), 2)
    identity = next(iter(delivery._executing_deliveries))
    record = await journal.get(identity)
    assert delivery.request_delivery_cancellation(record) == "cancellation_requested"
    assert delivery.request_delivery_cancellation(record) == "cancellation_requested"
    response = await asyncio.wait_for(task, 2)
    assert exited.is_set()
    assert _payload(response)["status"] == "cancelled_by_user"
    assert _payload(response)["delivery_state"] == DeliveryState.FAILED
    assert identity not in delivery._executing_deliveries
    replay = await handler(body=body, request=request)
    assert _payload(replay)["status"] == "cancelled_by_user"
    assert _payload(replay)["delivery_replayed"]
    assert calls == 1
    assert delivery.request_delivery_cancellation(await journal.get(identity)) == "already_terminal"


@pytest.mark.asyncio
async def test_stale_or_other_session_record_cannot_cancel_the_owner(journal):
    entered = asyncio.Event()
    release = asyncio.Event()

    @delivery._paired_chat_response_boundary
    async def handler(*, body, request):
        entered.set()
        await release.wait()
        return JSONResponse({"response": "Completed normally.", "status": "ok"})

    task = asyncio.create_task(handler(
        body=chat_mod.ChatRequest(message="Continue", session_id="session-1"),
        request=_request("isolated-cancel"),
    ))
    await asyncio.wait_for(entered.wait(), 2)
    identity = next(iter(delivery._executing_deliveries))
    record = await journal.get(identity)
    assert delivery.request_delivery_cancellation(replace(record, generation=record.generation + 1)) == "owner_changed"
    assert delivery.request_delivery_cancellation(replace(record, identity=_identity("isolated-cancel", session_id="other"))) == "execution_not_cancellable"
    assert not task.cancelling()
    release.set()
    assert _payload(await task)["response"] == "Completed normally."


@pytest.mark.asyncio
async def test_endpoint_derives_principal_and_session_before_cancelling(journal, monkeypatch):
    admission = await journal.reserve(_identity("endpoint-cancel"), "a" * 64, wait_timeout_s=0.1)
    seen = []
    monkeypatch.setattr(delivery, "request_delivery_cancellation", lambda record: seen.append(record.identity) or "cancellation_requested")
    request = _request("endpoint-cancel")
    wrong_session = await chat_mod.api_chat_delivery_cancel("endpoint-cancel", request, session_id="other")
    assert wrong_session.status_code == 404
    monkeypatch.setattr(delivery, "_authenticated_chat_principal", lambda _request: "owner:someone-else")
    wrong_principal = await chat_mod.api_chat_delivery_cancel("endpoint-cancel", request, session_id="session-1")
    assert wrong_principal.status_code == 404
    assert seen == []
    monkeypatch.setattr(delivery, "_authenticated_chat_principal", lambda _request: "owner:bryan")
    own = await chat_mod.api_chat_delivery_cancel("endpoint-cancel", request, session_id="session-1")
    assert own.status_code == 202
    assert _payload(own)["cancellation_status"] == "cancellation_requested"
    assert seen == [admission.record.identity]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_desktop_cancellation_keeps_terminal_delivery_with_its_original_observer():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [shutil.which("node"), str(root / "tests/js/chat_cancellation.mjs"), str(root / "interface/static/aura.js")],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
