from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from core.container import ServiceContainer
from core.runtime.principal_context import (
    current_relational_principal,
    relational_principal_scope,
)
from core.runtime.receipts import get_receipt_store, reset_receipt_store
from core.social.other_agent_model import OtherAgentStateEstimator
from core.social.relational_memory import RelationalMemoryAuthority
from core.utils.output_gate import AutonomousOutputGate
from interface.routes import chat as chat_routes
import interface.routes.chat_delivery as _chat_delivery
from tests.chat_lane_support import patch_chat_lane


def _authority(tmp_path) -> RelationalMemoryAuthority:
    return RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"h" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )


def _request(*, idempotency_key: str = "") -> Request:
    headers = []
    if idempotency_key:
        headers.append((b"x-idempotency-key", idempotency_key.encode("ascii")))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": headers,
            "client": ("127.0.0.1", 4242),
        }
    )


class _Observer:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def observe_agent(self, agent_id, **kwargs):
        self.events.append((agent_id, kwargs["evidence_digest"]))

    def register_interaction(self, agent_id, snapshot):
        self.events.append((agent_id, snapshot["evidence_digest"]))


class _Tracker:
    def __init__(self) -> None:
        self.tasks: list[asyncio.Task] = []

    def track(self, awaitable, *, name):
        task = asyncio.create_task(awaitable, name=name)
        self.tasks.append(task)
        return task


@pytest.mark.asyncio
async def test_http_turn_applies_consent_and_observes_exact_principal_idempotently(
    tmp_path,
    monkeypatch,
):
    authority = _authority(tmp_path)
    estimator = OtherAgentStateEstimator(
        storage_path=tmp_path / "legacy.json",
        authority=authority,
        autosave=False,
    )
    observer = _Observer()
    tracker = _Tracker()
    request = _request(idempotency_key="same-request")
    body = SimpleNamespace(message="Aura, remember always.", session_id="session-1")
    monkeypatch.setattr(
        _chat_delivery,
        "_authenticated_chat_principal",
        lambda _request: "bryan",
    )
    patch_chat_lane(monkeypatch, "get_task_tracker", lambda: tracker)
    ServiceContainer.clear()
    ServiceContainer.register_instance("relational_memory", authority, required=False)
    ServiceContainer.register_instance("other_agent_model", estimator, required=False)
    ServiceContainer.register_instance("recursive_tom", observer, required=False)

    try:
        assert chat_routes._observe_authenticated_chat_turn(request, body) == "bryan"
        assert chat_routes._observe_authenticated_chat_turn(request, body) == "bryan"
        await asyncio.gather(*tracker.tasks)
    finally:
        ServiceContainer.clear()

    assert request.state.relational_memory_control["mode"] == "remember_always"
    assert authority.allows("bryan", "derived_profile", "persist") is True
    assert authority.allows("alice", "derived_profile", "persist") is False
    assert estimator.estimate("bryan").observations == 1
    assert len(observer.events) == 4
    assert len({digest for _, digest in observer.events}) == 1


def test_http_turn_rebinds_empty_estimator_provenance_before_recursive_observation(
    monkeypatch,
):
    from core.consciousness.recursive_tom import ObserverContextModel

    class _AbstainingEstimator:
        def observe_message(self, *_args, **_kwargs):
            return None

        def cognitive_snapshot(self, principal, _observed_at):
            return {
                "agent_id": principal,
                "confidence": 0.0,
                "observations": 0,
                "affect_hypotheses": {},
                "evidence_digest": "",
                "at": 0.0,
            }

    degradations = []
    observer = ObserverContextModel()
    monkeypatch.setattr(
        _chat_delivery,
        "_authenticated_chat_principal",
        lambda _request: "bryan",
    )
    patch_chat_lane(monkeypatch, "record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )
    ServiceContainer.clear()
    ServiceContainer.register_instance(
        "other_agent_model",
        _AbstainingEstimator(),
        required=False,
    )
    ServiceContainer.register_instance("recursive_tom", observer, required=False)

    try:
        principal = chat_routes._observe_authenticated_chat_turn(
            _request(idempotency_key="social-proof"),
            SimpleNamespace(message="Are you with me?", session_id="session-1"),
        )
        snapshot = observer.get_mind("bryan")
    finally:
        ServiceContainer.clear()

    assert principal == "bryan"
    assert snapshot is not None
    assert len(snapshot.evidence_digest) == 64
    assert snapshot.captured_at > 0.0
    assert degradations == []


@pytest.mark.asyncio
async def test_http_feedback_window_opens_only_after_response_background_runs(
    tmp_path,
    monkeypatch,
):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["recall", "prompt"],
        receipt_id="session-social-consent",
    )
    estimator = OtherAgentStateEstimator(
        storage_path=tmp_path / "legacy.json",
        authority=authority,
        autosave=False,
    )
    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    monkeypatch.setattr(
        "core.runtime.receipts.get_receipt_store",
        lambda *args, **kwargs: store,
    )
    monkeypatch.setattr(
        _chat_delivery,
        "_authenticated_chat_principal",
        lambda _request: "bryan",
    )
    ServiceContainer.clear()
    ServiceContainer.register_instance("other_agent_model", estimator, required=False)
    response = JSONResponse({"response": "I fixed the configuration.", "status": "ok"})
    background_events: list[str] = []

    async def existing_background() -> None:
        background_events.append("existing")

    response.background = BackgroundTask(existing_background)
    chat_routes._attach_http_chat_delivery_receipt(
        response,
        request=_request(),
        body=SimpleNamespace(message="fix it", session_id="session-1"),
        payload={"response": "I fixed the configuration.", "status": "ok"},
    )

    try:
        assert estimator.get_health()["pending_response_count"] == 0
        assert response.background is not None
        await response.background()
        assert background_events == ["existing"]
        assert estimator.get_health()["pending_response_count"] == 1
        receipt_id = estimator._pending_responses["bryan"][2]
        receipt = store.get(receipt_id)
        assert receipt is not None
        assert receipt.metadata["delivery_stage"] == "transport_accepted"
        assert receipt.metadata["accepted_sinks"] == ["http_response_body"]
        assert receipt.metadata["recipient_principal_digest"] == hashlib.sha256(
            b"bryan"
        ).hexdigest()
        assert "bryan" not in str(receipt.metadata)

        feedback = estimator.observe_message(
            "bryan",
            "perfect, that works",
            evidence_digest=hashlib.sha256(b"http-feedback").hexdigest(),
        )
        assert feedback.response_feedback_context is True
        assert feedback.belief_confidence["aura_capable"] > 0.0
    finally:
        ServiceContainer.clear()
        reset_receipt_store()


@pytest.mark.asyncio
async def test_output_gate_receipt_preserves_valid_principal_binding(
    tmp_path,
    monkeypatch,
):
    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    monkeypatch.setattr(
        "core.runtime.receipts.get_receipt_store",
        lambda *args, **kwargs: store,
    )
    principal_digest = hashlib.sha256(b"bryan").hexdigest()
    gate = AutonomousOutputGate()

    receipt_id = await gate._emit_output_receipt(
        "principal-bound output",
        origin="user",
        target="primary",
        metadata={
            "accepted_sinks": ["reply_queue"],
            "recipient_principal_digest": principal_digest,
        },
    )

    assert receipt_id
    receipt = store.get(receipt_id)
    assert receipt is not None
    assert receipt.metadata["recipient_principal_digest"] == principal_digest
    reset_receipt_store()


def test_http_social_path_abstains_without_authenticated_principal(monkeypatch):
    monkeypatch.setattr(
        _chat_delivery,
        "_authenticated_chat_principal",
        lambda _request: "",
    )
    response = JSONResponse({"response": "hello", "status": "ok"})

    assert chat_routes._observe_authenticated_chat_turn(
        _request(),
        SimpleNamespace(message="hello", session_id="session-1"),
    ) == ""
    chat_routes._attach_http_chat_delivery_receipt(
        response,
        request=_request(),
        body=SimpleNamespace(message="hello", session_id="session-1"),
        payload={"response": "hello", "status": "ok"},
    )
    assert response.background is None


@pytest.mark.asyncio
async def test_failed_turn_poll_reuses_immutable_answer_receipt(tmp_path, monkeypatch):
    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    monkeypatch.setattr("core.runtime.receipts.get_receipt_store", lambda: store)
    monkeypatch.setattr(_chat_delivery, "_authenticated_chat_principal", lambda request: "bryan")
    monkeypatch.setattr(_chat_delivery, "optional_service", lambda name: None)
    degradations = []
    monkeypatch.setattr(_chat_delivery, "record_degradation", lambda *args, **kwargs: degradations.append(args))
    record = SimpleNamespace(turn_id="failed-turn", terminal_at=12345.0, http_status=503)
    payload = {"response": "The operation failed.", "status": "failed"}
    try:
        # Status polling may finish before the original POST background hook.
        for transport_status in (200, 503, 200):
            response = JSONResponse(payload, status_code=transport_status)
            _chat_delivery._attach_http_chat_delivery_receipt(
                response, request=_request(),
                body=SimpleNamespace(message="test", session_id="session-1"),
                payload=payload, record=record,
            )
            await response.background()
        receipt = store.get("output-chat-http-failed-turn")
        assert receipt.metadata["status_code"] == 503
        assert receipt.created_at == 12345.0
        assert not degradations
    finally:
        reset_receipt_store()


@pytest.mark.asyncio
async def test_relational_principal_scope_is_task_local_and_restored():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def read_scoped(principal: str) -> tuple[str, str]:
        with relational_principal_scope(principal):
            entered.set()
            await release.wait()
            during = current_relational_principal()
        return during, current_relational_principal()

    first = asyncio.create_task(read_scoped("bryan"))
    await entered.wait()
    second = asyncio.create_task(read_scoped("alice"))
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(first, second)

    assert sorted(results) == [("alice", ""), ("bryan", "")]
    assert current_relational_principal() == ""


@pytest.mark.asyncio
async def test_paired_chat_boundary_holds_exact_principal_for_full_concurrent_turn(
    monkeypatch,
):
    entered = 0
    both_entered = asyncio.Event()
    observed: list[tuple[str, str]] = []

    monkeypatch.setattr(
        _chat_delivery,
        "_authenticated_chat_principal",
        lambda request: str(request.headers.get("x-test-principal") or ""),
    )

    @chat_routes._paired_chat_response_boundary
    async def handler(body, request):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await both_entered.wait()
        before = current_relational_principal()
        await asyncio.sleep(0)
        observed.append((before, current_relational_principal()))
        return JSONResponse({"response": f"hello {body.message}", "status": "ok"})

    def scoped_request(principal: str) -> Request:
        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/chat",
                "headers": [(b"x-test-principal", principal.encode("ascii"))],
                "client": ("127.0.0.1", 4242),
            }
        )

    ServiceContainer.clear()
    try:
        await asyncio.gather(
            handler(SimpleNamespace(message="one", session_id="s1"), scoped_request("bryan")),
            handler(SimpleNamespace(message="two", session_id="s2"), scoped_request("alice")),
        )
    finally:
        ServiceContainer.clear()

    assert sorted(observed) == [("alice", "alice"), ("bryan", "bryan")]
    assert current_relational_principal() == ""
