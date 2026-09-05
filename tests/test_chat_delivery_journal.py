from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import stat
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse
from starlette.requests import Request

from core.runtime.chat_delivery_journal import (
    AdmissionKind,
    ChatDeliveryFenceLost,
    ChatDeliveryJournal,
    ChatDeliveryJournalCorruption,
    ChatDeliveryJournalUnavailable,
    DeliveryIdentity,
    DeliveryState,
    canonical_request_hash,
)
import interface.routes.chat_delivery as _chat_delivery
from tests.chat_lane_support import patch_chat_lane


def _identity(
    key: str = "turn-1",
    *,
    principal: str = "owner:bryan",
    session_id: str = "session-1",
) -> DeliveryIdentity:
    return DeliveryIdentity.create(
        principal=principal,
        session_id=session_id,
        idempotency_key=key,
    )


def _request_hash(message: str = "hello") -> str:
    return canonical_request_hash({"message": message})


def _request(
    key: str,
    *,
    path: str = "/api/chat",
    method: str = "POST",
    benchmark: bool = False,
) -> Request:
    headers = [(b"x-idempotency-key", key.encode("ascii"))]
    if benchmark:
        headers.append((b"x-aura-benchmark", b"true"))
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 50123),
            "server": ("127.0.0.1", 8000),
        }
    )


def _payload(response: JSONResponse) -> dict[str, object]:
    decoded = json.loads(bytes(response.body))
    assert isinstance(decoded, dict)
    return decoded


@pytest.fixture
def journal(tmp_path: Path) -> ChatDeliveryJournal:
    return ChatDeliveryJournal(
        tmp_path / "runtime" / "chat.sqlite3",
        stale_after_s=2.0,
        retention_s=60.0,
        abandon_after_s=30.0,
        poll_interval_s=0.01,
    )


def test_initialization_uses_private_filesystem_permissions(tmp_path: Path) -> None:
    path = tmp_path / "private" / "chat.sqlite3"

    ChatDeliveryJournal(path)

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_existing_malformed_database_fails_closed_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chat.sqlite3"
    original = b"not-a-sqlite-database"
    path.write_bytes(original)

    with pytest.raises(ChatDeliveryJournalCorruption):
        ChatDeliveryJournal(path)

    assert path.read_bytes() == original


def test_cached_factory_does_not_resolve_away_symlink_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.runtime import chat_delivery_journal as journal_module

    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    link = tmp_path / "chat.sqlite3"
    link.symlink_to(target)
    monkeypatch.setenv("AURA_CHAT_DELIVERY_DB", str(link))
    journal_module.reset_chat_delivery_journals_for_test()

    with pytest.raises(ChatDeliveryJournalCorruption):
        journal_module.get_chat_delivery_journal()


@pytest.mark.parametrize(
    ("principal", "session_id"),
    (
        ("p" * 241, "session"),
        ("principal", "s" * 241),
        ("principal", "session\x00collision"),
    ),
)
def test_identity_rejects_values_that_could_alias_or_poison_storage(
    principal: str,
    session_id: str,
) -> None:
    with pytest.raises(ValueError):
        DeliveryIdentity.create(
            principal=principal,
            session_id=session_id,
            idempotency_key="safe-key",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("stale_after_s", float("nan")),
        ("retention_s", float("inf")),
        ("poll_interval_s", 0.0),
        ("busy_timeout_s", -1.0),
        ("max_rows", True),
    ),
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError):
        ChatDeliveryJournal(tmp_path / f"{field}.sqlite3", **kwargs)


@pytest.mark.asyncio
async def test_concurrent_same_request_executes_once_then_replays(
    journal: ChatDeliveryJournal,
) -> None:
    identity = _identity()
    digest = _request_hash()
    owner = await journal.reserve(identity, digest, wait_timeout_s=0)
    assert owner.kind is AdmissionKind.EXECUTE

    waiter = asyncio.create_task(journal.reserve(identity, digest, wait_timeout_s=1.0))
    await asyncio.sleep(0.03)
    assert not waiter.done()

    terminal = await journal.finalize(
        owner,
        state=DeliveryState.COMPLETED,
        http_status=200,
        response={"response": "hello", "status": "ok"},
    )
    replay = await waiter

    assert replay.kind is AdmissionKind.REPLAY
    assert replay.record.turn_id == owner.record.turn_id == terminal.turn_id
    assert replay.record.response == {"response": "hello", "status": "ok"}


@pytest.mark.asyncio
async def test_same_key_different_payload_is_rejected_without_wait(
    journal: ChatDeliveryJournal,
) -> None:
    identity = _identity()
    owner = await journal.reserve(identity, _request_hash("first"), wait_timeout_s=0)

    mismatch = await journal.reserve(
        identity,
        _request_hash("different"),
        wait_timeout_s=5.0,
    )

    assert owner.kind is AdmissionKind.EXECUTE
    assert mismatch.kind is AdmissionKind.MISMATCH
    assert mismatch.record.turn_id == owner.record.turn_id


@pytest.mark.asyncio
async def test_same_key_isolated_by_principal_and_session(
    journal: ChatDeliveryJournal,
) -> None:
    digest = _request_hash()
    owner = await journal.reserve(_identity(), digest, wait_timeout_s=0)
    other_principal = await journal.reserve(
        _identity(principal="paired:device-2"),
        digest,
        wait_timeout_s=0,
    )
    other_session = await journal.reserve(
        _identity(session_id="session-2"),
        digest,
        wait_timeout_s=0,
    )

    assert owner.kind is AdmissionKind.EXECUTE
    assert other_principal.kind is AdmissionKind.EXECUTE
    assert other_session.kind is AdmissionKind.EXECUTE
    assert (
        len(
            {
                owner.record.turn_id,
                other_principal.record.turn_id,
                other_session.record.turn_id,
            }
        )
        == 3
    )


@pytest.mark.asyncio
async def test_expired_running_owner_becomes_ambiguous_not_reexecuted(
    tmp_path: Path,
) -> None:
    short = ChatDeliveryJournal(
        tmp_path / "chat.sqlite3",
        stale_after_s=0.05,
        poll_interval_s=0.01,
    )
    identity = _identity()
    digest = _request_hash()
    owner = await short.reserve(identity, digest, wait_timeout_s=0)
    await asyncio.sleep(0.07)

    recovered = await short.reserve(identity, digest, wait_timeout_s=0)

    assert recovered.kind is AdmissionKind.REPLAY
    assert recovered.record.state is DeliveryState.AMBIGUOUS
    assert recovered.record.http_status == 409
    assert recovered.record.response is not None
    assert recovered.record.response["status"] == "delivery_ambiguous"
    with pytest.raises(ChatDeliveryFenceLost):
        await short.finalize(
            owner,
            state=DeliveryState.COMPLETED,
            http_status=200,
            response={"response": "late"},
        )


@pytest.mark.asyncio
async def test_status_read_reconciles_expired_running_owner(
    tmp_path: Path,
) -> None:
    short = ChatDeliveryJournal(
        tmp_path / "chat.sqlite3",
        stale_after_s=0.05,
        poll_interval_s=0.01,
    )
    identity = _identity()
    owner = await short.reserve(identity, _request_hash(), wait_timeout_s=0)
    await asyncio.sleep(0.07)

    status = await short.get(identity)

    assert status is not None
    assert status.state is DeliveryState.AMBIGUOUS
    assert status.turn_id == owner.record.turn_id
    assert status.http_status == 409


@pytest.mark.asyncio
async def test_compaction_fences_expired_owner_from_another_turn(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chat.sqlite3"
    short = ChatDeliveryJournal(
        path,
        stale_after_s=0.05,
        abandon_after_s=30.0,
        poll_interval_s=0.01,
    )
    abandoned_identity = _identity("abandoned-turn")
    owner = await short.reserve(
        abandoned_identity,
        _request_hash("abandoned"),
        wait_timeout_s=0,
    )
    await asyncio.sleep(0.07)

    # Make the next unrelated reservation run compaction. The expired running
    # row must be fenced by its lease, not left active until abandon_after_s.
    with contextlib.closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.execute(
                "UPDATE chat_delivery_meta SET value='0' WHERE key='last_compaction_at'"
            )
    unrelated = await short.reserve(
        _identity("unrelated-turn"),
        _request_hash("new request"),
        wait_timeout_s=0,
    )
    recovered = await short.get(abandoned_identity)

    assert unrelated.kind is AdmissionKind.EXECUTE
    assert recovered is not None
    assert recovered.state is DeliveryState.AMBIGUOUS
    assert recovered.http_status == 409
    assert recovered.response is not None
    assert recovered.response["status"] == "delivery_ambiguous"
    with pytest.raises(ChatDeliveryFenceLost):
        await short.finalize(
            owner,
            state=DeliveryState.COMPLETED,
            http_status=200,
            response={"response": "late"},
        )


@pytest.mark.asyncio
async def test_terminal_receipt_survives_journal_recreation(tmp_path: Path) -> None:
    path = tmp_path / "chat.sqlite3"
    first = ChatDeliveryJournal(path)
    identity = _identity()
    digest = _request_hash()
    owner = await first.reserve(identity, digest, wait_timeout_s=0)
    await first.finalize(
        owner,
        state=DeliveryState.COMPLETED,
        http_status=200,
        response={"response": "durable", "status": "ok"},
    )

    second = ChatDeliveryJournal(path)
    replay = await second.reserve(identity, digest, wait_timeout_s=0)

    assert replay.kind is AdmissionKind.REPLAY
    assert replay.record.turn_id == owner.record.turn_id
    assert replay.record.response == {"response": "durable", "status": "ok"}


@pytest.mark.asyncio
async def test_running_progress_is_durable_fenced_and_public(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chat.sqlite3"
    first = ChatDeliveryJournal(path)
    identity = _identity("progress-turn")
    owner = await first.reserve(identity, _request_hash(), wait_timeout_s=0)

    updated = await first.publish_progress(
        owner,
        phase="executing",
        message="Completed the first verified step.",
        details={"steps_completed": 1, "steps_total": 3, "tool": "desktop_task"},
    )

    assert updated.progress_sequence == 1
    assert updated.progress is not None
    assert updated.progress["phase"] == "executing"
    assert updated.progress["details"]["steps_completed"] == 1
    public = updated.public_status()
    assert public["progress"]["message"] == "Completed the first verified step."

    reopened = ChatDeliveryJournal(path)
    durable = await reopened.get(identity)
    assert durable is not None
    assert durable.progress == updated.progress

    await reopened.finalize(
        owner,
        state=DeliveryState.COMPLETED,
        http_status=200,
        response={"response": "done", "status": "ok"},
    )
    with pytest.raises(ChatDeliveryFenceLost):
        await reopened.publish_progress(
            owner,
            phase="executing",
            message="This stale owner must not publish.",
        )


@pytest.mark.asyncio
async def test_generation_progress_keeps_owner_across_callback_contexts(journal) -> None:
    from core.runtime.chat_delivery_progress import (
        bind_chat_delivery_progress,
        capture_generation_progress,
    )

    owner = await journal.reserve(_identity("generation-progress"), _request_hash(), wait_timeout_s=0)
    with bind_chat_delivery_progress(journal, owner):
        report = capture_generation_progress()
    assert report is not None
    assert capture_generation_progress() is None
    await asyncio.to_thread(report, phase="prefill", completed=128, total=256)
    for _ in range(100):
        record = await journal.get(owner.record.identity)
        if record.progress_sequence:
            break
        await asyncio.sleep(0.01)
    assert record.progress["phase"] == "prefill"
    assert record.progress["details"]["completed_tokens"] == 128
    for count in range(129, 256):
        report(phase="prefill", completed=count, total=256)
    await asyncio.sleep(0.05)
    record = await journal.get(owner.record.identity)
    assert record.progress_sequence == 1
    await journal.finalize(owner, state=DeliveryState.COMPLETED, http_status=200,
                           response={"response": "done"})
    report(phase="generating", completed=1)
    await asyncio.sleep(0.05)
    record = await journal.get(owner.record.identity)
    assert record.response == {"response": "done"}
    assert record.progress_sequence == 1


def test_version_one_journal_migrates_progress_columns(tmp_path: Path) -> None:
    path = tmp_path / "chat.sqlite3"
    ChatDeliveryJournal(path)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.execute(
                "UPDATE chat_delivery_meta SET value='1' WHERE key='schema_version'"
            )
            # Simulate the v1 table shape rather than merely relabeling v2.
            conn.execute("ALTER TABLE chat_deliveries DROP COLUMN progress_hash")
            conn.execute("ALTER TABLE chat_deliveries DROP COLUMN progress_json")
            conn.execute("ALTER TABLE chat_deliveries DROP COLUMN progress_at")
            conn.execute("ALTER TABLE chat_deliveries DROP COLUMN progress_sequence")

    migrated = ChatDeliveryJournal(path)
    with contextlib.closing(sqlite3.connect(migrated.db_path)) as conn:
        version = conn.execute(
            "SELECT value FROM chat_delivery_meta WHERE key='schema_version'"
        ).fetchone()
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(chat_deliveries)").fetchall()
        }

    assert version == ("2",)
    assert {
        "progress_sequence",
        "progress_at",
        "progress_json",
        "progress_hash",
    } <= columns


@pytest.mark.asyncio
async def test_tampered_terminal_receipt_fails_closed(
    journal: ChatDeliveryJournal,
) -> None:
    identity = _identity()
    owner = await journal.reserve(identity, _request_hash(), wait_timeout_s=0)
    await journal.finalize(
        owner,
        state=DeliveryState.COMPLETED,
        http_status=200,
        response={"response": "sealed"},
    )
    with contextlib.closing(sqlite3.connect(journal.db_path)) as conn:
        with conn:
            conn.execute(
                "UPDATE chat_deliveries SET response_hash=? WHERE turn_id=?",
                ("0" * 64, owner.record.turn_id),
            )

    with pytest.raises(ChatDeliveryJournalCorruption):
        await journal.get(identity)


@pytest.mark.asyncio
async def test_active_capacity_fails_closed_instead_of_growing_unbounded(
    tmp_path: Path,
) -> None:
    bounded = ChatDeliveryJournal(tmp_path / "chat.sqlite3", max_rows=10)
    digest = _request_hash()
    for index in range(10):
        admission = await bounded.reserve(
            _identity(f"turn-{index}"),
            digest,
            wait_timeout_s=0,
        )
        assert admission.kind is AdmissionKind.EXECUTE

    with pytest.raises(ChatDeliveryJournalUnavailable):
        await bounded.reserve(_identity("turn-overflow"), digest, wait_timeout_s=0)


@pytest.mark.asyncio
async def test_capacity_evicts_oldest_terminal_receipt_before_rejecting_new_work(
    tmp_path: Path,
) -> None:
    bounded = ChatDeliveryJournal(tmp_path / "chat.sqlite3", max_rows=10)
    digest = _request_hash()
    first_identity = _identity("turn-0")
    for index in range(10):
        identity = _identity(f"turn-{index}")
        admission = await bounded.reserve(identity, digest, wait_timeout_s=0)
        await bounded.finalize(
            admission,
            state=DeliveryState.COMPLETED,
            http_status=200,
            response={"response": str(index)},
        )

    replacement = await bounded.reserve(
        _identity("turn-replacement"),
        digest,
        wait_timeout_s=0,
    )

    assert replacement.kind is AdmissionKind.EXECUTE
    assert await bounded.get(first_identity) is None
    with contextlib.closing(sqlite3.connect(bounded.db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chat_deliveries").fetchone()[0] == 10


def _patch_route_identity(monkeypatch: pytest.MonkeyPatch, journal: ChatDeliveryJournal) -> None:
    from interface.routes import chat as chat_mod

    patch_chat_lane(monkeypatch, "get_chat_delivery_journal", lambda: journal)
    patch_chat_lane(monkeypatch, "request_access_profile",
        lambda _request: {"surface": "owner", "conversation_only": False},
    )
    monkeypatch.setattr(
        _chat_delivery,
        "_authenticated_chat_principal",
        lambda _request: "owner:bryan",
    )
    monkeypatch.setattr(
        _chat_delivery,
        "_observe_authenticated_chat_turn",
        lambda _request, _body: "owner:bryan",
    )
    monkeypatch.setattr(
        _chat_delivery,
        "_attach_http_chat_delivery_receipt",
        lambda *_args, **_kwargs: None,
    )


@pytest.mark.asyncio
async def test_pending_answer_ack_follows_rendered_terminal_receipt(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from core.conversation import chat_preflight
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    events: list[str] = []
    monkeypatch.setattr(
        _chat_delivery,
        "_attach_http_chat_delivery_receipt",
        lambda *_args, **_kwargs: events.append("receipt-attached"),
    )
    monkeypatch.setattr(
        chat_preflight,
        "acknowledge_delivery",
        lambda ids, *, delivery_owner, **_kwargs: (
            events.append(f"ack:{delivery_owner}:{','.join(ids)}") or len(ids)
        ),
    )
    monkeypatch.setattr(
        chat_preflight,
        "release_delivery_claims",
        lambda *_args, **_kwargs: events.append("released") or 1,
    )

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        chat_mod._CHAT_PENDING_DELIVERY_CLAIM.set(("turn-owner", ("pending-1",)))
        return JSONResponse({"response": "late answer delivered", "status": "ok"})

    response = await handler(
        body=chat_mod.ChatRequest(message="next", session_id="session-1"),
        request=_request("pending-ack-order"),
    )

    assert response.status_code == 200
    assert events == ["receipt-attached", "ack:turn-owner:pending-1"]


@pytest.mark.asyncio
async def test_pending_answer_claim_released_when_response_receipt_cannot_attach(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from core.conversation import chat_preflight
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    events: list[str] = []

    def fail_receipt(*_args, **_kwargs) -> None:
        events.append("receipt-failed")
        raise RuntimeError("receipt attachment failed")

    monkeypatch.setattr(_chat_delivery, "_attach_http_chat_delivery_receipt", fail_receipt)
    monkeypatch.setattr(
        chat_preflight,
        "acknowledge_delivery",
        lambda *_args, **_kwargs: events.append("acked") or 1,
    )
    monkeypatch.setattr(
        chat_preflight,
        "release_delivery_claims",
        lambda ids, *, delivery_owner, **_kwargs: (
            events.append(f"released:{delivery_owner}:{','.join(ids)}") or len(ids)
        ),
    )

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        chat_mod._CHAT_PENDING_DELIVERY_CLAIM.set(("turn-owner", ("pending-1",)))
        return JSONResponse({"response": "late answer", "status": "ok"})

    with pytest.raises(RuntimeError, match="receipt attachment failed"):
        await handler(
            body=chat_mod.ChatRequest(message="next", session_id="session-1"),
            request=_request("pending-release-on-failure"),
        )

    assert events == ["receipt-failed", "released:turn-owner:pending-1"]


@pytest.mark.asyncio
async def test_route_concurrent_duplicate_runs_handler_once(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return JSONResponse({"response": "one", "status": "ok"})

    request = _request("route-race")
    body = chat_mod.ChatRequest(message="hello", session_id="session-1")
    first_task = asyncio.create_task(handler(body=body, request=request))
    await entered.wait()
    second_task = asyncio.create_task(handler(body=body, request=request))
    await asyncio.sleep(0.03)
    assert calls == 1
    release.set()

    first, second = await asyncio.gather(first_task, second_task)
    first_payload = _payload(first)
    second_payload = _payload(second)
    assert calls == 1
    assert first_payload["turn_id"] == second_payload["turn_id"]
    assert first_payload["delivery_replayed"] is False
    assert second_payload["delivery_replayed"] is True


@pytest.mark.asyncio
async def test_route_binds_and_seals_durable_progress(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from core.runtime.chat_delivery_progress import report_chat_delivery_progress
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        published = await report_chat_delivery_progress(
            phase="executing",
            message="Reading the requested source.",
            details={"source_count": 1},
        )
        assert published is True
        return JSONResponse({"response": "done", "status": "ok"})

    body = chat_mod.ChatRequest(message="read it", session_id="session-1")
    response = await handler(body=body, request=_request("route-progress"))
    record = await journal.get(_identity("route-progress"))

    assert response.status_code == 200
    assert record is not None
    assert record.terminal is True
    assert record.progress_sequence == 3
    assert record.progress is not None
    assert record.progress["phase"] == "finalizing"
    assert record.progress["message"] == "Checking the result and its evidence before replying."


@pytest.mark.asyncio
async def test_route_cancellation_seals_ambiguous_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    entered = asyncio.Event()
    calls = 0

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.Event().wait()
        return JSONResponse({"response": "unreachable"})

    request = _request("route-cancel")
    body = chat_mod.ChatRequest(message="act", session_id="session-1")
    task = asyncio.create_task(handler(body=body, request=request))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    replay = await handler(body=body, request=request)
    payload = _payload(replay)
    assert calls == 1
    assert replay.status_code == 409
    assert payload["status"] == "delivery_ambiguous"
    assert payload["delivery_replayed"] is True


@pytest.mark.asyncio
async def test_route_same_key_different_message_returns_409_without_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    calls = 0

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        nonlocal calls
        calls += 1
        return JSONResponse({"response": body.message, "status": "ok"})

    request = _request("route-mismatch")
    first = chat_mod.ChatRequest(message="first", session_id="session-1")
    second = chat_mod.ChatRequest(message="second", session_id="session-1")
    assert (await handler(body=first, request=request)).status_code == 200

    mismatch = await handler(body=second, request=request)

    assert mismatch.status_code == 409
    assert _payload(mismatch)["status"] == "idempotency_payload_mismatch"
    assert calls == 1


@pytest.mark.asyncio
async def test_route_delivers_unusable_output_shape_in_band_for_real_chat(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        return None

    response = await handler(
        body=chat_mod.ChatRequest(message="hello", session_id="session-1"),
        request=_request("route-invalid-shape"),
    )
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["status"] == "chat_response_format_rejected"
    assert payload["response_confidence"] == "failed"
    assert payload["delivery_state"] == DeliveryState.FAILED.value


@pytest.mark.asyncio
async def test_route_keeps_strict_http_failure_for_benchmark_output_shape(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        return None

    response = await handler(
        body=chat_mod.ChatRequest(message="hello", session_id="session-1"),
        request=_request("route-invalid-shape-benchmark", benchmark=True),
    )
    payload = _payload(response)

    assert response.status_code == 500
    assert payload["status"] == "chat_response_format_rejected"
    assert payload["delivery_state"] == DeliveryState.FAILED.value


@pytest.mark.asyncio
async def test_route_strips_malformed_affordance_control_before_journaling(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        return JSONResponse(
            {
                "response": (
                    'I can show that. ⟦affordance:show_sketch prompt="an orca" Done.'
                ),
                "status": "ok",
                "response_confidence": "high",
                "live_turn_contract": {
                    "answer_delivery_proven": True,
                    "certification_complete": True,
                    "authentic_cognitive_reply": True,
                    "authored_generation_source_proven": True,
                    "authored_answer_completion_proven": True,
                    "final_requested_output_contract_evaluated": True,
                    "final_requested_output_contract_satisfied": True,
                    "final_requested_output_contract_proven": True,
                    "model_native_output": True,
                    "final_text_authorship": "model_native",
                    "post_generation_repair_applied": False,
                    "deterministic_repair_applied": False,
                    "authorship_replacement_applied": False,
                    "unreceipted_runtime_replacement": False,
                    "full_mind_missing_proofs": [],
                },
            }
        )

    body = chat_mod.ChatRequest(message="show me", session_id="session-1")
    request = _request("route-malformed-affordance")
    response = await handler(body=body, request=request)
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["response"] == "I can show that. Done."
    assert payload["status"] == "chat_affordance_control_sanitized"
    assert payload["response_confidence"] == "degraded"
    contract = payload["live_turn_contract"]
    assert contract["answer_delivery_proven"] is False
    assert contract["certification_complete"] is False
    assert contract["authored_generation_source_proven"] is False
    assert contract["final_requested_output_contract_proven"] is False
    assert contract["model_native_output"] is False
    assert contract["final_text_authorship"] == "delivery_boundary_rewrite"
    assert contract["authorship_replacement_applied"] is True
    assert contract["unreceipted_runtime_replacement"] is True
    assert contract["delivery_payload_mutated_after_proof"] is True
    assert "delivery_bytes_changed_after_proof" in contract["full_mind_missing_proofs"]
    assert contract["pre_mutation_response_sha256"] != contract[
        "delivered_response_sha256"
    ]

    replay = await handler(body=body, request=request)
    replay_payload = _payload(replay)
    assert replay_payload["response"] == "I can show that. Done."
    assert "affordance:" not in str(replay_payload).lower()


@pytest.mark.asyncio
async def test_route_fails_in_band_when_affordance_visibility_cannot_be_verified(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from core.cognition import expressive_affordances
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)

    def fail_sanitization(_text: str):
        raise RuntimeError("sanitizer unavailable")

    monkeypatch.setattr(
        expressive_affordances,
        "sanitize_affordance_control_syntax",
        fail_sanitization,
    )

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        return JSONResponse(
            {
                "response": "Draft ⟦affordance:show_sketch prompt=orca⟧",
                "status": "ok",
                "response_confidence": "high",
            }
        )

    response = await handler(
        body=chat_mod.ChatRequest(message="show me", session_id="session-1"),
        request=_request("route-affordance-sanitizer-failed"),
    )
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["status"] == "chat_affordance_visibility_unavailable"
    assert payload["response_confidence"] == "failed"
    assert "affordance:" not in str(payload).lower()
    assert payload["delivery_state"] == DeliveryState.FAILED.value


@pytest.mark.asyncio
async def test_route_settles_surface_with_the_actual_sanitized_payload(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from core.conversation import surface_delivery
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    delivered: list[str] = []
    monkeypatch.setattr(
        surface_delivery,
        "note_route_delivered",
        lambda text, **_identity: delivered.append(text),
    )

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        return JSONResponse(
            {
                "response": "Visible ⟦affordance:unknown target=private⟧ answer",
                "status": "ok",
            }
        )

    response = await handler(
        body=chat_mod.ChatRequest(message="show me", session_id="session-1"),
        request=_request("route-surface-settled"),
    )
    payload = _payload(response)

    assert payload["response"] == "Visible answer"
    assert delivered == [payload["response"]]


@pytest.mark.asyncio
async def test_route_settles_surface_after_terminal_receipt_substitution(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from core.conversation import surface_delivery
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    delivered: list[str] = []
    monkeypatch.setattr(
        surface_delivery,
        "note_route_delivered",
        lambda text, **_identity: delivered.append(text),
    )

    async def fail_terminal_receipt(*_args, **_kwargs):
        raise ChatDeliveryJournalUnavailable("terminal receipt unavailable")

    monkeypatch.setattr(_chat_delivery, "_finalize_chat_delivery", fail_terminal_receipt)

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        return JSONResponse({"response": "draft that must be withheld", "status": "ok"})

    response = await handler(
        body=chat_mod.ChatRequest(message="hello", session_id="session-1"),
        request=_request("route-terminal-substitution"),
    )
    payload = _payload(response)

    assert response.status_code == 503
    assert payload["status"] == "chat_delivery_terminal_unsealed"
    assert delivered == [payload["response"]]
    assert delivered != ["draft that must be withheld"]


def test_request_contract_binds_method_and_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from interface.routes import chat as chat_mod

    patch_chat_lane(monkeypatch, "request_access_profile",
        lambda _request: {"surface": "owner", "conversation_only": False},
    )
    body = chat_mod.ChatRequest(message="hello", session_id="session-1")
    _, chat_hash, _ = chat_mod._chat_delivery_request_contract(
        _request("same-key", path="/api/chat"),
        body,
        exact_principal="owner:bryan",
    )
    _, regenerate_hash, _ = chat_mod._chat_delivery_request_contract(
        _request("same-key", path="/api/chat/regenerate"),
        body,
        exact_principal="owner:bryan",
    )

    assert chat_hash != regenerate_hash


@pytest.mark.asyncio
async def test_authenticated_status_endpoint_returns_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    identity = _identity("status-key")
    owner = await journal.reserve(identity, _request_hash(), wait_timeout_s=0)
    terminal = await journal.finalize(
        owner,
        state=DeliveryState.COMPLETED,
        http_status=200,
        response={"response": "recovered", "status": "ok"},
    )

    response = await chat_mod.api_chat_delivery_status(
        "status-key",
        _request("ignored", method="GET"),
        session_id="session-1",
    )
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["turn_id"] == terminal.turn_id
    assert payload["result"] == {"response": "recovered", "status": "ok"}


@pytest.mark.asyncio
async def test_authenticated_status_endpoint_returns_live_progress(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    identity = _identity("status-progress")
    owner = await journal.reserve(identity, _request_hash(), wait_timeout_s=0)
    await journal.publish_progress(
        owner,
        phase="verifying",
        message="Checking two effect receipts.",
        details={"receipts": 2},
    )

    response = await chat_mod.api_chat_delivery_status(
        "status-progress",
        _request("ignored", method="GET"),
        session_id="session-1",
    )
    payload = _payload(response)

    assert response.status_code == 202
    assert payload["terminal"] is False
    assert payload["progress"]["phase"] == "verifying"
    assert payload["progress"]["details"] == {"receipts": 2}
