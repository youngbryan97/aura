from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from chat_lane_support import patch_chat_lane
import interface.routes.chat_memory_state as _chat_memory_state


@pytest.mark.asyncio
async def test_bounded_chat_blocking_work_does_not_own_event_loop():
    from interface.routes import chat as chat_routes

    started = threading.Event()
    release = threading.Event()

    def blocking() -> str:
        started.set()
        release.wait(2.0)
        return "late"

    operation = asyncio.create_task(
        chat_routes._await_bounded_chat_blocking(
            blocking,
            timeout_s=0.05,
            operation_name="test_blocking",
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 1.0)
        loop_advanced = False

        async def tick() -> None:
            nonlocal loop_advanced
            await asyncio.sleep(0)
            loop_advanced = True

        await tick()
        assert loop_advanced is True
        with pytest.raises(TimeoutError):
            await operation
    finally:
        release.set()
        pending = list(chat_routes._chat_blocking_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_bounded_deterministic_work_can_finish_in_turn_during_grace():
    from interface.routes import chat as chat_routes

    def slightly_late() -> str:
        time.sleep(0.07)
        return "complete deterministic result"

    result = await chat_routes._await_bounded_chat_blocking(
        slightly_late,
        timeout_s=0.02,
        completion_grace_s=0.2,
        operation_name="recoverable_deterministic_test",
    )

    assert result == "complete deterministic result"
    assert not chat_routes._chat_blocking_tasks


@pytest.mark.asyncio
async def test_hard_timed_out_blocking_work_remains_supervised_until_exit():
    from interface.routes import chat as chat_routes

    started = threading.Event()
    release = threading.Event()

    def late() -> str:
        started.set()
        release.wait(1.0)
        return "too late for this turn"

    try:
        with pytest.raises(TimeoutError):
            await chat_routes._await_bounded_chat_blocking(
                late,
                timeout_s=0.02,
                operation_name="supervised_timeout_test",
            )
        assert started.is_set()
        assert chat_routes._chat_blocking_tasks
    finally:
        release.set()
        pending = list(chat_routes._chat_blocking_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    assert not chat_routes._chat_blocking_tasks


@pytest.mark.asyncio
async def test_bounded_chat_work_rejects_saturation_without_running_operation(monkeypatch):
    from interface.routes import chat as chat_routes

    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(_chat_memory_state, "_chat_blocking_slots", slots)
    called = False

    def operation() -> None:
        nonlocal called
        called = True

    try:
        with pytest.raises(chat_routes._ChatBlockingBudgetSaturatedError):
            await chat_routes._await_bounded_chat_blocking(
                operation,
                timeout_s=0.2,
                operation_name="saturated_test",
            )
    finally:
        slots.release()

    assert called is False


@pytest.mark.asyncio
async def test_live_mind_collection_timeout_is_explicit_and_fail_closed(monkeypatch):
    from interface.routes import chat as chat_routes

    started = threading.Event()
    release = threading.Event()

    def blocking_collector(**_kwargs):
        started.set()
        release.wait(2.0)
        return {"required_subsystems_ok": True}

    monkeypatch.setattr(chat_routes, "_build_live_mind_context_payload", blocking_collector)
    monkeypatch.setattr(chat_routes, "_CHAT_LIVE_MIND_COLLECTION_TIMEOUT_S", 0.05)
    try:
        result = await chat_routes._collect_live_mind_context_payload(
            user_message="Are you here?",
            lane={"conversation_ready": True},
            require_engine=True,
        )
    finally:
        release.set()

    assert started.is_set()
    assert result["collection_status"] == "unavailable"
    assert result["required_subsystems_ok"] is False
    assert result["must_answer_from_full_mind_path"] is True


@pytest.mark.asyncio
async def test_recent_response_symbolic_audit_is_supervised_off_loop(monkeypatch):
    from interface.routes import chat as chat_routes

    started = threading.Event()
    release = threading.Event()

    def blocking_audit(_text: str) -> None:
        started.set()
        release.wait(2.0)

    patch_chat_lane(monkeypatch, "_audit_recent_response_reasoning_sync", blocking_audit)
    monkeypatch.setattr(chat_routes, "_CHAT_REASONING_AUDIT_TIMEOUT_S", 0.05)
    chat_routes._reasoning_audit_tasks.clear()
    try:
        chat_routes._record_recent_response("A fresh measured answer.", "What happened?")
        assert await asyncio.to_thread(started.wait, 1.0)
        assert chat_routes._reasoning_audit_tasks
        await asyncio.sleep(0.08)
    finally:
        release.set()
        tasks = list(chat_routes._reasoning_audit_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        chat_routes._reasoning_audit_tasks.clear()


def test_repo_probe_refuses_oversized_file_without_partial_claim(tmp_path, monkeypatch):
    from core.conversation import demo_support
    from interface.routes import chat as chat_routes

    target = tmp_path / "large.py"
    target.write_bytes(b"x" * (chat_routes._REPO_PROBE_MAX_BYTES + 1))
    monkeypatch.setattr(demo_support, "_resolve_target_path", lambda _target: target)

    result = chat_routes._read_repo_probe_reply(
        "Read large.py and tell me how many lines it has."
    )

    assert result is not None
    assert result["status"] == "repo_probe_too_large"
    assert "did not pretend" in result["reply"]


def test_memory_pin_recall_reads_bounded_tail(tmp_path, monkeypatch):
    from interface.routes import chat as chat_routes

    # This test went stale in three ways behind deliberate refactors, and the
    # property it guards — recall reads a bounded TAIL, not the whole file —
    # is unchanged by both.
    #
    # 1. The read bound moved out of the route and into the ledger that owns
    #    it, so it is read from there now.
    # 2. The ledger became encrypted-records-only, so a plaintext v2 row can
    #    no longer be recalled and the cipher has to be supplied the way
    #    every other session-pin test supplies it (a fixed test key, rather
    #    than a macOS Keychain that does not exist in CI).
    from core.memory.session_pin_cipher import SessionPinCipher
    from core.memory.session_pin_ledger import SESSION_PIN_LEDGER_MAX_BYTES

    cipher = SessionPinCipher(b"k" * 32)
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_cipher", lambda: cipher)

    # 3. The ledger pins its own filename (a governed write-scope
    #    constraint), so the fixture has to use the real one.
    ledger = tmp_path / "session_memory_pins.jsonl"
    prefix = b"not-json\n" * (
        SESSION_PIN_LEDGER_MAX_BYTES // len(b"not-json\n") + 10
    )
    record = cipher.seal(
        content="orcas remain my favorite animal",
        source="Remember this",
        timestamp="2026-08-08T00:00:00Z",
        session_id="owner-session",
        principal_id="owner:test",
        principal_surface="owner",
    )
    ledger.write_bytes(prefix + json.dumps(record).encode() + b"\n")
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_ledger_path", lambda: ledger)

    recalled = chat_routes._recall_session_memory_pin_from_ledger(
        session_id="owner-session",
        principal_id="owner:test",
        principal_surface="owner",
    )

    assert recalled is not None
    assert recalled["content"] == "orcas remain my favorite animal"
    assert recalled["storage"] == "durable"


def test_pending_queue_reads_only_bounded_tail(tmp_path, monkeypatch):
    from core.conversation import chat_preflight

    queue = tmp_path / "pending.jsonl"
    monkeypatch.setattr(chat_preflight, "MAX_PENDING_QUEUE_BYTES", 512)
    valid = {
        "session_id": "current",
        "user_message": "question",
        "queued_at": time.time(),
        "reason": "timeout",
        "answered": True,
        "answer_text": "finished",
        "answered_at": time.time(),
    }
    queue.write_bytes(b"old-partial-data" * 100 + b"\n" + json.dumps(valid).encode() + b"\n")

    records = chat_preflight._read_all(queue)

    assert len(records) == 1
    assert records[0]["session_id"] == "current"


def test_expired_resume_deadline_does_not_consume_answer(tmp_path):
    from core.conversation import chat_preflight

    queue = tmp_path / "pending.jsonl"
    chat_preflight.enqueue("owner", "question", path=queue)
    assert chat_preflight.answer_pending("owner", "finished", path=queue) is True

    delivered = chat_preflight.consume_for_session(
        "owner",
        path=queue,
        deadline_monotonic=time.monotonic() - 1.0,
    )

    assert delivered == []
    assert any(record.get("answered") for record in chat_preflight._read_all(queue))


def test_export_record_budget_does_not_scan_beyond_item_limit():
    from interface.routes import chat as chat_routes

    consumed: list[int] = []

    def records():
        for index in range(1000):
            consumed.append(index)
            yield {"index": index}

    exported, receipt = chat_routes._bounded_export_records(
        records(),
        max_items=3,
        total_chars=10_000,
    )

    assert exported == [{"index": 0}, {"index": 1}, {"index": 2}]
    assert consumed == [0, 1, 2]
    assert receipt["source_items"] == 3


def test_export_is_truthful_when_one_record_is_not_json_serializable():
    from interface.routes import chat as chat_routes

    circular: dict[str, object] = {"value": float("nan")}
    circular["self"] = circular

    exported, receipt = chat_routes._bounded_export_records(
        [circular, {"valid": True}],
        max_items=2,
        total_chars=10_000,
    )

    assert exported[0]["export_unserializable"] is True
    assert exported[1] == {"valid": True}
    assert receipt["truncated_items"] == 1


@pytest.mark.asyncio
async def test_full_export_collects_memory_and_goals_off_loop(monkeypatch):
    from interface.routes import chat as chat_routes

    main_thread = threading.get_ident()
    called_threads: list[int] = []

    class Episodic:
        def get_recent(self, *, limit: int):
            called_threads.append(threading.get_ident())
            return [{"kind": "episode", "limit": limit}]

    class Semantic:
        def search(self, *, query: str, limit: int):
            called_threads.append(threading.get_ident())
            return [{"kind": "semantic", "query": query, "limit": limit}]

    class Goals:
        def get_active_goals(self):
            called_threads.append(threading.get_ident())
            return [SimpleNamespace(name="finish closeout")]

    services = {
        "episodic_memory": Episodic(),
        "semantic_memory": Semantic(),
        "goal_manager": Goals(),
    }
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )
    original_log = list(chat_routes._conversation_log)
    chat_routes._conversation_log[:] = [
        {"exchange_id": "e1", "role": "user", "content": "hello"}
    ]
    try:
        response = await chat_routes.api_export(SimpleNamespace(), None)
    finally:
        chat_routes._conversation_log[:] = original_log

    payload = json.loads(response.body)
    assert len(called_threads) == 3
    assert all(thread_id != main_thread for thread_id in called_threads)
    assert payload["episodic_memories"][0]["kind"] == "episode"
    assert payload["semantic_memories"][0]["kind"] == "semantic"
    assert "finish closeout" in payload["active_goals"][0]
    assert all(
        receipt["status"] == "complete"
        for receipt in payload["export_receipts"].values()
    )
