from __future__ import annotations

import asyncio
import sys
import types

from core.conversation import persistence as persistence_module
from core.conversation.persistence import ConversationPersistence


def _install_event_bus(monkeypatch, bus):
    module = types.ModuleType("core.event_bus")
    module.get_event_bus = lambda: bus
    monkeypatch.setitem(sys.modules, "core.event_bus", module)


def test_conversation_persistence_records_turn_and_publishes_threadsafe(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []

    class Bus:
        def publish_threadsafe(self, topic, payload):
            published.append((topic, payload))

    _install_event_bus(monkeypatch, Bus())

    store = ConversationPersistence(tmp_path / "conversations.db")
    session_id = store.start_session({"non_json": object()})
    turn_id = store.record_turn(
        "user\x00",
        "hello from persistence",
        origin="text",
        cid="cid-123",
    )

    history = store.get_session_history(session_id, limit="10000")

    assert turn_id
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello from persistence"
    assert len(history) == 1
    assert published[0][0] == "turn_recorded"
    assert published[0][1]["session_id"] == session_id
    assert published[0][1]["turn_id"] == turn_id
    assert published[0][1]["content_chars"] == len("hello from persistence")


def test_conversation_persistence_async_publish_is_scheduled(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []
    scheduled: list[str] = []

    class Bus:
        async def publish(self, topic, payload):
            await asyncio.sleep(0)
            published.append((topic, payload))

    class Tracker:
        def create_task(self, coro, name=None):
            scheduled.append(name or "")
            return asyncio.create_task(coro)

    _install_event_bus(monkeypatch, Bus())
    monkeypatch.setattr(persistence_module, "get_task_tracker", lambda: Tracker())

    async def scenario():
        store = ConversationPersistence(tmp_path / "async-conversations.db")
        turn_id = store.record_turn("aura", "scheduled event", cid="cid-async")
        await asyncio.sleep(0.01)
        return turn_id

    turn_id = asyncio.run(scenario())

    assert turn_id
    assert scheduled == ["conversation.turn_recorded.publish"]
    assert published[0][0] == "turn_recorded"
    assert published[0][1]["cid"] == "cid-async"


def test_conversation_persistence_scheduler_failure_records_receipt(monkeypatch, tmp_path):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    class TaskSpec:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Scheduler:
        async def register(self, _spec):
            self.attempted = True
            raise RuntimeError("scheduler unavailable")

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    scheduler_module = types.ModuleType("core.scheduler")
    scheduler_module.TaskSpec = TaskSpec
    scheduler_module.scheduler = Scheduler()
    monkeypatch.setitem(sys.modules, "core.scheduler", scheduler_module)
    monkeypatch.setattr(persistence_module, "record_degradation", record_degradation)

    store = ConversationPersistence(tmp_path / "scheduler-conversations.db")
    asyncio.run(store.on_start_async())

    assert store.get_retention_status()["last_persist_error_at"] > 0
    assert recorded
    assert recorded[0][0] == "persistence"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True
    assert "scheduled conversation pruning" in str(recorded[0][2]["action"])
