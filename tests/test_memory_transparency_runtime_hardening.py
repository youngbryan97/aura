import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_dev_mode_callback_runtime_errors_degrade_but_invariants_surface(monkeypatch):
    import core.transparency.dev_mode as dev_mode_module
    from core.transparency.dev_mode import DevMode

    recorded: list[tuple[str, str]] = []
    runtime_events: list[str] = []
    invariant_events: list[str] = []

    monkeypatch.setattr(
        dev_mode_module,
        "record_degradation",
        lambda component, error: recorded.append((component, str(error))),
    )

    def runtime_callback(event_type, _data):
        runtime_events.append(event_type)
        raise RuntimeError("websocket disconnected")

    def invariant_callback(event_type, _data):
        invariant_events.append(event_type)
        raise AssertionError("callback invariant broke")

    runtime_dev_mode = DevMode()
    await runtime_dev_mode.register_callback(runtime_callback)
    await runtime_dev_mode._emit_event("tool_trace", {"ok": True})

    assert runtime_events == ["tool_trace"]
    assert recorded == [("dev_mode.callback", "websocket disconnected")]

    invariant_dev_mode = DevMode()
    await invariant_dev_mode.register_callback(invariant_callback)
    with pytest.raises(AssertionError, match="callback invariant broke"):
        await invariant_dev_mode._emit_event("tool_trace", {"ok": True})
    assert invariant_events == ["tool_trace"]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    {"ok": False, "status": "deferred", "reason": "recent_user_287"},
    {"ok": False, "status": "deferred_by_executive", "deferred": True},
    {"ok": False, "error": "background_deferred:foreground_generation_active"},
])
async def test_dev_mode_represents_policy_deferral_without_failure_marker(caplog, result):
    from core.transparency.dev_mode import DevMode

    dev_mode = DevMode()
    trace = await dev_mode.record_tool_execution(
        "auto_refactor",
        {"path": "."},
        origin="autonomy",
    )

    with caplog.at_level(logging.INFO, logger="core.transparency.dev_mode"):
        await dev_mode.complete_tool_execution(
            trace,
            result,
            0.4,
        )

    assert trace.status == "deferred"
    assert trace.error is None
    messages = [record.getMessage() for record in caplog.records]
    assert any("Tool Deferred: auto_refactor" in message for message in messages)
    assert not any("❌ Tool Result: auto_refactor" in message for message in messages)


def test_vector_memory_chroma_writer_stops_and_rejects_after_close(monkeypatch, tmp_path):
    import core.memory.vector_memory as vector_module
    from core.memory.vector_memory import VectorMemory

    class FakeCollection:
        def count(self):
            return 0

        def upsert(self, *, ids, documents, metadatas):
            assert len(ids) == len(documents) == len(metadatas)

    class FakeClient:
        def get_or_create_collection(self, **_kwargs):
            return FakeCollection()

    monkeypatch.setattr(vector_module, "_CHROMA_AVAILABLE", True)
    monkeypatch.setattr(
        vector_module,
        "chromadb",
        SimpleNamespace(PersistentClient=lambda **_kwargs: FakeClient()),
        raising=False,
    )
    monkeypatch.setattr(
        vector_module,
        "ChromaSettings",
        lambda **_kwargs: SimpleNamespace(),
        raising=False,
    )

    memory = VectorMemory(collection_name="test", persist_directory=str(tmp_path))
    assert memory._upsert_thread is not None
    assert memory._upsert_thread.is_alive()

    memory.close(timeout_s=1.0)

    assert not memory._upsert_thread.is_alive()
    assert memory.add_memory("should not enqueue after close", _id="closed") is False


def test_id_rag_and_vector_memory_have_explicit_writer_stop_contracts() -> None:
    vector_source = Path("core/memory/vector_memory.py").read_text(encoding="utf-8")
    id_rag_source = Path("core/identity/id_rag.py").read_text(encoding="utf-8")

    assert "while True" not in vector_source
    assert "while True" not in id_rag_source
    assert "def close(" in vector_source
    assert "_UPSERT_QUEUE_STOP" in vector_source
