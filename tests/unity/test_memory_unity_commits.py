from __future__ import annotations

from core.memory.memory_facade import MemoryFacade


class _VectorRecorder:
    def __init__(self):
        self.calls = []

    def add_memory(self, content, metadata):
        self.calls.append((content, dict(metadata)))
        return True


def test_memory_metadata_carries_unity_fields(monkeypatch):
    facade = MemoryFacade()
    recorder = _VectorRecorder()
    facade._vector = recorder
    monkeypatch.setattr(
        facade,
        "_current_unity_metadata",
        lambda: {
            "unity_id": "unity_123",
            "unity_level": "strained",
            "unity_score": 0.61,
            "fragmentation_score": 0.39,
            "unity_memory_commit_mode": "qualified",
            "unity_suppressed_draft_ids": ["draft_b"],
            "unity_ownership_confidence": 0.84,
        },
    )

    ok = __import__("asyncio").run(facade.add_memory("remember this carefully", {"importance": 0.8}))

    assert ok is True
    assert recorder.calls
    _content, metadata = recorder.calls[0]
    assert metadata["unity_id"] == "unity_123"
    assert metadata["unity_memory_commit_mode"] == "qualified"
    assert metadata["unity_suppressed_draft_ids"] == ["draft_b"]


def test_memory_write_defers_when_unity_requires_it(monkeypatch):
    facade = MemoryFacade()
    recorder = _VectorRecorder()
    facade._vector = recorder
    monkeypatch.setattr(
        facade,
        "_current_unity_metadata",
        lambda: {
            "unity_id": "unity_456",
            "unity_memory_commit_mode": "defer",
        },
    )

    ok = __import__("asyncio").run(facade.add_memory("do not store this as settled", {}))

    assert ok is False
    assert recorder.calls == []
    assert facade._last_add_memory_status["reason"] == "unity_memory_defer"
