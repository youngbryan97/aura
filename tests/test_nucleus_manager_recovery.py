from __future__ import annotations

import pytest

from core.brain.llm.nucleus_manager import NucleusManager
from core.runtime.errors import get_degradation_tracker


def _manager_with_missing_models(tmp_path) -> NucleusManager:
    manager = NucleusManager()
    manager.bus = None
    manager.brainstem_path = str(tmp_path / "missing-brainstem")
    manager.cortex_path = str(tmp_path / "missing-cortex")
    return manager


@pytest.mark.asyncio
async def test_missing_model_path_marks_lane_unavailable(tmp_path):
    tracker = get_degradation_tracker()
    tracker.reset()
    manager = _manager_with_missing_models(tmp_path)
    manager.models.pop("brainstem", None)

    loaded = await manager.load_model("brainstem")

    assert loaded is False
    assert manager.models["brainstem"]["loaded"] is False
    assert "missing-brainstem" in manager.models["brainstem"]["last_error"]
    assert tracker.count("nucleus_manager", "warning") >= 1


@pytest.mark.asyncio
async def test_generate_text_missing_models_returns_deterministic_offline_marker(tmp_path):
    tracker = get_degradation_tracker()
    tracker.reset()
    manager = _manager_with_missing_models(tmp_path)

    text = await manager.generate_text_async("hello")

    assert text == "[NUCLEUS ERROR] Internal inference offline."
    assert manager.models["cortex"]["loaded"] is False
    assert manager.models["brainstem"]["loaded"] is False
    assert tracker.count("nucleus_manager", "degraded") >= 1


@pytest.mark.asyncio
async def test_generate_stream_missing_brainstem_entry_returns_offline_marker(tmp_path):
    tracker = get_degradation_tracker()
    tracker.reset()
    manager = _manager_with_missing_models(tmp_path)
    manager.models.pop("brainstem", None)

    chunks = [
        chunk
        async for chunk in manager.generate_stream_async("status", origin="health_monitor")
    ]

    assert chunks == ["[NUCLEUS ERROR] Internal inference offline."]
    assert "brainstem" in manager.models
    assert manager.models["brainstem"]["loaded"] is False
    assert tracker.count("nucleus_manager", "degraded") >= 1


@pytest.mark.asyncio
async def test_unload_models_clears_loaded_entries_even_when_cache_clear_is_unavailable(tmp_path):
    manager = _manager_with_missing_models(tmp_path)
    manager.models["cortex"].update({
        "model": object(),
        "tokenizer": object(),
        "loaded": True,
        "cache": object(),
    })

    await manager.unload_models()

    assert manager.models["cortex"]["model"] is None
    assert manager.models["cortex"]["tokenizer"] is None
    assert manager.models["cortex"]["loaded"] is False
    assert manager.models["cortex"]["cache"] is None
