from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.orchestrator.mixins.context_streaming import ContextStreamingMixin
from core.resilience.dream_cycle import DreamCycle
from core.safe_mode import apply_orchestrator_patches


class _DummyStreaming(ContextStreamingMixin):
    def __init__(self, history: list[dict[str, str]], runtime_config: dict[str, object] | None = None):
        self.conversation_history = history
        self.cognitive_engine = object()
        self._runtime_mode_config = runtime_config or {}


def test_apply_orchestrator_patches_is_configuration_only():
    async def _process_user_input(message: str, origin: str = "user") -> str:
        return f"{origin}:{message}"

    async def _prune_history_async() -> None:
        return None

    async def _consolidate_long_term_memory() -> None:
        return None

    orchestrator = SimpleNamespace(
        kernel=SimpleNamespace(volition_level=2),
        conversation_history=[{"role": "user", "content": f"m{i}"} for i in range(120)],
        cognitive_engine=SimpleNamespace(singularity_factor=3.0),
        singularity_monitor=SimpleNamespace(acceleration_factor=4.0),
        process_user_input=_process_user_input,
        _prune_history_async=_prune_history_async,
        _consolidate_long_term_memory=_consolidate_long_term_memory,
    )

    original_process = orchestrator.process_user_input
    original_prune = orchestrator._prune_history_async
    original_consolidate = orchestrator._consolidate_long_term_memory

    apply_orchestrator_patches(orchestrator, safe_mode=True)

    assert orchestrator.process_user_input is original_process
    assert orchestrator._prune_history_async is original_prune
    assert orchestrator._consolidate_long_term_memory is original_consolidate
    assert orchestrator._runtime_mode_config["context_pruning"] is False
    assert orchestrator._runtime_mode_config["persona_evolution"] is False
    assert orchestrator._min_thought_interval == 60.0
    assert len(orchestrator.conversation_history) == 50
    assert orchestrator.cognitive_engine.singularity_factor == 1.0
    assert orchestrator.singularity_monitor.acceleration_factor == 1.0


@pytest.mark.asyncio
async def test_context_streaming_prune_reverts_suspicious_pruner_output(monkeypatch):
    history = [{"role": "user", "content": f"m{i}"} for i in range(60)]
    stream = _DummyStreaming(history, runtime_config={"context_pruning": True, "max_conversation_history": 50})

    prune_history = AsyncMock(return_value=[{"role": "user", "content": "tiny"}])
    monkeypatch.setattr("core.memory.context_pruner.context_pruner.prune_history", prune_history)

    await stream._prune_history_async()

    assert len(stream.conversation_history) == 50
    assert stream.conversation_history[0]["content"] == "m10"


@pytest.mark.asyncio
async def test_context_streaming_prune_uses_bounded_tail_when_disabled(monkeypatch):
    history = [{"role": "user", "content": f"m{i}"} for i in range(80)]
    stream = _DummyStreaming(history, runtime_config={"context_pruning": False, "max_conversation_history": 30})

    async def _unexpected_prune(*_args, **_kwargs):
        raise AssertionError("context pruner should not be called when feature is disabled")

    monkeypatch.setattr("core.memory.context_pruner.context_pruner.prune_history", _unexpected_prune)

    await stream._prune_history_async()

    assert len(stream.conversation_history) == 30
    assert stream.conversation_history[0]["content"] == "m50"


@pytest.mark.asyncio
async def test_dream_cycle_skips_when_runtime_mode_disables_feature(tmp_path: Path):
    dlq_path = tmp_path / "dlq.jsonl"
    dlq_path.write_text(json.dumps({"message": "repair this"}) + "\n")

    orchestrator = SimpleNamespace(
        _runtime_mode_config={"dream_cycle": False},
        enqueue_message=MagicMock(),
    )
    dream_cycle = DreamCycle(orchestrator, dlq_path)

    await dream_cycle.process_dreams()

    orchestrator.enqueue_message.assert_not_called()
