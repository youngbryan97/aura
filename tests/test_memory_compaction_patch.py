from types import SimpleNamespace

import pytest

from core.memory_compaction_patch import _patched_memory_consolidation_execute


@pytest.mark.asyncio
async def test_memory_compaction_patch_forwards_extra_kwargs():
    captured = {}

    async def _original_execute(state, objective=None, **kwargs):
        captured["objective"] = objective
        captured["kwargs"] = kwargs
        return state

    phase = SimpleNamespace(_original_execute=_original_execute)
    state = SimpleNamespace(cognition=SimpleNamespace(working_memory=[]))

    result = await _patched_memory_consolidation_execute(
        phase,
        state,
        objective="summarize",
        is_background=True,
    )

    assert result is state
    assert captured["objective"] == "summarize"
    assert captured["kwargs"]["is_background"] is True
