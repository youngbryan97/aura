from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.scratchpad import ScratchpadEngine
from core.brain.types import ThinkingMode


class _BrainStub:
    def __init__(self) -> None:
        self.calls: list[ThinkingMode] = []

    async def think(self, objective, context=None, mode=None, **kwargs):
        self.calls.append(mode)
        return SimpleNamespace(content=f"mode={getattr(mode, 'name', mode)}")


@pytest.mark.asyncio
async def test_scratchpad_uses_slow_then_reflective_modes_for_shallow_depth():
    brain = _BrainStub()
    scratchpad = ScratchpadEngine(cognitive_engine=brain)

    await scratchpad.think_recursive("Audit the current response strategy.", {"history": []}, depth=1)

    assert brain.calls == [ThinkingMode.SLOW, ThinkingMode.REFLECTIVE]


@pytest.mark.asyncio
async def test_scratchpad_uses_deep_mode_for_multi_pass_planning():
    brain = _BrainStub()
    scratchpad = ScratchpadEngine(cognitive_engine=brain)

    await scratchpad.think_recursive("Audit the current response strategy.", {"history": []}, depth=2)

    assert brain.calls[0] == ThinkingMode.DEEP
    assert brain.calls[1:] == [ThinkingMode.REFLECTIVE, ThinkingMode.REFLECTIVE]
