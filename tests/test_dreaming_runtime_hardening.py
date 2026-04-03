from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.consciousness.dreaming import DreamingProcess


def test_dream_reflection_is_lightweight_and_bounded():
    reflection = DreamingProcess._compose_reflection(
        "\n".join(
            [
                "Context: alpha cluster | Action: explored avenue | Outcome: insight gained",
                "Context: alpha cluster | Action: explored avenue | Outcome: insight gained",
                "Context: beta lane | Action: stabilized state | Outcome: continuity preserved",
                "Context: gamma lane | Action: audited drift | Outcome: coherence restored",
                "Context: delta lane | Action: ignored overflow | Outcome: should be trimmed",
            ]
        )
    )

    assert "integrating" in reflection
    assert "alpha cluster" in reflection
    assert "beta lane" in reflection
    assert "delta lane" not in reflection


@pytest.mark.asyncio
async def test_dream_cycle_avoids_brain_think_on_event_loop(service_container):
    orchestrator = SimpleNamespace(_last_user_interaction_time=0)
    dreamer = DreamingProcess(orchestrator, interval=0.1)
    recorded_growth = []

    dreamer._identity = SimpleNamespace(
        record_evolution=lambda **kwargs: recorded_growth.append(kwargs)
    )
    dreamer._narrator = object()

    async def _recent_summary():
        return (
            "Context: alpha cluster | Action: explored avenue | Outcome: insight gained\n"
            "Context: beta lane | Action: stabilized state | Outcome: continuity preserved"
        )

    dreamer._get_recent_summary = _recent_summary

    class _VectorMemory:
        def __init__(self):
            self.brains = []

        async def consolidate(self, brain=None):
            self.brains.append(brain)
            return 0

    vector_memory = _VectorMemory()
    brain = SimpleNamespace(
        think=AsyncMock(side_effect=AssertionError("dream cycle should not invoke brain.think"))
    )

    service_container.register_instance("vector_memory_engine", vector_memory, required=False)
    service_container.register_instance("cognitive_engine", brain, required=False)

    await dreamer.dream()

    assert vector_memory.brains == [None]
    assert recorded_growth
    assert "alpha cluster" in recorded_growth[0]["reflection"]
    brain.think.assert_not_called()
