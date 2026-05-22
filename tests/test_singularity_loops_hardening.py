import asyncio

import pytest

from core.evolution.singularity_loops import SingularityLoops
from core.runtime.errors import get_degradation_tracker


def test_singularity_tick_records_failed_loop_and_preserves_siblings(monkeypatch):
    get_degradation_tracker().reset()
    loops = SingularityLoops()
    ran: list[str] = []

    async def failing_loop():
        detail = "metacognitive bridge failed"
        raise RuntimeError(detail)

    async def healthy_loop(name: str = "healthy"):
        ran.append(name)

    monkeypatch.setattr(loops, "_loop_metacognition_to_self_model", failing_loop)
    monkeypatch.setattr(loops, "_loop_curiosity_to_exploration", healthy_loop)
    monkeypatch.setattr(loops, "_loop_goal_advancement", healthy_loop)
    monkeypatch.setattr(loops, "_loop_profile_injection", healthy_loop)
    monkeypatch.setattr(loops, "_loop_distillation_trigger", healthy_loop)
    monkeypatch.setattr(loops, "_loop_affect_to_exploration", healthy_loop)

    asyncio.run(loops._tick())

    status = loops.get_status()
    assert ran == ["healthy", "healthy", "healthy", "healthy", "healthy"]
    assert status["consecutive_failures"]["metacognition_to_self_model"] == 1
    assert "RuntimeError" in status["last_loop_error"]["metacognition_to_self_model"]
    assert get_degradation_tracker().count("singularity_loops", "degraded") >= 1


def test_singularity_tick_propagates_cancellation(monkeypatch):
    loops = SingularityLoops()

    async def cancelled_loop():
        should_cancel = True
        if not should_cancel:
            return None
        raise asyncio.CancelledError()

    async def healthy_loop():
        return None

    monkeypatch.setattr(loops, "_loop_metacognition_to_self_model", cancelled_loop)
    monkeypatch.setattr(loops, "_loop_curiosity_to_exploration", healthy_loop)
    monkeypatch.setattr(loops, "_loop_goal_advancement", healthy_loop)
    monkeypatch.setattr(loops, "_loop_profile_injection", healthy_loop)
    monkeypatch.setattr(loops, "_loop_distillation_trigger", healthy_loop)
    monkeypatch.setattr(loops, "_loop_affect_to_exploration", healthy_loop)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loops._tick())
