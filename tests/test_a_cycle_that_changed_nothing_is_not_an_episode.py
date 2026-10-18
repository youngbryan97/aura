"""A morphogenesis cycle goes into her episodic memory only when something happened.

On 17 October 32,431 of her 36,138 episodes were the same line, "MorphogeneticRuntime
self-organization cycle", with no failure and no change to the topology.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.morphogenesis.runtime import MorphogeneticRuntime
from core.morphogenesis.types import MorphogenesisConfig


class _Episodic:
    def __init__(self) -> None:
        self.recorded: list[dict] = []

    def record_episode_async(self, **kwargs):
        self.recorded.append(kwargs)

        async def _done() -> None:
            return None

        return _done()


@pytest.fixture
def remembered(monkeypatch):
    episodic = _Episodic()
    from core.container import ServiceContainer

    real_get = ServiceContainer.get

    def _get(name, default=None):
        if name == "episodic_memory":
            return episodic
        return real_get(name, default=default)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_get))
    monkeypatch.setattr(
        "core.runtime.task_ownership.fire_and_forget",
        lambda coroutine, **_: coroutine.close() or True,
    )
    return episodic


@pytest.mark.asyncio
async def test_only_a_failure_or_a_moved_topology_is_remembered(remembered):
    rt = MorphogeneticRuntime(config=MorphogenesisConfig(enabled=False, episode_every_events=1))
    rt.graph = SimpleNamespace(version=4)
    ok = [{"cell": "a", "success": True}]

    await rt._maybe_record_episode(ok)  # the first cycle seen is remembered
    await rt._maybe_record_episode(ok)  # same topology, nothing failed
    await rt._maybe_record_episode(ok)
    assert len(remembered.recorded) == 1

    rt.graph = SimpleNamespace(version=5)
    await rt._maybe_record_episode(ok)
    assert len(remembered.recorded) == 2

    await rt._maybe_record_episode([{"cell": "b", "success": False}])
    assert len(remembered.recorded) == 3
    assert remembered.recorded[-1]["success"] is False
