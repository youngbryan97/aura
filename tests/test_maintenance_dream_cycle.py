import pytest

from core.maintenance import dream_cycle
from core.runtime.errors import get_degradation_tracker


class _Memory:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.consolidated = False

    async def consolidate(self):
        if self.fail:
            raise RuntimeError("memory consolidation offline")
        self.consolidated = True


class _Coordinator:
    def __init__(self):
        self.checkpointed = False

    def checkpoint_wal(self):
        self.checkpointed = True


class _Emitter:
    def __init__(self):
        self.events = []

    def emit(self, *args, **kwargs):
        self.events.append((args, kwargs))


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    get_degradation_tracker().reset()

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(dream_cycle.asyncio, "sleep", _no_sleep)
    yield
    get_degradation_tracker().reset()


@pytest.mark.asyncio
async def test_dream_cycle_continues_after_memory_consolidation_failure(monkeypatch):
    memory = _Memory(fail=True)
    coordinator = _Coordinator()
    emitter = _Emitter()

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: memory if name == "episodic_memory" else default,
    )
    monkeypatch.setattr(
        "core.resilience.database_coordinator.get_db_coordinator",
        lambda: coordinator,
    )
    monkeypatch.setattr("core.thought_stream.get_emitter", lambda: emitter)

    result = await dream_cycle.run_dream_cycle()

    assert result["ok"] is False
    assert "episodic_memory_consolidation" in result["degraded_steps"]
    assert "wal_checkpoint" in result["completed_steps"]
    assert coordinator.checkpointed is True
    assert emitter.events
    last = get_degradation_tracker().recent(subsystem="dream_cycle")[-1]
    assert last.action == "continued dream cycle after episodic memory consolidation failed"


@pytest.mark.asyncio
async def test_dream_cycle_returns_degraded_result_when_wal_checkpoint_fails(monkeypatch):
    memory = _Memory()
    emitter = _Emitter()

    class _BrokenCoordinator:
        def checkpoint_wal(self):
            raise RuntimeError("wal unavailable")

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: memory if name == "episodic_memory" else default,
    )
    monkeypatch.setattr(
        "core.resilience.database_coordinator.get_db_coordinator",
        lambda: _BrokenCoordinator(),
    )
    monkeypatch.setattr("core.thought_stream.get_emitter", lambda: emitter)

    result = await dream_cycle.run_dream_cycle()

    assert result["ok"] is False
    assert "episodic_memory_consolidation" in result["completed_steps"]
    assert "wal_checkpoint" in result["degraded_steps"]
    assert emitter.events
    last = get_degradation_tracker().recent(subsystem="dream_cycle")[-1]
    assert last.action == "completed remaining dream-cycle steps after WAL checkpoint failed"
