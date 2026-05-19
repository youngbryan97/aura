import pytest

import core.state.state_repository as state_module
from core.state.state_repository import StateRepository, _schedule_state_task


class ClosingAwaitable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def __await__(self):
        if False:
            yield None
        return None


class FailingTracker:
    def create_task(self, _awaitable, *, name=None):
        self.last_name = name
        raise RuntimeError(f"{name}: loop unavailable")


def test_state_scheduler_closes_unscheduled_awaitable():
    awaitable = ClosingAwaitable()

    task = _schedule_state_task(awaitable, name="state.contract", tracker=FailingTracker())

    assert task is None
    assert awaitable.closed is True


@pytest.mark.asyncio
async def test_state_repair_reports_deferred_consumer_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(state_module, "get_task_tracker", lambda: FailingTracker())
    repo = StateRepository(db_path=str(tmp_path / "state.db"), is_vault_owner=True)
    repo._is_processing = True

    result = await repo.repair_runtime()

    assert result["actions"] == ["consumer_restart_deferred", "reconnected_db"]
    assert result["status"]["local_consumer_alive"] is False

    await repo.close()


def test_state_queue_coalescing_is_bounded_and_keeps_latest(tmp_path):
    repo = StateRepository(db_path=str(tmp_path / "state.db"), is_vault_owner=True)
    repo._mutation_queue.put_nowait({"version": 1})
    repo._mutation_queue.put_nowait({"version": 2})
    repo._mutation_queue.put_nowait({"version": 3})

    dropped = repo._coalesce_pending_mutations(keep_latest=True)

    assert dropped == 2
    assert repo._mutation_queue.qsize() == 1
    assert repo._mutation_queue.get_nowait() == {"version": 3}
