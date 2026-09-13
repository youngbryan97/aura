import asyncio

import pytest

from core.runtime.shared_read import SharedRead


@pytest.mark.asyncio
async def test_late_result_survives_waiter_timeout_and_cancellation():
    reads = SharedRead()
    release = asyncio.Event()
    started = asyncio.Event()
    calls = 0

    async def load():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "durable result"

    with pytest.raises(TimeoutError):
        await reads.read("owner", load, wait_s=0.01)
    assert started.is_set()
    waiter = asyncio.create_task(reads.read("owner", load, wait_s=2))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    assert await reads.read("owner", load, wait_s=1) == "durable result"
    assert await reads.read("owner", load, wait_s=1) == "durable result"
    assert calls == 1


@pytest.mark.asyncio
async def test_concurrent_waiters_share_one_read_but_not_another_scope():
    reads = SharedRead()
    calls = []

    async def load(owner):
        calls.append(owner)
        await asyncio.sleep(0)
        return owner

    results = await asyncio.gather(*(
        reads.read(owner, lambda owner=owner: load(owner), wait_s=1)
        for owner in ("a", "a", "b", "a")
    ))
    assert results == ["a", "a", "b", "a"]
    assert calls == ["a", "b"]


@pytest.mark.asyncio
async def test_failed_read_is_retryable_and_preserves_the_error():
    reads = SharedRead()
    calls = 0

    async def load():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("unavailable")
        return "recovered"

    with pytest.raises(OSError, match="unavailable"):
        await reads.read("a", load, wait_s=1)
    assert await reads.read("a", load, wait_s=1) == "recovered"


@pytest.mark.asyncio
async def test_failure_after_the_waiter_leaves_is_still_reported(caplog):
    reads = SharedRead()
    release = asyncio.Event()
    failed = asyncio.Event()

    async def load():
        await release.wait()
        failed.set()
        raise OSError("late disk failure")

    with pytest.raises(TimeoutError):
        await reads.read("a", load, wait_s=0.01)
    release.set()
    await failed.wait()
    assert "Shared read failed before result delivery" in caplog.text
    assert "late disk failure" in caplog.text


@pytest.mark.asyncio
async def test_capacity_never_evicts_an_inflight_owner():
    reads = SharedRead(capacity=1)
    release = asyncio.Event()

    async def load():
        await release.wait()
        return "complete"

    with pytest.raises(TimeoutError):
        await reads.read("a", load, wait_s=0.01)
    with pytest.raises(RuntimeError, match="capacity_busy"):
        await reads.read("b", load, wait_s=1)
    release.set()
    assert await reads.read("a", load, wait_s=1) == "complete"
    assert await reads.read("b", load, wait_s=1) == "complete"


@pytest.mark.asyncio
async def test_completed_values_expire(monkeypatch):
    from core.runtime import shared_read

    now = [10.0]
    monkeypatch.setattr(shared_read.time, "monotonic", lambda: now[0])
    reads = SharedRead(retention_s=5)
    calls = 0

    async def load():
        nonlocal calls
        calls += 1
        return calls

    assert await reads.read("a", load, wait_s=1) == 1
    now[0] += 4
    assert await reads.read("a", load, wait_s=1) == 1
    now[0] += 2
    assert await reads.read("a", load, wait_s=1) == 2


@pytest.mark.asyncio
async def test_eager_task_factory_keeps_completed_entry_reusable():
    loop = asyncio.get_running_loop()
    prior = loop.get_task_factory()
    reads = SharedRead(capacity=1)

    async def load():
        return "ready"

    try:
        loop.set_task_factory(asyncio.eager_task_factory)
        assert await reads.read("a", load, wait_s=1) == "ready"
        assert await reads.read("b", load, wait_s=1) == "ready"
    finally:
        loop.set_task_factory(prior)


@pytest.mark.asyncio
async def test_unbounded_executor_owner_preserves_worker_timeout(monkeypatch):
    from core.runtime import executors

    def fail():
        raise TimeoutError("database busy")

    with pytest.raises(TimeoutError, match="database busy"):
        await executors.run_durable_receipt_io(fail, timeout_s=None)
