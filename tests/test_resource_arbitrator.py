import asyncio
import uuid

import pytest

from core.resilience.resource_arbitrator import ResourceArbitrator


@pytest.mark.asyncio
async def test_worker_token_timeout_does_not_leak_permit():
    arbitrator = ResourceArbitrator()
    worker = f"MLX-Cortex-test-{uuid.uuid4().hex}"

    assert await arbitrator.acquire_inference(worker=worker, timeout=0.05)
    assert await arbitrator.acquire_inference(worker=worker, timeout=0.05) is False

    await arbitrator.release_inference(worker=worker)
    await asyncio.sleep(0.05)

    assert await arbitrator.acquire_inference(worker=worker, timeout=0.05)
    await arbitrator.release_inference(worker=worker)


@pytest.mark.asyncio
async def test_inference_context_raises_when_worker_token_times_out():
    arbitrator = ResourceArbitrator()
    worker = f"MLX-Cortex-timeout-{uuid.uuid4().hex}"

    assert await arbitrator.acquire_inference(worker=worker, timeout=0.05)

    with pytest.raises(asyncio.TimeoutError):
        async with arbitrator.inference_context(worker=worker, timeout=0.05):
            pass

    await arbitrator.release_inference(worker=worker)


@pytest.mark.asyncio
async def test_priority_worker_timeout_respects_caller_budget():
    arbitrator = ResourceArbitrator()
    worker = f"MLX-Cortex-priority-{uuid.uuid4().hex}"

    assert await arbitrator.acquire_inference(worker=worker, timeout=0.05)

    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await arbitrator.acquire_inference(priority=True, worker=worker, timeout=0.05) is False
    elapsed = loop.time() - started

    await arbitrator.release_inference(worker=worker)

    assert elapsed < 0.5
