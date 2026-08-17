"""Joining an in-flight warmup from another loop must WAIT, not report failure.

LIVE 2026-08-17. Every launch answered the first message with "the live answer
lane could not finish preparing before a reasoning turn began."

    RuntimeError: Task <InferenceGate.ensure_foreground_ready
      coro=<MLXLocalClient.warmup()>> got Future <...> attached to a different loop
      -> reported warmup failure to a joined singleflight caller
    [DEGRADATION] chat.conversation_lane_admission:
      RuntimeError: worker_not_alive,init_not_complete,lane_warming

Boot starts the warmup on the boot loop. The chat turn arrives on the server
loop and joins the singleflight with `await asyncio.shield(inflight)`, which
across loops raises instantly — and the handler reports that as warmup failure
to the joiner. So admission failed on every cold start regardless of budget.

Three budget-side fixes moved the failure time (15s -> 25s) and none removed
it, because the failure was never about time. A symptom that shifts but does
not clear is evidence the cause is elsewhere.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from core.brain.llm.mlx_client import _join_inflight_across_loops


def test_a_same_loop_join_still_waits() -> None:
    async def scenario() -> bool:
        async def _work() -> bool:
            await asyncio.sleep(0.01)
            return True

        inflight = asyncio.ensure_future(_work())
        return await _join_inflight_across_loops(inflight)

    assert asyncio.run(scenario()) is True


def test_a_cross_loop_join_waits_instead_of_raising() -> None:
    """The regression: this raised 'attached to a different loop' instantly."""
    owner_loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run_owner() -> None:
        asyncio.set_event_loop(owner_loop)
        ready.set()
        owner_loop.run_forever()

    thread = threading.Thread(target=_run_owner, daemon=True)
    thread.start()
    ready.wait(5)

    async def _work() -> bool:
        await asyncio.sleep(0.05)
        return True

    inflight = asyncio.run_coroutine_threadsafe(
        asyncio.sleep(0), owner_loop
    )
    inflight.result(5)
    fut = asyncio.run_coroutine_threadsafe(_wrap(_work()), owner_loop)
    joined_task = fut.result(5)

    async def joiner() -> bool:
        return await _join_inflight_across_loops(joined_task)

    try:
        assert asyncio.run(joiner()) is True
    finally:
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        thread.join(timeout=5)


async def _wrap(coro):
    """Create the task ON the owner loop so it belongs to that loop."""
    return asyncio.ensure_future(coro)


def test_a_failing_warmup_still_propagates() -> None:
    """Waiting correctly must not swallow a genuine warmup failure."""

    async def scenario() -> None:
        async def _boom() -> bool:
            raise RuntimeError("worker died")

        inflight = asyncio.ensure_future(_boom())
        with pytest.raises(RuntimeError, match="worker died"):
            await _join_inflight_across_loops(inflight)

    asyncio.run(scenario())
