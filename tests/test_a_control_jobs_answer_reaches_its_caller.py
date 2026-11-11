"""A control job's answer reaches the caller waiting on its id, whatever the job.

LIVE 11 November 2026: her client's listener delivered only replies whose action it knew as terminal. The answers to
read_ahead (her voice mode's live context), pin_weights and clear_cache (the memory shed ladder) fell through and were
dropped: the live context never opened, every pin "timed out", and a shed waited out 10 s and reported nothing freed.
"""
from __future__ import annotations

import asyncio
import contextlib
import queue
from concurrent import futures as cfutures

import pytest

from core.brain.llm.mlx_client import MLXLocalClient


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    {"id": "control-1", "status": "ok", "prompt_cache_cleared": True, "prompt_cache_bytes_freed": 4096},
    {"id": "control-1", "action": "read_ahead", "status": "ok", "held": 25, "read": 25, "seconds": 0.4},
    {"id": "control-1", "status": "ok", "pinned": True, "wired_limit": 1 << 34},
])
async def test_the_answer_reaches_the_caller_waiting_on_its_id(monkeypatch, reply):
    from core.container import ServiceContainer

    monkeypatch.setattr(ServiceContainer, "get", lambda *_args, **_kwargs: None)
    client = MLXLocalClient(model_path="/models/test-1.5b")
    client._res_q = queue.Queue()
    waiting: cfutures.Future = cfutures.Future()
    client._pending_generations["control-1"] = waiting
    client._current_request_id = "a-reply-in-flight"
    client._current_gen_future = asyncio.get_running_loop().create_future()
    listener = asyncio.create_task(client._response_listener_loop())
    try:
        client._res_q.put(dict(reply))
        answer = await asyncio.wait_for(asyncio.wrap_future(waiting), timeout=2.0)
    finally:
        listener.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listener
    assert answer == reply
    assert "control-1" not in client._pending_generations
    assert not client._current_gen_future.done()  # nobody else's request was completed with it
