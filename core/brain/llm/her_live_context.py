"""Read text into her cortex's cache for a voice session's live context (core/voice/duplex/live_context.py).

A job of its own (mlx_worker_read_ahead.py), sent the way her other control messages are and answered by id. Blocking:
call it off the event loop.
"""
from __future__ import annotations

import uuid
from typing import Any


def read_ahead(client: Any, text: str, scope: str, *, timeout_s: float = 60.0) -> dict[str, Any] | None:
    """Ask her worker to read ``text`` into the cache scope ``scope``; its answer, or None when it could not be asked."""
    process = getattr(client, "_process", None)
    request_queue = getattr(client, "_req_q", None)
    if not (process is not None and process.is_alive() and getattr(client, "_init_done", False)) or request_queue is None:
        return None
    from core.brain.llm.mlx_client import _new_shared_future

    req_id = uuid.uuid4().hex
    fut = _new_shared_future()
    client._pending_generations[req_id] = fut
    try:
        job = {"id": req_id, "action": "read_ahead", "prompt": str(text), "prompt_cache_scope": str(scope)}
        request_queue.put(client._authorize_job(job, principal="voice_live.read_ahead"), True, 2.0)
        return fut.result(timeout=max(1.0, float(timeout_s)))
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        client._pending_generations.pop(req_id, None)
