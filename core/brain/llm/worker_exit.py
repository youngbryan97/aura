"""Waiting for a model worker's init handshake, and whether its process has ended.

The handshake bound was an absolute wall clock: 300s, then the worker is wedged
and respawned. LIVE 2026-09-16, load 34 on 18 cores: the 27B worker was at 9GB
of RSS and loading when the clock ran out, and the respawn started the load
again from nothing. The bound is now on the worker's progress. Its RSS and CPU
advance while it loads; ``stall_s`` of neither is the wedge the clock was for.

A worker that has ended cannot answer, so the wait ends at the next look and
says how the process ended. LIVE 14 November 2026: her
cortex's worker logged its last init step at 14:39:57 and was gone soon after
(her RAM at 35%, its admission lease reaped "for dead cortex worker"). The
handshake wait samples a loading worker's memory and CPU to tell a load from a
wedge; with no process to sample it ran on the clock alone, until 14:45:17,
and nothing recorded how the worker ended.
"""

from __future__ import annotations

import logging
import os
import signal
from typing import Any

logger = logging.getLogger("LLM.MLX")


class WorkerEndedDuringInit(TimeoutError):
    """The worker process ended before its init handshake; the caller's retry for a timed-out init applies."""


def how_it_ended(process: Any) -> str:
    """How a worker process ended ("exit code 1", "signal SIGKILL"), or "" while it runs or before it starts."""
    try:
        code = None if process is None else process.exitcode
    except (ValueError, AttributeError, OSError):
        return ""  # not a failure: a closed or foreign handle says nothing of the worker
    if code is None:
        return ""
    if code < 0:
        try:
            return f"signal {signal.Signals(-code).name}"
        except ValueError:
            return f"signal {-code}"
    return f"exit code {code}"


def the_worker_has_not_ended(process: Any, model_path: Any = "") -> None:
    """Raise WorkerEndedDuringInit, with how it ended, when the worker's process has ended."""
    ended = how_it_ended(process)
    if ended:
        name = os.path.basename(str(model_path or ""))
        logger.error("🛑 [MLX] %s worker ended during init: %s", name, ended)
        raise WorkerEndedDuringInit(f"{name} worker ended during init: {ended}")


async def await_init_while_the_worker_loads(client: Any, fut: Any, *, stall_s: float) -> Any:
    """The init handshake's result, waited for while the worker's memory and CPU advance and its process runs."""
    from .mlx_client import _await_shared_future

    stall_s = max(0.0, float(stall_s))
    period = max(0.05, min(15.0, stall_s / 20.0)) if stall_s else 15.0
    seen = client.worker_load_progress()
    still_for = 0.0
    while True:
        try:
            return await _await_shared_future(fut, timeout_s=period)
        except TimeoutError:
            pass  # not a failure: a look, not a deadline
        the_worker_has_not_ended(getattr(client, "_process", None), getattr(client, "model_path", ""))
        progress = client.worker_load_progress()
        if progress is None:
            # No process to read and none ended: nothing can advance, so the wait is bounded by the clock alone.
            still_for += period
        elif seen is None or progress[0] > seen[0] or progress[1] > seen[1]:
            seen, still_for = progress, 0.0
        else:
            still_for += period
        if still_for >= stall_s:
            raise TimeoutError(
                f"worker init made no progress for {still_for:.0f}s "
                f"(rss={0 if seen is None else seen[0] / 1e9:.1f}GB, "
                f"cpu={0 if seen is None else seen[1]:.0f}s)"
            )
