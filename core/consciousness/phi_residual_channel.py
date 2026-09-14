"""Grassmann state integers, from the worker process to the one that measures Φ.

The blocker, stated exactly. The residual sampler inside the steering hook
resolves PhiCore with ``ServiceContainer.has("phi_core")`` — an IN-PROCESS
lookup. Generation does not run in that process:

    core/brain/llm/mlx_client.py:7515
    p = ctx.Process(target=_mlx_worker_loop, args=(..., self._substrate_mem, ...))

The hook lives in the MLX worker subprocess; PhiCore is registered in the main
runtime. They have never shared a process, so the lookup returned False on every
token and the activation-grounded complex stayed empty forever. Measured on a
boot with three hooks installed and seven real generations:

    residual_stream_grassmann (insufficient_history:0/50 grassmann transitions)

Zero. That is the whole reason no activation-grounded live Φ has ever existed.

WHY A RING AND NOT THE RESPONSE PAYLOAD. Threading samples through the
generation result would edit the hot path and serialise per token, on the one
path where latency decides whether a turn survives. The precedent for crossing
this boundary is already here — ``_substrate_mem`` is a shared array passed
parent→worker carrying substrate state for steering. This mirrors it in
reverse, and it is cheap because the Grassmann encoder has already done the
expensive part: a ~5120-dimensional residual vector is reduced, in the worker,
to an 8-BIT STATE INTEGER. One byte per sample.

Single writer (worker), single reader (parent), one monotonic counter. No lock:
the reader tolerates having missed entries that wrapped, because a Φ estimate
built from a slightly gappy sample of transitions is a real measurement, and
blocking the decode loop to guarantee delivery would not be.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.PhiResidualChannel")

#: Slots in the ring. At the default sample-every-32-tokens this holds far more
#: than MIN_HISTORY_FOR_TPM, so a parent that drains on any reasonable cadence
#: loses nothing.
RING_SLOTS = 4096

#: Layout: [0] = monotonic write count, [1:] = the states themselves.
_COUNTER_INDEX = 0
_FIRST_SLOT = 1
CHANNEL_LENGTH = RING_SLOTS + _FIRST_SLOT


def create_channel(mp_context: Any) -> Any:
    """Allocate the shared ring. Called in the parent, before the fork."""
    try:
        return mp_context.Array("i", CHANNEL_LENGTH, lock=False)
    except (AttributeError, OSError, ValueError) as exc:
        logger.warning("Phi residual channel unavailable: %s", exc)
        return None


def publish_state(channel: Any, state: int) -> bool:
    """Worker side: record one Grassmann state. Never raises, never blocks.

    A failure here must cost a telemetry sample and nothing else — this runs
    inside the forward pass of a transformer block.
    """
    if channel is None:
        return False
    try:
        count = int(channel[_COUNTER_INDEX])
        channel[_FIRST_SLOT + (count % RING_SLOTS)] = int(state) & 0xFF
        # The counter is bumped AFTER the slot is written, so a reader that
        # races sees a smaller count rather than an unwritten slot.
        channel[_COUNTER_INDEX] = count + 1
        return True
    except (IndexError, OSError, TypeError, ValueError):
        return False


def drain(channel: Any, since: int) -> tuple[list[int], int]:
    """Parent side: states written after ``since``, and the new counter.

    Returns ``([], since)`` when nothing is new. When more than ``RING_SLOTS``
    have been written since the last drain the oldest are gone; the reader takes
    what survives and moves on, because a gap in the transition sample widens
    the interval on Φ rather than invalidating it.
    """
    if channel is None:
        return ([], int(since))
    try:
        count = int(channel[_COUNTER_INDEX])
    except (IndexError, OSError, TypeError, ValueError):
        return ([], int(since))

    start = max(int(since), 0)
    if count <= start:
        return ([], count)
    if count - start > RING_SLOTS:
        logger.debug(
            "Phi residual ring wrapped: %d states written since the last drain, "
            "keeping the most recent %d.",
            count - start,
            RING_SLOTS,
        )
        start = count - RING_SLOTS

    states: list[int] = []
    try:
        for index in range(start, count):
            states.append(int(channel[_FIRST_SLOT + (index % RING_SLOTS)]) & 0xFF)
    except (IndexError, OSError, TypeError, ValueError):
        return (states, start + len(states))
    return (states, count)


__all__ = [
    "CHANNEL_LENGTH",
    "RING_SLOTS",
    "create_channel",
    "drain",
    "publish_state",
]
