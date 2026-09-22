"""Latent readouts, from the worker that has the activations to the substrate.

The backward half of the latent bridge — ``h_layer · v_i → substrate.x[i]`` —
was written, tested in isolation, and could not have worked in any process.
Two independent reasons, both in ``SubstrateInjectionThread._loop``:

1. It resolved the substrate with ``ServiceContainer.get("conscious_substrate")``.
   The readout hooks live in the MLX worker subprocess; the substrate is
   registered in the main runtime. The lookup returns None there, forever —
   the same boundary that stopped the Φ residual samples.
2. It called ``asyncio.get_running_loop()`` from a plain daemon thread, which
   raises ``RuntimeError`` unconditionally. Even in the main process, with the
   substrate right there, the injection was inside a ``try`` that could only
   ever take the ``except``.

So ``attach_latent_bridge()`` having no caller was not the whole defect. It
was the *merciful* part: wiring it as written would have produced a
live-looking backward path that injected nothing, which is worse than an
honest absence.

This is the missing transport, mirroring
``core/consciousness/phi_residual_channel.py``: shared memory allocated in
the parent before the fork, written by the worker, drained by the parent
where the substrate and a real event loop both exist.

CUMULATIVE, NOT INCREMENTAL. Each slot holds a running total rather than a
pending delta, and the reader injects the difference since its last read.
A single writer adding to a slot and a single reader subtracting a snapshot
cannot lose an update to a race, and a reader that misses a cycle picks up
the whole interval on the next one instead of dropping it. Deltas are small
and bounded; the totals are float64.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.LatentReadoutChannel")

#: Substrate neurons the steering library addresses. The declared dimensions
#: use indices 0-5; 32 leaves room for the library to grow without a protocol
#: change, and costs 256 bytes.
SLOTS = 32

#: Layout: [0] = monotonic publish counter, [1:] = cumulative delta per index.
_COUNTER_INDEX = 0
_FIRST_SLOT = 1
CHANNEL_LENGTH = SLOTS + _FIRST_SLOT

#: Per-read clamp on any single injected delta, matching the bound the old
#: in-process injector applied. Feedback that runs away is worse than
#: feedback that saturates.
MAX_DELTA = 0.5


def create_channel(mp_context: Any) -> Any:
    """Allocate the shared array. Called in the parent, before the fork."""
    try:
        return mp_context.Array("d", CHANNEL_LENGTH, lock=False)
    except (AttributeError, OSError, ValueError) as exc:
        logger.warning("Latent readout channel unavailable: %s", exc)
        return None


def publish_deltas(channel: Any, deltas: dict[int, float]) -> bool:
    """Worker side: add this cycle's readout deltas. Never raises, never blocks.

    Runs on the readout thread, downstream of a transformer forward pass. A
    failure here must cost one feedback cycle and nothing else.
    """
    if channel is None or not deltas:
        return False
    try:
        for index, delta in deltas.items():
            slot = int(index)
            if slot < 0 or slot >= SLOTS:
                continue
            value = float(delta)
            if value != value:  # NaN
                continue
            channel[_FIRST_SLOT + slot] += value
        channel[_COUNTER_INDEX] = int(channel[_COUNTER_INDEX]) + 1
        return True
    # not a failure: a channel that will not take the write is a channel nothing was
    # published on, and False says so to the caller.
    except (IndexError, OSError, OverflowError, TypeError, ValueError):
        return False


def drain(
    channel: Any, previous: list[float] | None
) -> tuple[dict[int, float], list[float]]:
    """Parent side: deltas accumulated since ``previous``, and a new snapshot.

    Pass ``None`` on the first call to establish a baseline without injecting
    — a fresh reader must not inject the entire history of the worker as one
    stimulus.
    """
    if channel is None:
        return ({}, previous or [0.0] * SLOTS)
    try:
        snapshot = [float(channel[_FIRST_SLOT + i]) for i in range(SLOTS)]
    except (IndexError, OSError, TypeError, ValueError):
        return ({}, previous or [0.0] * SLOTS)

    if previous is None:
        return ({}, snapshot)

    deltas: dict[int, float] = {}
    for index, total in enumerate(snapshot):
        change = total - (previous[index] if index < len(previous) else 0.0)
        if change == 0.0 or change != change:
            continue
        deltas[index] = max(-MAX_DELTA, min(MAX_DELTA, change))
    return (deltas, snapshot)


def publish_count(channel: Any) -> int:
    """How many cycles the worker has published. Zero means silence."""
    if channel is None:
        return 0
    try:
        return int(channel[_COUNTER_INDEX])
    # not a failure: no readable counter is silence, which the docstring says zero means.
    except (IndexError, OSError, TypeError, ValueError):
        return 0


__all__ = [
    "CHANNEL_LENGTH",
    "MAX_DELTA",
    "SLOTS",
    "create_channel",
    "drain",
    "publish_count",
    "publish_deltas",
]
