"""Whether the turn is still producing, for every clock that might end it.

A desktop turn passes through five nested deadlines: the chat route's turn
budget, the cognitive cycle, the inference gate's request deadline, the MLX
client's wait, and the worker's own decode loop. Each was a number chosen
before anything knew what the answer would cost, and each capped the next, so
raising an inner one changed nothing while an outer one was smaller.

Raising all five is not the fix either. It is the same design with larger
numbers, and the number is not the thing that was wrong: a stopwatch cannot
tell a generation that is working from one that is stuck, and on this runtime
it was only ever stopping the first kind. The second kind — a silent worker, a
decode looping forever — is caught by watching the output, which is what the
first-token ceiling, the livelock ceiling and the token sentinel already do.

So there is one signal here and every clock asks it the same question: has
anything arrived recently? A turn that is producing is not out of time. A turn
that has gone quiet is, whatever its deadline says, and that is the case worth
ending.

The signal is one timestamp, written by the lane that decodes and read by the
layers above it, because a turn is one thing happening once even though five
places are waiting on it.
"""

from __future__ import annotations

import threading
import time

__all__ = [
    "note_progress",
    "seconds_since_progress",
    "still_producing",
    "forget_progress",
]

_lock = threading.Lock()

#: When a token was last seen, on the monotonic clock. Zero means nothing has
#: been seen at all, which is silence rather than slowness.
_last_progress_at: float = 0.0


def note_progress() -> None:
    """A token arrived. Called from the lane that decodes."""

    global _last_progress_at
    now = time.monotonic()
    with _lock:
        _last_progress_at = now


def seconds_since_progress() -> float:
    """How long since anything arrived, or -1.0 when nothing ever has."""

    with _lock:
        last = _last_progress_at
    if last <= 0.0:
        return -1.0
    return max(0.0, time.monotonic() - last)


def still_producing(*, within_s: float) -> bool:
    """True when a token arrived recently enough to call this turn alive.

    ``within_s`` is how long a gap between tokens is still normal. Above it
    the generation has stopped saying anything, and a deadline is then
    reporting something real rather than reporting the length of the answer.

    False when nothing has arrived at all: that is silence, and the
    first-token ceiling owns it.
    """

    try:
        window = float(within_s)
    except (TypeError, ValueError):
        return False
    if not (window > 0.0):
        return False
    since = seconds_since_progress()
    if since < 0.0:
        return False
    return since <= window


def forget_progress() -> None:
    """Drop the reading. For tests, and between unrelated turns."""

    global _last_progress_at
    with _lock:
        _last_progress_at = 0.0


def normal_gap_between_tokens(measured_for_64_tokens: float = 0.0) -> float:
    """How long a silence is still ordinary, in seconds.

    A deadline that defers to progress needs to know what "no progress" means,
    and the honest measure is the machine's own decode rate. This package is a
    foundation and may not reach up to the lane that measures it, so the rate
    is handed in by the caller that already knows it — which is also the layer
    that would have to keep any import here in step.

    Floored, because the gap before the FIRST token is prefill and can be far
    longer than any gap after it, and because an unmeasured rate must not make
    every pause look like a stall.
    """

    try:
        measured = float(measured_for_64_tokens)
    except (TypeError, ValueError):
        measured = 0.0
    return max(20.0, measured)
