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

Each TurnOutcome owns its progress. The model client captures that owner at
dispatch; a tool keeps its completion handle. Readers use the same turn even
when worker callbacks and tool completion run on different async tasks.
"""

from __future__ import annotations

import asyncio
import time
import weakref

from core.runtime.lockdep import checked_lock

__all__ = [
    "TurnProgress",
    "ToolActivity",
    "capture_progress",
    "note_progress",
    "tool_started",
    "tool_finished",
    "seconds_since_progress",
    "still_producing",
    "forget_progress",
]

class TurnProgress:
    """Activity owned by one TurnOutcome, including callbacks on other tasks."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.runtime.turn_progress")
        self._last_at = 0.0
        self._closed = False
        self._tools: set[ToolActivity] = set()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._last_at = 0.0
            self._tools.clear()


class ToolActivity:
    """Completion handle bound to the turn and task that started the tool."""

    def __init__(self, progress: TurnProgress) -> None:
        self.progress = progress
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        self._task = weakref.ref(task) if task is not None else None

    def owner_is_running(self) -> bool:
        if self._task is None:
            return True
        task = self._task()
        return task is not None and not task.done()


# Legacy unbound callers have no foreground authority. A bound turn always
# owns a separate reading; background tokens cannot renew its clocks.
_unbound_progress = TurnProgress()


def capture_progress() -> TurnProgress:
    from core.runtime.turn_outcome import current_turn

    outcome = current_turn()
    return outcome.progress if outcome is not None else _unbound_progress


def note_progress(*, progress: TurnProgress | None = None) -> None:
    """Publish advancing work to the captured owner, not the callback's context."""

    state = progress if progress is not None else capture_progress()
    now = time.monotonic()
    with state._lock:
        if not state._closed:
            state._last_at = now


def tool_started() -> ToolActivity:
    state = capture_progress()
    activity = ToolActivity(state)
    now = time.monotonic()
    with state._lock:
        if not state._closed:
            state._tools.add(activity)
            state._last_at = now
    return activity


def tool_finished(activity: ToolActivity | None) -> None:
    """Complete exactly this activity, even from a different async context."""

    if activity is None:
        return
    state = activity.progress
    now = time.monotonic()
    with state._lock:
        if activity in state._tools:
            state._tools.remove(activity)
            state._last_at = now


def seconds_since_progress(*, progress: TurnProgress | None = None) -> float:
    """How long since anything arrived, or -1.0 when nothing ever has."""

    state = progress if progress is not None else capture_progress()
    with state._lock:
        last = state._last_at
    if last <= 0.0:
        return -1.0
    return max(0.0, time.monotonic() - last)


def still_producing(*, within_s: float, progress: TurnProgress | None = None) -> bool:
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
    state = progress if progress is not None else capture_progress()
    with state._lock:
        state._tools = {tool for tool in state._tools if tool.owner_is_running()}
        if state._tools:
            return True
    since = seconds_since_progress(progress=state)
    if since < 0.0:
        return False
    return since <= window


def forget_progress() -> None:
    """Drop the reading. For tests, and between unrelated turns."""

    state = capture_progress()
    with state._lock:
        state._last_at = 0.0
        state._tools.clear()


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
