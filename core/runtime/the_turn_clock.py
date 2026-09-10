"""One turn, one clock, and the extension the waiting half can see.

A turn is priced twice. The HTTP route sets a wall-clock SLA when the request
is admitted, from the lane's state and a guess at how long the prompt will be.
The answer clock inside the inference gate prices the same turn again, later,
from the prompt that actually got built and the rates the worker just measured.
The second number is the better one — it is measured rather than guessed — and
it was invisible to the first.

LIVE, 2026-09-10: "deadline 103s to 251s", the gate extended its own request
deadline to honour it, and the route gave up at its own budget while the cortex
was still generating. The person was told the answer took too long to finish
cleanly. The gate had 251 seconds; nobody was waiting that long.

The clock here is that one object. The route opens it with what it granted; the
gate asks it for more time and says why; the route reads what was adopted. It
lives in a ContextVar holding a mutable clock rather than a value, because a
task copies the context down and cannot hand a new value back up — the object
it copies is the same object either way.

A turn with no clock open is normal: background work, a tool call, a test. Every
reader handles ``None`` by using the budget it already had.
"""

from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import dataclass, field

#: The ceiling no extension may pass, and the one the gate already honours.
#: Imported lazily so this module stays free of the response policy's imports.
def _the_hard_ceiling() -> float:
    try:
        from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S

        return float(USER_FACING_COMPLETION_DEADLINE_MAX_S)
    except (ImportError, TypeError, ValueError):
        return 480.0


@dataclass
class TheTurnClock:
    """What this turn was granted, what it has since been shown it needs."""

    granted_s: float
    started_at: float = field(default_factory=time.monotonic)
    adopted_s: float = 0.0
    why: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.granted_s = max(0.0, float(self.granted_s or 0.0))
        self.adopted_s = self.granted_s

    def ask_for(self, seconds: float, why: str = "") -> float:
        """Ask the turn for more wall clock. Returns what it now stands at.

        Only ever upward, and never past the ceiling: a deadline inside a wait
        that gives up sooner is two numbers disagreeing again, with the outer
        one winning silently.
        """
        try:
            wanted = float(seconds or 0.0)
        except (TypeError, ValueError):
            return self.adopted_s
        with self._lock:
            capped = min(wanted, _the_hard_ceiling())
            if capped > self.adopted_s:
                self.adopted_s = capped
                self.why = str(why or "")
            return self.adopted_s

    def budget(self) -> float:
        with self._lock:
            return max(self.granted_s, self.adopted_s)

    def remaining(self, *, reserve: float = 0.0) -> float:
        return max(0.0, self.budget() - (time.monotonic() - self.started_at) - reserve)

    def was_extended(self) -> bool:
        with self._lock:
            return self.adopted_s > self.granted_s

    def to_dict(self) -> dict[str, object]:
        with self._lock:
            return {
                "granted_s": round(self.granted_s, 1),
                "adopted_s": round(self.adopted_s, 1),
                "extended": self.adopted_s > self.granted_s,
                "why": self.why,
            }


_THE_CLOCK: contextvars.ContextVar[TheTurnClock | None] = contextvars.ContextVar(
    "aura_the_turn_clock", default=None
)


def open_a_turn_clock(granted_s: float) -> TheTurnClock:
    """Start this turn's clock at what the route granted it."""
    clock = TheTurnClock(granted_s=granted_s)
    _THE_CLOCK.set(clock)
    return clock


def the_turn_clock() -> TheTurnClock | None:
    """This turn's clock, or None where no turn opened one."""
    return _THE_CLOCK.get()


def ask_the_turn_for(seconds: float, why: str = "") -> float:
    """Ask for more wall clock, wherever there is a turn to ask.

    Returns what the turn now stands at, or 0.0 where there is no turn — a
    caller with no clock keeps whatever budget it already had.
    """
    clock = _THE_CLOCK.get()
    if clock is None:
        return 0.0
    return clock.ask_for(seconds, why)


def close_the_turn_clock() -> None:
    _THE_CLOCK.set(None)


__all__ = [
    "TheTurnClock",
    "ask_the_turn_for",
    "close_the_turn_clock",
    "open_a_turn_clock",
    "the_turn_clock",
]
