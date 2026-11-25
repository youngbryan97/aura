"""How long one thread may hold the interpreter while another waits for it.

Everything she does in her own process takes turns with one interpreter. A thread that works something out in Python
holds it for five milliseconds at a time, and a loop that hands it back at every small wait, as her event loop does, gets
one small thing done in each of them. So what is working in Python beside something that cannot wait asks for shorter
turns while it does.
"""
from __future__ import annotations

import sys
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["A_SHORT_TURN_S", "HER_TURN_S", "short_turns", "take_short_turns"]

#: How long one thread may hold the interpreter while another waits, in seconds, while somebody has asked for short turns.
A_SHORT_TURN_S = 0.001
#: The same for her whole process, in seconds. Measured 24 November 2026 on this host: beside one thread working in Python,
#: an event loop gets 144 of its passes a second done with turns of five milliseconds, 657 with one and 1,231 with a half,
#: where it does 40,358 alone, and its timers are 15.1, 4.2 and 2.8 ms late. Three threads at work in Python took no longer
#: for it (0.73 s, 0.92, 0.87); two of them calling into numpy on forty numbers at a time beside a third took twice as long
#: (0.94 s, 1.55, 1.63), and 2.00 with turns of a fifth of a millisecond.
HER_TURN_S = 0.0005
_TURNS: dict[str, Any] = {"asked": 0, "before": None}
_LOCK = checked_lock("interpreter_turns")


def take_short_turns() -> float:
    """Her whole process changes hands every ``HER_TURN_S``; what it was. Her event loop waits at every read, write and
    select it makes, and until then each wait cost it five milliseconds of whatever else was working. LIVE 24 November 2026,
    19:08, it was starved for 14.4 s, on 6% of a core, beside one thread working out how a board's pieces move."""
    was = sys.getswitchinterval()
    sys.setswitchinterval(min(HER_TURN_S, was))
    return was


class short_turns:  # noqa: N801 - read as what it does, in a with
    """The interpreter changes hands every millisecond instead of every five, and as it was after. Measured offline
    2026-10-13 with three other threads busy beside play: 99 ms a picture with turns of five milliseconds, 46 with turns
    of one, 41 alone at most. LIVE 24 November 2026 working a small ending out between games starved her event loop for
    24.7 s, on 4% of a core, while the thread that did it had 94% of one."""

    def __enter__(self) -> short_turns:
        with _LOCK:
            if _TURNS["asked"] == 0:
                _TURNS["before"] = sys.getswitchinterval()
                sys.setswitchinterval(min(A_SHORT_TURN_S, _TURNS["before"]))
            _TURNS["asked"] += 1
        return self

    def __exit__(self, *_: Any) -> None:
        with _LOCK:
            _TURNS["asked"] = max(0, _TURNS["asked"] - 1)
            if _TURNS["asked"] == 0 and _TURNS["before"] is not None:
                sys.setswitchinterval(_TURNS["before"])
                _TURNS["before"] = None
