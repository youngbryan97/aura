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

__all__ = ["A_SHORT_TURN_S", "short_turns"]

#: How long one thread may hold the interpreter while another waits, in seconds, while somebody has asked for short turns.
A_SHORT_TURN_S = 0.001
_TURNS: dict[str, Any] = {"asked": 0, "before": None}
_LOCK = checked_lock("interpreter_turns")


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
