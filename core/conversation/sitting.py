"""Where a sitting begins: the run of a conversation since the last break long enough to end one.

One definition for every part of her that asks what "now" covers: what they have watched this sitting
(core/skills/watching_together.py) and which earlier exchanges a request is read with
(core/conversation/history_reach.py).

LIVE 7 November 2026, after a day of rehearsals: her store held seven earlier exchanges asking "So what did you think
of all three?", each from a sitting of its own and each answered of three other videos. Asked it again, the request
bore on all seven by meaning, and the earlier answers came with it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

__all__ = ["A_SITTING_BREAKS_AFTER_S", "sitting_begins", "when"]

#: A quiet this long ends a sitting: the inactivity timeout that has divided sessions of use since Catledge and Pitkow
#: (1995) put it at 25.5 minutes, rounded up as it is everywhere it is used. On 7 November 2026 the rehearsals were 41
#: to 47 minutes apart, and within one, her remarks and the person's requests left no quiet over five minutes.
A_SITTING_BREAKS_AFTER_S = 30 * 60.0


def when(value: Any) -> float | None:
    """Seconds since the epoch, from an epoch number or an ISO 8601 time; None when it says neither."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def sitting_begins(times: Sequence[float | None], now: float) -> int:
    """The index of the first of ``times`` (oldest first) in the sitting that is still going at ``now``.

    ``len(times)`` when the last is a sitting ago; 0 when any time is unknown, since a break that cannot be measured
    divides nothing.
    """
    if any(at is None for at in times):
        return 0
    later = float(now)
    for index in range(len(times) - 1, -1, -1):
        at = float(times[index])  # type: ignore[arg-type]
        if later - at > A_SITTING_BREAKS_AFTER_S:
            return index + 1
        later = at
    return 0
