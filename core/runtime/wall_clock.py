"""The machine's clock, for a time that crosses from one process to another.

An experiment run installs a clock of its own as `time.time`
(core/subject/clock.py), so that two arms of an intervention see one timeline,
and a restore rewinds it with everything else. A time sent to another process
is read there against that process's clock. The model client stamped the
worker's deadline with the rewound clock, so in a whole run every arm after an
anchor's first reached the worker already past its deadline and was refused
(`deadline_exceeded_before_decode`): on 23 September no report arm after the
first got an answer from her cortex, in any whole run.

The experiment clock replaces `time.time` only. This reads the real-time clock
beneath it, which is the one every process on the machine shares.
"""

from __future__ import annotations

import time

__all__ = ["wall_time"]


def wall_time() -> float:
    """Seconds since the epoch on the machine's clock, whatever `time.time` has been replaced with."""
    return time.clock_gettime(time.CLOCK_REALTIME)
