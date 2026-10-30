"""A run gives the machine back between turns, and takes it up where it stopped.

Bryan has one computer. When he opens her app, a run measuring her gives the
room back, and later carries on from where it was. Stopping a process in the
middle of a turn is not safe: a phase that overruns its budget is abandoned on
the machine's clock (core/subject/clock.py), so a turn held for an hour would
come back recorded as a failure the arms never earned. So a run parks itself
between turns, where nothing is in flight.

The park directory (`AURA_PARK_DIR`) is the whole protocol. A run writes its
pid there, with the longest turn it has taken, so whoever holds it knows what
to wait for. A file named HOLD in that directory asks every run to park: the
next turn to begin waits, before it does anything, until the file is gone.
Without the variable nothing is installed, so the desktop and the test suite
never see this.

No signal is used. A process that stopped itself with SIGSTOP under a zsh
subshell with more to run after it (every stage of the decisive chain) was
continued at once, because zsh continues a stopped job's whole process group,
and a SIGCONT meant to release one run would release all of them (30 October).
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from core.runtime.wall_clock import monotonic_time, wall_time

__all__ = ["HOLD", "offer_to_park", "park_if_asked", "note_turn", "parked_seconds"]

logger = logging.getLogger("Aura.Subject.Parking")

#: The file that asks every run in the park directory to wait between turns.
HOLD = "HOLD"

#: How often a parked run looks for the hold to be lifted. The run is between
#: turns, so this only sets how soon it carries on; it measures nothing.
_LOOK_EVERY_S = 0.5

_STATE: dict[str, Any] = {"directory": None, "longest_turn_s": 0.0, "parked_s": 0.0, "parks": 0}


def _marker() -> Path | None:
    directory = _STATE["directory"]
    return None if directory is None else Path(directory) / f"{os.getpid()}.json"


def _write(parked: bool) -> None:
    from core.subject.archive import write_json

    directory = _STATE["directory"]
    if directory is None:
        return
    write_json(Path(directory), f"{os.getpid()}.json", {
        "pid": os.getpid(),
        "command": " ".join(sys.argv)[:400],
        "parked": parked,
        "longest_turn_s": round(float(_STATE["longest_turn_s"]), 3),
        "parks": int(_STATE["parks"]),
        "parked_s": round(float(_STATE["parked_s"]), 1),
        "written_at": wall_time(),
    })


def _forget() -> None:
    marker = _marker()
    if marker is not None:
        try:
            marker.unlink()
        except FileNotFoundError:
            pass


def offer_to_park(directory: str | os.PathLike[str] | None = None) -> bool:
    """Let this process be held between turns, if a park directory is named. True when it can be."""
    chosen = directory if directory is not None else os.environ.get("AURA_PARK_DIR", "")
    if not chosen or threading.current_thread() is not threading.main_thread():
        return False
    if _STATE["directory"] is not None:
        return True
    Path(chosen).mkdir(parents=True, exist_ok=True)
    _STATE["directory"] = str(chosen)
    _write(parked=False)
    atexit.register(_forget)
    return True


def note_turn(seconds: float) -> None:
    """The longest turn so far, so whoever holds the runs knows how long a park can take to arrive.

    Written only when it has doubled, so a run writes a handful of times.
    """
    seconds = float(seconds)
    if _STATE["directory"] is None or seconds <= max(1.0, 2.0 * float(_STATE["longest_turn_s"])):
        if seconds > float(_STATE["longest_turn_s"]):
            _STATE["longest_turn_s"] = seconds
        return
    _STATE["longest_turn_s"] = seconds
    _write(parked=False)


def park_if_asked() -> float:
    """Wait here while the hold is in place. Seconds parked, or 0.0."""
    directory = _STATE["directory"]
    if directory is None or not (Path(directory) / HOLD).exists():
        return 0.0
    _STATE["parks"] += 1
    _write(parked=True)
    logger.warning("parked between turns (park %d) until %s is removed", _STATE["parks"], Path(directory) / HOLD)
    began = monotonic_time()
    while (Path(directory) / HOLD).exists():
        time.sleep(_LOOK_EVERY_S)
    seconds = monotonic_time() - began
    _STATE["parked_s"] += seconds
    _write(parked=False)
    logger.warning("carried on after %.0f s parked", seconds)
    return seconds


def parked_seconds() -> float:
    """All the time this process has spent parked."""
    return float(_STATE["parked_s"])
