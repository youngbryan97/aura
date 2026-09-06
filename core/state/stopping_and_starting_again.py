"""Interrupting work so that starting it again is not starting over.

LangGraph makes interrupt and resume first class. The closure asked for the
same, and Aura has both halves apart: a turn can be cancelled (`whose_turn_it_is`)
and a state can be checkpointed (`a_checkpoint_and_its_writes`). What it did
not have is the thing that makes an interrupt useful — a record of WHERE the
work stopped, so the next attempt starts there.

An interrupt without that is a cancellation. The work is gone and the next
attempt repeats it, which is why a long task interrupted near the end costs
the same as one interrupted at the start.

So an interruption carries three things: the checkpoint the work had reached,
what it was about to do, and why it stopped. Resuming restores the first,
hands back the second, and refuses if the reason is one that has not gone
away — resuming into the same wall is how a retry loop is born.

**The record has to outlive the process.** The first version of this kept the
interruptions in a module-level dict, which made the whole module a promise it
could not keep: it said "resume from the checkpoint" while the metadata saying
what was interrupted and what to do next died with the process. The one case
that matters most — the runtime stopped — is exactly the case where a
process-local registry has already been lost by the time anyone asks.

So the registry is a file, written through the gateway, read back on first
use. A store that cannot be read is not fatal: the work is then genuinely
unresumable and saying so is better than crashing the thing that was trying to
recover.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.StoppingAndStartingAgain")

__all__ = [
    "AnInterruption",
    "WhyItStopped",
    "interrupt",
    "resume",
    "what_was_interrupted",
    "where_it_is_kept",
    "reload_from_disk",
    "forget_everything",
    "MOST_KEPT",
]


class WhyItStopped(StrEnum):
    """Why work stopped, and whether resuming is sensible.

    The distinction is the point. Waiting for a person is resumable the moment
    they answer; running out of budget is resumable only with more budget; a
    refusal is not resumable at all, and pretending otherwise builds a loop.
    """

    ASKED_A_PERSON = "waiting for a person"
    OUT_OF_BUDGET = "out of budget"
    OUT_OF_TIME = "out of time"
    THE_RUNTIME_STOPPED = "the runtime stopped"
    REFUSED = "refused"


#: Reasons that do not go away by themselves. Resuming into one of these is
#: resuming into the same wall.
_WILL_NOT_CLEAR: frozenset[WhyItStopped] = frozenset({WhyItStopped.REFUSED})


@dataclass
class AnInterruption:
    """Where the work stopped, what it was about to do, and why."""

    what: str
    why: WhyItStopped
    #: The checkpoint the work had reached. Restoring it is what makes
    #: resuming cheaper than starting over.
    checkpoint: str = ""
    #: What it was about to do. Handed back on resume so the next attempt
    #: does not have to work it out again.
    was_about_to: Any = None
    said: str = ""
    at: float = field(default_factory=time.time)
    resumed: int = 0

    @property
    def resumable(self) -> bool:
        return self.why not in _WILL_NOT_CLEAR

    def to_dict(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "why": str(self.why),
            "checkpoint": self.checkpoint,
            "was_about_to": self.was_about_to,
            "said": self.said,
            "at": self.at,
            "resumed": self.resumed,
            "resumable": self.resumable,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AnInterruption | None":
        """Read a record back. None where the reason is no longer a reason.

        A stored ``why`` this build does not recognise means the file was
        written by a version that knew something this one does not, and
        guessing at it would resume work under a rule nobody wrote.
        """
        try:
            why = WhyItStopped(str(row.get("why") or ""))
        except ValueError:
            return None
        return cls(
            what=str(row.get("what") or ""),
            why=why,
            checkpoint=str(row.get("checkpoint") or ""),
            was_about_to=row.get("was_about_to"),
            said=str(row.get("said") or ""),
            at=float(row.get("at") or 0.0),
            resumed=int(row.get("resumed") or 0),
        )


#: The most interruptions kept. Work that stopped and was never picked up is
#: not evidence of anything after a while, and an unbounded file on the
#: recovery path is a file that eventually stops being readable.
MOST_KEPT = 200

_INTERRUPTED: dict[str, AnInterruption] = {}
_LOADED = False
_LOCK = checked_lock("stopping_and_starting_again")


def where_it_is_kept() -> Path:
    """The file the registry lives in.

    Under the state root, so a test with its own AURA_STATE_ROOT never reads
    or writes the live one.
    """
    from core.runtime.state_ownership import state_root

    return Path(state_root()) / "interruptions.json"


def _load_if_needed() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    path = where_it_is_kept()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        # Unreadable is not fatal. The work is unresumable, which is a worse
        # answer than resuming and a much better one than taking down the
        # thing that was trying to recover.
        logger.warning("interruptions could not be read from %s: %s", path, exc)
        return
    rows = raw.get("interrupted") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        one = AnInterruption.from_dict(row)
        if one is not None and one.what:
            _INTERRUPTED[one.what] = one
    if _INTERRUPTED:
        logger.info(
            "%d interruption(s) survived the last process", len(_INTERRUPTED)
        )


_GENERATION = 0
_WRITTEN = 0


def _snapshot() -> tuple[int, str]:
    """The registry as bytes, and the generation it belongs to.

    Called with the lock held. Does no I/O: an fsync under a lock is how a
    runtime freezes, and lockdep refuses it — correctly, and it caught this
    the first time the write was inside.
    """
    global _GENERATION
    _GENERATION += 1
    kept = sorted(_INTERRUPTED.values(), key=lambda one: one.at)[-MOST_KEPT:]
    payload = {
        "schema": "aura.interruptions.v1",
        "written_at": time.time(),
        "generation": _GENERATION,
        "interrupted": [one.to_dict() for one in kept],
    }
    return _GENERATION, json.dumps(payload, indent=2, sort_keys=True, default=str)


def _write(generation: int, body: str) -> None:
    """Put it on disk. Called with no lock held, and never raises.

    A generation counter rather than a second lock: two writers can leave the
    lock in one order and reach the disk in the other, and the older one
    landing last would undo a stop that has already happened. A write older
    than what has already landed is dropped.
    """
    global _WRITTEN
    if generation <= _WRITTEN:
        return
    try:
        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        path = where_it_is_kept()
        gateway.ensure_directory(path.parent, source="stopping_and_starting_again")
        gateway.write_text(path, body, source="stopping_and_starting_again")
        _WRITTEN = max(_WRITTEN, generation)
    except Exception as exc:  # noqa: BLE001 - recording a stop must not stop anything
        logger.warning("interruptions could not be written down: %s", exc)


def reload_from_disk() -> int:
    """Read the registry again, discarding what is in memory.

    For a process that starts after another one stopped, and for tests.
    """
    global _LOADED
    with _LOCK:
        _INTERRUPTED.clear()
        _LOADED = False
        _load_if_needed()
        return len(_INTERRUPTED)


def interrupt(
    what: str,
    why: WhyItStopped,
    *,
    checkpoint: str = "",
    was_about_to: Any = None,
    said: str = "",
) -> AnInterruption:
    """Record that this work stopped here.

    Interrupting the same work twice replaces the record: the second stop is
    where it is now, and keeping the first would resume into the past.
    """
    one = AnInterruption(
        what=str(what),
        why=why,
        checkpoint=str(checkpoint),
        was_about_to=was_about_to,
        said=str(said),
    )
    with _LOCK:
        _load_if_needed()
        _INTERRUPTED[one.what] = one
        generation, body = _snapshot()
    _write(generation, body)
    logger.info("%s stopped: %s", one.what, one.why)
    return one


def resume(what: str, *, channels: Any = None) -> tuple[Any, AnInterruption | None]:
    """Pick the work up where it stopped.

    Returns what it was about to do, and the interruption it came from. The
    checkpoint is restored where one was recorded and a store was given.

    Refuses a reason that will not clear. Resuming into the same wall is how a
    retry loop is born, and the caller has to decide something different
    instead.
    """
    with _LOCK:
        _load_if_needed()
        one = _INTERRUPTED.get(str(what))
    if one is None:
        return None, None
    if not one.resumable:
        logger.info("%s will not resume: %s", one.what, one.why)
        return None, one
    if one.checkpoint and channels is not None:
        try:
            channels.restore(one.checkpoint)
        except Exception as exc:  # noqa: BLE001 — a missing checkpoint is not a crash
            logger.warning(
                "%s could not restore %s: %s", one.what, one.checkpoint, exc
            )
    one.resumed += 1
    with _LOCK:
        _INTERRUPTED.pop(str(what), None)
        generation, body = _snapshot()
    _write(generation, body)
    return one.was_about_to, one


def what_was_interrupted() -> dict[str, Any]:
    """Everything stopped and not yet picked up."""
    with _LOCK:
        _load_if_needed()
        each = {name: one.to_dict() for name, one in sorted(_INTERRUPTED.items())}
        resumable = sum(1 for one in _INTERRUPTED.values() if one.resumable)
        count = len(_INTERRUPTED)
    return {
        "interrupted": count,
        "resumable": resumable,
        "kept_at": str(where_it_is_kept()),
        "survives_the_process": True,
        "each": each,
    }


def forget_everything() -> None:
    """For tests. The live runtime never calls this."""
    global _LOADED
    with _LOCK:
        _INTERRUPTED.clear()
        _LOADED = True
        generation, body = _snapshot()
    _write(generation, body)
