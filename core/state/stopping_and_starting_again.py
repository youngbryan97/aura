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
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.StoppingAndStartingAgain")

__all__ = [
    "AnInterruption",
    "WhyItStopped",
    "interrupt",
    "resume",
    "what_was_interrupted",
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


_INTERRUPTED: dict[str, AnInterruption] = {}


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
    _INTERRUPTED[one.what] = one
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
    _INTERRUPTED.pop(str(what), None)
    return one.was_about_to, one


def what_was_interrupted() -> dict[str, Any]:
    """Everything stopped and not yet picked up."""
    return {
        "interrupted": len(_INTERRUPTED),
        "resumable": sum(1 for one in _INTERRUPTED.values() if one.resumable),
        "each": {name: one.to_dict() for name, one in sorted(_INTERRUPTED.items())},
    }


def forget_everything() -> None:
    """For tests. The live runtime never calls this."""
    _INTERRUPTED.clear()
