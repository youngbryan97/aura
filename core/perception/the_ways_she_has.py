"""core/perception/the_ways_she_has.py — the real inventory, declared.

The controller in `how_she_finds_out.py` deliberately knows nothing about
screens or people. This is where the things she can actually do to find
something out are declared, one entry each, by name and cost and outcome.

The costs are relative and they are the point of the comparison: looking at
the screen is nearly free and often ambiguous; asking is expensive and
usually settles it; running the thing costs what running it costs and is the
only one that can be wrong in a way the other two cannot. A controller
holding all three picks between them on what each would tell her, which is
the decision perception did not previously make.

Nothing here fabricates a reliability. Every way starts unmeasured, is
explored because unmeasured is uncertain rather than because a number says
so, and earns its standing from `how_it_went`.
"""

from __future__ import annotations

import logging
from typing import Any

from core.perception.how_she_finds_out import WayOfFindingOut, register_a_way

logger = logging.getLogger("Aura.Perception.TheWaysSheHas")

__all__ = ["declare_the_ways_she_has", "HOW_MUCH_EACH_COSTS"]


#: What each costs, on one scale, in the unit the value of a question is in.
#: Relative rather than absolute, and ordered by the only thing that is
#: actually comparable between them: what it takes from the person and from
#: the machine to make the observation once.
HOW_MUCH_EACH_COSTS: dict[str, float] = {
    # A frame is already being captured. Reading one more costs a read.
    "look at the screen": 0.02,
    # Reading a file is cheap and only tells her about the file.
    "read the file": 0.05,
    # Running it is the only one that can be wrong the way the world is.
    "run it and see": 0.30,
    # Asking spends his attention, which is the most expensive thing here.
    "ask him": 2.00,
}


def _look_at_the_screen(subject: str) -> str | None:
    """What the running perception loop last saw, as an outcome label.

    The last observation rather than a fresh capture, which is what makes the
    cost above honest: a frame is already being taken every couple of
    seconds, and reading the one that arrived costs a read. Capturing on
    demand would cost a capture, and calling for one from inside the event
    loop raises — an availability failure that would have been counted
    against the instrument's accuracy.

    None when the loop is not running. A sensor that is off has not observed
    anything, which is a different answer from observing an absence.
    """

    try:
        from core.perception.perception_daemon import PerceptionDaemon
    except ImportError:
        return None
    try:
        daemon = PerceptionDaemon.get_sync()
        seen = str(getattr(daemon, "last_active_window", "") or "")
    except (AttributeError, RuntimeError, OSError, TypeError, ValueError):
        return None
    if not seen:
        return None
    return "present" if str(subject).lower() in seen.lower() else "absent"


def _read_the_file(subject: str) -> str | None:
    """Whether the named thing is where it is supposed to be."""

    from pathlib import Path

    try:
        where = Path(str(subject))
        if not where.is_absolute() or not where.exists():
            return None
        return "present" if where.stat().st_size > 0 else "absent"
    except (OSError, ValueError):
        return None


def declare_the_ways_she_has() -> tuple[str, ...]:
    """Register everything she can do to find out. Idempotent.

    Re-registration keeps what each way has learned about itself, so calling
    this at every boot does not make every way look untried every morning.
    """

    declared: list[str] = []
    for name, take in (
        ("look at the screen", _look_at_the_screen),
        ("read the file", _read_the_file),
    ):
        try:
            register_a_way(
                WayOfFindingOut(
                    name=name,
                    about=(),
                    cost=HOW_MUCH_EACH_COSTS[name],
                    outcomes=("present", "absent"),
                    take=take,
                    description=name,
                )
            )
            declared.append(name)
        except (ValueError, KeyError, RuntimeError) as exc:
            logger.warning("Could not declare %s as a way of finding out: %s", name, exc)
    return tuple(declared)


def register_asking(ask: Any) -> str:
    """Declare asking him, given something that can actually ask.

    Kept separate and injected because the controller must never be able to
    ask a question by importing a module. Something that owns a channel to a
    person hands it over; nothing here reaches for one.
    """

    register_a_way(
        WayOfFindingOut(
            name="ask him",
            about=(),
            cost=HOW_MUCH_EACH_COSTS["ask him"],
            outcomes=("present", "absent"),
            take=ask,
            description="a question to the person, which spends his attention",
        )
    )
    return "ask him"
