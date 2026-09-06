"""One lifecycle for an organ, a phase or a service, and who may start it.

AutoGPT gives every component the same lifecycle and orders them by declared
dependencies. The closure asked for the same here: start, stop, health,
dependencies and authority on one protocol, with a single supervisor.

Aura starts things in several ways — a phase registered in an order, a service
registered in a container, an organ constructed at boot — and each knows its
own dependencies. What none of them share is a way to say *when* it may start
and *who* allowed it, so boot order is a fact about the code rather than a
statement anyone made.

Two things here are not just a protocol.

The order is computed from the declarations and refuses a cycle by naming it.
A boot that deadlocks because two parts wait for each other is a bad hour; a
boot that says "affect waits for memory waits for affect" is a minute.

And a part declares the authority it needs. A supervisor that starts anything
that asks is not a supervisor, and "who allowed this to run" was a question
with no answer.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("Aura.WhatAPartOfHerDeclares")

__all__ = [
    "APart",
    "Alive",
    "TheSupervisor",
    "WhatItNeeds",
]


class Alive(StrEnum):
    """Where a part is in its life. Nothing else is a state."""

    NOT_STARTED = "not started"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    #: It was asked to start and would not. Distinct from stopped, which is a
    #: part that ran and finished.
    REFUSED = "refused"


@runtime_checkable
class WhatItNeeds(Protocol):
    """What every organ, phase and service says about itself."""

    name: str
    #: Names of the parts that must be running first.
    needs: tuple[str, ...]
    #: What it is allowed to do. A supervisor that starts anything that asks
    #: is not a supervisor.
    authority: str

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def healthy(self) -> bool:
        ...


@dataclass
class APart:
    """One part, and what has happened to it."""

    name: str
    needs: tuple[str, ...] = ()
    authority: str = "unclassified"
    alive: Alive = Alive.NOT_STARTED
    #: When it entered that state. A report without this cannot tell a part
    #: that started a second ago from one that has been wedged since boot,
    #: and both of them read as the same word.
    alive_since: float = field(default_factory=time.time)
    why_refused: str = ""
    start: Any = None
    stop: Any = None
    healthy: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "needs": list(self.needs),
            "authority": self.authority,
            "alive": str(self.alive),
            "alive_since": self.alive_since,
            "for_seconds": round(max(0.0, time.time() - self.alive_since), 3),
            "why_refused": self.why_refused,
        }


class TheSupervisor:
    """The one thing that starts parts, in an order it can explain."""

    def __init__(self, *, may_start: frozenset[str] | None = None) -> None:
        self._parts: dict[str, APart] = {}
        #: The authorities this supervisor will start. None means every one,
        #: which is what a boot uses; a narrower set is what a test uses.
        self._may_start = may_start

    def declare(self, part: APart) -> APart:
        if not part.name.strip():
            raise ValueError("a part needs a name")
        if part.authority in ("", "unclassified"):
            raise ValueError(
                f"{part.name} does not say what authority it needs; "
                "'who allowed this to run' has to have an answer"
            )
        self._parts[part.name] = part
        return part

    def the_order(self) -> list[str]:
        """The order to start them in, or a named cycle.

        Refuses by naming the cycle. A boot that deadlocks because two parts
        wait for each other is a bad hour; one that says which two is a
        minute.
        """
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(name: str, path: tuple[str, ...]) -> None:
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                loop = " waits for ".join([*path[path.index(name):], name])
                raise ValueError(f"nothing can start first: {loop}")
            state[name] = 1
            part = self._parts.get(name)
            for needed in (part.needs if part else ()):
                if needed in self._parts:
                    visit(needed, (*path, name))
                else:
                    logger.debug("%s needs %s, which nothing declared", name, needed)
            state[name] = 2
            order.append(name)

        for name in sorted(self._parts):
            visit(name, ())
        return order

    def start_everything(self) -> dict[str, Any]:
        """Start them in order. A part that will not start stops its dependents.

        Named rather than silent: a dependent skipped because its dependency
        refused is a different thing from one that refused itself, and a boot
        report that conflates them sends somebody to the wrong file.
        """
        started: list[str] = []
        refused: list[str] = []
        skipped: list[str] = []
        for name in self.the_order():
            part = self._parts[name]
            missing = [
                one
                for one in part.needs
                if self._parts.get(one) is None
                or self._parts[one].alive is not Alive.RUNNING
            ]
            if missing:
                part.alive = Alive.REFUSED
                part.alive_since = time.time()
                part.why_refused = f"waiting on {', '.join(sorted(missing))}"
                skipped.append(name)
                continue
            if self._may_start is not None and part.authority not in self._may_start:
                part.alive = Alive.REFUSED
                part.alive_since = time.time()
                part.why_refused = f"this supervisor may not start {part.authority}"
                refused.append(name)
                continue
            part.alive = Alive.STARTING
            part.alive_since = time.time()
            try:
                if callable(part.start):
                    part.start()
            except Exception as exc:  # noqa: BLE001 — one part is not the boot
                part.alive = Alive.REFUSED
                part.alive_since = time.time()
                part.why_refused = f"{type(exc).__name__}: {exc}"
                refused.append(name)
                continue
            part.alive = Alive.RUNNING
            part.alive_since = time.time()
            started.append(name)
        return {
            "started": started,
            "refused": refused,
            "skipped_because_something_it_needs_did_not_start": skipped,
        }

    def stop_everything(self) -> list[str]:
        """Stop in the reverse of the start order."""
        stopped: list[str] = []
        for name in reversed(self.the_order()):
            part = self._parts[name]
            if part.alive is not Alive.RUNNING:
                continue
            part.alive = Alive.STOPPING
            part.alive_since = time.time()
            try:
                if callable(part.stop):
                    part.stop()
            except Exception as exc:  # noqa: BLE001 — a stuck stop is not a crash
                logger.warning("%s would not stop: %s", name, exc)
            part.alive = Alive.STOPPED
            part.alive_since = time.time()
            stopped.append(name)
        return stopped

    def report(self) -> dict[str, Any]:
        parts = [one.to_dict() for one in sorted(self._parts.values(), key=lambda p: p.name)]
        for one, part in zip(parts, sorted(self._parts.values(), key=lambda p: p.name), strict=True):
            one["healthy"] = bool(part.healthy()) if callable(part.healthy) else None
        return {
            "parts": len(parts),
            "running": sum(1 for one in parts if one["alive"] == "running"),
            "refused": [one["name"] for one in parts if one["alive"] == "refused"],
            "each": parts,
        }
