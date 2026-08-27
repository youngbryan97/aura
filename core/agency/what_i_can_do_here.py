"""Which of her actions do anything here, found out rather than declared.

A solver written for one thing is handed its action set: four moves, named in
the source. She was handed hers the same way — a tuple of four arrow keys in
core/runtime/watched_goal.py — and that is the last large thing about an
unfamiliar world that somebody else was still establishing for her. Everything
else she now works out: which window she is in, which part of it answers, how
it moves when she pushes it, what changes on its own, and what a good situation
looks like.

So the action set is a hypothesis like every other one here. She starts with
whatever she was told, tries the rest when what she was told is not working,
and keeps what answers. A key that never changes anything is not one of her
actions in this world, whoever wrote it down.

One line is not crossed. Only inputs that commit to nothing are tried without
knowing what they will do. An arrow moves a view or a piece and can be undone
by moving back; Return and Space activate whatever has focus, which may be a
Send, a Buy or a Delete. She may find out what moves things. She may not find
out what a button does by pressing it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "COMMITS_TO_NOTHING",
    "ENOUGH_TO_JUDGE",
    "WhatWorksHere",
    "worth_trying",
]

logger = logging.getLogger("Aura.WhatICanDoHere")

#: Inputs she may try without knowing what they will do.
#:
#: Every one of these moves something — a view, a selection, a piece — and
#: every one is undone by moving back. Nothing here activates a control.
COMMITS_TO_NOTHING: tuple[str, ...] = ("up", "down", "left", "right")

#: Inputs that act on whatever has focus. Never tried to find out what they
#: do: what they do is press whatever button is under them, and that is a
#: decision rather than an experiment.
COMMITS_TO_SOMETHING: tuple[str, ...] = ("return", "enter", "space", "tab", "delete")

#: How many times an input has to have done nothing before it is not one of
#: her actions here. Once is a bad moment — a board can refuse a direction it
#: will accept two moves later. Several times running is a fact about the
#: world rather than about the moment.
ENOUGH_TO_JUDGE = 4


def worth_trying(told: Sequence[str] = ()) -> tuple[str, ...]:
    """Everything she could try here, hers first and the rest after.

    What she was told comes first because somebody usually knows, and the
    others follow because sometimes nobody does.
    """
    named = [str(key or "").strip().lower() for key in told]
    ordered = [key for key in named if key in COMMITS_TO_NOTHING]
    ordered += [key for key in COMMITS_TO_NOTHING if key not in ordered]
    return tuple(ordered)


@dataclass
class WhatWorksHere:
    """Which inputs have done anything, and which have never done anything."""

    #: What she was told her actions were, if anything.
    told: tuple[str, ...] = ()
    did_something: dict[str, int] = field(default_factory=dict)
    did_nothing: dict[str, int] = field(default_factory=dict)
    #: Said once, when what she was told turns out to be wrong.
    said_it_differs: bool = False

    # ── finding out ──────────────────────────────────────────────────────

    def tried(self, key: str, changed: bool) -> None:
        """One input, and whether the world answered it."""
        name = str(key or "").strip().lower()
        if not name:
            return
        if changed:
            self.did_something[name] = self.did_something.get(name, 0) + 1
            self.did_nothing.pop(name, None)
        else:
            self.did_nothing[name] = self.did_nothing.get(name, 0) + 1

    # ── using it ─────────────────────────────────────────────────────────

    def dead(self) -> tuple[str, ...]:
        """Inputs that have done nothing, every time, enough times to say so."""
        return tuple(
            key
            for key, times in sorted(self.did_nothing.items())
            if times >= ENOUGH_TO_JUDGE and key not in self.did_something
        )

    def works(self) -> tuple[str, ...]:
        """Inputs that have done something at least once."""
        return tuple(sorted(self.did_something))

    def untried(self) -> tuple[str, ...]:
        """Inputs she could try and has not."""
        seen = set(self.did_something) | set(self.did_nothing)
        return tuple(key for key in worth_trying(self.told) if key not in seen)

    def available(self) -> tuple[str, ...]:
        """What to offer her now.

        What she was told, minus anything that has proved inert, plus
        anything else worth trying while some of what she was told is not
        working. A world where the named keys do the job never widens.
        """
        told = tuple(key for key in self.told if key not in self.dead())
        if told and not self.dead():
            return told
        wider = [key for key in worth_trying(self.told) if key not in self.dead()]
        if wider and set(wider) != set(self.told) and not self.said_it_differs:
            self.said_it_differs = True
            logger.info(
                "what she was told her moves were (%s) is not what works here (%s)",
                ", ".join(self.told) or "nothing",
                ", ".join(wider),
            )
        return tuple(wider)

    def still_finding_out(self) -> bool:
        """Whether anything is left to try."""
        return bool(self.untried())

    def says(self) -> str:
        """What she has found out, for whoever has to answer for it."""
        works = self.works()
        dead = self.dead()
        if not works and not dead:
            return "which of my moves do anything here is not worked out yet"
        said = f"these do something here: {', '.join(works) or 'none so far'}"
        if dead:
            said = f"{said}; these never do: {', '.join(dead)}"
        return said

    # ── keeping it ───────────────────────────────────────────────────────

    def as_memory(self) -> dict[str, object]:
        return {
            "told": list(self.told),
            "did_something": dict(self.did_something),
            "did_nothing": dict(self.did_nothing),
        }

    @classmethod
    def from_memory(cls, held: object, told: Sequence[str] = ()) -> "WhatWorksHere":
        """What worked here last time, as a starting point rather than a fact."""
        named = tuple(str(key or "").strip().lower() for key in told)
        if not isinstance(held, dict):
            return cls(told=named)

        def counts(value: object) -> dict[str, int]:
            if not isinstance(value, dict):
                return {}
            return {
                str(key): int(times)
                for key, times in value.items()
                if isinstance(times, (int, float))
            }

        return cls(
            told=named or tuple(str(key) for key in (held.get("told") or ())),
            did_something=counts(held.get("did_something")),
            did_nothing=counts(held.get("did_nothing")),
        )
