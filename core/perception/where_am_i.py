"""Whether this is the place she should be acting in, worked out by looking.

Every check she had for this asked something other than the screen. Is the
application she was told about frontmost. Does the reading say which window it
was scoped to. Did the browser report the address it is on. All of them are
about the machine's bookkeeping, and all of them can be perfectly true while
she is typing into the wrong window — a reading scoped to the right
application still contains whatever that application is showing, which after
a stray click is a different page.

She can see the screen. So the question is one she can answer the way anybody
answers it: look at what is there, and say whether the thing she was acting in
is in it. The strongest evidence needs no vocabulary at all — the thing she is
acting in is the thing whose places she has been acting on, and she is holding
those places. A reading that lands in them is that thing. A reading that lands
nowhere near them is somewhere else, however correct the bookkeeping.

Before she has been anywhere, there are no places to land in, and then what
she has is the request: the words she was given for the thing she is looking
for, checked against the words in front of her.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = ["WhereSheIs", "where_am_i", "SOME_OF_IT"]

#: How much of what she was told to look for has to be in front of her before
#: a reading counts as being of that thing. Not all of it: a request names a
#: thing in the words of somebody describing it, and a screen shows the thing
#: itself, so the two overlap rather than match.
SOME_OF_IT = 0.5


@dataclass(frozen=True)
class WhereSheIs:
    """What she can tell about the place in front of her, from looking at it."""

    #: Whether the thing she means to act in is what she is looking at.
    the_thing_is_here: bool
    #: What she saw that says so, in words she can say out loud.
    because: str
    #: What this looks like instead, when it is not the thing.
    looks_like: str = ""

    def said(self) -> str:
        """The answer as a person would give it."""
        if self.the_thing_is_here:
            return f"This is the right place: {self.because}."
        if self.looks_like:
            return f"This does not look like where I should be — {self.because}, {self.looks_like}."
        return f"This does not look like where I should be: {self.because}."


def _words_in(said: Any) -> set[str]:
    """The words of a thing, for comparing one description with another."""
    return {
        word
        for word in re.findall(r"[a-z0-9]+", str(said or "").lower())
        if len(word) > 2
    }


def _what_this_looks_like(reading: Any) -> str:
    """What kind of thing this is, from its shape rather than its subject.

    Enough to tell somebody what she is looking at instead. A thing laid out
    is regular and full of its own places; a page of prose is one long column;
    an empty reading is a reading of nothing.
    """
    rows = int(getattr(reading, "rows", 0) or 0)
    columns = int(getattr(reading, "columns", 0) or 0)
    filled = int(getattr(reading, "occupied", lambda: 0)() or 0)
    if not filled:
        return "there is nothing in front of me"
    if columns <= 1:
        return f"I am looking at a single column of {filled} things"
    if rows <= 1:
        return f"I am looking at a single row of {filled} things"
    return f"I am looking at a {rows} by {columns} thing with {filled} in it"


def where_am_i(
    reading: Any,
    *,
    lattice: Any = None,
    asked_for: str = "",
) -> WhereSheIs:
    """Whether the thing she means to act in is what is in front of her.

    ``lattice`` is the frame she is holding, and where she has one it settles
    the question: the thing she is acting in is the thing whose places she has
    been acting on. Everything else is a fallback for before she has been
    anywhere.
    """
    filled = int(getattr(reading, "occupied", lambda: 0)() or 0)
    if reading is None or not filled:
        return WhereSheIs(False, "I cannot read anything here", _what_this_looks_like(reading))

    if lattice is not None and getattr(lattice, "held", False):
        rows = int(getattr(reading, "rows", 0) or 0)
        columns = int(getattr(reading, "columns", 0) or 0)
        if (rows, columns) == (lattice.rows, lattice.columns):
            return WhereSheIs(
                True,
                f"the {lattice.rows} by {lattice.columns} thing I have been "
                f"acting in is in front of me, with {filled} of its places filled",
            )
        return WhereSheIs(
            False,
            f"the {lattice.rows} by {lattice.columns} thing I have been acting in is not here",
            _what_this_looks_like(reading),
        )

    # Before she has been anywhere, she can recognise the thing but not miss it.
    #
    # A request names where she is going, not what is in front of her at the
    # start: "get to a 256 tile" is about a tile that does not exist yet, and
    # a check that refused every screen not already showing the goal would
    # refuse the first keystroke of every task. Recognising the words she was
    # given is worth saying; failing to is not yet evidence of anything, and
    # absence of a control is not a finding.
    wanted = _words_in(asked_for)
    here = _words_in(getattr(reading, "as_text", lambda: "")())
    shared = wanted & here
    if shared and len(shared) >= max(1, round(len(wanted) * SOME_OF_IT)):
        return WhereSheIs(True, f"I can see {', '.join(sorted(shared))} in front of me")
    return WhereSheIs(
        True, "I have not acted anywhere yet, so I have nothing to tell this apart from"
    )
