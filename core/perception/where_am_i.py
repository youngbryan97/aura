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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.verify import invariant

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
    aimed_at: str = "",
    observation: Mapping[str, Any] | None = None,
) -> WhereSheIs:
    """Whether the thing she means to act in is what is in front of her.

    ``lattice`` is the frame she is holding, and where she has one it settles
    the question: the thing she is acting in is the thing whose places she has
    been acting on. Everything else is a fallback for before she has been
    anywhere.

    ``aimed_at`` is the thing she is about to click or carry, as she saw it in
    this same look: where there is one, there is something in front of her.
    """
    if _current_pointer_target(observation, aimed_at):
        return WhereSheIs(True, 'the current captured target answers the pointer')
    filled = int(getattr(reading, "occupied", lambda: 0)() or 0)
    rows = int(getattr(reading, "rows", 0) or 0)
    columns = int(getattr(reading, "columns", 0) or 0)
    # Places she can see are the thing, whether or not anything is in them.
    # A board that has just been started is empty for an instant, and reading
    # that instant as "I cannot read anything here" stopped her from acting
    # in the thing she had just started (live, 2026-09-17).
    seen_places = bool(getattr(reading, "places_seen", False)) and rows >= 2 and columns >= 2
    if reading is None or (not filled and not seen_places):
        # A picture with no words in it is not nothing. Where she is about to act on a thing she saw in this very look
        # (a door, an object, a place where the pointer turns into a hand), the thing is in front of her: she is
        # aiming at it. Offline 2026-10-12 a game played among the rooms of a house had no words on its screen of play,
        # and she declined every click on the doors and things she had found for two minutes, until it was lost.
        if str(aimed_at or "").strip():
            return WhereSheIs(True, "what I am about to act on is in front of me")
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


def _current_pointer_target(observation: Mapping[str, Any] | None, label: str) -> bool:
    """A measured pointer affordance does not depend on the screen's inferred grid."""
    if not observation or observation.get('ok') is not True or not str(label or '').strip():
        return False
    from core.perception.observed_transfer import freeze_screen

    frame = freeze_screen(observation)
    if (frame.picture is None or not frame.surface or frame.bounds is None or not frame.settled or frame.occluded
            or frame.capture_at is None or not observation.get('_capture_epoch')
            or observation.get('her_window_showing') is False or observation.get('refused_because')):
        return False
    named = ' '.join(str(label).casefold().split())
    candidates = [region for field in ('layout', 'shapes', 'objects') for region in observation.get(field) or ()
                  if isinstance(region, Mapping) and ' '.join(str(region.get('text') or '').casefold().split()) == named]
    if len(candidates) != 1 or candidates[0].get('answers_the_pointer') is not True or candidates[0].get('pointer_ambiguous'):
        return False
    region = candidates[0]
    # Use the same explicit anchor preference as delivery. A malformed
    # anchor cannot regain authority through its label's centre.
    x, y = ((region.get('pointer_x'), region.get('pointer_y')) if 'pointer_x' in region or 'pointer_y' in region
            else (region.get('center_x'), region.get('center_y')))
    from core.perception.where_the_pointer_turns import has_current_cursor_sample

    return (type(x) in (float, int) and type(y) in (float, int) and 0 <= x <= 1 and 0 <= y <= 1
            and has_current_cursor_sample(observation, (x, y)))


@invariant('perception.current_pointer_targets_do_not_depend_on_a_grid', scope='perception',
           owner='core/perception/where_am_i.py', observational=False)
def _pointer_target_frame_invariant() -> tuple:
    from core.perception.observed_transfer import PixelSnapshot, ScreenReading
    from core.perception.where_the_pointer_turns import bind_cursor_samples

    region = dict(text='a current target', x=.2, y=.2, width=.2, height=.2,
                  center_x=.3, center_y=.3, answers_the_pointer=True)
    seen = ScreenReading(dict(ok=True, shapes=[region], surface_id='one', bounds=[0, 0, 8, 8],
                              capture_at=1., _capture_epoch='one', settled=True),
                         picture_snapshot=PixelSnapshot(bytes(192), 8, 8))
    bind_cursor_samples(seen, ((.3, .3),))
    assert _current_pointer_target(seen, 'a current target')
    assert not _current_pointer_target(dict(seen), 'a current target')
    assert not _current_pointer_target(seen, 'an old target')
    return ()
