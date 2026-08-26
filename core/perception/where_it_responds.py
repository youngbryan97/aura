"""Which part of the screen answers to her.

A reading of a screen is everything on it. On the page holding a game that is
the board, the score, two advertising rails, a cookie footer and a copyright
line, and the situation she reasons about is all of it — so what she recalls
about "a board like this one" is dominated by whichever advertisement was
loaded at the time, and two readings of the same board look like different
situations because the advertising rotated.

Nothing about the page says which part is the task. But something about HER
does: the part that changes when she acts is the part she is acting on. Ads
change too, and pages animate, so a single coincidence proves nothing — it
takes repetition, and the answer is the region that keeps moving right after
she does something and does not otherwise.

That is general to any screen. It finds the board in a game, the editor in a
window full of panels, the results in a page of chrome. It knows nothing
about any of them; it knows what she did and what happened next.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Responds")

#: How many acts it takes before the answer is worth using. One coincidence
#: is a coincidence.
ENOUGH_ACTS = 4

#: How much of the changing area to keep. Padding a tight band by a little
#: allows for a value appearing one row further out than any seen so far.
MARGIN = 0.03

#: How responsive a place has to be, against the most responsive place on
#: the screen, before it counts as answering to her.
#:
#: Everything on a page changes eventually — advertising rotates, clocks
#: tick, banners cycle — so "it changed" is not the test. The test is whether
#: it changes when SHE does something, and how often is only meaningful
#: beside the rest of the screen. A fixed share of her acts cannot serve
#: both: a game board where each square changes a third of the time and an
#: editor where one line changes every keystroke are the same question with
#: different numbers. Measured against the busiest place, both answer it.
ANSWERS_OFTEN = 0.34


@dataclass
class Responsive:
    """Where her actions have been having their effects."""

    acts: int = 0
    #: How often the text at each position changed on a move that had an
    #: effect.
    answered: dict[tuple[int, int], int] = field(default_factory=dict)
    #: How often it changed on a move that had NO effect. Those moves are the
    #: control: whatever still changed then was changing on its own.
    regardless: dict[tuple[int, int], int] = field(default_factory=dict)
    #: How many of each kind of move there have been, so the two rates can be
    #: compared rather than the two counts.
    effective: int = 0
    idle: int = 0

    #: How many acts in a row can have no effect before the world she is
    #: working in has stopped answering her altogether.
    #:
    #: One is a bad move. Two is a bad idea. Several in a row, with nothing
    #: anywhere on the screen responding, is not a run of bad moves — it is a
    #: thing that has ended. A finished game, a session that expired, a form
    #: already submitted, a connection that dropped.
    DEAD_AFTER = 4

    #: Acts since anything last answered her.
    unanswered: int = 0

    def settled(self) -> bool:
        return self.effective >= ENOUGH_ACTS and bool(self.answered)

    def nothing_answers(self) -> bool:
        """Whether the thing she is working in has stopped responding at all.

        Read from what happened rather than from what the screen says. A page
        that has ended says so in its own words — "Game Over", "Session
        expired", "Thanks for your submission" — and there is no list of
        those words that covers the next one. What every ending has in common
        is that nothing she does changes anything any more.
        """
        return self.unanswered >= self.DEAD_AFTER

    def band(self) -> tuple[float, float, float, float] | None:
        """The area that answers to her, as (left, top, right, bottom).

        Only the positions that answer to a good share of her acts. Everything
        on a page changes eventually, so "it changed" would widen the band to
        the whole screen given long enough.
        """
        if not self.settled():
            return None
        # How much more often a place moves when she has an effect than when
        # she does not. A place that moves either way is moving on its own.
        idle_rate = {
            where: times / float(max(1, self.idle)) for where, times in self.regardless.items()
        }
        because_of_her = {
            where: (times / float(max(1, self.effective))) - idle_rate.get(where, 0.0)
            for where, times in self.answered.items()
        }
        because_of_her = {where: rate for where, rate in because_of_her.items() if rate > 0.0}
        if not because_of_her:
            return None
        busiest = max(because_of_her.values())
        repeated = [
            where for where, rate in because_of_her.items() if rate >= busiest * ANSWERS_OFTEN
        ]
        if not repeated:
            return None
        xs = _middle(sorted(where[0] / 100.0 for where in repeated))
        ys = _middle(sorted(where[1] / 100.0 for where in repeated))
        return (
            max(0.0, xs[0] - MARGIN),
            max(0.0, ys[0] - MARGIN),
            min(1.0, xs[1] + MARGIN),
            min(1.0, ys[1] + MARGIN),
        )


#: What share of the responsive positions to keep at each edge.
#:
#: A bounding box around everything that answered is as wide as its worst
#: outlier, and a screen read by OCR always has one: a run of text jitters by
#: a pixel, a banner redraws, a word is read differently between frames.
#: Measured on a real page, eight moves of that noise stretched the answer
#: from the board to the whole window. Dropping the outermost tenth at each
#: edge keeps the part that answers and loses the strays.
TRIM = 0.1


def _middle(values: Sequence[float]) -> tuple[float, float]:
    """The range these values occupy, ignoring the strays at each end."""
    if not values:
        return (0.0, 1.0)
    if len(values) < 5:
        return (values[0], values[-1])
    cut = max(1, int(len(values) * TRIM))
    kept = values[cut : len(values) - cut] or values
    return (kept[0], kept[-1])


def _cells(observation: dict[str, Any]) -> dict[tuple[int, int], str]:
    """The reading as text by position, rounded so a pixel of drift is not a change."""
    found: dict[tuple[int, int], str] = {}
    for region in observation.get("layout") or []:
        text = str(region.get("text") or "").strip()
        if not text:
            continue
        try:
            x = float(region.get("center_x", region.get("x", 0.0)))
            y = float(region.get("center_y", region.get("y", 0.0)))
        except (TypeError, ValueError):
            continue
        found[(round(x * 100), round(y * 100))] = text
    return found


def noticed(
    state: Responsive,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    worked: bool = True,
) -> Responsive:
    """Record what changed between one reading and the next, after she acted.

    ``worked`` says whether that act had any effect. A move that changed
    nothing is the control this needs: everything that still changed across
    it was changing on its own, and a page whose advertising animates as
    often as the task does cannot be separated any other way.
    """
    was, now = _cells(before), _cells(after)
    if not was and not now:
        return state
    state.acts += 1
    changed = any(was.get(where) != now.get(where) for where in set(was) | set(now))
    state.unanswered = 0 if changed else state.unanswered + 1
    if worked:
        state.effective += 1
    else:
        state.idle += 1
    counted = state.answered if worked else state.regardless
    for where in set(was) | set(now):
        if was.get(where) != now.get(where):
            counted[where] = counted.get(where, 0) + 1
    return state


def within(observation: dict[str, Any], band: tuple[float, float, float, float] | None) -> str:
    """The reading, kept to the part that answers to her.

    Returns the whole reading when there is no answer yet, because a guess
    about where the task is would be worse than reading everything.
    """
    text = str(observation.get("text") or "")
    if band is None:
        return text
    left, top, right, bottom = band
    inside: list[str] = []
    for region in observation.get("layout") or []:
        said = str(region.get("text") or "").strip()
        if not said:
            continue
        try:
            x = float(region.get("center_x", region.get("x", 0.0)))
            y = float(region.get("center_y", region.get("y", 0.0)))
        except (TypeError, ValueError):
            continue
        if left <= x <= right and top <= y <= bottom:
            inside.append(said)
    return " ".join(inside) if inside else text


def describe(band: tuple[float, float, float, float] | None) -> str:
    if band is None:
        return ""
    left, top, right, bottom = band
    return f"the part of the screen that responds to me ({left:.2f}–{right:.2f} across, {top:.2f}–{bottom:.2f} down)"
