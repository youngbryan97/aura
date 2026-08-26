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

#: What share of her acts a place has to answer to before it counts as
#: answering to her.
#:
#: Everything on a page changes eventually — advertising rotates, clocks
#: tick, banners cycle — so "it changed" is not the test. The test is whether
#: it changes when SHE does something. A board answers nearly every move; an
#: advertisement on a thirty-second rotation answers one move in fifty.
ANSWERS_OFTEN = 0.4


@dataclass
class Responsive:
    """Where her actions have been having their effects."""

    acts: int = 0
    #: How often the text at each position changed right after she acted.
    answered: dict[tuple[int, int], int] = field(default_factory=dict)

    def settled(self) -> bool:
        return self.acts >= ENOUGH_ACTS and bool(self.answered)

    def band(self) -> tuple[float, float, float, float] | None:
        """The area that answers to her, as (left, top, right, bottom).

        Only the positions that answer to a good share of her acts. Everything
        on a page changes eventually, so "it changed" would widen the band to
        the whole screen given long enough.
        """
        if not self.settled():
            return None
        needed = max(2, int(self.acts * ANSWERS_OFTEN))
        repeated = [where for where, times in self.answered.items() if times >= needed]
        if not repeated:
            return None
        xs = [where[0] / 100.0 for where in repeated]
        ys = [where[1] / 100.0 for where in repeated]
        return (
            max(0.0, min(xs) - MARGIN),
            max(0.0, min(ys) - MARGIN),
            min(1.0, max(xs) + MARGIN),
            min(1.0, max(ys) + MARGIN),
        )


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


def noticed(state: Responsive, before: dict[str, Any], after: dict[str, Any]) -> Responsive:
    """Record what changed between one reading and the next, after she acted."""
    was, now = _cells(before), _cells(after)
    if not was and not now:
        return state
    state.acts += 1
    for where in set(was) | set(now):
        if was.get(where) != now.get(where):
            state.answered[where] = state.answered.get(where, 0) + 1
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
