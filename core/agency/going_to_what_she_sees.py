"""Going to something she can see, in a world seen through a camera, and using it.

Navigation and using things are most of what SIMA 2 was measured on (its
skill profile, arXiv 2512.04797, Fig. 7), and both come down to a loop any
person runs without thinking: find the thing, turn until it is in front of
you, walk while it gets bigger, and do what the world says to do once you are
there. The world usually says so in words — "Press E to open", "Hold F" —
and SIMA 2 reads them the same way (The Gunk's ABSORB and HOLD cues, §4.2.2).

Nothing here knows a game. What turns the camera and what walks were measured
by `how_the_view_moves`; what is where comes from reading the screen; which
key to press comes from the words on it. The loop stops for three reasons,
each of which the world gives: a prompt appeared and was answered, the thing
stopped getting bigger however she walked, or it is not there to go to.

Where more than one thing on screen answers to the name, she does not pick
one. Which of two doors was meant is the person's to say.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.agency.what_hands_do import KEYS, Chunk, Slot

__all__ = ["GoingTo", "Sighting", "a_cue_to_press", "seen_named"]


@dataclass(frozen=True)
class Sighting:
    """Something read on screen, where it is, as shares of the frame."""

    text: str
    across: float
    down: float
    wide: float
    high: float


def _words(said: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(said or "").lower())


def seen_named(layout: Sequence[dict[str, Any]], named: str) -> list[Sighting]:
    """Everything on screen whose words include every word of ``named``."""
    wanted = set(_words(named))
    if not wanted:
        return []
    found: list[Sighting] = []
    for region in layout or ():
        if wanted <= set(_words(region.get("text", ""))):
            found.append(
                Sighting(
                    str(region.get("text", "")),
                    float(region.get("center_x", 0.5)),
                    float(region.get("center_y", 0.5)),
                    float(region.get("width", 0.0)),
                    float(region.get("height", 0.0)),
                )
            )
    return found


#: How a world tells you what to press: "Press E", "Hold [F]", "E) Open".
_A_CUE = re.compile(
    r"\b(?P<how>press|hold|tap)\s*\[?\s*(?P<key>[a-z0-9]|space|enter|return|tab|shift|ctrl|alt)\s*\]?(?![a-z0-9])",
    re.IGNORECASE,
)


def a_cue_to_press(layout: Sequence[dict[str, Any]]) -> tuple[str, bool] | None:
    """The key the screen says to press, and whether it says to hold it. None when it says none."""
    for region in layout or ():
        found = _A_CUE.search(str(region.get("text", "")))
        if found:
            key = found.group("key").lower()
            key = "return" if key == "enter" else key
            if key in KEYS:
                return key, found.group("how").lower() == "hold"
    return None


@dataclass
class GoingTo:
    """One trip to one thing she can see.

    ``turn_for`` says how far to move the mouse to slide the view by a share of
    its width; ``walks`` is the key measured to walk forward. Both come from
    `how_the_view_moves`, and neither is guessed.
    """

    named: str
    turn_for: Callable[[float], int]
    walks: str
    #: How tall the thing looked each time she walked toward it.
    heights: list[float] = field(default_factory=list)
    #: How much the whole view grew over each walk, where it was measured.
    growths: list[float] = field(default_factory=list)
    #: Why the trip ended, once it has.
    ended: str = ""

    def next_chunk(self, layout: Sequence[dict[str, Any]], *, slot_s: float, slots: int) -> Chunk:
        """What the hands do between this look and the next."""
        cue = a_cue_to_press(layout)
        if cue is not None:
            key, hold = cue
            self.ended = f"the screen said to {'hold' if hold else 'press'} {key}"
            pressed = [Slot(frozenset({key}))] * (slots if hold else 1)
            return Chunk(tuple(pressed), slot_s, done=True)
        seen = seen_named(layout, self.named)
        if len(seen) > 1:
            self.ended = f"{len(seen)} things answer to {self.named!r}: " + "; ".join(
                sorted(sight.text for sight in seen)
            )
            return Chunk((), slot_s, think=True)
        if not seen:
            if not self.heights:
                self.ended = f"nothing on screen answers to {self.named!r}"
                return Chunk((), slot_s, think=True)
            # It was there and has gone from view: walked past it or turned
            # off it. Looking again is the next decision, not a reflex.
            self.ended = f"{self.named!r} went out of view"
            return Chunk((), slot_s, think=True)
        sight = seen[0]
        off = sight.across - 0.5
        # In front means inside its own width of the middle: a wide thing is
        # faced sooner than a narrow one, and nothing here picks a tolerance.
        if abs(off) > max(sight.wide / 2.0, 1e-3):
            travel = self.turn_for(-off)
            return Chunk((Slot(moved=(travel, 0)),), slot_s)
        if self._stopped_getting_nearer(sight):
            self.ended = f"{self.named!r} stopped getting nearer"
            return Chunk((), slot_s, done=True)
        self.heights.append(sight.high)
        return Chunk(tuple(Slot(frozenset({self.walks})) for _ in range(max(1, slots))), slot_s)


    def walked(self, grew: float) -> None:
        """How much the whole view grew over the walk just made, as her body measured it."""
        self.growths.append(float(grew))

    def _stopped_getting_nearer(self, sight: Sighting) -> bool:
        """Two walks running that brought her no nearer.

        The whole view's growth says it best where it was measured: a box
        drawn round a few letters of text is a pixel taller or shorter from
        one reading to the next, and live, on the first trip, two readings of
        that noise ended a walk that was getting nearer at five per cent a
        step. The box is what is left where nothing measured the view.
        """
        if self.growths:
            return len(self.growths) >= 2 and all(grew <= 1.0 for grew in self.growths[-2:])
        return len(self.heights) >= 2 and sight.high <= self.heights[-1] <= self.heights[-2]


def turn_by_share(body: Any, small_wide: int) -> Callable[[float], int]:
    """Mouse travel for a slide of a share of the view, from what her hands were measured doing."""
    return lambda share: int(body.turn_for(share * small_wide))
