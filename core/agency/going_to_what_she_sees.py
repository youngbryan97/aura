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

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.agency.what_hands_do import KEYS, Chunk, Slot

__all__ = ["GoingTo", "Sighting", "a_cue_to_press", "reads_as_a_thing", "seen_named"]


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


#: Words that make a run of text the interface talking — a prompt, a status,
#: a sentence — rather than the name of a thing in the world.
_TALK = re.compile(r"\b(press|hold|tap|click|open|opened|closed|is|are|was|the|to)\b", re.IGNORECASE)


def reads_as_a_thing(region: dict[str, Any]) -> bool:
    """Whether a run of text on screen names a thing, whole.

    Two ways it does not. A sentence: "The Chest is open" is the world saying
    something, and live, it made her own success message a second chest to
    choose between. And a word cut by the edge of the frame: live, a chest
    half out of view was read as "hest" and she looked all the way round for
    one. A glyph cut by the frame sits within half its own width of the
    edge, so a box that comes nearer than that may be missing letters. A
    whole letter was too wide a margin: up close, where letters are large, it
    turned away a whole word sitting comfortably off the middle, and she lost
    a crate she was walking to.
    """
    said = " ".join(str(region.get("text", "")).split())
    if not said or len(said.split()) > 3 or _TALK.search(said):
        return False
    wide = float(region.get("width", 0.0))
    left = float(region.get("x", float(region.get("center_x", 0.5)) - wide / 2.0))
    half_a_letter = wide / max(1, len(said)) / 2.0
    return left > half_a_letter and left + wide < 1.0 - half_a_letter


def seen_named(layout: Sequence[dict[str, Any]], named: str) -> list[Sighting]:
    """Everything on screen that names a thing and whose words include every word of ``named``."""
    wanted = set(_words(named))
    if not wanted:
        return []
    found: list[Sighting] = []
    for region in layout or ():
        if not reads_as_a_thing(region):
            continue
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
    #: Whether to aim where a moving thing will be when she arrives rather
    #: than where it is. Off: measured 2026-09-26 in thirty generated worlds
    #: with a thing circling at under her pace, walking while steering at
    #: where it is caught it in 20, and leading by its drift over the time to
    #: arrive caught it in 11. The drift carried over twenty looks aims at the
    #: edge of the view and she loses it; the right lead is an angle, and that
    #: needs the field of view in degrees, which nothing measures yet.
    leads: bool = False
    #: Where the thing was at the last look.
    _last: float | None = None
    #: How far the whole picture slid since the last look, as her eyes measured it.
    _slid: float | None = None
    #: How far the thing moved on its own between looks, each time it was measured.
    _drifts: list[float] = field(default_factory=list)

    def _where_it_will_be(self, sight: Sighting) -> float:
        """Across, where the thing will be at the next look, on a constant-velocity guess.

        What she sees move between two looks is her own turning plus the
        thing's own motion. Her turning she knows: she slid the view by a
        share she chose. What is left is the thing moving, and a thing seen
        moving at some pace is aimed for where that pace takes it by the next
        look. SIMA 2 is weakest at exactly this — combat, 25% against 64% for
        people — and a reactive policy aims where the target was.
        """
        if self._last is not None and self._slid is not None:
            # The picture's own slide is measured, not assumed from the turn
            # she meant: assumed, whatever her gain got wrong read as the thing
            # moving, and a still thing was led off the screen. What of its
            # motion is not the picture's is its own.
            self._drifts.append(sight.across - (self._last + self._slid))
        self._slid = None
        if not self.leads or len(self._drifts) < 3:
            return sight.across
        drift = sum(self._drifts) / len(self._drifts)
        spread = math.sqrt(sum((d - drift) ** 2 for d in self._drifts) / (len(self._drifts) - 1))
        # A thing that is not moving shows a drift of nothing give or take its
        # reading; it is led only when the drift clears twice its own standard
        # error, the ordinary line between a motion and noise.
        if abs(drift) <= 2.0 * spread / math.sqrt(len(self._drifts)):
            return sight.across
        # Aim where it will be when she gets there, not one look on: a thing
        # moving across is caught by heading for where it is going. How many
        # looks away she is comes from her own walking: a step that makes the
        # view grow by g leaves her g/(g-1) steps from it. Never further off
        # than half a view, because past that she aims at something she can
        # no longer see.
        horizon = 1.0
        if self.growths and self.growths[-1] > 1.0:
            grew = self.growths[-1]
            horizon = grew / (grew - 1.0)
        lead = drift * horizon
        if abs(lead) > 0.5:
            lead = 0.5 if lead > 0 else -0.5
        return sight.across + lead

    def _the_cue_is_ours(self, layout: Sequence[dict[str, Any]], seen: list[Sighting]) -> bool:
        """Whether a prompt on screen belongs to the thing she is going to.

        A prompt that names a thing belongs to that thing. One that names
        nothing belongs to whatever is in front, so it is hers only when her
        thing is. Measured in generated worlds, with another thing standing
        between her and hers: its prompt came up as she passed, she pressed
        it, and used the wrong thing.
        """
        wanted = set(_words(self.named))
        for region in layout or ():
            said = str(region.get("text", ""))
            if not _A_CUE.search(said):
                continue
            words = set(_words(said))
            if wanted <= words:
                return True
            others = {
                word
                for other in layout or ()
                if other is not region and reads_as_a_thing(other)
                for word in _words(other.get("text", ""))
            } - wanted
            if words & others:
                return False
            return len(seen) == 1 and abs(seen[0].across - 0.5) <= max(seen[0].wide / 2.0, 1e-3)
        return False

    def next_chunk(self, layout: Sequence[dict[str, Any]], *, slot_s: float, slots: int) -> Chunk:
        """What the hands do between this look and the next."""
        seen = seen_named(layout, self.named)
        cue = a_cue_to_press(layout)
        if cue is not None and self._the_cue_is_ours(layout, seen):
            key, hold = cue
            self.ended = f"the screen said to {'hold' if hold else 'press'} {key}"
            pressed = [Slot(frozenset({key}))] * (slots if hold else 1)
            return Chunk(tuple(pressed), slot_s, done=True)
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
        off = self._where_it_will_be(sight) - 0.5
        self._last = sight.across
        # In front means inside its own width of the middle: a wide thing is
        # faced sooner than a narrow one, and nothing here picks a tolerance.
        if abs(off) > max(sight.wide / 2.0, 1e-3):
            travel = self.turn_for(-off)
            # Once it has been walked toward, she keeps walking as she turns,
            # the way a person runs and steers at once: anything in view is
            # within half the field of straight ahead, so every step still
            # brings her nearer. Facing first and walking after, she never
            # walked at a thing that kept moving: live in generated worlds,
            # two hundred turns and one step.
            held = frozenset({self.walks}) if self.heights else frozenset()
            if held:
                self.heights.append(sight.high)
            return Chunk((Slot(held, moved=(travel, 0)),), slot_s)
        if self._stopped_getting_nearer(sight):
            self.ended = f"{self.named!r} stopped getting nearer"
            return Chunk((), slot_s, done=True)
        self.heights.append(sight.high)
        return Chunk(tuple(Slot(frozenset({self.walks})) for _ in range(max(1, slots))), slot_s)


    def walked(self, grew: float) -> None:
        """How much the whole view grew over the walk just made, as her body measured it."""
        self.growths.append(float(grew))

    def slid(self, share: float) -> None:
        """How far the whole picture slid since the last look, in shares of its width."""
        self._slid = float(share)

    def _stopped_getting_nearer(self, sight: Sighting) -> bool:
        """Two walks running that brought her no nearer.

        The box drawn round a few letters of text is a pixel taller or
        shorter from one reading to the next, and live, on the first trip,
        two readings of that noise ended a walk that was getting nearer at
        five per cent a step; the whole view's growth is what steadies it.
        """
        # Both have to agree. The view growing says she walked, not that she
        # walked toward this: chasing a thing that moved round her, the wall
        # ahead grew while the thing did not, and neither alone is nearness.
        # So two walks running where neither the view nor the thing grew.
        thing_grew = not (len(self.heights) >= 2 and sight.high <= self.heights[-1] <= self.heights[-2])
        if self.growths:
            view_grew = not (len(self.growths) >= 2 and all(grew <= 1.0 for grew in self.growths[-2:]))
            return not view_grew and not thing_grew
        return not thing_grew


def turn_by_share(body: Any, small_wide: int) -> Callable[[float], int]:
    """Mouse travel for a slide of a share of the view, from what her hands were measured doing."""
    return lambda share: int(body.turn_for(share * small_wide))
