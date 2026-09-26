"""What her hands do on a keyboard and a mouse, written as one line she can read and a machine can play.

Until now she could tap one key per look. That is enough for a board and
nothing like enough for a world: walking is a key held down while the camera
turns, a jump is a key pressed in the middle of a run, and aiming is the mouse
moving a little at a time between looks. SIMA 2 writes its actions as text —
key names, clicks and relative mouse moves, frame by frame — and a fixed
parser turns the text into input events (arXiv 2512.04797, §3.2). ByteDance's
Game-TARS showed one such grammar serving the desktop, the web and games
alike, and doing better for the mixture.

So a chunk is a run of slots, one for each interval between looks. A slot
says which keys are down during it, how far the mouse moves, where a button
clicks and how far the page scrolls. A key named in two slots running stays
down across both; one missing from the next slot comes up. An empty slot is
written ".", and is how a hand waits.

    w | w shift | w mouse(40,0) | . | click(0.41,0.62) | done

Two words end a chunk. ``done`` says what she was asked for is done, so the
hands stop and she says so. ``think`` hands the next decision back to her
language, because something needs more than a reflex.

Nothing is guessed. A token this does not know is refused with the reason,
because a hand that plays a misread chunk presses keys nobody meant. Command
is never a key here: every application on this machine quits on Command-Q.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["KEYS", "NEVER", "Chunk", "NotAHand", "Slot", "chunk_of", "hands_read"]

#: Keys a hand may hold, by the name a person uses.
KEYS: frozenset[str] = frozenset(
    {
        "up", "down", "left", "right",
        "return", "enter", "tab", "space", "escape", "backspace", "delete",
        "shift", "ctrl", "alt",
        *"abcdefghijklmnopqrstuvwxyz0123456789",
    }
)

#: Keys that are never pressed, whatever asks. Command quits, closes and
#: deletes in every application; function keys move whole desktops.
NEVER: frozenset[str] = frozenset({"cmd", "command", "super", "win", "fn", "globe"})

#: Other names people give the same keys.
_SAME_KEY = {"control": "ctrl", "option": "alt", "opt": "alt", "esc": "escape", "spacebar": "space"}

_MOUSE = re.compile(r"^mouse\((-?\d+),(-?\d+)\)$")
_CLICK = re.compile(r"^click\((?:(left|right),)?(\d*\.?\d+),(\d*\.?\d+)\)$")
_SCROLL = re.compile(r"^scroll\((-?\d+),(-?\d+)\)$")


class NotAHand(ValueError):  # noqa: N818 - named for what it says, like the rest here
    """A chunk that cannot be played, and why."""


@dataclass(frozen=True)
class Slot:
    """What the hands do for one interval between looks."""

    held: frozenset[str] = frozenset()
    #: How far the mouse moves, in points, from wherever it is.
    moved: tuple[int, int] = (0, 0)
    #: A button and where it clicks, as shares of the window across and down.
    click: tuple[str, float, float] | None = None
    #: How far the view scrolls, in lines.
    scrolled: tuple[int, int] = (0, 0)

    def empty(self) -> bool:
        return not self.held and self.moved == (0, 0) and self.click is None and self.scrolled == (0, 0)

    def as_text(self) -> str:
        said = sorted(self.held)
        if self.moved != (0, 0):
            said.append(f"mouse({self.moved[0]},{self.moved[1]})")
        if self.click is not None:
            button, x, y = self.click
            where = f"{x:g},{y:g}"
            said.append(f"click({where})" if button == "left" else f"click({button},{where})")
        if self.scrolled != (0, 0):
            said.append(f"scroll({self.scrolled[0]},{self.scrolled[1]})")
        return " ".join(said) or "."


@dataclass(frozen=True)
class Chunk:
    """A run of slots, played one interval each."""

    slots: tuple[Slot, ...] = field(default_factory=tuple)
    #: How long each slot lasts, in seconds: the interval between her looks.
    slot_s: float = 0.1
    #: What she was asked for is done.
    done: bool = False
    #: The next decision is for her language, not her hands.
    think: bool = False

    def keys(self) -> frozenset[str]:
        return frozenset().union(*(slot.held for slot in self.slots)) if self.slots else frozenset()

    def as_text(self) -> str:
        said = [slot.as_text() for slot in self.slots]
        if self.done:
            said.append("done")
        if self.think:
            said.append("think")
        return " | ".join(said)


def _a_key(token: str) -> str:
    name = _SAME_KEY.get(token, token)
    if name in NEVER:
        raise NotAHand(f"{token!r} is never pressed: it quits, closes or moves the desktop")
    if name not in KEYS:
        raise NotAHand(f"{token!r} is not a key a hand here can hold")
    return name


def _a_slot(said: str) -> Slot:
    held: set[str] = set()
    moved = (0, 0)
    click: tuple[str, float, float] | None = None
    scrolled = (0, 0)
    for token in said.replace("+", " ").split():
        token = token.strip().lower()
        if token in {"", "."}:
            continue
        found = _MOUSE.match(token)
        if found:
            moved = (moved[0] + int(found.group(1)), moved[1] + int(found.group(2)))
            continue
        found = _CLICK.match(token)
        if found:
            x, y = float(found.group(2)), float(found.group(3))
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise NotAHand(f"{token!r} clicks outside the window")
            if click is not None:
                raise NotAHand("two clicks in one slot")
            click = (found.group(1) or "left", x, y)
            continue
        found = _SCROLL.match(token)
        if found:
            scrolled = (scrolled[0] + int(found.group(1)), scrolled[1] + int(found.group(2)))
            continue
        held.add(_a_key(token))
    return Slot(frozenset(held), moved, click, scrolled)


def hands_read(text: str, *, slot_s: float) -> Chunk:
    """A chunk from its written form. Raises NotAHand, with the reason, for anything it cannot play."""
    if slot_s <= 0.0:
        raise NotAHand("a slot has to last some time")
    slots: list[Slot] = []
    done = think = False
    parts = [part.strip() for part in str(text or "").split("|")]
    for index, part in enumerate(parts):
        words = part.lower().split()
        if words and words[-1] in {"done", "think"}:
            if index != len(parts) - 1:
                raise NotAHand(f"nothing may follow {words[-1]!r}")
            done, think = words[-1] == "done", words[-1] == "think"
            part = " ".join(words[:-1])
            if not part:
                break
        slots.append(_a_slot(part))
    return Chunk(tuple(slots), float(slot_s), done=done, think=think)


def chunk_of(*slots: Slot, slot_s: float, done: bool = False, think: bool = False) -> Chunk:
    """A chunk made from slots in code rather than read from text."""
    return Chunk(tuple(slots), float(slot_s), done=done, think=think)
