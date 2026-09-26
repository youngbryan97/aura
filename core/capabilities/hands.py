"""Her hands: a chunk played on the keyboard and mouse, one slot per interval between looks.

The grammar is in core/agency/what_hands_do.py. This plays it. Keys go down
when a slot first names them and come up when a slot stops naming them, so a
key held across ten slots is one press held, not ten taps. The mouse moves by
the slot's delta from wherever it is, with the delta set on the event too,
because a game that turns its camera reads how far the mouse moved and not
where it ended up. Clicks land at their share of the window.

Three things are never left to chance. Before every slot the window she
means has to still be in front, and if it is not she stops: keys sent to
whatever the person clicked since are keys sent to the wrong thing. Every key
she pressed is released however the chunk ends, including by an error. And
the time each slot really took is kept, because a slot that runs late is a
look taken later than she thinks.

Posting goes through a sink so the whole of this can be tested without
touching the machine. The one that touches it posts Quartz events to the
stream a keyboard and a mouse would post to.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.agency.what_hands_do import Chunk

logger = logging.getLogger("Aura.Hands")

__all__ = ["Played", "QuartzHands", "Sink", "play"]


class Sink(Protocol):
    """Where hand events go."""

    def key(self, name: str, down: bool) -> bool: ...

    def move_by(self, dx: int, dy: int) -> bool: ...

    def click(self, button: str, x: float, y: float) -> bool: ...

    def scroll(self, dx: int, dy: int) -> bool: ...


@dataclass
class Played:
    """What a chunk actually did."""

    slots: int = 0
    events: int = 0
    #: Why it stopped before its end, when it did.
    stopped: str = ""
    #: How far behind its own clock the chunk finished, in seconds.
    late_s: float = 0.0
    #: The keys that were down at some point, all released by the end.
    pressed: set[str] = field(default_factory=set)


async def play(
    chunk: Chunk,
    *,
    sink: Sink,
    window: tuple[float, float, float, float],
    still_ours: Callable[[], bool],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> Played:
    """Play ``chunk`` into ``window`` (left, top, width, height in points).

    ``still_ours`` answers whether the window she means is still the one in
    front; it is asked before every slot.
    """
    left, top, width, height = window
    played = Played()
    down: set[str] = set()
    began = clock()
    try:
        for index, slot in enumerate(chunk.slots):
            if not still_ours():
                played.stopped = "the window she was acting in is no longer in front"
                break
            for name in sorted(down - slot.held):
                if sink.key(name, False):
                    played.events += 1
                down.discard(name)
            for name in sorted(slot.held - down):
                if sink.key(name, True):
                    played.events += 1
                    down.add(name)
                    played.pressed.add(name)
            if slot.moved != (0, 0) and sink.move_by(*slot.moved):
                played.events += 1
            if slot.click is not None:
                button, x, y = slot.click
                if sink.click(button, left + x * width, top + y * height):
                    played.events += 1
            if slot.scrolled != (0, 0) and sink.scroll(*slot.scrolled):
                played.events += 1
            played.slots = index + 1
            due = began + (index + 1) * chunk.slot_s
            wait = due - clock()
            if wait > 0.0:
                await sleep(wait)
    finally:
        # Every key she pressed comes up, however the chunk ended.
        for name in sorted(down):
            try:
                if sink.key(name, False):
                    played.events += 1
            except Exception as exc:  # noqa: BLE001 - a key left down is worse than a lost error
                logger.warning("could not release %r: %s", name, exc)
    played.late_s = max(0.0, clock() - (began + played.slots * chunk.slot_s))
    return played


# ── the sink that touches the machine ────────────────────────────────────

#: Key codes for the keys a game reads by position, by the name a person uses.
#: The ANSI layout's virtual key codes; named keys come from window_server.
_POSITIONS: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29,
    "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "shift": 56, "alt": 58, "ctrl": 59,
}

#: What a held modifier adds to every other key's event, as Quartz flags.
_FLAGS = {"shift": 0x00020000, "ctrl": 0x00040000, "alt": 0x00080000}


class QuartzHands:
    """Posts hand events where a keyboard and a mouse post theirs."""

    def __init__(self) -> None:
        from core.capabilities import window_server

        self._window_server = window_server
        self._quartz = window_server._quartz()
        self._modifiers: set[str] = set()

    def _code(self, name: str) -> int | None:
        found = _POSITIONS.get(name)
        return found if found is not None else self._window_server.key_code(name)

    def key(self, name: str, down: bool) -> bool:
        quartz, code = self._quartz, self._code(name)
        if quartz is None or code is None:
            return False
        if name in _FLAGS:
            (self._modifiers.add if down else self._modifiers.discard)(name)
        event = self._window_server._key_event(quartz, name, code, down)
        flags = 0
        for held in self._modifiers:
            flags |= _FLAGS[held]
        if flags:
            quartz.CGEventSetFlags(event, quartz.CGEventGetFlags(event) | flags)
        quartz.CGEventPost(quartz.kCGHIDEventTap, event)
        self._window_server.her_hands_posted()
        return True

    def move_by(self, dx: int, dy: int) -> bool:
        quartz = self._quartz
        if quartz is None:
            return False
        at = quartz.CGEventGetLocation(quartz.CGEventCreate(None))
        event = quartz.CGEventCreateMouseEvent(
            None, quartz.kCGEventMouseMoved, (at.x + dx, at.y + dy), quartz.kCGMouseButtonLeft
        )
        # What a game turning its camera reads.
        quartz.CGEventSetIntegerValueField(event, quartz.kCGMouseEventDeltaX, int(dx))
        quartz.CGEventSetIntegerValueField(event, quartz.kCGMouseEventDeltaY, int(dy))
        quartz.CGEventPost(quartz.kCGHIDEventTap, event)
        self._window_server.her_hands_posted()
        return True

    def click(self, button: str, x: float, y: float) -> bool:
        quartz = self._quartz
        if quartz is None:
            return False
        right = button == "right"
        downs = quartz.kCGEventRightMouseDown if right else quartz.kCGEventLeftMouseDown
        ups = quartz.kCGEventRightMouseUp if right else quartz.kCGEventLeftMouseUp
        which = quartz.kCGMouseButtonRight if right else quartz.kCGMouseButtonLeft
        for kind in (downs, ups):
            quartz.CGEventPost(
                quartz.kCGHIDEventTap, quartz.CGEventCreateMouseEvent(None, kind, (x, y), which)
            )
        self._window_server.her_hands_posted()
        return True

    def scroll(self, dx: int, dy: int) -> bool:
        quartz = self._quartz
        if quartz is None:
            return False
        event = quartz.CGEventCreateScrollWheelEvent(None, quartz.kCGScrollEventUnitLine, 2, int(dy), int(dx))
        quartz.CGEventPost(quartz.kCGHIDEventTap, event)
        self._window_server.her_hands_posted()
        return True
