"""The window server, asked directly, from inside the process.

Everything a loop acting on an application needs to know about its window is
already known to the window server: where the window is, which process owns
it, what is drawn in it, and what sits in front of it. It was being asked
through AppleScript and a `screencapture` subprocess. Spawning either costs
about a third of a second before any work is done, every glance left a picture
of the person's screen on disk that then had to be deleted under governance,
and a question about an application that was not running waited out a
timeout. Measured 2026-09-17: fourteen seconds of every cycle were the window's
bounds, asked of System Events about a process that did not exist.

Here the same questions take milliseconds and leave nothing behind:

* :func:`windows` and :func:`window_of` read the window list.
* :func:`capture` takes the pixels of ONE window by its number, so a window
  sitting over hers does not end up in the picture of hers.
* :func:`post_keys` delivers keystrokes to the process that owns a window
  rather than to whatever happens to be in front, which is the difference
  between a key with an address and a key without one.

Nothing here knows what any application is for. Every function answers for
any window on the machine, and every one of them returns an empty answer
rather than raising when the window server cannot be reached, because a
missing answer is something the caller can reason about and an exception in
the middle of an act is not.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.WindowServer")

__all__ = [
    "Window",
    "capture",
    "front_owner",
    "key_code",
    "post_keys",
    "window_of",
    "windows",
]

#: Keys by the name a person uses, as the virtual key codes the keyboard sends.
#: Named keys only; a printable character is typed, not pressed.
_KEY_CODES: dict[str, int] = {
    "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51,
    "backspace": 51, "escape": 53, "esc": 53,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
}


def key_code(name: str) -> int | None:
    """The virtual key code for a named key, or None for a key with no name."""
    return _KEY_CODES.get(str(name or "").strip().lower())


@dataclass(frozen=True)
class Window:
    """One window as the window server reports it. Pixels are points."""

    number: int
    owner: str
    pid: int
    title: str
    left: float
    top: float
    width: float
    height: float
    layer: int
    on_screen: bool

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (int(self.left), int(self.top), int(self.width), int(self.height))

    @property
    def area(self) -> float:
        return self.width * self.height


def _quartz() -> Any:
    try:
        import Quartz  # noqa: PLC0415

        return Quartz
    # not a failure: the module is optional here, and its absence is the answer.
    except ImportError:
        return None


def windows(*, on_screen_only: bool = True) -> tuple[Window, ...]:
    """Every window the window server will describe, front to back."""
    quartz = _quartz()
    if quartz is None:
        return ()
    option = (
        quartz.kCGWindowListOptionOnScreenOnly
        if on_screen_only
        else quartz.kCGWindowListOptionAll
    )
    try:
        listed = quartz.CGWindowListCopyWindowInfo(option, quartz.kCGNullWindowID) or []
    except (AttributeError, RuntimeError, TypeError, ValueError) as why:
        logger.debug("the window server did not answer: %s", why)
        return ()
    found: list[Window] = []
    for one in listed:
        try:
            box = one.get("kCGWindowBounds") or {}
            found.append(
                Window(
                    number=int(one.get("kCGWindowNumber", 0) or 0),
                    owner=str(one.get("kCGWindowOwnerName", "") or ""),
                    pid=int(one.get("kCGWindowOwnerPID", 0) or 0),
                    title=str(one.get("kCGWindowName", "") or ""),
                    left=float(box.get("X", 0.0)),
                    top=float(box.get("Y", 0.0)),
                    width=float(box.get("Width", 0.0)),
                    height=float(box.get("Height", 0.0)),
                    layer=int(one.get("kCGWindowLayer", 0) or 0),
                    on_screen=bool(one.get("kCGWindowIsOnscreen", False)),
                )
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return tuple(found)


def _folded(name: str) -> str:
    return " ".join(str(name or "").replace(".app", "").split()).casefold()


def _names_the_owner(wanted: str, owner: str) -> int:
    """How closely a name somebody used names a window's owner. 0 is not at all.

    A person says "2048" or "Chrome" and the window server says "2048 Game" or
    "Google Chrome". The name an application calls itself is rarely the name it
    is called by, so a whole-word containment either way counts, with the exact
    name preferred when there is one.
    """
    asked, owned = _folded(wanted), _folded(owner)
    if not asked or not owned:
        return 0
    if asked == owned:
        return 3
    if f" {asked} " in f" {owned} " or f" {owned} " in f" {asked} ":
        return 2
    return 0


def window_of(app: str, *, on_screen_only: bool = False) -> Window | None:
    """The main window of the application someone named, or None.

    Ordinary windows only, and the largest of the best-named owner's, because
    an application's toolbars, sheets and hidden helper windows are windows
    too and none of them is the thing a person means.
    """
    best: tuple[int, int, float, Window] | None = None
    for one in windows(on_screen_only=on_screen_only):
        if one.layer != 0 or one.width < 50 or one.height < 50:
            continue
        match = _names_the_owner(app, one.owner)
        if not match:
            continue
        key = (match, int(one.on_screen), one.area, one)
        if best is None or key[:3] > best[:3]:
            best = key
    return best[3] if best else None


def front_owner() -> str:
    """Whose ordinary window is in front of every other ordinary window."""
    for one in windows(on_screen_only=True):
        if one.layer == 0 and one.owner and one.width >= 50 and one.height >= 50:
            return one.owner
    return ""


def capture(window: Window) -> Any:
    """The pixels of one window as a BGR array, or None.

    Taken by window number, so whatever is drawn over the window is not in the
    picture. A rectangle of the display is a different question: it answers
    with whatever happens to be on top, which is how a reading of her game
    once came back full of another application's headlines.
    """
    quartz = _quartz()
    if quartz is None or window is None:
        return None
    try:
        import numpy as np  # noqa: PLC0415

        image = quartz.CGWindowListCreateImage(
            quartz.CGRectNull,
            quartz.kCGWindowListOptionIncludingWindow,
            int(window.number),
            quartz.kCGWindowImageBoundsIgnoreFraming
            | quartz.kCGWindowImageNominalResolution,
        )
        if image is None:
            return None
        wide = int(quartz.CGImageGetWidth(image))
        tall = int(quartz.CGImageGetHeight(image))
        if wide < 2 or tall < 2:
            return None
        per_row = int(quartz.CGImageGetBytesPerRow(image))
        raw = quartz.CGDataProviderCopyData(quartz.CGImageGetDataProvider(image))
        pixels = np.frombuffer(bytes(raw), dtype=np.uint8)
        pixels = pixels[: per_row * tall].reshape(tall, per_row)[:, : wide * 4]
        pixels = pixels.reshape(tall, wide, 4)
        info = int(quartz.CGImageGetBitmapInfo(image))
        little = (info & quartz.kCGBitmapByteOrderMask) == quartz.kCGBitmapByteOrder32Little
        alpha = int(quartz.CGImageGetAlphaInfo(image))
        alpha_first = alpha in (
            quartz.kCGImageAlphaPremultipliedFirst,
            quartz.kCGImageAlphaFirst,
            quartz.kCGImageAlphaNoneSkipFirst,
        )
        if little and alpha_first:
            return np.ascontiguousarray(pixels[:, :, :3])  # BGRA
        if not little and alpha_first:
            return np.ascontiguousarray(pixels[:, :, [3, 2, 1]])  # ARGB
        return np.ascontiguousarray(pixels[:, :, [2, 1, 0]])  # RGBA
    except (AttributeError, RuntimeError, TypeError, ValueError, ImportError) as why:
        logger.debug("could not take the pixels of window %s: %s", getattr(window, "number", "?"), why)
        return None


#: What a keyboard's arrow keys carry and a key made from its code alone does
#: not: they sit on the numeric pad and the function layer. Some applications
#: read an arrow by those, and a synthetic one without them is a key they do
#: not recognise as an arrow.
_ARROW_FLAGS = 0x00200000 | 0x00800000
_ARROWS = frozenset({"up", "down", "left", "right"})


def _key_event(quartz: Any, name: str, code: int, down: bool) -> Any:
    event = quartz.CGEventCreateKeyboardEvent(None, code, down)
    if str(name or "").strip().lower() in _ARROWS:
        quartz.CGEventSetFlags(event, _ARROW_FLAGS)
    return event


def type_keys(keys: Sequence[str], *, between_s: float = 0.0) -> int:
    """Send named keys the way a keyboard does. Returns how many were sent.

    Into the stream typing goes into, so they reach whatever is in front and
    nothing else. The caller makes sure what is in front is what it means to
    type into; an application that ignores keys addressed to it still takes
    these, as it takes a person's.
    """
    quartz = _quartz()
    if quartz is None:
        return 0
    sent = 0
    for name in keys:
        code = key_code(name)
        if code is None:
            break
        try:
            quartz.CGEventPost(quartz.kCGHIDEventTap, _key_event(quartz, name, code, True))
            quartz.CGEventPost(quartz.kCGHIDEventTap, _key_event(quartz, name, code, False))
        except (AttributeError, RuntimeError, TypeError, ValueError) as why:
            logger.debug("a key did not go: %s", why)
            break
        sent += 1
        if between_s > 0.0 and sent < len(keys):
            time.sleep(between_s)
    return sent


def seconds_since_someone_touched_it() -> float:
    """Seconds since the keyboard or the mouse last did anything, as the hardware reports it.

    Infinite where it cannot be read, which reads as nobody there; the caller
    that is about to take the front decides what that is worth.
    """
    quartz = _quartz()
    if quartz is None:
        return float("inf")
    try:
        return float(
            quartz.CGEventSourceSecondsSinceLastEventType(
                quartz.kCGEventSourceStateHIDSystemState, quartz.kCGAnyInputEventType
            )
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return float("inf")


def owns_the_front(app: str) -> bool:
    """Whether the ordinary window in front belongs to the application someone named."""
    front = front_owner()
    return bool(front) and _names_the_owner(app, front) > 0


def post_keys(pid: int, keys: Sequence[str], *, between_s: float = 0.0) -> int:
    """Send named keys to one process. Returns how many were sent.

    Addressed to the process, so a key goes to the application it was meant
    for whatever is in front, and the person can keep using the rest of the
    machine while it happens. A key with no known code stops the batch there,
    because a sequence with a hole in it is a different sequence.
    """
    quartz = _quartz()
    if quartz is None or int(pid or 0) <= 0:
        return 0
    sent = 0
    for name in keys:
        code = key_code(name)
        if code is None:
            break
        try:
            quartz.CGEventPostToPid(int(pid), _key_event(quartz, name, code, True))
            quartz.CGEventPostToPid(int(pid), _key_event(quartz, name, code, False))
        except (AttributeError, RuntimeError, TypeError, ValueError) as why:
            logger.debug("a key to %s did not go: %s", pid, why)
            break
        sent += 1
        if between_s > 0.0 and sent < len(keys):
            time.sleep(between_s)
    return sent
