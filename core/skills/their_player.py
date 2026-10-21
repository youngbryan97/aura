"""What the player in their browser says of itself, read through her Accessibility access: how far into what it plays
it is, how long that is, and whether it offers to skip it.

LIVE 20 October 2026, rehearsal 7 on YouTube (Kurzgesagt, 9:14): three and a half minutes of adverts played first. The
watching's clock was the time since the window opened, so at 8:54 by that clock she said "Twenty seconds left", and
then "it just ended. We made it all the way to 9:14" with three and a half minutes still to play. "Skip" was on screen
in most looks and never pressed: the window was looked for in the accessibility tree by its title and not found.

A web player reports its place the way the WAI-ARIA practices give it to anyone using assistive technology: a seek
slider whose value text reads "2:15 of 9:13". While an advert plays in the same player, the slider is the advert's, so
a length that is not the length the page stated for its video says something else is playing. Nothing here knows a
site.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PlayerReading", "ax_window_for", "press", "read_their_player", "seconds_of", "skip_on_screen"]

_CLOCK = r"\d{1,2}(?::\d{2}){1,2}"
_PLACE = re.compile(rf"({_CLOCK})\s*(?:of|/)\s*({_CLOCK})", re.IGNORECASE)
#: How many of a window's accessibility elements are read: a video site's page is a few thousand.
ELEMENTS_READ = 6000


def seconds_of(clock: str) -> float:
    """Seconds in a clock reading such as "9:13" or "1:02:03"."""
    total = 0.0
    for part in str(clock).split(":"):
        total = total * 60 + float(part)
    return total


@dataclass
class PlayerReading:
    """One reading of their player: where it is and how long what it plays is (None where no seek slider was found),
    and the skip control on screen, if one is."""

    at: float | None = None
    length: float | None = None
    skip_words: str = ""
    skip: Any = field(default=None, repr=False)
    window_found: bool = False
    #: Its pause control, where it offers one on screen: what plays is playing.
    pause: Any = field(default=None, repr=False)


def skip_on_screen(elements: list[tuple[str, str, tuple[float, float], tuple[float, float]]],
                   bounds: dict[str, Any]) -> int | None:
    """Which of ``elements`` (role, words, position, size) is a control offering to skip what plays first, inside the
    window's ``bounds`` on screen; None when none is. "On screen" is read from where the element is: a page keeps a
    "Skip navigation" link above its top for keyboards (LIVE 20 October 2026, at y -877 in a window whose top was 33)."""
    left, top = float(bounds.get("X") or 0), float(bounds.get("Y") or 0)
    right, bottom = left + float(bounds.get("Width") or 0), top + float(bounds.get("Height") or 0)
    for index, (role, words, (x, y), (width, height)) in enumerate(elements):
        if role in ("AXButton", "AXLink") and re.match(r"\s*skip\b", words, re.IGNORECASE) and width > 1 and \
                height > 1 and left <= x and x + width <= right and top <= y and y + height <= bottom:
            return index
    return None


def _read(element: Any, name: str) -> Any:
    from ApplicationServices import AXUIElementCopyAttributeValue

    error, value = AXUIElementCopyAttributeValue(element, name, None)
    return None if error else value


def _frame(element: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    from ApplicationServices import AXValueGetValue, kAXValueCGPointType, kAXValueCGSizeType

    position, size = _read(element, "AXPosition"), _read(element, "AXSize")
    if position is None or size is None:
        return None
    _ok, point = AXValueGetValue(position, kAXValueCGPointType, None)
    _ok, extent = AXValueGetValue(size, kAXValueCGSizeType, None)
    return (float(point.x), float(point.y)), (float(extent.width), float(extent.height))


#: What a timeline calls itself, and how its value text speaks: LIVE 20 October 2026 YouTube's in Safari was "Seek
#: slider", value 2 of 608, saying "Childhood classified ad 0 Minutes 2 Seconds of 10 Minutes 8 Seconds".
_TIMELINE_NAME = re.compile(r"\b(seek|progress|position|playback|scrub|timeline)\b", re.IGNORECASE)
_TIME_WORDS = re.compile(r"\b(seconds?|minutes?|hours?)\b", re.IGNORECASE)


def _a_timeline(element: Any) -> tuple[float, float] | None:
    """A slider's place and length in seconds where it is a player's timeline; None for any other slider."""
    named = f"{_read(element, 'AXTitle') or ''} {_read(element, 'AXDescription') or ''}"
    said = str(_read(element, "AXValueDescription") or "")
    try:
        value, top = float(_read(element, "AXValue")), float(_read(element, "AXMaxValue"))
    except (TypeError, ValueError):
        value = top = 0.0
    if top > 0 and (_TIMELINE_NAME.search(named) or _TIME_WORDS.search(said) or _PLACE.search(said)):
        return value, top
    place = _PLACE.search(said)
    return (seconds_of(place.group(1)), seconds_of(place.group(2))) if place else None


def ax_window_for(window: dict[str, Any]) -> Any:
    """The accessibility element of a window seen on screen (a Quartz window record): the one with its title, or else
    the one where it is and as large as it is. A browser may give every window its own name in accessibility."""
    from ApplicationServices import AXUIElementCreateApplication

    windows = _read(AXUIElementCreateApplication(int(window["kCGWindowOwnerPID"])), "AXWindows") or []
    named = str(window.get("kCGWindowName") or "")
    for candidate in windows:
        if named and str(_read(candidate, "AXTitle") or "") == named:
            return candidate
    bounds = window.get("kCGWindowBounds") or {}
    wanted = (float(bounds.get("X") or 0), float(bounds.get("Y") or 0),
              float(bounds.get("Width") or 0), float(bounds.get("Height") or 0))
    for candidate in windows:
        frame = _frame(candidate)
        if frame and all(abs(a - b) <= 2.0 for a, b in zip((*frame[0], *frame[1]), wanted, strict=True)):
            return candidate
    return None


def read_their_player(window: dict[str, Any]) -> PlayerReading:
    """One pass over the window's accessibility tree: its seek slider's place and length, and a skip control."""
    reading = PlayerReading()
    start = ax_window_for(window)
    if start is None:
        return reading
    reading.window_found = True
    controls: list[tuple[str, str, tuple[float, float], tuple[float, float]]] = []
    elements: list[Any] = []
    places: list[tuple[float, float]] = []
    waiting, seen = [start], 0
    while waiting and seen < ELEMENTS_READ:
        element = waiting.pop(0)
        seen += 1
        role = str(_read(element, "AXRole") or "")
        if role == "AXSlider":
            place = _a_timeline(element)
            if place is not None:
                places.append(place)
        elif role in ("AXButton", "AXLink"):
            frame = _frame(element)
            words = " ".join(str(_read(element, name) or "") for name in ("AXTitle", "AXDescription")).strip()
            if frame and words:
                controls.append((role, words, *frame))
                elements.append(element)
        waiting.extend(_read(element, "AXChildren") or [])
    if places:
        # The player's place is the longest timeline on the page; a volume slider says no clock.
        reading.at, reading.length = max(places, key=lambda one: one[1])
    chosen = skip_on_screen(controls, window.get("kCGWindowBounds") or {})
    if chosen is not None:
        reading.skip, reading.skip_words = elements[chosen], controls[chosen][1][:40]
    reading.pause = next((element for element, (role, words, *_where) in zip(elements, controls, strict=True)
                          if role == "AXButton" and re.match(r"\s*pause\b", words, re.IGNORECASE)), None)
    return reading


def press(element: Any) -> bool:
    """A control pressed as a person presses it, through Accessibility."""
    from ApplicationServices import AXUIElementPerformAction

    return AXUIElementPerformAction(element, "AXPress") == 0
