"""Point at it on the screen, when pointing is the answer.

"Which one is the failing test?" has a better answer than a paragraph
describing where to look. Sometimes the answer is a rectangle around the
thing.

WHAT THIS IS AND IS NOT
───────────────────────
It is a transient, non-interactive overlay: a rectangle drawn over the
screen for a few seconds, click-through, that disappears on its own. It
never covers anything permanently and never intercepts input, because an
overlay that can swallow a click is an overlay that can cost someone work.

It is asked for ("show me", "which one", "point at it"), or, while she
keeps the person company in the bubble, it is hers to use: Bryan, 15
November 2026, "she's allowed to physically act and react to things that
are going on. highlight, show me something ... full companion". Then she
points by a call she makes in her turn at the desk
(core/skills/at_the_desk.py). The governance is the same as everything
else in companion mode: the suppression that stops her speaking stops her
drawing.

FINDING THE THING
─────────────────
The overlay needs coordinates, and the only trustworthy source is the
accessibility tree, which reports each element's real position and size. A
guess from character offsets in an OCR dump is a rectangle over the wrong
thing, and a rectangle over the wrong thing is worse than no rectangle: it
is a confident wrong answer that looks authoritative.

So an element that cannot be located is REFUSED, and she says where to look
in words instead. That refusal is the honest outcome and it is not a
degraded one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

HIGHLIGHT_SCHEMA = "aura.perception.screen_highlight.v1"

#: How long a highlight stays up. Long enough to look at, short enough that
#: a forgotten one cannot become furniture on someone's desktop.
DEFAULT_DURATION_S = 3.0
MAX_DURATION_S = 10.0

#: A rectangle bigger than this fraction of the screen is not pointing at
#: anything — it is covering the screen, and "here, everywhere" is not an
#: answer. Refused rather than drawn.
MAX_SCREEN_FRACTION = 0.55


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def is_usable(self) -> bool:
        # A zero or negative rectangle is not a location, and a one-pixel one
        # points at nothing a person can see.
        return self.width >= 4.0 and self.height >= 4.0


@dataclass(frozen=True)
class HighlightResult:
    shown: bool
    reason: str
    rect: Rect | None = None
    matched_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HIGHLIGHT_SCHEMA,
            "shown": self.shown,
            "reason": self.reason,
            "matched": self.matched_text[:120],
            "rect": (
                None
                if self.rect is None
                else {
                    "x": round(self.rect.x, 1),
                    "y": round(self.rect.y, 1),
                    "width": round(self.rect.width, 1),
                    "height": round(self.rect.height, 1),
                }
            ),
        }


_TIMEOUT_S = 2.5


def locate_on_screen(needle: str) -> Rect | None:
    """Where the accessibility tree says this text is, or None.

    None means "could not locate", which the caller must render as words
    rather than as a guessed rectangle. A rectangle over the wrong thing is
    a confident wrong answer wearing the appearance of a measurement.

    The front window's whole tree is searched, breadth first, within the time
    allowed. It asked only the window's own children ("every UI element of
    front window"), which in a real window are groups and toolbars: LIVE 15
    November 2026, with Finder in front, "Downloads", "Applications" and
    "Search" were each not found, so every requested highlight was refused.
    """
    text = str(needle or "").strip()
    if len(text) < 2:
        return None
    try:
        return _found_in_the_front_window(text, within_s=_TIMEOUT_S)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        record_degradation(
            "screen_highlight", exc, severity="debug",
            action="could not locate the element; she will describe it in words",
        )
        return None


#: What an element is called, in the attributes a person sees or a screen reader says. Not its help text, which
#: talks about other things: Finder's full-screen button's is "this button also has an action to zoom the window", and
#: its history group's "See folders you viewed previously", so "the save button" found a button with no Save in it.
_NAMING_ATTRIBUTES = ("AXTitle", "AXDescription", "AXValue", "AXPlaceholderValue")


#: Roles whose children are not on the screen until it is opened: a menu's items. The menu bar and its titles are.
_NOT_ON_SCREEN_UNTIL_OPENED = frozenset({"AXMenu"})


def _found_in_the_front_window(needle: str, *, within_s: float) -> Rect | None:
    """The element of the front window that ``needle`` names, by the window's own words; None when it names none.

    A word of the needle that names some element in the window must name this one ("Downloads"); a word the window
    uses only for a kind of control ("button", "text field") is preferred and not required; a word the window does not
    use at all ("folder" in "the Downloads folder", "in this window") says nothing about which element is meant.
    Among the elements left, the one whose name has the fewest other words, then the smallest. A needle that names
    nothing in the window finds nothing: a rectangle around some other button is worse than none.
    """
    import re
    from collections import deque

    from AppKit import NSWorkspace
    from ApplicationServices import (
        AXUIElementCopyAttributeValue,
        AXUIElementCreateApplication,
        AXValueGetValue,
        kAXValueCGPointType,
        kAXValueCGSizeType,
    )

    def attribute(element: Any, name: str) -> Any:
        error, value = AXUIElementCopyAttributeValue(element, name, None)
        return None if error else value

    def words_of(*values: Any) -> frozenset[str]:
        return frozenset(re.findall(r"\w+", " ".join(str(v) for v in values if isinstance(v, str)).lower()))

    def rect_of(element: Any) -> Rect | None:
        position, size = attribute(element, "AXPosition"), attribute(element, "AXSize")
        if position is None or size is None:
            return None
        has_point, point = AXValueGetValue(position, kAXValueCGPointType, None)
        has_size, extent = AXValueGetValue(size, kAXValueCGSizeType, None)
        if not (has_point and has_size):
            return None
        found = Rect(x=float(point.x), y=float(point.y), width=float(extent.width), height=float(extent.height))
        return found if found.is_usable else None

    front = NSWorkspace.sharedWorkspace().frontmostApplication()
    if front is None:
        return None
    application = AXUIElementCreateApplication(int(front.processIdentifier()))
    window = attribute(application, "AXFocusedWindow") or attribute(application, "AXMainWindow")
    if window is None:
        return None
    ends = time.monotonic() + max(0.1, float(within_s))
    seen: list[tuple[Any, frozenset[str], frozenset[str]]] = []
    walked: set[Any] = set()
    queue = deque([window])
    while queue and time.monotonic() < ends:
        element = queue.popleft()
        # Each element once: a tree that lists an element under itself is walked in circles otherwise. LIVE 16
        # November 2026, the Claude app's window gave the walk 36,969 elements in 3 s, 33,133 of them menu items and
        # its application element 140 times, and the conversation on screen was never reached.
        key: Any = element  # an accessibility element is equal to, and hashes as, every other copy of itself
        try:
            hash(element)
        except TypeError:
            key = id(element)
        if key in walked:
            continue
        walked.add(key)
        named = words_of(*(attribute(element, name) for name in _NAMING_ATTRIBUTES))
        if named:
            seen.append((element, named, words_of(attribute(element, "AXRoleDescription"))))
        if attribute(element, "AXRole") in _NOT_ON_SCREEN_UNTIL_OPENED:
            continue
        children = attribute(element, "AXChildren")
        if children:
            queue.extend(children)
    asked = frozenset(re.findall(r"\w+", needle.lower()))
    names = asked & frozenset().union(*(named for _element, named, _kind in seen))
    if not names:
        return None
    kinds = (asked - names) & frozenset().union(*(kind for _element, _named, kind in seen))
    best, best_rank = None, None
    for element, named, kind in seen:
        if not names <= named:
            continue
        found = rect_of(element)
        if found is None:
            continue
        rank = (not kinds <= kind, len(named - asked), found.area)
        if best_rank is None or rank < best_rank:
            best, best_rank = found, rank
    return best


def _screen_area() -> float:
    """The area of her displays, in the points the accessibility tree measures in; 0.0 when it cannot be read.

    It asked Finder for its desktop window's bounds over AppleScript, an Apple Event from her process: LIVE 16
    November 2026, 00:40, the back button was located in 0.36 s and the highlight then waited 120 s on that question,
    so the rectangle it drew was declined as stale. The display list is read directly.
    """
    try:
        import Quartz

        error, _none, count = Quartz.CGGetActiveDisplayList(0, None, None)
        if error or not count:
            return 0.0
        error, displays, count = Quartz.CGGetActiveDisplayList(count, None, None)
        if error:
            return 0.0
        bounds = [Quartz.CGDisplayBounds(display) for display in displays[:count]]
        return float(sum(rect.size.width * rect.size.height for rect in bounds))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("the displays could not be read (%s: %s); the highlighted region reads as zero area",
                     type(exc).__name__, exc)
        return 0.0


async def highlight(
    needle: str, *, duration_s: float = DEFAULT_DURATION_S, requested: bool = False, invited: bool = False
) -> HighlightResult:
    """Draw a transient rectangle around ``needle``, or refuse and say why.

    ``requested`` must be True. Highlighting is something she is ASKED to do;
    an unrequested rectangle appearing on someone's screen is a jump scare,
    not help. The parameter exists so a caller has to state that the person
    asked, rather than the check living in one caller and being forgotten by
    the next.
    """
    if not (requested or invited):
        # ``invited``: she is keeping the person company, where pointing is hers (Bryan, 15 November 2026).
        return HighlightResult(
            shown=False, reason="highlighting is only done when asked for, or while keeping company"
        )
    if _suppressed():
        # The same suppression that stops her speaking stops her drawing.
        return HighlightResult(shown=False, reason="proactivity is suppressed")
    if rate_limited():
        # Enforced HERE rather than left to callers. The limiter existed and
        # was exported and tested, and nothing on the drawing path called it,
        # so "repeated highlights are rate limited" was true of the helper and
        # false of the feature. A rectangle that reappears faster than the eye
        # settles is not pointing, it is flickering.
        return HighlightResult(
            shown=False, reason="a highlight was drawn a moment ago"
        )

    from core.perception.ambient_presence import is_private_context
    from core.senses.screen_context import frontmost_window_hint

    app, title = frontmost_window_hint()
    if is_private_context(app, title):
        # Drawing on a private window means having located something in it.
        return HighlightResult(
            shown=False, reason="a private window is in the foreground"
        )

    import asyncio

    looked = time.monotonic()
    rect = await asyncio.to_thread(locate_on_screen, needle)  # a walk of the window's tree, off her event loop
    logger.info("🖍️ %r located %s in %.2fs", needle, "at " + str(rect) if rect else "nowhere", time.monotonic() - looked)
    if rect is None:
        return HighlightResult(
            shown=False,
            reason="could not locate it in the accessibility tree; describing it instead",
        )

    area = _screen_area()
    if area and rect.area / area > MAX_SCREEN_FRACTION:
        # "Here, everywhere" is not an answer.
        return HighlightResult(
            shown=False,
            reason="the match covers most of the screen, which points at nothing",
            rect=rect,
        )

    seconds = min(MAX_DURATION_S, max(0.5, float(duration_s)))
    try:
        from core.governance_context import local_internal_governed_scope

        with local_internal_governed_scope(
            "screen_highlight.draw", domain="environment_action"
        ):
            drawn = await _draw(rect, seconds)
    except (ImportError, RuntimeError, OSError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_highlight", exc, severity="warning",
            action="highlight not drawn; she will describe the location in words",
        )
        return HighlightResult(shown=False, reason=f"overlay failed: {type(exc).__name__}")

    return HighlightResult(
        shown=drawn,
        reason="drawn" if drawn else "overlay declined",
        rect=rect,
        matched_text=str(needle),
    )


async def with_pointing(objective: str, payload: dict[str, Any]) -> dict[str, Any]:
    """``payload`` with the pointing its request asked for, drawn or refused; unchanged when it asked for none.

    "Where is the submit button?" is answered better by a rectangle than by a paragraph. The refusal is carried as
    deliberately as the success: a highlight that did not draw must reach the answer as a fact, or she says "I've
    highlighted it" over a screen with no highlight on it. Moved here from core/skills/desktop_task.py.
    """
    from core.runtime.desktop_objective_intent import asks_to_be_shown_where

    needle = asks_to_be_shown_where(objective)
    if not needle:
        return payload
    try:
        result = await highlight(needle, requested=True)
    except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
        # Pointing never costs her the answer, and a highlight that raised is said as one not drawn.
        payload["pointing_refused_because"] = f"the highlight could not be drawn: {type(exc).__name__}: {exc}"
        return payload
    payload["highlight"] = result.to_dict()
    payload["pointed_at"] = result.matched_text if result.shown else ""
    payload["pointing_refused_because"] = "" if result.shown else result.reason
    return payload


async def point_as_asked(objective: str) -> dict[str, Any] | None:
    """A request to be shown where something is, done as a desktop step: the rectangle drawn, or why it was not; None
    when the request asks for no pointing.

    Pointing was attached only to a screen question answered from her ambient reading
    (with_pointing, from core/skills/desktop_task.py), so "Can you highlight the back button?" drew nothing: it is not a
    question about what the screen shows, and it never reached a step that points (15 November 2026).
    """
    from core.runtime.desktop_objective_intent import asks_to_be_shown_where

    needle = asks_to_be_shown_where(objective)
    if not needle:
        return None
    began = time.monotonic()
    result = await highlight(needle, requested=True)
    logger.info("🖍️ asked to point at %r: %s in %.2fs", needle, result.reason, time.monotonic() - began)
    where = result.to_dict().get("rect") or {}
    evidence = (f"drew a rectangle around {needle!r} at x {where.get('x')}, y {where.get('y')}, "
                f"{where.get('width')} by {where.get('height')}" if result.shown else result.reason)
    receipt = {"index": 1, "action": "highlight_on_screen", "reason": "point at what was asked about",
               "expect": f"a rectangle around {needle!r}", "ok": result.shown, "effect_verified": result.shown,
               "effect_evidence": evidence, "result": result.to_dict()}
    payload = {"ok": result.shown, "status": "pointed_on_screen" if result.shown else "pointing_refused",
               "objective": objective, "steps_requested": 1, "steps_completed": int(result.shown),
               "receipts": [receipt], "highlight": result.to_dict(), "pointed_at": needle if result.shown else "",
               "pointing_refused_because": "" if result.shown else result.reason}
    if not result.shown:
        payload["error"] = result.reason
    return payload


async def _draw(rect: Rect, seconds: float) -> bool:
    """Ask the host launcher to draw the overlay.

    The rectangle is drawn by the launcher, not here: it owns the AppKit
    surface, and a Python process cannot put a click-through window over
    another app's screen without one. When the launcher is not present —
    running headless, or from a terminal — there is nothing to draw on and
    the refusal is honest.
    """
    from core.runtime.service_registry import get_runtime_service

    overlay = get_runtime_service("desktop_overlay", default=None)
    if overlay is None or not hasattr(overlay, "show_rect"):
        return False
    result = overlay.show_rect(
        x=rect.x, y=rect.y, width=rect.width, height=rect.height, seconds=seconds
    )
    import inspect

    if inspect.isawaitable(result):
        result = await result
    return bool(result)


def _suppressed() -> bool:
    """Is she barred from drawing on someone's screen right now? Fails CLOSED.

    Recorded, not swallowed: this reached for the gate on a module that does
    not define it, so it answered "suppressed" forever and every highlight
    refused before it ever looked for the thing it was asked to point at.
    """
    try:
        from core.brain.initiative_engine import proactivity_suppressed_now

        return bool(proactivity_suppressed_now())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_highlight",
            exc,
            severity="warning",
            action=(
                "highlight refused because the proactivity gate could not be "
                "reached — a wiring fault, not a quiet window; she will "
                "describe the location in words instead"
            ),
        )
        return True


_LAST_HIGHLIGHT_AT = 0.0


def rate_limited(min_gap_s: float = 1.5) -> bool:
    """Is a highlight arriving too soon after the last one?

    A rectangle flashing repeatedly is not pointing, it is flickering.
    """
    global _LAST_HIGHLIGHT_AT
    now = time.monotonic()
    if now - _LAST_HIGHLIGHT_AT < max(0.0, min_gap_s):
        return True
    _LAST_HIGHLIGHT_AT = now
    return False


__all__ = [
    "DEFAULT_DURATION_S",
    "HIGHLIGHT_SCHEMA",
    "MAX_DURATION_S",
    "MAX_SCREEN_FRACTION",
    "HighlightResult",
    "Rect",
    "highlight",
    "locate_on_screen",
    "point_as_asked",
    "with_pointing",
    "rate_limited",
]
