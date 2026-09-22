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

It is NOT a pointer she uses on her own initiative. Highlighting is
ASKED FOR — "show me", "which one", "point at it" — and an unrequested
rectangle appearing on someone's screen is not help, it is a jump scare.
The governance is the same as everything else in companion mode: the
suppression that stops her speaking stops her drawing.

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
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

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


_LOCATE_SCRIPT = """
on run argv
    set needle to item 1 of argv
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
        tell process frontApp
            set matches to (every UI element of front window whose value of ¬
                attribute "AXTitle" contains needle)
            if (count of matches) is 0 then
                set matches to (every UI element of front window whose value of ¬
                    attribute "AXValue" contains needle)
            end if
            if (count of matches) is 0 then return "none"
            set target to item 1 of matches
            set pos to value of attribute "AXPosition" of target
            set siz to value of attribute "AXSize" of target
            return ((item 1 of pos) as text) & "," & ((item 2 of pos) as text) & ¬
                "," & ((item 1 of siz) as text) & "," & ((item 2 of siz) as text)
        end tell
    end tell
end run
"""

_TIMEOUT_S = 2.5


def locate_on_screen(needle: str) -> Rect | None:
    """Where the accessibility tree says this text is, or None.

    None means "could not locate", which the caller must render as words
    rather than as a guessed rectangle. A rectangle over the wrong thing is
    a confident wrong answer wearing the appearance of a measurement.
    """
    text = str(needle or "").strip()
    if len(text) < 2:
        return None
    try:
        completed = get_subprocess_gateway().run(
            ["osascript", "-", text],
            input=_LOCATE_SCRIPT,
            timeout=_TIMEOUT_S,
            read_only=True,
            capture_output=True,
            source="screen_highlight.locate",
            accelerator_capability="none",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        record_degradation(
            "screen_highlight", exc, severity="debug",
            action="could not locate the element; she will describe it in words",
        )
        return None
    raw = str(completed.stdout or "").strip()
    if completed.returncode != 0 or not raw or raw == "none":
        return None
    parts = raw.split(",")
    if len(parts) != 4:
        return None
    try:
        x, y, width, height = (float(part.strip()) for part in parts)
    # not a failure: a value that is not a number is not one this can read.
    except ValueError:
        return None
    rect = Rect(x=x, y=y, width=width, height=height)
    return rect if rect.is_usable else None


def _screen_area() -> float:
    try:
        completed = get_subprocess_gateway().run(
            [
                "osascript",
                "-e",
                'tell application "Finder" to get bounds of window of desktop',
            ],
            timeout=_TIMEOUT_S,
            read_only=True,
            capture_output=True,
            source="screen_highlight.screen_bounds",
            accelerator_capability="none",
        )
        parts = [float(p.strip()) for p in str(completed.stdout or "").split(",")]
        if len(parts) == 4:
            return max(1.0, (parts[2] - parts[0]) * (parts[3] - parts[1]))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.debug(
            "%s unavailable (%s: %s); the highlighted region reads as zero area",
            "it",
            type(exc).__name__,
            exc,
        )
    return 0.0


async def highlight(
    needle: str, *, duration_s: float = DEFAULT_DURATION_S, requested: bool = False
) -> HighlightResult:
    """Draw a transient rectangle around ``needle``, or refuse and say why.

    ``requested`` must be True. Highlighting is something she is ASKED to do;
    an unrequested rectangle appearing on someone's screen is a jump scare,
    not help. The parameter exists so a caller has to state that the person
    asked, rather than the check living in one caller and being forgotten by
    the next.
    """
    if not requested:
        return HighlightResult(
            shown=False, reason="highlighting is only done when asked for"
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

    rect = locate_on_screen(needle)
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
    "rate_limited",
]
