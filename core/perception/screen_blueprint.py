"""What is actually on the screen, as structure rather than as pixels.

Aura could see the screen two ways, and neither answered the question a
person answers instantly.

    a screenshot + OCR      a wall of text with no idea what owns it
    a list of window titles a flat set with no geometry and no order

Both leave "which app is in front", "is my note window actually visible or
is Chrome covering it", and "where do I click to type" unanswerable — so an
OS action had to guess, act, and hope. Guessing is why the Notes step used to
land in the wrong window: nothing could tell her the window she meant was
behind something else.

This module is the third way: a blueprint. Every on-screen window, in true
front-to-back order, with its owner, its rectangle, and how much of it is
still visible after the windows above it are subtracted. It is the same
information a person gets from a glance, in the form a machine can reason
over.

Three layers, each paid for only when asked:

    frame       every window, z-order, geometry, occlusion    ~30ms, no
                permission beyond what a screenshot needs, no subprocess
    elements    the controls inside one window and where they are
                Accessibility permission, one bounded osascript
    document    the live DOM of a browser tab, which is the only honest
                answer to "what does that page say"

Occlusion is computed, not estimated. A window's visible area is its
rectangle minus the union of every rectangle in front of it, evaluated
exactly by coordinate decomposition. So "Chrome is covering Notes" is a
measurement — and when she says a window is hidden, that is why.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.ScreenBlueprint")

__all__ = [
    "ScreenBlueprint",
    "WindowFrame",
    "blueprint_is_available",
    "blueprint_is_enabled",
    "capture_blueprint",
    "read_browser_document",
    "read_window_elements",
]

#: Windows above this layer are system furniture — the menu bar, Control
#: Centre, the cursor's status dots. They are on the screen but they are not
#: what anyone means by "what's open".
_NORMAL_WINDOW_LAYER = 0

#: A window smaller than this in either dimension is a shadow, a tooltip, or
#: a status item, not something a person is working in.
_MIN_REAL_WINDOW_EDGE = 40

#: An accessibility read is a bounded question, never a stall.
_ELEMENT_READ_TIMEOUT_S = 6.0
_DOCUMENT_READ_TIMEOUT_S = 8.0

#: A blueprint is cheap but not free, and a single turn may ask several
#: times. One capture serves a burst; anything older is re-measured.
_CACHE_TTL_S = 1.5

_CACHE_LOCK = checked_lock("screen_blueprint")
_CACHED: tuple[float, ScreenBlueprint] | None = None

#: Set to "0" to make every capture report itself unavailable.
#:
#: This exists because the blueprint answers a question that used to have
#: exactly one seam. Callers such as ComputerUseSkill._frontmost_app_name ask
#: the window server first and fall back to AppleScript — so a caller that
#: controls only the AppleScript transport (every test in the desktop suite)
#: stopped controlling the answer, and quietly began reading the real screen
#: of whatever machine the suite happened to run on. A test that reads the
#: live desktop is not a test.
#:
#: It is a production switch as well as a test one: a machine whose window
#: server misreports z-order should be able to fall back to the slow path
#: without an edit.
_BLUEPRINT_ENV = "AURA_SCREEN_BLUEPRINT"


def blueprint_is_enabled() -> bool:
    """False when the window-server path has been switched off."""
    return str(os.environ.get(_BLUEPRINT_ENV, "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@dataclass(frozen=True)
class WindowFrame:
    """One window on the screen, with everything needed to reason about it."""

    app: str
    title: str
    pid: int
    window_id: int
    x: int
    y: int
    width: int
    height: int
    layer: int
    alpha: float
    #: Front-to-back position. 0 is the window nearest the viewer.
    z_index: int
    #: Fraction of this window's area not covered by any window in front of
    #: it. 1.0 is fully visible, 0.0 is entirely hidden.
    visible_fraction: float = 1.0
    #: The apps whose windows are covering this one, nearest first.
    covered_by: tuple[str, ...] = ()

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def is_visible(self) -> bool:
        """Would a person say they can see this window?"""
        return self.visible_fraction > 0.02

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def describe(self) -> str:
        where = f"{self.width}x{self.height} at ({self.x}, {self.y})"
        if self.visible_fraction >= 0.999:
            seen = "fully visible"
        elif not self.is_visible:
            seen = "completely hidden"
            if self.covered_by:
                seen += f" behind {self.covered_by[0]}"
        else:
            seen = f"{round(self.visible_fraction * 100)}% visible"
            if self.covered_by:
                seen += f", partly behind {self.covered_by[0]}"
        title = f' "{self.title}"' if self.title else ""
        return f"{self.app}{title} — {where}, {seen}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "app": self.app,
            "title": self.title,
            "pid": self.pid,
            "window_id": self.window_id,
            "bounds": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "z_index": self.z_index,
            "visible_fraction": round(self.visible_fraction, 4),
            "covered_by": list(self.covered_by),
            "is_visible": self.is_visible,
        }


@dataclass(frozen=True)
class ScreenBlueprint:
    """The whole screen, as structure."""

    windows: tuple[WindowFrame, ...] = ()
    frontmost_app: str = ""
    captured_at: float = field(default_factory=lambda: time.time())
    #: True when the blueprint could not be taken at all. An empty screen and
    #: an unreadable one are different facts and must never be confused.
    unavailable: bool = False
    unavailable_reason: str = ""

    @property
    def apps(self) -> tuple[str, ...]:
        """Every app with a window, front-to-back, each named once."""
        seen: list[str] = []
        for window in self.windows:
            if window.app and window.app not in seen:
                seen.append(window.app)
        return tuple(seen)

    def windows_for(self, app: str) -> tuple[WindowFrame, ...]:
        wanted = str(app or "").strip().lower()
        if not wanted:
            return ()
        return tuple(
            window
            for window in self.windows
            if wanted in window.app.lower() or window.app.lower() in wanted
        )

    def is_app_frontmost(self, app: str) -> bool:
        wanted = str(app or "").strip().lower()
        if not wanted or not self.windows:
            return False
        front = self.windows[0].app.lower()
        return wanted in front or front in wanted

    def app_is_visible(self, app: str) -> bool:
        """Can a person see any part of this app right now?"""
        return any(window.is_visible for window in self.windows_for(app))

    def describe(self, limit: int = 8) -> str:
        """The screen in plain words, front to back."""
        if self.unavailable:
            return f"I can't read the screen layout right now ({self.unavailable_reason})."
        if not self.windows:
            return "There are no application windows open on the screen."
        lines = [f"{len(self.windows)} window(s) open, front to back:"]
        for index, window in enumerate(self.windows[:limit], start=1):
            lines.append(f"  {index}. {window.describe()}")
        if len(self.windows) > limit:
            lines.append(f"  ... and {len(self.windows) - limit} more")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontmost_app": self.frontmost_app,
            "apps": list(self.apps),
            "window_count": len(self.windows),
            "windows": [window.to_dict() for window in self.windows],
            "captured_at": self.captured_at,
            "unavailable": self.unavailable,
            "unavailable_reason": self.unavailable_reason,
        }


def _rect_union_area_within(
    target: tuple[int, int, int, int],
    covers: Sequence[tuple[int, int, int, int]],
) -> int:
    """Exact area of `target` covered by the union of `covers`.

    Coordinate decomposition rather than a sampling estimate: the x-edges of
    every rectangle cut the target into vertical bands, and within a band the
    covering intervals are constant, so a 1-D merge gives the covered height
    exactly. Overlapping covers are counted once, which is the whole reason
    not to just sum areas.
    """
    tx0, ty0, tx1, ty1 = target
    if tx1 <= tx0 or ty1 <= ty0:
        return 0

    clipped: list[tuple[int, int, int, int]] = []
    for cx0, cy0, cx1, cy1 in covers:
        x0, y0 = max(cx0, tx0), max(cy0, ty0)
        x1, y1 = min(cx1, tx1), min(cy1, ty1)
        if x1 > x0 and y1 > y0:
            clipped.append((x0, y0, x1, y1))
    if not clipped:
        return 0

    edges = sorted({tx0, tx1} | {x for rect in clipped for x in (rect[0], rect[2])})
    covered = 0
    for left, right in zip(edges, edges[1:]):
        band_width = right - left
        if band_width <= 0:
            continue
        spans = sorted(
            (y0, y1) for x0, y0, x1, y1 in clipped if x0 <= left and x1 >= right
        )
        if not spans:
            continue
        merged_height = 0
        run_start, run_end = spans[0]
        for y0, y1 in spans[1:]:
            if y0 > run_end:
                merged_height += run_end - run_start
                run_start, run_end = y0, y1
            else:
                run_end = max(run_end, y1)
        merged_height += run_end - run_start
        covered += band_width * merged_height
    return covered


def _is_real_window(entry: Any) -> bool:
    try:
        if int(entry.get("kCGWindowLayer", 0) or 0) != _NORMAL_WINDOW_LAYER:
            return False
        if float(entry.get("kCGWindowAlpha", 1.0) or 0.0) <= 0.01:
            return False
        bounds = entry.get("kCGWindowBounds") or {}
        if int(bounds.get("Width", 0) or 0) < _MIN_REAL_WINDOW_EDGE:
            return False
        if int(bounds.get("Height", 0) or 0) < _MIN_REAL_WINDOW_EDGE:
            return False
    except (AttributeError, TypeError, ValueError):
        # not a failure: an entry whose bounds are unreadable is not one
        # this can call a real window, which is the question.
        return False
    return True


def blueprint_is_available() -> bool:
    """Can this machine produce a blueprint at all?"""
    if not blueprint_is_enabled():
        return False
    try:
        import Quartz  # noqa: F401
    except (ImportError, OSError):
        # not a failure: no Quartz means this machine cannot produce a
        # blueprint, which is exactly what the docstring asks.
        return False
    return True


def _raw_window_list() -> list[Any]:
    import Quartz

    options = Quartz.kCGWindowListOptionOnScreenOnly
    try:
        options |= Quartz.kCGWindowListExcludeDesktopElements
    except AttributeError:  # pragma: no cover - older bindings
        # not a failure: older bindings have no such flag, and the list
        # then includes desktop elements the caller filters anyway.
        pass
    return list(Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or [])


def capture_blueprint(*, fresh: bool = False) -> ScreenBlueprint:
    """Measure the screen's structure. Cheap, bounded, cached for a moment.

    Never raises: a blueprint that cannot be taken reports itself unavailable
    with the reason, because "I couldn't look" and "nothing is open" must
    stay distinguishable to everything reasoning above this.
    """
    global _CACHED
    if not blueprint_is_enabled():
        return ScreenBlueprint(
            unavailable=True,
            unavailable_reason=f"{_BLUEPRINT_ENV} is switched off",
        )
    if not fresh:
        with _CACHE_LOCK:
            cached = _CACHED
        if cached is not None and (time.time() - cached[0]) < _CACHE_TTL_S:
            return cached[1]

    try:
        entries = _raw_window_list()
    except Exception as exc:  # noqa: BLE001 - perception may never take a turn down
        logger.debug("Screen blueprint capture failed: %s", exc)
        return ScreenBlueprint(
            unavailable=True,
            unavailable_reason=f"the window list could not be read ({type(exc).__name__})",
        )

    frames: list[WindowFrame] = []
    for z_index, entry in enumerate(entries):
        if not _is_real_window(entry):
            continue
        bounds = entry.get("kCGWindowBounds") or {}
        try:
            frames.append(
                WindowFrame(
                    app=str(entry.get("kCGWindowOwnerName") or "").strip(),
                    title=str(entry.get("kCGWindowName") or "").strip(),
                    pid=int(entry.get("kCGWindowOwnerPID", 0) or 0),
                    window_id=int(entry.get("kCGWindowNumber", 0) or 0),
                    x=int(bounds.get("X", 0) or 0),
                    y=int(bounds.get("Y", 0) or 0),
                    width=int(bounds.get("Width", 0) or 0),
                    height=int(bounds.get("Height", 0) or 0),
                    layer=int(entry.get("kCGWindowLayer", 0) or 0),
                    alpha=float(entry.get("kCGWindowAlpha", 1.0) or 0.0),
                    z_index=len(frames),
                )
            )
        except (TypeError, ValueError):
            continue
        del z_index

    resolved: list[WindowFrame] = []
    for index, frame in enumerate(frames):
        above = frames[:index]
        covered = _rect_union_area_within(frame.rect, [w.rect for w in above])
        area = frame.area
        visible = 1.0 if area <= 0 else max(0.0, 1.0 - (covered / area))
        coverers: list[str] = []
        if covered > 0:
            for other in above:
                ox0, oy0, ox1, oy1 = other.rect
                fx0, fy0, fx1, fy1 = frame.rect
                if min(ox1, fx1) > max(ox0, fx0) and min(oy1, fy1) > max(oy0, fy0):
                    if other.app and other.app not in coverers:
                        coverers.append(other.app)
        resolved.append(
            WindowFrame(
                app=frame.app,
                title=frame.title,
                pid=frame.pid,
                window_id=frame.window_id,
                x=frame.x,
                y=frame.y,
                width=frame.width,
                height=frame.height,
                layer=frame.layer,
                alpha=frame.alpha,
                z_index=index,
                visible_fraction=visible,
                covered_by=tuple(coverers[:4]),
            )
        )

    blueprint = ScreenBlueprint(
        windows=tuple(resolved),
        frontmost_app=resolved[0].app if resolved else "",
    )
    with _CACHE_LOCK:
        _CACHED = (time.time(), blueprint)
    return blueprint


_ELEMENT_SCRIPT = """
on esc(s)
  set out to ""
  repeat with c in characters of s
    if c is "\\"" then
      set out to out & "\\\\\\""
    else if c is "\\\\" then
      set out to out & "\\\\\\\\"
    else if (id of c) < 32 then
      set out to out & " "
    else
      set out to out & c
    end if
  end repeat
  return out
end esc

tell application "System Events"
  if not (exists application process "%(app)s") then return "{\\"error\\":\\"app not running\\"}"
  tell application process "%(app)s"
    if (count of windows) is 0 then return "{\\"error\\":\\"no windows\\"}"
    set w to window 1
    set items_json to {}
    set kinds to {"button", "text field", "text area", "checkbox", "pop up button", "menu button", "static text", "radio button"}
    repeat with k in kinds
      try
        set els to (every UI element of w whose role description is (k as text))
        repeat with e in els
          try
            set nm to ""
            try
              set nm to name of e as text
            end try
            if nm is "" then
              try
                set nm to description of e as text
              end try
            end if
            set p to position of e
            set sz to size of e
            set end of items_json to "{\\"kind\\":\\"" & (k as text) & "\\",\\"name\\":\\"" & my esc(nm) & "\\",\\"x\\":" & (item 1 of p) & ",\\"y\\":" & (item 2 of p) & ",\\"w\\":" & (item 1 of sz) & ",\\"h\\":" & (item 2 of sz) & "}"
          end try
        end repeat
      end try
    end repeat
    set AppleScript's text item delimiters to ","
    set body to items_json as text
    set AppleScript's text item delimiters to ""
    set wt to ""
    try
      set wt to name of w as text
    end try
    return "{\\"window\\":\\"" & my esc(wt) & "\\",\\"elements\\":[" & body & "]}"
  end tell
end tell
"""


def read_window_elements(app: str, *, timeout: float = _ELEMENT_READ_TIMEOUT_S) -> dict[str, Any]:
    """The controls inside an app's front window, and where each one is.

    This is the answer to "where do I type" that does not involve guessing at
    a screenshot: a text area reports its own position and size, so a click
    lands in it because it was aimed there.
    """
    name = str(app or "").strip()
    if not name:
        return {"ok": False, "error": "no app named"}
    osascript = shutil.which("osascript") or "/usr/bin/osascript"
    script = _ELEMENT_SCRIPT % {"app": name.replace('"', '\\"')}
    try:
        result = get_subprocess_gateway().run(
            [osascript, "-e", script],
            capture_output=True,
            read_only=True,
            text=True,
            timeout=timeout,
            source="perception.screen_blueprint.accessibility_read",
            accelerator_capability="none",
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"the accessibility read timed out after {exc.timeout:g}s",
        }
    except OSError as exc:
        return {"ok": False, "error": f"accessibility read failed ({exc})"}

    raw = (result.stdout or "").strip()
    if not raw:
        stderr = (result.stderr or "").strip()
        if "not allowed" in stderr.lower() or "1002" in stderr:
            return {"ok": False, "error": "macOS has not granted Accessibility access"}
        return {"ok": False, "error": stderr[:200] or "no accessibility data"}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return {
            "ok": False,
            "error": f"unreadable accessibility data ({exc}): {raw[:120]}",
        }
    if isinstance(payload, dict) and payload.get("error"):
        return {"ok": False, "error": str(payload["error"])}
    elements = payload.get("elements") if isinstance(payload, dict) else None
    return {
        "ok": True,
        "app": name,
        "window": (payload or {}).get("window", ""),
        "elements": elements if isinstance(elements, list) else [],
    }


#: Browsers that expose their page to AppleScript, and how to ask.
_BROWSER_DOCUMENT_SCRIPTS: dict[str, str] = {
    "Google Chrome": (
        'tell application "Google Chrome" to return '
        "(execute active tab of front window javascript "
        '"JSON.stringify({url:location.href,title:document.title,'
        'text:document.body?document.body.innerText.slice(0,%(limit)d):\\"\\",'
        'html:document.documentElement.outerHTML.slice(0,%(html)d)})")'
    ),
    "Safari": (
        'tell application "Safari" to return '
        "(do JavaScript "
        '"JSON.stringify({url:location.href,title:document.title,'
        'text:document.body?document.body.innerText.slice(0,%(limit)d):\\"\\",'
        'html:document.documentElement.outerHTML.slice(0,%(html)d)})" '
        "in front document)"
    ),
}


def read_browser_document(
    browser: str = "Google Chrome",
    *,
    text_limit: int = 12_000,
    html_limit: int = 0,
    timeout: float = _DOCUMENT_READ_TIMEOUT_S,
) -> dict[str, Any]:
    """What the front browser tab actually contains — its own words, not OCR.

    `html_limit` above zero returns the live DOM as well. It is off by
    default because a page's text is what a reader needs and its markup is
    fifty times the size; the structure is there when structure is the
    question.
    """
    name = str(browser or "").strip() or "Google Chrome"
    template = _BROWSER_DOCUMENT_SCRIPTS.get(name)
    if template is None:
        return {"ok": False, "error": f"{name} does not expose its page for reading"}
    osascript = shutil.which("osascript") or "/usr/bin/osascript"
    script = template % {"limit": max(0, int(text_limit)), "html": max(0, int(html_limit))}
    try:
        result = get_subprocess_gateway().run(
            [osascript, "-e", script],
            capture_output=True,
            read_only=True,
            text=True,
            timeout=timeout,
            source="perception.screen_blueprint.browser_document_read",
            accelerator_capability="none",
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": f"the page read timed out after {exc.timeout:g}s"}
    except OSError as exc:
        return {"ok": False, "error": f"page read failed ({exc})"}

    raw = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip().lower()
    if not raw:
        if "allow javascript from apple events" in stderr or "not allowed" in stderr:
            return {
                "ok": False,
                "error": (
                    "the browser has not enabled 'Allow JavaScript from Apple Events'"
                ),
            }
        return {"ok": False, "error": stderr[:200] or "the browser returned nothing"}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"unreadable page data ({exc}): {raw[:120]}"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "the page data was not an object"}
    payload["ok"] = True
    payload["browser"] = name
    return payload


def blueprint_summary_lines(blueprint: ScreenBlueprint | None = None) -> Iterable[str]:
    """The blueprint as short lines, for a prompt or a log."""
    report = blueprint if blueprint is not None else capture_blueprint()
    yield from report.describe().splitlines()
