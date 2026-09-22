"""What window is in front, cheaply, without reading it.

This exists so the privacy rule can run BEFORE a capture. Every path that
reads the screen needs to know whether the foreground is private, and it
needs to know without looking at the screen — a check that has to capture in
order to decide whether capturing was allowed has already done the thing it
was deciding about.

macOS gives the frontmost application name and its window title through the
Accessibility API without any screen content, so the question "may I look?"
is answerable for the price of one attribute read.

Deliberately tiny and dependency-light: it is called on the hot path of every
screen read, including the ambient loop's tick, and a heavyweight import here
would make the cheap gate expensive enough to skip.
"""

from __future__ import annotations

import subprocess
import sys

from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

#: Bounded hard. This runs on every screen read, and a hung osascript would
#: stall the caller — including the response lane answering "what's on my
#: screen".
_TIMEOUT_S = 1.5

_SCRIPT = """
tell application "System Events"
    set frontApp to name of first application process whose frontmost is true
    set windowTitle to ""
    try
        tell process frontApp
            set windowTitle to name of front window
        end tell
    end try
    return frontApp & "|" & windowTitle
end tell
"""


def _native_frontmost_window_hint() -> tuple[str, str]:
    """Read foreground metadata in-process when macOS exposes it.

    The application PID from ``NSWorkspace`` is used to select the matching
    Quartz window.  Taking the first global window can attribute another
    application's title to the foreground app, which is unsafe for a privacy
    decision.
    """

    if sys.platform != "darwin":
        return "", ""
    try:
        import Quartz
        from AppKit import NSWorkspace

        application = NSWorkspace.sharedWorkspace().frontmostApplication()
        if application is None:
            return "", ""
        app = str(application.localizedName() or "").strip()
        pid = int(application.processIdentifier())
        options = Quartz.kCGWindowListOptionOnScreenOnly
        try:
            options |= Quartz.kCGWindowListExcludeDesktopElements
        # not a failure: an older Quartz without that option keeps the base list, which
        # is the one this asked for.
        except AttributeError:
            pass
        windows = Quartz.CGWindowListCopyWindowInfo(
            options,
            Quartz.kCGNullWindowID,
        ) or []
        for window in windows:
            if int(window.get("kCGWindowOwnerPID", 0) or 0) != pid:
                continue
            if int(window.get("kCGWindowLayer", 0) or 0) != 0:
                continue
            return app, str(window.get("kCGWindowName") or "").strip()
        return app, ""
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return "", ""


def frontmost_window_hint() -> tuple[str, str]:
    """(app, title) for the frontmost window, or ("", "") when unknown.

    An unknown foreground returns EMPTY rather than raising.  This function
    reports metadata rather than making policy; the universal capture policy
    treats unknown as refusal because intent cannot prove a foreground is
    non-private.
    """
    native = _native_frontmost_window_hint()
    if native[0] and native[1]:
        return native

    try:
        # Through the gateway, not raw: every process this runtime spawns is
        # accounted for in one place, and a read of the frontmost window is
        # not special enough to be the exception.
        completed = get_subprocess_gateway().run(
            ["osascript", "-e", _SCRIPT],
            timeout=_TIMEOUT_S,
            read_only=True,
            capture_output=True,
            source="screen_context.frontmost_window",
            accelerator_capability="none",
        )
    except (
        OSError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        record_degradation(
            "screen_context",
            exc,
            severity="debug",
            action="frontmost window unknown; the privacy gate had nothing to check",
        )
        return native
    if completed.returncode != 0:
        # Accessibility permission not granted is the common case here, and
        # it is not an error worth escalating: the caller finds out it cannot
        # read the screen either, one step later.
        return native
    raw = str(completed.stdout or "").strip()
    app, _, title = raw.partition("|")
    resolved = app.strip(), title.strip()
    return resolved if any(resolved) else native


__all__ = ["frontmost_window_hint"]
