"""Frontmost-application sensing without a subprocess per poll.

The perception daemon and the app-focus sensor both need the name of the frontmost
macOS application. They each shelled out to ``osascript`` (an AppleScript/System-Events
round trip) on every tick — the perception daemon does this on a ~500ms cadence, so it
was forking a process twice a second just to read one string.

``NSWorkspace.sharedWorkspace().frontmostApplication()`` answers the same question with
an in-process Cocoa call: no fork, no AppleScript compile, microseconds instead of a
process spawn with a 1.5s timeout budget. We import PyObjC lazily and cache the result of
the probe, falling back to the old osascript path only when AppKit is genuinely
unavailable (non-macOS, headless, or a stripped runtime).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("Perception.FrontmostApp")

# Tri-state cache for the PyObjC probe: None = not yet probed, True/False = result.
_PYOBJC_OK: bool | None = None
_NSWorkspace = None  # cached class once the import succeeds


def _load_nsworkspace() -> bool:
    """Lazily import AppKit.NSWorkspace once; cache whether it's usable."""
    global _PYOBJC_OK, _NSWorkspace
    if _PYOBJC_OK is not None:
        return _PYOBJC_OK
    try:
        from AppKit import NSWorkspace  # type: ignore

        _NSWorkspace = NSWorkspace
        _PYOBJC_OK = True
    except (ImportError, Exception) as exc:  # noqa: BLE001 - any import-time failure → fallback
        logger.debug("PyObjC NSWorkspace unavailable, will fall back to osascript: %s", exc)
        _PYOBJC_OK = False
    return _PYOBJC_OK


def frontmost_app_name_fast() -> str | None:
    """Return the frontmost application's name via NSWorkspace, or None.

    Pure in-process Cocoa call — safe to invoke on a hot polling loop. Returns None if
    PyObjC is unavailable or the lookup fails, so callers can fall back to osascript.
    """
    if not _load_nsworkspace():
        return None
    try:
        app = _NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        name = app.localizedName()
        return str(name) if name else None
    except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.debug("NSWorkspace frontmostApplication failed: %s", exc)
        return None


def pyobjc_available() -> bool:
    """Whether the fast in-process path is usable (for diagnostics/tests)."""
    return _load_nsworkspace()
