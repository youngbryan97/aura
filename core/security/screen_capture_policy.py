"""One privacy boundary for every read of the user's desktop.

Screen pixels and accessibility text are equivalent from a privacy
perspective: either can expose an incognito page, password manager, or other
foreground content.  Callers therefore do not decide this independently.
They ask this module immediately before a read, and an unknown foreground is
not treated as permission.

The admission object intentionally never contains the application or window
title.  A denial receipt that names a private window has leaked the metadata
the denial exists to protect.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.runtime.permission_gates import screen_allowed

_POLICY_SCHEMA = "aura.security.screen_capture_privacy_policy.v1"
SCREEN_CAPTURE_ADMISSION_SCHEMA = "aura.security.screen_capture_admission.v1"
_ADMISSION_SCHEMA = SCREEN_CAPTURE_ADMISSION_SCHEMA
_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "screen_capture_privacy_policy.json"


@dataclass(frozen=True, slots=True)
class _ScreenCapturePrivacyPolicy:
    private_window_markers: tuple[str, ...]
    private_apps: frozenset[str]
    private_browsing_apps: frozenset[str]
    private_pattern: re.Pattern[str]


def _normalized_policy_values(payload: Any, key: str) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    raw_values = payload.get(key)
    if not isinstance(raw_values, list):
        return ()
    values = tuple(
        value.strip().lower() for value in raw_values if isinstance(value, str) and value.strip()
    )
    if len(values) != len(raw_values) or len(values) != len(set(values)):
        return ()
    return values


@lru_cache(maxsize=1)
def _load_privacy_policy() -> _ScreenCapturePrivacyPolicy | None:
    """Load the one source of truth shared with the signed Swift bridge."""

    try:
        payload = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != _POLICY_SCHEMA:
        return None

    markers = _normalized_policy_values(payload, "private_window_markers")
    private_apps = _normalized_policy_values(payload, "private_apps")
    private_browsing_apps = _normalized_policy_values(payload, "private_browsing_apps")
    if not markers or not private_apps or not private_browsing_apps:
        return None
    return _ScreenCapturePrivacyPolicy(
        private_window_markers=markers,
        private_apps=frozenset(private_apps),
        private_browsing_apps=frozenset(private_browsing_apps),
        private_pattern=re.compile(
            "|".join(re.escape(marker) for marker in markers),
            re.IGNORECASE,
        ),
    )


class ScreenCaptureDenial(StrEnum):
    """Stable, non-disclosing reasons a desktop read did not happen."""

    NONE = "none"
    RUNTIME_SETTING_DISABLED = "runtime_setting_disabled"
    PRIVATE_FOREGROUND = "private_foreground"
    PRIVATE_VISIBLE = "private_visible"
    FOREGROUND_UNKNOWN = "foreground_unknown"
    BROWSER_TITLE_UNKNOWN = "browser_title_unknown"
    SESSION_LOCKED = "session_locked"
    POLICY_UNAVAILABLE = "policy_unavailable"


@dataclass(frozen=True, slots=True)
class ScreenCaptureAdmission:
    """Privacy-safe result of checking whether Aura may read the desktop."""

    allowed: bool
    reason: ScreenCaptureDenial = ScreenCaptureDenial.NONE
    context_known: bool = False
    authority: str = "python_fallback"

    @property
    def public_error(self) -> str:
        if self.allowed:
            return ""
        if self.reason is ScreenCaptureDenial.RUNTIME_SETTING_DISABLED:
            return "screen capture is disabled by permissions.screen"
        if self.reason in {
            ScreenCaptureDenial.PRIVATE_FOREGROUND,
            ScreenCaptureDenial.PRIVATE_VISIBLE,
        }:
            return "screen capture refused because private content is visible"
        if self.reason is ScreenCaptureDenial.SESSION_LOCKED:
            # Say what is true. "The interactive session is unavailable" is
            # accurate and tells the person nothing they can act on, and the
            # thing it is hiding is theirs: it is their own screen, and they
            # are the one who locked it.
            return "the screen is locked, so there is nothing for me to look at yet"
        return "screen capture refused because foreground privacy could not be verified"

    def to_receipt(self) -> dict[str, str | bool]:
        return {
            "schema": _ADMISSION_SCHEMA,
            "allowed": self.allowed,
            "reason": self.reason.value,
            "context_known": self.context_known,
            "authority": self.authority,
        }


class ScreenCaptureDeniedError(PermissionError):
    """Raised before a backend is touched when desktop reading is not admitted."""

    def __init__(self, admission: ScreenCaptureAdmission) -> None:
        self.admission = admission
        super().__init__(admission.public_error)


def is_private_screen_context(app: str, title: str) -> bool:
    """Return whether foreground metadata identifies a private context."""

    policy = _load_privacy_policy()
    if policy is None:
        return True
    normalized_app = str(app or "").strip().lower()
    if normalized_app in policy.private_apps:
        return True
    return bool(policy.private_pattern.search(f"{app or ''} {title or ''}"))


def _admission_from_context(
    context: tuple[str, str] | None,
    *,
    authority: str,
) -> ScreenCaptureAdmission:
    policy = _load_privacy_policy()
    if policy is None:
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.POLICY_UNAVAILABLE,
            authority=authority,
        )

    try:
        app, title = context or ("", "")
    except (TypeError, ValueError):
        app, title = "", ""
    app = str(app or "").strip()
    title = str(title or "").strip()

    if not app and not title:
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.FOREGROUND_UNKNOWN,
            authority=authority,
        )
    if is_private_screen_context(app, title):
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.PRIVATE_FOREGROUND,
            context_known=True,
            authority=authority,
        )
    # A browser name without a readable title cannot prove that the active
    # window is not private. Other apps can legitimately have no titled window.
    if app.lower() in policy.private_browsing_apps and not title:
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.BROWSER_TITLE_UNKNOWN,
            context_known=False,
            authority=authority,
        )
    return ScreenCaptureAdmission(
        allowed=True,
        context_known=True,
        authority=authority,
    )


def _resident_bridge_capture_admission() -> ScreenCaptureAdmission | None:
    """Return the resident Aura.app decision, never a one-shot bridge decision."""

    if sys.platform != "darwin":
        return None
    try:
        from core.security.native_desktop_bridge import invoke_native_desktop_bridge

        result = invoke_native_desktop_bridge(
            "foreground_capture_admission",
            read_only=True,
            timeout=0.75,
            allow_one_shot=False,
        )
    except (ImportError, OSError, RuntimeError, TimeoutError, TypeError, ValueError):
        return None
    if result.get("bridge_transport") != "resident_ipc" or not result.get("ok"):
        return None
    receipt = result.get("capture_admission")
    if not isinstance(receipt, dict):
        return None
    if receipt.get("schema") != _ADMISSION_SCHEMA or receipt.get("authority") != "resident_bridge":
        return None
    try:
        reason = ScreenCaptureDenial(str(receipt.get("reason", "")))
    except ValueError:
        return None
    allowed = receipt.get("allowed")
    context_known = receipt.get("context_known")
    if not isinstance(allowed, bool) or not isinstance(context_known, bool):
        return None
    if allowed != (reason is ScreenCaptureDenial.NONE):
        return None
    if allowed and not context_known:
        return None
    return ScreenCaptureAdmission(
        allowed=allowed,
        reason=reason,
        context_known=context_known,
        authority="resident_bridge",
    )


def _admission_from_visible_windows(
    windows: Any,
    *,
    foreground_pid: int = 0,
    authority: str,
) -> ScreenCaptureAdmission:
    """Evaluate a complete macOS on-screen window inventory.

    A full-display screenshot can expose any visible window, not only the
    foreground one. This is the Python equivalent of the signed Swift bridge's
    admission pass: every ordinary layer-zero window is checked against the
    same repository policy before pixels are acquired. No owner or title is
    retained in the receipt.
    """

    policy = _load_privacy_policy()
    if policy is None:
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.POLICY_UNAVAILABLE,
            authority=authority,
        )
    if not isinstance(windows, (list, tuple)) or not windows:
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.FOREGROUND_UNKNOWN,
            authority=authority,
        )

    inspected = 0
    for raw_window in windows:
        if not isinstance(raw_window, Mapping):
            return ScreenCaptureAdmission(
                allowed=False,
                reason=ScreenCaptureDenial.FOREGROUND_UNKNOWN,
                authority=authority,
            )
        try:
            layer = int(raw_window.get("kCGWindowLayer", raw_window.get("layer", 0)) or 0)
            owner_pid = int(
                raw_window.get("kCGWindowOwnerPID", raw_window.get("owner_pid", 0)) or 0
            )
        except (TypeError, ValueError):
            return ScreenCaptureAdmission(
                allowed=False,
                reason=ScreenCaptureDenial.FOREGROUND_UNKNOWN,
                authority=authority,
            )
        if layer != 0:
            continue
        owner = str(raw_window.get("kCGWindowOwnerName", raw_window.get("owner", "")) or "").strip()
        title = str(raw_window.get("kCGWindowName", raw_window.get("title", "")) or "").strip()
        if not owner:
            # An unnamed layer-zero window cannot be checked against the app
            # denylist. Treat an incomplete inventory as unknown, never public.
            return ScreenCaptureAdmission(
                allowed=False,
                reason=ScreenCaptureDenial.FOREGROUND_UNKNOWN,
                authority=authority,
            )
        inspected += 1
        normalized_owner = owner.lower()
        combined = f"{owner} {title}"
        if normalized_owner in policy.private_apps or policy.private_pattern.search(combined):
            return ScreenCaptureAdmission(
                allowed=False,
                reason=(
                    ScreenCaptureDenial.PRIVATE_FOREGROUND
                    if foreground_pid > 0 and owner_pid == foreground_pid
                    else ScreenCaptureDenial.PRIVATE_VISIBLE
                ),
                context_known=True,
                authority=authority,
            )
        if normalized_owner in policy.private_browsing_apps and not title:
            return ScreenCaptureAdmission(
                allowed=False,
                reason=ScreenCaptureDenial.BROWSER_TITLE_UNKNOWN,
                authority=authority,
            )

    if inspected == 0:
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.FOREGROUND_UNKNOWN,
            authority=authority,
        )
    return ScreenCaptureAdmission(
        allowed=True,
        context_known=True,
        authority=authority,
    )


def _python_macos_capture_admission() -> ScreenCaptureAdmission | None:
    """Use the host process's complete CoreGraphics inventory when available."""

    if sys.platform != "darwin":
        return None
    authority = "python_visible_windows"
    try:
        import Quartz
        from AppKit import NSWorkspace

        session = Quartz.CGSessionCopyCurrentDictionary() or {}
        if bool(session.get("CGSSessionScreenIsLocked", False)):
            return ScreenCaptureAdmission(
                allowed=False,
                reason=ScreenCaptureDenial.SESSION_LOCKED,
                authority=authority,
            )
        on_console = session.get("kCGSSessionOnConsoleKey")
        login_done = session.get("kCGSessionLoginDoneKey")
        if (on_console is not None and not bool(on_console)) or (
            login_done is not None and not bool(login_done)
        ):
            return ScreenCaptureAdmission(
                allowed=False,
                reason=ScreenCaptureDenial.SESSION_LOCKED,
                authority=authority,
            )

        foreground = NSWorkspace.sharedWorkspace().frontmostApplication()
        foreground_pid = int(foreground.processIdentifier()) if foreground is not None else 0
        options = Quartz.kCGWindowListOptionOnScreenOnly
        options |= Quartz.kCGWindowListExcludeDesktopElements
        windows = (
            Quartz.CGWindowListCopyWindowInfo(
                options,
                Quartz.kCGNullWindowID,
            )
            or []
        )
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return _admission_from_visible_windows(
        list(windows),
        foreground_pid=foreground_pid,
        authority=authority,
    )


def evaluate_screen_capture_admission(
    *,
    context: tuple[str, str] | None = None,
) -> ScreenCaptureAdmission:
    """Evaluate the universal desktop-read policy without acquiring pixels.

    ``context`` is injectable for deterministic tests.  Production callers
    omit it and use the bounded foreground metadata probe.
    """

    if not screen_allowed():
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.RUNTIME_SETTING_DISABLED,
            authority="runtime_setting",
        )

    if context is not None:
        return _admission_from_context(context, authority="provided_context")

    resident_admission = _resident_bridge_capture_admission()
    if resident_admission is not None:
        return resident_admission
    if sys.platform == "darwin":
        visible_windows_admission = _python_macos_capture_admission()
        if visible_windows_admission is not None:
            return visible_windows_admission
        # A full-display capture can include visible windows on every monitor.
        # If neither authority can enumerate all of them, a frontmost-only
        # probe is not strong enough to permit capture.
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.FOREGROUND_UNKNOWN,
            authority="visible_window_authority_unavailable",
        )

    try:
        from core.senses.screen_context import frontmost_window_hint

        context = frontmost_window_hint()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        context = ("", "")
    return _admission_from_context(context, authority="python_fallback")


def require_screen_capture_admission(
    *,
    context: tuple[str, str] | None = None,
) -> ScreenCaptureAdmission:
    admission = evaluate_screen_capture_admission(context=context)
    if not admission.allowed:
        raise ScreenCaptureDeniedError(admission)
    return admission


async def evaluate_screen_capture_admission_async() -> ScreenCaptureAdmission:
    """Run the bounded metadata probe off the event loop."""

    return await asyncio.to_thread(evaluate_screen_capture_admission)


async def require_screen_capture_admission_async() -> ScreenCaptureAdmission:
    admission = await evaluate_screen_capture_admission_async()
    if not admission.allowed:
        raise ScreenCaptureDeniedError(admission)
    return admission


__all__ = [
    "SCREEN_CAPTURE_ADMISSION_SCHEMA",
    "ScreenCaptureAdmission",
    "ScreenCaptureDeniedError",
    "ScreenCaptureDenial",
    "evaluate_screen_capture_admission",
    "evaluate_screen_capture_admission_async",
    "is_private_screen_context",
    "require_screen_capture_admission",
    "require_screen_capture_admission_async",
]
