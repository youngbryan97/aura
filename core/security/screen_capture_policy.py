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
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.runtime.permission_gates import screen_allowed

_POLICY_SCHEMA = "aura.security.screen_capture_privacy_policy.v1"
SCREEN_CAPTURE_ADMISSION_SCHEMA = "aura.security.screen_capture_admission.v1"
_ADMISSION_SCHEMA = SCREEN_CAPTURE_ADMISSION_SCHEMA
_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "screen_capture_privacy_policy.json"
)


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
        value.strip().lower()
        for value in raw_values
        if isinstance(value, str) and value.strip()
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
    private_browsing_apps = _normalized_policy_values(
        payload, "private_browsing_apps"
    )
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
            return "screen capture deferred while the interactive session is unavailable"
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
    if (
        receipt.get("schema") != _ADMISSION_SCHEMA
        or receipt.get("authority") != "resident_bridge"
    ):
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
        # A full-display capture can include visible windows on every monitor.
        # The Python fallback sees only the frontmost title, so it cannot prove
        # that a background incognito/password-manager window is absent.
        return ScreenCaptureAdmission(
            allowed=False,
            reason=ScreenCaptureDenial.FOREGROUND_UNKNOWN,
            authority="resident_bridge_unavailable",
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
