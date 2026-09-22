"""The seam between "she found it" and a rectangle on the actual screen.

WHY THIS FILE EXISTS
────────────────────
``screen_highlight`` locates a thing in the accessibility tree and then asks a
``desktop_overlay`` runtime service to draw a box around it. Nothing ever
registered that service, and nothing ever sent the launcher the message it was
waiting for, so the whole pointing half of "she can point at it, or say she
could not find it" was unreachable: ``get_runtime_service`` returned None,
``_draw`` returned False, and every single highlight came back "overlay
declined". The Swift handler that draws the rectangle was dead code the day it
was written.

A Python process cannot put a click-through window over another app's screen.
The launcher owns that AppKit surface, and the only channel to it is the
WebKit message handler on the bubble's page. So the rectangle travels:

    screen_highlight  →  this service  →  AmbientPresence queue
                      →  /api/ambient/state?surface=native-bubble  (the bubble polls)
                      →  window.webkit.messageHandlers.auraBubble
                      →  AuraLauncher.showHighlight

using the poll the bubble is already making. No new socket, no new port, no
second lane into the launcher.

THE HONEST FALSE
────────────────
``show_rect`` returns False whenever nothing can actually draw — no launcher,
bubble hidden, running headless from a terminal. That False is the whole point
of routing through a liveness check rather than fire-and-forget: the caller
turns it into "I could not point at it, it is in the top-right of the
sidebar", which is true. Claiming a highlight nobody drew is the one outcome
worse than not pointing, because the person then goes looking for a box that
was never on their screen.
"""

from __future__ import annotations

from typing import Any

from core.runtime.errors import record_degradation

OVERLAY_SERVICE = "desktop_overlay"
OVERLAY_SCHEMA = "aura.perception.desktop_overlay.v1"


class BubbleOverlay:
    """Draws by asking the bubble's host, or honestly reports that it cannot."""

    schema = OVERLAY_SCHEMA

    def show_rect(
        self,
        *,
        x: float,
        y: float,
        width: float,
        height: float,
        seconds: float,
    ) -> bool:
        """Queue a rectangle for the bubble's host to draw. False if none can.

        Synchronous and cheap: this only enqueues. The draw happens when the
        bubble next polls, which is within its active cadence — fast enough
        that the rectangle is up before she has finished saying where to look.
        """
        try:
            from core.perception.ambient_presence import get_ambient_presence

            return bool(
                get_ambient_presence().request_highlight(
                    x=x, y=y, width=width, height=height, seconds=seconds
                )
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "desktop_overlay",
                exc,
                severity="warning",
                action=(
                    "highlight not queued; she will describe the location in "
                    "words instead of claiming to have pointed at it"
                ),
            )
            return False

    def available(self) -> bool:
        """Is there a live surface that would draw right now?"""
        try:
            from core.perception.ambient_presence import get_ambient_presence

            return bool(get_ambient_presence().drawing_surface_attached())
        # not a failure: no ambient presence means there is no surface drawing.
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return False

    def move_to(self, *, x: float, y: float) -> int | None:
        """Request native movement and return its acknowledgement sequence."""
        try:
            from core.perception.ambient_presence import get_ambient_presence

            return get_ambient_presence().request_bubble_move(x=x, y=y)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "desktop_overlay",
                exc,
                severity="warning",
                action="companion movement not queued; the bubble stayed where it was",
            )
            return None


def install_desktop_overlay() -> bool:
    """Publish the overlay service. Safe to call more than once.

    Registered as NOT required: a runtime with no launcher — headless, a test,
    a terminal session — is a runtime where she describes locations in words,
    which is a supported way for her to answer and not a degraded boot.
    """
    try:
        from core.runtime.service_registry import register_runtime_service

        return bool(
            register_runtime_service(
                OVERLAY_SERVICE,
                BubbleOverlay(),
                required=False,
                owner="core.perception.desktop_overlay",
                registered_by="install_desktop_overlay",
                required_for="pointing at things on screen",
            )
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "desktop_overlay",
            exc,
            severity="warning",
            action=(
                "overlay service not registered; highlights will refuse and "
                "she will describe locations in words"
            ),
        )
        return False


def get_desktop_overlay() -> Any:
    from core.runtime.service_registry import get_runtime_service

    return get_runtime_service(OVERLAY_SERVICE, default=None)


__all__ = [
    "OVERLAY_SCHEMA",
    "OVERLAY_SERVICE",
    "BubbleOverlay",
    "get_desktop_overlay",
    "install_desktop_overlay",
]
