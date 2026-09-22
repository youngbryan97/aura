"""Shared conversation-lane health semantics.

The desktop UI and transport heartbeats need to distinguish a broken chat lane
from a healthy lane that is currently occupied by the foreground generation.
"""
from __future__ import annotations

from typing import Any


def conversation_lane_is_busy(lane: dict[str, Any] | None) -> bool:
    """Return true when the foreground conversation lane is actively working."""

    if not isinstance(lane, dict):
        return False
    state = str(lane.get("state", "") or "").strip().lower()
    blockers = {
        str(item or "").strip()
        for item in (lane.get("readiness_blockers") or [])
        if str(item or "").strip()
    }
    reason = str(lane.get("last_failure_reason", "") or lane.get("last_error", "") or "").strip()
    try:
        active_generations = int(lane.get("active_generations", 0) or 0)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError, OverflowError):
        active_generations = 0
    active_work = (
        active_generations > 0
        or "active_generation_in_flight" in blockers
        or reason == "active_generation_in_flight"
    )
    if active_work:
        return True
    if state in {"spawning", "handshaking"}:
        return True
    if state == "warming" and (
        bool(lane.get("warmup_attempted", False))
        or bool(lane.get("warmup_in_flight", False))
        or any("warmup" in item or item == "visible_conversation_probe_missing" for item in blockers)
        or "warmup" in reason
        or reason == "visible_conversation_probe_missing"
    ):
        return True
    if state == "recovering" and bool(lane.get("warmup_in_flight", False)):
        return True
    return False


def conversation_lane_is_available(lane: dict[str, Any] | None) -> bool:
    """Return true when the lane is ready or legitimately busy answering."""

    if not isinstance(lane, dict):
        return False
    return bool(lane.get("conversation_ready", False)) or conversation_lane_is_busy(lane)


def conversation_lane_is_serving(lane: dict[str, Any] | None) -> bool:
    """True when a *functional* lane is actively answering a turn.

    This is the distinction the boot-readiness gate needs: "busy answering on
    a ready lane" (the desktop must connect and show the streaming reply) is a
    different state from "busy because the lane is still warming up /
    handshaking / spawning" (not yet able to serve). Conflating the two made a
    healthy, sustained-busy instance report ``ready: false``, so the desktop
    shell sat at "Connecting to runtime" through a long turn or a run of
    back-to-back turns (observed live, July 2026).

    Readiness here follows Kubernetes semantics: a server actively serving a
    request is ready, never un-ready.
    """
    if not isinstance(lane, dict):
        return False
    state = str(lane.get("state", "") or "").strip().lower()
    if state != "ready":
        return False
    blockers = {
        str(item or "").strip()
        for item in (lane.get("readiness_blockers") or [])
        if str(item or "").strip()
    }
    reason = str(lane.get("last_failure_reason", "") or lane.get("last_error", "") or "").strip()
    try:
        active_generations = int(lane.get("active_generations", 0) or 0)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError, OverflowError):
        active_generations = 0
    return (
        active_generations > 0
        or "active_generation_in_flight" in blockers
        or reason == "active_generation_in_flight"
    )
