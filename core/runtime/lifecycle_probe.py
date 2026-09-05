"""Bounded lifecycle barriers for deterministic external shutdown proofs."""

from __future__ import annotations

import asyncio
import logging
import os
import time

logger = logging.getLogger("Aura.LifecycleProbe")

_MAX_HOLD_SECONDS = 2.0



def _flag_value(name, kind_name, default, description, owner):
    """Read a knob through the typed flag layer (C1); env stays live."""
    try:
        from core.runtime.flags import FlagKind, declare

        return declare(
            name,
            kind=getattr(FlagKind, kind_name),
            default=default,
            description=description,
            owner=owner,
        ).value()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return default

def shutdown_probe_hold_seconds(target: str) -> float:
    if not _flag_value("AURA_SHUTDOWN_PROBE_ENABLED", "BOOL", False,
                   "Enable the shutdown-probe hold harness", "core.runtime.lifecycle_probe"):
        return 0.0
    configured_target = str(_flag_value("AURA_SHUTDOWN_PROBE_TARGET", "STRING", "",
                       "Which shutdown probe point to hold", "core.runtime.lifecycle_probe")).strip()
    if configured_target != str(target).strip():
        return 0.0
    try:
        requested = float(_flag_value("AURA_SHUTDOWN_PROBE_HOLD_SECONDS", "FLOAT", 0.75,
                  "Seconds to hold at the shutdown probe point", "core.runtime.lifecycle_probe"))
    except (TypeError, ValueError):
        requested = 0.75
    return min(_MAX_HOLD_SECONDS, max(0.05, requested))


def _log_started(target: str, hold_seconds: float) -> None:
    logger.info(
        "Lifecycle probe hold started "
        "(target=%s started_at_unix=%.6f hold_seconds=%.3f)",
        target,
        time.time(),
        hold_seconds,
    )


def _log_completed(target: str) -> None:
    logger.info(
        "Lifecycle probe hold completed (target=%s completed_at_unix=%.6f)",
        target,
        time.time(),
    )


async def hold_shutdown_probe_async(target: str) -> float:
    hold_seconds = shutdown_probe_hold_seconds(target)
    if hold_seconds <= 0.0:
        return 0.0
    _log_started(target, hold_seconds)
    await asyncio.sleep(hold_seconds)
    _log_completed(target)
    return hold_seconds


def hold_shutdown_probe_sync(target: str) -> float:
    hold_seconds = shutdown_probe_hold_seconds(target)
    if hold_seconds <= 0.0:
        return 0.0
    _log_started(target, hold_seconds)
    time.sleep(hold_seconds)
    _log_completed(target)
    return hold_seconds
