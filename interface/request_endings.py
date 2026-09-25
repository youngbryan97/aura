"""Whether a request that ended without a response had anybody waiting for it."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.Server")

__all__ = ["nobody_is_waiting"]


async def nobody_is_waiting(request: Any, exc: Exception) -> str:
    """Why a request that ended without a response has nobody to answer, or nothing.

    A route cancelled under a middleware reaches the exception handler as "No
    response returned." That is an error when somebody is still waiting for the
    answer, and an ending when the client has gone or Aura is shutting down:
    LIVE 2026-09-24, logged as an unhandled exception on a clean shutdown.
    """
    if not (isinstance(exc, RuntimeError) and str(exc) == "No response returned."):
        return ""
    try:
        if await request.is_disconnected():
            return "the client went away"
    except (RuntimeError, OSError) as gone:
        logger.debug("could not ask whether the client is still there: %s", gone)
    from core.runtime.shutdown_coordinator import is_shutdown_requested

    return "Aura is shutting down" if is_shutdown_requested() else ""
