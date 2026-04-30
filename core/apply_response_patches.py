"""Legacy compatibility shim for the retired response patch hook.

The response-layer fixes that once lived behind runtime monkey-patching are now
implemented directly in the first-class modules. This entry point remains only
so older boot paths can call it safely without mutating live classes.
"""
from __future__ import annotations


import logging
from typing import Any

logger = logging.getLogger("Aura.ResponsePatches")

# Module-level sentinel
_applied = False


def apply_response_patches(orchestrator: Any | None = None) -> None:
    """No-op compatibility hook.

    Kept for older startup code, but deliberately performs no monkey-patching.
    """
    global _applied
    if _applied:
        logger.debug("apply_response_patches: already applied")
        return

    _applied = True
    logger.info("🧠 Response pipeline is running on native modules; legacy patch hook is a no-op.")
