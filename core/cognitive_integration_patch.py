"""Retired CognitiveIntegrationLayer patch compatibility shim.

The Phase 7 history threading, inline inference, and phenomenal context
injection now live directly in :mod:`core.cognitive_integration_layer`.
Keeping this module as a no-op preserves old imports without reintroducing
runtime monkey-patching.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("Aura.CILPatch")

__all__ = ["patch_cognitive_integration"]


def patch_cognitive_integration() -> None:
    """Compatibility entry point; the canonical layer is already patched."""
    logger.debug("patch_cognitive_integration is retired; canonical CIL is active")
