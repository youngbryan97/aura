"""
core/apply_response_patches.py
================================
Unified entry point — applies all response/cognition pipeline patches.

One call after orchestrator boot:

    from core.apply_response_patches import apply_response_patches
    apply_response_patches()

Patches applied:
  1. ContextAssembler — casual routing, fake ack removal, personality
                        preservation, attention_focus writer
  2. CognitiveIntegrationLayer — history threading, inline inference,
                                  phenomenal context injection
  3. MemoryConsolidationPhase — wires existing compact_working_memory

All patches are idempotent. Failures are logged but do not raise —
the system degrades gracefully to original behaviour.

Also pairs with core/consciousness/apply_patches.py:

    from core.consciousness.apply_patches import apply_consciousness_patches
    from core.apply_response_patches import apply_response_patches

    apply_consciousness_patches(orchestrator)   # experience layer
    apply_response_patches()                    # response layer
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("Aura.ResponsePatches")

# Module-level sentinel
_applied = False


def apply_response_patches(orchestrator: Optional[Any] = None) -> None:
    """Apply all three response/cognition pipeline patches."""
    global _applied
    if _applied:
        logger.debug("apply_response_patches: already applied")
        return

    # ── Patch 1: ContextAssembler ─────────────────────────────────────────────
    try:
        from core.brain.llm.context_assembler_patch import patch_context_assembler
        patch_context_assembler()
    except Exception as exc:
        logger.error("apply_response_patches: ContextAssembler patch failed — %s",
                     exc, exc_info=True)

    # ── Patch 2: CognitiveIntegrationLayer ───────────────────────────────────
    try:
        from core.cognitive_integration_patch import patch_cognitive_integration
        patch_cognitive_integration()
    except Exception as exc:
        logger.error("apply_response_patches: CIL patch failed — %s",
                     exc, exc_info=True)

    # ── Patch 3: Memory compaction ────────────────────────────────────────────
    try:
        from core.memory_compaction_patch import patch_memory_compaction
        patch_memory_compaction()
    except Exception as exc:
        logger.error("apply_response_patches: memory compaction patch failed — %s",
                     exc, exc_info=True)

    _applied = True
    logger.info("🧠 All response pipeline patches applied")
