"""What she brings to a turn besides the conversation: what she senses for it, and what she has verified that bears
on it, each as data for her desktop turn's grounding (core/brain/cognitive_engine_quick_reply.py).

What she has verified is her own learning from what she checked online (core/epistemics/what_she_has_verified.py,
L10): a claim she checked is used again rather than checked again, with its standing, its sources and when they were
read, offline as well as online.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.CognitiveEngine")


def grounding_she_brings(context: dict[str, Any], request: str) -> list[str]:
    """The sensory evidence read for this exact turn and the checked facts the request bears on, as blocks."""
    blocks: list[str] = []
    try:
        from core.senses.turn_evidence import sensory_evidence_grounding_block

        sensed = sensory_evidence_grounding_block(context.get("turn_sensory_evidence"))
        if sensed:
            blocks.append(sensed)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Turn sensory evidence unavailable: %s", exc)
    try:
        from core.epistemics.what_she_has_verified import as_evidence, her_verified_facts

        known = as_evidence(her_verified_facts().bearing_on(request))
        if known:
            blocks.append(known)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        logger.debug("Her checked facts unavailable for this turn: %s", exc)
    return blocks
