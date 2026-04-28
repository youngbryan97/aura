"""
core/memory_compaction_patch.py
=================================
Response Pipeline Patch — Working Memory Compaction

GAP — compact_working_memory() is defined but never called.

  Location: core/brain/llm/context_limit.py
  Function: compact_working_memory(chat_history, max_raw_turns=4)
  Status:   Fully implemented. Called 0 times from the live pipeline.

  What it does (already written, just unwired):
    - Keeps the most recent max_raw_turns*2 messages verbatim
    - Compresses older messages into a semantic LLM-generated summary
    - Keeps token count flat indefinitely across long sessions

  Without it: working_memory grows unbounded. In a long session,
  every response generation call passes the full conversation history
  to the LLM, burning context window and degrading response quality
  as the window fills with stale content the TokenGovernor has to
  prune heuristically anyway.

WHAT THIS PATCH DOES:
  Wires compact_working_memory() into the MemoryConsolidationPhase
  and into the legacy orchestrator tick. Specifically:

  1. Patches MemoryConsolidationPhase.execute() to call compaction
     when working_memory exceeds COMPACTION_THRESHOLD messages.
     Runs after the response is generated, before state is saved.
     Async, non-blocking to the user response.

  2. Adds a standalone compact_if_needed() function that the
     legacy orchestrator can call from its post-response hook.

  3. Adds session-level compaction stats to state for observability.

THRESHOLDS:
  COMPACTION_THRESHOLD = 30 messages (15 turns)
    Below this, TokenGovernor allocation handles it fine.
    Above this, compaction pays for itself in quality + speed.
  MAX_RAW_TURNS = 6
    Keep 12 messages verbatim (last 6 turns).
    Everything older becomes a compressed summary.

INSTALL:
  from core.memory_compaction_patch import patch_memory_compaction
  patch_memory_compaction()
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.MemoryCompactionPatch")

COMPACTION_THRESHOLD = 30   # messages before compaction triggers
MAX_RAW_TURNS        = 6    # turns to keep verbatim after compaction


# ─────────────────────────────────────────────────────────────────────────────
# Core compaction call
# ─────────────────────────────────────────────────────────────────────────────

async def compact_if_needed(
    working_memory: List[Dict[str, Any]],
    force: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run compaction if working_memory exceeds COMPACTION_THRESHOLD.
    Returns the (possibly compacted) memory list.
    Safe to call every turn — threshold check is cheap.
    """
    if not force and len(working_memory) < COMPACTION_THRESHOLD:
        return working_memory

    try:
        from core.brain.llm.context_limit import compact_working_memory
        compacted = await compact_working_memory(
            working_memory,
            max_raw_turns=MAX_RAW_TURNS,
        )
        original_len = len(working_memory)
        new_len      = len(compacted) if compacted else len(working_memory)
        if compacted and new_len < original_len:
            logger.info(
                "🗜️  Working memory compacted: %d → %d messages",
                original_len, new_len,
            )
            return compacted
    except Exception as exc:
        record_degradation('memory_compaction_patch', exc)
        logger.warning("MemoryCompactionPatch: compaction failed — %s", exc)

    return working_memory


# ─────────────────────────────────────────────────────────────────────────────
# Patched MemoryConsolidationPhase.execute()
# ─────────────────────────────────────────────────────────────────────────────

async def _patched_memory_consolidation_execute(
    self: Any,
    state: Any,
    objective: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Replacement for MemoryConsolidationPhase.execute().

    Runs the original consolidation logic first, then triggers
    compaction if working_memory has grown past threshold.
    The compaction is awaited before state is returned so the
    next turn starts with a clean, compact history.
    """
    # Run original consolidation
    try:
        state = await self._original_execute(state, objective, **kwargs)
    except Exception as exc:
        record_degradation('memory_compaction_patch', exc)
        logger.error("MemoryCompactionPatch: original execute failed — %s", exc)
        return state

    # Compact if needed
    if hasattr(state, "cognition") and hasattr(state.cognition, "working_memory"):
        wm = state.cognition.working_memory
        if isinstance(wm, list) and len(wm) >= COMPACTION_THRESHOLD:
            compacted = await compact_if_needed(wm)
            if compacted is not wm:
                state.cognition.working_memory = compacted
                # Track stats
                if not hasattr(state.cognition, "compaction_count"):
                    state.cognition.compaction_count = 0
                state.cognition.compaction_count += 1

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Patch application
# ─────────────────────────────────────────────────────────────────────────────

def patch_memory_compaction() -> None:
    """
    Wire compact_working_memory into MemoryConsolidationPhase.
    Idempotent.
    """
    try:
        from core.phases.memory_consolidation import MemoryConsolidationPhase
    except ImportError as exc:
        logger.warning(
            "MemoryCompactionPatch: MemoryConsolidationPhase not found — %s. "
            "Patch skipped. Call compact_if_needed() manually from your "
            "post-response hook if needed.", exc
        )
        return

    if getattr(MemoryConsolidationPhase, "_compaction_patched", False):
        logger.debug("MemoryCompactionPatch: already applied")
        return

    # Preserve original execute under a new name
    MemoryConsolidationPhase._original_execute = MemoryConsolidationPhase.execute

    # Replace with patched version
    MemoryConsolidationPhase.execute = _patched_memory_consolidation_execute
    MemoryConsolidationPhase._compaction_patched = True

    logger.info(
        "✅ MemoryCompactionPatch applied — compaction triggers at %d messages, "
        "keeps last %d turns verbatim",
        COMPACTION_THRESHOLD, MAX_RAW_TURNS,
    )
