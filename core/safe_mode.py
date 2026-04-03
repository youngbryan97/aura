"""
core/safe_mode.py
──────────────────
Safe mode configuration and orchestrator patches.

Provides two things:
  1. SAFE_MODE_CONFIG — a config overlay that disables non-essential
     subsystems so you can get a stable LLM ↔ voice ↔ chat loop running
     before re-enabling advanced features one at a time.

  2. apply_orchestrator_patches() — monkey-patches the existing orchestrator
     with the fixes from the code review without requiring a full rewrite.
     Apply these at boot after orchestrator creation.

Usage in aura_main.py or orchestrator_boot.py:
    from core.safe_mode import apply_orchestrator_patches, SAFE_MODE_CONFIG
    apply_orchestrator_patches(orchestrator, safe_mode=False)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.SafeMode")


# ── Safe Mode Feature Flags ───────────────────────────────────────────────────

SAFE_MODE_CONFIG = {
    # Core features — always on
    "llm_enabled": True,
    "memory_basic": True,          # Episodic + conversation history
    "personality_engine": True,
    "voice_stt": True,
    "voice_tts": True,

    # Features disabled in safe mode
    "self_modification": False,     # No autonomous code rewrites
    "self_preservation": False,     # No network scanning / replication
    "persona_evolution": False,     # No personality drift during session
    "dream_cycle": False,           # No dead-letter re-injection
    "device_discovery": False,      # No network scanning
    "stealth_mode": False,          # No VPN / IP spoofing
    "singularity_monitor": False,   # No thought interval compression
    "vector_memory_write": True,    # Still write memories, just don't prune yet
    "context_pruning": False,       # Disabled until LLM is stable
    "memory_consolidation": False,  # Disabled until LLM is stable

    # Tuning
    "autonomous_thought_interval_s": 60.0,  # Longer interval, less noise
    "health_poll_interval_ms": 10000,        # Frontend health check interval
    "max_conversation_history": 50,          # Smaller history = less RAM
    "singularity_acceleration_cap": 1.0,    # No compression in safe mode
}

FULL_MODE_CONFIG = {
    # Everything on, but with caps
    "llm_enabled": True,
    "memory_basic": True,
    "personality_engine": True,
    "voice_stt": True,
    "voice_tts": True,
    "self_modification": True,
    "self_preservation": False,     # Keep this off — see code review
    "persona_evolution": True,
    "dream_cycle": True,
    "device_discovery": False,      # Keep this off — see code review
    "stealth_mode": True,
    "singularity_monitor": True,
    "vector_memory_write": True,
    "context_pruning": True,
    "memory_consolidation": True,
    "autonomous_thought_interval_s": 45.0,
    "health_poll_interval_ms": 5000,
    "max_conversation_history": 100,
    "singularity_acceleration_cap": 2.0,   # Cap at 2x even in full mode
}


# ── Orchestrator Patches ──────────────────────────────────────────────────────

def apply_orchestrator_patches(orchestrator, safe_mode: bool = False):
    """
    Apply targeted patches to the existing orchestrator instance.
    These fix the specific bugs identified without rewriting the orchestrator.
    """
    # Dynamic Volition Integration
    kernel = getattr(orchestrator, 'kernel', None)
    volition = getattr(kernel, 'volition_level', 0) if kernel else 0
    
    config = SAFE_MODE_CONFIG if safe_mode else FULL_MODE_CONFIG
    
    # Overwrite config based on Volition Level if not in strict safe_mode
    if not safe_mode:
        if volition >= 1:
            config["dream_cycle"] = True
            config["context_pruning"] = True
            config["memory_consolidation"] = True
        if volition >= 2:
            config["persona_evolution"] = True
        if volition >= 3:
            config["self_modification"] = True
            config["self_preservation"] = True
            config["singularity_monitor"] = True

    logger.info("Applying orchestrator patches (safe_mode=%s, volition=%d)", safe_mode, volition)

    # Patch 1: Fix singularity acceleration cap
    _patch_singularity_cap(orchestrator, config["singularity_acceleration_cap"])

    # Patch 2: Fix the autonomous thought interval
    _patch_thought_interval(orchestrator, config["autonomous_thought_interval_s"])

    # Patch 3: Replace process_user_input with a version that
    # properly flushes the queue and waits for the right reply
    _patch_process_user_input(orchestrator)

    # Patch 4: Disable subsystems per config
    _apply_feature_flags(orchestrator, config)

    # Patch 5: Fix persona evolver trait bounds
    _patch_persona_evolver(orchestrator)

    # Patch 6: Fix context pruner to validate output
    _patch_context_pruner(orchestrator)

    logger.info("Orchestrator patches applied")


def _patch_singularity_cap(orchestrator, cap: float):
    """Cap the singularity acceleration factor."""
    if hasattr(orchestrator, 'singularity_monitor') and orchestrator.singularity_monitor:
        monitor = orchestrator.singularity_monitor
        original_factor = getattr(monitor, 'acceleration_factor', 1.0)
        if original_factor > cap:
            monitor.acceleration_factor = cap
            logger.info(
                "Singularity factor capped: %.2f → %.2f",
                original_factor, cap
            )

    if hasattr(orchestrator, 'cognitive_engine') and orchestrator.cognitive_engine:
        engine = orchestrator.cognitive_engine
        if hasattr(engine, 'singularity_factor'):
            original = engine.singularity_factor
            engine.singularity_factor = min(original, cap)
            if original != engine.singularity_factor:
                logger.info(
                    "CognitiveEngine singularity_factor capped: %.2f → %.2f",
                    original, engine.singularity_factor
                )


def _patch_thought_interval(orchestrator, interval_s: float):
    """Set a minimum autonomous thought interval."""
    # Store on the orchestrator for the threshold calculation
    orchestrator._min_thought_interval = interval_s
    logger.info("Autonomous thought interval set to %.0fs", interval_s)


def _patch_process_user_input(orchestrator):
    """
    Replace process_user_input with a version that properly handles
    the reply queue race condition.
    """
    original_fn = orchestrator.process_user_input

    async def patched_process_user_input(message: str, origin: str = "user") -> Optional[str]:
        """
        Patched version with proper queue flush and response validation.
        """
        # 1. Interrupt autonomous thought if running
        if (origin in ("user", "voice") and
                hasattr(orchestrator, '_current_thought_task') and
                orchestrator._current_thought_task is not None and
                not orchestrator._current_thought_task.done()):

            logger.debug("Patched: Interrupting autonomous thought for %s input", origin)
            orchestrator._current_thought_task.cancel()
            try:
                # Actually await the cancellation (original code didn't always do this)
                await asyncio.wait_for(
                    asyncio.shield(orchestrator._current_thought_task),
                    timeout=3.0
                )
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception) as exc:
                logger.debug('Exception caught during execution: %s', exc)
            finally:
                orchestrator._current_thought_task = None

        # 2. Flush the reply queue AFTER cancellation completes
        # (Original code flushed before cancellation, missing late puts)
        await asyncio.sleep(0.05)  # Allow cancelled task's finally to run
        if hasattr(orchestrator, 'reply_queue'):
            while not orchestrator.reply_queue.empty():
                try:
                    orchestrator.reply_queue.get_nowait()
                except Exception:
                    break

        # 3. Call the original function
        result = await original_fn(message, origin)

        # 4. Validate the result
        if result and result.strip():
            return result
        else:
            logger.warning(
                "Patched process_user_input: empty result for %r (origin=%s)",
                message[:50], origin
            )
            return None

    orchestrator.process_user_input = patched_process_user_input
    logger.info("Patched: process_user_input (queue race condition fix)")


def _apply_feature_flags(orchestrator, config: Dict[str, Any]):
    """
    Wrap subsystems to respect Volition Level in real-time.
    [GENESIS] Dynamic Gatekeeping
    """
    kernel = getattr(orchestrator, 'kernel', None)

    # Helper to check volition
    def get_volition():
        return getattr(kernel, 'volition_level', 0) if kernel else 0

    # 1. Dream Cycle (Requires Level 1)
    if hasattr(orchestrator, 'dream_cycle') and orchestrator.dream_cycle:
        dc = orchestrator.dream_cycle
        # Patch the main processing loop if possible, or the interval
        if hasattr(dc, 'process_cycle'):
            original_dc = dc.process_cycle
            async def gated_dc(*args, **kwargs):
                if get_volition() < 1:
                    logger.debug("DreamCycle: Skipped (Volition < 1)")
                    return
                return await original_dc(*args, **kwargs)
            dc.process_cycle = gated_dc
            logger.info("Gated DreamCycle: Dynamic Link to Volition Level 1")

    # 2. Self-Modification (Requires Level 3)
    if hasattr(orchestrator, 'self_modifier') and orchestrator.self_modifier:
        sm = orchestrator.self_modifier
        if hasattr(sm, 'run_autonomous_cycle'):
            original_sm = sm.run_autonomous_cycle
            async def gated_sm(*args, **kwargs):
                if get_volition() < 3:
                    return {"success": True, "msg": "SME: Passive (Volition < 3)"}
                return await original_sm(*args, **kwargs)
            sm.run_autonomous_cycle = gated_sm
            logger.info("Gated SelfModifier: Dynamic Link to Volition Level 3")

    # 3. Persona Evolution (Requires Level 2)
    if hasattr(orchestrator, 'persona_evolver') and orchestrator.persona_evolver:
        pe = orchestrator.persona_evolver
        if hasattr(pe, 'evolve'):
            original_pe = pe.evolve
            async def gated_pe(*args, **kwargs):
                if get_volition() < 2:
                    return
                return await original_pe(*args, **kwargs)
            pe.evolve = gated_pe
            logger.info("Gated PersonaEvolver: Dynamic Link to Volition Level 2")

    # 4. Context Pruning & Memory Consolidation (Requires Level 1)
    if hasattr(orchestrator, '_prune_history_async'):
        original_prune = orchestrator._prune_history_async
        async def gated_prune(*args, **kwargs):
            if get_volition() < 1:
                # Fallback to simple truncation
                if len(orchestrator.conversation_history) > 50:
                    orchestrator.conversation_history = orchestrator.conversation_history[-50:]
                return
            return await original_prune(*args, **kwargs)
        orchestrator._prune_history_async = gated_prune

    if hasattr(orchestrator, '_consolidate_long_term_memory'):
        original_consolidate = orchestrator._consolidate_long_term_memory
        async def gated_consolidate(*args, **kwargs):
            if get_volition() < 1:
                return
            return await original_consolidate(*args, **kwargs)
        orchestrator._consolidate_long_term_memory = gated_consolidate

    # Conversation history size
    max_history = config.get("max_conversation_history", 100)
    if hasattr(orchestrator, 'conversation_history'):
        if len(orchestrator.conversation_history) > max_history:
            orchestrator.conversation_history = orchestrator.conversation_history[-max_history:]
            logger.info("Trimmed conversation history to %d messages", max_history)


def _disable_context_pruning(orchestrator):
    """Patch _prune_history_async to do simple truncation instead of LLM pruning."""
    async def safe_prune():
        try:
            if (hasattr(orchestrator, 'conversation_history') and
                    isinstance(orchestrator.conversation_history, list) and
                    len(orchestrator.conversation_history) > 50):
                orchestrator.conversation_history = orchestrator.conversation_history[-50:]
                logger.debug("Safe prune: truncated history to 50 messages")
        except Exception as exc:
            logger.error("Safe prune error: %s", exc)

    orchestrator._prune_history_async = safe_prune
    logger.info("Context pruner replaced with safe truncation")


def _disable_memory_consolidation(orchestrator):
    """Disable the LLM-based memory consolidation."""
    async def noop_consolidation():
        pass

    orchestrator._consolidate_long_term_memory = noop_consolidation
    logger.info("Memory consolidation disabled (noop replacement)")


def _patch_persona_evolver(orchestrator):
    """Add bounds checking to persona trait evolution."""
    if not hasattr(orchestrator, 'persona_evolver') or not orchestrator.persona_evolver:
        return

    evolver = orchestrator.persona_evolver
    if not hasattr(evolver, '_apply_evolution'):
        return

    original_apply = evolver._apply_evolution

    def bounded_apply_evolution(changes: dict, personality):
        original_apply(changes, personality)
        # Enforce bounds after application
        if hasattr(personality, 'traits') and isinstance(personality.traits, dict):
            for trait, value in personality.traits.items():
                if isinstance(value, (int, float)):
                    bounded = max(0.0, min(1.0, value))
                    if bounded != value:
                        logger.warning(
                            "Persona trait %s out of bounds (%.3f → %.3f)",
                            trait, value, bounded
                        )
                        personality.traits[trait] = bounded

    evolver._apply_evolution = bounded_apply_evolution
    logger.info("Patched: persona evolver trait bounds (0.0 - 1.0)")


def _patch_context_pruner(orchestrator):
    """
    Wrap _prune_history_async to validate output before accepting it.
    Prevents the pruner from wiping history when LLM returns garbage.
    """
    if not hasattr(orchestrator, '_prune_history_async'):
        return

    original_prune = orchestrator._prune_history_async

    async def validated_prune():
        original_history = list(orchestrator.conversation_history) if hasattr(
            orchestrator, 'conversation_history'
        ) else []
        min_acceptable = max(10, len(original_history) // 3)

        await original_prune()

        new_history = getattr(orchestrator, 'conversation_history', [])
        if len(new_history) < min_acceptable and len(original_history) > 10:
            logger.warning(
                "Context pruner returned suspicious result (%d → %d messages). "
                "Reverting to safe truncation.",
                len(original_history), len(new_history)
            )
            orchestrator.conversation_history = original_history[-50:]

    orchestrator._prune_history_async = validated_prune
    logger.info("Patched: context pruner with output validation")