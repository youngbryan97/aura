"""core/personality_integration.py - Personality Systems Integration
Bridge for enforcing Aura's technical personality across all modules.
"""
from core.runtime.errors import record_degradation
import logging
import time
from typing import Any, Dict

logger = logging.getLogger("Aura.Integration")

def integrate_all_personality_systems(orchestrator):
    """Integrates identity parameters across all system components.
    """
    logger.info("-" * 40)
    logger.info("Initiating personality integration")
    logger.info("-" * 40)

    try:
        from .personality_engine import get_personality_engine
        engine = get_personality_engine()
        
        # 1. Patch Proactive Communication
        if hasattr(orchestrator, 'proactive_comm'):
            _patch_proactive_comm(orchestrator.proactive_comm, engine)

        # 2. Patch Orchestrator Response Expresser
        _patch_orchestrator(orchestrator, engine)

        logger.info("Identity core integrated - v3.5.5")
        return True
    except Exception as e:
        record_degradation('personality_integration', e)
        logger.error("Personality integration failed: %s", e)
        return False

def _patch_orchestrator(orchestrator, engine):
    """Add a final filter to all outgoing messages to catch 'Assistant' leaks."""
    if hasattr(orchestrator, 'reply_queue'):
        original_put = orchestrator.reply_queue.put_nowait
        def filtered_put(item):
            if isinstance(item, str):
                item = engine.filter_response(item)
            elif isinstance(item, dict) and 'message' in item:
                item['message'] = engine.filter_response(item['message'])
            return original_put(item)
        orchestrator.reply_queue.put_nowait = filtered_put
        logger.info("Orchestrator reply queue filter active")
    logger.info("Orchestrator output integrity filter active")

def _patch_proactive_comm(comm, engine):
    """Ensure autonomous messages are also filtered."""
    if hasattr(comm, 'queue_message'):
        original_queue = comm.queue_message
        
        def filtered_queue(content, emotion, urgency, context=None):
            content = engine.filter_response(content)
            return original_queue(content, emotion, urgency, context)
            
        comm.queue_message = filtered_queue
        logger.info("Proactive communication aligned with personality core")

def verify_all_systems_aligned(orchestrator) -> bool:
    """Check if the personality core is active and verified."""
    from .personality_kernel import get_kernel
    kernel = get_kernel()
    return kernel.version == "3.5.5"
