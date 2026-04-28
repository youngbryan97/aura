"""core/presence_integration.py
───────────────────────────
The "Presence Patch" that wires the v30 components into the Orchestrator.
"""

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import logging
from core.container import ServiceContainer

logger = logging.getLogger("Aura.PresenceIntegration")

def apply_presence_patch(orchestrator):
    """
    Wires OpinionEngine, ProactivePresence, and social/discourse systems
    into a running RobustOrchestrator.
    Called during _init_proactive_systems in boot.py.
    """
    logger.info("🔧 [PresencePatch] Applying Phase 30 communication hierarchy...")

    # 1. Initialize & Register OpinionEngine
    from core.opinion_engine import OpinionEngine
    opinion_engine = OpinionEngine(orchestrator)
    ServiceContainer.register_instance("opinion_engine", opinion_engine)
    orchestrator.opinion_engine = opinion_engine
    logger.info("✅ OpinionEngine registered.")

    # 2. Initialize & Register ProactivePresence
    from core.proactive_presence import ProactivePresence
    presence = ProactivePresence(orchestrator)
    ServiceContainer.register_instance("proactive_presence", presence)
    orchestrator.proactive_presence = presence
    logger.info("✅ ProactivePresence registered.")

    # 3. Start ProactivePresence background task
    import asyncio
    get_task_tracker().create_task(presence.start())
    logger.info("🚀 ProactivePresence loop started.")

    # 4. Hook VAD to prevent interruption
    try:
        from core.senses.voice_engine import get_voice_engine
        voice_engine = get_voice_engine()
        voice_engine._on_vad_change = presence.mark_user_speaking
        logger.info("🎤 VAD pinned to ProactivePresence.")
    except Exception as e:
        record_degradation('presence_integration', e)
        logger.warning("Failed to hook VAD: %s", e)

    # 5. Shared Ground Buffer (inside jokes, established references, running callbacks)
    try:
        from core.memory.shared_ground import get_shared_ground
        sg = get_shared_ground()
        ServiceContainer.register_instance("shared_ground", sg)
        logger.info("✅ SharedGroundBuffer registered (%d entries).", len(sg.entries))
    except Exception as e:
        record_degradation('presence_integration', e)
        logger.warning("SharedGroundBuffer init failed: %s", e)

    # 6. SocialMemory (relationship depth & milestones)
    try:
        # Only register if not already present (another boot path may have registered it)
        if not ServiceContainer.get("social_memory", default=None):
            from core.memory.social_memory import SocialMemory
            social_mem = SocialMemory()
            ServiceContainer.register_instance("social_memory", social_mem)
            logger.info("✅ SocialMemory registered.")
    except Exception as e:
        record_degradation('presence_integration', e)
        logger.warning("SocialMemory init failed: %s", e)

    # 7. TheoryOfMind (user model: rapport, trust, emotional state)
    try:
        if not ServiceContainer.get("theory_of_mind", default=None):
            from core.consciousness.theory_of_mind import get_theory_of_mind
            ce = getattr(orchestrator, "cognitive_engine", None)
            tom = get_theory_of_mind(ce)
            ServiceContainer.register_instance("theory_of_mind", tom)
            logger.info("✅ TheoryOfMind registered.")
    except Exception as e:
        record_degradation('presence_integration', e)
        logger.warning("TheoryOfMind init failed: %s", e)

    # 8. DiscourseTracker (topic threading, user emotional trend, conversation energy)
    try:
        from core.brain.discourse_tracker import DiscourseTracker
        ce = getattr(orchestrator, "cognitive_engine", None)
        discourse_tracker = DiscourseTracker(ce)
        ServiceContainer.register_instance("discourse_tracker", discourse_tracker)
        logger.info("✅ DiscourseTracker registered.")
    except Exception as e:
        record_degradation('presence_integration', e)
        logger.warning("DiscourseTracker init failed: %s", e)

    return True
