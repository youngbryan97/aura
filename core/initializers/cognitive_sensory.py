"""Cognitive sensory layer initialization.

Registers sensory subsystems (vision, hearing, continuous perception)
into the service container for use by the cognitive pipeline.
"""
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging

logger = logging.getLogger("Aura.Init.CognitiveSensory")


def init_cognitive_sensory_layer(container):
    """Register sensory subsystems in the service container."""
    try:
        from core.sensory_integration import get_sensory_system
        sensory = get_sensory_system()
        if sensory is not None:
            container.register_instance("sensory_system", sensory, required=False)
            logger.debug("Sensory system registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('cognitive_sensory', e)
        logger.debug("Sensory system init deferred: %s", e)

    try:
        from core.senses.continuous_vision import ContinuousSensoryBuffer
        from core.config import config
        buffer = ContinuousSensoryBuffer(data_dir=getattr(config.paths, "data_dir", None))
        container.register_instance("continuous_vision", buffer, required=False)
        logger.debug("Continuous vision buffer registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('cognitive_sensory', e)
        logger.debug("Continuous vision init deferred: %s", e)

    # Research-inspired memory systems
    try:
        from core.memory.conceptual_gravitation import get_gravitation_engine
        grav = get_gravitation_engine()
        container.register_instance("conceptual_gravitation", grav, required=False)
        logger.debug("Conceptual gravitation engine registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('cognitive_sensory', e)
        logger.debug("Conceptual gravitation init deferred: %s", e)

    try:
        from core.memory.knowledge_compression import get_knowledge_compressor
        compressor = get_knowledge_compressor()
        container.register_instance("knowledge_compressor", compressor, required=False)
        logger.debug("Knowledge compressor registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('cognitive_sensory', e)
        logger.debug("Knowledge compressor init deferred: %s", e)

    try:
        from core.memory.navigating_graph import get_navigating_graph
        nsg = get_navigating_graph()
        container.register_instance("navigating_graph", nsg, required=False)
        logger.debug("Navigating graph registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('cognitive_sensory', e)
        logger.debug("Navigating graph init deferred: %s", e)

    try:
        from core.consciousness.stdp_learning import get_stdp_engine
        stdp = get_stdp_engine()
        container.register_instance("stdp_engine", stdp, required=False)
        logger.debug("STDP learning engine registered (with MESU plasticity).")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('cognitive_sensory', e)
        logger.debug("STDP engine init deferred: %s", e)

    # ── Phase 3/4: Research-grade cognitive modules ──
    # All non-required — boot never fails on these.

    try:
        from core.meta.metacognitive_monitor import MetaCognitiveMonitor
        metacog = MetaCognitiveMonitor()
        container.register_instance("metacognitive_monitor", metacog, required=False)
        logger.debug("MetaCognitive monitor registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('cognitive_sensory', e)
        logger.debug("MetaCognitive monitor init deferred: %s", e)

    try:
        from core.adaptation.intrinsic_motivation import IntrinsicMotivationEngine
        im_engine = IntrinsicMotivationEngine()
        container.register_instance("intrinsic_motivation", im_engine, required=False)
        logger.debug("Intrinsic motivation engine registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('cognitive_sensory', e)
        logger.debug("Intrinsic motivation init deferred: %s", e)

    try:
        from core.meta.experience_distillery import ExperienceDistillery
        distillery = ExperienceDistillery()
        container.register_instance("experience_distillery", distillery, required=False)
        logger.debug("Experience distillery registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('cognitive_sensory', e)
        logger.debug("Experience distillery init deferred: %s", e)

    try:
        from core.adaptation.plasticity_governor import PlasticityGovernor
        gov = PlasticityGovernor()
        container.register_instance("plasticity_governor", gov, required=False)
        logger.debug("Plasticity governor (EWC) registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('cognitive_sensory', e)
        logger.debug("Plasticity governor init deferred: %s", e)

    # Code graph — Aura's self-knowledge of her own codebase
    try:
        from core.introspection.code_graph import get_code_graph
        import asyncio
        graph = get_code_graph()
        container.register_instance("code_graph", graph, required=False)
        # Build incrementally in background (don't block boot)
        get_task_tracker().create_task(_build_code_graph_background(graph))
        logger.info("Code graph registered (building in background).")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('cognitive_sensory', e)
        logger.debug("Code graph init deferred: %s", e)


async def _build_code_graph_background(graph):
    """Build the code graph without blocking boot."""
    import logging
    _logger = logging.getLogger("Aura.Init.CodeGraph")
    try:
        await asyncio.sleep(10)  # Let critical subsystems boot first
        stats = await graph.build(incremental=True)
        _logger.info("Code graph ready: %d files, %d symbols, %d relationships (%.1fs)",
                     stats["files"], stats["symbols"], stats["relationships"], stats.get("build_time_s", 0))
    except (OSError, ConnectionError, TimeoutError) as e:
        record_degradation('cognitive_sensory', e)
        _logger.warning("Code graph background build failed: %s", e)
