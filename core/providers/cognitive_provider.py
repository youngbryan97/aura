"""core/providers/cognitive_provider.py — Cognitive, LLM, Learning & Reasoning Registration
"""

import logging
import subprocess
from core.container import ServiceLifetime

logger = logging.getLogger("Aura.Providers.Cognitive")

def register_cognitive_services(container, is_proxy: bool = False):
    # 1. Cognitive Engine
    def create_cognitive_engine():
        if is_proxy:
            logger.info("📡 Proxy Mode: Skipping full CognitiveEngine.")
            return None
        try:
            from core.brain.cognitive_engine import CognitiveEngine
            return CognitiveEngine()
        except Exception:
            logger.exception("Failed to create cognitive_engine")
            return None
    container.register('cognitive_engine', create_cognitive_engine, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 1.1 Cognitive Manager
    def create_cognitive_manager():
        if is_proxy:
            return None
        try:
            from core.brain.cognitive_manager import CognitiveManager
            return CognitiveManager()
        except Exception:
            logger.exception("Failed to create cognitive_manager")
            return None
    container.register('cognitive_manager', create_cognitive_manager, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 1.2 LLM Router
    def create_llm_router():
        if is_proxy:
            logger.info("📡 Proxy Mode: Skipping LLM Router (Prevents MLX Load).")
            # Return a mock router that doesn't trigger heavy dependencies
            class MockRouter:
                async def think(self, *args, **kwargs):
                    return "LLM unavailable in GUI Proxy Mode"
                def route(self, *args, **kwargs):
                    return self
            return MockRouter()
        
        try:
            from core.brain.llm_health_router import HealthAwareLLMRouter
            router = HealthAwareLLMRouter()
            
            # Phase 15: Populating the Router
            # The AutonomousCognitiveEngine is responsible for Tier discovery
            from core.brain.llm.autonomous_brain_integration import AutonomousCognitiveEngine
            from core.capability_engine import CapabilityEngine
            from core.event_bus import get_event_bus
            
            # We don't need to store the engine instance here, its __init__ handles registration
            AutonomousCognitiveEngine(
                registry=CapabilityEngine(), # Temporary registry for init
                llm_router=router,
                event_bus=get_event_bus()
            )
            
            return router
        except Exception as e:
            logger.error("🛑 LLMRouter Initialization Critical Failure: %s", e)
            # Do NOT return None; let the container know it failed so it can be re-attempted
            # or handled by the lazy-fetching LanguageCenter.
            raise
    container.register('llm_router', create_llm_router, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 1.2.1 Autonomous Brain (for LLM Router population)
    def create_autonomous_brain():
        if is_proxy:
            logger.info("📡 Proxy Mode: Skipping AutonomousCognitiveEngine.")
            return None
        try:
            from core.brain.llm.autonomous_brain_integration import AutonomousCognitiveEngine
            from core.capability_engine import CapabilityEngine
            from core.event_bus import get_event_bus
            # The actual instance is created within create_llm_router, this is just for registration
            # and to allow other services to depend on it if needed.
            # We return None here because the instance is managed by the router's creation.
            return None
        except Exception:
            logger.exception("Failed to create autonomous_brain")
            return None
    container.register('autonomous_brain', create_autonomous_brain, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 1.3 LLM Interfaces
    def create_router_interface():
        from core.brain.llm.router_interface import RouterLLMInterface
        router = container.get("llm_router")
        return RouterLLMInterface(router)
    container.register('llm_interface', create_router_interface, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_deliberator():
        from core.brain.deliberation import DeliberationController
        llm = container.get("llm_interface")
        return DeliberationController(llm)
    container.register('deliberator', create_deliberator, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 4. Capability Engine (Unified)
    def create_capability_engine():
        from core.capability_engine import CapabilityEngine
        orch = container.get("orchestrator", None)
        return CapabilityEngine(orchestrator=orch)
    container.register('capability_engine', create_capability_engine, lifetime=ServiceLifetime.SINGLETON, required=True)
    container.register('skill_registry', lambda: container.get("capability_engine"), lifetime=ServiceLifetime.SINGLETON, required=False)

    # 4.5 Personality Engine
    def create_personality_engine():
        from core.brain.personality_engine import PersonalityEngine
        return PersonalityEngine()
    container.register('personality_engine', create_personality_engine, lifetime=ServiceLifetime.SINGLETON, required=True)

    # Phase 7: Cognitive Inversion (The Brain-LLM Split)
    def create_api_adapter():
        from core.api_adapter import get_api_adapter
        return get_api_adapter()
    container.register('api_adapter', create_api_adapter, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_cognitive_kernel():
        from core.cognitive_kernel import get_cognitive_kernel
        return get_cognitive_kernel()
    container.register('cognitive_kernel', create_cognitive_kernel, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_inner_monologue():
        from core.inner_monologue import get_inner_monologue
        return get_inner_monologue()
    container.register('inner_monologue', create_inner_monologue, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_language_center():
        from core.language_center import get_language_center
        return get_language_center()
    container.register('language_center', create_language_center, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_memory_synthesizer():
        from core.memory_synthesizer import get_memory_synthesizer
        return get_memory_synthesizer()
    container.register('memory_synthesizer', create_memory_synthesizer, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_cognitive_integration():
        from core.cognitive_integration_layer import CognitiveIntegrationLayer
        return CognitiveIntegrationLayer()
    container.register('cognitive_integration', create_cognitive_integration, lifetime=ServiceLifetime.SINGLETON, required=True)

    # Phase 7 Extension: Epistemic Tracking & Inquiry Loop
    def create_epistemic_tracker():
        from core.epistemic_tracker import get_epistemic_tracker
        return get_epistemic_tracker()
    container.register('epistemic_tracker', create_epistemic_tracker, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_inquiry_engine():
        from core.inquiry_engine import get_inquiry_engine
        return get_inquiry_engine()
    container.register('inquiry_engine', create_inquiry_engine, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_concept_linker():
        from core.concept_linker import ConceptLinker
        return ConceptLinker()
    container.register('concept_linker', create_concept_linker, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_belief_challenger():
        from core.belief_challenger import BeliefChallenger
        return BeliefChallenger()
    container.register('belief_challenger', create_belief_challenger, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_insight_journal():
        from core.insight_journal import InsightJournal
        return InsightJournal()
    container.register('insight_journal', create_insight_journal, lifetime=ServiceLifetime.SINGLETON, required=True)

    # Phase 9: Self-Architect & Recursive Mastery
    def create_code_refiner():
        from core.code_refiner import CodeRefinerService
        return CodeRefinerService()
    container.register('code_refiner', create_code_refiner, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_skill_evolution():
        from core.skill_evolution import SkillEvolutionEngine
        return SkillEvolutionEngine()
    container.register('skill_evolution', create_skill_evolution, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_system_monitor():
        from core.system_monitor import SystemStateMonitor
        return SystemStateMonitor()
    container.register('system_monitor', create_system_monitor, lifetime=ServiceLifetime.SINGLETON, required=True)

    # Phase 10: Full-Mind Integration (Zenith)
    def create_narrative_thread():
        from core.narrative_thread import NarrativeThread
        return NarrativeThread()
    container.register('narrative_thread', create_narrative_thread, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_emotional_coloring():
        from core.emotional_coloring import EmotionalColoring
        return EmotionalColoring()
    container.register('emotional_coloring', create_emotional_coloring, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_plasticity_controller():
        from core.plasticity_controller import PlasticityController
        return PlasticityController()
    container.register('plasticity_controller', create_plasticity_controller, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_conversation_reflector():
        from core.conversation_reflector import ConversationReflector
        return ConversationReflector()
    container.register('conversation_reflector', create_conversation_reflector, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 13. Orchestrator
    if not container.has('orchestrator'):
        def orchestrator_factory():
            from core.orchestrator import create_orchestrator
            return create_orchestrator()
        container.register('orchestrator', orchestrator_factory, lifetime=ServiceLifetime.SINGLETON, required=True)
    else:
        logger.debug("Orchestrator already registered, skipping provider registration.")

    # 38.2 Cognitive Context Manager
    def create_context_manager():
        from core.brain.cognitive_context_manager import CognitiveContextManager
        orch = container.get("orchestrator", None)
        return CognitiveContextManager(orch)
    container.register('context_manager', create_context_manager, lifetime=ServiceLifetime.SINGLETON, required=True)
    container.register('cognition', lambda: container.get("context_manager"), lifetime=ServiceLifetime.SINGLETON, required=False)

    # Nucleus (MLX) - Apple Silicon Only
    def _is_apple_silicon():
        import platform
        if platform.system() != "Darwin":
            return False

        machine = (platform.machine() or "").lower()
        if machine in {"arm64", "aarch64"}:
            return True

        try:
            res = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64"],
                capture_output=True,
                text=True,
                check=False,
            )
            return res.stdout.strip() == "1"
        except Exception:
            return False

    if _is_apple_silicon():
        def create_nucleus():
            try:
                from core.brain.llm.nucleus_manager import NucleusManager
                return NucleusManager()
            except Exception:
                logger.debug("Nucleus manager unavailable")
                return None
        container.register('nucleus', create_nucleus, lifetime=ServiceLifetime.SINGLETON, required=False)

    # Phase 13: Continuity
    def create_continuity():
        from core.continuity import get_continuity
        return get_continuity()
    container.register('continuity', create_continuity, lifetime=ServiceLifetime.SINGLETON, required=True)

    # Phase 13: Goal Beliefs
    def create_goal_manager():
        from core.world_model.goal_beliefs import GoalBeliefManager
        belief_system = container.get("belief_graph", default=None)
        if not belief_system:
            # Fallback for late init
            from core.world_model.belief_graph import get_belief_graph
            belief_system = get_belief_graph()
        return GoalBeliefManager(belief_system)
    container.register('goal_belief_manager', create_goal_manager, lifetime=ServiceLifetime.SINGLETON, required=True)

    # Phase 13: The Bryan Model
    def create_bryan_model():
        from core.world_model.user_model import BryanModelEngine
        return BryanModelEngine()
    container.register('bryan_model', create_bryan_model, lifetime=ServiceLifetime.SINGLETON, required=True)

    # --- New Cognitive Engines (Phase XXII) ---

    # Continuous Learner
    def create_continuous_learner():
        try:
            from core.learning.genuine_learning_pipeline import register_continuous_learner
            return register_continuous_learner()
        except Exception:
            logger.exception("Failed to create continuous_learner")
            return None
    container.register('continuous_learner', create_continuous_learner, lifetime=ServiceLifetime.SINGLETON, required=False)

    # ReAct Loop
    def create_react_loop():
        try:
            from core.brain.react_loop import ReActLoop
            engine = container.get("cognitive_engine")
            orch = container.get("orchestrator", default=None)
            return ReActLoop(brain=engine, orchestrator=orch)
        except Exception:
            logger.exception("Failed to create react_loop")
            return None
    container.register('react_loop', create_react_loop, lifetime=ServiceLifetime.SINGLETON, required=False)

    # Personality Bridge (Systemic Influence)
    def create_personality_bridge():
        try:
            from core.brain.personality_bridge import PersonalityBridge
            return PersonalityBridge()
        except Exception:
            logger.exception("Failed to create personality_bridge")
            return None
    container.register('personality_bridge', create_personality_bridge, lifetime=ServiceLifetime.SINGLETON, required=False)

    # Phase 25: Critic Engine
    def create_critic_engine():
        try:
            from core.critic_engine import get_critic_engine
            return get_critic_engine()
        except Exception:
            logger.exception("Failed to create critic_engine")
            return None
    container.register('critic_engine', create_critic_engine, lifetime=ServiceLifetime.SINGLETON, required=True)

    # Phase 29: Agent Swarm Delegator
    def create_agent_delegator():
        try:
            from core.collective.delegator import AgentDelegator
            orch = container.get("orchestrator", default=None)
            return AgentDelegator(orchestrator=orch)
        except Exception:
            logger.exception("Failed to create agent_delegator")
            return None
    container.register('agent_delegator', create_agent_delegator, lifetime=ServiceLifetime.SINGLETON, required=True)

    # Paraconsistent Logic Core (holds contradictory beliefs without crashing)
    def create_paraconsistent_engine():
        try:
            from core.cognition.paraconsistent_logic import ParaconsistentEngine
            return ParaconsistentEngine()
        except Exception:
            logger.exception("Failed to create paraconsistent_engine")
            return None
    container.register('paraconsistent_engine', create_paraconsistent_engine, lifetime=ServiceLifetime.SINGLETON, required=False)
