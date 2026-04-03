from __future__ import annotations
import logging
from typing import Any, Optional
from .orchestrator_types import SystemStatus

logger = logging.getLogger(__name__)

class OrchestratorServicesMixin:
    """Mixin for service resolution and property getters."""

    def _get_service(self, name: str, alias: Optional[str] = None) -> Any:
        """Standardized resolver for orchestrator dependencies.
        
        v50: Fail-soft stabilization.
        """
        from core.container import ServiceNotFoundError, get_container
        container = get_container()
        
        # Check for overrides first (v14.2)
        override_attr = f"_{name}_override"
        if hasattr(self, override_attr):
            return getattr(self, override_attr)
            
        try:
            val = container.get(name, default=None)
            if val is not None:
                return val
        except Exception as e:
            logger.debug("Primary service lookup failed for '%s': %s", name, e)
            
        if alias:
            try:
                val = container.get(alias, default=None)
                if val is not None:
                    return val
            except Exception as e:
                logger.debug("Alias lookup failed for '%s' (alias of %s): %s", alias, name, e)
        
        # v50: Fallback to existing attribute if set via earlier registration
        cached_attr = f"_{name.replace(' ', '_')}"
        if hasattr(self, cached_attr):
            return getattr(self, cached_attr)
            
        # Patch 9: Fail-soft during stabilization
        logger.debug("Service '%s' (alias: %s) not found. Returning None.", name, alias)
        return None

    @property
    def cognitive_engine(self): return self._get_service("cognitive_engine")
    @cognitive_engine.setter
    def cognitive_engine(self, value): setattr(self, "_cognitive_engine_override", value)

    @property
    def liquid_state(self): 
        """Resolved via 'liquid_state' or 'affect_engine' alias."""
        return self._get_service("liquid_state", "affect_engine")
    @liquid_state.setter
    def liquid_state(self, value): setattr(self, "_liquid_state_override", value)


    @property
    def personality_engine(self): return self._get_service("personality_engine")
    @personality_engine.setter
    def personality_engine(self, value): setattr(self, "_personality_engine_override", value)

    @property
    def memory(self): return self._get_service("memory_facade")
    @memory.setter
    def memory(self, value): setattr(self, "_memory_facade_override", value)

    @property
    def capability_engine(self): 
        """Resolved via 'capability_engine' or 'skill_registry' alias."""
        return self._get_service("capability_engine", "skill_registry")
    @capability_engine.setter
    def capability_engine(self, value): setattr(self, "_capability_engine_override", value)

    @property
    def strategic_planner(self): return self._get_service("strategic_planner")
    @strategic_planner.setter
    def strategic_planner(self, value): setattr(self, "_strategic_planner_override", value)

    @property
    def intent_router(self): return self._get_service("cognitive_router", "intent_router")
    @intent_router.setter
    def intent_router(self, value): setattr(self, "_cognitive_router_override", value)

    @property
    def identity(self): return self._get_service("self_model", "identity")
    @identity.setter
    def identity(self, value): setattr(self, "_self_model_override", value)

    @property
    def mycelium(self): return self._get_service("mycelium", "mycelial_network")
    @mycelium.setter
    def mycelium(self, value): setattr(self, "_mycelium_override", value)

    @property
    def metabolic_monitor(self): return self._get_service("metabolic_monitor")
    @metabolic_monitor.setter
    def metabolic_monitor(self, value): setattr(self, "_metabolic_monitor_override", value)

    @property
    def metabolic_coordinator(self): return self._get_service("metabolic_coordinator")
    @metabolic_coordinator.setter
    def metabolic_coordinator(self, value): setattr(self, "_metabolic_coordinator_override", value)

    @property
    def consciousness(self): return self._get_service("liquid_state", "conscious_substrate")
    @consciousness.setter
    def consciousness(self, value): setattr(self, "_conscious_substrate_override", value)

    @property
    def curiosity(self): return self._get_service("curiosity_engine")
    @curiosity.setter
    def curiosity(self, value): setattr(self, "_curiosity_engine_override", value)

    @property
    def motivation(self): return self._get_service("motivation_engine")
    @motivation.setter
    def motivation(self, value): setattr(self, "_motivation_engine_override", value)

    @property
    def knowledge_graph(self): return self._get_service("knowledge_graph")
    @knowledge_graph.setter
    def knowledge_graph(self, value): setattr(self, "_knowledge_graph_override", value)

    @property
    def goal_hierarchy(self): return self._get_service("goal_hierarchy")
    @goal_hierarchy.setter
    def goal_hierarchy(self, value): setattr(self, "_goal_hierarchy_override", value)
    
    @property
    def alignment(self): return self._get_service("alignment", "alignment_engine")
    @alignment.setter
    def alignment(self, value): setattr(self, "_alignment_engine_override", value)

    @property
    def watchdog(self): return self._get_service("watchdog")
    @watchdog.setter
    def watchdog(self, value): setattr(self, "_watchdog_override", value)

    @property
    def lnn(self): return self._get_service("lnn", "liquid_substrate")
    @lnn.setter
    def lnn(self, value): setattr(self, "_lnn_override", value)
    
    @property
    def sovereign_swarm(self):
        """Resolved via AgencyCore's swarm instance (Internal Thinking Shards)."""
        core = self._get_service("agency_core")
        return getattr(core, "swarm", None) if core else None
    @sovereign_swarm.setter
    def sovereign_swarm(self, value): setattr(self, "_agency_core_override", value)
    
    @property
    def affect(self): return self._get_service("affect_engine")
    @affect.setter
    def affect(self, value): setattr(self, "_affect_engine_override", value)

    @property
    def state_machine(self): return self._get_service("state_machine")
    @state_machine.setter
    def state_machine(self, value): setattr(self, "_state_machine_override", value)
    
    @property
    def project_store(self): return self._get_service("project_store")
    @project_store.setter
    def project_store(self, value): setattr(self, "_project_store_override", value)

    @property
    def self_modifier(self): return self._get_service("self_modification_engine")
    @self_modifier.setter
    def self_modifier(self, value): setattr(self, "_self_modification_engine_override", value)

    @property
    def virtual_body(self): return self._get_service("virtual_body")
    @virtual_body.setter
    def virtual_body(self, value): setattr(self, "_virtual_body_override", value)

    @property
    def meta_learning(self): return self._get_service("meta_learning")
    @meta_learning.setter
    def meta_learning(self, value): setattr(self, "_meta_learning_override", value)

    @property
    def drives(self): return self._get_service("drive_engine", "drives")
    @drives.setter
    def drives(self, value): setattr(self, "_drive_engine_override", value)

    @property
    def hierarchical_memory(self): return self._get_service("hierarchical_memory_orchestrator")
    @hierarchical_memory.setter
    def hierarchical_memory(self, value): setattr(self, "_hierarchical_memory_override", value)

    @property
    def meta_cognition(self): return self._get_service("meta_cognition_shard")
    @meta_cognition.setter
    def meta_cognition(self, value): setattr(self, "_meta_cognition_override", value)

    @property
    def healing_swarm(self): return self._get_service("healing_swarm")
    @healing_swarm.setter
    def healing_swarm(self, value): setattr(self, "_healing_swarm_override", value)
    
    @property
    def output_gate(self):
        val = getattr(self, "_output_gate_override", None)
        if val is not None:
            return val
        from core.utils.output_gate import get_output_gate
        gate = get_output_gate(self)
        setattr(self, "_output_gate_override", gate)
        return gate
    @output_gate.setter
    def output_gate(self, value): setattr(self, "_output_gate_override", value)

    # Dynamic Alias Mapping (v11.0 Clean Room)
    def __getattr__(self, name: str) -> Any:
        """Dynamic service resolution for remaining aliases."""
        aliases = {
            # Memory
            "memory_facade": "memory_facade",
            "memory_manager": "memory_facade",
            "memory_optimizer": "memory_manager",
            "persistent_state": "persistent_state",
            # Cognition
            "cognitive_manager": "cognitive_engine",
            "personality_manager": "personality_engine",
            "personality_engine": "personality_engine",
            "cognition": "cognitive_engine",
            "context_manager": "context_manager",
            "metacognition": "meta_cognition_shard",
            "meta_learning": "metacognition",
            "meta_evolution": "meta_cognition_shard",
            "meta_cognition": "meta_cognition_shard",
            "healing_swarm": "healing_swarm",
            "healing_service": "healing_swarm",
            "self_model": "self_model",
            "identity": "self_model",
            "identity_kernel": "self_model",
            "drives": "drive_engine",
            "drive_engine": "drive_engine",
            "hierarchical_memory": "hierarchical_memory_orchestrator",
            "hierarchical_memory_orchestrator": "hierarchical_memory_orchestrator",
            "memory_compressor": "hierarchical_memory_orchestrator",
            # Capability
            "router": "capability_engine",
            "skill_manager": "capability_engine",
            "intent_router": "cognitive_router",
            # Embodiment/World
            "liquid_state": "conscious_substrate",
            "conscious_substrate": "conscious_substrate",
            "consciousness_core": "consciousness_core",
            "embodied": "agency_core",
            "world_model": "belief_graph",
            "world_state": "belief_graph",
            "belief_graph": "belief_graph",
            # Network
            "mycelial_network": "mycelium",
            "pulse_manager": "pulse_manager",
            "proactive_comm": "proactive_communication",
            "proactive_communication": "proactive_communication",
            "proactive_comm": "proactive_communication",
            "singularity_monitor": "singularity_monitor",
            "world_state": "belief_graph",
            "self_healer": "self_healer",
            "memory_optimizer": "memory_facade",
            "immune_system": "immune_system",
            "predictive_engine": "predictive_engine",
            # Security/Monitor
            "alignment": "alignment_engine",
            "alignment_engine": "alignment_engine",
            "constitutional_guard": "alignment_engine",
            "autonomy_guardian": "autonomy_guardian",
            "watchdog": "watchdog",
            "_watchdog": "watchdog",
            "vision": "sovereign_eyes",
            "ears": "sovereign_ears",
            "audio": "sovereign_ears",
            "llm": "llm_router",
            "brain": "cognitive_engine",
            "embodied": "agency_core",
            "agency_core": "agency_core",
            "agency": "agency_core",
            "memory": "knowledge_graph",
            "state": "state_repository",
            # Sensory
            "ears": "ears",
            "voice": "voice_engine",
            "voice_engine": "voice_engine",
            "mouth": "tts_stream",
            "vision": "vision_engine",
            "vision_engine": "vision_engine",
            "scanner": "integrity_guard",
            "sovereign_scanner": "integrity_guard",
            "integrity_guard": "integrity_guard",
            # Cognitive Core
            "cognitive_engine": "cognitive_engine",
            "knowledge_graph": "knowledge_graph",
            "self_modification_engine": "self_modification_engine",
            "self_modifier": "self_modification_engine",
            "sme": "self_modification_engine",
            "hephaestus": "hephaestus_engine",
            "hephaestus_engine": "hephaestus_engine",
            "curiosity_engine": "curiosity_engine",
            "curiosity": "curiosity_engine",
            "motivation_engine": "motivation_engine",
            "motivation": "motivation_engine",
            "affect_engine": "affect_engine",
            "affect": "affect_engine",
            "affect_manager": "affect_engine",
            "llm": "cerebellum",
            "cerebellum": "cerebellum",
            "nucleus": "nucleus",
            "curiosity": "curiosity_engine",
            "hotfix_engine": "hotfix_engine",
            "metabolism": "metabolic_coordinator",
            "metabolic_monitor": "metabolic_coordinator",
            "metabolic_coordinator": "metabolic_coordinator",
            "consciousness": "consciousness",
            "lazarus": "lazarus",
            "persona_evolver": "persona_evolver",
            "belief_sync": "belief_sync",
            "momentum": "conversational_momentum_engine",
            "substrate": "liquid_substrate",
            "liquid_substrate": "liquid_substrate",
            "lnn": "liquid_substrate",
            "liquid_neural_network": "liquid_substrate",
            # State Management & Handlers
            "state_machine": "state_machine",
            "classifier": "cognitive_router",
            "cognitive_router": "cognitive_router",
            # Internal Subs
            "autonomic_core": "autonomic_core",
            "subsystem_audit": "subsystem_audit",
            "degradation_manager": "degradation_manager",
            "health_monitor": "health_monitor",
            "drive_controller": "drive_engine",
            "memory_governor": "memory_governor",
            "permission_guard": "permission_guard",
            # Memory & Data
            "episodic": "episodic_memory",
            "semantic": "semantic_memory",
            "vector_memory": "vector_memory",
            "knowledge_ledger": "knowledge_ledger",
            "ledger": "knowledge_ledger",
            "project_store": "project_store",
            # Advanced Consciousness
            "global_workspace": "global_workspace",
            "conscious_core": "consciousness_core",
            "qualia_synthesizer": "qualia_synthesizer",
            "affect_engine": "affect_engine",
            "drive_engine": "drive_engine",
            "self_model": "self_model",
            "embodiment": "embodiment",
            "soma": "soma",
            "mind_model": "mind_model",
            "theory_of_mind": "theory_of_mind",
            "moral_reasoning": "moral_reasoning",
            "homeostasis": "homeostasis",
            # External/Swarm
            "swarm": "swarm_protocol",
            "agent_delegator": "agent_delegator",
            "delegator": "agent_delegator",
            "web_search": "web_search",
            "sandbox": "sandbox",
            # Drives & Motivation
            "drives": "drive_engine",
            "drive_engine": "drive_engine",
            # Resilience
            "snapshot_manager": "snapshot_manager",
            "hotfix_engine": "hotfix_engine",
            "singularity_monitor": "singularity_monitor",
            "self_healer": "self_healer",
            "memory_optimizer": "memory_optimizer",
            "proactive_comm": "proactive_communication",
            "personality_manager": "personality_engine",
            "terminal_monitor": "terminal_monitor",
            "latent_distiller": "latent_distiller",
            "ast_guard": "ast_guard",
            "fictional_engines": "fictional_engines",
            "capability_engine": "capability_engine",
            "drive_controller": "drive_engine",
            "cryptolalia_decoder": "cryptolalia_decoder"
        }
        
        target = aliases.get(name)
        if target:
            return self._get_service(target)
            
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
