"""core/skill_evolution.py — Recursive Skill Improvement for Aura Zenith.

Uses distributed agency (SovereignSwarm) to research and refine its own skills.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("Aura.SkillEvolution")

@dataclass
class SkillMutation:
    """A proposed change to a skill's logic or parameters."""
    skill_name: str
    target_logic: str
    rationale: str
    proposed_change: Dict[str, Any]
    benefit_prediction: float  # Expected success rate improvement

class SkillEvolutionEngine:
    """Managing the recursive evolution of Aura's capabilities."""
    
    def __init__(self):
        logger.info("SkillEvolutionEngine initialized.")

    async def identify_evolution_targets(self) -> List[str]:
        """Identify skills that need improvement based on [OMNI] execution logs."""
        from core.container import ServiceContainer
        omni = ServiceContainer.get("omni_tool", default=None)
        if not omni:
            return []
            
        targets = []
        for name, logs in omni._execution_logs.items():
            # Check for error state in recent execution history
            recent_errors = [l for l in logs[-10:] if l["status"] == "error"]
            if len(recent_errors) > 3:
                targets.append(name)
        
        # Fallback to general skill list if no active errors detected via Omni
        if not targets:
            capability_engine = ServiceContainer.get("capability_engine", default=None)
            if capability_engine:
                all_skills = list(capability_engine.skills.keys())
            if all_skills:
                import random
                targets = random.sample(all_skills, min(2, len(all_skills)))
            else:
                targets = ["web_search", "memory_query"]
            
        return targets

    async def spawn_evolution_shard(self, skill_name: str):
        """Spawn a SovereignSwarm shard to research a skill optimization."""
        from core.container import ServiceContainer
        swarm = ServiceContainer.get("sovereign_swarm", default=None)
        if not swarm:
            logger.warning("No SovereignSwarm available for skill evolution.")
            return

        objective = f"Research and propose an optimization for the '{skill_name}' skill. Focus on parameter parsing and error handling."
        await swarm.spawn_shard(
            objective=objective,
            context={"target_skill": skill_name},
            lifespan=600 # 10 minutes research
        )
        logger.info(f"Spawned evolution shard for skill: {skill_name}")

    def propose_mutation(self, mutation: SkillMutation):
        """Publish a mutation proposed by a shard to the EventBus (Issue 93)."""
        logger.info(f"New mutation proposed for {mutation.skill_name}: {mutation.rationale}")
        
        # Publish to EventBus
        from core.event_bus import get_event_bus
        bus = get_event_bus()
        bus.publish_threadsafe("skill_mutation_proposed", {
            "skill": mutation.skill_name,
            "rationale": mutation.rationale,
            "benefit": mutation.benefit_prediction
        })

# Service Registration
def register_skill_evolution():
    """Register the skill evolution engine."""
    from core.container import ServiceContainer, ServiceLifetime
    ServiceContainer.register(
        "skill_evolution",
        factory=lambda: SkillEvolutionEngine(),
        lifetime=ServiceLifetime.SINGLETON
    )
