import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from core.brain.cognitive_engine import CognitiveEngine
from core.capability_engine import CapabilityEngine as RobustSkillRegistry

from .glue_factory import GlueFactory
from .structures import Intent, Neuron, Synapse

logger = logging.getLogger("NeuroWeb.CNS")

class CentralNervousSystem:
    """Central Nervous System (CNS) v2.0
    Orchestrates the neural processing pipeline with real cognitive grounding.
    """

    def __init__(self, memory_system, brain: CognitiveEngine, registry: Optional[RobustSkillRegistry] = None):
        if registry is None:
            # Lazy import to avoid circular dependency
            from core.container import get_container
            registry = get_container().get('capability_engine')
            
        self.memory = memory_system
        self.brain = brain
        self.registry = registry
        self.glue_factory = GlueFactory(memory_system=memory_system)
        self.neurons: Dict[str, Neuron] = {}
        
        # Initialize internal neuron maps from registry
        self._refresh_neurons()

    def _refresh_neurons(self):
        """Syncs the internal neuron map with the skill registry."""
        available_skills = self.registry.get_available_skills()
        for skill_name in available_skills:
            metadata = self.registry.get(skill_name)
            if metadata:
                neuron = Neuron(
                    id=f"skill:{skill_name}",
                    type="skill",
                    path=f"{metadata.skill_class.__module__}.{metadata.skill_class.__name__}"
                )
                self.neurons[neuron.id] = neuron
        
        self.glue_factory.neurons = list(self.neurons.values())
        logger.info("CNS: %d neurons loaded from registry.", len(self.neurons))

    async def process_stimulus(self, text: str) -> Dict[str, Any]:
        """Process a text stimulus (user input) using the brain to extract intent.
        Returns an execution plan.
        """
        # 0. Fast Validation (Gating)
        if not text or len(text) < 2:
            return {"status": "ignored", "reason": "Stimulus too short or empty."}

        # 1. Intent Extraction via CognitiveEngine
        prompt = f"""
        Extract the INTENT and CONFIDENCE for the following stimulus:
        "{text}"
        
        Return JSON:
        {{
            "intent": "Short descriptive intent label",
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation"
        }}
        """
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            
            # Phase 27: Aggressive Memory Override
            if mem.percent > 85:
                logger.info("🧠 CNS: Critical Memory (%s%%) - Bypassing intent extraction.", mem.percent)
                intent = Intent(text=text, confidence=1.0)
            else:
                # Reduced timeouts for Phase 27 stability
                timeout = 4.0 if mem.percent > 75 else 8.0
                
                # Concurrent preparation: Start thinking while pre-checking memory/cache (future expansion)
                thought = await asyncio.wait_for(
                    self.brain.think(
                        objective=prompt,
                        context={"role": "cns_intent_extractor"},
                        mode="fast"
                    ),
                    timeout=timeout
                )
                
                json_match = re.search(r"\{.*\}", thought.content, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                        intent = Intent(text=data.get("intent", text), confidence=data.get("confidence", 0.5))
                    except (json.JSONDecodeError, KeyError):
                        intent = Intent(text=text, confidence=0.5)
                else:
                    intent = Intent(text=text, confidence=0.5)
        except (asyncio.TimeoutError, Exception) as e:
            if isinstance(e, asyncio.TimeoutError):
                logger.warning("CNS Intent Extraction timed out - using raw stimulus.")
            else:
                logger.error("CNS Intent Extraction failed: %s", e)
            intent = Intent(text=text, confidence=0.5)

        # 2. Competitive Bottleneck (Phase 1 Consciousness)
        gwt = getattr(getattr(self.brain, 'consciousness', None), 'global_workspace', None)
        if gwt:
            from core.consciousness.global_workspace import CognitiveCandidate
            candidate = CognitiveCandidate(
                content=intent.text,
                source="cns_stimulus",
                priority=intent.confidence
            )
            await gwt.submit(candidate)
            try:
                # Wait for competition winner (max 2s)
                winner = await asyncio.wait_for(gwt.run_competition(), timeout=2.0)
                if winner and winner.source == "cns_stimulus":
                    logger.debug("CNS: Stimulus won GW competition (pri=%.2f)", winner.effective_priority)
                else:
                    logger.info("CNS: Stimulus silenced by GW competition bottleneck.")
            except asyncio.TimeoutError:
                logger.debug("CNS: GW competition timed out - proceeding with stimulus.")
            except Exception as e:
                logger.error("CNS GW bottleneck error: %s", e)

        # 3. Forge Synapse (via GlueFactory - now using semantic search)
        synapse = await self.glue_factory.forge(intent)
        
        if synapse:
            neuron = self.neurons.get(synapse.neuron_id)
            return {
                "status": "active",
                "intent": intent,
                "execution": {
                    "neuron": neuron,
                    "synapse": synapse
                }
            }
        
        return {
            "status": "unresolved",
            "intent": intent,
            "reason": "No functional neuron mapped to this intent."
        }