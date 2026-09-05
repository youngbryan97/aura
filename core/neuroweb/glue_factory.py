import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from .structures import Neuron, Synapse

logger = logging.getLogger("NeuroWeb.GlueFactory")

class GlueFactory:
    """Creates synapses (connections) between concepts/intents and functional neurons.
    Uses semantic vector search to find the best functional mapping.
    """

    def __init__(self, memory_system=None):
        self.neurons = [] # Active neurons in current context
        self.memory = memory_system
        
    async def forge(self, intent) -> Optional[Synapse]:
        """Forge a synapse for the given intent using semantic search.
        """
        if not self.neurons:
            logger.warning("No neurons available to forge synapse.")
            return None
            
        intent_text = intent.text.lower().strip()
        
        # Guard against accidental skill mapping for simple greetings
        greetings = ["hey", "hi", "hello", "aura", "bryan", "how are you", "what's up"]
        words = intent_text.split()
        if len(words) <= 3 and all(re.sub(r'[^\w]', '', w) in greetings for w in words):
            logger.debug("GlueFactory: '%s' identified as greeting. Skipping skill mapping.", intent_text)
            return None
        
        # 1. Semantic Search via VectorMemory
        if self.memory:
            # Query memory for existing synapses or similar neurons
            matches = await self.memory.search_similar(intent_text, limit=3)
            for match in matches:
                neuron_id = match.get("metadata", {}).get("neuron_id")
                if neuron_id:
                    # Verify neuron exists in current context
                    if any(n.id == neuron_id for n in self.neurons):
                        return await self.forge_synapse(intent_text, neuron_id, strength=0.8)

        # 2. Heuristic Matching (Fallback)
        best_neuron = None
        best_strength = 0.0
        
        # Exact match on neuron ID (case-insensitive)
        for neuron in self.neurons:
            if neuron.id.lower() == intent_text:
                best_neuron = neuron
                best_strength = 0.9
                break
                
        # Keyword matching and semantic weighting
        if not best_neuron:
            intent_words = set(re.findall(r'\w+', intent_text))
            
            # Weighted Scoring
            scores = []
            for neuron in self.neurons:
                score = 0.0
                n_id_lower = neuron.id.lower()
                
                # Check path/ID for keywords
                if intent_text in n_id_lower or n_id_lower in intent_text:
                    score += 0.5
                
                # Word-based overlap
                n_words = set(re.findall(r'\w+', n_id_lower))
                overlap = intent_words.intersection(n_words)
                if overlap:
                    score += 0.1 * len(overlap)
                
                if score > 0:
                    scores.append((neuron, score))
            
            if scores:
                scores.sort(key=lambda x: x[1], reverse=True)
                best_neuron, best_strength = scores[0]
                best_strength = min(0.8, best_strength)

        # Fallback for common tool categories if no clear neuron found
        if not best_neuron or best_strength < 0.4:
            if any(k in intent_text for k in ["browser", "web", "search", "google", "chrome", "internet", "open url", "visit"]):
                # Favor browser skill
                best_neuron = next((n for n in self.neurons if "browser" in n.id.lower()), best_neuron)
                best_strength = 0.5
            elif any(k in intent_text for k in ["code", "python", "script", "file", "edit", "write", "save", "terminal", "shell", "run"]):
                best_neuron = next((n for n in self.neurons if any(k in n.id.lower() for k in ["code", "file", "ops", "shell"])), best_neuron)
                best_strength = 0.5

        if best_neuron:
            return await self.forge_synapse(intent_text, best_neuron.id, strength=best_strength)
            
        logger.warning("GlueFactory could not find a neuron for intent: %s", intent.text)
        return None

    async def forge_synapse(self, intent_text: str, neuron_id: str, strength: float = 0.5) -> Optional[Synapse]:
        """Creates a synapse and optionally persists it."""
        # Find the neuron to ensure it exists
        target_neuron = next((n for n in self.neurons if n.id == neuron_id), None)
        if not target_neuron:
            logger.error("Cannot forge synapse: Neuron %s not found.", neuron_id)
            return None

        synapse = Synapse(
            id=f"syn_{uuid.uuid4().hex[:8]}",
            intent_pattern=intent_text,
            neuron_id=neuron_id,
            strength=strength,
            created_at=time.time(),
            status="active"
        )
        
        # Persist the connection for future learning
        if self.memory:
            await self.memory.log_event(
                event_type="synapse",
                content=f"Synapse forged: {intent_text} -> {neuron_id}",
                metadata={
                    "intent": intent_text,
                    "neuron_id": neuron_id,
                    "strength": strength
                }
            )
        
        logger.info("✨ Synapse Forged: '%s' <==> %s", intent_text, neuron_id)
        return synapse