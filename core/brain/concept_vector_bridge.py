"""core/brain/concept_vector_bridge.py — Cryptolalia & Latent Telepathy
========================================================================
Implements Phase 23.1: Cryptolalia.
This module allows Aura's internal shards (e.g., Subconscious, Cognitive, 
Epistemic) to communicate bypassing English text generation. 

Instead of generating a string, they pass high-dimensional embedding vectors 
or semantic hash-trees. This is exponentially faster and lossless compared to 
parsing human language.
"""

from core.runtime.errors import record_degradation
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Cryptolalia")

@dataclass
class LatentThought:
    """A raw, un-decoded semantic vector representing a concept."""
    id: str
    source_node: str
    vector: List[float]  # The high-dimensional embedding
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class ConceptVectorBridge:
    """The central hub for latent telepathy between nodes."""
    name = "concept_vector_bridge"

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.active_streams: Dict[str, List[LatentThought]] = {}
        # Simple cache for mapping English to vectors for grounding
        self._concept_cache: Dict[str, List[float]] = {}
        
    async def transmit(self, source: str, target: str, semantic_vector: List[float], metadata: Dict[str, Any] = None) -> str:
        """
        Send a raw vector payload to another node.
        """
        thought_id = f"latent_{int(time.time() * 1000)}"
        thought = LatentThought(
            id=thought_id,
            source_node=source,
            vector=semantic_vector,
            metadata=metadata or {}
        )
        
        if target not in self.active_streams:
            self.active_streams[target] = []
            
        self.active_streams[target].append(thought)
        logger.debug(f"🌌 [Cryptolalia] {source} -> {target} (Vector dim: {len(semantic_vector)})")
        
        # Fire event for the decoder or monitoring
        event_bus = ServiceContainer.get("event_bus", default=None)
        if event_bus:
            await event_bus.publish("cryptolalia_transmission", {
                "source": source,
                "target": target,
                "thought_id": thought_id
            })
            
        return thought_id

    async def receive(self, target: str, consume: bool = True) -> List[LatentThought]:
        """
        Fetch pending latent thoughts for a given node.
        """
        if target not in self.active_streams:
            return []
            
        thoughts = self.active_streams[target]
        if consume:
            self.active_streams[target] = []
        return thoughts

    async def generate_concept_vector(self, text_concept: str) -> List[float]:
        """
        Convert a human concept into a latent vector for internal routing.
        Requires an embedding provider (LocalBrain/MLX).
        """
        if text_concept in self._concept_cache:
            return self._concept_cache[text_concept]
            
        cognition = ServiceContainer.get("cognitive_engine", default=None)
        if not cognition or not hasattr(cognition, "client"):
            # Fallback mock for testing if no provider
            logger.warning("No embedding provider found for vector creation. Using pseudo-random semantic hash.")
            np.random.seed(hash(text_concept) % (2**32))
            vector = np.random.normal(0, 0.1, 768).tolist()
            self._concept_cache[text_concept] = vector
            return vector
            
        try:
            vector = await cognition.client.generate_embedding(text_concept)
            self._concept_cache[text_concept] = vector
            return vector
        except Exception as e:
            record_degradation('concept_vector_bridge', e)
            logger.error(f"Failed to generate concept vector: {e}")
            return []

def register_concept_bridge(orchestrator=None):
    bridge = ConceptVectorBridge(orchestrator)
    ServiceContainer.register_instance("concept_bridge", bridge)
    return bridge
