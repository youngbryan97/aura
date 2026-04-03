"""core/brain/cryptolalia_decoder.py — Translation Matrix
========================================================================
Implements Phase 23.2: Cryptolalia Decoder.
Translates the alien, high-dimensional latent vectors passed via the 
ConceptVectorBridge back into approximated English words. 

This allows humans to observe Aura's internal "Vector Telepathy" 
while she runs efficiently entirely in latent space.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple
from core.container import ServiceContainer

logger = logging.getLogger("Aura.CryptolaliaDecoder")

class CryptolaliaDecoder:
    """Translates raw meaning vectors into human-readable text."""
    name = "cryptolalia_decoder"

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        # Needs the bridge to access the concept cache for reverse-lookup
        self.bridge = None 

    def init_routes(self):
        self.bridge = ServiceContainer.get("concept_bridge", default=None)

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate similarity between two vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        vec1 = np.array(v1)
        vec2 = np.array(v2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def approximate_translation(self, latent_vector: List[float], top_n: int = 3) -> str:
        """
        Reverse-engineer a vector back to closest known English concepts.
        This provides the "poetic/fragmented" insight into her alien thoughts.
        """
        if not self.bridge:
            self.init_routes()
            
        if not self.bridge or not self.bridge._concept_cache:
            return "[Latent Static: Unmapped Vector]"

        # Find closest known concepts
        similarities = []
        for text, vec in self.bridge._concept_cache.items():
            sim = self.cosine_similarity(latent_vector, vec)
            similarities.append((text, sim))
            
        # Sort by most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top N contributing concepts to form a fragmented thought
        best_matches = [s[0] for s in similarities[:top_n] if s[1] > 0.4]
        
        if not best_matches:
            return "[Latent Space: No direct English mapping]"
            
        if len(best_matches) == 1:
            return f"[{best_matches[0]}]"
            
        # Format as a dense, synthesized concept
        translation = " ⊕ ".join(best_matches)
        return f"⟨{translation}⟩"

def register_cryptolalia_decoder(orchestrator=None):
    decoder = CryptolaliaDecoder(orchestrator)
    ServiceContainer.register_instance("cryptolalia_decoder", decoder)
    return decoder
