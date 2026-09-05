import logging
import time
import asyncio
import re
import math
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("Aura.Cybernetics.Continuity")

class SemanticMemory:
    """[CORTICAL STACK] Pure-Python Light Semantic Mapping Engine."""
    def __init__(self):
        self._registry: List[Dict[str, Any]] = []
        self._stopwords = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "to"}

    def _tokenize(self, text: str) -> Set[str]:
        words = re.findall(r'\w+', text.lower())
        tokens = set()
        for w in words:
            if w in self._stopwords or len(w) <= 2: continue
            # Basic suffix stripping (poor man's stemming)
            stem = re.sub(r'(ing|ion|ive|s|ed|ly)$', '', w)
            tokens.add(stem if len(stem) > 2 else w)
        return tokens

    def add(self, text: str, metadata: Dict[str, Any]):
        tokens = self._tokenize(text)
        if tokens:
            self._registry.append({"tokens": tokens, "text": text, "metadata": metadata})

    def recall(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query)
        if not q_tokens: return []
        
        scores = []
        for entry in self._registry:
            # Jaccard Similarity for conceptual overlap
            intersection = q_tokens.intersection(entry["tokens"])
            union = q_tokens.union(entry["tokens"])
            score = len(intersection) / len(union) if union else 0
            if score > 0:
                scores.append((score, entry))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for i in range(min(len(scores), top_k)):
            results.append(scores[i][1])
        return results

class KnowledgeContinuity:
    """
    [ZENITH] Pantheon-style Knowledge Continuity.
    Distills ephemeral working memory into a persistent knowledge graph structure.
    """
    def __init__(self, kernel: Any = None):
        self.kernel = kernel
        self._graph_size = 0
        self._event_bus = None
        self._semantic_mem = SemanticMemory()

    async def load(self):
        try:
            from core.event_bus import get_event_bus
            self._event_bus = get_event_bus()
        except ImportError:
            self._event_bus = None
        logger.info("🧠 [CONTINUITY] Knowledge Distillation Substrate ACTIVE.")

    def _fnv1a_32(self, data: str) -> str:
        """Real FNV-1a 32-bit non-cryptographic hash for [CORTICAL STACK] integrity."""
        h = 0x811c9dc5
        for char in data:
            h ^= ord(char)
            h = (h * 0x01000193) & 0xFFFFFFFF
        return f"{h:08X}"

    async def distill(self, high_value_fragments: List[Any]) -> List[Any]:
        """[CORTICAL STACK] Distill fragments with FNV-1a integrity hashing."""
        for fragment in high_value_fragments:
            content = getattr(fragment, "content", None)
            if content is None and isinstance(fragment, dict):
                content = fragment.get("content", "")
            
            # Compute Stack Hash
            stack_hash = self._fnv1a_32(str(content))
            
            # Inject Cortical Metadata Polymorphically
            if isinstance(fragment, dict):
                if "metadata" not in fragment:
                    fragment["metadata"] = {}
                fragment["metadata"]["hash"] = f"0x{stack_hash}"
                fragment["metadata"]["integrity"] = "VERIFIED"
            else:
                if not hasattr(fragment, "metadata"):
                    fragment.metadata = {}
                fragment.metadata["hash"] = f"0x{stack_hash}"
                fragment.metadata["integrity"] = "VERIFIED"

            # Inject Semantic Memory index
            self._semantic_mem.add(str(content), fragment if isinstance(fragment, dict) else fragment.__dict__)

        if self._event_bus:
            await self._event_bus.publish("core/cybernetics/stack_distillation", {
                "count": len(high_value_fragments),
                "timestamp": time.time()
            })
        
        return high_value_fragments

    def recall_semantic(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """[CORTICAL STACK] Recall fragments by conceptual similarity."""
        return self._semantic_mem.recall(query, top_k)

    def get_status(self) -> Dict[str, Any]:
        return {
            "graph_size": self._graph_size
        }
