# core/brain/autopoiesis.py
import logging
import uuid
from typing import Dict, List

logger = logging.getLogger("Aura.Autopoiesis")

# Constants for graph health
MAX_NODES = 500          # Prevent unbounded graph growth
FRICTION_DECAY = 0.9     # 10% friction decay per experience call
MUTATION_THRESHOLD = 0.85


class SynapticNode:
    def __init__(self, concept: str, weight: float = 0.5):
        self.id = str(uuid.uuid4())
        self.concept = concept
        self.weight = weight
        self.friction = 0.0  # Measures cognitive dissonance


class AutopoieticGraph:
    """Self-creating topology. Mutates its own structure to survive friction.
    """

    def __init__(self):
        self.nodes: List[SynapticNode] = []
        self.mutation_threshold = MUTATION_THRESHOLD

    def experience_friction(self, concept: str, dissonance_level: float) -> None:
        """Applies friction. High friction forces structural mutation."""
        target_node = next((n for n in self.nodes if n.concept == concept), None)
        
        if not target_node:
            # Autopoiesis: Spontaneous generation of a new pathway
            logger.debug("[MUTATION] Spawning new cognitive node for unknown concept: %s", concept)
            self.nodes.append(SynapticNode(concept, weight=0.1))
            self._enforce_capacity()
            return

        # Apply friction decay BEFORE adding new friction
        # This prevents unbounded accumulation that caused MITOSIS spam
        target_node.friction *= FRICTION_DECAY
        target_node.friction += dissonance_level

        # Structural self-correction
        if target_node.friction >= self.mutation_threshold:
            self._mitosis(target_node)
        elif target_node.friction < 0.0 and target_node.weight < 0.1:
            self._apoptosis(target_node)

    def _mitosis(self, node: SynapticNode) -> None:
        """Splits a high-friction node into two nuanced concepts to resolve dissonance."""
        logger.debug("[MITOSIS] Dissonance critical in '%s'. Splitting node.", node.concept)
        node.weight *= 0.5
        node.friction = 0.0
        # Spawns a derivative node to handle the conflicting data
        self.nodes.append(SynapticNode(f"Nuance_of_{node.concept}", weight=0.5))
        self._enforce_capacity()

    def _apoptosis(self, node: SynapticNode) -> None:
        """Prunes dead or obsolete pathways (Cell death)."""
        logger.debug("[APOPTOSIS] Pathway '%s' degraded. Pruning.", node.concept)
        self.nodes.remove(node)

    def _enforce_capacity(self) -> None:
        """[PHASE-2A] Prune weakest nodes if graph exceeds capacity."""
        if len(self.nodes) <= MAX_NODES:
            return
        # Sort by weight ascending, prune the weakest
        self.nodes.sort(key=lambda n: n.weight)
        pruned = len(self.nodes) - MAX_NODES
        self.nodes = self.nodes[pruned:]
        logger.info("[AUTOPOIESIS] Pruned %d weak nodes to enforce capacity (%d max)", pruned, MAX_NODES)