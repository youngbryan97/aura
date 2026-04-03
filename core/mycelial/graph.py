import networkx as nx
import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger("Aura.Mycelial")

class MycelialNetwork:
    """
    Zenith Audit Fix 4.1: Minimal safe Mycelial DAG with cycle detection.
    Prevents infinite feedback loops between memory nodes and skill execution.
    """
    def __init__(self):
        self.G = nx.DiGraph()
        self._lock = asyncio.Lock()

    async def add_edge(self, memory_node: str, skill_node: str):
        """Add a directed edge with mandatory cycle detection."""
        async with self._lock:
            self.G.add_edge(memory_node, skill_node)
            # Cycle check: if skill points back to memory, it's a hallucination loop risk
            if nx.has_path(self.G, skill_node, memory_node):
                logger.warning("🕸️ Mycelial cycle detected: %s -> %s. Rejecting edge.", memory_node, skill_node)
                self.G.remove_edge(memory_node, skill_node)
                return False
            
            logger.info("🕸️ Mycelial edge added: %s -> %s", memory_node, skill_node)
            return True

    async def plan_path(self, start_memory: str, goal_skill: str) -> List[str]:
        """Find the shortest safe path through the mycelial graph."""
        async with self._lock:
            try:
                return nx.shortest_path(self.G, start_memory, goal_skill)
            except nx.NetworkXNoPath:
                return []

# Singleton
_mycelial: Optional[MycelialNetwork] = None

def get_mycelial() -> MycelialNetwork:
    global _mycelial
    if _mycelial is None:
        _mycelial = MycelialNetwork()
    return _mycelial
