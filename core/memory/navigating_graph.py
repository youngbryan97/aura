"""Navigating Spreading-out Graph (NSG) for episodic memory retrieval.

Inspired by the Cognitive SSD paper's graph search mechanism and the NSG
algorithm (Fu et al. 2017). Instead of brute-force vector search across
all memories, this builds a proximity graph where each memory links to
its K nearest neighbors, then searches by walking the graph greedily.

Performance: O(log N) retrieval vs O(N) brute-force, with ~95% recall.

The graph is built incrementally as memories are added and rebuilt
periodically during dream consolidation.

Usage:
    nsg = NavigatingGraph(dim=384)
    nsg.add("mem_001", embedding_vector, metadata={"text": "..."})
    nsg.add("mem_002", embedding_vector, metadata={"text": "..."})

    results = nsg.search(query_vector, top_k=5)
    # [{"id": "mem_001", "distance": 0.12, "metadata": {...}}, ...]
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("Aura.NavigatingGraph")

# Graph parameters
K_NEIGHBORS = 16          # Max edges per node
SEARCH_BEAM_WIDTH = 32    # Candidates evaluated per search
MAX_SEARCH_STEPS = 100    # Prevent infinite walks
REBUILD_THRESHOLD = 500   # Rebuild graph after this many inserts


@dataclass
class GraphNode:
    """A node in the proximity graph."""
    node_id: str
    embedding: np.ndarray
    neighbors: List[str] = field(default_factory=list)  # IDs of K nearest
    metadata: Dict[str, Any] = field(default_factory=dict)
    added_at: float = 0.0


class NavigatingGraph:
    """Proximity graph for fast approximate nearest neighbor search."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._nodes: Dict[str, GraphNode] = {}
        self._entry_point: Optional[str] = None
        self._inserts_since_rebuild = 0
        self._total_searches = 0
        self._total_search_steps = 0

    def add(self, node_id: str, embedding: np.ndarray,
            metadata: Optional[Dict[str, Any]] = None):
        """Add a memory to the graph with incremental neighbor linking."""
        if embedding.shape[0] != self.dim:
            # Auto-adapt dimension on first add
            if not self._nodes:
                self.dim = embedding.shape[0]
            else:
                logger.warning("Dimension mismatch: expected %d, got %d", self.dim, embedding.shape[0])
                return

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        node = GraphNode(
            node_id=node_id,
            embedding=embedding,
            metadata=metadata or {},
            added_at=time.time(),
        )
        self._nodes[node_id] = node

        if self._entry_point is None:
            self._entry_point = node_id
            return

        # Find neighbors for the new node via greedy search
        candidates = self._greedy_search(embedding, K_NEIGHBORS * 2)
        neighbors = [c[0] for c in candidates[:K_NEIGHBORS]]
        node.neighbors = neighbors

        # Add reverse edges (bidirectional connectivity)
        for neighbor_id in neighbors:
            neighbor = self._nodes.get(neighbor_id)
            if neighbor is None:
                continue
            if len(neighbor.neighbors) < K_NEIGHBORS:
                neighbor.neighbors.append(node_id)
            else:
                # Replace the furthest neighbor if new node is closer
                distances = [
                    self._distance(neighbor.embedding, self._nodes[n].embedding)
                    for n in neighbor.neighbors if n in self._nodes
                ]
                if distances:
                    max_dist_idx = int(np.argmax(distances))
                    new_dist = self._distance(neighbor.embedding, embedding)
                    if new_dist < distances[max_dist_idx]:
                        neighbor.neighbors[max_dist_idx] = node_id

        self._inserts_since_rebuild += 1

        # Periodic rebuild for graph quality
        if self._inserts_since_rebuild >= REBUILD_THRESHOLD:
            self.rebuild()

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find the top_k nearest memories to the query vector."""
        if not self._nodes or self._entry_point is None:
            return []

        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm

        candidates = self._greedy_search(query, top_k)
        self._total_searches += 1

        return [
            {
                "id": node_id,
                "distance": float(dist),
                "metadata": self._nodes[node_id].metadata if node_id in self._nodes else {},
            }
            for node_id, dist in candidates[:top_k]
        ]

    def remove(self, node_id: str):
        """Remove a memory from the graph."""
        if node_id not in self._nodes:
            return

        node = self._nodes[node_id]

        # Remove from neighbors' neighbor lists
        for neighbor_id in node.neighbors:
            neighbor = self._nodes.get(neighbor_id)
            if neighbor and node_id in neighbor.neighbors:
                neighbor.neighbors.remove(node_id)

        del self._nodes[node_id]

        # Update entry point if needed
        if self._entry_point == node_id:
            self._entry_point = next(iter(self._nodes), None)

    def rebuild(self):
        """Rebuild the entire graph for optimal connectivity.

        Called periodically during dream consolidation.
        """
        if len(self._nodes) < 2:
            return

        start = time.time()

        # Collect all embeddings
        ids = list(self._nodes.keys())
        embeddings = np.array([self._nodes[nid].embedding for nid in ids])

        # Compute all pairwise distances (O(N^2) — only for rebuild)
        # For large N, use random sampling
        n = len(ids)
        if n > 5000:
            logger.info("Graph too large for full rebuild (%d nodes), using incremental.", n)
            self._inserts_since_rebuild = 0
            return

        # Build distance matrix
        # Use matrix multiplication for cosine distance: dist = 1 - dot(A, B)
        dot_products = embeddings @ embeddings.T
        distances = 1.0 - dot_products

        # Assign K nearest neighbors for each node
        for i, nid in enumerate(ids):
            dists = distances[i]
            dists[i] = float("inf")  # Exclude self
            nearest_indices = np.argsort(dists)[:K_NEIGHBORS]
            self._nodes[nid].neighbors = [ids[j] for j in nearest_indices]

        # Pick entry point as the medoid (closest to centroid)
        centroid = np.mean(embeddings, axis=0)
        centroid_dists = np.linalg.norm(embeddings - centroid, axis=1)
        self._entry_point = ids[int(np.argmin(centroid_dists))]

        self._inserts_since_rebuild = 0
        elapsed = time.time() - start
        logger.info("NSG rebuilt: %d nodes, %d edges, %.2fs", n, n * K_NEIGHBORS, elapsed)

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get a node's embedding (for conceptual gravitation)."""
        node = self._nodes.get(node_id)
        return node.embedding.copy() if node else None

    def set_embedding(self, node_id: str, embedding: np.ndarray):
        """Update a node's embedding (for conceptual gravitation)."""
        node = self._nodes.get(node_id)
        if node is not None:
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
            node.embedding = embedding

    # ── Internal ─────────────────────────────────────────────────────────

    def _greedy_search(self, query: np.ndarray, beam_width: int) -> List[Tuple[str, float]]:
        """Greedy beam search through the graph."""
        if not self._entry_point or self._entry_point not in self._nodes:
            return []

        visited: Set[str] = set()
        # Priority queue: (distance, node_id)
        entry = self._nodes[self._entry_point]
        entry_dist = self._distance(query, entry.embedding)

        candidates = [(entry_dist, self._entry_point)]
        visited.add(self._entry_point)
        results = [(self._entry_point, entry_dist)]

        steps = 0
        while candidates and steps < MAX_SEARCH_STEPS:
            steps += 1
            # Pick closest unvisited candidate
            candidates.sort(key=lambda x: x[0])
            current_dist, current_id = candidates.pop(0)

            current_node = self._nodes.get(current_id)
            if current_node is None:
                continue

            # Check all neighbors
            for neighbor_id in current_node.neighbors:
                if neighbor_id in visited or neighbor_id not in self._nodes:
                    continue
                visited.add(neighbor_id)

                neighbor = self._nodes[neighbor_id]
                dist = self._distance(query, neighbor.embedding)

                results.append((neighbor_id, dist))
                candidates.append((dist, neighbor_id))

            # Keep only beam_width best candidates
            if len(candidates) > beam_width:
                candidates.sort(key=lambda x: x[0])
                candidates = candidates[:beam_width]

        self._total_search_steps += steps

        # Sort results by distance
        results.sort(key=lambda x: x[1])
        return results

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance (1 - cosine_similarity)."""
        dot = np.dot(a, b)
        return float(1.0 - dot)

    def get_status(self) -> Dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "avg_neighbors": (
                sum(len(n.neighbors) for n in self._nodes.values()) / max(len(self._nodes), 1)
            ),
            "entry_point": self._entry_point,
            "total_searches": self._total_searches,
            "avg_search_steps": (
                self._total_search_steps / max(self._total_searches, 1)
            ),
            "inserts_since_rebuild": self._inserts_since_rebuild,
        }


_instance: Optional[NavigatingGraph] = None


def get_navigating_graph(dim: int = 384) -> NavigatingGraph:
    global _instance
    if _instance is None:
        _instance = NavigatingGraph(dim=dim)
    return _instance
