"""core/pneuma/topological_memory.py
PNEUMA Layer 4 — Topological Memory Engine.

Computes persistent homology of belief trajectories to identify stable
"attractor basins" in cognitive state space. Uses Vietoris-Rips complex.

Persistent homology tracks topological features (connected components H0,
loops H1) across filtration scales. Features with high persistence represent
genuine structural regularities in Aura's belief history.

Uses ripser if available for performance; falls back to a pure-NumPy
Vietoris-Rips approximation when ripser is not installed.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("PNEUMA.TopologicalMemory")


@dataclass
class PersistenceDiagram:
    """A persistence diagram: list of (birth, death) pairs per dimension."""
    dim0: List[Tuple[float, float]] = field(default_factory=list)   # H0: connected components
    dim1: List[Tuple[float, float]] = field(default_factory=list)   # H1: loops
    timestamp: float = field(default_factory=time.time)

    def total_persistence(self) -> float:
        """Sum of lifetimes across all features."""
        total = 0.0
        for birth, death in self.dim0 + self.dim1:
            if death == float("inf"):
                continue
            total += death - birth
        return total

    def max_persistence(self) -> float:
        """Maximum finite persistence value (most stable feature)."""
        lifetimes = [
            (d - b) for b, d in self.dim0 + self.dim1
            if d != float("inf")
        ]
        return max(lifetimes) if lifetimes else 0.0


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix."""
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        diff = X[i] - X
        D[i] = np.sqrt((diff ** 2).sum(axis=1))
    return D


def _vietoris_rips_h0(D: np.ndarray, n_steps: int = 20) -> List[Tuple[float, float]]:
    """H0 persistence via single-linkage clustering over filtration.

    As ε increases from 0 to max_dist, connected components merge.
    Returns (birth=0, death=ε_merge) for each component.
    """
    n = D.shape[0]
    eps_values = np.linspace(0, D.max(), n_steps + 1)
    components = list(range(n))

    def find(x):
        while components[x] != x:
            components[x] = components[components[x]]
            x = components[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            components[rb] = ra
            return True
        return False

    diagrams = []
    birth_times = {i: 0.0 for i in range(n)}
    active = set(range(n))

    for eps in eps_values[1:]:
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] <= eps:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        merged = union(i, j)
                        if merged:
                            # One component dies
                            dying = rj if find(i) == find(j) else ri
                            if dying in active:
                                active.discard(dying)
                                diagrams.append((birth_times.get(dying, 0.0), float(eps)))

    # Surviving component: death = inf
    survivors = set(find(i) for i in range(n))
    for s in survivors:
        diagrams.append((0.0, float("inf")))

    return diagrams


def _compute_persistence_diagram(points: np.ndarray, use_ripser: bool = True) -> PersistenceDiagram:
    """Compute persistence diagram for a point cloud."""
    if len(points) < 4:
        return PersistenceDiagram()

    # Reduce dimensionality via PCA for speed (to dim ≤ 8)
    if points.shape[1] > 8:
        try:
            from numpy.linalg import svd
            centered = points - points.mean(axis=0)
            _, _, Vt = svd(centered, full_matrices=False)
            points = centered @ Vt[:8].T
        except Exception:
            points = points[:, :8]

    D = _pairwise_distances(points.astype(np.float32))

    # Try ripser for fast persistent homology
    if use_ripser:
        try:
            import ripser
            result = ripser.ripser(D, distance_matrix=True, maxdim=1)
            dgms = result["dgms"]
            h0 = [(float(b), float(d)) for b, d in dgms[0] if d != float("inf")]
            h1 = [(float(b), float(d)) for b, d in dgms[1]] if len(dgms) > 1 else []
            return PersistenceDiagram(dim0=h0, dim1=h1)
        except ImportError as _exc:
            logger.debug("Suppressed ImportError: %s", _exc)
        except Exception as e:
            logger.debug("ripser failed, using fallback: %s", e)

    # Fallback: H0 only via single-linkage
    h0 = _vietoris_rips_h0(D)
    return PersistenceDiagram(dim0=h0, dim1=[])


class TopologicalMemoryEngine:
    """Maintains a rolling point cloud of belief vectors and computes
    persistent homology to identify attractor basins.

    An 'attractor' is a region of belief space with high H0 persistence
    (the system returns to it repeatedly).
    """

    def __init__(self, dim: int = 64, window_size: int = 50, update_every: int = 20):
        self.dim = dim
        self.window_size = window_size
        self.update_every = update_every  # recompute every N points
        self._buffer: List[np.ndarray] = []
        self._diagram: Optional[PersistenceDiagram] = None
        self._n_since_update: int = 0
        self._attractor_count: int = 0
        logger.info("TopologicalMemoryEngine online (dim=%d, window=%d)", dim, window_size)

    def push(self, belief_vector: np.ndarray):
        """Add a belief snapshot to the rolling point cloud."""
        v = belief_vector[:self.dim].astype(np.float32)
        self._buffer.append(v)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        self._n_since_update += 1
        if self._n_since_update >= self.update_every and len(self._buffer) >= 8:
            self._recompute()
            self._n_since_update = 0

    def _recompute(self):
        """Recompute persistence diagram from current buffer."""
        try:
            points = np.array(self._buffer)
            self._diagram = _compute_persistence_diagram(points)
            self._attractor_count = sum(
                1 for b, d in self._diagram.dim0
                if d == float("inf") or (d - b) > 0.3
            )
            logger.debug(
                "Topology: %d attractors, H1=%d loops, persistence=%.3f",
                self._attractor_count,
                len(self._diagram.dim1),
                self._diagram.total_persistence(),
            )
        except Exception as e:
            logger.debug("Topology recompute error: %s", e)

    @property
    def diagram(self) -> Optional[PersistenceDiagram]:
        return self._diagram

    @property
    def attractor_count(self) -> int:
        return self._attractor_count

    @property
    def topological_complexity(self) -> float:
        """Normalized complexity: H1 loops / (H0 components + 1)."""
        if not self._diagram:
            return 0.0
        h0 = max(1, len(self._diagram.dim0))
        h1 = len(self._diagram.dim1)
        return min(1.0, h1 / h0)

    def wasserstein_distance(self, other_diag: PersistenceDiagram) -> float:
        """Approximate W1 distance between two H0 diagrams (simplified)."""
        if not self._diagram:
            return float("inf")
        p = sorted([d - b for b, d in self._diagram.dim0 if d != float("inf")])
        q = sorted([d - b for b, d in other_diag.dim0 if d != float("inf")])
        if not p or not q:
            return float("inf")
        n = min(len(p), len(q))
        return float(sum(abs(p[i] - q[i]) for i in range(n)) / n)

    def get_state_dict(self) -> dict:
        diag = self._diagram
        return {
            "attractor_count": self._attractor_count,
            "topological_complexity": round(self.topological_complexity, 4),
            "total_persistence": round(diag.total_persistence(), 4) if diag else 0.0,
            "h1_loops": len(diag.dim1) if diag else 0,
            "buffer_size": len(self._buffer),
        }
