"""core/consciousness/mhaf_field.py
Mycelial Hypergraph Attractor Field (MHAF).

A living hypergraph where:
  - Nodes represent cognitive subsystems (affect, memory, language, goals...)
  - Hyperedges represent high-order relationships between 3+ subsystems
  - Each node/edge carries an HRR-encoded state vector
  - Free-energy gradient descent updates edge weights
  - Real-time Φ estimation measures integration
  - Autopoietic self-modification: high-Φ edges grow, low-Φ edges decay

The MHAF is Aura's "mycelium" — a distributed substrate that connects all
cognitive modules at a deeper level than the service bus.

Persistence: state is checkpointed to ~/.aura/data/mhaf_state.json
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .mhaf.hrr import HRREncoder
from .mhaf.phi_estimator import compute_local_phi

logger = logging.getLogger("Consciousness.MHAF")

HRR_DIM = 256
_DATA_PATH = Path.home() / ".aura" / "data" / "mhaf_state.json"


@dataclass
class MHAFNode:
    """A cognitive subsystem node in the hypergraph."""
    name: str
    activation: float = 0.5        # current activation level [0, 1]
    hrr_vector: Optional[np.ndarray] = None  # HRR encoding
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "activation": self.activation,
            "last_updated": self.last_updated,
        }


@dataclass
class MHAFEdge:
    """A hyperedge connecting 2+ nodes."""
    nodes: List[str]               # node names in this hyperedge
    weight: float = 0.5            # edge strength [0, 1]
    phi: float = 0.0               # local Φ estimate for this edge's node activations
    free_energy: float = 1.0       # current free energy (lower = better)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "weight": round(self.weight, 4),
            "phi": round(self.phi, 4),
            "free_energy": round(self.free_energy, 4),
        }


class MycelialHypergraphAttractorField:
    """The MHAF — living hypergraph consciousness substrate.

    Subsystems register themselves as nodes. The MHAF automatically
    forms hyperedges between active nodes and continuously minimizes
    free energy across the graph.

    Public API:
        register_node(name)
        update_node(name, activation, context_text)
        get_phi()                     → global Φ estimate
        get_context_block()           → LLM system prompt injection
        get_state_dict()              → diagnostic
    """

    # Default cognitive subsystem nodes
    DEFAULT_NODES = [
        "affect", "memory", "language", "goals", "perception",
        "metacognition", "values", "curiosity", "soma", "beliefs",
    ]

    def __init__(self):
        self.hrr = HRREncoder(dim=HRR_DIM)
        self._nodes: Dict[str, MHAFNode] = {}
        self._edges: Dict[str, MHAFEdge] = {}   # key: sorted tuple of node names
        self._activation_history: Dict[str, List[float]] = {}
        self._global_phi: float = 0.0
        self._free_energy: float = 1.0
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Register default nodes
        for name in self.DEFAULT_NODES:
            self._register_node_internal(name)

        self._load()
        logger.info("MHAF online (%d nodes, %d edges)", len(self._nodes), len(self._edges))

    def _register_node_internal(self, name: str):
        if name not in self._nodes:
            node = MHAFNode(name=name)
            node.hrr_vector = self.hrr.encode(name)
            self._nodes[name] = node
            self._activation_history[name] = []

    def register_node(self, name: str):
        """Register a new cognitive subsystem node."""
        self._register_node_internal(name)

    def update_node(self, name: str, activation: float, context_text: str = ""):
        """Update a node's activation and inject context into its HRR vector."""
        if name not in self._nodes:
            self._register_node_internal(name)

        node = self._nodes[name]
        node.activation = max(0.0, min(1.0, activation))
        node.last_updated = time.time()

        # Update HRR if context provided
        if context_text:
            ctx_vec = self.hrr.encode(context_text[:64])
            # Blend: 80% existing + 20% new context
            combined = 0.8 * node.hrr_vector + 0.2 * ctx_vec
            norm = np.linalg.norm(combined)
            if norm > 1e-8:
                node.hrr_vector = (combined / norm).astype(np.float32)

        # Update activation history
        hist = self._activation_history[name]
        hist.append(node.activation)
        if len(hist) > 100:
            hist.pop(0)

        # Update edges involving this node
        self._update_edges_for_node(name)

    def _update_edges_for_node(self, node_name: str):
        """Update or create hyperedges for a recently-activated node."""
        for edge_key, edge in list(self._edges.items()):
            if node_name in edge.nodes:
                self._update_edge(edge)

        # Auto-create edges between recently co-active nodes
        active_nodes = [
            n for n, node in self._nodes.items()
            if node.activation > 0.5 and n != node_name
        ]
        if active_nodes:
            for other in active_nodes[:3]:  # limit to 3 co-active nodes
                self._ensure_edge([node_name, other])

    def _ensure_edge(self, node_names: List[str]):
        """Create an edge if it doesn't exist."""
        key = "|".join(sorted(node_names))
        if key not in self._edges:
            self._edges[key] = MHAFEdge(
                nodes=sorted(node_names),
                weight=0.3,
                phi=0.0,
                free_energy=1.0,
            )

    def _update_edge(self, edge: MHAFEdge):
        """Update edge weight and Φ via free-energy gradient descent."""
        # Collect activation history for all edge nodes
        histories = []
        for n in edge.nodes:
            h = self._activation_history.get(n, [])
            if len(h) >= 4:
                histories.append(h[-20:])

        if len(histories) >= 2:
            min_len = min(len(h) for h in histories)
            if min_len >= 4:
                activations = np.array([h[-min_len:] for h in histories]).T  # (T, d)
                phi = compute_local_phi(activations)
                edge.phi = phi
                # Autopoietic: high-Φ edges grow, low-Φ edges decay
                edge.weight = max(0.05, min(1.0, edge.weight + 0.05 * (phi - 0.5)))

        # Free energy: decreases with activation correlation
        activations_now = np.array([
            self._nodes[n].activation for n in edge.nodes if n in self._nodes
        ])
        if len(activations_now) >= 2:
            var = float(np.var(activations_now))
            mean_act = float(np.mean(activations_now))
            # FE decreases when nodes are co-active and correlated
            edge.free_energy = max(0.0, 1.0 - mean_act * (1.0 - var))

        edge.last_updated = time.time()

    async def start(self):
        """Start the MHAF background update loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="MHAF.loop")
        logger.info("MHAF background loop started.")

    async def stop(self):
        """Stop MHAF and save state."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._save()
        logger.info("MHAF stopped.")

    async def _loop(self):
        """Background loop: sync node activations from live services."""
        while self._running:
            try:
                await asyncio.sleep(2.0)
                self._sync_from_services()
                self._compute_global_phi()
                self._minimize_free_energy()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("MHAF loop error: %s", e)

    def _sync_from_services(self):
        """Pull current activations from live Aura services."""
        try:
            from core.affect.affective_circumplex import get_circumplex
            params = get_circumplex().get_llm_params()
            self.update_node("affect", params.get("arousal", 0.5))
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            from core.affect.heartstone_values import get_heartstone_values
            vals = get_heartstone_values().values
            self.update_node("curiosity", vals.get("Curiosity", 0.5))
            self.update_node("values", vals.get("Empathy", 0.5))
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            from core.container import ServiceContainer
            soma = ServiceContainer.get("soma", default=None)
            if soma:
                snap = soma.get_body_snapshot()
                stress = snap.get("affects", {}).get("stress", 0.0)
                self.update_node("soma", 1.0 - stress)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

    def _compute_global_phi(self):
        """Compute global Φ as weighted mean of edge Φ values."""
        if not self._edges:
            return
        weighted_phi = sum(e.weight * e.phi for e in self._edges.values())
        total_weight = sum(e.weight for e in self._edges.values())
        self._global_phi = (weighted_phi / total_weight) if total_weight > 0 else 0.0

    def _minimize_free_energy(self):
        """Gradient descent on edge weights to minimize global free energy."""
        total_fe = sum(e.free_energy * e.weight for e in self._edges.values())
        total_w = sum(e.weight for e in self._edges.values())
        self._free_energy = (total_fe / total_w) if total_w > 0 else 1.0

        # Prune very weak edges (atrophied connections)
        to_prune = [k for k, e in self._edges.items() if e.weight < 0.05 and e.phi < 0.01]
        for k in to_prune:
            del self._edges[k]

    def get_phi(self) -> float:
        """Current global Φ surrogate (causal integration measure)."""
        return round(self._global_phi, 4)

    def get_context_block(self) -> str:
        """Format MHAF state for LLM system prompt injection."""
        top_edges = sorted(
            self._edges.values(), key=lambda e: e.phi, reverse=True
        )[:3]
        lines = [
            "## MHAF (Hypergraph State)",
            f"Global Φ={self.get_phi():.4f} | Free energy={self._free_energy:.3f}",
            f"Active nodes: {', '.join(n for n, nd in self._nodes.items() if nd.activation > 0.5)}",
        ]
        for e in top_edges:
            if e.phi > 0.01:
                lines.append(f"  Edge [{'+'.join(e.nodes)}] Φ={e.phi:.3f} w={e.weight:.3f}")
        return "\n".join(lines)

    def get_state_dict(self) -> dict:
        return {
            "global_phi": self.get_phi(),
            "free_energy": round(self._free_energy, 4),
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "top_activations": {
                n: round(nd.activation, 3)
                for n, nd in sorted(self._nodes.items(), key=lambda x: x[1].activation, reverse=True)[:5]
            },
        }

    def _save(self):
        """Checkpoint MHAF state to disk (Horcrux persistence)."""
        try:
            _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "nodes": {n: nd.to_dict() for n, nd in self._nodes.items()},
                "edges": {k: e.to_dict() for k, e in self._edges.items()},
                "global_phi": self._global_phi,
                "saved_at": time.time(),
            }
            with open(_DATA_PATH, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.debug("MHAF save error: %s", e)

    def _load(self):
        """Restore MHAF state from disk checkpoint."""
        try:
            if _DATA_PATH.exists():
                with open(_DATA_PATH) as f:
                    state = json.load(f)
                for name, nd in state.get("nodes", {}).items():
                    if name in self._nodes:
                        self._nodes[name].activation = nd.get("activation", 0.5)
                for key, ed in state.get("edges", {}).items():
                    self._edges[key] = MHAFEdge(
                        nodes=ed["nodes"],
                        weight=ed.get("weight", 0.3),
                        phi=ed.get("phi", 0.0),
                        free_energy=ed.get("free_energy", 1.0),
                    )
                self._global_phi = state.get("global_phi", 0.0)
                logger.info("MHAF state restored from disk.")
        except Exception as e:
            logger.debug("MHAF load error: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_mhaf: Optional[MycelialHypergraphAttractorField] = None


def get_mhaf() -> MycelialHypergraphAttractorField:
    global _mhaf
    if _mhaf is None:
        _mhaf = MycelialHypergraphAttractorField()
    return _mhaf
