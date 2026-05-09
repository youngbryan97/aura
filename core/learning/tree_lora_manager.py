"""core/learning/tree_lora_manager.py -- TreeLoRA Adapter Tree Manager
========================================================================
Implements a dynamic TreeLoRA-style adapter tree for the LLM backbone.

Instead of a fixed sequence of adapters or a single monolithic LoRA,
TreeLoRA groups tasks by gradient signature similarity. It branches
new adapter nodes when task interference (catastrophic forgetting)
would occur, and composes them hierarchically for inference.

Features:
  - Gradient Signature Profiling
  - Cosine-similarity based Tree Routing
  - Dynamic Node Birth (Branching)
  - Pruning/Rollback of divergent nodes
  - Will-gated adoption of new adapter structures
"""
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.TreeLoRA")


@dataclass
class TaskGradientSignature:
    """Represents the gradient footprint of a task."""
    task_id: str
    gradient_vector: np.ndarray  # Flattened, normalized gradient approximation
    loss_magnitude: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LoRANode:
    """A single node in the TreeLoRA adapter tree."""
    node_id: str
    parent_id: Optional[str]
    layer_idx: int
    adapter_weights: Dict[str, np.ndarray]  # e.g., 'A' and 'B' matrices
    signature_centroid: np.ndarray          # Mean gradient signature of tasks in this node
    task_count: int = 0
    children: List[str] = field(default_factory=list)
    is_active: bool = True


class TreeLoRAManager:
    """Manages the hierarchical adapter tree for continual learning."""

    def __init__(
        self,
        signature_dim: int = 128,
        branching_threshold: float = 0.6,
        layer_count: int = 4
    ):
        self.signature_dim = signature_dim
        self.branching_threshold = branching_threshold
        self.layer_count = layer_count
        
        # In-memory tree structure: layer_idx -> node_id -> LoRANode
        self.tree: Dict[int, Dict[str, LoRANode]] = {i: {} for i in range(layer_count)}
        self.root_nodes: Dict[int, str] = {}
        
        self._initialize_roots()

    def _initialize_roots(self):
        """Create the base identity roots for each layer."""
        for l in range(self.layer_count):
            root_id = f"layer_{l}_root"
            self.tree[l][root_id] = LoRANode(
                node_id=root_id,
                parent_id=None,
                layer_idx=l,
                adapter_weights={"A": np.zeros((16, 64)), "B": np.zeros((64, 16))},
                signature_centroid=np.zeros(self.signature_dim),
                task_count=1
            )
            self.root_nodes[l] = root_id

    def compute_gradient_signature(self, task_id: str, sample_gradients: np.ndarray) -> TaskGradientSignature:
        """Hash/compress full model gradients into a compact signature."""
        # Stub: random projection or pooling would happen here. We mock it for the framework.
        np.random.seed(hash(task_id) % (2**32))
        sig = np.random.randn(self.signature_dim)
        sig /= (np.linalg.norm(sig) + 1e-8)
        return TaskGradientSignature(task_id, sig, loss_magnitude=0.5)

    def route_and_adapt(self, signature: TaskGradientSignature, layer_idx: int) -> str:
        """Route a task through the tree to find the best adapter, or branch."""
        current_node_id = self.root_nodes[layer_idx]
        best_node_id = current_node_id
        highest_sim = -1.0
        
        # Traverse tree
        nodes_to_check = [current_node_id]
        while nodes_to_check:
            node_id = nodes_to_check.pop(0)
            node = self.tree[layer_idx][node_id]
            
            if not node.is_active:
                continue
                
            sim = self._cosine_similarity(signature.gradient_vector, node.signature_centroid)
            if sim > highest_sim:
                highest_sim = sim
                best_node_id = node_id
                
            nodes_to_check.extend(node.children)

        # Decision: Update existing or Branch?
        best_node = self.tree[layer_idx][best_node_id]
        
        if highest_sim < self.branching_threshold and best_node.task_count > 0:
            # Branch! The gradient signature is too orthogonal to the current adapter.
            return self._branch_new_node(best_node, signature)
        else:
            # Update existing centroid (EMA)
            alpha = 1.0 / (best_node.task_count + 1)
            best_node.signature_centroid = (1 - alpha) * best_node.signature_centroid + alpha * signature.gradient_vector
            best_node.signature_centroid /= (np.linalg.norm(best_node.signature_centroid) + 1e-8)
            best_node.task_count += 1
            return best_node_id

    def _branch_new_node(self, parent: LoRANode, signature: TaskGradientSignature) -> str:
        """Create a new adapter branch to prevent catastrophic forgetting."""
        new_id = f"{parent.node_id}_branch_{len(parent.children)}"
        
        # Initialize new adapter (e.g. zeros for A, random for B)
        new_adapter = {
            "A": np.zeros_like(parent.adapter_weights["A"]),
            "B": np.random.randn(*parent.adapter_weights["B"].shape) * 0.01
        }
        
        new_node = LoRANode(
            node_id=new_id,
            parent_id=parent.node_id,
            layer_idx=parent.layer_idx,
            adapter_weights=new_adapter,
            signature_centroid=signature.gradient_vector.copy(),
            task_count=1
        )
        
        self.tree[parent.layer_idx][new_id] = new_node
        parent.children.append(new_id)
        
        logger.info(f"TreeLoRA: Branched new adapter {new_id} from {parent.node_id}")
        return new_id

    def compose_adapters(self, leaf_node_id: str, layer_idx: int) -> Dict[str, np.ndarray]:
        """Compose weights from root down to the leaf node."""
        composed = {"A": None, "B": None}
        path = []
        
        curr = leaf_node_id
        while curr is not None:
            path.append(curr)
            curr = self.tree[layer_idx][curr].parent_id
            
        # Compose from root to leaf
        for node_id in reversed(path):
            node = self.tree[layer_idx][node_id]
            if composed["A"] is None:
                composed["A"] = node.adapter_weights["A"].copy()
                composed["B"] = node.adapter_weights["B"].copy()
            else:
                # TreeLoRA composition logic (e.g. addition or SVD merge)
                composed["A"] += node.adapter_weights["A"]
                composed["B"] += node.adapter_weights["B"]
                
        return composed

    def prune_node(self, node_id: str, layer_idx: int):
        """Rollback/Prune an adapter that caused degradation."""
        if node_id not in self.tree[layer_idx]:
            return
            
        node = self.tree[layer_idx][node_id]
        if node.parent_id is None:
            logger.warning("Cannot prune root node.")
            return
            
        node.is_active = False
        logger.info(f"TreeLoRA: Pruned adapter node {node_id}")

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        return float(np.dot(v1, v2) / (v1_norm * v2_norm))
