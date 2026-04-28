"""core/brain/causal_world_model.py — Causal Inference & Simulation
===================================================================
Aura's predictive engine. Instead of just reacting to the current state, 
the Causal World Model allows Aura to run "What if" counterfactuals before
executing a plan.

It models the digital environment as a probabilistic causal graph:
- Nodes: Entities, states, or concepts (e.g., 'API Latency', 'User Mood')
- Edges: Causal relationships ('High API Latency' -> 'User Frustration' with p=0.8)

This is core to Phase 22.8, enabling proactive planning over reactive scripting.
"""

from core.runtime.errors import record_degradation
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from core.container import ServiceContainer

logger = logging.getLogger("Aura.CausalWorldModel")

@dataclass
class CausalNode:
    """A variable in Aura's world model."""
    name: str
    activation: float = 0.0  # Current active state (0.0 to 1.0)
    variance: float = 0.1    # Uncertainty of this state

@dataclass
class CausalEdge:
    """A directional causal link: A -> B."""
    source: str
    target: str
    relationship: str = "causes"  # "causes", "correlates_with", "contradicts", "enables"
    weight: float = 1.0           # -1.0 (inhibits) to 1.0 (excites)
    confidence: float = 1.0       # 0.0 (guess) to 1.0 (proven fact)
    observations: int = 1
    last_confirmed: float = field(default_factory=time.time)
    last_disconfirmed: Optional[float] = None

class CausalWorldModel:
    """The counterfactual simulation engine."""
    name = "causal_world_model"

    def __init__(self):
        from core.config import config
        self.data_path = config.paths.data_dir / "causal_world.json"
        
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        
        self._load()

    def get_node(self, name: str) -> CausalNode:
        """Fetch or lazily create a node."""
        name = name.lower()
        if name not in self.nodes:
            self.nodes[name] = CausalNode(name=name)
        return self.nodes[name]

    def add_observation(self, source: str, target: str, correlation: float):
        """
        Record a real-world observation. 
        Updates or creates the causal link between source and target.
        """
        source = source.lower()
        target = target.lower()
        
        # Ensure nodes exist
        self.get_node(source)
        self.get_node(target)
        
        edge = next((e for e in self.edges if e.source == source and e.target == target), None)
        if not edge:
            edge = CausalEdge(source=source, target=target, weight=correlation, confidence=0.1, observations=1)
            self.edges.append(edge)
        else:
            # Moving average of weight based on new observation
            alpha = 1.0 / (edge.observations + 1)
            edge.weight = (1 - alpha) * edge.weight + (alpha * correlation)
            edge.observations += 1
            # Confidence grows asymptotically with observations
            edge.confidence = 1.0 - math.exp(-0.1 * edge.observations)
            edge.last_confirmed = time.time()
            
        self._save()

    def disconfirm(self, source: str, target: str) -> None:
        """A prediction based on this edge failed. Weaken it."""
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                edge.weight *= 0.8  # Decay on disconfirmation
                edge.last_disconfirmed = time.time()
                self._save()
                return

    def predict_effects(self, source_id: str) -> list[tuple[str, float]]:
        """Given a cause, what effects does the model predict?"""
        predictions = []
        for edge in self.edges:
            if edge.source == source_id and edge.weight > 0.3:
                predictions.append((edge.target, edge.weight))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def simulate(self, interventions: Dict[str, float], steps: int = 3) -> Dict[str, float]:
        """
        Run a counterfactual simulation. 
        'What if I force node X to activation Y?'
        
        Args:
            interventions: A dict targeting nodes with specific activations.
            steps: Number of causal propagation steps to simulate.
            
        Returns:
            The predicted final state of all nodes.
        """
        # Create a temporary sandbox state
        state = {name: node.activation for name, node in self.nodes.items()}
        
        # Apply interventions
        for node_name, val in interventions.items():
            state[node_name.lower()] = max(0.0, min(1.0, val))
            
        # Propagate causal influence
        for _ in range(steps):
            next_state = state.copy()
            for edge in self.edges:
                if edge.source in state:
                    source_val = state[edge.source]
                    # Influence = source_activation * edge_weight * edge_confidence
                    influence = source_val * edge.weight * edge.confidence
                    
                    target_val = next_state.get(edge.target, 0.0)
                    # Merge influence via simple logistic-style bounding
                    new_val = max(0.0, min(1.0, target_val + influence))
                    next_state[edge.target] = new_val
            state = next_state
            
        return state

    def analyze_preventative_actions(self, undesirable_node: str) -> List[Tuple[str, float]]:
        """
        Reverse-traverse the graph to find nodes that negatively influence the undesirable node.
        Used by the planner to figure out how to avoid a bad outcome.
        """
        undesirable_node = undesirable_node.lower()
        preventers = []
        for edge in self.edges:
            if edge.target == undesirable_node and edge.weight < -0.2 and edge.confidence > 0.3:
                preventers.append((edge.source, edge.weight))
                
        # Sort by strongest preventative weight (most negative)
        return sorted(preventers, key=lambda x: x[1])

    def get_prompt_context(self) -> str:
        """Returns the strongest proven causal rules for prompt injection."""
        strong_rules = [e for e in self.edges if e.confidence > 0.7 and abs(e.weight) > 0.5]
        if not strong_rules:
            return ""
            
        rules_text = []
        for e in sorted(strong_rules, key=lambda x: x.confidence, reverse=True)[:5]:
            effect = "INCREASES" if e.weight > 0 else "DECREASES"
            rules_text.append(f"- [{e.source}] {effect} [{e.target}] (Confidence: {e.confidence:.2f})")
            
        return "\n### ESTABLISHED WORLD CASCADES\n" + "\n".join(rules_text) + "\n"

    def _save(self):
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "nodes": {k: asdict(v) for k, v in self.nodes.items()},
                "edges": [asdict(e) for e in self.edges]
            }
            with open(self.data_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            record_degradation('causal_world_model', e)
            logger.error(f"Failed to save Causal World Model: {e}")

    def _load(self):
        if not self.data_path.exists():
            # Seed with some baseline digital intuition
            self.add_observation("high cpu usage", "system lag", 0.9)
            self.add_observation("sandbox violation", "orchestrator crash", 0.95)
            self.add_observation("unclear prompt", "hallucination", 0.7)
            self.add_observation("deep dreaming", "memory consolidation", 0.8)
            return
            
        try:
            with open(self.data_path, "r") as f:
                data = json.load(f)
                
            self.nodes = {k: CausalNode(**v) for k, v in data.get("nodes", {}).items()}
            self.edges = [CausalEdge(**v) for v in data.get("edges", [])]
        except Exception as e:
            record_degradation('causal_world_model', e)
            logger.error(f"Failed to load Causal World Model: {e}")

def register_causal_world_model(orchestrator=None):
    model = CausalWorldModel()
    ServiceContainer.register_instance("causal_world_model", model)
    return model
