"""core/brain/causal_world_model.py — Structural Causal Model (SCM)
===================================================================
Aura's predictive engine. Instead of just tracking correlations, this
implements a FOCUS-style Structural Causal Model (SCM) with true
intervention-based causal discovery (do-calculus).

It allows Aura to:
1. Track observational correlations.
2. Perform active interventions (do-calculus) to discover true causal edges.
3. Answer counterfactual "what if I break this correlation?" queries.

This enables proactive planning over reactive scripting.
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
    relationship: str = "correlates_with"  # Upgraded to "causes" upon intervention
    weight: float = 1.0           # -1.0 (inhibits) to 1.0 (excites)
    confidence: float = 1.0       # 0.0 (guess) to 1.0 (proven fact)
    observations: int = 1
    intervention_count: int = 0   # Number of times do(source) was tested
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

    def discover_causality_via_intervention(self, source: str, target: str, source_val: float, target_val_observed: float) -> None:
        """
        FOCUS-style do-calculus intervention.
        We actively forced `do(source = source_val)` and observed `target_val_observed`.
        This proves true causality vs mere correlation.
        """
        source = source.lower()
        target = target.lower()
        
        # Ensure nodes exist in the graph
        self.get_node(source)
        self.get_node(target)
        
        edge = next((e for e in self.edges if e.source == source and e.target == target), None)
        if not edge:
            edge = CausalEdge(source=source, target=target, weight=0.0, confidence=0.0)
            self.edges.append(edge)
            
        edge.relationship = "causes"  # Upgraded from mere correlation
        edge.intervention_count += 1
        
        # Interventions provide massively higher confidence than passive observation
        alpha = 1.0 / edge.intervention_count
        implied_weight = target_val_observed if source_val > 0.5 else -target_val_observed
        edge.weight = (1 - alpha) * edge.weight + (alpha * implied_weight)
        
        # Max out confidence quickly on direct intervention
        edge.confidence = min(1.0, edge.confidence + 0.3)
        edge.last_confirmed = time.time()
        self._save()

    def simulate_counterfactual(self, do_interventions: Dict[str, float], steps: int = 3) -> Dict[str, float]:
        """
        Run a true counterfactual SCM simulation using do-calculus.
        'What if I break the correlation and force do(X=x)?'
        
        Unlike passive simulation, this explicitly severs inbound causal edges to
        the intervened nodes (Pearl's graph surgery).
        """
        # Create state
        state = {name: node.activation for name, node in self.nodes.items()}
        
        # Apply do-calculus graph surgery (intervened nodes cannot be changed by parents)
        intervened_nodes = {k.lower(): v for k, v in do_interventions.items()}
        for k, v in intervened_nodes.items():
            state[k] = max(0.0, min(1.0, v))
            
        for _ in range(steps):
            next_state = state.copy()
            for edge in self.edges:
                # SCM Surgery: If the target is intervened upon, ignore inbound edges
                if edge.target in intervened_nodes:
                    continue
                    
                if edge.source in state and edge.relationship == "causes":
                    influence = state[edge.source] * edge.weight * edge.confidence
                    target_val = next_state.get(edge.target, 0.0)
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('causal_world_model', e)
            logger.error("Failed to save Causal World Model: %s", e)

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
        except (httpx.HTTPError, OSError, ConnectionError, TimeoutError) as e:
            record_degradation('causal_world_model', e)
            logger.error("Failed to load Causal World Model: %s", e)

def register_causal_world_model(orchestrator=None):
    model = CausalWorldModel()
    ServiceContainer.register_instance("causal_world_model", model)
    return model
