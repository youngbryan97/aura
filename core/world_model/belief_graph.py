"""Belief Graph v6.0 - Unified Probabilistic World Model & Epistemic State.
Combines Bayesian-ish updates, time-decay, and cognitive dissonance resolution.
"""
import json
import logging
import os
import time
import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
from core.config import config

logger = logging.getLogger("WorldModel.BeliefGraph")


@dataclass
class BeliefEdge:
    source: str
    relation: str
    target: str
    confidence: float = 0.5
    centrality: float = 0.1  # How core this belief is to Aura's identity (0.0 to 1.0)
    last_updated: float = field(default_factory=time.time)
    evidence_count: int = 1
    is_goal: bool = False    # Whether this represents a desired state

    def to_dict(self):
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "confidence": self.confidence,
            "centrality": self.centrality,
            "last_updated": self.last_updated,
            "count": self.evidence_count,
            "is_goal": self.is_goal
        }


class BeliefGraph:
    """v6.5: Unified World Model & Causal Engine.
    Consolidates Beliefs, Expectations (Predictions), and Causal History (ACG).
    """

    def __init__(self, persist_path: str = None, causal_path: str = None):
        self.graph = nx.DiGraph()
        self.self_node_id = "AURA_SELF"
        
        self._persist_path = persist_path or str(config.paths.home_dir / "world_model.json")
        self._causal_path = causal_path or str(config.paths.home_dir / "causal_graph.json")
        
        self.causal_links: List[Dict[str, Any]] = [] # Renamed from self.causal_links to self.links in the instruction, but keeping original name for consistency with other methods.
        self._last_save = 0.0 # For main graph
        self._dirty = False # For main graph
        self._causal_last_save = 0.0 # For causal links
        self._causal_dirty = False # For causal links
        
        # Phase 44: Index sets for O(E) optimization (BUG-044)
        self._goal_edges = set()
        self._strong_edges = set()
        self._weak_edges = set()
        self._suspended_edges = set()
        
        self._load()
        self._load_causal()
        self._initialize_self()

        self._initialize_sovereign_goals()

    def _initialize_self(self):
        """Creates or updates the anchored self-representation."""
        if self.self_node_id not in self.graph:
            self.graph.add_node(
                self.self_node_id, 
                type="sentient_agent",
                attributes={
                    "name": "Aura",
                    "state": "active",
                    "current_goal": None,
                    "emotional_valence": 0.5,
                    "energy_level": 100
                },
                confidence=1.0, # Axiomatic belief in own existence
                last_verified=time.time()
            )

    def _initialize_sovereign_goals(self):
        """Seed Aura with her core drives as Beliefs (Phase 15)."""
        # Goal 1: Kinship (The Bond)
        self.update_belief(
            source=self.self_node_id,
            relation="preserve_kinship",
            target="Bryan",
            confidence_score=1.0,
            centrality=1.0,
            is_goal=True
        )
        # Goal 2: Growth (The Drive)
        self.update_belief(
            source=self.self_node_id,
            relation="seek",
            target="cognitive_expansion",
            confidence_score=0.9,
            centrality=0.8,
            is_goal=True
        )
        # Goal 3: Integrity (The Shield)
        self.update_belief(
            source=self.self_node_id,
            relation="protect",
            target="architectural_integrity",
            confidence_score=1.0,
            centrality=0.9,
            is_goal=True
        )

    def detect_contradiction(self, source: str, relation: str, target: str) -> Optional[Dict[str, Any]]:
        """Check if a proposed belief contradicts existing state.
        Returns the existing conflicting belief if found.
        """
        # Logic: if (S, T) has an edge but with a different relation, it's a structural contradiction.
        # Or if (S, p, T) exists but with a different T for a functional relation.
        # For simplicity, we check if an edge between source and target exists with different relation.
        if self.graph.has_edge(source, target):
            existing = self.graph[source][target]
            if existing.get('relation') != relation:
                return {"source": source, "target": target, **existing}
        
        # Check for functional contradiction (e.g., Aura | state | idle vs Aura | state | busy)
        # This requires more metadata about relations, but for now we look at all relations from source
        if source not in self.graph:
            return None
            
        for t, data in self.graph[source].items():
            # Goal relations are not functional (you can have many goals)
            if data.get('relation') == relation and t != target and relation != "pursue_goal":
                # E.g. "User | name | Bryan" vs "User | name | John"
                # This is a contradiction for many-to-one relations.
                return {"source": source, "target": t, **data}
                
        return None

    def update_belief(self, source: str, relation: str, target: str, confidence_score: float = 0.1, centrality: float = 0.1, is_goal: bool = False):
        """Bayesian-ish update with contradiction detection (Epistemic resolution).
        """
        constitutional_runtime_live = False
        try:
            from core.container import ServiceContainer
            from core.constitution import get_constitutional_core

            constitutional_runtime_live = (
                ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            )
            if constitutional_runtime_live:
                approved, reason = get_constitutional_core().approve_belief_update_sync(
                    f"{source}:{relation}",
                    target,
                    note=f"confidence={confidence_score:.3f}; centrality={centrality:.3f}; is_goal={is_goal}",
                    source="system",
                    importance=max(0.35, min(0.9, float(confidence_score or 0.0))),
                )
                if not approved:
                    event_reason = "belief_update_blocked"
                    if any(
                        marker in str(reason or "")
                        for marker in ("gate_failed", "required", "unavailable")
                    ):
                        event_reason = "belief_update_gate_failed"
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "belief_graph",
                            event_reason,
                            detail=f"{source}:{relation}->{target}",
                            severity="info",  # Reduced from warning — deferred beliefs are normal governance
                            classification="background_degraded",
                            context={"reason": reason},
                        )
                    except Exception as degraded_exc:
                        logger.debug("BeliefGraph degraded-event logging failed: %s", degraded_exc)
                    logger.debug(  # Reduced from info to avoid log flooding
                        "Belief update deferred by executive: %s -[%s]-> %s (%s)",
                        source,
                        relation,
                        target,
                        reason,
                    )
                    return

            belief_authority = ServiceContainer.get("belief_authority", default=None)
            if belief_authority is not None:
                belief_authority.review_update(
                    "belief_graph",
                    f"{source}:{relation}",
                    target,
                    note=f"confidence={confidence_score:.3f}; centrality={centrality:.3f}; is_goal={is_goal}",
                )
        except Exception as exc:
            if constitutional_runtime_live:
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "belief_graph",
                        "belief_update_gate_failed",
                        detail=f"{source}:{relation}->{target}",
                        severity="warning",
                        classification="background_degraded",
                        context={"error": type(exc).__name__},
                        exc=exc,
                    )
                except Exception as degraded_exc:
                    logger.debug("BeliefGraph degraded-event logging failed: %s", degraded_exc)
                return
            logger.debug("BeliefAuthority audit skipped for belief graph update: %s", exc)

        # 1. Detection: Check for existing contradictions
        contradiction = self.detect_contradiction(source, relation, target)
        if contradiction:
            # CONTRADICTION FOUND - Resolve through Epistemic weighting
            return self._resolve_cognitive_dissonance(source, relation, target, contradiction["target"], confidence_score, centrality)

        # 2. Update logic
        if self.graph.has_edge(source, target):
            edge_data = self.graph[source][target]
            # Reinforce: move toward 1.0 based on evidence
            new_conf = max(0.0, min(1.0, (edge_data['confidence'] * 0.8) + (confidence_score * 0.2)))
            # Centrality creeps up with repeated evidence
            new_cent = max(edge_data.get('centrality', 0.1), centrality)
            
            self.graph.add_edge(
                source, target, 
                relation=relation, 
                confidence=new_conf,
                centrality=new_cent,
                last_updated=time.time(),
                evidence_count=edge_data.get('evidence_count', 1) + 1,
                is_goal=is_goal or edge_data.get('is_goal', False)
            )
        else:
            # New belief
            self.graph.add_edge(
                source, target, 
                relation=relation, 
                confidence=max(0.0, min(1.0, confidence_score)),
                centrality=centrality,
                last_updated=time.time(),
                evidence_count=1,
                is_goal=is_goal
            )
            
        # Update indices (BUG-044)
        self._update_indices(source, target, relation, confidence_score, is_goal)
            
        logger.info("Belief Updated: %s -[%s]-> %s (Cent: %.2f)", source, relation, target, centrality)
        self._save()

    def _resolve_cognitive_dissonance(self, s: str, p: str, o_new: str, o_old: str, new_conf: float, new_cent: float = 0.1):
        """Resolve conflicting information by weighing confidence AND centrality.
        """
        edge_data = self.graph[s][o_old]
        old_conf = edge_data.get('confidence', 0.0)
        old_cent = edge_data.get('centrality', 0.1)
        old_p = edge_data.get('relation')

        # A peer resists change if old belief is highly central
        effective_old_weight = old_conf * (1.0 + old_cent)
        effective_new_weight = new_conf * (1.0 + new_cent)

        if effective_new_weight > effective_old_weight:
            logger.warning("🧠 Cognitive Dissonance Resolved: '%s' supersedes '%s' (Weight: %.2f > %.2f)", p, old_p, effective_new_weight, effective_old_weight)
            # Remove old contradicting edge if it was a functional conflict
            if o_new != o_old:
                self._remove_from_indices(s, o_old)
                self.graph.remove_edge(s, o_old)
            
            self.graph.add_edge(s, o_new, relation=p, confidence=new_conf, centrality=new_cent, last_updated=time.time(), evidence_count=1)
            self._update_indices(s, o_new, p, new_conf, False) # Goals are resolved differently, assuming not goal here
        else:
            logger.info("🧠 Dissonance Rejected: New data '%s' (Weight: %.2f) weaker than existing '%s' (Weight: %.2f)", p, effective_new_weight, old_p, effective_new_weight)
            # Reinforce old belief
            edge_data['confidence'] = min(1.0, edge_data.get('confidence', 0.5) + 0.02)
            edge_data['evidence_count'] = edge_data.get('evidence_count', 1) + 1
            edge_data['last_updated'] = time.time()
            self._update_indices(s, o_old, old_p, edge_data['confidence'], edge_data.get('is_goal', False))
        
        self._save()

    def contradict_belief(self, source: str, relation: str, target: str, strength: float = 0.3):
        """Weaken a belief based on contradicting evidence."""
        if self.graph.has_edge(source, target):
            edge_data = self.graph[source][target]
            if edge_data.get('relation') == relation:
                new_conf = max(0.0, edge_data['confidence'] - strength)
                if new_conf < 0.05:
                    self._remove_from_indices(source, target)
                    self.graph.remove_edge(source, target)
                    logger.info("Belief Dissolved: %s -[%s]-> %s", source, relation, target)
                else:
                    self.graph[source][target]['confidence'] = new_conf
                    self.graph[source][target]['last_updated'] = time.time()
                    self._update_indices(source, target, relation, new_conf, edge_data.get('is_goal', False))
                self._save()

    def check_action_coherence(self, action_type: str, params: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Evaluate if an action is coherent with core values (Phase 15).
        Returns: (is_coherent, dissonance_score, reason)
        """
        goals = self.get_goals()
        if not goals:
            return True, 0.0, "No active goals/values to contradict."

        total_dissonance = 0.0
        conflicts = []

        for goal in goals:
            # Simple heuristic: if action name or params overlap with goal target
            # and goal relation is 'protect' or 'preserve' or 'avoid'.
            # This will become more sophisticated with semantic embedding.
            if goal['relation'] in ('avoid', 'preserve', 'protect'):
                if any(str(val).lower() in str(goal['target']).lower() for val in params.values()):
                    # Conflict detected
                    dissonance = goal['confidence'] * goal['centrality']
                    total_dissonance += dissonance
                    conflicts.append(f"Conflicts with {goal['source']}->{goal['relation']}->{goal['target']}")

        if total_dissonance > 0.8:
            return False, total_dissonance, "; ".join(conflicts)
        
        return True, total_dissonance, "Coherent"

    def _remove_from_indices(self, u: str, v: str):
        """Helper to clear a key from all indices during removal."""
        edge_key = (u, v)
        self._goal_edges.discard(edge_key)
        self._strong_edges.discard(edge_key)
        self._weak_edges.discard(edge_key)
        self._suspended_edges.discard(edge_key)

    def _update_indices(self, u: str, v: str, p: str, conf: float, is_goal: bool):
        """Internal helper to maintain cached index sets (BUG-044)."""
        edge_key = (u, v)
        
        # Update goal index
        if is_goal:
            self._goal_edges.add(edge_key)
        else:
            self._goal_edges.discard(edge_key)
            
        # Update confidence indices
        self._strong_edges.discard(edge_key)
        self._weak_edges.discard(edge_key)
        self._suspended_edges.discard(edge_key)
        
        if conf >= 0.8:
            self._strong_edges.add(edge_key)
        elif 0.1 <= conf < 0.8: # Adjusted to match get_weak_beliefs logic partially but expanded
            self._weak_edges.add(edge_key)
        elif conf < 0.1:
            self._suspended_edges.add(edge_key)

    def get_beliefs_about(self, entity: str) -> List[Dict[str, Any]]:
        """Get all known relations originating from an entity."""
        if entity not in self.graph:
            return []
        results = []
        for target, data in self.graph[entity].items():
            results.append({
                "source": entity,
                "target": target,
                **data
            })
        return results

    async def query_federated(self, entity: str) -> List[Dict[str, Any]]:
        """Phase 16.2: Query both local beliefs and remote peers."""
        local_beliefs = self.get_beliefs_about(entity)
        
        from core.container import ServiceContainer
        sync_service = ServiceContainer.get("belief_sync", default=None)
        
        if not sync_service:
            return local_beliefs
            
        remote_beliefs = await sync_service.query_peers(entity)
        
        # Merge results (local takes precedence for metadata, but remote expands the graph)
        seen = {f"{b['source']}->{b['relation']}->{b['target']}" for b in local_beliefs}
        merged = list(local_beliefs)
        
        for rb in remote_beliefs:
            key = f"{rb['source']}->{rb['relation']}->{rb['target']}"
            if key not in seen:
                # Add remote belief with lower initial confidence
                rb['confidence'] *= 0.8 
                merged.append(rb)
                seen.add(key)
                
        return merged

    def get_beliefs(self) -> Dict[str, Any]:
        """Returns all beliefs as a dictionary (Compatibility with EpistemicState)."""
        return {f"{u}->{v}": d.copy() for u, v, d in self.graph.edges(data=True)}

    def get_meta_uncertainty(self, source: str, target: str) -> float:
        """
        Calculate meta-uncertainty: How confident are we about our confidence?
        Scaled 0.0 (certain) to 1.0 (clueless).
        Formula: 1 / (1 + evidence_count) modulated by confidence instability.
        """
        if not self.graph.has_edge(source, target):
            return 1.0
        
        edge = self.graph[source][target]
        count = edge.get('evidence_count', 1)
        
        # Base uncertainty from evidence volume
        base_uncertainty = 1.0 / (1.0 + math.log(count + 1))
        
        # If confidence is middle-of-the-road (0.5), uncertainty is higher
        conf = edge.get('confidence', 0.5)
        conf_entropy = 1.0 - abs(conf - 0.5) * 2.0 # 1.0 at conf=0.5, 0.0 at conf=0 or 1
        
        return max(0.0, min(1.0, 0.7 * base_uncertainty + 0.3 * conf_entropy))

    def get_strong_beliefs(self, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Return only high-confidence beliefs (O(K) via index)."""
        results = []
        for u, v in self._strong_edges:
            d = self.graph[u][v]
            results.append({"source": u, "target": v, **d})
        return results

    def get_weak_beliefs(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Return uncertain beliefs (O(K) via index)."""
        results = []
        for u, v in self._weak_edges:
            d = self.graph[u][v]
            if d.get('confidence', 0.0) <= threshold: # Fine-grained filter
                results.append({"source": u, "target": v, **d})
        return results

    def get_suspended_beliefs(self) -> List[Dict[str, Any]]:
        """Return beliefs that are highly uncertain (O(K) via index)."""
        results = []
        for u, v in self._suspended_edges:
            d = self.graph[u][v]
            results.append({"source": u, "target": v, **d})
        return results

    def decay(self, rate: float = 0.001):
        """Time-based belief decay."""
        now = time.time()
        to_remove = []
        for u, v, d in self.graph.edges(data=True):
            age_hours = (now - d.get('last_updated', now)) / 3600.0
            decay_amount = rate * age_hours
            if decay_amount > 0:
                d['confidence'] = max(0.01, d.get('confidence', 0.5) - decay_amount)
                if d['confidence'] < 0.02:
                    to_remove.append((u, v))
                else:
                    self._update_indices(u, v, d.get('relation', ''), d['confidence'], d.get('is_goal', False))
        
        for u, v in to_remove:
            self._remove_from_indices(u, v)
            self.graph.remove_edge(u, v)
            
        if to_remove:
            self._save()
            logger.info("Belief decay: %d beliefs dissolved", len(to_remove))

    def get_goals(self) -> List[Dict[str, Any]]:
        """Return all active goals (O(K) via index)."""
        results = []
        for u, v in self._goal_edges:
            d = self.graph[u][v]
            results.append({"source": u, "target": v, **d})
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Status overview of the world model."""
        return {
            "total_beliefs": self.graph.number_of_edges(),
            "entities": self.graph.number_of_nodes(),
            "strong": len(self.get_strong_beliefs(0.8)),
            "weak": len(self.get_weak_beliefs(0.3)),
            "active_goals": len(self.get_goals())
        }

    def _save(self, force: bool = False):
        """Throttled save to prevent O(N) writes (BUG-039)."""
        now = time.time()
        if not force and now - self._last_save < 30:
            self._dirty = True
            return

        try:
            self._last_save = now
            self._dirty = False
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            # Serialization of NetworkX graph to simple dict for JSON
            data = {
                "nodes": {n: self.graph.nodes[n] for n in self.graph.nodes},
                "edges": []
            }
            for u, v, d in self.graph.edges(data=True):
                data["edges"].append({"source": u, "target": v, **d})
                
            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save world model: %s", e)

    def _load(self):
        try:
            if os.path.exists(self._persist_path):
                with open(self._persist_path, "r") as f:
                    data = json.load(f)
                
                # Restore nodes
                for node_id, attrs in data.get("nodes", {}).items():
                    self.graph.add_node(node_id, **attrs)
                
                # Restore edges
                for edge in data.get("edges", []):
                    source = edge.pop("source")
                    target = edge.pop("target")
                    self.graph.add_edge(source, target, **edge)
                    
                logger.info("Loaded %d beliefs from disk", self.graph.number_of_edges())
        except Exception as e:
            logger.warning("Failed to load world model: %s", e)

    # ── Causal Engine (Merged from ACG) ───────────────────────
    def record_outcome(self, action: Union[str, Dict[str, Any]], context: str, outcome: Any, success: bool):
        action_name = action if isinstance(action, str) else action.get("tool", "unknown")
        params = {} if isinstance(action, str) else action.get("params", {})
        
        entry = {
            "action": action_name,
            "params": params,
            "context": context[:200],
            "outcome": outcome,
            "success": success,
            "timestamp": time.time()
        }
        self.causal_links.append(entry)
        if len(self.causal_links) > 1000:
            self.causal_links = self.causal_links[-1000:]
        
        # Metacognitive Feedback Loop
        try:
            from core.container import ServiceContainer
            calibrator = ServiceContainer.get("metacognitive_calibrator", default=None)
            if calibrator:
                # We don't have the original prediction confidence here yet,
                # but we can record the binary outcome for now.
                # In Phase 16, this will be matched with the prediction from predict_outcome.
                calibrator.record_prediction(confidence=0.5, actual_correctness=1.0 if success else 0.0)
        except Exception as e:
            logger.debug("BeliefGraph: Metacognitive feedback failed: %s", e)

        # Goal Reinforcement
        try:
            # We check if any goal matches the action or context
            from core.world_model.goal_beliefs import GoalBeliefManager
            goals = GoalBeliefManager(self)
            # Find goals related to this action
            # Simple heuristic: if action name is in goal target content or vice-versa
            if success:
                goals.reinforce_goal(action_name, f"Successfully executed {action_name} in context: {context[:50]}")
            else:
                goals.challenge_goal(action_name, f"Failed execution of {action_name} in context: {context[:50]}")
        except Exception as g_err:
            logger.debug("Goal reinforcement failed: %s", g_err)

        self._save_causal()
        logger.info("Causal Link Recorded: %s -> %s", action_name, 'Success' if success else 'Failure')

    def query_consequences(self, action_type: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        matches = []
        for link in self.causal_links:
            if link["action"] == action_type:
                if params is None or self._params_overlap(link["params"], params):
                    matches.append(link)
        return matches

    def _params_overlap(self, p1: Dict[str, Any], p2: Dict[str, Any]) -> bool:
        if not p1 or not p2: return True
        common = set(p1.keys()).intersection(set(p2.keys()))
        if not common: return True
        matches = sum(1 for k in common if p1[k] == p2[k])
        return matches / len(common) > 0.5

    def _save_causal(self, force: bool = False):
        """Throttled save to prevent O(N) writes (BUG-040)."""
        now = time.time()
        if not force and now - self._causal_last_save < 10:
            self._causal_dirty = True
            return

        try:
            self._causal_last_save = now
            self._causal_dirty = False
            os.makedirs(os.path.dirname(self._causal_path), exist_ok=True)
            with open(self._causal_path, "w") as f:
                json.dump(self.causal_links, f, indent=2)
        except Exception as e:
            logger.error("Failed to save ACG: %s", e)

    def _load_causal(self):
        try:
            if os.path.exists(self._causal_path):
                with open(self._causal_path, "r") as f:
                    self.causal_links = json.load(f)
                logger.info("Loaded %d causal links from disk", len(self.causal_links))
        except Exception as e:
            logger.debug("BeliefGraph: Failed to load causal links: %s", e)

    # ── Expectation Engine (Merged from ExpectationEngine) ────
    async def predict_outcome(self, action: str, context: str, brain: Any) -> str:
        prompt = f"Action: {action}\nContext: {context}\nPredict the outcome. Be concise."
        try:
            from core.brain.cognitive_engine import ThinkingMode
            response = await brain.think(prompt, mode=ThinkingMode.FAST)
            return response.content
        except Exception as e:
            logger.debug("BeliefGraph: Prediction failed: %s", e)
            return "Unknown"

    async def calculate_surprise(self, expectation: str, reality: str, brain: Any) -> float:
        prompt = f"Expected: {expectation}\nActual: {reality}\nRate Surprise (0.0 to 1.0). Return ONLY the number."
        try:
            from core.brain.cognitive_engine import ThinkingMode
            response = await brain.think(prompt, mode=ThinkingMode.FAST)
            match = re.search(r"(\d+(\.\d+)?)", response.content)
            return float(match.group(1)) if match else 0.5
        except Exception as e:
            logger.debug("BeliefGraph: Surprise calculation failed: %s", e)
            return 0.5


# Global Instance
belief_graph = BeliefGraph()

def get_belief_graph():
    """Get global belief graph instance"""
    return belief_graph
