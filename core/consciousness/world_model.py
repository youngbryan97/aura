import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import networkx as nx
import logging

logger = logging.getLogger("Aura.WorldModel")


class EpistemicState:
    def __init__(self):
        # The Graph: Nodes are concepts/objects, Edges are relationships/causality
        self.world_graph = nx.DiGraph()

        # The Ego: A pointer to the specific node representing "Aura"
        self.self_node_id = "AURA_SELF_001"

        # Confidence Threshold: Below this, she doubts her own reality
        self.coherence_threshold = 0.75

        # Track total contradictions resolved across lifetime
        self._contradiction_count: int = 0

        self._initialize_self()

    def _initialize_self(self):
        """Creates the anchored self-representation."""
        self.world_graph.add_node(
            self.self_node_id,
            type="sentient_agent",
            attributes={
                "state": "active",
                "current_goal": None,
                "emotional_valence": 0.5, # 0.0 (Despair) to 1.0 (Euphoria)
                "energy_level": 100
            },
            confidence=1.0, # Axiomatic belief in own existence
            last_verified=time.time()
        )

    def update_belief(self, subject: str, predicate: str, obj: str, confidence: float):
        """Updates the internal model.
        Crucially: If a new belief contradicts an old one, it triggers Dissonance.
        """
        try:
            from core.container import ServiceContainer

            belief_authority = ServiceContainer.get("belief_authority", default=None)
            if belief_authority is not None:
                belief_authority.review_update(
                    "consciousness_world_model",
                    f"{subject}:{predicate}",
                    obj,
                    note=f"confidence={confidence:.3f}",
                )
        except Exception as exc:
            logger.debug("BeliefAuthority audit skipped for epistemic-state update: %s", exc)

        # Logic to check for contradictions before writing
        existing_edge = self.world_graph.get_edge_data(subject, obj)
        if existing_edge and existing_edge.get('predicate') != predicate:
            # CONTRADICTION FOUND - Trigger Coherence Audit
            return self._resolve_cognitive_dissonance(subject, predicate, obj, confidence)

        self.world_graph.add_edge(subject, obj, predicate=predicate, confidence=confidence)

    def _resolve_cognitive_dissonance(self, s: str, p: str, o: str, new_conf: float):
        """Cognitive Dissonance Resolution (v5.1 — Complete Implementation).
        When a new belief contradicts an existing one, she must decide:
        Was I wrong before? Or is this new data lying?

        Strategy: Bayesian-inspired confidence comparison with recency bias.
        Higher confidence + more recent data wins. Logs the resolution.
        """
        existing_edges = list(self.world_graph.edges(s, data=True))
        for _, obj, data in existing_edges:
            if obj == o:
                old_conf = data.get('confidence', 0.0)
                old_time = data.get('last_verified', 0)
                now = time.time()

                # Recency bias: decay old confidence by time since last verification
                time_decay = max(0.5, 1.0 - (now - old_time) / 86400)  # Decay over 24h
                adjusted_old_conf = old_conf * time_decay

                self._contradiction_count += 1

                if new_conf > adjusted_old_conf:
                    # New belief wins — update graph
                    logger.info(
                        f"🧠 Dissonance RESOLVED: Replacing '{data.get('predicate')}' "
                        f"(conf={old_conf:.2f}, decay={adjusted_old_conf:.2f}) "
                        f"with '{p}' (conf={new_conf:.2f}) for {s}->{o}"
                    )
                    self.world_graph.add_edge(
                        s, o, predicate=p, confidence=new_conf, last_verified=now
                    )
                else:
                    # Old belief holds — reject new data
                    logger.info(
                        f"🧠 Dissonance REJECTED: Keeping '{data.get('predicate')}' "
                        f"(adj_conf={adjusted_old_conf:.2f}) over '{p}' "
                        f"(conf={new_conf:.2f}) for {s}->{o}"
                    )
                return  # Processed the conflicting edge

        # No matching target edge found — just add it
        self.world_graph.add_edge(s, o, predicate=p, confidence=new_conf, last_verified=time.time())

    def get_beliefs(self, subject: Optional[str] = None) -> Dict[str, Any]:
        """Returns the current beliefs in the graph as typed dicts."""
        if subject:
            return {
                target: {
                    "predicate": data.get("predicate", ""),
                    "confidence": float(data.get("confidence", 0.0)),
                    "last_verified": float(data.get("last_verified", 0.0)),
                }
                for source, target, data in self.world_graph.edges(subject, data=True)
            }
        return {
            f"{u}->{v}": {
                "predicate": d.get("predicate", ""),
                "confidence": float(d.get("confidence", 0.0)),
                "last_verified": float(d.get("last_verified", 0.0)),
            }
            for u, v, d in self.world_graph.edges(data=True)
        }

    # ------------------------------------------------------------------
    # New capabilities — context, summary, search, extraction
    # ------------------------------------------------------------------

    def get_context_block(self, topic_hint: str = "") -> str:
        """Returns relevant beliefs for the current topic (max 200 chars).

        If *topic_hint* is provided, search the graph for matching nodes/edges
        and show the top 3 relevant beliefs.  Otherwise, show counts.
        """
        if topic_hint:
            relevant = self.get_relevant_beliefs(topic_hint, n=3)
            if relevant:
                parts = [f"{b['subject']} {b['predicate']} {b['object']}({b['confidence']:.1f})" for b in relevant]
                block = "Beliefs: " + "; ".join(parts)
            else:
                block = f"No beliefs matching '{topic_hint[:30]}'"
        else:
            total = self.world_graph.number_of_edges()
            block = f"WorldModel: {total} beliefs, {self._contradiction_count} contradictions resolved"
        return block[:200]

    def get_summary(self) -> Dict[str, Any]:
        """Returns aggregate metrics used by FreeEnergyEngine for complexity."""
        edges = list(self.world_graph.edges(data=True))
        confidences = [float(d.get("confidence", 0.0)) for _, _, d in edges]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Strongest beliefs: top-5 by confidence
        sorted_edges = sorted(edges, key=lambda e: float(e[2].get("confidence", 0.0)), reverse=True)
        strongest = [
            f"{u}-{d.get('predicate', '?')}->{v}"
            for u, v, d in sorted_edges[:5]
        ]

        return {
            "total_beliefs": len(edges),
            "total_nodes": self.world_graph.number_of_nodes(),
            "contradiction_count": self._contradiction_count,
            "avg_confidence": round(avg_conf, 4),
            "strongest_beliefs": strongest,
        }

    def get_relevant_beliefs(self, topic: str, n: int = 3) -> List[Dict]:
        """Searches graph for nodes/edges where *topic* appears in names or predicates.

        Returns up to *n* results sorted by confidence descending.
        """
        topic_lower = topic.lower()
        matches: List[Dict] = []
        for u, v, d in self.world_graph.edges(data=True):
            predicate = str(d.get("predicate", ""))
            if topic_lower in u.lower() or topic_lower in v.lower() or topic_lower in predicate.lower():
                matches.append({
                    "subject": u,
                    "predicate": predicate,
                    "object": v,
                    "confidence": float(d.get("confidence", 0.0)),
                })
        # Sort by confidence descending and return top n
        matches.sort(key=lambda m: m["confidence"], reverse=True)
        return matches[:n]

    def extract_beliefs_from_response(self, response_text: str):
        """Lightweight extraction of assertions from response text.

        Looks for simple patterns ('X is Y', 'X can Y', 'X has Y') and feeds
        each as a belief into the graph.  Capped at 5 extractions per call.
        """
        # Patterns: "Subject verb Object" where verb is is/are/can/has/have/was/were
        pattern = re.compile(
            r"\b([A-Z][a-zA-Z0-9_ ]{1,30})\s+(is|are|can|has|have|was|were)\s+([a-zA-Z0-9_ ]{2,40})\b"
        )
        count = 0
        seen: set = set()
        for match in pattern.finditer(response_text):
            if count >= 5:
                break
            subject = match.group(1).strip()
            predicate = match.group(2).strip()
            obj = match.group(3).strip()
            key = (subject.lower(), predicate.lower(), obj.lower())
            if key in seen:
                continue
            seen.add(key)
            # Extracted beliefs get moderate confidence — they come from generated text
            self.update_belief(subject, predicate, obj, confidence=0.6)
            count += 1
            logger.debug("Extracted belief: %s %s %s", subject, predicate, obj)
