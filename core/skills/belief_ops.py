"""core/skills/belief_ops.py

Belief Graph LLM Tools — Add/Query Aura's world model during conversation.

These skills give the LLM the ability to persistently update and query
the HardenedBeliefGraph (core/world_model/belief_graph.py) while talking.

Skills registered:
  add_belief    — Assert a new fact or reinforce an existing belief
  query_beliefs — Retrieve what Aura believes about a subject
"""
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.BeliefOps")


# ─── Input models ─────────────────────────────────────────────────────────────

class AddBeliefInput(BaseModel):
    source: str     = Field(..., description="The subject of the belief (e.g. 'Bryan', 'AURA_SELF')")
    relation: str   = Field(..., description="The predicate/relation (e.g. 'prefers', 'is_building', 'dislikes')")
    target: str     = Field(..., description="The object (e.g. 'Python', 'enterprise AI', 'corporate speak')")
    confidence: float = Field(0.75, ge=0.0, le=1.0, description="Certainty from 0.0 (wild guess) to 1.0 (axiomatic)")
    centrality: float = Field(0.3,  ge=0.0, le=1.0, description="How core to identity this belief is (0.0 peripheral → 1.0 foundational)")


class QueryBeliefsInput(BaseModel):
    subject: str = Field(..., description="The entity to look up beliefs about (e.g. 'Bryan', 'AURA_SELF', 'Python')")
    limit: int   = Field(10, ge=1, le=30, description="Max number of beliefs to return")


# ─── Skills ───────────────────────────────────────────────────────────────────

class AddBeliefSkill(BaseSkill):
    """
    Assert or reinforce a belief in Aura's world model.

    Use this to persistently record facts learned during conversation:
    preferences, opinions, relationships, plans, or world-state facts.
    The belief graph uses Bayesian-ish updates — adding the same belief
    repeatedly increases confidence; contradictory beliefs trigger
    epistemic resolution (the stronger belief wins).

    Example:
      source="Bryan", relation="prefers", target="direct feedback"
      source="AURA_SELF", relation="is_working_on", target="belief graph integration"
    """
    name = "add_belief"
    description = (
        "Add or reinforce a belief in Aura's world model. "
        "Use source/relation/target triples (e.g. Bryan | prefers | Python). "
        "Confidence 0-1, centrality 0-1 (how core to identity)."
    )
    input_model = AddBeliefInput
    metabolic_cost = 1
    timeout_seconds = 5

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        # params may be a Pydantic model (if input_model validated it) or a raw dict
        if hasattr(params, "source"):
            source, relation, target = params.source, params.relation, params.target
            confidence, centrality   = params.confidence, params.centrality
        else:
            source     = str(params.get("source", "")).strip()
            relation   = str(params.get("relation", "")).strip()
            target     = str(params.get("target", "")).strip()
            confidence = float(params.get("confidence", 0.75))
            centrality = float(params.get("centrality", 0.3))

        if not (source and relation and target):
            return {"ok": False, "error": "source, relation, and target are all required"}

        try:
            from core.container import ServiceContainer
            belief_graph = ServiceContainer.get("belief_graph", default=None)
            if belief_graph is None:
                from core.world_model.belief_graph import BeliefGraph
                belief_graph = BeliefGraph()
                ServiceContainer.register_instance("belief_graph", belief_graph)

            belief_graph.update_belief(
                source=source,
                relation=relation,
                target=target,
                confidence_score=confidence,
                centrality=centrality,
            )

            summary = f"Belief recorded: {source} —[{relation}]→ {target} (confidence={confidence:.2f})"
            logger.info("💡 %s", summary)

            # Emit to neural feed so it's visible in thought cards
            try:
                from core.thought_stream import get_emitter
                get_emitter().emit(
                    "Belief Updated",
                    summary,
                    level="info",
                    category="WorldModel",
                )
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

            return {
                "ok": True,
                "summary": summary,
                "data": {
                    "source": source,
                    "relation": relation,
                    "target": target,
                    "confidence": confidence,
                    "centrality": centrality,
                },
            }

        except Exception as e:
            logger.error("add_belief failed: %s", e)
            return {"ok": False, "error": str(e)}


class QueryBeliefsSkill(BaseSkill):
    """
    Retrieve what Aura currently believes about a subject.

    Returns all outgoing edges from the subject node in the belief graph,
    sorted by confidence. Use this before answering questions about
    preferences, relationships, or factual claims to ensure the response
    reflects persistent knowledge rather than just in-context inference.
    """
    name = "query_beliefs"
    description = (
        "Query Aura's persistent belief graph about a subject. "
        "Returns known facts, preferences, and relationships. "
        "Use before answering questions about Bryan, yourself, or any tracked entity."
    )
    input_model = QueryBeliefsInput
    metabolic_cost = 1
    timeout_seconds = 5

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(params, "subject"):
            subject = str(params.subject).strip()
            limit = int(params.limit)
        else:
            subject = str(params.get("subject", "")).strip()
            limit   = int(params.get("limit", 10))

        if not subject:
            return SkillResult(ok=False, skill=self.name, error="subject is required")

        try:
            from core.container import ServiceContainer
            belief_graph = ServiceContainer.get("belief_graph", default=None)
            if belief_graph is None:
                from core.world_model.belief_graph import BeliefGraph
                belief_graph = BeliefGraph()

            g = belief_graph.graph

            # Normalize subject — try exact match first, then case-insensitive
            if subject not in g:
                for node in g.nodes:
                    if str(node).lower() == subject.lower():
                        subject = node
                        break

            if subject not in g:
                return {"ok": True, "summary": f"No beliefs found about '{subject}'.", "data": {"subject": subject, "beliefs": []}}

            beliefs = []
            for target, edge_data in g[subject].items():
                beliefs.append({
                    "source":     subject,
                    "relation":   edge_data.get("relation", "?"),
                    "target":     str(target),
                    "confidence": round(float(edge_data.get("confidence", 0.5)), 3),
                    "centrality": round(float(edge_data.get("centrality", 0.1)), 3),
                    "is_goal":    bool(edge_data.get("is_goal", False)),
                })

            beliefs.sort(key=lambda b: b["confidence"], reverse=True)
            beliefs = beliefs[:limit]

            if beliefs:
                lines = [f"{b['source']} —[{b['relation']}]→ {b['target']} (conf={b['confidence']:.2f})" for b in beliefs]
                summary = f"Beliefs about '{subject}':\n" + "\n".join(lines)
            else:
                summary = f"'{subject}' is in the graph but has no outgoing beliefs."

            return {"ok": True, "summary": summary, "data": {"subject": subject, "beliefs": beliefs}}

        except Exception as e:
            logger.error("query_beliefs failed: %s", e)
            return {"ok": False, "error": str(e)}
