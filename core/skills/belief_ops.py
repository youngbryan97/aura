"""core/skills/belief_ops.py

Belief Graph LLM Tools — Add/Query Aura's world model during conversation.

These skills give the LLM the ability to persistently update and query
the HardenedBeliefGraph (core/world_model/belief_graph.py) while talking.

Skills registered:
  add_belief    — Assert a new fact or reinforce an existing belief
  query_beliefs — Retrieve what Aura believes about a subject
"""
from core.runtime.errors import record_degradation
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

        fact = f"{source} —[{relation}]→ {target}"
        stored_layers = []

        # Layer 1: BeliefRevisionEngine (primary — real Bayesian-ish updates)
        try:
            from core.container import ServiceContainer
            bre = ServiceContainer.get("belief_revision_engine", default=None)
            if bre and hasattr(bre, "add_belief"):
                bre.add_belief(
                    content=fact,
                    confidence=confidence,
                    domain=f"belief:{relation}",
                    source=source,
                )
                stored_layers.append("belief_engine")
        except Exception as e:
            record_degradation('belief_ops', e)
            logger.debug("BeliefRevisionEngine store failed: %s", e)

        # Layer 2: EpistemicState / World Model (knowledge graph triples)
        try:
            from core.container import ServiceContainer
            world_model = ServiceContainer.get("epistemic_state", default=None)
            if world_model and hasattr(world_model, "add_belief"):
                world_model.add_belief(
                    subject=source, predicate=relation, obj=target,
                    confidence=confidence,
                )
                stored_layers.append("world_model")
        except Exception as e:
            record_degradation('belief_ops', e)
            logger.debug("EpistemicState store failed: %s", e)

        # Layer 3: MemFS text (legacy compatibility + human-readable backup)
        try:
            from core.skills.memory_ops import MemoryOpsSkill
            mem_ops = MemoryOpsSkill()
            block = "persona" if source.lower() in ("aura", "aura_self", "me") else "user"
            mem_resp = await mem_ops.execute(
                {"action": "core_append", "block": block,
                 "content": f"{fact} (confidence={confidence:.2f})"},
                context,
            )
            if mem_resp.get("ok"):
                stored_layers.append(f"memfs:{block}")
        except Exception as e:
            record_degradation('belief_ops', e)
            logger.debug("MemFS store failed: %s", e)

        # Layer 4: Vector memory for semantic retrieval
        try:
            from core.container import ServiceContainer
            vmem = ServiceContainer.get("vector_memory_engine", default=None)
            if vmem and hasattr(vmem, "store"):
                await vmem.store(
                    content=f"Belief: {fact} (confidence: {confidence})",
                    memory_type="semantic",
                    source="belief_ops",
                    tags=["belief", source.lower(), relation.lower()],
                )
                stored_layers.append("vector_memory")
        except Exception as e:
            record_degradation('belief_ops', e)
            logger.debug("Vector memory store failed: %s", e)

        if not stored_layers:
            return {"ok": False, "error": "All belief storage backends failed"}

        summary = f"Belief stored in {len(stored_layers)} layers: {fact}"
        logger.info("%s (layers: %s)", summary, stored_layers)

        # Emit to thought stream
        try:
            from core.thought_stream import get_emitter
            get_emitter().emit("Belief Updated", summary, level="info", category="WorldModel")
        except Exception:
            pass

        return {
            "ok": True,
            "summary": summary,
            "data": {
                "source": source, "relation": relation, "target": target,
                "confidence": confidence, "centrality": centrality,
                "layers": stored_layers,
            },
        }


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

        beliefs = []
        sources_checked = []

        # Layer 1: BeliefRevisionEngine
        try:
            from core.container import ServiceContainer
            bre = ServiceContainer.get("belief_revision_engine", default=None)
            if bre and hasattr(bre, "beliefs"):
                for b in bre.beliefs:
                    content = getattr(b, "content", str(b))
                    if subject.lower() in content.lower():
                        conf = getattr(b, "confidence", 0.5)
                        beliefs.append(f"{content} (confidence: {conf:.2f})")
                sources_checked.append("belief_engine")
        except Exception as e:
            record_degradation('belief_ops', e)
            logger.debug("BeliefRevisionEngine query failed: %s", e)

        # Layer 2: EpistemicState / World Model
        try:
            from core.container import ServiceContainer
            world_model = ServiceContainer.get("epistemic_state", default=None)
            if world_model and hasattr(world_model, "get_relevant_beliefs"):
                wm_beliefs = world_model.get_relevant_beliefs(subject, n=limit)
                for wb in wm_beliefs:
                    belief_str = f"{wb.get('subject', '')} {wb.get('predicate', '')} {wb.get('object', '')} (conf: {wb.get('confidence', 0):.2f})"
                    if belief_str not in beliefs:
                        beliefs.append(belief_str)
                sources_checked.append("world_model")
        except Exception as e:
            record_degradation('belief_ops', e)
            logger.debug("EpistemicState query failed: %s", e)

        # Layer 3: Vector memory semantic search
        try:
            from core.container import ServiceContainer
            vmem = ServiceContainer.get("vector_memory_engine", default=None)
            if vmem and hasattr(vmem, "search"):
                results = await vmem.search(
                    query=f"beliefs about {subject}",
                    limit=limit,
                    memory_type="semantic",
                )
                if results:
                    for r in results:
                        content = r.get("content", str(r)) if isinstance(r, dict) else str(r)
                        if content not in beliefs:
                            beliefs.append(content[:200])
                    sources_checked.append("vector_memory")
        except Exception as e:
            record_degradation('belief_ops', e)
            logger.debug("Vector memory query failed: %s", e)

        # Layer 4: MemFS fallback
        try:
            from core.config import config
            from pathlib import Path
            mem_fs_dir = Path(getattr(config.paths, "base_dir", ".")) / ".aura" / "memfs"
            for block in ["user", "persona"]:
                path = mem_fs_dir / f"{block}.txt"
                if path.exists():
                    lines = path.read_text(encoding="utf-8").splitlines()
                    for line in lines:
                        if subject.lower() in line.lower() and line.strip() not in beliefs:
                            beliefs.append(line.strip())
            sources_checked.append("memfs")
        except Exception as e:
            record_degradation('belief_ops', e)
            logger.debug("MemFS query failed: %s", e)

        beliefs = beliefs[:limit]

        if beliefs:
            summary = f"Beliefs about '{subject}' ({len(beliefs)} found, sources: {', '.join(sources_checked)}):\n" + "\n".join(f"  - {b}" for b in beliefs)
        else:
            summary = f"No beliefs found about '{subject}' (checked: {', '.join(sources_checked)})."

        return {
            "ok": True,
            "summary": summary,
            "data": {"subject": subject, "beliefs": beliefs, "sources": sources_checked},
        }
