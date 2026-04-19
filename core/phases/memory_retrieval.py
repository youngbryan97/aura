import asyncio
import logging
from typing import Any, Optional
from . import BasePhase
from ..state.aura_state import AuraState
from core.utils.queues import decode_stringified_priority_message, role_for_origin

logger = logging.getLogger(__name__)

class MemoryRetrievalPhase(BasePhase):
    """
    Phase 2: Memory Retrieval.
    Uses current working memory to retrieve relevant long-term context (RAG)
    and updates the state's long_term_memory field.
    """
    
    def __init__(self, container: Any):
        self.container = container

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        Retrieve relevant long-term memories for the most recent user message.

        Queries both the dual-memory RAG store and the knowledge graph in parallel,
        then merges the results into state.cognition.long_term_memory.
        Returns state unchanged if working memory is empty or the last message is not
        from a user.
        """
        if not state.cognition.working_memory:
            return state
            
        # Use the most recent user entry or objective for retrieval
        last_msg = state.cognition.working_memory[-1]
        query = last_msg.get("content", "")
        decoded_payload, decoded_origin, was_decoded = decode_stringified_priority_message(query)
        if was_decoded:
            if isinstance(decoded_payload, dict):
                query = decoded_payload.get("content", "")
                if decoded_payload.get("origin"):
                    decoded_origin = decoded_payload["origin"]
            else:
                query = str(decoded_payload)
            if decoded_origin:
                last_msg = {**last_msg, "origin": decoded_origin, "role": role_for_origin(decoded_origin)}
        
        if not query or last_msg.get("role") != "user":
            # Only retrieve on new user input for now to save cycles
            return state
            
        if len(query) < 5:
            return state

        affect_signature = (
            state.affect.get_cognitive_signature()
            if hasattr(state.affect, "get_cognitive_signature")
            else {}
        )
        contract = dict(getattr(state, "response_modifiers", {}) or {}).get("response_contract", {}) or {}
        retrieval_limit = 5
        if contract.get("requires_memory_grounding"):
            retrieval_limit += 2
        if float(affect_signature.get("memory_salience", 0.0) or 0.0) > 0.65:
            retrieval_limit += 1
        hot_limit = 4 if float(affect_signature.get("social_hunger", 0.0) or 0.0) > 0.65 else 3

        # ── Consciousness-driven memory modulation ──
        # High attention coherence (flow state) → retrieve more (deeper context)
        # High free energy (surprise) → retrieve more (need grounding)
        # Low homeostasis vitality → retrieve less (conserve resources)
        try:
            from core.container import ServiceContainer
            attention = ServiceContainer.get("attention_schema", default=None)
            if attention and hasattr(attention, "is_in_flow") and attention.is_in_flow():
                retrieval_limit += 2  # Flow state: deeper retrieval
            fe_engine = ServiceContainer.get("free_energy_engine", default=None)
            if fe_engine and fe_engine.current and fe_engine.current.free_energy > 0.6:
                retrieval_limit += 1  # High surprise: need more grounding
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis and homeostasis.compute_vitality() < 0.35:
                retrieval_limit = max(2, retrieval_limit - 2)  # Low energy: conserve
        except Exception as _cmod_e:
            pass  # Non-critical — proceed with default limits

        logger.info("🧠 MemoryRetrieval: Searching for context: %s...", query[:50])
        
        async def _get_dual():
            try:
                mm = self.container.get("memory_manager", default=None)
                if mm and hasattr(mm, "dual_memory"):
                    async with asyncio.timeout(5.0):
                        return await mm.dual_memory.retrieve_context(query)
            except Exception as e:
                logger.debug("MemoryRetrieval: DualMemory RAG failed: %s", e)
                return None
            return None

        async def _get_kg():
            try:
                kg = self.container.get("knowledge_graph", default=None)
                if kg:
                    method = kg.search_knowledge
                    if asyncio.iscoroutinefunction(method):
                        return await method(query, limit=retrieval_limit)
                    else:
                        return await asyncio.to_thread(method, query, limit=retrieval_limit)
            except Exception as e:
                logger.debug("MemoryRetrieval: KnowledgeGraph search failed: %s", e)
                return None
            return None

        async def _get_facade():
            try:
                memory = self.container.get("memory_facade", default=None)
                if not memory:
                    return None

                recalled = []
                if hasattr(memory, "search"):
                    async with asyncio.timeout(5.0):
                        recalled.extend(list(await memory.search(query, limit=retrieval_limit) or []))

                if hasattr(memory, "get_hot_memory"):
                    async with asyncio.timeout(5.0):
                        hot = await memory.get_hot_memory(limit=hot_limit)
                    for episode in hot.get("recent_episodes", []) or []:
                        recalled.append({
                            "content": str(episode),
                            "metadata": {"type": "recent_episode"},
                        })

                return recalled or None
            except Exception as e:
                logger.debug("MemoryRetrieval: MemoryFacade search failed: %s", e)
                return None

        async def _get_episodic():
            try:
                from core.container import ServiceContainer
                ep = self.container.get("episodic_memory", default=None)
                if ep is None:
                    ep = ServiceContainer.get("episodic_memory", default=None)
                if ep and hasattr(ep, "recall_similar_async"):
                    async with asyncio.timeout(3.0):
                        return await ep.recall_similar_async(query, limit=retrieval_limit)
                elif ep and hasattr(ep, "recall_similar"):
                    return await asyncio.to_thread(ep.recall_similar, query, retrieval_limit)
            except Exception as e:
                logger.debug("MemoryRetrieval: Episodic recall failed: %s", e)
            return None

        dual_res, kg_res, facade_res, episodic_res = await asyncio.gather(
            _get_dual(),
            _get_kg(),
            _get_facade(),
            _get_episodic(),
        )
        
        memories: list[str] = []
        memory_candidates: list[tuple[float, str]] = []

        # ── Gap 3 Fix: Memory Affect → Steering ──
        total_valence_hit = 0.0
        total_arousal_hit = 0.0
        memory_hits = 0

        if dual_res:
            memory_candidates.append((0.45 + float(affect_signature.get("memory_salience", 0.0) or 0.0) * 0.1, str(dual_res)))
        if kg_res:
            for km in kg_res:
                if isinstance(km, dict):
                    metadata = km.get("metadata", {}) or {}
                    emotional_valence = float(metadata.get("emotional_valence", 0.0) or 0.0)
                    importance = float(metadata.get("importance", 0.0) or 0.0)
                    valence_alignment = 1.0 - min(1.0, abs(float(state.affect.valence or 0.0) - emotional_valence))
                    weighted_score = (0.3 + (importance * 0.3) + (valence_alignment * 0.2) + (float(affect_signature.get("memory_salience", 0.0) or 0.0) * 0.2))
                    memory_candidates.append((weighted_score, f"[{km.get('type', 'fact')}] {km.get('content')}"))

                    if abs(emotional_valence) > 0.3:
                        total_valence_hit += emotional_valence * importance
                        total_arousal_hit += importance * 0.5
                        memory_hits += 1

        if facade_res:
            for item in facade_res:
                if isinstance(item, dict):
                    content = item.get("content") or item.get("text") or ""
                    if content:
                        metadata = item.get("metadata", {}) or {}
                        emotional_valence = float(metadata.get("emotional_valence", 0.0) or 0.0)
                        importance = float(metadata.get("importance", 0.0) or 0.0)
                        score = float(item.get("score", 0.0) or 0.0)
                        salience = float(affect_signature.get("memory_salience", 0.0) or 0.0)
                        valence_alignment = 1.0 - min(1.0, abs(float(state.affect.valence or 0.0) - emotional_valence))
                        weighted_score = round((score * 0.35) + (importance * 0.25) + (valence_alignment * 0.25) + (salience * 0.15), 3)
                        memory_candidates.append((weighted_score, f"[memory score={weighted_score:.3f}] {content}"))

                        if abs(emotional_valence) > 0.3:
                            total_valence_hit += emotional_valence * importance
                            total_arousal_hit += importance * 0.5
                            memory_hits += 1
                elif item:
                    memory_candidates.append((0.35, f"[memory] {item}"))

        if episodic_res:
            for ep in episodic_res:
                desc = getattr(ep, "description", "") or getattr(ep, "context", "") or str(ep)
                outcome = getattr(ep, "outcome", "") or ""
                importance = float(getattr(ep, "importance", 0.5) or 0.5)
                valence = float(getattr(ep, "emotional_valence", 0.0) or 0.0)
                content = f"{desc}" + (f" → {outcome}" if outcome and outcome != desc else "")
                if content and len(content) > 10:
                    score = 0.5 + importance * 0.3 + abs(valence) * 0.2
                    memory_candidates.append((score, f"[episodic] {content}"))

                    if abs(valence) > 0.3:
                        total_valence_hit += valence * importance
                        total_arousal_hit += importance * 0.5
                        memory_hits += 1

        # Push accumulated affect from memory retrieval
        if memory_hits > 0:
            try:
                from core.container import ServiceContainer
                affect_engine = ServiceContainer.get("affect_engine", default=None)
                if affect_engine and hasattr(affect_engine, "modify"):
                    val_shift = (total_valence_hit / memory_hits) * 0.4
                    arousal_shift = (total_arousal_hit / memory_hits) * 0.3
                    
                    logger.debug("💥 Memory retrieval triggered affective hit: val_shift=%.2f, arousal_shift=%.2f",
                                 val_shift, arousal_shift)
                    
                    asyncio.create_task(
                        affect_engine.modify(dv=val_shift, da=arousal_shift, de=0.0, source="memory_retrieval")
                    )
            except Exception as e:
                logger.debug("Failed to push memory affect: %s", e)

        if memory_candidates:
            memory_candidates.sort(key=lambda item: item[0], reverse=True)
            memories = [text for _, text in memory_candidates[:retrieval_limit]]
            
        if not memories:
            return state
            
        # Derive new state with retrieved context
        new_state = state.derive("memory_retrieval")
        new_state.cognition.long_term_memory = memories
        new_state.response_modifiers["memory_retrieval_signature"] = {
            "query": query[:160],
            "retrieval_limit": retrieval_limit,
            "hot_limit": hot_limit,
            "affect": affect_signature,
        }
        return new_state
