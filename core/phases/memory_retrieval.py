import asyncio
import inspect
import json
import logging
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation
from core.soma.effort import note_effort
from core.utils.queues import decode_stringified_priority_message, role_for_origin
from core.utils.task_tracker import get_task_tracker

from ..state.aura_state import AuraState
from . import BasePhase

logger = logging.getLogger(__name__)

_MEMORY_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)


def _record_memory_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    metadata["stage"] = stage
    try:
        record_degradation(
            "memory_retrieval",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata,
        )
    except TypeError:
        record_degradation(
            "memory_retrieval",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


def _safe_text(value: Any, *, max_chars: int = 12_000) -> str:
    if value is None:
        return ""
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return ""
    text = text.replace("\x00", "").strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (RuntimeError, TypeError, ValueError):
        return default


def _iter_retrieval_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except (RuntimeError, TypeError, ValueError):
        return []


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _safe_metadata(raw: Any) -> dict[str, Any]:
    """Coerce metadata to a dict.  Knowledge-graph rows store metadata as a
    JSON TEXT column; if the upstream forgot to parse it we handle it here."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError) as _exc:
            logger.debug(
                "MemoryRetrieval: ignored malformed memory metadata: %s",
                _exc,
            )
        return {}
    return {}


class MemoryRetrievalPhase(BasePhase):
    """
    Phase 2: Memory Retrieval.
    Uses current working memory to retrieve relevant long-term context (RAG)
    and updates the state's long_term_memory field.
    """

    def __init__(self, container: Any):
        self.container = container

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
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
        if isinstance(last_msg, dict):
            query = last_msg.get("content", "")
        else:
            query = _safe_text(last_msg)
            last_msg = {"role": "user", "content": query}
        decoded_payload, decoded_origin, was_decoded = decode_stringified_priority_message(query)
        if was_decoded:
            if isinstance(decoded_payload, dict):
                query = decoded_payload.get("content", "")
                if decoded_payload.get("origin"):
                    decoded_origin = decoded_payload["origin"]
            else:
                query = str(decoded_payload)
            if decoded_origin:
                last_msg = {
                    **last_msg,
                    "origin": decoded_origin,
                    "role": role_for_origin(decoded_origin),
                }
        query = _safe_text(query)

        # What she is working on, when the last thing in mind is not something
        # a person said. Retrieval used to run only when the newest
        # working-memory entry came from the user — "to save cycles" — and the
        # newest entry after a turn is her own reply, so across an ordinary
        # hour of thinking she recalled nothing at all. Every autonomous turn,
        # every turn of her own problem solving, every idle turn: no recall,
        # and the recalled set every consumer reads went stale and stayed
        # stale. A memory that can only be reached when somebody speaks is not
        # hers.
        #
        # The cycles are still saved, by the thing that was actually costing
        # them: a query already answered is not asked again.
        if not query or last_msg.get("role") != "user":
            query = _safe_text(
                getattr(state.cognition, "current_objective", "") or objective or ""
            )
        if not query:
            return state

        if len(query) < 5:
            return state

        if query == getattr(state.cognition, "last_retrieval_query", None):
            return state

        try:
            from core.runtime.proof_policy import is_strict_proof_answer_prompt

            proof_origin = (
                last_msg.get("origin")
                or getattr(state.cognition, "current_origin", None)
                or kwargs.get("origin")
            )
            if is_strict_proof_answer_prompt(query or objective or "", origin=proof_origin):
                new_state = state.derive("memory_retrieval_skipped_for_strict_proof")
                new_state.cognition.long_term_memory = []
                new_state.cognition.memory_scores = []
                new_state.response_modifiers["proof_memory_retrieval_skipped"] = True
                return new_state
        except _MEMORY_RECOVERABLE_ERRORS as exc:
            _record_memory_degradation(
                exc,
                action="continued memory retrieval after strict proof guard failed",
                stage="strict_proof_memory_guard",
            )

        try:
            affect_signature = (
                state.affect.get_cognitive_signature()
                if hasattr(state.affect, "get_cognitive_signature")
                else {}
            )
            if not isinstance(affect_signature, dict):
                affect_signature = {}
        except _MEMORY_RECOVERABLE_ERRORS as exc:
            affect_signature = {}
            _record_memory_degradation(
                exc,
                action="used neutral affect signature after affect state read failed",
                stage="affect_signature",
            )
        response_modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        contract = response_modifiers.get("response_contract", {}) or {}
        imagination_memory_pressure = _safe_float(
            response_modifiers.get("imagination_memory_pressure")
        )
        bicameral_causal_effects = response_modifiers.get("bicameral_causal_effects")
        if not isinstance(bicameral_causal_effects, dict):
            bicameral_causal_effects = {}
        bicameral_memory_priority = _safe_float(
            response_modifiers.get("bicameral_memory_priority")
            or bicameral_causal_effects.get("memory_priority")
        )
        imagination_verification_pressure = _safe_float(
            response_modifiers.get("imagination_verification_pressure")
            or response_modifiers.get("verification_pressure")
        )
        bicameral_verification_pressure = _safe_float(
            response_modifiers.get("bicameral_verification_pressure")
            or bicameral_causal_effects.get("verification_pressure")
        )
        verification_pressure = max(
            imagination_verification_pressure,
            bicameral_verification_pressure,
            _safe_float(response_modifiers.get("verification_pressure")),
        )
        effective_memory_salience = max(
            _safe_float(affect_signature.get("memory_salience")),
            imagination_memory_pressure,
            bicameral_memory_priority,
        )
        retrieval_limit = 5
        if (
            contract.get("requires_memory_grounding")
            or response_modifiers.get("requires_memory_grounding")
            or imagination_memory_pressure > 0.55
            or bicameral_memory_priority > 0.55
        ):
            retrieval_limit += 2
        if effective_memory_salience > 0.65:
            retrieval_limit += 1
        if verification_pressure > 0.55:
            retrieval_limit += 1
        hot_limit = 4 if _safe_float(affect_signature.get("social_hunger")) > 0.65 else 3

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
            if fe_engine and fe_engine.current and _safe_float(fe_engine.current.free_energy) > 0.6:
                retrieval_limit += 1  # High surprise: need more grounding
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis and _safe_float(homeostasis.compute_vitality(), default=0.5) < 0.35:
                retrieval_limit = max(2, retrieval_limit - 2)  # Low energy: conserve
        except _MEMORY_RECOVERABLE_ERRORS as exc:
            _record_memory_degradation(
                exc,
                action="kept bounded default retrieval limits after substrate modulation failed",
                stage="retrieval_modulation",
            )

        # Entity memory targets the search. When Aura has recognised a person,
        # place, or thing in what was said, that entity's canonical name and its
        # best-evidenced associations are appended to the query, so retrieval
        # looks for what she KNOWS about it rather than only the literal words
        # the user happened to type ("he" retrieves nothing; the resolved name
        # and its bound episode ids retrieve the history).
        try:
            entity_cues = response_modifiers.get("entity_retrieval_cues")
            if isinstance(entity_cues, (list, tuple)) and entity_cues:
                seen_cue: set[str] = set()
                extra: list[str] = []
                lowered_query = query.lower()
                for cue in entity_cues:
                    text = str(cue or "").strip()
                    if not text or len(text) > 120:
                        continue
                    folded = text.lower()
                    if folded in seen_cue or folded in lowered_query:
                        continue
                    seen_cue.add(folded)
                    extra.append(text)
                    if len(extra) >= 6:
                        break
                if extra:
                    query = f"{query} {' '.join(extra)}".strip()[:2000]
                    logger.debug(
                        "🧠 MemoryRetrieval: entity memory added %d retrieval cue(s).",
                        len(extra),
                    )
        except _MEMORY_RECOVERABLE_ERRORS as exc:
            _record_memory_degradation(
                exc,
                action="searched without entity-memory retrieval cues",
                stage="entity_cue_targeting",
            )

        logger.info("🧠 MemoryRetrieval: Searching for context: %s...", query[:50])

        async def _get_dual():
            try:
                mm = self.container.get("memory_manager", default=None)
                if mm and hasattr(mm, "dual_memory"):
                    async with asyncio.timeout(15.0):
                        return await mm.dual_memory.retrieve_context(query)
            except TimeoutError as exc:
                logger.debug(
                    "MemoryRetrieval: optional DualMemory RAG timed out; continuing without it: %s",
                    exc,
                )
                return None
            except _MEMORY_RECOVERABLE_ERRORS as exc:
                _record_memory_degradation(
                    exc,
                    action="continued retrieval without dual-memory context",
                    stage="dual_memory",
                )
                logger.debug("MemoryRetrieval: DualMemory RAG failed: %s", exc)
                return None
            return None

        async def _get_kg():
            try:
                kg = self.container.get("knowledge_graph", default=None)
                if kg:
                    method = kg.search_knowledge
                    async with asyncio.timeout(15.0):
                        if inspect.iscoroutinefunction(method):
                            return await method(query, limit=retrieval_limit)
                        else:
                            return await asyncio.to_thread(method, query, limit=retrieval_limit)
            except TimeoutError as exc:
                logger.debug(
                    "MemoryRetrieval: optional KnowledgeGraph retrieval timed out; continuing without it: %s",
                    exc,
                )
                return None
            except _MEMORY_RECOVERABLE_ERRORS as exc:
                _record_memory_degradation(
                    exc,
                    action="continued retrieval without knowledge-graph context",
                    stage="knowledge_graph",
                )
                logger.debug("MemoryRetrieval: KnowledgeGraph search failed: %s", exc)
                return None
            return None

        async def _get_facade():
            try:
                memory = self.container.get("memory_facade", default=None)
                if not memory:
                    return None

                recalled = []
                if hasattr(memory, "search"):
                    async with asyncio.timeout(15.0):
                        recalled.extend(
                            list(
                                await _maybe_await(memory.search(query, limit=retrieval_limit))
                                or []
                            )
                        )

                if hasattr(memory, "get_hot_memory"):
                    async with asyncio.timeout(15.0):
                        hot = await _maybe_await(memory.get_hot_memory(limit=hot_limit))
                    if isinstance(hot, dict):
                        for episode in hot.get("recent_episodes", []) or []:
                            content_str = ""
                            if hasattr(episode, "to_retrieval_text"):
                                content_str = episode.to_retrieval_text()
                            elif hasattr(episode, "full_description"):
                                content_str = episode.full_description
                            elif isinstance(episode, dict):
                                content_str = episode.get("content") or episode.get("description") or episode.get("context") or str(episode)
                            else:
                                content_str = _safe_text(episode, max_chars=2_000)

                            metadata_dict = {"type": "recent_episode"}
                            valence_val = 0.0
                            importance_val = 0.5

                            if hasattr(episode, "emotional_valence"):
                                valence_val = _safe_float(episode.emotional_valence)
                            elif isinstance(episode, dict):
                                valence_val = _safe_float(episode.get("emotional_valence", 0.0))

                            if hasattr(episode, "importance"):
                                importance_val = _safe_float(episode.importance, default=0.5)
                            elif isinstance(episode, dict):
                                importance_val = _safe_float(episode.get("importance", 0.5), default=0.5)

                            metadata_dict["emotional_valence"] = valence_val
                            metadata_dict["importance"] = importance_val

                            recalled.append(
                                {
                                    "content": content_str,
                                    "metadata": metadata_dict,
                                    "score": 0.85,
                                }
                            )

                return recalled or None
            except TimeoutError as exc:
                logger.debug(
                    "MemoryRetrieval: optional MemoryFacade retrieval timed out; continuing without it: %s",
                    exc,
                )
                return None
            except _MEMORY_RECOVERABLE_ERRORS as exc:
                _record_memory_degradation(
                    exc,
                    action="continued retrieval without memory-facade context",
                    stage="memory_facade",
                )
                logger.debug("MemoryRetrieval: MemoryFacade search failed: %s", exc)
                return None

        async def _get_episodic():
            try:
                from core.container import ServiceContainer

                ep = self.container.get("episodic_memory", default=None)
                if ep is None:
                    ep = ServiceContainer.get("episodic_memory", default=None)
                if ep and hasattr(ep, "recall_similar_async"):
                    async with asyncio.timeout(15.0):
                        return await ep.recall_similar_async(query, limit=retrieval_limit)
                elif ep and hasattr(ep, "recall_similar"):
                    return await asyncio.to_thread(ep.recall_similar, query, retrieval_limit)
            except TimeoutError as exc:
                logger.debug(
                    "MemoryRetrieval: optional EpisodicMemory recall timed out; continuing without it: %s",
                    exc,
                )
                return None
            except _MEMORY_RECOVERABLE_ERRORS as exc:
                _record_memory_degradation(
                    exc,
                    action="continued retrieval without episodic context",
                    stage="episodic_memory",
                )
                logger.debug("MemoryRetrieval: Episodic recall failed: %s", exc)
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
            memory_candidates.append(
                (
                    0.45 + effective_memory_salience * 0.1,
                    _safe_text(dual_res, max_chars=2_000),
                )
            )
        if kg_res:
            for km in _iter_retrieval_items(kg_res):
                if isinstance(km, dict):
                    metadata = _safe_metadata(km.get("metadata", {}))
                    emotional_valence = _safe_float(metadata.get("emotional_valence"))
                    importance = _safe_float(metadata.get("importance"))
                    valence_alignment = 1.0 - min(
                        1.0,
                        abs(_safe_float(getattr(state.affect, "valence", 0.0)) - emotional_valence),
                    )
                    weighted_score = (
                        0.3
                        + (importance * 0.3)
                        + (valence_alignment * 0.2)
                        + (effective_memory_salience * 0.2)
                    )
                    content = _safe_text(km.get("content"), max_chars=2_000)
                    if content:
                        memory_candidates.append(
                            (weighted_score, f"[{km.get('type', 'fact')}] {content}")
                        )

                    if abs(emotional_valence) > 0.3:
                        total_valence_hit += emotional_valence * importance
                        total_arousal_hit += importance * 0.5
                        memory_hits += 1

        if facade_res:
            for item in _iter_retrieval_items(facade_res):
                if isinstance(item, dict):
                    content = _safe_text(item.get("content") or item.get("text"), max_chars=2_000)
                    if content:
                        metadata = _safe_metadata(item.get("metadata", {}))
                        emotional_valence = _safe_float(metadata.get("emotional_valence"))
                        importance = _safe_float(metadata.get("importance"))
                        score = _safe_float(item.get("score"))
                        salience = effective_memory_salience
                        valence_alignment = 1.0 - min(
                            1.0,
                            abs(
                                _safe_float(getattr(state.affect, "valence", 0.0))
                                - emotional_valence
                            ),
                        )
                        weighted_score = round(
                            (score * 0.35)
                            + (importance * 0.25)
                            + (valence_alignment * 0.25)
                            + (salience * 0.15),
                            3,
                        )
                        memory_candidates.append(
                            (weighted_score, f"[memory score={weighted_score:.3f}] {content}")
                        )

                        if abs(emotional_valence) > 0.3:
                            total_valence_hit += emotional_valence * importance
                            total_arousal_hit += importance * 0.5
                            memory_hits += 1
                elif item:
                    memory_candidates.append(
                        (0.35, f"[memory] {_safe_text(item, max_chars=2_000)}")
                    )

        if episodic_res:
            for ep in _iter_retrieval_items(episodic_res):
                desc = _safe_text(
                    getattr(ep, "description", "") or getattr(ep, "context", "") or ep,
                    max_chars=2_000,
                )
                outcome = _safe_text(getattr(ep, "outcome", ""), max_chars=1_000)
                importance = _safe_float(getattr(ep, "importance", 0.5), default=0.5)
                valence = _safe_float(getattr(ep, "emotional_valence", 0.0))
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

                    logger.debug(
                        "💥 Memory retrieval triggered affective hit: val_shift=%.2f, arousal_shift=%.2f",
                        val_shift,
                        arousal_shift,
                    )

                    modification = affect_engine.modify(
                        dv=val_shift,
                        da=arousal_shift,
                        de=0.0,
                        source="memory_retrieval",
                    )
                    get_task_tracker().create_task(
                        modification,
                        name="memory_retrieval.affective_hit",
                    )
            except _MEMORY_RECOVERABLE_ERRORS as exc:
                close = getattr(locals().get("modification", None), "close", None)
                if callable(close):
                    close()
                _record_memory_degradation(
                    exc,
                    action="kept retrieved memories after affect scheduling failed",
                    stage="affective_memory_hit",
                )
                logger.debug("Failed to push memory affect: %s", exc)

        # Recall costs the search, not the finding. Reported after the early
        # return below, a recall that came back empty cost nothing — so the
        # body could not feel a fruitless search, and the column that carries
        # this was constant for the whole of every recording.
        note_effort("recall", max(1, len(memory_candidates)))

        scores: list[float] = []
        if memory_candidates:
            memory_candidates.sort(key=lambda item: item[0], reverse=True)
            kept = memory_candidates[:retrieval_limit]
            memories = [text for _, text in kept]
            # And how well each one matched. The ranking is computed here and
            # was discarded here, so every consumer downstream saw an unranked
            # list of strings and had to treat a recollection that answered the
            # question exactly the same as one that scraped in last.
            scores = [round(float(score), 4) for score, _ in kept]

        if not memories:
            return state

        # Derive new state with retrieved context
        new_state = state.derive("memory_retrieval")
        new_state.cognition.long_term_memory = memories
        new_state.cognition.memory_scores = scores
        new_state.cognition.last_retrieval_query = query
        new_state.response_modifiers["memory_retrieval_signature"] = {
            "query": query[:160],
            "retrieval_limit": retrieval_limit,
            "hot_limit": hot_limit,
            "affect": affect_signature,
            "imagination_memory_pressure": round(imagination_memory_pressure, 4),
            "imagination_verification_pressure": round(imagination_verification_pressure, 4),
            "bicameral_memory_priority": round(bicameral_memory_priority, 4),
            "bicameral_verification_pressure": round(bicameral_verification_pressure, 4),
            "verification_pressure": round(verification_pressure, 4),
        }
        return new_state
