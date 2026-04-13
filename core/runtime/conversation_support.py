from __future__ import annotations

import logging
import time
from typing import Any

from core.runtime import service_access
from core.runtime.coding_session_memory import (
    build_coding_context_block,
    get_coding_session_memory,
)
from core.runtime.turn_analysis import analyze_turn

logger = logging.getLogger("Aura.ConversationSupport")


def _is_task_context_priority(objective: str) -> bool:
    lowered = str(objective or "").lower()
    if not lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "keep going",
            "keep it going",
            "continue",
            "resume",
            "let's do it",
            "lets do it",
            "do it",
            "are you done",
            "did you finish",
            "status",
            "progress",
            "still running",
            "task",
            "follow up",
            "what happened",
        )
    )


def _is_goal_context_priority(objective: str) -> bool:
    lowered = str(objective or "").lower()
    if not lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "project",
            "roadmap",
            "milestone",
            "priority",
            "long term",
            "long-term",
            "goal",
            "plan",
            "status",
            "progress",
            "resume",
            "continue",
            "keep going",
            "keep it going",
        )
    )


def resolve_primary_user_id(state: Any) -> str:
    try:
        world = getattr(state, "world", None)
        for collection_name in ("relationship_graph", "known_entities"):
            collection = getattr(world, collection_name, {}) or {}
            for key in collection.keys():
                normalized = str(key).strip().lower()
                if normalized:
                    return normalized
    except Exception as _exc:
        logger.debug("Suppressed Exception: %s", _exc)
    return "bryan"


async def record_shared_ground_callbacks(response_text: str) -> None:
    try:
        from core.memory.shared_ground import get_shared_ground

        shared_ground = get_shared_ground()
        if not shared_ground.entries:
            return

        resp_lower = response_text.lower()
        for entry in shared_ground.entries:
            ref_words = entry.reference.lower().split()
            matches = sum(1 for word in ref_words if len(word) > 3 and word in resp_lower)
            if matches >= 2:
                shared_ground.record_callback(entry.reference)
                logger.debug("SharedGround callback recorded: %s", entry.reference)
    except Exception as _exc:
        logger.debug("Suppressed Exception: %s", _exc)


def build_conversational_context_blocks(state: Any, objective: str = "") -> list[str]:
    user_id = resolve_primary_user_id(state)
    priority_blocks: list[str] = []
    blocks: list[str] = []

    try:
        profiler = service_access.optional_service("conversational_profiler", default=None)
        if profiler and hasattr(profiler, "get_context_injection"):
            profile_block = profiler.get_context_injection(user_id)
            if profile_block:
                blocks.append(profile_block)
    except Exception as exc:
        logger.debug("ConversationalProfile injection failed: %s", exc)

    try:
        dialogue = service_access.resolve_dialogue_cognition(default=None)
        if dialogue and hasattr(dialogue, "get_context_injection"):
            source_ids = dialogue.default_source_ids() if hasattr(dialogue, "default_source_ids") else None
            dialogue_block = dialogue.get_context_injection(
                user_id,
                current_text=objective or "",
                source_ids=source_ids,
            )
            if dialogue_block:
                blocks.append(dialogue_block)
    except Exception as exc:
        logger.debug("DialogueCognition injection failed: %s", exc)

    try:
        humor = service_access.optional_service("humor_engine", default=None)
        if humor:
            humor_guide = humor.get_humor_guidance(user_id)
            if humor_guide:
                blocks.append(humor_guide)
            banter = humor.get_banter_directive()
            if banter:
                blocks.append(banter)
    except Exception as exc:
        logger.debug("HumorEngine injection failed: %s", exc)

    try:
        conv_intel = service_access.optional_service("conversation_intelligence", default=None)
        if conv_intel and hasattr(conv_intel, "get_context_injection"):
            ci_block = conv_intel.get_context_injection()
            if ci_block:
                blocks.append(ci_block)
    except Exception as exc:
        logger.debug("ConversationIntelligence injection failed: %s", exc)

    try:
        rel_intel = service_access.optional_service("relational_intelligence", default=None)
        if rel_intel and hasattr(rel_intel, "get_context_injection"):
            ri_block = rel_intel.get_context_injection(user_id)
            if ri_block:
                blocks.append(ri_block)
    except Exception as exc:
        logger.debug("RelationalIntelligence injection failed: %s", exc)

    try:
        social_imagination = service_access.resolve_social_imagination(default=None)
        if social_imagination and hasattr(social_imagination, "get_context_injection"):
            si_block = social_imagination.get_context_injection(
                user_id,
                current_text=objective or "",
            )
            if si_block:
                blocks.append(si_block)
    except Exception as exc:
        logger.debug("SocialImagination injection failed: %s", exc)

    try:
        coding_block = build_coding_context_block(objective or "")
        if coding_block:
            priority_blocks.append(coding_block)
    except Exception as exc:
        logger.debug("Coding session context injection failed: %s", exc)

    try:
        from core.agency.task_commitment_verifier import get_task_commitment_verifier

        verifier = get_task_commitment_verifier()
        if verifier and hasattr(verifier, "get_context_block"):
            task_block = verifier.get_context_block(objective or "")
            if task_block:
                if _is_task_context_priority(objective or ""):
                    priority_blocks.append(task_block)
                else:
                    blocks.append(task_block)
    except Exception as exc:
        logger.debug("Task verifier context injection failed: %s", exc)

    try:
        goal_engine = service_access.resolve_goal_engine(default=None)
        if goal_engine and hasattr(goal_engine, "get_context_block"):
            goal_block = goal_engine.get_context_block(objective=objective or "")
            if goal_block:
                if _is_goal_context_priority(objective or "") or priority_blocks:
                    priority_blocks.append(goal_block)
                else:
                    blocks.append(goal_block)
    except Exception as exc:
        logger.debug("Goal engine context injection failed: %s", exc)

    return priority_blocks + blocks


async def update_conversational_intelligence(user_input: str, aura_response: str, state: Any) -> None:
    user_id = resolve_primary_user_id(state)

    try:
        profiler = service_access.optional_service("conversational_profiler", default=None)
        if profiler:
            await profiler.update_from_interaction(user_id, user_input, aura_response, {})
    except Exception as exc:
        logger.debug("ConversationalProfile update skipped: %s", exc)

    try:
        dialogue = service_access.resolve_dialogue_cognition(default=None)
        if dialogue:
            await dialogue.update_from_interaction(user_id, user_input, aura_response, {})
    except Exception as exc:
        logger.debug("DialogueCognition update skipped: %s", exc)

    try:
        humor = service_access.optional_service("humor_engine", default=None)
        if humor:
            humor.record_reaction(user_id, user_input, time.time())
            dynamics = service_access.resolve_conversational_dynamics(default=None)
            if dynamics:
                humor.update_banter_state(user_input, dynamics.get_current_state())
    except Exception as exc:
        logger.debug("HumorEngine update skipped: %s", exc)

    try:
        conv_intel = service_access.optional_service("conversation_intelligence", default=None)
        if conv_intel:
            dynamics = service_access.resolve_conversational_dynamics(default=None)
            dynamics_state = dynamics.get_current_state() if dynamics else None
            discourse_state = {
                "topic": getattr(state.cognition, "discourse_topic", None),
                "depth": getattr(state.cognition, "discourse_depth", 0),
                "energy": getattr(state.cognition, "conversation_energy", 0.5),
                "trend": getattr(state.cognition, "user_emotional_trend", "neutral"),
            } if state else {}
            await conv_intel.update(user_input, aura_response, dynamics_state, discourse_state)
    except Exception as exc:
        logger.debug("ConversationIntelligence update skipped: %s", exc)

    try:
        rel_intel = service_access.optional_service("relational_intelligence", default=None)
        if rel_intel:
            dynamics = service_access.resolve_conversational_dynamics(default=None)
            dynamics_state = dynamics.get_current_state() if dynamics else None
            await rel_intel.update_from_interaction(user_id, user_input, aura_response, dynamics_state)
    except Exception as exc:
        logger.debug("RelationalIntelligence update skipped: %s", exc)

    try:
        social_imagination = service_access.resolve_social_imagination(default=None)
        if social_imagination:
            await social_imagination.update_from_interaction(user_id, user_input, aura_response, {})
    except Exception as exc:
        logger.debug("SocialImagination update skipped: %s", exc)


def _conversation_emotional_valence(user_input: str) -> float:
    lowered = str(user_input or "").lower()
    if any(token in lowered for token in ("love", "thanks", "thank you", "appreciate", "glad")):
        return 0.4
    if any(token in lowered for token in ("upset", "angry", "hurt", "afraid", "sad", "frustrated")):
        return -0.3
    return 0.1


def _conversation_importance(user_input: str) -> float:
    analysis = analyze_turn(user_input)
    importance = 0.35
    if analysis.requires_live_aura_voice:
        importance += 0.25
    if analysis.suggests_deliberate_mode:
        importance += 0.15
    if analysis.intent_type in {"SKILL", "TASK"}:
        importance += 0.1
    return min(0.95, importance)


async def record_conversation_experience(user_input: str, aura_response: str, state: Any = None) -> None:
    if not str(user_input or "").strip() or not str(aura_response or "").strip():
        return

    state_obj = state
    if state_obj is None:
        repo = service_access.resolve_state_repository(default=None)
        state_obj = getattr(repo, "_current", None) if repo is not None else None

    user_id = resolve_primary_user_id(state_obj)
    importance = _conversation_importance(user_input)
    emotional_valence = _conversation_emotional_valence(user_input)
    analysis = analyze_turn(user_input)

    await update_conversational_intelligence(user_input, aura_response, state_obj)
    await record_shared_ground_callbacks(aura_response)

    try:
        get_coding_session_memory().record_conversation_turn(
            user_input,
            aura_response,
            analysis=analysis,
        )
    except Exception as exc:
        logger.debug("Coding session turn recording skipped: %s", exc)

    try:
        episodic = service_access.optional_service("episodic_memory", default=None)
        if episodic and hasattr(episodic, "record_episode_async"):
            await episodic.record_episode_async(
                context=f"User said: {str(user_input).strip()}",
                action=f"Aura replied: {str(aura_response).strip()}",
                outcome="Conversation changed shared context and should inform future continuity.",
                success=True,
                emotional_valence=emotional_valence,
                tools_used=["conversation"],
                lessons=[
                    f"User interaction classified as {analysis.intent_type.lower()}:{analysis.semantic_mode}.",
                    "Conversational exchanges should remain available as lived context, not just transcript data.",
                ],
                importance=importance,
            )
    except Exception as exc:
        logger.debug("Episodic conversation recording skipped: %s", exc)

    try:
        entity_graph = service_access.optional_service("entity_graph", "relationship_graph", default=None)
        if entity_graph and hasattr(entity_graph, "register_interaction"):
            await entity_graph.register_interaction("aura_self", user_id, "conversation", "self", "person")
    except Exception as exc:
        logger.debug("Relationship graph update skipped: %s", exc)

    try:
        user_model = service_access.optional_service("user_model", "theory_of_mind_user_model", default=None)
        if user_model and hasattr(user_model, "update_from_interaction"):
            user_model.update_from_interaction(user_input, aura_response, {"source": "chat_api"})
    except Exception as exc:
        logger.debug("User model update skipped: %s", exc)

    try:
        learner = service_access.optional_service("continuous_learning", "continuous_learning_engine", default=None)
        if learner and hasattr(learner, "record_interaction"):
            await learner.record_interaction(
                user_input=user_input,
                aura_response=aura_response,
                user_name=user_id,
                domain="conversation",
                strategy="dialogic_exchange",
            )
    except Exception as exc:
        logger.debug("Continuous learning update skipped: %s", exc)

    try:
        bryan_model = service_access.optional_service(
            "bryan_model_engine",
            "bryan_model",
            "user_model_engine",
            default=None,
        )
        if bryan_model and hasattr(bryan_model, "_model"):
            bryan_model._model.total_messages += 2
            bryan_model._model.conversation_count += 1
            if analysis.semantic_mode in {"technical", "critical", "philosophical"} and hasattr(
                bryan_model, "observe_pattern"
            ):
                bryan_model.observe_pattern(
                    f"Bryan often brings {analysis.semantic_mode} conversation into the foreground."
                )
            if hasattr(bryan_model, "save"):
                bryan_model.save()
    except Exception as exc:
        logger.debug("Bryan model update skipped: %s", exc)
