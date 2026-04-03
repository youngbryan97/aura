from __future__ import annotations

import logging
import time
from typing import Any

from core.runtime import service_access

logger = logging.getLogger("Aura.ConversationSupport")


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

    return blocks


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
