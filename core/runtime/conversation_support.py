from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.runtime import service_access
from core.runtime.coding_session_memory import (
    build_coding_context_block,
    get_coding_session_memory,
)
from core.runtime.errors import record_degradation
from core.runtime.principal_context import (
    current_relational_principal,
    relational_principal_scope_is_bound,
)
from core.runtime.task_ownership import create_tracked_task
from core.runtime.turn_analysis import analyze_turn

logger = logging.getLogger("Aura.ConversationSupport")


def _record_conversation_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("conversation_support", exc, severity=severity, action=action)


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


def _normalize_agent_id(value: Any) -> str | None:
    """Canonical identity key: whitespace-collapsed, bounded, CASE-FOLDED.

    'Bryan' and 'bryan' must resolve to ONE relational identity — cased
    variants were forking profile/dialogue/relational-memory keys for the
    same person.
    """
    normalized = " ".join(str(value or "").strip().split())[:160].lower()
    return normalized or None


def resolve_exact_partner_id(state: Any) -> str | None:
    """Resolve a live partner without inventing a cross-session identity."""

    try:
        estimator = service_access.optional_service("other_agent_model", default=None)
        active_agent = _normalize_agent_id(getattr(estimator, "active_agent_id", ""))
        if active_agent:
            return active_agent

        cognition = getattr(state, "cognition", None)
        current_partner = _normalize_agent_id(getattr(cognition, "current_partner", ""))
        if current_partner:
            return current_partner
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="left exact conversation partner unresolved after identity lookup failed",
        )
        logger.debug("Exact conversation partner lookup failed: %s", exc)
    return None


def resolve_primary_user_id(state: Any) -> str:
    exact_partner = resolve_exact_partner_id(state)
    if exact_partner:
        return exact_partner

    try:
        world = getattr(state, "world", None)
        for collection_name in ("relationship_graph", "known_entities"):
            collection = getattr(world, collection_name, {}) or {}
            if isinstance(collection, dict) and len(collection) == 1:
                normalized = _normalize_agent_id(next(iter(collection)))
                if normalized:
                    return normalized
    except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
        _record_conversation_degradation(
            _exc,
            action="fell back to local user id after exact partner lookup failed",
        )
        logger.debug("Suppressed Exception: %s", _exc)
    return "local_user"


def relational_memory_allows(user_id: str, kind: str, operation: str) -> bool:
    try:
        authority = service_access.optional_service("relational_memory", default=None)
        return bool(
            authority
            and hasattr(authority, "allows")
            and authority.allows(user_id, kind, operation)
        )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="excluded relational profile data after consent lookup failed",
        )
        return False


def relationship_topology_consent_allows(agent_id: str | None) -> bool:
    """Require the complete session topology authority, not identity alone."""
    exact_id = _normalize_agent_id(agent_id)
    if exact_id is None:
        return False
    return relational_memory_allows(
        exact_id,
        "shared_ground",
        "recall",
    ) and relational_memory_allows(
        exact_id,
        "shared_ground",
        "prompt",
    )


async def record_shared_ground_callbacks(
    response_text: str,
    *,
    agent_id: str | None = None,
) -> None:
    try:
        from core.memory.shared_ground import get_shared_ground

        shared_ground = get_shared_ground()
        exact_agent_id = _normalize_agent_id(agent_id) or _normalize_agent_id(
            shared_ground.active_agent_id
        )
        if exact_agent_id is None:
            logger.debug("Shared-ground callback skipped: no exact active partner.")
            return

        entries = shared_ground.get_top_entries(
            shared_ground.MAX_ENTRIES,
            agent_id=exact_agent_id,
        )
        resp_lower = response_text.lower()
        for entry in entries:
            ref_words = entry.reference.lower().split()
            matches = sum(1 for word in ref_words if len(word) > 3 and word in resp_lower)
            if matches >= 2:
                shared_ground.record_callback(
                    entry.reference,
                    agent_id=exact_agent_id,
                )
                logger.debug("SharedGround callback recorded: %s", entry.reference)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
        _record_conversation_degradation(
            _exc,
            action="skipped shared-ground callback recording after shared-ground store failed",
        )
        logger.debug("Suppressed Exception: %s", _exc)


def build_conversational_context_blocks(state: Any, objective: str = "") -> list[str]:
    user_id = resolve_primary_user_id(state)
    priority_blocks: list[str] = []
    blocks: list[str] = []

    try:
        profiler = service_access.optional_service("conversational_profiler", default=None)
        if (
            profiler
            and hasattr(profiler, "get_context_injection")
            and relational_memory_allows(user_id, "derived_profile", "prompt")
        ):
            profile_block = profiler.get_context_injection(user_id)
            if profile_block:
                blocks.append(profile_block)
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without conversational profile block",
        )
        logger.debug("ConversationalProfile injection failed: %s", exc)

    try:
        # What she knows about this person, rendered from the typed store rather
        # than from a prose block. The distinction is the whole point: this text
        # is generated from records with fields, so a qualifier cannot go missing
        # between what she observed and what she reads back — there is no
        # summarisation step in which it could.
        #
        # Gated here for the same reason as the profile block above; the store
        # re-checks consent itself, because it is reachable from more than one
        # seam and a gate that lives only at the seams is one the next seam
        # forgets.
        interpersonal = service_access.optional_service("interpersonal_memory", default=None)
        if (
            interpersonal
            and hasattr(interpersonal, "render")
            and relational_memory_allows(user_id, "derived_profile", "prompt")
        ):
            interpersonal_block = interpersonal.render(user_id)
            if interpersonal_block:
                blocks.append(interpersonal_block)
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without the interpersonal block",
        )
        logger.debug("Interpersonal injection failed: %s", exc)

    try:
        dialogue = service_access.resolve_dialogue_cognition(default=None)
        if dialogue and hasattr(dialogue, "get_context_injection"):
            source_ids = (
                dialogue.default_source_ids() if hasattr(dialogue, "default_source_ids") else None
            )
            if relational_memory_allows(user_id, "dialogue_preference", "prompt"):
                dialogue_block = dialogue.get_context_injection(
                    user_id,
                    current_text=objective or "",
                    source_ids=source_ids,
                )
            elif hasattr(dialogue, "get_source_context_injection"):
                dialogue_block = dialogue.get_source_context_injection(source_ids)
            else:
                dialogue_block = ""
            if dialogue_block:
                blocks.append(dialogue_block)
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without dialogue cognition block",
        )
        logger.debug("DialogueCognition injection failed: %s", exc)

    try:
        humor = service_access.optional_service("humor_engine", default=None)
        if humor and relational_memory_allows(user_id, "style_preference", "prompt"):
            humor_guide = humor.get_humor_guidance(user_id)
            if humor_guide:
                blocks.append(humor_guide)
            banter = humor.get_banter_directive(user_id)
            if banter:
                blocks.append(banter)
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without humor guidance block",
        )
        logger.debug("HumorEngine injection failed: %s", exc)

    try:
        conv_intel = service_access.optional_service("conversation_intelligence", default=None)
        if conv_intel and hasattr(conv_intel, "get_context_injection"):
            ci_block = conv_intel.get_context_injection()
            if ci_block:
                blocks.append(ci_block)
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without conversation intelligence block",
        )
        logger.debug("ConversationIntelligence injection failed: %s", exc)

    try:
        rel_intel = service_access.optional_service("relational_intelligence", default=None)
        if (
            rel_intel
            and hasattr(rel_intel, "get_context_injection")
            and relational_memory_allows(user_id, "derived_profile", "prompt")
        ):
            ri_block = rel_intel.get_context_injection(user_id)
            if ri_block:
                blocks.append(ri_block)
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without relational intelligence block",
        )
        logger.debug("RelationalIntelligence injection failed: %s", exc)

    try:
        social_imagination = service_access.resolve_social_imagination(default=None)
        if (
            social_imagination
            and hasattr(social_imagination, "get_context_injection")
            and relational_memory_allows(user_id, "social_imagination", "prompt")
        ):
            si_block = social_imagination.get_context_injection(
                user_id,
                current_text=objective or "",
            )
            if si_block:
                blocks.append(si_block)
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without social imagination block",
        )
        logger.debug("SocialImagination injection failed: %s", exc)

    try:
        joy_social = service_access.optional_service("joy_social", default=None)
        if joy_social and hasattr(joy_social, "get_context_injection"):
            joy_block = joy_social.get_context_injection()
            if joy_block:
                blocks.append(joy_block)
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without joy/social context block",
        )
        logger.debug("JoySocial context injection failed: %s", exc)

    try:
        coding_block = build_coding_context_block(objective or "")
        if coding_block:
            priority_blocks.append(coding_block)
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without coding session context block",
        )
        logger.debug("Coding session context injection failed: %s", exc)

    try:
        from core.agency.task_commitment_verifier import get_task_commitment_verifier

        verifier = get_task_commitment_verifier()
        if verifier and hasattr(verifier, "get_context_block"):
            task_block = verifier.get_context_block(objective or "")
            if task_block and _is_task_context_priority(objective or ""):
                priority_blocks.append(task_block)
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without task verifier context block",
        )
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
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without goal engine context block",
        )
        logger.debug("Goal engine context injection failed: %s", exc)

    try:
        # Grounded self-knowledge: when the user asks why Aura was slow,
        # stuck, or restarted, inject the receipt-backed incident narrative so
        # she answers from forensics instead of confabulating. The narrator
        # gates internally on incident-shaped questions, so ordinary turns pay
        # nothing here.
        from core.observability.incident_narrator import get_incident_narrator

        incident_block = get_incident_narrator().get_context_injection(objective or "")
        if incident_block:
            priority_blocks.append(incident_block)
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError, ImportError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without incident self-knowledge block",
        )
        logger.debug("Incident narrator injection failed: %s", exc)

    try:
        # Grounded learning self-knowledge: when the user asks what Aura has
        # been learning or practicing, inject the receipt-backed status from
        # the lineage ledger / flywheel / scheduler so claims about her own
        # learning are the same numbers the API serves. Gates internally on
        # learning-shaped questions; ordinary turns pay nothing.
        from core.learning.learning_selfreport import get_learning_selfreport

        learning_block = get_learning_selfreport().get_context_injection(objective or "")
        if learning_block:
            priority_blocks.append(learning_block)
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError, ImportError) as exc:
        _record_conversation_degradation(
            exc,
            action="continued context assembly without learning self-knowledge block",
        )
        logger.debug("Learning self-report injection failed: %s", exc)

    # One section per header.
    #
    # Two subsystems can inject the same kind of block — the goal engine is
    # asked for its execution state by more than one assembler — and the prompt
    # then carried "## GOAL EXECUTION STATE" twice, at 1,966 and 1,078
    # characters, in a 46,996-character system message. Neither copy was wrong
    # and the second cost a thousand characters of a prompt the model reads at
    # about twelve tokens a second.
    #
    # The first wins: priority blocks are assembled first and are the ones
    # chosen for this objective.
    seen: set[str] = set()
    kept: list[str] = []
    for block in priority_blocks + blocks:
        header = str(block or "").lstrip().split("\n", 1)[0].strip()
        if header and header in seen:
            logger.debug("Dropped a repeated context section: %s", header[:60])
            continue
        if header:
            seen.add(header)
        kept.append(block)
    return kept


async def update_conversational_intelligence(
    user_input: str,
    aura_response: str,
    state: Any,
    *,
    agent_id: str | None = None,
) -> None:
    exact_agent_id = _normalize_agent_id(agent_id) or resolve_exact_partner_id(state)

    try:
        profiler = service_access.optional_service("conversational_profiler", default=None)
        if exact_agent_id and profiler and relational_memory_allows(
            exact_agent_id,
            "derived_profile",
            "recall",
        ):
            await profiler.update_from_interaction(
                exact_agent_id,
                user_input,
                aura_response,
                {},
            )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="skipped conversational profile update for this exchange",
        )
        logger.debug("ConversationalProfile update skipped: %s", exc)

    try:
        # Dialogue cognition may cold-start encrypted relational memory. Keep
        # native secure-store and file initialization off the request loop so
        # post-turn learning cannot stall model admission or UI health polls.
        dialogue = await asyncio.to_thread(
            service_access.resolve_dialogue_cognition,
            default=None,
        )
        if exact_agent_id and dialogue and relational_memory_allows(
            exact_agent_id,
            "dialogue_preference",
            "recall",
        ):
            await dialogue.update_from_interaction(
                exact_agent_id,
                user_input,
                aura_response,
                {},
            )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="skipped dialogue cognition update for this exchange",
        )
        logger.debug("DialogueCognition update skipped: %s", exc)

    try:
        humor = service_access.optional_service("humor_engine", default=None)
        if exact_agent_id and humor and relational_memory_allows(
            exact_agent_id,
            "style_preference",
            "recall",
        ):
            dynamics = service_access.resolve_conversational_dynamics(default=None)
            if dynamics:
                humor.update_banter_state(
                    user_input,
                    dynamics.get_current_state(),
                    user_id=exact_agent_id,
                )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="skipped humor banter-state update for this exchange",
        )
        logger.debug("HumorEngine update skipped: %s", exc)

    try:
        conv_intel = service_access.optional_service("conversation_intelligence", default=None)
        if conv_intel:
            dynamics = service_access.resolve_conversational_dynamics(default=None)
            dynamics_state = dynamics.get_current_state() if dynamics else None
            discourse_state = (
                {
                    "topic": getattr(state.cognition, "discourse_topic", None),
                    "depth": getattr(state.cognition, "discourse_depth", 0),
                    "energy": getattr(state.cognition, "conversation_energy", 0.5),
                    "trend": getattr(state.cognition, "user_emotional_trend", "neutral"),
                }
                if state
                else {}
            )
            await conv_intel.update(user_input, aura_response, dynamics_state, discourse_state)
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_conversation_degradation(
            exc,
            action="skipped conversation intelligence update for this exchange",
        )
        logger.debug("ConversationIntelligence update skipped: %s", exc)

    try:
        rel_intel = service_access.optional_service("relational_intelligence", default=None)
        if exact_agent_id and rel_intel and relational_memory_allows(
            exact_agent_id,
            "derived_profile",
            "recall",
        ):
            dynamics = service_access.resolve_conversational_dynamics(default=None)
            dynamics_state = dynamics.get_current_state() if dynamics else None
            await rel_intel.update_from_interaction(
                exact_agent_id,
                user_input,
                aura_response,
                dynamics_state,
            )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="skipped relational intelligence update for this exchange",
        )
        logger.debug("RelationalIntelligence update skipped: %s", exc)

    try:
        social_imagination = service_access.resolve_social_imagination(default=None)
        if exact_agent_id and social_imagination and relational_memory_allows(
            exact_agent_id,
            "social_imagination",
            "recall",
        ):
            await social_imagination.update_from_interaction(
                exact_agent_id,
                user_input,
                aura_response,
                {},
            )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="skipped social imagination update for this exchange",
        )
        logger.debug("SocialImagination update skipped: %s", exc)


async def _run_conversation_support_updates(
    user_input: str,
    aura_response: str,
    state: Any,
    *,
    agent_id: str | None,
) -> None:
    await update_conversational_intelligence(
        user_input,
        aura_response,
        state,
        agent_id=agent_id,
    )
    await record_shared_ground_callbacks(
        aura_response,
        agent_id=agent_id,
    )


def schedule_conversation_support_updates(
    user_input: str,
    aura_response: str,
    state: Any,
) -> asyncio.Task[Any] | None:
    """Schedule one bounded, named owner for post-generation social updates."""

    if not str(user_input or "").strip() or not str(aura_response or "").strip():
        return None
    exact_agent_id = resolve_exact_partner_id(state)
    awaitable = _run_conversation_support_updates(
        str(user_input),
        str(aura_response),
        state,
        agent_id=exact_agent_id,
    )
    try:
        return create_tracked_task(
            awaitable,
            name="conversation_support.turn_updates",
            owner="response_generation",
            bounded=True,
        )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="skipped post-generation conversation updates after owned task scheduling failed",
        )
        return None


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


def _clip_continuity_text(text: Any, *, limit: int = 520) -> str:
    clipped = " ".join(str(text or "").strip().split())
    if len(clipped) <= limit:
        return clipped
    return clipped[: max(0, limit - 1)].rstrip() + "..."


def _resolve_conversation_user_id(
    exact_agent_id: Any,
    *,
    explicit_principal: bool,
    scoped_principal: bool,
    state_obj: Any,
) -> str:
    """Who this turn is attributed to.

    Extracted so the continuity-only path and the full learning path cannot
    attribute the same exchange to different people. An explicitly
    identity-less request must never be re-attributed from mutable process
    state, which is why the fallback is a named unattributed session rather
    than whoever the runtime happens to think is present.
    """
    return exact_agent_id or (
        "unattributed_session"
        if explicit_principal or scoped_principal
        else resolve_primary_user_id(state_obj)
    )


async def _record_continuity_only(
    user_input: str,
    aura_response: str,
    *,
    user_id: str,
    reasons: tuple[str, ...],
) -> bool:
    """Remember that the exchange happened, without learning from the reply.

    Written when the learning gate refuses. The metadata says plainly that
    this reply was NOT admitted as experience, so a retrieval path looking
    for examples of good answers can exclude it while a retrieval path
    looking for "what did we talk about" still finds it.
    """
    try:
        memory_facade = service_access.optional_service("memory_facade", default=None)
        if memory_facade is None or not hasattr(memory_facade, "add_memory"):
            return False
        continuity_text = _build_conversation_continuity_memory(
            user_input,
            aura_response,
            user_id=user_id,
        )
        result = memory_facade.add_memory(
            continuity_text,
            metadata={
                "origin": "api",
                "source": "chat_api",
                "domain": "conversation",
                "memory_type": "conversation_continuity",
                "conversation_turn": True,
                "preserve_for_continuity": True,
                "searchable_conversation_context": True,
                "user_id": user_id,
                "user_utterance": _clip_continuity_text(user_input, limit=700),
                "aura_response": _clip_continuity_text(aura_response, limit=700),
                "provenance_source": "live_conversation_turn",
                "confidence": 1.0,
                # The two fields that keep this out of the priming loop.
                "learning_admission": "refused",
                "reply_is_exemplary": False,
                "reply_quality_reasons": list(reasons),
                "importance": 0.5,
            },
        )
        if hasattr(result, "__await__"):
            result = await result
        return bool(result)
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        _record_conversation_degradation(
            exc,
            action="lost the continuity record for a turn that was not admitted as learning",
        )
        return False


def _build_conversation_continuity_memory(
    user_input: str,
    aura_response: str,
    *,
    user_id: str,
) -> str:
    return (
        "Conversation continuity memory. "
        f"User({user_id}) said: {_clip_continuity_text(user_input)} "
        f"Aura replied: {_clip_continuity_text(aura_response)}"
    )


async def record_conversation_experience(
    user_input: str,
    aura_response: str,
    state: Any = None,
    *,
    principal_id: str | None = None,
) -> None:
    if not str(user_input or "").strip() or not str(aura_response or "").strip():
        return
    try:
        from core.conversation.response_reliability import (
            assess_conversation_learning_admission,
            is_self_condition_turn,
        )

        learning_admission = assess_conversation_learning_admission(
            user_input,
            aura_response,
        )
        if not learning_admission.ok:
            logger.warning(
                "Not learning from this reply (%s); recording the exchange for "
                "continuity without treating the reply as an example.",
                ",".join(learning_admission.reasons) or "unknown",
            )
            # The rationale for refusing to LEARN here is sound and evidenced:
            # a repaired or rejected reply stored as experience comes back as
            # memory evidence and primes the model to repeat the broken shape
            # (live 2026-07-27, the truncated marbles answer, turn after turn).
            #
            # What was wrong is the scope. Refusing to learn from a bad reply
            # also threw away the USER'S half of the exchange — that they asked
            # at all, and what about. Seen in the 2026-07-30 demo: Aura answered
            # a desktop request in her own internal vocabulary, the reply
            # tripped pseudo_internal_jargon, and the entire turn vanished from
            # continuity. The worse her wording, the less she remembered of the
            # conversation.
            #
            # So: no experience commit, but the continuity record still
            # happens — marked so nothing can retrieve it as a model of how to
            # answer — and ONLY when every objection is about wording. A
            # grounding failure (host telemetry offered as a feeling, say) is
            # not merely badly worded, and its record would be retrieved as
            # evidence about the state it got wrong.
            from core.conversation.surface_disposition import (
                CONTINUITY_SAFE_REASONS,
            )

            wording_only = bool(learning_admission.reasons) and set(
                learning_admission.reasons
            ) <= CONTINUITY_SAFE_REASONS
            if not wording_only:
                return
            refused_state = state
            if refused_state is None:
                repo = service_access.resolve_state_repository(default=None)
                refused_state = getattr(repo, "_current", None) if repo else None
            await _record_continuity_only(
                user_input,
                aura_response,
                user_id=_resolve_conversation_user_id(
                    _normalize_agent_id(
                        principal_id
                        if principal_id is not None
                        else current_relational_principal()
                    )
                    if (
                        principal_id is not None
                        or relational_principal_scope_is_bound()
                    )
                    else resolve_exact_partner_id(refused_state),
                    explicit_principal=principal_id is not None,
                    scoped_principal=relational_principal_scope_is_bound(),
                    state_obj=refused_state,
                ),
                reasons=learning_admission.reasons,
            )
            return
        self_condition_grounded = bool(is_self_condition_turn(user_input))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="failed conversational learning admission closed while preserving transcript storage",
        )
        return

    state_obj = state
    if state_obj is None:
        repo = service_access.resolve_state_repository(default=None)
        state_obj = getattr(repo, "_current", None) if repo is not None else None

    explicit_principal = principal_id is not None
    scoped_principal = relational_principal_scope_is_bound()
    exact_agent_id = _normalize_agent_id(
        principal_id if explicit_principal else current_relational_principal()
    )
    if exact_agent_id is None and not explicit_principal and not scoped_principal:
        exact_agent_id = resolve_exact_partner_id(state_obj)
    # An explicitly identity-less request must never be re-attributed from
    # mutable process state. Non-request callers retain the legacy state
    # resolver until they pass a causal principal of their own.
    user_id = _resolve_conversation_user_id(
        exact_agent_id,
        explicit_principal=explicit_principal,
        scoped_principal=scoped_principal,
        state_obj=state_obj,
    )
    importance = _conversation_importance(user_input)
    emotional_valence = _conversation_emotional_valence(user_input)
    analysis = analyze_turn(user_input)

    await update_conversational_intelligence(
        user_input,
        aura_response,
        state_obj,
        agent_id=exact_agent_id,
    )
    await record_shared_ground_callbacks(
        aura_response,
        agent_id=exact_agent_id,
    )

    try:
        get_coding_session_memory().record_conversation_turn(
            user_input,
            aura_response,
            analysis=analysis,
        )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_conversation_degradation(
            exc,
            action="skipped coding session turn recording for this exchange",
        )
        logger.debug("Coding session turn recording skipped: %s", exc)

    used_memory_facade = False
    memory_commit_result = None
    try:
        memory_facade = service_access.optional_service("memory_facade", default=None)
        if memory_facade and hasattr(memory_facade, "commit_interaction"):
            used_memory_facade = True
            memory_commit_result = await memory_facade.commit_interaction(
                context=str(user_input).strip(),
                action="conversation_reply",
                outcome=str(aura_response).strip(),
                success=True,
                emotional_valence=emotional_valence,
                importance=importance,
                metadata={
                    "origin": "api",
                    "source": "chat_api",
                    "domain": "conversation",
                    "objective": str(user_input).strip(),
                    "intent_type": str(analysis.intent_type).lower(),
                    "semantic_mode": str(analysis.semantic_mode),
                    "memory_salience": round(float(importance), 4),
                    "conversation_turn": True,
                    # Who said which half. `context` is the person's words and
                    # `outcome` is Aura's; without this the record recalls as
                    # one voice and the "I" in it lands on whoever reads it.
                    "context_speaker": "user",
                    "outcome_speaker": "aura",
                    "preserve_for_continuity": True,
                    "learning_admission": "verified",
                    "self_condition_grounded": self_condition_grounded,
                },
            )
            
            # Check if facade actually saved it (not deferred or blocked)
            if memory_commit_result is None:
                logger.warning("⚠️ Memory facade commit_interaction returned None (governance/deferral may have blocked it). Falling back to episodic...")
                used_memory_facade = False
            else:
                logger.debug(f"✓ Conversation turn saved to memory facade: {memory_commit_result}")
                if hasattr(memory_facade, "add_memory"):
                    continuity_text = _build_conversation_continuity_memory(
                        user_input,
                        aura_response,
                        user_id=user_id,
                    )
                    continuity_result = memory_facade.add_memory(
                        continuity_text,
                        metadata={
                            "origin": "api",
                            "source": "chat_api",
                            "domain": "conversation",
                            "memory_type": "conversation_continuity",
                            "conversation_turn": True,
                            "preserve_for_continuity": True,
                            "searchable_conversation_context": True,
                            "user_id": user_id,
                            "user_utterance": _clip_continuity_text(user_input, limit=700),
                            "aura_response": _clip_continuity_text(aura_response, limit=700),
                            "intent_type": str(analysis.intent_type).lower(),
                            "semantic_mode": str(analysis.semantic_mode),
                            "importance": min(0.95, max(importance, 0.62)),
                            "provenance_source": "live_conversation_turn",
                            "confidence": 1.0,
                            "learning_admission": "verified",
                            "self_condition_grounded": self_condition_grounded,
                        },
                    )
                    if hasattr(continuity_result, "__await__"):
                        continuity_result = await continuity_result
                    if not bool(continuity_result):
                        logger.debug("Conversation continuity add_memory returned false after commit.")
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_conversation_degradation(
            exc,
            action="fell back from memory facade to episodic conversation recording",
        )
        logger.debug("Conversation memory facade commit skipped: %s", exc)

    if not used_memory_facade:
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
        except (RuntimeError, AttributeError, TypeError) as exc:
            _record_conversation_degradation(
                exc,
                action="skipped episodic conversation recording after memory facade was unavailable",
            )
            logger.debug("Episodic conversation recording skipped: %s", exc)

    if relationship_topology_consent_allows(exact_agent_id):
        try:
            from core.social.relationship_graph import RelationshipConsentRequiredError

            entity_graph = service_access.optional_service(
                "entity_graph", "relationship_graph", default=None
            )
            if entity_graph and hasattr(entity_graph, "register_interaction"):
                await entity_graph.register_interaction(
                    "aura_self", exact_agent_id, "conversation", "self", "person"
                )
        except RelationshipConsentRequiredError:
            # Consent can expire or be revoked between admission and mutation.
            # That is a governed abstention, not a runtime health failure.
            logger.debug(
                "Relationship topology update abstained after consent changed for %s.",
                exact_agent_id,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError) as exc:
            _record_conversation_degradation(
                exc,
                action="skipped relationship graph update for this exchange",
            )
            logger.debug("Relationship graph update skipped: %s", exc)
    elif exact_agent_id is None and user_id not in (
        "local_user",
        "unattributed_session",
    ):
        # No exact THIRD-PARTY partner, but the turn has a resolved owner
        # (Bryan). The aura↔owner edge is the owner's own relationship with
        # their AI — they are its data controller, so it needs no third-party
        # relational-memory grant (that gate protects other people's data).
        # A consent-DENIED exact partner never reaches here: exact_agent_id
        # is non-None in that case, so this branch is skipped.
        try:
            from core.social.relationship_graph import RelationshipConsentRequiredError

            entity_graph = service_access.optional_service(
                "entity_graph", "relationship_graph", default=None
            )
            if entity_graph and hasattr(entity_graph, "register_interaction"):
                await entity_graph.register_interaction(
                    "aura_self", user_id, "conversation", "self", "person"
                )
        except RelationshipConsentRequiredError:
            logger.debug(
                "Owner relationship edge abstained after consent changed for %s.",
                user_id,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError) as exc:
            _record_conversation_degradation(
                exc,
                action="skipped owner relationship edge for this exchange",
            )
            logger.debug("Owner relationship edge skipped: %s", exc)
    else:
        logger.debug(
            "Relationship topology update abstained: exact-agent recall/prompt consent absent."
        )

    try:
        user_model = service_access.optional_service(
            "user_model", "theory_of_mind_user_model", default=None
        )
        if user_model and hasattr(user_model, "update_from_interaction"):
            user_model.update_from_interaction(user_input, aura_response, {"source": "chat_api"})
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_conversation_degradation(
            exc,
            action="skipped user model update for this exchange",
        )
        logger.debug("User model update skipped: %s", exc)

    try:
        learner = service_access.optional_service(
            "continuous_learning", "continuous_learning_engine", default=None
        )
        if learner and hasattr(learner, "record_interaction"):
            await learner.record_interaction(
                user_input=user_input,
                aura_response=aura_response,
                user_name=user_id,
                domain="conversation",
                strategy="dialogic_exchange",
            )
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_conversation_degradation(
            exc,
            action="skipped continuous learning update for this exchange",
        )
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
    except (RuntimeError, AttributeError, TypeError) as exc:
        _record_conversation_degradation(
            exc,
            action="skipped Bryan model counters update for this exchange",
        )
        logger.debug("Bryan model update skipped: %s", exc)
