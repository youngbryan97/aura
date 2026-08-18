from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation

from ..consciousness.executive_authority import get_executive_authority
from ..state.aura_state import AuraState
from . import BasePhase

logger = logging.getLogger(__name__)

_MEMORY_CONSOLIDATION_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_memory_consolidation_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        record_degradation(
            "memory_consolidation",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation("memory_consolidation", error, severity=severity, action=action)
        except TypeError:
            logger.debug("Memory consolidation degradation could not be recorded: %s", signature_exc)


class MemoryConsolidationPhase(BasePhase):
    """
    Phase 6: Memory Consolidation.
    Commits recent interactions and insights to long-term storage (RAG).
    Ensures that the experience is persisted beyond working memory.
    """
    
    def __init__(self, container: Any):
        self.container = container

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        if parsed != parsed:
            return default
        return max(0.0, min(1.0, parsed))

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        """
        Persist recent interactions to long-term storage and prune working memory.

        Detects completed user/assistant turns (or high-arousal forced consolidation),
        optionally distils content through the SovereignPruner, queues a knowledge
        evolution entry on the ColdStore, detects and degrades stability on
        conversational loops, and caps working memory at max_working_memory entries.
        """
        # Pure Transformation: Stop direct side-effects.
        # Create a derived state for any modifications.
        new_state = await state.derive_async(cause="memory_consolidation_cycles", origin="MemoryConsolidationPhase")

        # 0. Defensive Hygiene: Filter out non-dict trash from working_memory
        # This prevents AttributeError if objects (like the Orchestrator) leak in.
        clean_memory = [m for m in new_state.cognition.working_memory if isinstance(m, dict)]
        if len(clean_memory) != len(new_state.cognition.working_memory):
            logger.warning("💾 MemoryConsolidation: Dropped %d non-dict items from working memory.", len(new_state.cognition.working_memory) - len(clean_memory))
            new_state.cognition.working_memory = clean_memory

        try:
            from core.runtime.proof_policy import is_strict_proof_answer_prompt

            proof_origin = getattr(new_state.cognition, "current_origin", None) or kwargs.get("origin")
            proof_text = objective or ""
            for item in reversed(new_state.cognition.working_memory):
                if isinstance(item, dict) and item.get("role") == "user":
                    proof_origin = item.get("origin") or proof_origin
                    proof_text = str(item.get("content", "") or proof_text)
                    break
            if is_strict_proof_answer_prompt(proof_text, origin=proof_origin):
                new_state.cognition.long_term_memory = []
                new_state.response_modifiers["proof_memory_consolidation_skipped"] = True
                return new_state
        except _MEMORY_CONSOLIDATION_ERRORS as exc:
            self._mark_consolidation_status(
                new_state,
                status="partial",
                stage="strict_proof_consolidation_guard",
                error=exc,
            )
            _record_memory_consolidation_degradation(
                exc,
                action="continued memory consolidation after strict proof guard failed",
                severity="warning",
                extra={"stage": "strict_proof_consolidation_guard"},
            )

        # ISSUE-81: Consolidation Skip Fix
        # Allow consolidation if there's high arousal or a pending action,
        # even if the turn is not strictly completed.
        response_modifiers = dict(getattr(new_state, "response_modifiers", {}) or {})
        imagination_memory_pressure = self._safe_float(
            response_modifiers.get("imagination_memory_pressure")
        )
        bicameral_causal_effects = response_modifiers.get("bicameral_causal_effects")
        if not isinstance(bicameral_causal_effects, dict):
            bicameral_causal_effects = {}
        bicameral_memory_priority = self._safe_float(
            response_modifiers.get("bicameral_memory_priority")
            or bicameral_causal_effects.get("memory_priority")
        )
        force_consolidation = (
            new_state.affect.arousal > 0.8
            or imagination_memory_pressure > 0.72
            or bicameral_memory_priority > 0.72
            or (
                len(new_state.cognition.working_memory) > 0
                and new_state.cognition.working_memory[-1].get("action")
            )
        )

        # ── Consciousness-driven consolidation triggers ──
        # GWT ignition (high-priority broadcast) → force consolidation (significant event)
        # High prediction surprise → force consolidation (unexpected = memorable)
        # Low free energy + rest action → ideal time for background consolidation
        if not force_consolidation:
            try:
                from core.container import ServiceContainer
                gw = ServiceContainer.get("global_workspace", default=None)
                if gw and hasattr(gw, "is_ignited") and gw.is_ignited():
                    force_consolidation = True  # GWT ignition = significant cognitive event
                    logger.debug("💾 Force consolidation: GWT ignition detected")
                fe = ServiceContainer.get("free_energy_engine", default=None)
                if fe and fe.current:
                    if fe.current.surprise > 0.7:
                        force_consolidation = True  # High surprise = memorable event
                        logger.debug("💾 Force consolidation: high surprise (%.2f)", fe.current.surprise)
            except _MEMORY_CONSOLIDATION_ERRORS as exc:
                self._mark_consolidation_status(
                    new_state,
                    status="partial",
                    stage="consolidation_trigger_probe",
                    error=exc,
                )
                _record_memory_consolidation_degradation(
                    exc,
                    action="continued memory consolidation after optional trigger probe failed",
                    severity="warning",
                    extra={"stage": "consolidation_trigger_probe"},
                )
        
        if len(new_state.cognition.working_memory) < 1:
            return new_state
            
        # 1.1 Conversational Loop Detection (v46)
        # If the latest assistant message repeats a previous one, degrade stability to force shift.
        assistant_msgs = [m for m in new_state.cognition.working_memory if isinstance(m, dict) and m.get("role") == "assistant"]
        if len(assistant_msgs) >= 2:
            latest = str(assistant_msgs[-1].get("content", "")).strip()
            if assistant_msgs[-1].get("ephemeral"):
                latest = ""
            # vResilience: Workaround for slice limitations
            duplicate_assistant_ids: set[int] = set()
            latest_assistant_id = id(assistant_msgs[-1])
            for i in range(len(assistant_msgs) - 1):
                prev = assistant_msgs[i]
                if latest == str(prev.get("content", "")).strip() and len(latest) > 20:
                    duplicate_assistant_ids.add(id(prev))
                    logger.warning("🔄 [LOOP DETECTED] Assistant repeated content: '%s...'", latest[:30])
                    new_state.identity.stability = max(0.1, new_state.identity.stability - 0.3)
                    # CRITICAL FIX: Clear the stuck pending_initiatives that caused the loop.
                    # Without this, the same objective re-queues indefinitely.
                    stuck_count = len(new_state.cognition.pending_initiatives)
                    if stuck_count > 0:
                        logger.warning("🔄 [LOOP BREAK] Suppressing %d stuck pending_initiatives to escape loop.", stuck_count)
                        new_state, _ = await get_executive_authority().suppress_initiatives(
                            new_state,
                            predicate=lambda _item: True,
                            reason="loop_detected_repeated_assistant_output",
                            source="memory_consolidation",
                        )
                    break
            if duplicate_assistant_ids:
                # Keep the current user-facing turn intact. Memory consolidation
                # may prune older duplicate assistant messages, but it must not
                # erase the latest answer and cause the live chat path to see
                # "no assistant response" after Cortex succeeded.
                new_state.cognition.working_memory = [
                    m
                    for m in new_state.cognition.working_memory
                    if id(m) == latest_assistant_id or id(m) not in duplicate_assistant_ids
                ]
                new_state.response_modifiers["memory_consolidation_loop_signal"] = {
                    "duplicate_assistant_messages_pruned": len(duplicate_assistant_ids),
                    "latest_answer_preserved": True,
                }

        # vResilience: Workaround for slice limitations
        start_idx = max(0, len(new_state.cognition.working_memory) - 2)
        last_msgs = [new_state.cognition.working_memory[i] for i in range(start_idx, len(new_state.cognition.working_memory))]
        
        # Check for turn completion OR forced consolidation
        is_completed_turn = len(last_msgs) == 2 and last_msgs[0].get("role") == "user" and last_msgs[1].get("role") == "assistant"
        
        if not is_completed_turn and not force_consolidation:
            return new_state
            
        # 2. Extract content to store
        content = ""
        source = "conversation"
        if is_completed_turn:
            content = f"User: {last_msgs[0]['content']}\nAura: {last_msgs[1]['content']}"
        elif force_consolidation and len(new_state.cognition.working_memory) > 0:
            # If forced, consolidate the last message, especially if it's an action
            last_message = new_state.cognition.working_memory[-1]
            if last_message.get("ephemeral") and not last_message.get("action"):
                logger.debug("💾 MemoryConsolidation: Skipping ephemeral fallback message.")
                return new_state
            if last_message.get("role") == "assistant" and last_message.get("action"):
                content = f"Aura Action: {last_message['action']}"
            else:
                content = f"{last_message.get('role', 'unknown').capitalize()}: {last_message.get('content', '')}"
        
        if not content: # If no content was extracted, don't proceed with consolidation
            return new_state

        interaction_context = (
            str(last_msgs[0].get("content", "")) if is_completed_turn and last_msgs else (objective or "")
        )
        interaction_action = "conversation_reply" if is_completed_turn else "background_consolidation"
        interaction_outcome = (
            str(last_msgs[1].get("content", "")) if is_completed_turn and len(last_msgs) > 1 else content
        )
            
        # v40: Sovereign Pruner
        # Forget experience, keep insight. Protect contradictions.
        pruner = self.container.get("sovereign_pruner", default=None)
        if pruner:
            # v40: Dynamic weight from affect
            # High arousal plus strong valence at either pole should preserve
            # memories more aggressively than neutral-but-busy states.
            arousal = max(0.0, float(getattr(new_state.affect, "arousal", 0.0) or 0.0))
            valence_magnitude = abs(float(getattr(new_state.affect, "valence", 0.0) or 0.0))
            emotional_weight = min(1.0, arousal * (0.6 + 0.5 * valence_magnitude))
            
            # Convert dicts to MemoryRecords for the pruner
            from core.memory.sovereign_pruner import MemoryRecord
            records = [
                MemoryRecord(
                    id=str(uuid.uuid4()),
                    content=m["content"],
                    timestamp=m.get("timestamp", time.time()),
                    source="conversation",
                    emotional_weight=emotional_weight,
                    identity_relevance=0.7 # default
                ) for m in last_msgs
                if not m.get("ephemeral")
            ]
            if not records:
                logger.debug("💾 MemoryConsolidation: No durable messages qualified for sovereign pruning.")
                return new_state
            
            # v40: Pull importance weights from state
            values = {
                "Sovereignty": 0.9,
                "Curiosity": new_state.affect.curiosity,
                "Integrity": 0.8,
                "Autonomy": 0.9
            }
            if hasattr(new_state.identity, "narrative_version"):
                values["IdentityEvolution"] = 0.5 + (new_state.identity.narrative_version * 0.05)
            
            surviving, pruner_log = await pruner.prune(records, values)
            for entry in pruner_log:
                logger.debug("💾 [SovereignPruner] %s", entry)
            
            # Use distilled content for consolidation
            if surviving:
                content = "\n".join([m.content for m in surviving])

        memory_facade = self.container.get("memory_facade", default=None)
        if memory_facade and hasattr(memory_facade, "commit_interaction"):
            try:
                affect_signature = (
                    new_state.affect.get_cognitive_signature()
                    if hasattr(new_state.affect, "get_cognitive_signature")
                    else {}
                )
                salience = max(
                    float(affect_signature.get("memory_salience", 0.0) or 0.0),
                    imagination_memory_pressure,
                    bicameral_memory_priority,
                )
                complexity = float(affect_signature.get("affective_complexity", 0.0) or 0.0)
                bicameral_verification_pressure = self._safe_float(
                    response_modifiers.get("bicameral_verification_pressure")
                    or bicameral_causal_effects.get("verification_pressure")
                )
                importance = 0.85 if is_completed_turn else 0.65
                importance = max(
                    importance,
                    min(1.0, 0.4 + float(new_state.affect.arousal or 0.0) * 0.3 + salience * 0.3),
                )
                await memory_facade.commit_interaction(
                    context=interaction_context or (objective or "conversation"),
                    action=interaction_action,
                    outcome=interaction_outcome or content,
                    success=True,
                    emotional_valence=float(getattr(new_state.affect, "valence", 0.0) or 0.0),
                    importance=importance,
                    metadata={
                        "source": source,
                        "objective": str(objective or "")[:160],
                        "origin": str(getattr(new_state.cognition, "current_origin", "") or ""),
                        "dominant_emotion": affect_signature.get("dominant_emotion", getattr(new_state.affect, "dominant_emotion", "neutral")),
                        "top_emotions": list(affect_signature.get("top_emotions", []) or []),
                        "social_hunger": float(affect_signature.get("social_hunger", getattr(new_state.affect, "social_hunger", 0.0)) or 0.0),
                        "physiological_strain": float(affect_signature.get("physiological_strain", 0.0) or 0.0),
                        "affective_complexity": complexity,
                        "memory_salience": salience,
                        "affective_memory_salience": float(affect_signature.get("memory_salience", 0.0) or 0.0),
                        "imagination_memory_pressure": imagination_memory_pressure,
                        "bicameral_memory_priority": bicameral_memory_priority,
                        "imagination_verification_pressure": self._safe_float(
                            response_modifiers.get("imagination_verification_pressure")
                            or response_modifiers.get("verification_pressure")
                        ),
                        "bicameral_verification_pressure": bicameral_verification_pressure,
                        "resonance": affect_signature.get("resonance", getattr(new_state.affect, "get_resonance_string", lambda: "")()),
                    },
                )
                self._mark_consolidation_status(
                    new_state,
                    status="committed",
                    stage="memory_facade",
                )
            except _MEMORY_CONSOLIDATION_ERRORS as e:
                self._mark_consolidation_status(
                    new_state,
                    status="degraded",
                    stage="memory_facade",
                    error=e,
                )
                self._queue_failed_commit(
                    error=e,
                    content=content,
                    objective=objective,
                    interaction_context=interaction_context,
                    interaction_action=interaction_action,
                    interaction_outcome=interaction_outcome,
                )
                _record_memory_consolidation_degradation(
                    e,
                    action="preserved consolidation in ColdStore and queued failed memory facade commit for retry",
                    severity="warning",
                    extra={"stage": "memory_facade"},
                )
                logger.debug("MemoryConsolidation: MemoryFacade commit failed: %s", e)
 
        # Queue the knowledge for the ColdStore to process asynchronously.
        if new_state.cold is not None:
            new_state.cold.evolution_log.append({
                "type": "knowledge_addition",
                "content": content,
                "source": source,
                "timestamp": float(time.time())
            })
        
        # Enforce cap on evolution log
        from ..state.aura_state import MAX_EVOLUTION_LOG
        if new_state.cold is not None and len(new_state.cold.evolution_log) > MAX_EVOLUTION_LOG:
            # vResilience: Workaround for slice limitations
            start_log = len(new_state.cold.evolution_log) - MAX_EVOLUTION_LOG
            new_state.cold.evolution_log = [new_state.cold.evolution_log[i] for i in range(start_log, len(new_state.cold.evolution_log))]
        
        logger.debug("MemoryConsolidation: Queued knowledge evolution to ColdStore.")

        # 4. Intelligent Context Trimming (Claude Code pattern: two-pass compression)
        # Pass 1: Drop verbose tool/skill results first (they're already in episodic memory)
        # Pass 2: If still over limit, drop oldest non-user messages
        # Always preserve: most recent user message, system messages, high-importance episodes
        max_working_memory: int = 30
        wm = new_state.cognition.working_memory
        if len(wm) > max_working_memory:
            # Pass 1: Remove tool/skill result messages (most verbose, already persisted)
            trimmed = []
            dropped_tools = 0
            for msg in wm:
                if not isinstance(msg, dict):
                    continue
                content = str(msg.get("content", ""))
                metadata = msg.get("metadata", {}) or {}
                is_tool_result = (
                    str(metadata.get("type", "")).lower() in {"skill_result", "tool_result"}
                    or content.startswith("[SKILL RESULT:")
                    or content.startswith("[TOOL RESULT:")
                )
                if is_tool_result and len(trimmed) > 2:
                    dropped_tools += 1
                    continue
                trimmed.append(msg)

            if dropped_tools > 0:
                logger.info("🧹 Context trim pass 1: dropped %d tool results (%d→%d)", dropped_tools, len(wm), len(trimmed))
                wm = trimmed

            # Pass 2: If still over, keep most recent messages with bias toward user turns
            # IMPORTANT: Messages with >2000 chars of user content are treated as
            # "high-importance" (stories, code blocks, etc.) and are exempt from pruning
            if len(wm) > max_working_memory:
                # Always keep last 4 messages (current conversation turn)
                tail = wm[-4:]
                older = wm[:-4]
                # Choose what to keep by INDEX, then emit in the order it
                # happened.
                #
                # This selected the same messages and concatenated them as
                # `kept_user_and_large + kept_recent_non_user + tail`, which is
                # not a conversation. On a plain alternating exchange of 18
                # turns it produced:
                #
                #   U1 U2 U3 ... U16  A7 A8 ... A16  U17 A17 U18 A18
                #
                # Sixteen consecutive user messages, then ten consecutive
                # replies. Every answer was torn away from the question it
                # answered, and A1-A6 were dropped outright. What she reasoned
                # over was a transcript that never happened — answers appearing
                # to respond to whichever question happened to precede them
                # after the shuffle.
                #
                # Reordering also reshapes the KV prefix on every trim, so the
                # prompt cache could never reuse more than the system block:
                # measured live, "prefix diverges at token 226 (9% of 2561
                # reused)".
                # Kept as EXCHANGES, not as loose messages.
                #
                # Preferring user turns on their own kept questions and dropped
                # the answers, so the retained history contained runs of seven
                # consecutive user messages — a conversation in which she was
                # asked seven things and replied to none. She then reasons over
                # her own unanswered questions, which is its own invitation to
                # invent what was said.
                #
                # A question and the reply it drew are one unit of context, so
                # they are kept or dropped together.
                priority: list[int] = []
                for index, message in enumerate(older):
                    if not isinstance(message, dict):
                        continue
                    is_user = message.get("role") == "user"
                    is_large = len(str(message.get("content", ""))) > 2000
                    if not (is_user or is_large):
                        continue
                    priority.append(index)
                    if is_user:
                        answer = index + 1
                        if (
                            answer < len(older)
                            and isinstance(older[answer], dict)
                            and older[answer].get("role") == "assistant"
                        ):
                            priority.append(answer)
                # Sorted and de-duplicated: an answer can be reached both as a
                # large message and as its question's partner.
                priority = sorted(dict.fromkeys(priority))
                # The priority set alone can exceed the budget: a long
                # conversation is mostly user turns, so `remaining` went
                # NEGATIVE and every one of them was kept regardless. Forty
                # turns produced 42 retained messages against a limit of 30 —
                # the trim did nothing exactly when it was needed most, and the
                # context it was protecting kept growing.
                budget_for_older = max(0, max_working_memory - len(tail))
                if len(priority) > budget_for_older:
                    priority = priority[-budget_for_older:]
                keep_indices = set(priority)
                remaining = budget_for_older - len(keep_indices)
                if remaining > 0:
                    fill = [
                        index
                        for index, message in enumerate(older)
                        if isinstance(message, dict)
                        and index not in keep_indices
                        and message.get("role") != "user"
                        and len(str(message.get("content", ""))) <= 2000
                    ]
                    keep_indices.update(fill[-remaining:])
                wm = [older[index] for index in sorted(keep_indices)] + tail
                logger.info("🧹 Context trim pass 2: %d messages retained", len(wm))

            new_state.cognition.working_memory = wm
            
        return new_state

    def _mark_consolidation_status(
        self,
        state: AuraState,
        *,
        status: str,
        stage: str,
        error: BaseException | None = None,
    ) -> None:
        modifiers = dict(getattr(state.cognition, "modifiers", {}) or {})
        payload: dict[str, Any] = {
            "status": status,
            "stage": stage,
            "at": time.time(),
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)[:240]
        modifiers["memory_consolidation_status"] = payload
        state.cognition.modifiers = modifiers

    def _queue_failed_commit(
        self,
        *,
        error: BaseException,
        content: str,
        objective: str | None,
        interaction_context: str,
        interaction_action: str,
        interaction_outcome: str,
    ) -> None:
        try:
            dlq = self.container.get("dead_letter_queue", default=None)
            if dlq is None:
                return
            payload = {
                "content": content[:2_000],
                "objective": str(objective or "")[:240],
                "context": interaction_context[:1_000],
                "action": interaction_action,
                "outcome": interaction_outcome[:1_000],
            }
            push = getattr(dlq, "push", None)
            if callable(push):
                push("memory_consolidation.commit_interaction", payload, str(error)[:500])
                return
            capture_failure = getattr(dlq, "capture_failure", None)
            if callable(capture_failure):
                capture_failure(
                    message=content,
                    context=payload,
                    error=error,
                    source="memory_consolidation",
                )
        except _MEMORY_CONSOLIDATION_ERRORS as dlq_exc:
            _record_memory_consolidation_degradation(
                dlq_exc,
                action="kept ColdStore fallback after failed memory commit could not be queued",
                severity="warning",
                extra={"stage": "dead_letter_queue"},
            )


# ─────────────────────────────────────────────────────────────────────────────
# Declared semantics. See core/runtime/cognitive_contract.py.
#
# `writes` is MEASURED — tools/observe_phase_writes.py ran this phase against a
# real AuraState and recorded which fields moved. It is not a reading of the
# code, which is how a declaration ends up describing what the author believed.
from core.runtime.cognitive_contract import (
    BranchSpec,
    CognitiveTransformContract,
    register_contract,
)

register_contract(
    CognitiveTransformContract(
        name="MemoryConsolidationPhase",
        version="1.0",
        module=__name__,
        purpose=(
            "Move what the tick learned into durable memory, and say so in the "
            "transition cause when it happens."
        ),
        reads=("cognition.working_memory", "cognition.last_response"),
        writes=(
            "cognition.coherence_score",
            "cognition.fragmentation_score",
            "cognition.long_term_memory",
            "cognition.modifiers",
            "cognition.working_memory",
            "health",
            "response_modifiers",
            "transition_cause",
        ),
        preconditions=("state carries a cognition block",),
        branches=(
            BranchSpec(
                "consolidated",
                "there is unconsolidated working memory",
                "write it through the memory service and record the cause",
            ),
            BranchSpec(
                "nothing_to_consolidate",
                "working memory holds nothing new",
                "return state unchanged",
            ),
        ),
        side_effects=("writes to the durable memory store",),
        calibration_source=(
            "writes measured by tools/observe_phase_writes.py on the no-model "
            "path and expanded from live provenance receipts for the cleanup, "
            "strict-proof, loop-removal, retention, and derived-health branches"
            "; reads reach state through this phase's delegate rather than "
            "appearing in this module, so they are declared from the "
            "delegate's behaviour, not by scanning this file"
        ),
    )
)
