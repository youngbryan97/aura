import logging
import time
import uuid
from typing import Any, Optional
from . import BasePhase
from ..state.aura_state import AuraState
from ..consciousness.executive_authority import get_executive_authority

logger = logging.getLogger(__name__)

class MemoryConsolidationPhase(BasePhase):
    """
    Phase 6: Memory Consolidation.
    Commits recent interactions and insights to long-term storage (RAG).
    Ensures that the experience is persisted beyond working memory.
    """
    
    def __init__(self, container: Any):
        self.container = container

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        Persist recent interactions to long-term storage and prune working memory.

        Detects completed user/assistant turns (or high-arousal forced consolidation),
        optionally distils content through the SovereignPruner, queues a knowledge
        evolution entry on the ColdStore, detects and degrades stability on
        conversational loops, and caps working memory at MAX_WORKING_MEMORY entries.
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

        # ISSUE-81: Consolidation Skip Fix
        # Allow consolidation if there's high arousal or a pending action,
        # even if the turn is not strictly completed.
        force_consolidation = new_state.affect.arousal > 0.8 or (len(new_state.cognition.working_memory) > 0 and new_state.cognition.working_memory[-1].get("action"))

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
            except Exception:
                pass
        
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
            for i in range(len(assistant_msgs) - 1):
                prev = assistant_msgs[i]
                if latest == str(prev.get("content", "")).strip() and len(latest) > 20:
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
                    # Also prune the repeated assistant messages from working memory
                    # so the next tick starts fresh — keep only the last user message
                    user_msgs = [m for m in new_state.cognition.working_memory if isinstance(m, dict) and m.get("role") == "user"]
                    if user_msgs:
                        new_state.cognition.working_memory = [user_msgs[-1]]
                    break

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
            # High arousal + low valence = high emotional weight (friction/distress)
            # High arousal + high valence = high emotional weight (excitement/joy)
            emotional_weight = min(1.0, new_state.affect.arousal * (1.1 - abs(new_state.affect.valence)))
            
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
                salience = float(affect_signature.get("memory_salience", 0.0) or 0.0)
                complexity = float(affect_signature.get("affective_complexity", 0.0) or 0.0)
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
                        "resonance": affect_signature.get("resonance", getattr(new_state.affect, "get_resonance_string", lambda: "")()),
                    },
                )
            except Exception as e:
                logger.debug("MemoryConsolidation: MemoryFacade commit failed: %s", e)
 
        # Queue the knowledge for the ColdStore to process asynchronously.
        if new_state.cold is not None:
            new_state.cold.evolution_log.append({
                "type": "knowledge_addition",
                "content": content,
                "source": source,
                "timestamp": float(time.time())
            })
        
        # vResilience: Enforce cap on evolution log (BUG-017)
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
        MAX_WORKING_MEMORY: int = 15
        wm = new_state.cognition.working_memory
        if len(wm) > MAX_WORKING_MEMORY:
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
            if len(wm) > MAX_WORKING_MEMORY:
                # Always keep last 4 messages (current conversation turn)
                tail = wm[-4:]
                older = wm[:-4]
                # From older, prefer user messages and high-importance ones
                keep_older = [m for m in older if isinstance(m, dict) and m.get("role") == "user"]
                remaining = MAX_WORKING_MEMORY - len(tail) - len(keep_older)
                if remaining > 0:
                    non_user = [m for m in older if isinstance(m, dict) and m.get("role") != "user"]
                    keep_older.extend(non_user[-remaining:])
                wm = keep_older + tail
                logger.info("🧹 Context trim pass 2: %d messages retained", len(wm))

            new_state.cognition.working_memory = wm
            
        return new_state
