from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import logging
import time
from typing import Optional, TYPE_CHECKING

from core.kernel.bridge import Phase
from core.phases.dialogue_policy import validate_dialogue_response
from core.phases.response_contract import ResponseContract
from core.state.aura_state import AuraState
from core.runtime.background_policy import background_activity_allowed

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.LearningPhase")


class LearningPhase(Phase):
    """
    Post-response learning phase.

    Detects follow-up and confusion signals from working memory,
    then records the interaction into LiveLearner for quality scoring.
    """

    def __init__(self, kernel: "AuraKernel"):
        super().__init__(kernel)
        self._learner = None   # Lazy-loaded

    def _get_learner(self):
        if self._learner is None:
            try:
                from core.learning.live_learner import get_live_learner
                self._learner = get_live_learner()
            except Exception as e:
                record_degradation('learning_phase', e)
                logger.debug("LearningPhase: could not load learner: %s", e)
        return self._learner

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        # ISSUE-88: Self-Modification Awareness
        has_mod = False
        if hasattr(state.cognition, "modifiers") and state.cognition.modifiers:
            if "self_modification" in state.cognition.modifiers:
                has_mod = True
                logger.info("🧠 SelfModification: Awareness triggered. Preserving modification context.")

        # Memory Consolidation Logic
        # Skip if response is too short and no modification occurred
        if not has_mod and len(state.cognition.last_response or "") < 10:
            return state

        # Metacognitive feedback: wire user corrections to the calibrator.
        # When the user signals confusion/correction, feed that as actual_correctness=0
        # for the previous response. When they follow up (engaged), score it as correct.
        follow_up = self._detect_follow_up(state)
        confusion = self._detect_confusion(state)
        try:
            from core.container import ServiceContainer
            calibrator = ServiceContainer.get("metacognitive_calibrator", default=None)
            if calibrator:
                prev_confidence = state.response_modifiers.get("learning_score", 0.7)
                if confusion:
                    calibrator.record_prediction(
                        confidence=prev_confidence, actual_correctness=0.0
                    )
                    logger.debug("MetaCal: correction signal → recording low correctness")
                elif follow_up:
                    calibrator.record_prediction(
                        confidence=prev_confidence, actual_correctness=1.0
                    )
        except Exception as e:
            record_degradation('learning_phase', e)
            logger.debug("LearningPhase: metacognitive feedback failed: %s", e)

        # Run standard learning + cross-domain synthesis
        try:
            state = await self._perform_standard_learning(state, objective or "")
        except Exception as e:
            record_degradation('learning_phase', e)
            logger.debug("LearningPhase: standard learning failed: %s", e)

        try:
            state = await self._map_cross_domain(state, objective or "")
        except Exception as e:
            record_degradation('learning_phase', e)
            logger.debug("LearningPhase: cross-domain mapping failed: %s", e)

        try:
            await self._wire_conversation_learning(state, objective or "")
        except Exception as e:
            record_degradation('learning_phase', e)
            logger.debug("LearningPhase: follow-up wiring failed: %s", e)

        return state

    async def _perform_standard_learning(self, state: AuraState, objective: str) -> AuraState:
        learner = self._get_learner()
        if learner is None: return state
        response = state.cognition.last_response
        if not response: return state
        
        follow_up = self._detect_follow_up(state)
        confusion = self._detect_confusion(state)
        
        try:
            # ISSUE-94: Affective Memory synchronization
            affect_data = {
                "valence": state.affect.valence,
                "arousal": state.affect.arousal,
                "dominant": state.affect.dominant_emotion
            }
            score = learner.record_tick(
                state=state, 
                user_input=objective, 
                response=response, 
                follow_up=follow_up, 
                confusion=confusion,
                affect=affect_data # Sync emotional context
            )
            if score:
                if not hasattr(state, "response_modifiers") or state.response_modifiers is None:
                    state.response_modifiers = {}
                state.response_modifiers["learning_score"] = round(score.raw_score, 3)
                state.response_modifiers["affective_sync"] = True
        except Exception as e:
            record_degradation('learning_phase', e)
            logger.debug("LearningPhase: record failed: %s", e)
        return state

    async def _map_cross_domain(self, state: AuraState, objective: str) -> AuraState:
        """[AGI Seed] Relates current objective to distant knowledge domains."""
        curiosity = getattr(state.affect, 'curiosity', 0.5)
        if curiosity > 0.8:
            try:
                from core.container import ServiceContainer
                orch = ServiceContainer.get("orchestrator", default=None)
            except Exception:
                orch = None
            if not background_activity_allowed(
                orch,
                min_idle_seconds=1200.0,
                max_memory_percent=78.0,
                max_failure_pressure=0.08,
                require_conversation_ready=True,
            ):
                return state
            logger.info("🔭 [AGI] High curiosity: Generating cross-domain synthesis...")
            state.cognition.pending_intents.append({
                "type": "cross_domain_synthesis",
                "trigger": objective,
                "curiosity_level": curiosity
            })
        return state

    async def _trigger_autotelic_curiosity(self, state: AuraState) -> AuraState:
        """[AGI Seed] Generates a new internal objective based on idle curiosity."""
        logger.info("🧩 [AGI] Idle detected. Triggering autotelic exploration.")
        state.cognition.pending_intents.append({
            "type": "autotelic_objective",
            "domain": "UNKNOWN", # To be filled by ResearchCycle
            "curiosity_vector": state.affect.curiosity
        })
        return state

    @staticmethod
    def _looks_generic(response: str) -> bool:
        text = str(response or "").strip().lower()
        if not text:
            return True
        generic_markers = (
            "how can i help",
            "what would you like",
            "let me know if you'd like",
            "as an ai",
            "as a language model",
            "i'm here to help",
            "what about you?",
            "how about you?",
            "what questions do you have?",
        )
        return any(marker in text for marker in generic_markers)

    @staticmethod
    def _recent_learning_messages(state: AuraState, limit: int = 8):
        messages = []
        for msg in list(getattr(state.cognition, "working_memory", []) or [])[-limit:]:
            if not isinstance(msg, dict):
                continue
            content = str(msg.get("content", "") or "").strip()
            metadata = msg.get("metadata", {}) or {}
            if not content or msg.get("ephemeral"):
                continue
            if msg.get("role") == "system" and metadata.get("type") != "skill_result":
                continue
            messages.append(dict(msg))
        return messages

    async def _wire_conversation_learning(self, state: AuraState, objective: str) -> None:
        response = str(getattr(state.cognition, "last_response", "") or "").strip()
        if not objective or not response:
            return

        from core.container import ServiceContainer

        contract = dict(getattr(state, "response_modifiers", {}) or {}).get("response_contract", {}) or {}
        dialogue_validation = dict(getattr(state, "response_modifiers", {}) or {}).get("dialogue_validation", {}) or {}
        learning_score = float((state.response_modifiers or {}).get("learning_score", 0.0) or 0.0)
        confusion = self._detect_confusion(state)
        recent_messages = self._recent_learning_messages(state)
        affect_signature = (
            state.affect.get_cognitive_signature()
            if hasattr(state.affect, "get_cognitive_signature")
            else {}
        )

        if recent_messages:
            try:
                from core.cognition.knowledge_enrichment import get_enricher
                from core.utils.task_tracker import get_task_tracker

                enricher = get_enricher(
                    knowledge_graph=ServiceContainer.get("knowledge_graph", default=None),
                    brain=ServiceContainer.get("cognitive_engine", default=None),
                    belief_engine=ServiceContainer.get("belief_engine", default=None),
                )
                force = bool(contract.get("requires_search") or contract.get("tool_evidence_available"))
                task = get_task_tracker().create_task(
                    enricher.enrich_from_conversation(recent_messages, force=force),
                    name="learning_phase.knowledge_enrichment",
                )
                get_task_tracker().track_task(task)
            except Exception as e:
                record_degradation('learning_phase', e)
                logger.debug("LearningPhase: knowledge enrichment scheduling failed: %s", e)

        should_distill = bool(
            confusion
            or (learning_score and learning_score < 0.55)
            or self._looks_generic(response)
            or (contract.get("requires_search") and not contract.get("tool_evidence_available"))
        )
        if not should_distill and contract:
            try:
                dialogue_contract = ResponseContract(**contract)
                should_distill = not validate_dialogue_response(response, dialogue_contract).ok
            except Exception as _exc:
                record_degradation('learning_phase', _exc)
                logger.debug("Suppressed Exception: %s", _exc)
        if should_distill:
            try:
                from core.adaptation.distillation_pipe import get_distillation_pipe

                confidence = max(0.05, min(0.99, learning_score or 0.45))
                await get_distillation_pipe().flag_for_distillation(
                    prompt=objective,
                    local_response=response,
                    confidence=confidence,
                    context={
                        "origin": str(getattr(state.cognition, "current_origin", "") or ""),
                        "confusion": confusion,
                        "learning_score": learning_score,
                        "response_contract": contract,
                        "dialogue_validation": dialogue_validation,
                        "affect_signature": affect_signature,
                    },
                )
            except Exception as e:
                record_degradation('learning_phase', e)
                logger.debug("LearningPhase: distillation flagging failed: %s", e)

    def _detect_follow_up(self, state: AuraState) -> bool:
        """
        A follow-up is when the user responded to our last message.
        The working memory should show: ... user, assistant, user
        """
        wm = state.cognition.working_memory
        if len(wm) < 3:
            return False
        last3_roles = [m.get("role") for m in wm[-3:]]
        # Pattern: user → assistant → user means they engaged with our response
        return last3_roles == ["user", "assistant", "user"]

    def _detect_confusion(self, state: AuraState) -> bool:
        """
        Confusion signals from the most recent user message.
        """
        wm = state.cognition.working_memory
        # Get the current user message (the one being processed)
        user_msgs = [m for m in wm if m.get("role") == "user"]
        if not user_msgs:
            return False
        last_user = user_msgs[-1].get("content", "").lower()
        confusion_signals = [
            "what?", "??", "that's wrong", "that's not right",
            "i don't understand", "what do you mean", "huh?",
            "you're wrong", "that makes no sense", "incorrect",
            "not what i asked", "stop", "nevermind",
        ]
        return any(sig in last_user for sig in confusion_signals)
