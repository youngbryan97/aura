"""core/phases/conversational_dynamics_phase.py — Conversational Dynamics Phase

Kernel phase that runs on every user message and computes the full
conversational dynamics state. Runs BEFORE response generation so the
LLM speaks from computed pragmatic state, not just raw text.

Position in pipeline: after SensoryIngestion, before CognitiveRouting.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, TYPE_CHECKING

from core.kernel.bridge import Phase
from core.state.aura_state import AuraState

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.ConversationalDynamics")


class ConversationalDynamicsPhase(Phase):
    """
    Computes full conversational state before response generation:
    - Pragmatic / illocutionary force
    - Emotional frame and trajectory
    - Topic trajectory, drift chain, open threads
    - Register and accommodation cues
    - Face threat assessment
    - Humor detection
    - Turn management
    """

    def __init__(self, kernel: "AuraKernel"):
        super().__init__(kernel)
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from core.conversational.dynamics import get_dynamics_engine
                self._engine = get_dynamics_engine()
            except Exception as e:
                logger.warning("ConversationalDynamics: Engine init failed: %s", e)
        return self._engine

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        if not objective:
            return state

        engine = self._get_engine()
        if not engine:
            return state

        origin = kwargs.get("origin", state.cognition.current_origin or "system")

        # Only run full analysis on user-facing messages
        if origin not in ("user", "voice", "admin", "web"):
            return state

        try:
            new_state = state.derive("conversational_dynamics", origin="ConversationalDynamicsPhase")

            # Compute dynamics from the latest user message
            dynamics = engine.update(
                message=objective,
                role="user",
                working_memory=state.cognition.working_memory
            )

            # Store the prompt injection in response_modifiers so UnitaryResponsePhase can use it
            new_state.response_modifiers["conversational_dynamics"] = engine.get_prompt_injection()

            # Also surface key fields directly to CognitiveContext for other phases to read
            cog = new_state.cognition

            # Update discourse state with richer data
            if dynamics.current_topic != "general":
                cog.discourse_topic = dynamics.current_topic
            cog.user_emotional_trend = dynamics.partner_frame

            # Mirror conversation_energy from partner intensity
            cog.conversation_energy = dynamics.partner_intensity

            try:
                from core.container import ServiceContainer

                interaction_signals = ServiceContainer.get("interaction_signals", default=None)
                if interaction_signals and hasattr(interaction_signals, "get_status"):
                    signal_status = interaction_signals.get_status() or {}
                    fused = dict(signal_status.get("fused", {}) or {})
                    guidance = interaction_signals.get_prompt_guidance() if hasattr(interaction_signals, "get_prompt_guidance") else ""
                    new_state.response_modifiers["interaction_signals"] = signal_status
                    if guidance:
                        new_state.response_modifiers["conversational_dynamics"] = (
                            f"{engine.get_prompt_injection()}\n\n{guidance}"
                        )
                    engagement = float(fused.get("engagement", 0.0) or 0.0)
                    activation = float(fused.get("activation", 0.0) or 0.0)
                    cog.conversation_energy = max(
                        cog.conversation_energy,
                        min(1.0, (engagement * 0.75) + (activation * 0.25)),
                    )
                    cog.modifiers["interaction_summary"] = str(fused.get("summary") or "")
                    cog.modifiers["interaction_pacing"] = str(fused.get("pacing") or "steady")
                    cog.modifiers["interaction_verbosity_bias"] = str(fused.get("verbosity_bias") or "balanced")
            except Exception as exc:
                logger.debug("ConversationalDynamics: interaction signal integration skipped: %s", exc)

            # Store callback topics in discourse_branches
            available_callbacks = [
                a.topic for a in dynamics.topic_anchors[-5:]
                if not a.is_resolved and a.topic != dynamics.current_topic
            ]
            cog.discourse_branches = available_callbacks

            # Store the full dynamics state for downstream phases
            new_state.response_modifiers["conv_dynamics_state"] = {
                "floor_state": dynamics.floor_state,
                "partner_frame": dynamics.partner_frame,
                "partner_intensity": dynamics.partner_intensity,
                "humor_frame_active": dynamics.humor_frame_active,
                "humor_type": dynamics.humor_type,
                "escalation_invited": dynamics.escalation_invited,
                "last_speech_act": dynamics.last_speech_act,
                "illocutionary_intent": dynamics.illocutionary_intent,
                "conditional_relevance_open": dynamics.conditional_relevance_open,
                "register": dynamics.register,
                "in_group_active": dynamics.in_group_active,
                "accommodation_cues": dynamics.accommodation_cues,
                "hedge_level": dynamics.hedge_level,
                "challenge_is_face_threat": dynamics.challenge_is_face_threat,
                "open_threads_count": len(dynamics.open_threads),
                "association_chain": dynamics.association_chain,
            }

            # ── Wire dormant personhood modules into the foreground path ──
            # These modules exist but were never called during live conversation.
            # Each provides context that shapes HOW Aura responds, not just WHAT.

            # Humor Engine: adaptive banter calibrated to this specific user
            try:
                from core.container import ServiceContainer
                humor = ServiceContainer.get("humor_engine", default=None)
                if humor and hasattr(humor, "update_banter_state"):
                    humor.update_banter_state(objective, dynamics)
                    guidance = humor.get_humor_guidance("owner") if hasattr(humor, "get_humor_guidance") else ""
                    banter = humor.get_banter_directive() if hasattr(humor, "get_banter_directive") else ""
                    if guidance or banter:
                        new_state.response_modifiers["humor_guidance"] = f"{guidance}\n{banter}".strip()
            except Exception as exc:
                logger.debug("ConversationalDynamics: humor engine skipped: %s", exc)

            # Conversation Intelligence: rhythm, pacing, arc awareness
            try:
                from core.container import ServiceContainer
                conv_intel = ServiceContainer.get("conversation_intelligence", default=None)
                if conv_intel and hasattr(conv_intel, "get_context_injection"):
                    ci_block = conv_intel.get_context_injection()
                    if ci_block:
                        new_state.response_modifiers["conversation_intelligence"] = ci_block
            except Exception as exc:
                logger.debug("ConversationalDynamics: conversation intelligence skipped: %s", exc)

            # Relational Intelligence: social modeling of this specific person
            try:
                from core.container import ServiceContainer
                rel_intel = ServiceContainer.get("relational_intelligence", default=None)
                if rel_intel and hasattr(rel_intel, "get_context_injection"):
                    ri_block = rel_intel.get_context_injection("owner")
                    if ri_block:
                        new_state.response_modifiers["relational_intelligence"] = ri_block
            except Exception as exc:
                logger.debug("ConversationalDynamics: relational intelligence skipped: %s", exc)

            # MetaCognition: reasoning strategy selection for this turn
            try:
                from core.container import ServiceContainer
                metacog = ServiceContainer.get("metacognition", default=None)
                if metacog and hasattr(metacog, "before_reasoning"):
                    meta_hints = metacog.before_reasoning(objective, {
                        "origin": origin,
                        "dynamics": new_state.response_modifiers.get("conv_dynamics_state", {}),
                    })
                    if meta_hints and isinstance(meta_hints, dict):
                        strategy = meta_hints.get("strategy", "")
                        if strategy:
                            new_state.response_modifiers["metacognitive_strategy"] = strategy
            except Exception as exc:
                logger.debug("ConversationalDynamics: metacognition skipped: %s", exc)

            # Credit Assignment: outcome-aware context from prior actions
            try:
                from core.container import ServiceContainer
                credit = ServiceContainer.get("credit_assignment", default=None)
                if credit and hasattr(credit, "get_context_block"):
                    ca_block = credit.get_context_block()
                    if ca_block:
                        new_state.response_modifiers["credit_assignment"] = ca_block
            except Exception as exc:
                logger.debug("ConversationalDynamics: credit assignment skipped: %s", exc)

            # Agency Comparator: sense of authorship over recent actions
            try:
                from core.consciousness.agency_comparator import get_agency_comparator
                agency = get_agency_comparator()
                agency_block = agency.get_context_block()
                if agency_block:
                    new_state.response_modifiers["agency_comparator"] = agency_block
            except Exception as exc:
                logger.debug("ConversationalDynamics: agency comparator skipped: %s", exc)

            # Narrative Memory: autobiographical narrative context from journal/arcs
            try:
                from core.container import ServiceContainer
                narrative = ServiceContainer.get("narrative_engine", default=None)
                if narrative and hasattr(narrative, "get_narrative_context"):
                    nm_block = narrative.get_narrative_context()
                    if nm_block:
                        new_state.response_modifiers["narrative_context"] = nm_block
            except Exception as exc:
                logger.debug("ConversationalDynamics: narrative memory skipped: %s", exc)

            # Natural Follow-up: whether Aura should ask a follow-up, make a statement, or stay quiet
            try:
                from core.container import ServiceContainer
                sve = ServiceContainer.get("substrate_voice_engine", default=None)
                if sve and hasattr(sve, "_followup_engine"):
                    followup_engine = sve._followup_engine
                    profile = sve.get_current_profile() if hasattr(sve, "get_current_profile") else None
                    if followup_engine and profile:
                        last_assistant = ""
                        for msg in reversed(state.cognition.working_memory or []):
                            if msg.get("role") == "assistant":
                                last_assistant = msg.get("content", "")
                                break
                        decision = followup_engine.decide(
                            profile=profile,
                            user_message=objective,
                            aura_response=last_assistant,
                            conversation_history=list(state.cognition.working_memory or []),
                        )
                        if decision and decision.should_followup:
                            new_state.response_modifiers["natural_followup"] = {
                                "should_followup": True,
                                "followup_type": decision.followup_type,
                                "delay_seconds": decision.delay_seconds,
                                "context_hint": decision.context_hint,
                                "word_budget": decision.word_budget,
                                "reason": decision.reason,
                            }
            except Exception as exc:
                logger.debug("ConversationalDynamics: natural followup skipped: %s", exc)

            logger.debug(
                "ConversationalDynamics: frame=%s intensity=%.2f speech_act=%s register=%s humor=%s",
                dynamics.partner_frame,
                dynamics.partner_intensity,
                dynamics.last_speech_act,
                dynamics.register,
                dynamics.humor_type or "none"
            )

            return new_state

        except Exception as e:
            logger.warning("ConversationalDynamics phase failed: %s", e)
            return state
