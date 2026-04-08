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
