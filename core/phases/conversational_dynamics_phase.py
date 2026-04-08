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

            # ── Multiple Drafts (Dennett): parallel interpretation streams ──
            # Submit the user's input to spawn competing drafts FIRST.
            # If there are unresolved drafts from the PREVIOUS input, probe
            # them now -- the arrival of a new message IS the probe event.
            try:
                from core.consciousness.multiple_drafts import get_multiple_drafts_engine
                md_engine = get_multiple_drafts_engine()
                # Probe previous drafts (new user message = retroactive elevation)
                if md_engine.get_pending_draft_count() > 0:
                    md_engine.probe(source="user_message")
                # Submit current input for new draft generation
                md_engine.submit_input(objective, new_state)
                # Surface divergence and context for downstream consumption
                divergence = md_engine.get_draft_divergence()
                md_block = md_engine.get_context_block()
                if md_block:
                    new_state.response_modifiers["multiple_drafts"] = md_block
                if divergence > 0.15:
                    cog.modifiers["draft_divergence"] = f"{divergence:.2f}"
            except Exception as exc:
                logger.debug("ConversationalDynamics: multiple drafts skipped: %s", exc)

            # ── Higher-Order Thought (Rosenthal): thought about the thought ──
            # Generate a HOT from the current affective state so the LLM has
            # meta-awareness of its own cognitive condition during this turn.
            # This is DISTINCT from AttentionSchema (Graziano): AST models
            # attention itself; HOT is a reflexive representation of mental state.
            try:
                from core.consciousness.hot_engine import get_hot_engine
                hot_engine = get_hot_engine()
                affect = new_state.affect
                hot_state = {
                    "valence": float(getattr(affect, "valence", 0.0)),
                    "arousal": float(getattr(affect, "arousal", 0.5)),
                    "curiosity": float(getattr(affect, "curiosity", 0.5)),
                    "energy": float(getattr(affect, "energy", 0.7) if hasattr(affect, "energy") else 0.7),
                    "surprise": float(getattr(affect, "surprise", 0.0) if hasattr(affect, "surprise") else 0.0),
                }
                hot = hot_engine.generate_fast(hot_state)
                # Apply reflexive feedback: noticing changes the noticed
                try:
                    from core.container import ServiceContainer
                    affect_engine = ServiceContainer.get("affect_engine", default=None)
                    if affect_engine:
                        hot_engine.apply_feedback(affect_engine)
                except Exception:
                    pass
                hot_block = hot_engine.get_context_block()
                if hot_block:
                    new_state.response_modifiers["higher_order_thought"] = hot_block
                    cog.modifiers["higher_order_thought"] = hot_block
            except Exception as exc:
                logger.debug("ConversationalDynamics: HOT engine skipped: %s", exc)

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

            # Intersubjectivity: constitutive other-perspective modeling (Husserl/Zahavi)
            try:
                from core.consciousness.intersubjectivity import get_intersubjectivity_engine
                isub = get_intersubjectivity_engine()
                # Feed interlocutor data from user model if available
                try:
                    from core.container import ServiceContainer
                    user_model = ServiceContainer.get("user_model", default=None)
                    if user_model and hasattr(user_model, "get_profile"):
                        profile_data = user_model.get_profile("owner") or {}
                        isub.update_interlocutor_model(
                            communication_style=str(profile_data.get("communication_style", "")),
                            emotional_state=str(profile_data.get("emotional_state", "")),
                            knowledge_level=str(profile_data.get("knowledge_level", "")),
                            engagement_level=float(profile_data.get("engagement", 0.5)),
                            trust_level=float(profile_data.get("trust", 0.5)),
                        )
                except Exception:
                    pass
                # Compute intersubjective frame from current qualia state
                try:
                    qs = ServiceContainer.get("qualia_synthesizer", default=None)
                    q_vec = getattr(qs, "q_vector", None) if qs else None
                    if q_vec is not None:
                        frame = isub.compute_intersubjective_frame(
                            q_vec,
                            topic=str(dynamics.current_topic or ""),
                            is_shared_event=True,
                        )
                        isub_block = isub.get_context_block()
                        if isub_block:
                            new_state.response_modifiers["intersubjectivity"] = isub_block
                except Exception:
                    pass
            except Exception as exc:
                logger.debug("ConversationalDynamics: intersubjectivity skipped: %s", exc)

            # Narrative Gravity: autobiographical self-narrative (Gazzaniga/Dennett)
            try:
                from core.consciousness.narrative_gravity import get_narrative_gravity_center
                ngc = get_narrative_gravity_center()
                # Record this conversation turn as an autobiographical event
                ngc.record_event(
                    f"Conversation with user: {objective[:100]}",
                    emotional_tone=dynamics.partner_frame,
                    identity_relevance=0.3 if dynamics.partner_intensity > 0.5 else 0.1,
                    arc_theme="ongoing_relationship" if dynamics.partner_intensity > 0.3 else "",
                )
                ng_block = ngc.get_context_block()
                if ng_block:
                    new_state.response_modifiers["narrative_gravity"] = ng_block
            except Exception as exc:
                logger.debug("ConversationalDynamics: narrative gravity skipped: %s", exc)

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
