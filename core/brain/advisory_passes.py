"""The four advisory passes, and what they learn from how the turn went.

Four advisors get a say in a turn before the model is asked anything: the
spiking active-inference model, the imagination workspace, the bicameral
advisory, and the cognitive-situation frame. Each one is asked, its answer is
written into ``state.response_modifiers`` and into the merged context, and
after the turn each is told what the answer was worth.

They lived in ``cognitive_engine`` and were the reason it reached into five
more packages than it needed to. None of the seven used ``self``. Moving them
here takes those five reaches with them, and the engine keeps thin methods
that forward, so an instance can still be patched in a test the way it always
could.

Degradations are still recorded against ``cognitive_engine``. The subsystem
name is a contract — the fail-closed list in ``core/config.py`` reads it — and
this is still the engine's work, done in a different file.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.numeric_guards import bounded_float
from core.runtime.service_registry import get_runtime_service
from core.state.aura_state import AuraState, CognitiveMode

logger = logging.getLogger(__name__)

#: What an advisor may fail with without taking the turn down. An advisory
#: that cannot answer leaves the turn exactly as it found it.
_RECOVERABLE_ERRORS = (
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


def _bounded_float(
    value: Any, default: float = 0.0, *, lower: float = 0.0, upper: float = 1.0
) -> float:
    """The shared guard, in the argument order these passes were written in."""

    return bounded_float(value, default=default, minimum=lower, maximum=upper)


def apply_spiking_active_inference(
    state: AuraState,
    objective: str,
    origin: str,
    context: dict[str, Any] | None,
    *,
    is_background: bool,
) -> dict[str, Any] | None:
    try:
        from core.cognitive.spiking_active_inference import (
            get_spiking_active_inference_advisor,
        )

        advisor = get_spiking_active_inference_advisor()
        advice = advisor.advise(
            objective,
            context=context,
            state=state,
            origin=origin,
            is_background=is_background,
        )
    except _RECOVERABLE_ERRORS as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="continued cognitive cycle without spiking active-inference advisory",
        )
        logger.debug("Spiking active-inference advisory unavailable: %s", exc)
        return context

    advice_dict = advice.to_dict()
    routing = dict(advice.routing_bias or {})
    sampling = dict(advice.sampling_bias or {})
    state.response_modifiers["spiking_active_inference"] = advice_dict
    state.response_modifiers["active_inference_action_tendency"] = advice.action
    state.response_modifiers["epistemic_uncertainty"] = advice.uncertainty
    state.response_modifiers["metacognition_depth"] = routing.get("metacognition_depth", 0.35)
    state.response_modifiers["tool_governance_pressure"] = bool(
        routing.get("use_tool_gateway")
    )
    state.response_modifiers["sampling_bias"] = sampling
    if routing.get("reduce_load"):
        state.response_modifiers["runtime_load_shed_requested"] = True
    if routing.get("repair_first"):
        state.response_modifiers["repair_first_pressure"] = True

    merged_context = dict(context or {})
    merged_context["spiking_active_inference"] = advice_dict
    return merged_context


def apply_imagination_workspace(
    state: AuraState,
    objective: str,
    origin: str,
    context: dict[str, Any] | None,
    *,
    is_background: bool,
) -> dict[str, Any] | None:
    try:
        from core.brain.imagination import get_imagination_engine

        engine = get_imagination_engine()
        frame = engine.imagine(
            objective,
            state=state,
            context=context,
            origin=origin,
            is_background=is_background,
        )
    except _RECOVERABLE_ERRORS as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="continued cognitive cycle without imagination workspace",
        )
        logger.debug("Imagination workspace unavailable: %s", exc)
        return context

    frame_dict = frame.to_dict()
    if frame.salience < 0.18:
        return context

    state.response_modifiers["imagination_workspace"] = frame_dict
    state.response_modifiers["creative_pressure"] = frame.salience
    state.response_modifiers["novelty_pressure"] = frame.novelty_pressure
    state.response_modifiers["imagination_sampling_bias"] = dict(frame.sampling_bias)
    state.response_modifiers["imagination_routing_bias"] = dict(frame.routing_bias)
    state.response_modifiers["imagination_memory_pressure"] = frame.memory_pressure
    state.response_modifiers["imagination_verification_pressure"] = frame.verification_pressure
    state.response_modifiers["imagination_working_memory"] = dict(frame.working_memory)
    state.response_modifiers["imagination_attractor_state"] = dict(frame.attractor_state)
    state.response_modifiers["verification_pressure"] = max(
        _bounded_float(state.response_modifiers.get("verification_pressure"), 0.0),
        frame.verification_pressure,
    )
    if frame.routing_bias.get("seek_verification") or frame.routing_bias.get("raise_metacognition"):
        state.response_modifiers["tool_governance_pressure"] = True
        state.response_modifiers["metacognition_depth"] = max(
            _bounded_float(state.response_modifiers.get("metacognition_depth"), 0.35),
            _bounded_float(frame.causal_effects.get("metacognition_depth"), 0.35),
        )

    cognition_mods = dict(getattr(state.cognition, "modifiers", {}) or {})
    cognition_mods["imagination_workspace"] = frame_dict
    cognition_mods["imagination_prompt_block_available"] = True
    cognition_mods["imagination_attention_targets"] = list(frame.attention_targets)
    cognition_mods["imagination_causal_effects"] = dict(frame.causal_effects)
    cognition_mods["imagination_ablation_predictions"] = dict(frame.ablation_predictions)
    cognition_mods["imagination_working_memory"] = dict(frame.working_memory)
    cognition_mods["imagination_attractor_state"] = dict(frame.attractor_state)
    if frame.routing_bias.get("requires_memory_grounding"):
        cognition_mods["requires_memory_grounding"] = True
    if frame.routing_bias.get("compress_imagination"):
        state.response_modifiers["runtime_load_shed_requested"] = True
        cognition_mods["runtime_load_shed_requested"] = True
    state.cognition.modifiers = cognition_mods
    if frame.attention_targets and not is_background:
        state.cognition.attention_focus = (
            f"{objective[:120]} | imagined focus: {', '.join(frame.attention_targets[:3])}"
        )

    merged_context = dict(context or {})
    merged_context["imagination_workspace"] = frame_dict
    return merged_context


def apply_bicameral_advisory(
    state: AuraState,
    objective: str,
    origin: str,
    context: dict[str, Any] | None,
    *,
    is_background: bool,
) -> dict[str, Any] | None:
    try:
        from core.brain.bicameral_advisory import get_bicameral_advisory

        advisor = get_bicameral_advisory()
        frame = advisor.advise(
            objective,
            state=state,
            context=context,
            origin=origin,
            is_background=is_background,
        )
    except _RECOVERABLE_ERRORS as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="continued cognitive cycle without bicameral advisory",
        )
        logger.debug("Bicameral advisory unavailable: %s", exc)
        return context

    if frame.salience < 0.18:
        return context

    frame_dict = frame.to_dict()
    # The issued frame is deeply immutable. AuraState is intentionally
    # deepcopy-able for phase retry/rebase, so only its fully materialized
    # signed transport payload may cross into state modifiers.
    causal = dict(frame_dict.get("causal_effects") or {})
    routing = dict(frame_dict.get("routing_bias") or {})
    sampling = dict(frame_dict.get("sampling_bias") or {})

    state.response_modifiers["bicameral_advisory"] = frame_dict
    state.response_modifiers["bicameral_consensus"] = frame.consensus
    state.response_modifiers["bicameral_dissent"] = frame.dissent
    state.response_modifiers["bicameral_sampling_bias"] = sampling
    state.response_modifiers["bicameral_routing_bias"] = routing
    state.response_modifiers["bicameral_attention_targets"] = list(frame.attention_targets)
    state.response_modifiers["bicameral_causal_effects"] = causal
    state.response_modifiers["bicameral_memory_priority"] = _bounded_float(
        causal.get("memory_priority"), 0.0
    )
    state.response_modifiers["bicameral_verification_pressure"] = _bounded_float(
        causal.get("verification_pressure"), 0.0
    )
    state.response_modifiers["self_model_update_pressure"] = max(
        _bounded_float(state.response_modifiers.get("self_model_update_pressure"), 0.0),
        _bounded_float(causal.get("self_model_update"), 0.0),
    )
    state.response_modifiers["metacognition_depth"] = max(
        _bounded_float(state.response_modifiers.get("metacognition_depth"), 0.35),
        _bounded_float(causal.get("metacognition_depth"), 0.35),
    )
    state.response_modifiers["verification_pressure"] = max(
        _bounded_float(state.response_modifiers.get("verification_pressure"), 0.0),
        _bounded_float(causal.get("verification_pressure"), 0.0),
    )
    state.response_modifiers["creative_pressure"] = max(
        _bounded_float(state.response_modifiers.get("creative_pressure"), 0.0),
        _bounded_float(causal.get("creative_pressure"), 0.0),
    )
    if routing.get("use_tool_gateway") or routing.get("seek_verification"):
        state.response_modifiers["tool_governance_pressure"] = True
    if routing.get("compact_foreground"):
        state.response_modifiers["runtime_load_shed_requested"] = True
    if (
        _bounded_float(causal.get("memory_priority"), 0.0) >= 0.45
        or _bounded_float(causal.get("self_model_update"), 0.0) >= 0.35
        or routing.get("preserve_conversation_context")
    ):
        state.response_modifiers["requires_memory_grounding"] = True

    cognition_mods = dict(getattr(state.cognition, "modifiers", {}) or {})
    cognition_mods["bicameral_advisory"] = frame_dict
    cognition_mods["bicameral_prompt_block_available"] = True
    cognition_mods["bicameral_attention_targets"] = list(frame.attention_targets)
    cognition_mods["bicameral_causal_effects"] = causal
    cognition_mods["bicameral_sampling_bias"] = sampling
    cognition_mods["bicameral_routing_bias"] = routing
    cognition_mods["self_model_update_pressure"] = state.response_modifiers[
        "self_model_update_pressure"
    ]
    if state.response_modifiers.get("requires_memory_grounding"):
        cognition_mods["requires_memory_grounding"] = True
    state.cognition.modifiers = cognition_mods

    if frame.attention_targets and not is_background:
        existing_focus = str(getattr(state.cognition, "attention_focus", "") or "").strip()
        advisory_focus = ", ".join(frame.attention_targets[:4])
        state.cognition.attention_focus = (
            f"{existing_focus} | advisory focus: {advisory_focus}"
            if existing_focus
            else f"{objective[:120]} | advisory focus: {advisory_focus}"
        )

    merged_context = dict(context or {})
    merged_context["bicameral_advisory"] = frame_dict
    merged_context["bicameral_sampling_bias"] = sampling
    return merged_context


def apply_cognitive_situation_frame(
    state: AuraState,
    objective: str,
    origin: str,
    context: dict[str, Any] | None,
    *,
    is_background: bool,
) -> dict[str, Any] | None:
    try:
        from core.brain.cognitive_situation import get_cognitive_situation_engine

        engine = get_cognitive_situation_engine()
        frame = engine.frame(
            objective,
            state=state,
            context=context,
            origin=origin,
            is_background=is_background,
        )
    except _RECOVERABLE_ERRORS as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="continued cognitive cycle without cognitive situation frame",
        )
        logger.debug("Cognitive situation frame unavailable: %s", exc)
        return context

    frame_dict = frame.to_dict()
    if frame.salience < 0.16:
        return context

    causal = dict(frame.causal_effects or {})
    routing = dict(frame.routing_bias or {})
    sampling = dict(frame.sampling_bias or {})

    state.response_modifiers["cognitive_situation_frame"] = frame_dict
    state.response_modifiers["semantic_flexibility_pressure"] = frame.semantic_flexibility
    state.response_modifiers["analogical_leap_pressure"] = frame.analogical_leap_pressure
    state.response_modifiers["sensorimotor_grounding_pressure"] = frame.sensorimotor_grounding
    state.response_modifiers["cognitive_situation_sampling_bias"] = sampling
    state.response_modifiers["cognitive_situation_routing_bias"] = routing
    state.response_modifiers["cognitive_situation_attention_targets"] = list(
        frame.attention_targets
    )
    state.response_modifiers["verification_pressure"] = max(
        _bounded_float(state.response_modifiers.get("verification_pressure"), 0.0),
        frame.verification_pressure,
    )
    state.response_modifiers["metacognition_depth"] = max(
        _bounded_float(state.response_modifiers.get("metacognition_depth"), 0.35),
        frame.metacognition_pressure,
    )
    state.response_modifiers["creative_pressure"] = max(
        _bounded_float(state.response_modifiers.get("creative_pressure"), 0.0),
        frame.analogical_leap_pressure,
    )
    if routing.get("use_tool_gateway") or routing.get("bind_sensorimotor_evidence"):
        state.response_modifiers["tool_governance_pressure"] = True
    if routing.get("perception_abstention_required"):
        state.response_modifiers["perception_abstention_required"] = True
    if routing.get("perception_repair_required"):
        state.response_modifiers["perception_repair_required"] = True
    perception_constraints = causal.get("perception_planning_constraints")
    if isinstance(perception_constraints, list):
        state.response_modifiers["perception_planning_constraints"] = list(
            perception_constraints[:8]
        )
    perception_repairs = causal.get("perception_repair_requirements")
    if isinstance(perception_repairs, list):
        state.response_modifiers["perception_repair_requirements"] = list(
            perception_repairs[:8]
        )
    social_constraints = causal.get("social_planning_constraints")
    if isinstance(social_constraints, list):
        state.response_modifiers["social_planning_constraints"] = list(
            social_constraints[:8]
        )
    state.response_modifiers["social_uncertainty"] = frame.social_uncertainty
    state.response_modifiers["social_repair_pressure"] = frame.social_repair_pressure
    if routing.get("social_repair_required"):
        state.response_modifiers["social_repair_required"] = True
    if routing.get("social_confirmation_required"):
        state.response_modifiers["social_confirmation_required"] = True
    if routing.get("social_state_clarification_required"):
        state.response_modifiers["social_state_clarification_required"] = True
    if routing.get("social_response_brevity"):
        state.response_modifiers["social_response_brevity"] = True
    if routing.get("requires_memory_grounding") or routing.get("preserve_conversation_context"):
        state.response_modifiers["requires_memory_grounding"] = True
    if routing.get("deliberate_mode") and not is_background:
        state.cognition.current_mode = CognitiveMode.DELIBERATE

    cognition_mods = dict(getattr(state.cognition, "modifiers", {}) or {})
    cognition_mods["cognitive_situation_frame"] = frame_dict
    cognition_mods["cognitive_situation_prompt_block_available"] = True
    cognition_mods["semantic_flexibility_pressure"] = frame.semantic_flexibility
    cognition_mods["analogical_leap_pressure"] = frame.analogical_leap_pressure
    cognition_mods["sensorimotor_grounding_pressure"] = frame.sensorimotor_grounding
    cognition_mods["cognitive_situation_sampling_bias"] = sampling
    cognition_mods["cognitive_situation_routing_bias"] = routing
    cognition_mods["cognitive_situation_causal_effects"] = causal
    if routing.get("requires_memory_grounding"):
        cognition_mods["requires_memory_grounding"] = True
    if routing.get("bind_sensorimotor_evidence"):
        cognition_mods["bind_sensorimotor_evidence"] = True
    if routing.get("perception_abstention_required"):
        cognition_mods["perception_abstention_required"] = True
    if routing.get("perception_repair_required"):
        cognition_mods["perception_repair_required"] = True
    if routing.get("social_repair_required"):
        cognition_mods["social_repair_required"] = True
    if routing.get("social_confirmation_required"):
        cognition_mods["social_confirmation_required"] = True
    if routing.get("social_state_clarification_required"):
        cognition_mods["social_state_clarification_required"] = True
    state.cognition.modifiers = cognition_mods

    if frame.attention_targets and not is_background:
        existing_focus = str(getattr(state.cognition, "attention_focus", "") or "").strip()
        situation_focus = ", ".join(frame.attention_targets[:4])
        state.cognition.attention_focus = (
            f"{existing_focus} | situation focus: {situation_focus}"
            if existing_focus
            else f"{objective[:120]} | situation focus: {situation_focus}"
        )

    merged_context = dict(context or {})
    merged_context["cognitive_situation_frame"] = frame_dict
    merged_context["cognitive_situation_sampling_bias"] = sampling
    return merged_context


def learn_spiking_active_inference_outcome(
    context: dict[str, Any] | None,
    *,
    outcome: str,
    reward: float,
) -> dict[str, Any] | None:
    if not isinstance(context, dict):
        return None
    advice = context.get("spiking_active_inference")
    if not isinstance(advice, dict):
        return None
    action = str(advice.get("action") or "").strip()
    features = advice.get("features")
    if not action or not isinstance(features, dict):
        return None
    try:
        advisor = get_runtime_service("spiking_active_inference", default=None)
        if advisor is None or not hasattr(advisor, "learn_from_feedback"):
            return None
        learned = advisor.learn_from_feedback(action, float(reward), features)
        if isinstance(learned, dict):
            learned["outcome"] = str(outcome or "unknown")[:80]
            return learned
    except _RECOVERABLE_ERRORS as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="continued cognitive cycle without spiking active-inference feedback learning",
        )
        logger.debug("Spiking active-inference feedback learning skipped: %s", exc)
    return None


def learn_imagination_workspace_outcome(
    context: dict[str, Any] | None,
    *,
    outcome: str,
    reward: float,
    evidence_basis: str = "",
    evidence_id: str = "",
) -> dict[str, Any] | None:
    if not isinstance(context, dict):
        return None
    frame = context.get("imagination_workspace")
    if not isinstance(frame, dict):
        return None
    try:
        from core.brain.imagination_basis import Basis, meets

        try:
            basis = Basis(str(evidence_basis or ""))
        except ValueError:
            basis = Basis.LEXICAL
        frame_id = str(frame.get("frame_id") or "")[:120]
        subject = str(
            context.get("user_id")
            or context.get("principal_id")
            or "anonymous"
        )[:64]
        if not evidence_id or not meets(basis, Basis.MEASURED):
            # A generation existing is not evidence that imagination made
            # it correct. Keep the eligibility record typed and pending;
            # do not call the durable learner until an evaluator, tool
            # receipt, or user outcome supplies measured evidence.
            return {
                "frame_id": frame_id,
                "subject": subject,
                "outcome": str(outcome or "unknown")[:80],
                "reward": round(max(-1.0, min(1.0, float(reward))), 4),
                "evidence_basis": basis.value,
                "evidence_id": str(evidence_id or "")[:120],
                "applied": False,
                "refusal": "measured outcome evidence required",
            }
        from core.brain.imagination import get_imagination_engine

        learned = get_imagination_engine().learn_from_feedback(
            frame,
            reward=float(reward),
            outcome=outcome,
            subject=subject,
            evidence_basis=basis.value,
            evidence_id=evidence_id,
        )
        return learned if isinstance(learned, dict) else None
    except _RECOVERABLE_ERRORS as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="continued cognitive cycle without imagination workspace feedback learning",
        )
        logger.debug("Imagination workspace feedback learning skipped: %s", exc)
    return None


def learn_bicameral_advisory_outcome(
    context: dict[str, Any] | None,
    *,
    outcome: str,
    reward: float,
) -> dict[str, Any] | None:
    if not isinstance(context, dict):
        return None
    frame = context.get("bicameral_advisory")
    if not isinstance(frame, dict):
        return None
    try:
        from core.brain.bicameral_advisory import get_bicameral_advisory

        learned = get_bicameral_advisory().learn_from_feedback(
            frame,
            reward=float(reward),
            outcome=outcome,
        )
        return learned if isinstance(learned, dict) else None
    except _RECOVERABLE_ERRORS as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="continued cognitive cycle without bicameral advisory feedback learning",
        )
        logger.debug("Bicameral advisory feedback learning skipped: %s", exc)
    return None
