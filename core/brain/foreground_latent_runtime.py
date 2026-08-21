"""Bounded foreground entry point for the Recursive Latent Cortex.

The active and compatibility response phases must make the same routing and
ownership decisions.  This module owns that contract so a latent episode is
never selected in one serving path but silently bypassed in another.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.structured_input import analyze_prompt_shape

logger = logging.getLogger("Aura.ForegroundLatentRuntime")

_LATENT_CLEANUP_RESERVE_SECONDS = 8.0
_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_PROMPT_SHAPE_KEYS = frozenset(
    {
        "question_parts",
        "explicit_question_marks",
        "question_like_lines",
        "connector_parts",
        "repeated_clause_parts",
        "numbered_parts",
        "imperative_parts",
        "prefers_extended_answer",
        "requires_single_reply_coverage",
    }
)


def _base_trace() -> dict[str, Any]:
    return {
        "latent_cortex_selected": False,
        "latent_cortex_attempted": False,
        "latent_cortex_succeeded": False,
        "latent_cortex_fallback_used": False,
        "latent_cortex_failure_reason": "",
        "latent_cortex_identity_bound": False,
        "latent_cortex_receipt": {},
        "latent_cortex_progress": {},
        "qualified_recurrent_eligible": False,
        "qualified_recurrent_attempted": False,
        "qualified_recurrent_succeeded": False,
        "qualified_recurrent_shadowed": False,
        "qualified_recurrent_shadow_recorded": False,
        "qualified_recurrent_reason": "",
        "qualified_recurrent_receipt": {},
    }


def select_foreground_episode(
    *,
    foreground: bool,
    desktop_required: bool,
    cognitive_mode: str,
    prompt_shape: dict[str, Any] | None,
    compact_contract: bool,
    strict_output_contract: bool,
    incompatible_contract: bool,
    proof_or_benchmark: bool,
    explicitly_required: bool = False,
    visible_objective: str | None = None,
) -> dict[str, Any]:
    """Return a closed-schema, auditable foreground routing decision."""

    analyzed_shape = analyze_prompt_shape(visible_objective).to_dict()
    supplied_shape = prompt_shape if isinstance(prompt_shape, dict) else {}
    shape: dict[str, Any] = {}
    rejected_shape_keys = sorted(set(supplied_shape) - _PROMPT_SHAPE_KEYS)[:8]
    for key in (
        "question_parts",
        "explicit_question_marks",
        "question_like_lines",
        "connector_parts",
        "repeated_clause_parts",
        "numbered_parts",
        "imperative_parts",
    ):
        supplied = supplied_shape.get(key)
        supplied = supplied if type(supplied) is int and 0 <= supplied <= 512 else 0
        shape[key] = max(supplied, int(analyzed_shape.get(key) or 0))
    for key in ("prefers_extended_answer", "requires_single_reply_coverage"):
        shape[key] = bool(supplied_shape.get(key) or analyzed_shape.get(key))

    question_parts = shape.get("question_parts", 0)
    question_parts = question_parts if type(question_parts) is int else 0
    extended = bool(shape.get("prefers_extended_answer"))
    single_reply_coverage = bool(shape.get("requires_single_reply_coverage"))
    imperative_parts = shape.get("imperative_parts", 0)
    imperative_parts = imperative_parts if type(imperative_parts) is int else 0
    mode = str(cognitive_mode or "").strip().lower()
    depth_worthy = bool(
        explicitly_required
        or mode == "deliberate"
        or extended
        or single_reply_coverage
        or question_parts > 1
    )
    # A compact conversation contract is an execution decision, not a weak
    # hint.  Two natural clauses (report state + distinguish inference) do not
    # justify spending an RLC episode before ordinary speech.  Only genuinely
    # compound structure may override a stale compact classification; explicit
    # RLC requests retain authority regardless of shape.
    exclusion = ""
    compact_shape_override = question_parts >= 3 or imperative_parts >= 3
    if not foreground:
        exclusion = "not_foreground"
    elif not desktop_required:
        exclusion = "desktop_cognitive_engine_not_required"
    elif compact_contract and not (explicitly_required or compact_shape_override):
        exclusion = "compact_contract"
    elif strict_output_contract:
        exclusion = "strict_output_contract"
    elif incompatible_contract:
        exclusion = "incompatible_contract"
    elif proof_or_benchmark and not explicitly_required:
        exclusion = "proof_lane_not_explicitly_opted_in"
    selected = bool(not exclusion and depth_worthy)
    reason = (
        "explicit_requirement"
        if selected and explicitly_required
        else "deliberate_cognitive_mode"
        if selected and mode == "deliberate"
        else "multipart_or_extended_prompt"
        if selected
        else exclusion or "depth_threshold_not_met"
    )
    depth_signal = min(1.0, 0.55 + 0.10 * min(3, question_parts))
    if extended or single_reply_coverage:
        depth_signal = max(depth_signal, 0.75)
    if mode == "deliberate":
        depth_signal = max(depth_signal, 0.80)
    if explicitly_required:
        depth_signal = max(depth_signal, 0.90)
    return {
        "latent_cortex_selected": selected,
        "latent_cortex_selection_reason": reason,
        "latent_cortex_depth_worthy": depth_worthy,
        "latent_cortex_prompt_shape": shape,
        "latent_cortex_prompt_shape_rejected_keys": rejected_shape_keys,
        "stakes": round(max(0.55, depth_signal - 0.05), 3),
        "uncertainty": round(depth_signal, 3),
        "signal_basis": "prompt_shape_heuristic",
        "signal_sources": ["prompt_text_shape"],
        "calibrated_uncertainty": False,
        "consequence_evidence": False,
    }


@dataclass(frozen=True)
class ForegroundLatentOutcome:
    """One complete foreground routing attempt and its ownership disposition."""

    text: str
    trace: dict[str, Any]
    fallback_allowed: bool
    evidence: tuple[str, ...] = ()
    shadow_text: str = ""

    @property
    def selected(self) -> bool:
        return self.trace.get("latent_cortex_selected") is True

    @property
    def attempted(self) -> bool:
        return self.trace.get("latent_cortex_attempted") is True

    @property
    def succeeded(self) -> bool:
        return self.trace.get("latent_cortex_succeeded") is True

    @property
    def answer_available(self) -> bool:
        """Whether this attempt already owns a complete visible answer."""

        return bool(self.text.strip())


def latent_owner_exhausted(reason: str, receipt: dict[str, Any]) -> bool:
    """Return whether a failed episode can still own the resident model.

    A completed receipt-contract failure and a soft cancellation release the
    owner.  Starting a second decode after a timeout, identity failure, or a
    non-terminal episode receipt can collide with a still-cleaning worker.
    """

    normalized = str(reason or "").strip()
    if normalized.startswith(
        (
            "latent_timeout:",
            "latent_integrity:",
            "worker_identity_failed:",
            "runtime_identity_deadline_exhausted",
            "runtime_identity_unbound",
        )
    ):
        return True

    if (
        receipt.get("resident_owner_released") is True
        and receipt.get("resident_state_reusable") is True
    ):
        return False

    terminal_stage = str(receipt.get("last_stage") or "").strip().lower()
    if normalized.startswith("receipt_contract_failed:") and terminal_stage in {
        "complete",
        "completed",
        "finished",
    }:
        return False
    if normalized.startswith("soft_cancel") or normalized in {"cancelled", "canceled"}:
        return False
    # Declining to spend releases the owner, exactly like a soft cancel.
    #
    # LIVE, 2026-08-10: a desktop turn died on
    # "compute budget cannot afford window [0:16) for 9 slots". The window was
    # refused on an accounting precondition BEFORE any layer ran, so nothing was
    # left mid-flight and the resident model was clean — but the episode had an
    # episode_id and had consumed input tokens, which is all the fallthrough
    # below looks at. It concluded the owner was spent, the ordinary generation
    # was refused, and the person got "I couldn't get to an answer I'd stand
    # behind" because an optional enhancement was priced out.
    if normalized.startswith("latent_budget_declined:"):
        return False

    input_tokens = receipt.get("input_token_count")
    consumed_input = type(input_tokens) is int and input_tokens > 0
    return bool(
        str(receipt.get("episode_id") or "").strip()
        and (str(receipt.get("last_stage") or "").strip() or consumed_input)
    )


def materialized_latent_incumbent(
    result: Any,
) -> tuple[str, dict[str, Any]] | None:
    """Return a completed ordinary answer preserved by a failed episode.

    A latent episode materializes its ordinary incumbent before adaptation. A
    later optional-stage failure must not erase that answer or open a second
    resident-model owner. Only the engine's causal receipt can authorize this
    path; arbitrary text on a failed result remains non-servable.
    """

    if not isinstance(result, dict) or result.get("ok") is True:
        return None
    text = str(result.get("text") or "")
    receipt = result.get("receipt")
    if not text.strip() or not isinstance(receipt, dict):
        return None
    raw_flags = receipt.get("honest_flags")
    flags = (
        {
            str(flag or "").strip()
            for flag in raw_flags
            if str(flag or "").strip()
        }
        if isinstance(raw_flags, list)
        else set()
    )
    required = {
        "fallback_reused_materialized_incumbent",
        "vanilla_incumbent_captured_before_adaptation",
    }
    worker_materialized = required.issubset(flags)
    host_materialized = False
    if (
        not worker_materialized
        and "vanilla_incumbent_captured_before_adaptation" in flags
    ):
        try:
            from core.brain.llm.latent_cortex.answer_replacement import (
                validate_host_incumbent_disposition,
            )

            tokens = result.get("tokens")
            if not isinstance(tokens, list):
                raise ValueError("host incumbent tokens are unavailable")
            validate_host_incumbent_disposition(
                receipt.get("host_incumbent_disposition"),
                answer_replacement_receipt=receipt.get("answer_replacement"),
                expected_text=text,
                expected_tokens=tokens,
            )
            host_materialized = True
        except (ImportError, TypeError, ValueError):
            return None
    if not worker_materialized and not host_materialized:
        return None
    if receipt.get("resident_owner_released") is not True:
        return None
    if receipt.get("resident_state_reusable") is not True:
        return None
    return text, dict(receipt)


def _resolve_service() -> Any:
    from core.container import ServiceContainer

    service = ServiceContainer.get("latent_cortex", default=None)
    if service is not None:
        return service
    try:
        from core.runtime.service_registry import get_runtime_service

        return get_runtime_service("latent_cortex", default=None)
    except _RECOVERABLE_ERRORS:
        return None


def _latest_service_evidence(service: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    getter = getattr(service, "get_status", None)
    if not callable(getter):
        return {}, {}
    try:
        status = getter()
    except _RECOVERABLE_ERRORS:
        return {}, {}
    if not isinstance(status, dict):
        return {}, {}
    receipt = status.get("last_failure_receipt") or status.get("last_receipt")
    progress = status.get("last_progress")
    return (
        dict(receipt) if isinstance(receipt, dict) else {},
        dict(progress) if isinstance(progress, dict) else {},
    )


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(maximum, parsed))


async def run_foreground_latent_episode(
    *,
    orchestrator: Any,
    messages: list[dict[str, Any]],
    visible_objective: str,
    foreground: bool,
    desktop_required: bool,
    cognitive_mode: str,
    request_timeout_s: float,
    prompt_shape: dict[str, Any] | None = None,
    compact_contract: bool = False,
    strict_output_contract: bool = False,
    incompatible_contract: bool = False,
    proof_or_benchmark: bool = False,
    explicitly_required: bool = False,
    tenant_id: str = "local",
    user_id: str = "owner",
    session_id: str = "local",
    domain: str = "desktop_conversation",
    decode_max_tokens: int = 768,
    decode_temperature: float = 0.58,
    decode_top_p: float = 0.88,
    recurrent_loops: int = 1,
    steering_alpha: float = 0.25,
    capability_modifiers: dict[str, Any] | None = None,
    service: Any = None,
) -> ForegroundLatentOutcome:
    """Select and, when warranted, execute one full-stack latent episode."""

    trace = _base_trace()
    # Certified recurrent tissue has a narrower but stronger contract than the
    # general foreground selector.  Try it before exact-format/incompatible
    # exclusions, but only when an answer-blind parser recognizes the complete
    # public task grammar.  Unsupported language never acquires the model lane.
    qualified_admission = None
    if foreground and desktop_required and not proof_or_benchmark:
        try:
            from core.brain.llm.qualified_recurrent_ingress import (
                admit_qualified_recurrent_objective,
            )

            qualified_admission = admit_qualified_recurrent_objective(
                visible_objective
            )
        except _RECOVERABLE_ERRORS as exc:
            record_degradation(
                "latent_cortex.qualified_recurrent_classification",
                exc,
                action="retained general foreground routing after typed classification failed",
                severity="warning",
            )
    if qualified_admission is not None:
        trace["qualified_recurrent_eligible"] = True
        semantic_admission = str(qualified_admission.family).startswith("frontier_")
        service = service or _resolve_service()
        runner = getattr(service, "qualified_recurrent_reason", None)
        if callable(runner):
            try:
                qualified = await runner(
                    visible_objective,
                    timeout_s=max(
                        0.0,
                        float(request_timeout_s) - _LATENT_CLEANUP_RESERVE_SECONDS,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - resident-owner safety boundary
                trace.update(
                    {
                        "qualified_recurrent_attempted": True,
                        "qualified_recurrent_reason": (
                            f"qualified_recurrent_runtime_error:{type(exc).__name__}"
                        ),
                    }
                )
                return ForegroundLatentOutcome(
                    text="",
                    trace=trace,
                    fallback_allowed=semantic_admission,
                )
            if isinstance(qualified, dict):
                raw_qualified_receipt = qualified.get("receipt")
                qualified_receipt = (
                    dict(raw_qualified_receipt)
                    if isinstance(raw_qualified_receipt, dict)
                    else {}
                )
                trace.update(
                    {
                        "qualified_recurrent_attempted": bool(
                            qualified.get("attempted")
                        ),
                        "qualified_recurrent_succeeded": qualified.get("ok") is True,
                        "qualified_recurrent_reason": str(
                            qualified.get("reason") or ""
                        )[:160],
                        "qualified_recurrent_receipt": qualified_receipt,
                    }
                )
                if qualified.get("ok") is True:
                    text = str(qualified.get("text") or "").strip()
                    if not text:
                        trace["qualified_recurrent_reason"] = (
                            "qualified_recurrent_empty_answer"
                        )
                        return ForegroundLatentOutcome(
                            text="", trace=trace, fallback_allowed=False
                        )
                    semantic_neural = (
                        qualified.get("reason")
                        == "qualified_semantic_neural_completed"
                    )
                    activation_receipt = qualified_receipt.get("activation_receipt")
                    promotion_mode = (
                        str(activation_receipt.get("promotion_mode") or "")
                        if isinstance(activation_receipt, dict)
                        else ""
                    )
                    if semantic_neural and promotion_mode == "shadow":
                        trace.update(
                            {
                                "latent_cortex_selected": True,
                                "latent_cortex_attempted": True,
                                "latent_cortex_succeeded": False,
                                "latent_cortex_selection_reason": (
                                    "qualified_semantic_neural_shadow"
                                ),
                                "latent_cortex_identity_bound": True,
                                "latent_cortex_receipt": qualified_receipt,
                                "qualified_recurrent_shadowed": True,
                            }
                        )
                        return ForegroundLatentOutcome(
                            text="",
                            trace=trace,
                            fallback_allowed=True,
                            evidence=("qualified_semantic_neural_shadow",),
                            shadow_text=text,
                        )
                    if semantic_neural and promotion_mode != "active":
                        trace["qualified_recurrent_reason"] = (
                            "semantic_neural_promotion_mode_invalid"
                        )
                        return ForegroundLatentOutcome(
                            text="", trace=trace, fallback_allowed=True
                        )
                    trace.update(
                        {
                            "latent_cortex_selected": True,
                            "latent_cortex_attempted": True,
                            "latent_cortex_succeeded": True,
                            "latent_cortex_selection_reason": (
                                "qualified_semantic_neural_exact_domain"
                                if semantic_neural
                                else "qualified_recurrent_exact_domain"
                            ),
                            "latent_cortex_identity_bound": True,
                            "latent_cortex_receipt": qualified_receipt,
                        }
                    )
                    return ForegroundLatentOutcome(
                        text=text,
                        trace=trace,
                        fallback_allowed=False,
                        evidence=(
                            "qualified_semantic_neural_execution"
                            if semantic_neural
                            else "qualified_recurrent_typed_execution",
                        ),
                    )
                if qualified.get("attempted") is True:
                    trace.update(
                        {
                            "latent_cortex_selected": True,
                            "latent_cortex_attempted": True,
                            "latent_cortex_failure_reason": str(
                                qualified.get("reason")
                                or "qualified_recurrent_failed"
                            )[:160],
                        }
                    )
                    return ForegroundLatentOutcome(
                        text="",
                        trace=trace,
                        fallback_allowed=semantic_admission,
                    )

    selection = select_foreground_episode(
        foreground=foreground,
        desktop_required=desktop_required,
        cognitive_mode=cognitive_mode,
        prompt_shape=prompt_shape,
        compact_contract=compact_contract,
        strict_output_contract=strict_output_contract,
        incompatible_contract=incompatible_contract,
        proof_or_benchmark=proof_or_benchmark,
        explicitly_required=explicitly_required,
        visible_objective=visible_objective,
    )
    trace.update(
        {key: value for key, value in selection.items() if key.startswith("latent_cortex_")}
    )
    if selection.get("latent_cortex_selected") is not True:
        return ForegroundLatentOutcome(text="", trace=trace, fallback_allowed=True)

    trace["latent_cortex_attempted"] = True
    service = service or _resolve_service()
    if service is None:
        trace.update(
            {
                "latent_cortex_fallback_used": True,
                "latent_cortex_failure_reason": "latent_service_not_registered",
            }
        )
        return ForegroundLatentOutcome(text="", trace=trace, fallback_allowed=True)

    admission_probe = getattr(service, "foreground_admission", None)
    if callable(admission_probe):
        try:
            service_admission = admission_probe()
        except _RECOVERABLE_ERRORS as exc:
            service_admission = {
                "admitted": False,
                "reason": f"foreground_admission_error:{type(exc).__name__}",
            }
        if isinstance(service_admission, dict) and service_admission.get("admitted") is False:
            trace.update(
                {
                    "latent_cortex_fallback_used": True,
                    "latent_cortex_attempted": False,
                    "latent_cortex_failure_reason": str(
                        service_admission.get("reason")
                        or "latent_service_circuit_open"
                    )[:160],
                    "latent_cortex_service_admission": dict(service_admission),
                }
            )
            return ForegroundLatentOutcome(
                text="",
                trace=trace,
                fallback_allowed=True,
                evidence=("general_latent_service_circuit_open",),
            )

    latent_timeout = max(
        0.0,
        float(request_timeout_s) - _LATENT_CLEANUP_RESERVE_SECONDS,
    )
    if latent_timeout < 15.0:
        trace.update(
            {
                "latent_cortex_fallback_used": True,
                "latent_cortex_failure_reason": "latent_cycle_budget_insufficient",
            }
        )
        return ForegroundLatentOutcome(text="", trace=trace, fallback_allowed=True)

    ingress_stakes = float(selection.get("stakes") or 0.75)
    ingress_uncertainty = float(selection.get("uncertainty") or 0.80)
    ingress_context: list[dict[str, Any]] | None = None
    epistemic_genesis = None
    epistemic_state = None
    memory_result = None
    admitted_evidence: tuple[str, ...] = ()
    try:
        from core.brain.cognitive_ingress import (
            assemble_cognitive_ingress_async,
            cognitive_context_items,
        )

        ingress = await assemble_cognitive_ingress_async(
            orchestrator,
            visible_objective,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
        )
        ingress_stakes = max(ingress.stakes, ingress_stakes - 0.15)
        ingress_uncertainty = ingress.uncertainty
        ingress_context = cognitive_context_items(ingress) or None
        epistemic_genesis = ingress.epistemic_genesis
        epistemic_state = ingress.epistemic_state
        memory_result = ingress.memory_result
        trace["latent_cortex_ingress"] = ingress.to_receipt()
    except _RECOVERABLE_ERRORS as exc:
        record_degradation(
            "latent_cortex.foreground_ingress",
            exc,
            action="used bounded routing estimates after typed cognitive ingress failed",
            severity="warning",
        )

    try:
        from core.brain.capability_evidence_context import (
            build_current_turn_capability_evidence,
            merge_capability_evidence,
        )

        capability_bundle = build_current_turn_capability_evidence(
            capability_modifiers,
            visible_objective,
        )
        ingress_context, merge_receipt = merge_capability_evidence(
            ingress_context,
            capability_bundle,
        )
        trace["latent_cortex_capability_evidence"] = capability_bundle.receipt
        trace["latent_cortex_context_merge"] = merge_receipt
        admitted_evidence = tuple(
            str(item.get("text") or "").strip()
            for item in ingress_context or []
            if isinstance(item, dict)
            and item.get("context_role") in {"memory_observation", "evidence_observation"}
            and str(item.get("text") or "").strip()
        )[:6]
    except _RECOVERABLE_ERRORS as exc:
        record_degradation(
            "latent_cortex.capability_evidence",
            exc,
            action="retained organ ingress after capability evidence admission failed",
            severity="warning",
        )
        trace["latent_cortex_capability_evidence"] = {
            "schema": "aura.rlc.capability_evidence.v1",
            "admitted": False,
            "reason": f"admission_error:{type(exc).__name__}",
        }

    reasoner = getattr(service, "deep_reason_with_acquisition", None)
    acquisition_kwargs: dict[str, Any] = {}
    if callable(reasoner):
        acquisition_kwargs = {
            "orchestrator": orchestrator,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id,
        }
    else:
        reasoner = getattr(service, "deep_reason", None)
    if not callable(reasoner):
        trace.update(
            {
                "latent_cortex_fallback_used": True,
                "latent_cortex_failure_reason": "latent_service_contract_missing",
            }
        )
        return ForegroundLatentOutcome(text="", trace=trace, fallback_allowed=True)

    try:
        result = await reasoner(
            messages=messages,
            **acquisition_kwargs,
            stakes=ingress_stakes,
            uncertainty=ingress_uncertainty,
            domain=str(domain or "desktop_conversation")[:64],
            config_overrides={
                "decode_max_tokens": _bounded_int(decode_max_tokens, 768, 64, 4096),
                "decode_temperature": _bounded_float(decode_temperature, 0.58, 0.0, 1.5),
                "decode_top_p": _bounded_float(decode_top_p, 0.88, 0.05, 1.0),
            },
            runtime_controls={
                "clean_user_surface_recurrent_loops": _bounded_int(recurrent_loops, 1, 1, 2),
                "clean_user_surface_steering_alpha": _bounded_float(
                    steering_alpha, 0.0, 0.0, 1.0
                ),
            },
            timeout_s=latent_timeout,
            require_full_stack=True,
            foreground_request=True,
            cognitive_context=ingress_context,
            epistemic_genesis=epistemic_genesis,
            epistemic_state=epistemic_state,
            selective_memory_result=memory_result,
        )
    except Exception as exc:  # noqa: BLE001 - resident-owner safety boundary
        receipt, progress = _latest_service_evidence(service)
        reason = (
            f"latent_timeout:{type(exc).__name__}"
            if isinstance(exc, TimeoutError)
            else f"latent_integrity:runtime_error:{type(exc).__name__}"
        )
        trace.update(
            {
                "latent_cortex_fallback_used": True,
                "latent_cortex_failure_reason": reason,
                "latent_cortex_receipt": receipt,
                "latent_cortex_progress": progress,
            }
        )
        exhausted = latent_owner_exhausted(reason, receipt)
        record_degradation(
            "latent_cortex.foreground_episode",
            exc,
            action=(
                "suppressed a colliding decoder retry after latent owner exhaustion"
                if exhausted
                else "released the latent lane and retained ordinary foreground decoding"
            ),
            severity="warning",
        )
        return ForegroundLatentOutcome(text="", trace=trace, fallback_allowed=not exhausted)

    if not isinstance(result, dict):
        result = {"ok": False, "reason": "invalid_latent_service_response"}
    receipt = dict(result.get("receipt") or {}) if isinstance(result.get("receipt"), dict) else {}
    progress = (
        dict(result.get("progress") or {}) if isinstance(result.get("progress"), dict) else {}
    )
    text = str(result.get("text") or "").strip()
    if result.get("ok") is True and text:
        runtime_identity = receipt.get("runtime_identity")
        trace.update(
            {
                "latent_cortex_succeeded": True,
                "latent_cortex_identity_bound": bool(
                    isinstance(runtime_identity, dict)
                    and runtime_identity.get("identity_bound") is True
                ),
                "latent_cortex_receipt": receipt,
                "latent_cortex_progress": progress,
                "response_path": "cognitive_engine_latent_cortex",
            }
        )
        return ForegroundLatentOutcome(
            text=text,
            trace=trace,
            fallback_allowed=False,
            evidence=admitted_evidence,
        )

    reason = str(result.get("reason") or "latent_episode_failed")
    if not receipt:
        receipt, latest_progress = _latest_service_evidence(service)
        progress = progress or latest_progress
    trace.update(
        {
            "latent_cortex_fallback_used": True,
            "latent_cortex_failure_reason": reason,
            "latent_cortex_receipt": receipt,
            "latent_cortex_progress": progress,
        }
    )
    incumbent = materialized_latent_incumbent(result)
    if incumbent is not None:
        incumbent_text, incumbent_receipt = incumbent
        trace.update(
            {
                "latent_cortex_receipt": incumbent_receipt,
                "latent_cortex_incumbent_fallback_served": True,
                "response_path": "cognitive_engine_latent_incumbent_fallback",
            }
        )
        return ForegroundLatentOutcome(
            text=incumbent_text,
            trace=trace,
            fallback_allowed=False,
            evidence=("materialized_ordinary_incumbent",),
        )
    exhausted = latent_owner_exhausted(reason, receipt)
    return ForegroundLatentOutcome(text="", trace=trace, fallback_allowed=not exhausted)


__all__ = [
    "ForegroundLatentOutcome",
    "latent_owner_exhausted",
    "materialized_latent_incumbent",
    "run_foreground_latent_episode",
    "select_foreground_episode",
]
