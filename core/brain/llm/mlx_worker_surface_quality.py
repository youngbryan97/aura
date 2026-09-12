"""Whether a draft is fit to show a person, and the controls that shaped it.

The worker can finish a generation and still have nothing worth serving. This
is where that is decided: the surface controls applied before decoding and the
receipt that records what they were, the reasons a draft fails, whether a stop
was semantic or a wall, and whether a continuation may resume from what is
already on screen.
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from typing import Any

from core.brain.live_mind_contract import (
    normalize_text_mutations,
    summarize_text_mutation_authorship,
)
from core.brain.llm.user_surface_recurrence import (
    admit_user_surface_recurrent_loops,
)
from core.conversation.user_surface_contract import (
    UserSurfacePromptResolution,
    resolve_user_surface_prompt,
)
from core.language.terminal_boundary import has_terminal_sentence_boundary
from core.runtime.errors import record_degradation
from core.runtime.model_layers import resolve_model_layers

logger = logging.getLogger("MLXWorker")

from .mlx_worker import _CORRUPT_LANGUAGE_MARKERS, _FUSION_MODEL_IDENTITY


def _record_mlx_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("mlx_worker", exc, severity=severity, action=action)


def _surface_prompt_resolution(job: dict[str, Any]) -> UserSurfacePromptResolution:
    return resolve_user_surface_prompt(job)


def _surface_validation_prompt(job: dict[str, Any]) -> str:
    return _surface_prompt_resolution(job).prompt


# Machine tokens whose IDENTITY is their casing: screaming-snake enum values and
# CamelCase internal symbols. Matching these case-INSENSITIVELY destroyed whole
# replies over ordinary English — "PROCEEDING" is a leaked enum, but
# "proceeding" is just a word, and this pattern is a FATAL check that returns
# None and annihilates the entire answer. Measured live: a conversational turn
# produced a 226-token draft and the user got "I couldn't get to an answer I'd
# stand behind on that one", with the log saying only "Hallucination detected by
# sanitizer. Returning empty text for caller-side recovery."
#
# The natural-language jargon that used to sit in this list ("field coherence",
# "system authority", "memory scar", "precognitive texture", "existence hash")
# is deliberately NOT here: those are style leaks, not model-state corruption,
# they occur legitimately in English, and the reliability gate already handles
# them as `pseudo_internal_jargon` — a reason that can be retried and repaired
# rather than one that throws the answer away.
_BACKEND_SYMBOLIC_SURFACE_MARKERS = re.compile(
    r"\b(?:PROCEEDING|TOOL_ACTION|CONVERGE_UNION|CONFORMED_METHODS|"
    r"TACTICAL_ORGANIZE|UI_SHUTDOWN_OR_DURATIVE_TIMEOUT|"
    r"MySelfEpsilon|CanonicalStabilityAnchor|currentInferenceProblem|"
    r"fieldOfPlay|INTRUSTION_DETECTED|INTRUSION_DETECTED|"
    r"ExistenceHash)\b"
)


def _safe_float(value: Any, default: float) -> float:
    """Finite-only float coercion — these helpers feed steering, sampling,
    retry, receipt, and token-budget paths, where NaN/inf silently poison
    comparisons and sampler construction."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _surface_generation_contract_enabled(job: dict[str, Any]) -> bool:
    """Decide whether this job's decode runs with the steering clamp.

    FAIL-SAFE INVERSION (July 2026 coherence incident): the clamp used to
    apply only to jobs that explicitly carried a surface/strict contract, so
    any route that dropped the flag decoded at full governor steering (up to
    alpha 3.0 — or the install-time 5.0 when the substrate sync was stale).
    Live symptom: fluent spliced-dialog nonsense served to the user. Every
    job is now clamped UNLESS it explicitly opts into full steering
    (latent-cortex episodes, steering experiments) — a dropped flag degrades
    to safe, never to hot.

    Strict/structured proof contracts still matter for the alpha TIER (see
    _surface_control_alpha): they need CORRECT symbolic tokens, not affective
    voice. Running them at full steering (alpha 5.0) corrupts the constrained
    first-token logits → zero-token generation that hangs to the 90s
    first-token timeout (DNU R011/R040/R022 wedges).
    """
    if bool(job.get("allow_full_affective_steering", False)):
        return False
    return True


def _surface_alpha_from_certificate() -> float:
    """How much residual steering this checkpoint has earned on a person's turn.

    Zero until measured. For a long time this was zero unconditionally, and the
    reason given was an A/B whose steered and baseline samples came out
    byte-identical while the statistic still passed. That A/B was void: alpha
    was an absolute number of units added to a residual stream whose magnitude
    grows with width and depth, so the shipped 3.0 sat under the threshold at
    which either model changes its output. Alpha became a fraction of the stream
    and the gate stayed shut, which left her substrate reaching the model as
    text in a prompt and never as part of the computation.

    The comment that closed it asked for a model-specific no-regression
    certificate. `core/consciousness/fusion_certificate.py` is that certificate
    and `tools/measure_fusion_channel.py` earns one. Absent, unreadable and
    failing certificates all return zero, so the failure direction is still
    shut.
    """
    identity = _FUSION_MODEL_IDENTITY
    if not identity:
        return 0.0
    try:
        from core.consciousness.fusion_certificate import certified_alpha

        return certified_alpha(identity)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Fusion certificate lookup unavailable: %s", exc)
        return 0.0


def _surface_control_alpha(job: dict[str, Any], current_alpha: Any) -> float:
    default_alpha = str(_surface_alpha_from_certificate())
    configured = job.get(
        "clean_user_surface_steering_alpha",
        os.environ.get("AURA_USER_SURFACE_STEERING_ALPHA", default_alpha),
    )
    requested = max(0.0, min(_safe_float(configured, 0.0), 1.0))
    try:
        current = float(current_alpha)
    except (TypeError, ValueError):
        current = requested
    if current > 0:
        requested = min(requested, current)
    return max(0.0, requested)


def _surface_control_recurrent_loops(job: dict[str, Any]) -> int:
    # Imported at call time: mlx_worker imports this module, so the other
    # direction cannot be a module-level import.
    from .mlx_worker import _FUSION_MODEL_IDENTITY

    return admit_user_surface_recurrent_loops(
        job.get("clean_user_surface_recurrent_loops"),
        descriptor_sha256=_FUSION_MODEL_IDENTITY,
    )


def _apply_surface_generation_controls(
    engine: Any,
    model: Any,
    job: dict[str, Any],
) -> dict[str, Any]:
    """Clamp latent embellishment when the next tokens are user-visible prose."""
    if not _surface_generation_contract_enabled(job):
        return {"enabled": False}

    state: dict[str, Any] = {"enabled": True, "apply_errors": []}
    alpha = _surface_control_alpha(job, getattr(engine, "_alpha", None))
    state["surface_alpha_requested"] = alpha

    if engine is not None:
        state["engine"] = engine
        state["surface_alpha_override_before"] = getattr(engine, "_surface_alpha_override", None)
        hooks = list(getattr(engine, "_hooks", []) or [])
        state["hook_alphas_before"] = [(hook, getattr(hook, "_alpha", None)) for hook in hooks]
        try:
            if hasattr(engine, "set_surface_alpha_override"):
                engine.set_surface_alpha_override(alpha)
            else:
                for hook in hooks:
                    hook._alpha = min(float(getattr(hook, "_alpha", alpha) or alpha), alpha)
            state["surface_alpha_applied"] = alpha
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            state["apply_errors"].append(f"steering_clamp:{type(exc).__name__}")
            _record_mlx_degradation(
                exc,
                action="recorded steering clamp failure for fail-closed surface admission",
                severity="error",
            )
            logger.warning("Surface steering clamp failed: %s", exc)
    elif alpha == 0.0:
        # A missing optional steering engine is exactly equivalent to a zero
        # steering request.  Treating this as an unapplied control made the
        # neutral, user-visible path depend on the embellishment it disabled.
        state["surface_alpha_applied"] = 0.0
    else:
        state["apply_errors"].append("steering_unavailable")

    layer_view = resolve_model_layers(model)
    inner = layer_view.owner if layer_view is not None else None
    if inner is not None and getattr(inner, "_recurrent_depth_config", None):
        state["recurrent_inner"] = inner
        state["had_recurrent_runtime_loops"] = hasattr(inner, "_recurrent_depth_runtime_loops")
        state["recurrent_runtime_loops_before"] = getattr(
            inner, "_recurrent_depth_runtime_loops", None
        )
        try:
            loops = _surface_control_recurrent_loops(job)
            inner._recurrent_depth_runtime_loops = loops
            state["recurrent_runtime_loops_applied"] = loops
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            state["apply_errors"].append(f"recurrent_clamp:{type(exc).__name__}")
            state["recurrent_clamp_failed"] = True
            _record_mlx_degradation(
                exc,
                action="recorded recurrent-depth clamp failure for fail-closed surface admission",
                severity="error",
            )
            logger.warning("Surface recurrent-depth clamp failed: %s", exc)

    # What a user-visible decode ACTUALLY ran with. Every diagnosis of a bad
    # reply on 2026-07-26 stalled here: the receipt carried these numbers but
    # nothing put them where a live log would show them, so "the clamp is
    # applied" and "the clamp silently no-opped" looked identical from outside.
    if bool(job.get("clean_user_surface_contract", False)):
        logger.info(
            "🎚️ [WORKER] Surface decode: steering α=%s (engine α=%s), "
            "recurrent loops=%s (was %s, depth_present=%s)%s",
            state.get("surface_alpha_applied"),
            getattr(engine, "_alpha", None),
            state.get("recurrent_runtime_loops_applied"),
            state.get("recurrent_runtime_loops_before"),
            state.get("recurrent_inner") is not None,
            (
                " APPLY_ERRORS=" + ",".join(state.get("apply_errors") or [])
                if state.get("apply_errors")
                else ""
            ),
        )
    return state


def _enforce_surface_controls_or_fail(job: dict[str, Any], state: dict[str, Any]) -> None:
    """Fail closed when a user-visible contract selected controls that did
    not apply.

    Decoding a clean-user-surface job WITHOUT the steering/recurrent clamps
    its contract selected serves latent-embellished prose to a user; the
    receipt honestly said applied=False but nothing enforced it. Strict,
    proof, and health jobs keep their own gates and are not blocked here.
    """
    errors = list(state.get("apply_errors") or [])
    if errors and bool(job.get("clean_user_surface_contract", False)):
        raise RuntimeError("surface_controls_unavailable:" + ";".join(errors)[:200])


def _restore_surface_generation_controls(state: dict[str, Any]) -> bool:
    """Restore pre-job control state; False means the resident model may be
    contaminated and the worker must not serve further jobs on it."""
    if not state.get("enabled"):
        return True

    restored = True
    engine = state.get("engine")
    if engine is not None:
        try:
            if hasattr(engine, "set_surface_alpha_override"):
                engine.set_surface_alpha_override(state.get("surface_alpha_override_before"))
            else:
                for hook, alpha in state.get("hook_alphas_before", []):
                    if alpha is not None:
                        hook._alpha = alpha
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            restored = False
            _record_mlx_degradation(
                exc,
                action="flagged worker for recycle after user-surface steering restore failed",
                severity="critical",
            )
            logger.error("Surface steering restore failed: %s", exc)

    inner = state.get("recurrent_inner")
    if inner is not None:
        try:
            if state.get("had_recurrent_runtime_loops"):
                inner._recurrent_depth_runtime_loops = state.get("recurrent_runtime_loops_before")
            elif hasattr(inner, "_recurrent_depth_runtime_loops"):
                delattr(inner, "_recurrent_depth_runtime_loops")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            restored = False
            _record_mlx_degradation(
                exc,
                action="flagged worker for recycle after user-surface recurrent-depth restore failed",
                severity="critical",
            )
            logger.error("Surface recurrent-depth restore failed: %s", exc)
    return restored


def _surface_generation_control_receipt(
    job: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Return an IPC-safe proof of user-surface generation control application."""
    enabled = bool(state.get("enabled"))
    try:
        generation_max_tokens = max(
            1,
            int(state.get("generation_max_tokens_applied") or job.get("max_tokens") or 1),
        )
    except (TypeError, ValueError, OverflowError):
        generation_max_tokens = 1
    receipt: dict[str, Any] = {
        "enabled": enabled,
        # PROVENANCE: the fields named in caller_declared_fields are ECHOES
        # of the caller's job booleans — the worker cannot independently
        # verify them and consumers must not read them as worker-proven
        # facts. Worker-measured evidence lives in worker_verified.
        "caller_declared_fields": [
            "live_mind_controls_bound",
            "clean_user_surface_contract",
            "strict_answer_contract",
            "strict_value_contract",
            "proof_evaluation_contract",
            "operator_evidence_contract",
            "health_probe",
            "runtime_fact_status_contract",
            "grounded_runtime_status_contract",
            "surface_validation_prompt_source",
        ],
        "live_mind_controls_bound": bool(job.get("live_mind_controls_bound", False)),
        "clean_user_surface_contract": bool(job.get("clean_user_surface_contract", False)),
        "surface_validation_prompt_present": bool(_surface_validation_prompt(job)),
        "surface_validation_prompt_bound": _surface_prompt_resolution(job).bound,
        "surface_validation_prompt_binding_valid": _surface_prompt_resolution(job).valid,
        "surface_validation_prompt_source": _surface_prompt_resolution(job).source,
        "surface_validation_prompt_sha256": _surface_prompt_resolution(job).sha256,
        "strict_answer_contract": bool(job.get("strict_answer_contract", False)),
        "strict_value_contract": bool(job.get("strict_value_contract", False)),
        "proof_evaluation_contract": bool(job.get("proof_evaluation_contract", False)),
        "operator_evidence_contract": bool(job.get("operator_evidence_contract", False)),
        "health_probe": bool(job.get("health_probe", False)),
        "runtime_fact_status_contract": bool(job.get("runtime_fact_status_contract", False)),
        "grounded_runtime_status_contract": bool(
            job.get("grounded_runtime_status_contract", False)
        ),
        "generation_max_tokens": generation_max_tokens,
        "memory_pressure_token_cap": job.get("memory_pressure_token_cap"),
        "user_surface_completion_floor": job.get("user_surface_completion_floor"),
        "completion_floor_applied": bool(job.get("completion_floor_applied", False)),
        "caller_requested_max_tokens": job.get("caller_requested_max_tokens"),
        "adaptive_suggested_max_tokens": job.get("adaptive_suggested_max_tokens"),
        "output_contract_generation_floor": job.get("output_contract_generation_floor"),
        "semantic_output_token_cap": job.get("semantic_output_token_cap"),
        "hard_output_token_ceiling": job.get("hard_output_token_ceiling"),
        "generation_stop_reason": state.get("generation_stop_reason"),
        "generation_configured_stop_sequence": state.get("generation_configured_stop_sequence"),
        "semantic_completion_contract": bool(state.get("semantic_completion_contract", False)),
        "semantic_completion_satisfied": bool(state.get("semantic_completion_satisfied", False)),
        "semantic_completion_incomplete": bool(state.get("semantic_completion_incomplete", False)),
        "semantic_completion_missing_part_count": max(
            0,
            _safe_int(state.get("semantic_completion_missing_part_count"), 0),
        ),
        "semantic_completion_missing_part_indexes": list(
            state.get("semantic_completion_missing_part_indexes") or []
        ),
        "semantic_completion_quality_reasons": list(
            state.get("semantic_completion_quality_reasons") or []
        ),
        "semantic_completion_epistemic_partition_covered": state.get(
            "semantic_completion_epistemic_partition_covered"
        ),
        "semantic_completion_terminal_boundary": bool(
            state.get("semantic_completion_terminal_boundary", False)
        ),
        "continuation_resume_requested": bool(state.get("continuation_resume_requested", False)),
        "continuation_resume_applied": bool(state.get("continuation_resume_applied", False)),
        "continuation_resume_available": bool(state.get("continuation_resume_available", False)),
        "conversation_resume_requested": bool(state.get("conversation_resume_requested", False)),
        "conversation_resume_applied": bool(state.get("conversation_resume_applied", False)),
        "conversation_resume_available": bool(state.get("conversation_resume_available", False)),
        "instruction_shape_repair_applied": bool(
            state.get("instruction_shape_repair_applied", False)
        ),
        "text_mutations": normalize_text_mutations(state.get("text_mutations")),
        "applied": False,
    }
    receipt["text_mutation_count"] = len(receipt["text_mutations"])
    resume_handle = str(state.get("continuation_resume_handle") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{32}", resume_handle):
        receipt["continuation_resume_handle"] = resume_handle
    resume_failure = str(state.get("continuation_resume_failure_reason") or "").strip()
    if resume_failure:
        receipt["continuation_resume_failure_reason"] = resume_failure[:120]
    conversation_resume_handle = str(state.get("conversation_resume_handle") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{32}", conversation_resume_handle):
        receipt["conversation_resume_handle"] = conversation_resume_handle
    conversation_resume_failure = str(state.get("conversation_resume_failure_reason") or "").strip()
    if conversation_resume_failure:
        receipt["conversation_resume_failure_reason"] = conversation_resume_failure[:120]
    conversation_resume_output_sha256 = (
        str(state.get("conversation_resume_output_sha256") or "").strip().lower()
    )
    if re.fullmatch(r"[0-9a-f]{64}", conversation_resume_output_sha256):
        receipt["conversation_resume_output_sha256"] = conversation_resume_output_sha256
    receipt["deterministic_repair_applied"] = any(
        bool(item.get("deterministic")) for item in receipt["text_mutations"]
    )
    receipt.update(summarize_text_mutation_authorship(receipt["text_mutations"]))
    for key in (
        "exact_reply_token_count",
        "exact_reply_required_termination_headroom",
        "exact_reply_available_termination_headroom",
        "exact_reply_content_capacity_sufficient",
        "exact_reply_termination_headroom_sufficient",
        "exact_reply_token_ceiling_valid",
        "exact_reply_native_capacity_sufficient",
    ):
        if key in job:
            receipt[key] = job.get(key)
    output_contract = job.get("requested_output_contract")
    if isinstance(output_contract, dict) and output_contract:
        receipt["requested_output_contract"] = dict(output_contract)
    if not enabled:
        return receipt

    if job.get("clean_user_surface_steering_alpha") is not None:
        receipt["surface_alpha_requested"] = _safe_float(
            job.get("clean_user_surface_steering_alpha"),
            0.0,
        )
    if "surface_alpha_applied" in state:
        receipt["surface_alpha_applied"] = state.get("surface_alpha_applied")
    receipt["surface_alpha_applied_ok"] = (
        "surface_alpha_applied" in state or state.get("engine") is None
    )
    # Attribution evidence: what the hooks ACTUALLY injected at (ceiling and
    # staleness derating happen inside the hook, so the requested/applied
    # values alone cannot explain an incident) plus how fresh the substrate
    # sync was. steering_sync_age_s = -1.0 means the sync never ran.
    receipt_engine = state.get("engine")
    # Worker-MEASURED facts (not caller echoes): live steering engine state
    # and clamp application outcomes, sampled at receipt time.
    try:
        receipt["worker_verified"] = {
            "steering_engine_present": receipt_engine is not None,
            "steering_engine_active": bool(receipt_engine.is_active())
            if receipt_engine is not None
            else False,
            "surface_clamp_errors": list(state.get("apply_errors") or []),
            "native_thinking_enabled": bool(state.get("native_thinking_enabled", False)),
            "native_thinking_boundary_closed": bool(
                state.get("native_thinking_boundary_closed", False)
            ),
            "native_thinking_private_chars": max(
                0,
                _safe_int(state.get("native_thinking_private_chars"), 0),
            ),
            # The rate is measured here and needed in the parent process,
            # which sizes the deadline. This receipt is the channel that
            # already crosses that boundary.
            "decode_tokens_per_second": float(state.get("decode_tokens_per_second") or 0.0),
        }
    except (AttributeError, RuntimeError, TypeError) as verify_exc:
        receipt["worker_verified"] = {
            "steering_engine_present": receipt_engine is not None,
            "steering_engine_active": False,
            "verification_error": f"{type(verify_exc).__name__}: {verify_exc}",
            "surface_clamp_errors": list(state.get("apply_errors") or []),
            "native_thinking_enabled": bool(state.get("native_thinking_enabled", False)),
            "native_thinking_boundary_closed": bool(
                state.get("native_thinking_boundary_closed", False)
            ),
            "native_thinking_private_chars": max(
                0,
                _safe_int(state.get("native_thinking_private_chars"), 0),
            ),
            # The rate is measured here and needed in the parent process,
            # which sizes the deadline. This receipt is the channel that
            # already crosses that boundary.
            "decode_tokens_per_second": float(state.get("decode_tokens_per_second") or 0.0),
        }
    receipt_hooks = (
        list(getattr(receipt_engine, "_hooks", []) or []) if receipt_engine is not None else []
    )
    if receipt_hooks:
        effective_alphas = [
            _safe_float(getattr(hook, "_last_effective_alpha", 0.0), 0.0) for hook in receipt_hooks
        ]
        receipt["steering_effective_alpha_max"] = round(max(effective_alphas), 4)
        sync_stamps = [
            _safe_float(getattr(hook, "_last_substrate_sync_monotonic", 0.0), 0.0)
            for hook in receipt_hooks
        ]
        newest_sync = max(sync_stamps)
        receipt["steering_sync_age_s"] = (
            round(max(0.0, time.monotonic() - newest_sync), 3) if newest_sync > 0 else -1.0
        )

    if job.get("clean_user_surface_recurrent_loops") is not None:
        receipt["recurrent_runtime_loops_requested"] = _safe_int(
            job.get("clean_user_surface_recurrent_loops"),
            1,
        )
    receipt["recurrent_depth_present"] = state.get("recurrent_inner") is not None
    if "recurrent_runtime_loops_applied" in state:
        receipt["recurrent_runtime_loops_applied"] = state.get("recurrent_runtime_loops_applied")
    requested_loops = receipt.get("recurrent_runtime_loops_requested")
    applied_loops = receipt.get("recurrent_runtime_loops_applied")
    recurrence_not_applicable = state.get("recurrent_inner") is None
    receipt["recurrent_runtime_loops_applied_ok"] = bool(
        recurrence_not_applicable
        or (
            type(requested_loops) is int
            and type(applied_loops) is int
            and requested_loops == applied_loops
        )
    )
    receipt["applied"] = bool(
        receipt.get("surface_alpha_applied_ok")
        and receipt.get("recurrent_runtime_loops_applied_ok")
        and ("surface_alpha_applied" in state or "recurrent_runtime_loops_applied" in state)
    )
    for key in (
        "surface_quality_gate_enabled",
        "surface_quality_gate_passed",
        "surface_quality_gate_attempts",
        "surface_quality_gate_reasons",
        "telemetry_sanitizer_reasons",
        # The draft the gate rejected, carried rather than destroyed.
        #
        # Blanking it turned "I wrote something a heuristic disliked" into
        # "the client returned no text", which opened the Cortex circuit,
        # tripped the sovereign no-fallback policy, and served Bryan a canned
        # apology while the turn was holding an answer. core/runtime/
        # turn_outcome.py states the rule this restores: a gate ANNOTATES or
        # TRANSFORMS a candidate, it does not destroy one.
        #
        # Carrying it does NOT make it servable. It travels marked as
        # suppressed, and only the caller's recovery path — when the
        # alternative is nothing at all — may serve it.
        "surface_quality_rejected_text",
        "surface_quality_rejected_reasons",
        "surface_quality_gate_error",
        "surface_quality_gate_exemption",
        "surface_quality_gate_waived_reasons",
        "instruction_shape_repair_applied",
        "sentinel_loop_prefix_preserved",
    ):
        if key in state:
            receipt[key] = state.get(key)
    return receipt


def _surface_quality_gate_enabled(job: dict[str, Any]) -> bool:
    if not bool(job.get("clean_user_surface_contract", False)):
        return False
    prompt_resolution = _surface_prompt_resolution(job)
    if not prompt_resolution.prompt and not prompt_resolution.bound:
        return False
    return not bool(
        job.get("health_probe", False)
        or job.get("runtime_fact_status_contract", False)
        or job.get("grounded_runtime_status_contract", False)
        or job.get("operator_evidence_contract", False)
        or job.get("strict_answer_contract", False)
        or job.get("strict_value_contract", False)
        or job.get("proof_evaluation_contract", False)
        or job.get("schema")
    )


def _recent_user_turns(job: dict[str, Any]) -> list[str]:
    """What the person has said, from the transcript the model was given.

    The check this feeds asks whether a reply invents a shared past, and it
    decides that by looking for content appearing nowhere in what was said. An
    empty history makes everything novel, so the check answers "fabricated"
    for a perfectly correct recall.

    It read `user_surface_recent_messages` off the job. Nothing in the tree
    ever put that key in a job — the client builds the payload field by field
    and this one is not among them — so the check has been running against an
    empty conversation since it was written.

    LIVE, 2026-09-07: "What did I just ask you?" had its draft rejected as
    `fabricated_shared_history`, which also disables the prompt cache on the
    repair pass, so a recall question costs a full re-prefill as well as the
    answer.

    Reading it off `messages` is the fix rather than filling the key in: the
    transcript is what the model saw, so the grounding cannot drift out of
    step with what it was answering from.
    """

    stated = job.get("user_surface_recent_messages")
    if isinstance(stated, (list, tuple)) and stated:
        return [str(message or "") for message in stated]
    messages = job.get("messages")
    if not isinstance(messages, (list, tuple)):
        return []
    said: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            said.append(content)
    # The last one is the turn being answered; the check gets that separately
    # as the prompt, and passing it twice narrows nothing.
    # Input admission already bounds this transcript. A second, shorter
    # window would reject a recollection the model was allowed to read.
    return said[:-1]


def _recent_assistant_turns(job: dict[str, Any]) -> list[str]:
    """Prior Aura speech in the exact transcript supplied to this worker."""

    messages = job.get("messages")
    if not isinstance(messages, (list, tuple)):
        return []
    latest_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], dict)
            and str(messages[index].get("role") or "").strip().lower() == "user"
        ),
        len(messages),
    )
    said: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if index >= latest_user_index:
            continue
        if str(message.get("role") or "").strip().lower() != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            said.append(content)
    return said


def _surface_quality_failure_reasons(
    job: dict[str, Any],
    response_text: Any,
) -> list[str]:
    """Validate user-visible drafts inside the worker before IPC success."""
    if not _surface_quality_gate_enabled(job):
        return []
    prompt_resolution = _surface_prompt_resolution(job)
    if prompt_resolution.bound and not prompt_resolution.valid:
        return [prompt_resolution.error or "surface_validation_prompt_binding_invalid"]
    prompt = prompt_resolution.prompt
    if not prompt:
        return []
    recent_messages = _recent_user_turns(job)
    grounding_raw = job.get("user_surface_grounding_evidence")
    transported_grounding = (
        [str(item or "") for item in grounding_raw]
        if isinstance(grounding_raw, (list, tuple))
        else []
    )
    grounding = list(_recent_assistant_turns(job))
    grounding.extend(
        item for item in transported_grounding if item and item not in grounding
    )
    try:
        from core.conversation.response_reliability import assess_user_facing_reply
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_mlx_degradation(
            exc,
            action="blocked live user-surface generation because quality gate was unavailable",
            severity="critical",
        )
        return ["surface_quality_gate_unavailable"]

    candidate = _surface_quality_candidate(job, response_text)
    assessment = assess_user_facing_reply(
        prompt,
        candidate,
        recent_user_messages=recent_messages,
        grounding=grounding,
        sensory_evidence=job.get("user_surface_sensory_evidence"),
        tool_receipts=job.get("user_surface_tool_receipts", ()),
    )
    sanitizer_reasons = _telemetry_sanitization_failure_reasons(
        candidate,
        is_proof=False,
    )
    self_claim_contradiction = False
    self_claim_verification_unavailable = False
    try:
        from core.conversation.self_claim_verifier import verify_self_claims

        self_claim_contradiction = not verify_self_claims(candidate).ok
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        self_claim_verification_unavailable = True
        _record_mlx_degradation(
            exc,
            action="continued surface validation after self-claim verification failed",
            severity="error",
        )
    if (
        assessment.ok
        and not assessment.retryable
        and not assessment.hard_failure
        and not sanitizer_reasons
        and not self_claim_contradiction
        and not self_claim_verification_unavailable
    ):
        return []
    reasons = list(assessment.reasons)
    reasons.extend(sanitizer_reasons)
    if self_claim_contradiction:
        reasons.append("self_claim_contradiction")
    if self_claim_verification_unavailable:
        reasons.append("self_claim_verification_unavailable")
    reasons = list(dict.fromkeys(reasons))
    if not reasons:
        reasons = ["surface_quality_gate_failed"]
    # This gate is about INTEGRITY — leaks, corruption, prompt artefacts and
    # text that is not language. Those reasons may suppress a draft, but the
    # draft remains intact for one bounded authored correction whose wall starts
    # when correction starts. COMPLETENESS is different: a draft that merely
    # fell short is real content and remains the floor the route can deliver.
    # See core/conversation/surface_disposition.py.
    try:
        from core.conversation.surface_disposition import integrity_failures

        reasons = list(integrity_failures(reasons))
    except (ImportError, RuntimeError, TypeError, ValueError):
        reasons = [reason for reason in reasons if reason != "final_answer_missing"]
    if not reasons:
        return []
    if bool(job.get("capability_inventory_contract", False)):
        grounded, _evidence = _capability_inventory_minimum_grounding(response_text)
        if grounded:
            reasons = [
                reason
                for reason in reasons
                if reason
                not in {
                    "too_thin_for_operational_status_turn",
                    "too_thin_for_status_turn",
                    "too_short_for_user_turn",
                    "too_thin_for_user_turn",
                }
            ]
    return reasons


def _surface_quality_candidate(job: dict[str, Any], response_text: Any) -> str:
    """Return the complete authored candidate represented by this decode."""

    tail = str(response_text or "")
    if not bool(job.get("user_surface_continuation_contract", False)):
        return tail
    head = str(job.get("user_surface_continuation_partial") or "")
    if not head:
        return tail
    if not tail:
        return head
    separator = ""
    if not head[-1].isspace() and not tail[0].isspace():
        separator = "" if tail[0] in ".,;:!?)]}" else " "
    return f"{head}{separator}{tail}"


def _semantic_surface_stop_ready(
    job: dict[str, Any],
    response_text: Any,
    *,
    generated_tokens: int,
    minimum_tokens: int | None = None,
) -> bool:
    """Assess visible coverage; this heuristic must not terminate decoding."""

    if not bool(job.get("semantic_completion_contract", False)):
        return False
    required_tokens = (
        max(1, int(minimum_tokens))
        if minimum_tokens is not None
        else 8
        if job.get("user_surface_continuation_contract")
        else 24
    )
    if int(generated_tokens) < required_tokens:
        return False
    candidate = _surface_quality_candidate(job, response_text).rstrip()
    if not has_terminal_sentence_boundary(candidate):
        return False
    try:
        from core.conversation.request_coverage import (
            requested_epistemic_partition_is_covered,
            unanswered_question_parts,
        )
        from core.language.discourse_commitments import unfulfilled_commitments
        from core.runtime.structured_input import analyze_prompt_shape

        if unfulfilled_commitments(candidate):
            return False
        validation_prompt = _surface_validation_prompt(job)
        if not requested_epistemic_partition_is_covered(validation_prompt, candidate):
            return False
        if unanswered_question_parts(
            candidate,
            analyze_prompt_shape(validation_prompt),
        ):
            return False
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action="continued bounded generation because semantic completion proof was unavailable",
            severity="warning",
        )
        return False
    return not _surface_quality_failure_reasons(job, candidate)


def _semantic_completion_receipt_state(
    job: dict[str, Any],
    response_text: Any,
    *,
    generated_tokens: int,
    generation_stop_reason: str = "",
) -> dict[str, Any]:
    """Describe completion without changing the model's decode distribution.

    Terminal punctuation is required while decoding because it is the only
    mechanical evidence that an early stop is safe. A natural model EOS is a
    different boundary: it is the author's explicit end of the utterance, and
    concise answers such as a name, number, path, or label need not be a
    punctuated sentence. EOS therefore closes an otherwise covered and clean
    candidate, while the canonical truncation detector still rejects dangling
    syntax such as ``The latest release is``.
    """

    required = bool(job.get("semantic_completion_contract", False))
    candidate = _surface_quality_candidate(job, response_text).rstrip()
    terminal_boundary = has_terminal_sentence_boundary(candidate)
    missing_indexes: list[int] = []
    discourse_missing: list[dict[str, Any]] = []
    quality_reasons: list[str] = []
    epistemic_covered: bool | None = None
    if required:
        try:
            from core.conversation.request_coverage import (
                requested_epistemic_partition_is_covered,
                unanswered_question_parts,
            )
            from core.conversation.response_reliability import _has_truncated_tail
            from core.language.discourse_commitments import unfulfilled_commitments
            from core.runtime.structured_input import analyze_prompt_shape

            validation_prompt = _surface_validation_prompt(job)
            shape = analyze_prompt_shape(validation_prompt)
            missing = list(unanswered_question_parts(candidate, shape))
            segments = list(getattr(shape, "question_segments", ()) or ())
            missing_indexes = [
                index for index, segment in enumerate(segments) if segment in missing
            ]
            epistemic_covered = requested_epistemic_partition_is_covered(
                validation_prompt,
                candidate,
            )
            quality_reasons = list(_surface_quality_failure_reasons(job, candidate))
            discourse_missing = [
                {
                    "expected_count": item.expected_count,
                    "observed_count": item.observed_count,
                    "kind": item.kind,
                    "declaration": item.declaration,
                }
                for item in unfulfilled_commitments(candidate)
            ]
            eos_completion_boundary = bool(
                str(generation_stop_reason or "").strip().lower() == "eos"
                and candidate
                and not _has_truncated_tail(
                    candidate,
                    generation_stop_reason="eos",
                )
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="reported semantic completion as incomplete because diagnostics were unavailable",
                severity="warning",
            )
            quality_reasons = ["completion_diagnostics_unavailable"]
            eos_completion_boundary = False
    else:
        eos_completion_boundary = False
    # The established validator remains the authority. Diagnostics explain its
    # decision; they do not reimplement it and accidentally create a second,
    # divergent completion contract.
    early_stop_ready = bool(
        required
        and _semantic_surface_stop_ready(
            job,
            response_text,
            generated_tokens=generated_tokens,
            # The default keeps an early sentence from stopping a longer
            # decode. After generation ends, semantic and terminal evidence,
            # rather than length alone, decides whether the answer is complete.
            minimum_tokens=1,
        )
    )
    eos_stop_ready = bool(
        required
        and eos_completion_boundary
        and not missing_indexes
        and not discourse_missing
        and not quality_reasons
        and epistemic_covered is True
    )
    satisfied = bool(early_stop_ready or eos_stop_ready)
    return {
        "semantic_completion_contract": required,
        "semantic_completion_satisfied": satisfied,
        "semantic_completion_incomplete": bool(required and not satisfied),
        "semantic_completion_missing_part_count": len(missing_indexes),
        "semantic_completion_missing_part_indexes": missing_indexes,
        "semantic_completion_unfulfilled_discourse": discourse_missing,
        "semantic_completion_unfulfilled_discourse_count": len(discourse_missing),
        "semantic_completion_quality_reasons": quality_reasons,
        "semantic_completion_epistemic_partition_covered": epistemic_covered,
        "semantic_completion_terminal_boundary": terminal_boundary,
        "semantic_completion_eos_boundary": eos_completion_boundary,
    }


def _semantic_terminal_grace_eligible(
    job: dict[str, Any],
    response_text: Any,
    *,
    generated_tokens: int,
) -> bool:
    """Whether a deadline-cut answer needs only its terminal boundary.

    This is deliberately narrower than a generic timeout extension. Every
    typed request obligation, quality check, and epistemic partition must
    already pass. The worker may then spend a few tokens from the caller's
    existing delivery reserve to close the sentence instead of paying for a
    second full prefill and decode.
    """

    receipt = _semantic_completion_receipt_state(
        job,
        response_text,
        generated_tokens=generated_tokens,
    )
    return bool(
        receipt.get("semantic_completion_contract")
        and not receipt.get("semantic_completion_missing_part_count")
        and not receipt.get("semantic_completion_quality_reasons")
        and receipt.get("semantic_completion_epistemic_partition_covered") is True
        and receipt.get("semantic_completion_terminal_boundary") is False
    )


def _loop_abort_prefix_is_servable(
    job: dict[str, Any],
    response_text: Any,
) -> bool:
    """Whether a repetition abort left enough clean authored work to retain.

    A late loop is a defect in the tail, not evidence that the prefix never
    existed. Restarting a resident decode from token zero discarded hundreds
    of clean tokens and routinely exhausted the owning request deadline before
    the replacement caught up. Keep a substantial prefix when the integrity
    gate finds no positively identified leak or corruption; ordinary
    completion recovery can extend it without re-paying for the whole answer.
    Tiny/unsafe prefixes still take the bounded clean-retry path.
    """

    if not bool(job.get("clean_user_surface_contract", False)):
        return False
    body = str(response_text or "").strip()
    if len(body) < 160 or len(body.split()) < 30:
        return False
    return not _surface_quality_failure_reasons(job, body)


def _classify_generation_stop_reason(
    *,
    soft_cancelled: bool,
    deadline_hit: bool,
    sentinel_aborted: bool,
    role_continuation_hit: bool,
    configured_stop_hit: bool,
    hard_token_limit_hit: bool,
    semantic_contract_satisfied: bool = False,
    generated_tokens: int,
    max_tokens: int,
) -> str:
    """Return the exact terminal event that ended one decode attempt."""

    if soft_cancelled:
        return "soft_cancelled"
    if deadline_hit:
        return "deadline_exceeded"
    if sentinel_aborted:
        return "sentinel_abort"
    if role_continuation_hit:
        return "role_continuation"
    if configured_stop_hit:
        return "configured_stop"
    if hard_token_limit_hit:
        return "hard_token_limit"
    if semantic_contract_satisfied:
        return "semantic_contract_satisfied"
    if int(generated_tokens) >= max(1, int(max_tokens)):
        return "max_tokens"
    return "eos"


def _continuation_resume_unavailable_reason(
    *,
    resume_required: bool,
    cache_lru_available: bool,
    cache_disabled: bool,
    final_cache_available: bool,
    sentinel_aborted: bool,
    response_present: bool,
) -> str:
    """Name a failed required resume, or return empty when none was required."""

    if not resume_required:
        return ""
    if not cache_lru_available:
        return "cache_lru_unavailable"
    if cache_disabled:
        return "cache_disabled_by_contract"
    if not final_cache_available:
        return "final_cache_unavailable"
    if sentinel_aborted:
        return "sentinel_aborted"
    if not response_present:
        return "empty_partial"
    return "cache_retention_refused"


def _continuation_resume_should_bind(
    *,
    generation_stop_reason: str,
    semantic_completion_incomplete: bool,
) -> bool:
    """Whether downstream completion still needs ownership of exact decode state.

    Deadline and token-cap stops need a resume only when the worker can already
    prove the visible answer incomplete. A semantic stop is different: later
    language projections inspect the assembled user surface and can discover an
    obligation that was not present in the worker's raw segment. Retaining its
    cache is therefore part of the successful transaction, not evidence that the
    worker itself judged the answer incomplete.
    """

    stop_reason = str(generation_stop_reason or "").strip().lower()
    if stop_reason == "semantic_contract_satisfied":
        return True
    return bool(
        semantic_completion_incomplete
        and stop_reason in {"deadline_exceeded", "max_tokens", "soft_cancelled"}
    )


def _conversation_resume_boundary_complete(generation_stop_reason: str) -> bool:
    """Whether the cache contains a native completed assistant turn.

    Conversation append is unlike same-turn continuation: it needs the
    model's end-of-turn token in the cache. Every synthetic stop can leave
    token state that is absent from the visible reply, so only natural EOS is
    an admissible boundary.
    """

    return str(generation_stop_reason or "").strip().lower() == "eos"


def _capability_inventory_minimum_grounding(
    response_text: Any,
) -> tuple[bool, dict[str, bool]]:
    """Evidence check for the capability-inventory thin-response exemption.

    Capability inventory questions contain "tools", "external", and
    "desktop", which overlap operational-status classifiers, so a concise
    governed inventory may be exempted from thin-response failures — but
    ONLY when it actually shows its documented evidence: concrete
    categories, governance, and the non-execution boundary. The boundary is
    non-negotiable; the old `or not effect-evidence` escape admitted
    answers with NEITHER effect evidence NOR a boundary.
    """
    reply = str(response_text or "").lower()
    evidence = {
        "category": "browser/web research" in reply
        or ("browser" in reply and ("file" in reply or "desktop" in reply)),
        "governance": any(
            marker in reply
            for marker in ("will/authority", "will and authority", "permission", "governed")
        ),
        "boundary": (
            "not executing" in reply
            or "not opening" in reply
            or "hypothetical" in reply
            or "in this turn" in reply
        ),
        "effect_evidence": "receipt" in reply or "effect verification" in reply,
    }
    grounded = evidence["category"] and evidence["governance"] and evidence["boundary"]
    return grounded, evidence


def _contains_corrupted_language(text: str) -> bool:
    try:
        from core.phases.dialogue_policy import contains_corrupted_language

        return contains_corrupted_language(text)
    except (ImportError, AttributeError):
        return bool(_CORRUPT_LANGUAGE_MARKERS.search(str(text or "")))


def _telemetry_sanitization_failure_reasons(
    text: str,
    is_proof: bool = False,
) -> list[str]:
    """Identify fatal surface leakage without destroying the authored draft."""
    if not text:
        return []

    reasons: list[str] = []

    # 1) Reject telemetry-path walls without blocking legitimate code, regex,
    # filesystem, or proof output. The old slash-count heuristic rejected any
    # answer with more than 15 "/" characters, which is common in live coding
    # tasks and path-aware proof/eval runs.
    # The path-wall check stays non-proof: coding and path-aware eval answers
    # legitimately contain many paths, and unlike the symbolic markers there
    # is no exact token that separates a wall from real content.
    if not is_proof:
        slash_count = text.count("/")
        if slash_count > 30 and "http" not in text.lower():
            path_like = re.findall(r"(?:/[A-Za-z0-9._-]+){3,}", text)
            path_chars = sum(len(path) for path in path_like)
            if len(path_like) >= 3 or path_chars > max(120, int(len(text) * 0.35)):
                reasons.append("telemetry_path_wall")

    # 3) Extreme numeric sequences: in conversational output a 20+ digit run
    # is a hallucination signature. Proof/eval answers are exempt — large
    # integers, hashes, and numeric test vectors are legitimate exact
    # answers there, and correctness is the eval harness's job to score.
    if not is_proof and re.search(r"\d{20,}", text):
        reasons.append("unbounded_numeric_identifier")

    # 4) Corrupted lexical output is a model-state failure, not a usable
    # answer — in EVERY mode. A proof answer containing corruption tokens is
    # corrupted evidence, so the proof exemption never applied here.
    if _contains_corrupted_language(text):
        reasons.append("corrupted_language")
    # 5) Backend-symbolic surface markers apply in EVERY mode. The exemption
    # here was justified by a pattern that no longer exists: it claimed the
    # regex matched common English words ("proceeding", "field coherence"),
    # but the markers are exact-case backend identifiers — PROCEEDING,
    # TOOL_ACTION, ExistenceHash — matched WITHOUT re.IGNORECASE, so
    # lowercase prose never matched, and "field coherence" is not in this
    # pattern at all. A proof answer containing a raw backend action code is
    # leaked internals wherever it appears.
    if _BACKEND_SYMBOLIC_SURFACE_MARKERS.search(text):
        reasons.append("backend_symbolic_surface_leak")

    return list(dict.fromkeys(reasons))


def _sanitize_telemetry_leakage(text: str, is_proof: bool = False) -> str | None:
    """Legacy strict-path adapter for the typed telemetry sanitizer.

    Strict/proof callers still receive ``None`` for an unspeakable draft. Live
    user surfaces consume the typed reasons through the quality-repair lane so
    the original draft remains available for bounded authored correction.
    """
    if _telemetry_sanitization_failure_reasons(text, is_proof=is_proof):
        return None

    return text
