"""Steps of `_mlx_worker_loop`: sizing the cache, reporting the worker ready, strict-answer sampling, the final pass's measurement, and taking a repaired surface.

Lifted whole out of `mlx_worker`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import Any


def _size_the_metal_cache_to_the_machine(
    *,
    mx: Any,
) -> None:
    """Size MLX's cache and active-memory limits to this machine's memory.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 1 name(s) from the turn and hands back
    0.
    """
    from .mlx_worker import (
        _record_mlx_degradation,
        logger,
    )

    try:
        # The fallback must not assume a large host: fixed 24GB/40GB
        # limits authorized more memory than smaller or pressured
        # machines could give. Derive conservatively from total RAM
        # when observable; use small-safe limits when it is not.
        try:
            from core.runtime.resource_observation import get_resource_observer

            _total_gb = float(
                get_resource_observer().memory(include_process_tree=False).total_bytes
            ) / float(1024**3)
        except (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError) as exc:
            logger.debug("Total memory unreadable, reporting 0GB: %s", exc)
            _total_gb = 0.0
        if _total_gb >= 96.0:
            _cache_gb, _active_gb = 24, 40
        elif _total_gb >= 48.0:
            _cache_gb, _active_gb = 12, 28
        elif _total_gb > 0.0:
            _cache_gb, _active_gb = 4, 12
        else:
            # Unobservable capacity: smallest useful limits.
            _cache_gb, _active_gb = 4, 12
        mx.metal.set_cache_limit(1024 * 1024 * 1024 * _cache_gb)
        if hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(1024 * 1024 * 1024 * _active_gb)
    except (AttributeError, RuntimeError, ValueError) as fallback_exc:
        _record_mlx_degradation(
            fallback_exc,
            action="continued without explicit Metal cache limit after fallback failed",
            severity="degraded",
        )


def _say_the_recurrent_tissue_was_admitted(
    *,
    unified_recurrent_qualified_activation_status: Any,
) -> None:
    """Log the unified recurrent tissue admitted for qualified typed serving.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 1 name(s) from the turn and hands back
    0.
    """
    from .mlx_worker import (
        logger,
    )

    if unified_recurrent_qualified_activation_status["loaded"]:
        logger.info(
            "Unified recurrent tissue admitted for QUALIFIED TYPED serving: activation=%s",
            unified_recurrent_qualified_activation_status["activation"]["activation_sha256"],
        )
    else:
        logger.info(
            "Optional unified typed-controller serving inactive (independent of "
            "the CP568 semantic-neural serving lane): %s",
            unified_recurrent_qualified_activation_status["reason"],
        )


def _report_the_worker_ready(
    *,
    _steering_active: Any,
    _steering_disposition: Any,
    device: Any,
    ipc_writer: Any,
    personality_adapter_status: Any,
    recurrent_adapter_activation: Any,
    recurrent_adapter_activation_receipt: Any,
    recurrent_depth_status: Any,
    token_budget_calibration: Any,
    unified_recurrent_qualified_activation_status: Any,
    unified_recurrent_shadow_status: Any,
    worker_identity: Any,
) -> None:
    """Tell the parent the worker is up, with its device, steering and recurrent state.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 12 name(s) from the turn and hands back
    0.
    """
    ipc_writer.put(
        {
            "status": "ok",
            "action": "init",
            "device": device,
            "steering_active": bool(_steering_active),
            "steering_disposition": _steering_disposition,
            "recurrent_depth": recurrent_depth_status,
            "recurrent_adapter_activation": dict(recurrent_adapter_activation),
            "recurrent_adapter_activation_receipt": (
                dict(recurrent_adapter_activation_receipt)
                if recurrent_adapter_activation_receipt is not None
                else None
            ),
            "unified_recurrent_shadow": dict(unified_recurrent_shadow_status),
            "unified_recurrent_qualified_activation": dict(
                unified_recurrent_qualified_activation_status
            ),
            "personality_adapter": dict(personality_adapter_status),
            "token_budget_calibration": token_budget_calibration,
            "worker_identity": dict(worker_identity),
        }
    )


def _sampling_for_a_strict_answer(
    *,
    artifact_generation_contract: Any,
    min_p: Any,
    operator_evidence_contract: Any,
    proof_evaluation_contract: Any,
    repetition_penalty: Any,
    strict_answer_contract: Any,
    strict_value_contract: Any,
    temp: Any,
    top_p: Any,
) -> tuple[Any, Any, Any, Any]:
    """Greedy sampling for a strict answer contract, and what it overrides.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 6 name(s) from the turn and hands back
    4.
    """
    from .mlx_worker import (
        _safe_float,
    )

    if strict_answer_contract:
        temp = 0.0
        top_p = 1.0
        min_p = 0.0
        repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.12)
    elif strict_value_contract:
        temp = 0.0
        top_p = 1.0
        min_p = 0.0
        repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.05)
    elif proof_evaluation_contract:
        if artifact_generation_contract:
            temp = 0.0
            top_p = 1.0
            min_p = 0.0
            repetition_penalty = max(_safe_float(repetition_penalty, 1.08), 1.05)
        else:
            temp = min(_safe_float(temp, 0.1), 0.15)
            top_p = min(_safe_float(top_p, 0.9), 0.9)
            min_p = min(_safe_float(min_p, 0.05), 0.05)
            repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.08)
    elif operator_evidence_contract:
        temp = min(_safe_float(temp, 0.1), 0.12)
        top_p = min(_safe_float(top_p, 0.8), 0.8)
        min_p = max(_safe_float(min_p, 0.03), 0.03)
        repetition_penalty = max(_safe_float(repetition_penalty, 1.15), 1.18)
    return min_p, repetition_penalty, temp, top_p


def _add_the_bound_logits_processor(
    *,
    _bound: Any,
    logits_processors: Any,
) -> None:
    """Add a bound logits processor to the generation, and say so.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 2 name(s) from the turn and hands back
    0.
    """
    from .mlx_worker import (
        logger,
    )

    if _bound is not None:
        logits_processors.append(_bound)
        logger.info(
            "[WORKER] Private channel bounded at %d tokens.",
            getattr(_bound, "budget_tokens", 0),
        )
    else:
        logger.info(
            "[WORKER] Private channel NOT bounded; this "
            "tokenizer has no single closing token."
        )


def _measure_the_final_pass(
    *,
    final_generation_response: Any,
    first_token_latency_s: Any,
    generation_passes: Any,
    generation_performance: Any,
    generation_stream_elapsed_s: Any,
    prefill_tokens: Any,
    token_count: Any,
) -> Any:
    """Measure the final generation pass and record its performance.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 6 name(s) from the turn and hands back
    1.
    """
    from .mlx_worker import (
        _generation_pass_performance,
        logger,
    )

    if final_generation_response is not None:
        generation_performance = _generation_pass_performance(
            generation_passes.final_responses,
            prompt_tokens=prefill_tokens,
            generation_tokens=token_count,
            first_token_seconds=first_token_latency_s,
            stream_seconds=generation_stream_elapsed_s,
        )
        logger.info(
            "[WORKER] generation performance: "
            "prefill=%d tokens/%.2fs (%.1f tok/s), "
            "decode=%d tokens/%.2fs (%.1f tok/s), "
            "first_token=%.2fs stream=%.2fs peak=%.2fGB",
            generation_performance["prompt_tokens"],
            generation_performance["prefill_seconds"] or 0.0,
            generation_performance["prompt_tps"],
            generation_performance["generation_tokens"],
            generation_performance["decode_seconds"] or 0.0,
            generation_performance["generation_tps"],
            first_token_latency_s or 0.0,
            generation_stream_elapsed_s,
            generation_performance["peak_memory_gb"],
        )
    return generation_performance


def _take_the_repaired_user_surface(
    *,
    pre_shape_reasons: Any,
    response_text: Any,
    shaped_surface: Any,
    surface_control_state: Any,
) -> Any:
    """Take a repaired explicit user surface in place of the draft, with its receipt.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 4 name(s) from the turn and hands back
    1.
    """
    from .mlx_worker import (
        append_text_mutation,
        logger,
    )

    if shaped_surface != response_text:
        logger.info(
            "[WORKER] Repaired explicit user-surface "
            "shape before quality validation."
        )
        surface_control_state[
            "instruction_shape_repair_applied"
        ] = True
        append_text_mutation(
            surface_control_state,
            stage="mlx_worker.instruction_shape",
            method="deterministic_instruction_shape",
            reasons=pre_shape_reasons or ["instruction_shape"],
            before=response_text,
            after=shaped_surface,
            deterministic=True,
            authorship_effect="preserved",
        )
        response_text = shaped_surface
    return response_text


def _take_the_restored_newlines(
    *,
    rejection_reasons: Any,
    response_text: Any,
    surface_control_state: Any,
    unescaped_reasons: Any,
    unescaped_surface: Any,
) -> tuple[Any, Any]:
    """Restore newlines the model emitted as literal escapes, with a receipt.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 4 name(s) from the turn and hands back
    2.
    """
    from .mlx_worker import (
        append_text_mutation,
        logger,
    )

    if (
        "escaped_control_artifact"
        not in unescaped_reasons
    ):
        logger.info(
            "[WORKER] Restored newlines the model "
            "emitted as literal escapes."
        )
        append_text_mutation(
            surface_control_state,
            stage="mlx_worker.escaped_control_artifact",
            method="unescape_control_sequences",
            reasons=["escaped_control_artifact"],
            before=response_text,
            after=unescaped_surface,
            deterministic=True,
            authorship_effect="preserved",
        )
        response_text = unescaped_surface
        rejection_reasons = unescaped_reasons
    return rejection_reasons, response_text


def _keep_a_complete_foreground_draft(
    *,
    completed_reasons: Any,
    completed_surface: Any,
    rejection_reasons: Any,
    response_text: Any,
    surface_control_state: Any,
) -> tuple[Any, Any]:
    """Keep a complete foreground draft that passed its checks, with a receipt.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 4 name(s) from the turn and hands back
    2.
    """
    from .mlx_worker import (
        append_text_mutation,
        logger,
    )

    if completed_surface and not completed_reasons:
        logger.info(
            "[WORKER] Kept complete foreground "
            "sentences after a clipped tail."
        )
        append_text_mutation(
            surface_control_state,
            stage="mlx_worker.truncated_tail",
            method="retain_complete_sentences",
            reasons=["truncated_tail"],
            before=response_text,
            after=completed_surface,
            deterministic=True,
            authorship_effect="preserved",
        )
        response_text = completed_surface
        rejection_reasons = []
    return rejection_reasons, response_text


def _take_the_repaired_status_draft(
    *,
    rejection_reasons: Any,
    response_text: Any,
    surface_control_state: Any,
    telemetry_reasons: Any,
    telemetry_surface: Any,
) -> tuple[Any, Any]:
    """Take a repaired live status draft in place of the original, with a receipt.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 5 name(s) from the turn and hands back
    2.
    """
    from .mlx_worker import (
        append_text_mutation,
        logger,
    )

    if telemetry_surface and not telemetry_reasons:
        logger.info(
            "[WORKER] Repaired live status draft "
            "with concrete runtime telemetry."
        )
        append_text_mutation(
            surface_control_state,
            stage="mlx_worker.operational_status",
            method="grounded_runtime_telemetry_repair",
            reasons=rejection_reasons,
            before=response_text,
            after=telemetry_surface,
            deterministic=True,
            authorship_effect="replaced_by_runtime",
        )
        response_text = telemetry_surface
        rejection_reasons = []
    return rejection_reasons, response_text


def _take_a_revalidated_repair(
    *,
    _method: Any,
    _reason: Any,
    candidate: Any,
    candidate_reasons: Any,
    rejection_reasons: Any,
    response_text: Any,
    surface_control_state: Any,
) -> tuple[Any, Any]:
    """Take a repair whose remaining authored answer revalidated, with a receipt.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 6 name(s) from the turn and hands back
    2.
    """
    from .mlx_worker import (
        append_text_mutation,
        logger,
    )

    if (
        candidate
        and candidate
        != str(response_text or "").strip()
    ):
        logger.info(
            "[WORKER] Repaired %s and revalidated the "
            "remaining authored answer.",
            _reason,
        )
        append_text_mutation(
            surface_control_state,
            stage=f"mlx_worker.{_reason}",
            method=_method,
            reasons=[_reason],
            before=response_text,
            after=candidate,
            deterministic=True,
            authorship_effect="preserved",
        )
        response_text = candidate
        rejection_reasons = candidate_reasons
    return rejection_reasons, response_text


def _say_the_inventory_draft_is_ungrounded(
    *,
    inventory_evidence: Any,
) -> None:
    """Warn that a clipped capability inventory draft lacks minimum grounding.

    Moved out of ``_mlx_worker_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 1 name(s) from the turn and hands back
    0.
    """
    from .mlx_worker import (
        logger,
    )

    logger.warning(
        "[WORKER] Clipped capability inventory draft lacks "
        "minimum grounding (%s); keeping the gate failure.",
        ",".join(
            key
            for key, present in inventory_evidence.items()
            if not present
        )
        or "unknown",
    )


