"""Shared controlled-recurrence math and its machine-checkable contract."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from core.brain.llm.latent_cortex.workspace import per_position_rms

LOOP_CORE_SCHEMA = "aura.rlc.loop_core.v1"
KV_BOUND_SCHEMA = "aura.rlc.kv_bound.v2"
UPDATE_IMPLEMENTATION = "core.brain.llm.latent_cortex.loop_core.controlled_recurrent_update"
ABSOLUTE_POSITION_LIMIT = 1_048_576


class LoopCoreError(RuntimeError):
    """The recurrent state violated a fail-closed numerical invariant."""


class ActionContinuationDrift(RuntimeError):
    """The restored continuation is not the state the episode left behind.

    Detection worked and the reason did not survive the trip out. Reasons
    reach the caller with the exception's message stripped, because messages
    here can carry local paths and processed text, so the caller saw
    ``latent_phase_failed:RuntimeError`` — the same string a numerical
    invariant produces, needing the opposite response. The class name is the
    part that is safe to publish, so the class is where the meaning goes.
    """


class ComputeBudgetUnaffordable(RuntimeError):
    """The episode declined to spend, which is not the same as breaking.

    LIVE, 2026-08-10. A desktop chat turn died with

        latent_phase_failed:RuntimeError:compute budget cannot afford
        window [0:16) for 9 slots

    and the person got "I couldn't get to an answer I'd stand behind."

    The window was refused BEFORE any layer ran, on an accounting
    precondition. Nothing about the model was left mid-flight. But the refusal
    arrived as a bare RuntimeError, indistinguishable from a numerical
    invariant blowing up inside a decode, and latent_owner_exhausted() — whose
    real question is "could a second decode collide with a still-cleaning
    worker?" — saw an episode that had reached a stage and consumed input
    tokens, concluded the resident owner was spent, and refused the ordinary
    generation that would have answered the turn.

    So this is its own type: a decision not to spend releases the owner, and
    the ordinary path can serve the answer.
    """


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def alpha_for_step(
    *,
    alpha: float,
    schedule: str,
    max_steps: int,
    step: int,
) -> float:
    """Return the shared train/live residual scale for one recurrent step."""

    if not _finite_number(alpha) or not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and inside (0, 1]")
    if schedule not in {"constant", "cosine"}:
        raise ValueError("alpha schedule must be constant or cosine")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if type(step) is not int or not 0 <= step < max_steps:
        raise ValueError("step must be inside the recurrent horizon")
    if schedule == "constant":
        return float(alpha)
    horizon = max(1, max_steps - 1)
    progress = min(1.0, step / horizon)
    return float(alpha) * (0.25 + 0.75 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def rms_match(new_state: Any, anchor_state: Any, clip_ratio: float) -> Any:
    """Clamp per-position RMS to a fixed anchor's activation trust band."""

    if not _finite_number(clip_ratio) or float(clip_ratio) < 1.0:
        raise ValueError("clip_ratio must be finite and at least 1")
    if tuple(new_state.shape) != tuple(anchor_state.shape):
        raise ValueError("RMSMatch state and anchor shapes differ")
    import mlx.core as mx

    new_rms = mx.maximum(per_position_rms(new_state), 1e-6)
    anchor_rms = per_position_rms(anchor_state)
    target = mx.clip(
        new_rms,
        anchor_rms / float(clip_ratio),
        anchor_rms * float(clip_ratio),
    )
    return new_state * (target / new_rms)


def controlled_recurrent_update(
    previous_state: Any,
    candidate_state: Any,
    anchor_state: Any,
    *,
    alpha: float,
    clip_ratio: float,
) -> Any:
    """The single train/live update operator for stable latent recurrence."""

    if tuple(previous_state.shape) != tuple(candidate_state.shape) or tuple(
        previous_state.shape
    ) != tuple(anchor_state.shape):
        raise ValueError("recurrent update state shapes differ")
    if not _finite_number(alpha) or not 0.0 < float(alpha) <= 1.0:
        raise ValueError("recurrent update alpha must be inside (0, 1]")
    for stage, value in (
        ("shared_update_input", previous_state),
        ("shared_update_candidate", candidate_state),
        ("shared_update_anchor", anchor_state),
    ):
        assert_finite_state(value, stage=stage)
    matched = rms_match(candidate_state, anchor_state, clip_ratio)
    blended = (1.0 - float(alpha)) * previous_state + float(alpha) * matched
    # A convex blend of two individually bounded vectors can still collapse
    # through cancellation. Re-apply the fixed-anchor band to the state that
    # actually enters the next recurrent step.
    import mlx.core as mx

    previous_or_anchor = mx.where(
        per_position_rms(previous_state) > 1e-6,
        previous_state,
        anchor_state,
    )
    stabilized = mx.where(
        per_position_rms(blended) > 1e-6,
        blended,
        previous_or_anchor,
    )
    bounded = rms_match(stabilized, anchor_state, clip_ratio)
    assert_finite_state(bounded, stage="shared_update_output")
    return bounded


def assert_finite_state(value: Any, *, stage: str) -> None:
    """Synchronously refuse a non-finite tensor before it enters bookkeeping."""

    import mlx.core as mx

    if not bool(mx.all(mx.isfinite(value))):
        raise LoopCoreError(f"recurrent state is non-finite at {stage}")


def build_loop_core_contract(
    *,
    prelude_end: int,
    coda_start: int,
    max_steps: int,
    min_steps: int,
    alpha: float,
    alpha_schedule: str,
    rms_clip_ratio: float,
    convergence_eps: float,
    divergence_ratio: float,
    fixed_depth: bool,
) -> dict[str, Any]:
    """Bind the exact stable-loop semantics used by training or inference."""

    payload = {
        "schema": LOOP_CORE_SCHEMA,
        "update_implementation": UPDATE_IMPLEMENTATION,
        "anchor_policy": "fixed_post_prelude_state_v1",
        "residual_policy": "mailbox_plus_mutable_hypothesis_mean_rms_v1",
        "finite_policy": "input_candidate_output_fail_closed_v1",
        "cache_policy": "snapshot_restore_and_position_bound_v1",
        "prelude_end": prelude_end,
        "coda_start": coda_start,
        "max_steps": max_steps,
        "min_steps": min_steps,
        "alpha": alpha,
        "alpha_schedule": alpha_schedule,
        "rms_clip_ratio": rms_clip_ratio,
        "convergence_eps": convergence_eps,
        "divergence_ratio": divergence_ratio,
        "fixed_depth": fixed_depth,
    }
    contract = {**payload, "semantic_sha256": canonical_sha256(payload)}
    return validate_loop_core_contract(contract)


def validate_loop_core_contract(
    value: Any,
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and optionally byte-match a stable-loop semantic contract."""

    fields = {
        "schema",
        "update_implementation",
        "anchor_policy",
        "residual_policy",
        "finite_policy",
        "cache_policy",
        "prelude_end",
        "coda_start",
        "max_steps",
        "min_steps",
        "alpha",
        "alpha_schedule",
        "rms_clip_ratio",
        "convergence_eps",
        "divergence_ratio",
        "fixed_depth",
        "semantic_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("loop-core contract fields do not match schema")
    payload = {key: value[key] for key in fields - {"semantic_sha256"}}
    if value["semantic_sha256"] != canonical_sha256(payload):
        raise ValueError("loop-core contract commitment mismatch")
    if (
        value["schema"] != LOOP_CORE_SCHEMA
        or value["update_implementation"] != UPDATE_IMPLEMENTATION
        or value["anchor_policy"] != "fixed_post_prelude_state_v1"
        or value["residual_policy"] != "mailbox_plus_mutable_hypothesis_mean_rms_v1"
        or value["finite_policy"] != "input_candidate_output_fail_closed_v1"
        or value["cache_policy"] != "snapshot_restore_and_position_bound_v1"
        or type(value["prelude_end"]) is not int
        or type(value["coda_start"]) is not int
        or not 0 <= value["prelude_end"] < value["coda_start"]
        or type(value["max_steps"]) is not int
        or type(value["min_steps"]) is not int
        or not 1 <= value["min_steps"] <= value["max_steps"] <= 64
        or not _finite_number(value["alpha"])
        or not 0.0 < float(value["alpha"]) <= 1.0
        or value["alpha_schedule"] not in {"constant", "cosine"}
        or not _finite_number(value["rms_clip_ratio"])
        or not 1.0 <= float(value["rms_clip_ratio"]) <= 100.0
        or not _finite_number(value["convergence_eps"])
        or not 0.0 < float(value["convergence_eps"]) <= 1.0
        or not _finite_number(value["divergence_ratio"])
        or not 1.0 < float(value["divergence_ratio"]) <= 1000.0
        or type(value["fixed_depth"]) is not bool
        or not _is_sha256(value["semantic_sha256"])
    ):
        raise ValueError("loop-core contract values are invalid")
    if expected is not None and value != validate_loop_core_contract(expected):
        raise ValueError("loop-core contract differs from expected execution")
    return dict(value)


def transition_metrics(
    previous_state: Any,
    next_state: Any,
    anchor_state: Any,
    *,
    alpha: float,
    convergence_eps: float,
    previous_residual: float | None = None,
    previous_delta: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    """Measure fixed-point dynamics without exposing the latent tensor."""

    import mlx.core as mx

    for stage, value in (
        ("diagnostic_input", previous_state),
        ("diagnostic_output", next_state),
        ("diagnostic_anchor", anchor_state),
    ):
        assert_finite_state(value, stage=stage)
    if tuple(previous_state.shape) != tuple(next_state.shape) or tuple(
        previous_state.shape
    ) != tuple(anchor_state.shape):
        raise ValueError("diagnostic state shapes differ")
    delta = next_state - previous_state
    residual_num = mx.mean(per_position_rms(delta))
    residual_den = mx.maximum(mx.mean(per_position_rms(previous_state)), 1e-6)
    residual = float(residual_num / residual_den)
    input_rms = float(mx.mean(per_position_rms(previous_state)))
    output_rms = float(mx.mean(per_position_rms(next_state)))
    anchor_rms = float(mx.mean(per_position_rms(anchor_state)))
    reported_residual = round(residual, 8)
    reported_input_rms = round(input_rms, 8)
    reported_output_rms = round(output_rms, 8)
    reported_anchor_rms = round(anchor_rms, 8)
    anchor_ratio = reported_output_rms / max(reported_anchor_rms, 1e-6)
    contraction_ratio = None
    if previous_residual is not None:
        if not _finite_number(previous_residual) or float(previous_residual) < 0.0:
            raise ValueError("previous residual is invalid")
        contraction_ratio = reported_residual / max(float(previous_residual), 1e-9)
    delta_cosine = None
    if previous_delta is not None:
        numerator = mx.sum(previous_delta * delta)
        denominator = mx.maximum(
            mx.sqrt(mx.sum(previous_delta * previous_delta)) * mx.sqrt(mx.sum(delta * delta)),
            1e-9,
        )
        delta_cosine = float(numerator / denominator)
        delta_cosine = max(-1.0, min(1.0, delta_cosine))
    values = (residual, input_rms, output_rms, anchor_rms, anchor_ratio)
    if (
        any(not math.isfinite(value) for value in values)
        or (contraction_ratio is not None and not math.isfinite(contraction_ratio))
        or (delta_cosine is not None and not math.isfinite(delta_cosine))
    ):
        raise LoopCoreError("recurrent diagnostics are non-finite")
    metrics = {
        "alpha": round(float(alpha), 8),
        "input_mean_rms": reported_input_rms,
        "output_mean_rms": reported_output_rms,
        "anchor_mean_rms": reported_anchor_rms,
        "anchor_rms_ratio": round(anchor_ratio, 8),
        "residual": reported_residual,
        "contraction_ratio": (None if contraction_ratio is None else round(contraction_ratio, 8)),
        "delta_cosine": (None if delta_cosine is None else round(delta_cosine, 8)),
        "contracting": (None if contraction_ratio is None else contraction_ratio < 1.0),
        "oscillating": bool(delta_cosine is not None and delta_cosine < -0.5),
        "fixed_point_candidate": reported_residual < float(convergence_eps),
        "all_finite": True,
    }
    mx.eval(delta)
    return metrics, delta


def validate_kv_bound_receipt(value: Any) -> dict[str, Any]:
    """Validate the per-call recurrent KV position-limit evidence."""

    fields = {
        "schema",
        "position_limit",
        "position_limit_source",
        "call_count",
        "max_context_tokens",
        "max_total_tokens",
        "all_within_limit",
        "calls",
        "calls_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("KV-bound receipt fields do not match schema")
    calls = value["calls"]
    if not isinstance(calls, list) or not calls:
        raise ValueError("KV-bound receipt has no calls")
    row_fields = {
        "ordinal",
        "start",
        "end",
        "tokens",
        "context_tokens",
        "total_tokens",
        "post_context_tokens",
        "persist",
        "restored",
    }
    legacy = value["schema"] == "aura.rlc.kv_bound.v1"

    def valid_context(row):
        context = row["context_tokens"]
        total = row["total_tokens"]
        post = row["post_context_tokens"]
        if context is None or total is None or post is None:
            return not legacy and context is None and total is None and post is None
        return (
            type(context) is int
            and context >= 0
            and type(total) is int
            and total == context + row["tokens"]
            and type(post) is int
            and post >= 0
            and post == (total if row["persist"] else context)
        )

    if any(
        not isinstance(row, dict)
        or set(row) != row_fields
        or row["ordinal"] != index
        or type(row["start"]) is not int
        or type(row["end"]) is not int
        or not 0 <= row["start"] < row["end"]
        or type(row["tokens"]) is not int
        or row["tokens"] < 1
        or not valid_context(row)
        or type(row["persist"]) is not bool
        or type(row["restored"]) is not bool
        or row["restored"] is not (not row["persist"])
        for index, row in enumerate(calls)
    ):
        raise ValueError("KV-bound call evidence is invalid")
    position_limit = value["position_limit"]
    if (
        value["schema"] not in {"aura.rlc.kv_bound.v1", KV_BOUND_SCHEMA}
        or type(position_limit) is not int
        or not 1 <= position_limit <= ABSOLUTE_POSITION_LIMIT
        or value["position_limit_source"] not in {"model_config", "absolute_safety_ceiling"}
        or value["call_count"] != len(calls)
        or value["max_context_tokens"]
        != max(
            (row["context_tokens"] for row in calls if row["context_tokens"] is not None),
            default=None,
        )
        or value["max_total_tokens"]
        != max(
            (row["total_tokens"] for row in calls if row["total_tokens"] is not None), default=None
        )
        or value["all_within_limit"] is not True
        or any(
            (row["total_tokens"] if row["total_tokens"] is not None else row["tokens"])
            > position_limit
            for row in calls
        )
        or value["calls_sha256"] != canonical_sha256(calls)
    ):
        raise ValueError("KV-bound receipt summary is invalid")
    return dict(value)


__all__ = [
    "ABSOLUTE_POSITION_LIMIT",
    "ComputeBudgetUnaffordable",
    "KV_BOUND_SCHEMA",
    "LOOP_CORE_SCHEMA",
    "LoopCoreError",
    "alpha_for_step",
    "assert_finite_state",
    "build_loop_core_contract",
    "canonical_sha256",
    "controlled_recurrent_update",
    "rms_match",
    "transition_metrics",
    "validate_kv_bound_receipt",
    "validate_loop_core_contract",
]
