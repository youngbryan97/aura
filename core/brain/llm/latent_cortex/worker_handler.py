"""Worker-side handler for the ``latent_reason`` IPC action.

Lives in its own module so ``mlx_worker.py`` gains one surgical elif and the
whole latent-reasoning surface stays independently testable. The handler is
synchronous and runs inside the worker process while the caller holds the
metal semaphore — the resident model is exclusively ours for the episode.

Job contract (all optional except the prompt source):
{
  "action": "latent_reason",
  "id": "...",
  "prompt": "..."            # or "messages": [...]
  "domain": "general",
  "response_contract": "{...}", # optional public shape DSL, no answer values
  "config": {                # conservative defaults; hard caps in types.py
     "n_slots": 16, "n_branches": 2, "max_steps": 8,
     "latent_opt": false, "fast_weights": false,
     "decode_max_tokens": 512, "decode_temperature": 0.0,
     "verifier_probe_max_tokens": 48,
     "verifier_probe_contract": "none",
     "verifier_accept_non_regression": false,
     "decode_bridge_policy": "none",
     "decode_incumbent_policy": "vanilla_incumbent",
     "schedule": {...}       # optional explicit program
  },
  "budget": {"max_layer_apps": ..., "wall_clock_s": ...}
}

Kill switch: AURA_LATENT_CORTEX=0 refuses every episode with an honest
reason — the caller falls back to ordinary generation, no silence.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.engine import (
    LatentCortexEngine,
    LatentEngineBusyError,
)
from core.brain.llm.latent_cortex.schedules import ScheduleLibrary
from core.brain.llm.latent_cortex.types import (
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    FastWeightsConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)

logger = logging.getLogger("Aura.LatentCortex.WorkerHandler")

_schedule_library: ScheduleLibrary | None = None

_CONFIG_KEYS = {
    "alpha",
    "alpha_schedule",
    "allow_vanilla_fallback",
    "anchor_scale",
    "coda_frac",
    "collapse_cos_threshold",
    "branch_correlation_evidence",
    "critic_blind_spot_evidence",
    "comm_slot",
    "convergence_eps",
    "decode_max_tokens",
    "decode_contract",
    "decode_contract_grace_tokens",
    "decode_min_tokens",
    "decode_bridge_policy",
    "decode_incumbent_policy",
    "decode_repetition_penalty",
    "decode_repetition_window",
    "decode_temperature",
    "decode_top_p",
    "divergence_ratio",
    "escape",
    "exchange_gamma",
    "exchange_interval",
    "fast_weights",
    "fast_weights_canary",
    "fast_weights_canary_generated",
    "fast_weights_canary_max_delta_rms",
    "fast_weights_canary_max_drop",
    "fast_weights_canary_max_tokens",
    "fast_weights_canary_rescale_attempts",
    "fast_weights_lr",
    "fast_weights_layer_placement",
    "fast_weights_max_layers",
    "fast_weights_export_candidates",
    "fast_weights_opt_steps",
    "fast_weights_output_memory_diagnostic",
    "fast_weights_query_gate",
    "fast_weights_query_gate_temperature",
    "fast_weights_query_gate_threshold",
    "fast_weights_rank",
    "fast_weights_scale",
    "fast_weights_target",
    "generative_verifier_enabled",
    "generative_verifier_max_atoms",
    "generative_verifier_max_tokens",
    "counterfactual_verifier_enabled",
    "counterfactual_verifier_max_atoms",
    "counterfactual_verifier_max_interventions",
    "counterfactual_verifier_max_tokens",
    "prefix_stability_enabled",
    "prefix_stability_samples",
    "prefix_stability_max_tokens",
    "prefix_stability_temperature",
    "prefix_stability_top_p",
    "prefill_chunk_tokens",
    "prefix_stability_seed",
    "prefix_stability_calibrator",
    "local_repair_enabled",
    "local_repair_max_attempts",
    "local_repair_max_tokens",
    "answer_replacement_enabled",
    "objective_program_enabled",
    "verified_objective_teacher_enabled",
    "answer_replacement_margin",
    "verifier_fusion_evidence",
    "jitter_scale",
    "input_context_max_chars",
    "isolation_steps",
    "latent_opt",
    "latent_opt_control",
    "latent_opt_lambda_manifold",
    "latent_opt_lambda_reconstruct",
    "latent_opt_lr",
    "latent_opt_max_grad_norm",
    "latent_opt_steps",
    "max_steps",
    "min_steps",
    "n_branches",
    "n_slots",
    "prelude_frac",
    "rms_clip_ratio",
    "schedule",
    "seed",
    "halting",
    "probe_cache",
    "telemetry",
    "contradiction_head",
    "contradiction_perturber",
    "local_exploration",
    "heterogeneous_integration",
    "transient_negative_constraints",
    "virtual_quanta",
    "latent_tree_search",
    "mistake_locator",
    "uncertainty_head",
    "update_gate",
    "verifier_accept_non_regression",
    "verifier_probe_contract",
    "verifier_probe_max_tokens",
}


def _typed_value(raw: dict[str, Any], key: str, default: Any, expected: type) -> Any:
    value = raw.get(key, default)
    if expected is bool:
        if type(value) is not bool:
            raise ValueError(f"{key} must be a JSON boolean")
        return value
    if expected is int:
        if type(value) is not int:
            raise ValueError(f"{key} must be a JSON integer")
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a JSON number")
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be a finite JSON number") from exc
    if expected is str:
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value
    raise TypeError(f"unsupported wire type for {key}")


def cortex_enabled() -> bool:
    return str(os.environ.get("AURA_LATENT_CORTEX", "1")).strip() != "0"


def _library() -> ScheduleLibrary | None:
    """Process-wide schedule library, persisted under the data dir."""
    global _schedule_library
    if _schedule_library is None:
        try:
            from core.config import DATA_DIR

            path = Path(DATA_DIR) / "latent_cortex" / "schedule_library.json"
            _schedule_library = ScheduleLibrary(path)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Schedule library unavailable (%s); using defaults.", exc)
            _schedule_library = ScheduleLibrary(None)
    return _schedule_library


def config_from_job(job_config: dict[str, Any] | None) -> CortexConfig:
    """Translate the wire config into a validated CortexConfig."""
    if job_config is not None and not isinstance(job_config, dict):
        raise ValueError("latent_reason config must be a mapping")
    raw = dict(job_config or {})
    unknown = sorted(set(raw) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"latent_reason config contains unknown keys: {unknown}")
    cfg = CortexConfig(
        workspace=WorkspaceConfig(
            n_slots=_typed_value(raw, "n_slots", 16, int),
            seed=_typed_value(raw, "seed", 0, int),
            anchor_scale=_typed_value(raw, "anchor_scale", 0.05, float),
        ),
        recurrence=RecurrenceConfig(
            max_steps=_typed_value(raw, "max_steps", 8, int),
            min_steps=_typed_value(raw, "min_steps", 2, int),
            alpha=_typed_value(raw, "alpha", 0.5, float),
            alpha_schedule=_typed_value(raw, "alpha_schedule", "cosine", str),
            rms_clip_ratio=_typed_value(raw, "rms_clip_ratio", 3.0, float),
            convergence_eps=_typed_value(raw, "convergence_eps", 0.02, float),
            divergence_ratio=_typed_value(raw, "divergence_ratio", 10.0, float),
        ),
        branches=BranchConfig(
            n_branches=_typed_value(raw, "n_branches", 2, int),
            isolation_steps=_typed_value(raw, "isolation_steps", 2, int),
            exchange_interval=_typed_value(raw, "exchange_interval", 4, int),
            exchange_gamma=_typed_value(raw, "exchange_gamma", 0.35, float),
            comm_slot=_typed_value(raw, "comm_slot", 0, int),
            collapse_cos_threshold=_typed_value(raw, "collapse_cos_threshold", 0.98, float),
            jitter_scale=_typed_value(raw, "jitter_scale", 0.02, float),
        ),
        latent_opt=LatentOptConfig(
            enabled=_typed_value(raw, "latent_opt", False, bool),
            steps=_typed_value(raw, "latent_opt_steps", 4, int),
            lr=_typed_value(raw, "latent_opt_lr", 0.05, float),
            lambda_reconstruct=_typed_value(raw, "latent_opt_lambda_reconstruct", 1.0, float),
            lambda_manifold=_typed_value(raw, "latent_opt_lambda_manifold", 0.5, float),
            max_grad_norm=_typed_value(raw, "latent_opt_max_grad_norm", 1.0, float),
            control_mode=_typed_value(raw, "latent_opt_control", False, bool),
        ),
        fast_weights=FastWeightsConfig(
            enabled=_typed_value(raw, "fast_weights", False, bool),
            rank=_typed_value(raw, "fast_weights_rank", 2, int),
            scale=_typed_value(raw, "fast_weights_scale", 1.0, float),
            target=_typed_value(raw, "fast_weights_target", "o_proj", str),
            layer_placement=_typed_value(
                raw,
                "fast_weights_layer_placement",
                "early",
                str,
            ),
            opt_steps=_typed_value(raw, "fast_weights_opt_steps", 4, int),
            lr=_typed_value(raw, "fast_weights_lr", 0.01, float),
            max_wrapped_layers=_typed_value(raw, "fast_weights_max_layers", 8, int),
            query_gate_enabled=_typed_value(
                raw, "fast_weights_query_gate", True, bool
            ),
            query_gate_threshold=_typed_value(
                raw, "fast_weights_query_gate_threshold", 0.8, float
            ),
            query_gate_temperature=_typed_value(
                raw, "fast_weights_query_gate_temperature", 0.05, float
            ),
            output_memory_diagnostic_enabled=_typed_value(
                raw,
                "fast_weights_output_memory_diagnostic",
                False,
                bool,
            ),
            export_candidates=_typed_value(raw, "fast_weights_export_candidates", False, bool),
            canary_enabled=_typed_value(raw, "fast_weights_canary", True, bool),
            canary_generated_enabled=_typed_value(
                raw, "fast_weights_canary_generated", True, bool
            ),
            canary_max_logprob_drop=_typed_value(raw, "fast_weights_canary_max_drop", 0.5, float),
            canary_max_effective_delta_rms=_typed_value(
                raw, "fast_weights_canary_max_delta_rms", 0.05, float
            ),
            canary_rescale_attempts=_typed_value(
                raw, "fast_weights_canary_rescale_attempts", 2, int
            ),
            canary_max_tokens=_typed_value(raw, "fast_weights_canary_max_tokens", 24, int),
        ),
        prefill_chunk_tokens=_typed_value(raw, "prefill_chunk_tokens", 128, int),
        prelude_frac=_typed_value(raw, "prelude_frac", 0.25, float),
        coda_frac=_typed_value(raw, "coda_frac", 0.25, float),
        schedule=raw.get("schedule"),
        decode_max_tokens=_typed_value(raw, "decode_max_tokens", 512, int),
        decode_contract=_typed_value(raw, "decode_contract", "none", str),
        decode_contract_grace_tokens=_typed_value(raw, "decode_contract_grace_tokens", 0, int),
        decode_min_tokens=_typed_value(raw, "decode_min_tokens", 0, int),
        verifier_probe_max_tokens=_typed_value(raw, "verifier_probe_max_tokens", 48, int),
        verifier_probe_contract=_typed_value(
            raw, "verifier_probe_contract", "none", str
        ),
        generative_verifier_enabled=_typed_value(raw, "generative_verifier_enabled", True, bool),
        generative_verifier_max_atoms=_typed_value(raw, "generative_verifier_max_atoms", 1, int),
        generative_verifier_max_tokens=_typed_value(
            raw, "generative_verifier_max_tokens", 160, int
        ),
        counterfactual_verifier_enabled=_typed_value(
            raw, "counterfactual_verifier_enabled", True, bool
        ),
        counterfactual_verifier_max_atoms=_typed_value(
            raw, "counterfactual_verifier_max_atoms", 1, int
        ),
        counterfactual_verifier_max_interventions=_typed_value(
            raw, "counterfactual_verifier_max_interventions", 2, int
        ),
        counterfactual_verifier_max_tokens=_typed_value(
            raw, "counterfactual_verifier_max_tokens", 128, int
        ),
        prefix_stability_enabled=_typed_value(raw, "prefix_stability_enabled", True, bool),
        prefix_stability_samples=_typed_value(raw, "prefix_stability_samples", 3, int),
        prefix_stability_max_tokens=_typed_value(raw, "prefix_stability_max_tokens", 128, int),
        prefix_stability_temperature=_typed_value(raw, "prefix_stability_temperature", 0.35, float),
        prefix_stability_top_p=_typed_value(raw, "prefix_stability_top_p", 0.9, float),
        prefix_stability_seed=_typed_value(raw, "prefix_stability_seed", 104_729, int),
        prefix_stability_calibrator=raw.get("prefix_stability_calibrator"),
        local_repair_enabled=_typed_value(raw, "local_repair_enabled", True, bool),
        local_repair_max_attempts=_typed_value(raw, "local_repair_max_attempts", 1, int),
        local_repair_max_tokens=_typed_value(raw, "local_repair_max_tokens", 128, int),
        answer_replacement_enabled=_typed_value(raw, "answer_replacement_enabled", True, bool),
        objective_program_enabled=_typed_value(raw, "objective_program_enabled", True, bool),
        verified_objective_teacher_enabled=_typed_value(
            raw,
            "verified_objective_teacher_enabled",
            True,
            bool,
        ),
        answer_replacement_margin=_typed_value(raw, "answer_replacement_margin", 0.05, float),
        verifier_fusion_evidence=raw.get("verifier_fusion_evidence"),
        verifier_accept_non_regression=_typed_value(
            raw, "verifier_accept_non_regression", False, bool
        ),
        decode_temperature=_typed_value(raw, "decode_temperature", 0.0, float),
        decode_top_p=_typed_value(raw, "decode_top_p", 1.0, float),
        decode_repetition_penalty=_typed_value(raw, "decode_repetition_penalty", 1.0, float),
        decode_repetition_window=_typed_value(raw, "decode_repetition_window", 72, int),
        decode_bridge_policy=_typed_value(raw, "decode_bridge_policy", "none", str),
        # The resident/live worker preserves the checkpoint's ordinary answer
        # as the output incumbent. Research callers can explicitly request the
        # historical latent decode while it is being improved and measured.
        decode_incumbent_policy=_typed_value(
            raw,
            "decode_incumbent_policy",
            "vanilla_incumbent",
            str,
        ),
        input_context_max_chars=_typed_value(raw, "input_context_max_chars", 0, int),
        allow_vanilla_fallback=_typed_value(raw, "allow_vanilla_fallback", True, bool),
        escape=raw.get("escape"),
        telemetry_enabled=_typed_value(raw, "telemetry", True, bool),
        probe_cache_enabled=_typed_value(raw, "probe_cache", True, bool),
        halting=raw.get("halting"),
        update_gate=raw.get("update_gate"),
        uncertainty_head=raw.get("uncertainty_head"),
        mistake_locator=raw.get("mistake_locator"),
        contradiction_head=raw.get("contradiction_head"),
        contradiction_perturber=raw.get("contradiction_perturber"),
        local_exploration=raw.get("local_exploration"),
        heterogeneous_integration=raw.get("heterogeneous_integration"),
        transient_negative_constraints=raw.get("transient_negative_constraints"),
        virtual_quanta=raw.get("virtual_quanta"),
        latent_tree_search=raw.get("latent_tree_search"),
        branch_correlation_evidence=raw.get("branch_correlation_evidence"),
        critic_blind_spot_evidence=raw.get("critic_blind_spot_evidence"),
    )
    problems = cfg.validate()
    if problems:
        raise ValueError(f"latent_reason config rejected: {problems}")
    return cfg


_BUDGET_COMPUTE_KEYS = frozenset({"max_layer_apps", "wall_clock_s"})

# Fields the allocator records ALONGSIDE the budget so a deeper-than-usual
# episode can be explained after the fact ("traceable to the reason it was
# deeper, not just observed to have been"). They are provenance, not compute
# limits, and they are named here rather than waved through: an unrecognised key
# is still a rejected budget, because a typo in a real limit must never be
# silently ignored.
#
# Reconciling these two intents is not cosmetic. The strict check rejected the
# allocator's own annotations, so on the live desktop path EVERY foreground turn
# recorded
#   mlx_worker (warning): ValueError: latent_reason budget contains unknown
#   keys: ['effective_uncertainty', 'effort', 'novelty', 'ontogeny_episode']
# and the Recursive Latent Cortex then declined the turn outright. The whole
# latent-reasoning lane was dark on the user surface, and the only symptom was a
# warning that read like a caller bug.
_BUDGET_ANNOTATION_KEYS = frozenset(
    {"effective_uncertainty", "novelty", "effort", "ontogeny_episode"}
)


def budget_from_job(job_budget: dict[str, Any] | None) -> ComputeBudget:
    if job_budget is not None and not isinstance(job_budget, dict):
        raise ValueError("latent_reason budget must be a mapping")
    raw = dict(job_budget or {})
    unknown = sorted(set(raw) - _BUDGET_COMPUTE_KEYS - _BUDGET_ANNOTATION_KEYS)
    if unknown:
        raise ValueError(f"latent_reason budget contains unknown keys: {unknown}")
    kwargs: dict[str, Any] = {}
    if "max_layer_apps" in raw:
        kwargs["max_layer_apps"] = _typed_value(raw, "max_layer_apps", 0, int)
    if "wall_clock_s" in raw:
        kwargs["wall_clock_s"] = _typed_value(raw, "wall_clock_s", 0.0, float)
    return ComputeBudget(**kwargs)


def handle_latent_reason(
    job: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    model_path: str,
    worker_identity: dict[str, Any] | None = None,
    worker_capture_signing_identity: Any | None = None,
    worker_capture_launch_challenge: Mapping[str, Any] | None = None,
    surface_control_state: dict[str, Any] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run one latent-reasoning episode on the resident model.

    Returns the IPC response body. Never raises for episode-level failures —
    the engine's fail-honest contract puts them in the receipt; only truly
    malformed jobs surface as status=error.
    """
    if not cortex_enabled():
        return {
            "status": "error",
            "message": "latent_cortex_disabled:AURA_LATENT_CORTEX=0",
        }
    prompt = job.get("prompt")
    messages = job.get("messages")
    if not prompt and not messages:
        return {"status": "error", "message": "latent_reason requires prompt or messages"}

    response_contract = job.get("response_contract")
    if response_contract is not None:
        if not isinstance(response_contract, str) or not response_contract.strip():
            return {
                "status": "error",
                "message": "latent_reason response_contract must be a non-empty string",
            }
        try:
            from core.brain.llm.latent_cortex.response_contracts import (
                parse_response_contract,
            )

            parse_response_contract(response_contract)
        except ValueError as exc:
            return {
                "status": "error",
                "message": f"latent_reason response_contract rejected: {exc}",
            }
        if tokenizer is None:
            return {
                "status": "error",
                "message": "latent_reason response_contract requires a tokenizer",
            }

    raw_config = job.get("config")
    if response_contract is not None:
        if raw_config is not None and not isinstance(raw_config, dict):
            return {"status": "error", "message": "latent_reason config must be a mapping"}
        raw_config = dict(raw_config or {})
        configured_contract = raw_config.get("decode_contract")
        if configured_contract not in (None, "final_answer_v1"):
            return {
                "status": "error",
                "message": (
                    "latent_reason response_contract conflicts with config.decode_contract"
                ),
            }
        raw_config["decode_contract"] = "final_answer_v1"
        raw_config.setdefault("verifier_probe_contract", "final_answer_v1")
        raw_config.setdefault(
            "decode_contract_grace_tokens",
            min(int(raw_config.get("decode_max_tokens", 512)), 512),
        )

    config = config_from_job(raw_config)
    budget = budget_from_job(job.get("budget"))
    cognitive_context = job.get("cognitive_context")
    try:
        from core.brain.llm.latent_cortex.cognitive_context import (
            normalize_cognitive_context,
        )

        cognitive_context = normalize_cognitive_context(cognitive_context) or None
    except (TypeError, ValueError) as exc:
        return {
            "status": "error",
            "message": f"latent_reason cognitive_context is invalid: {exc}",
        }
    operation_authority = job.get("operation_authority")
    action_policy_evidence = job.get("action_policy_evidence")
    action_intervention = job.get("action_intervention")
    action_state_runtime_wire = job.get("action_state_runtime")
    external_execution_offer = job.get("external_execution_offer")
    if external_execution_offer is not None:
        if operation_authority is None or action_policy_evidence is None:
            return {
                "status": "error",
                "message": (
                    "latent_reason external execution requires operation "
                    "authority and action-policy evidence"
                ),
            }
        try:
            from core.brain.llm.latent_cortex.external_execution import (
                validate_external_execution_offer,
            )

            external_execution_offer = validate_external_execution_offer(external_execution_offer)
        except (ImportError, TypeError, ValueError) as exc:
            return {
                "status": "error",
                "message": f"latent_reason external execution offer rejected: {exc}",
            }
    if (
        external_execution_offer is None
        and action_policy_evidence is None
        and isinstance(operation_authority, dict)
    ):
        try:
            from core.brain.llm.latent_cortex.value_of_computation import (
                build_evidence_snapshot,
            )

            action_policy_evidence = build_evidence_snapshot(
                bucket=str(operation_authority.get("controller_bucket") or ""),
                cells={},
            )
        except (ImportError, TypeError, ValueError) as exc:
            return {
                "status": "error",
                "message": f"latent_reason action policy rejected: {exc}",
            }
    if action_policy_evidence is not None:
        try:
            from core.brain.llm.latent_cortex.value_of_computation import (
                validate_evidence_snapshot,
            )

            action_policy_evidence = validate_evidence_snapshot(action_policy_evidence)
        except (ImportError, TypeError, ValueError) as exc:
            return {
                "status": "error",
                "message": f"latent_reason action policy rejected: {exc}",
            }
    action_intervention_consumption = None
    if action_intervention is not None:
        if action_policy_evidence is None:
            return {
                "status": "error",
                "message": ("latent_reason action intervention requires action-policy evidence"),
            }
        try:
            from core.brain.llm.latent_cortex.action_intervention import (
                validate_action_intervention,
            )

            action_intervention = validate_action_intervention(
                action_intervention,
                require_current_policy=True,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "status": "error",
                "message": f"latent_reason action intervention rejected: {exc}",
            }
    if operation_authority is not None:
        try:
            from core.brain.llm.latent_cortex.epistemic_runtime import (
                validate_runtime_operation_authority,
            )

            operation_authority = validate_runtime_operation_authority(
                operation_authority,
                prompt=prompt if isinstance(prompt, str) else None,
                messages=messages if isinstance(messages, list) else None,
                config=dict(job.get("config") or {}),
                budget=dict(job.get("budget") or {}),
                cognitive_context=cognitive_context,
                action_policy_evidence=action_policy_evidence,
                external_execution_offer=external_execution_offer,
            )
        except (ImportError, TypeError, ValueError) as exc:
            return {
                "status": "error",
                "message": f"latent_reason operation authority rejected: {exc}",
            }
    if action_state_runtime_wire is not None:
        try:
            from core.brain.llm.latent_cortex.runtime_identity import (
                latent_request_payload_sha256,
            )

            actual_capture_request_sha256 = latent_request_payload_sha256(
                prompt=prompt,
                messages=messages,
                domain=str(job.get("domain", "general")),
                config=job.get("config"),
                budget=job.get("budget"),
                runtime_controls=job.get("runtime_controls"),
                cognitive_context=cognitive_context,
                operation_authority=operation_authority,
                action_policy_evidence=action_policy_evidence,
                external_execution_offer=external_execution_offer,
                response_contract=response_contract,
                verifier_guidance=(True if job.get("verifier_guidance") else None),
                facet_reliability=job.get("facet_reliability"),
            )
            capture_payload = action_state_runtime_wire.get("capture_request", {}).get(
                "request_payload", {}
            )
            if (
                actual_capture_request_sha256
                != capture_payload.get("latent_reason_request_sha256")
            ):
                raise ValueError("action-state latent request differs")
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            return {
                "status": "error",
                "message": f"latent_reason action-state request rejected: {exc}",
            }
    if worker_identity is None:
        from core.brain.llm.latent_cortex.runtime_identity import build_worker_identity
        from core.brain.llm.latent_cortex.worker_capture_identity import (
            build_worker_capture_identity,
        )

        if worker_capture_signing_identity is None:
            worker_capture_signing_identity = build_worker_capture_identity(
                worker_boot_id=uuid.uuid4().hex,
            )
        worker_identity = build_worker_identity(
            model,
            model_path=model_path,
            worker_boot_id=worker_capture_signing_identity.public_identity[
                "worker_boot_id"
            ],
            worker_source_path=Path(__file__).resolve().parents[1] / "mlx_worker.py",
            worker_action_capture_identity=(
                worker_capture_signing_identity.public_identity
            ),
            tokenizer=tokenizer,
        )
    action_state_runtime = None
    action_state_store = None
    action_state_custodian = None
    action_state_runtime_identity: dict[str, Any] | None = None
    if action_state_runtime_wire is not None:
        if worker_capture_launch_challenge is None:
            return {
                "status": "error",
                "message": "latent_reason action-state runtime lacks launch challenge",
            }
        if worker_capture_signing_identity is None:
            return {
                "status": "error",
                "message": "latent_reason action-state runtime lacks worker signer",
            }
        try:
            from core.brain.llm.latent_cortex.action_state_runtime import (
                admit_action_state_runtime,
                resident_model_identity_for_worker,
            )
            from core.brain.llm.latent_cortex.runtime_identity import (
                collect_latent_runtime_identity,
            )

            action_state_runtime = admit_action_state_runtime(
                action_state_runtime_wire,
                worker_launch_challenge=worker_capture_launch_challenge,
                now_unix=int(time.time()),
            )
            actual_model_identity = resident_model_identity_for_worker(
                worker_identity
            )
            if actual_model_identity != action_state_runtime.model_identity:
                raise ValueError(
                    "action-state model identity differs from loaded resident"
                )
            if (
                action_state_runtime.resident_worker_origin_binding.get(
                    "worker_identity"
                )
                != worker_capture_signing_identity.public_identity
            ):
                raise ValueError(
                    "action-state resident origin differs from this worker"
                )
            action_state_runtime_identity = collect_latent_runtime_identity(
                Path(__file__).resolve().parents[4]
            )
            if action_state_runtime_identity.get("identity_bound") is not True:
                raise ValueError("action-state runtime identity is unbound")
            if action_state_runtime.mode == "capture" and action_intervention is not None:
                raise ValueError("capture must precede action intervention")
            if action_state_runtime.mode == "restore":
                if action_intervention is None:
                    raise ValueError("restore requires an action intervention")
                authority = action_intervention.get("authority_payload", {})
                if authority.get("arm") != action_state_runtime.arm:
                    raise ValueError("restore arm differs from intervention")
                capture_receipt = action_state_runtime.capture_receipt or {}
                if (
                    authority.get("starting_state_components")
                    != capture_receipt.get("state_components")
                    or authority.get("expected_pre_state_sha256")
                    != capture_receipt.get("state_sha256")
                    or authority.get("expected_pre_kv_sha256")
                    != capture_receipt.get("state_components", {}).get(
                        "kv_cache_sha256"
                    )
                ):
                    raise ValueError(
                        "restore capture receipt differs from intervention prestate"
                    )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "status": "error",
                "message": f"latent_reason action-state runtime rejected: {exc}",
            }
    engine = LatentCortexEngine(
        model,
        tokenizer,
        config,
        model_path=model_path,
        schedule_library=_library(),
    )
    episode_messages = messages if isinstance(messages, list) else None
    context_compaction: dict[str, Any] = {}
    if episode_messages is not None and config.input_context_max_chars:
        from core.brain.llm.latent_cortex.context_compaction import (
            compact_latent_messages,
        )

        episode_messages, context_compaction = compact_latent_messages(
            episode_messages,
            max_chars=config.input_context_max_chars,
        )
    # Verifier guidance: when the caller asks for it, candidate branches and
    # latent-opt proposals are scored by deterministic task-typed checks
    # (arithmetic recomputation, code syntax, facet coverage, grounding) —
    # the winner is picked because its answer CHECKS OUT, not because its
    # trajectory converged prettier. Tokenizer required: verification reads
    # decoded probe text.
    task_verifier = None
    critic_identity: dict[str, Any] = {}
    shared_blind_spots: dict[str, Any] = {}
    verifier_unavailable_reason = ""
    verifier_requested = bool(job.get("verifier_guidance")) or response_contract is not None
    if verifier_requested and tokenizer is None:
        # A REQUESTED verifier that cannot be built must not vanish. Without
        # a tokenizer the guidance was skipped silently, so the episode ran
        # with no task verifier while the caller — which had asked for one,
        # or supplied a response contract that implies one — received a
        # receipt that simply omitted it. Absence of guidance then read as
        # "no guidance was wanted" rather than "guidance was lost".
        from core.runtime.errors import record_degradation

        record_degradation(
            "latent_cortex_worker_handler",
            RuntimeError("verifier_guidance_requested_without_tokenizer"),
            severity="error",
            action="ran latent episode without the requested task verifier because no tokenizer was available",
        )
        logger.error(
            "Latent episode requested verifier guidance but no tokenizer is "
            "available; the episode runs UNVERIFIED and the receipt records it."
        )
        verifier_unavailable_reason = "tokenizer_unavailable"
    if verifier_requested and tokenizer is not None:
        from core.brain.llm.latent_cortex.task_verifiers import (
            _ANSWER_FACET_HINTS,
            EpisodeTaskVerifier,
        )

        facet_reliability = job.get("facet_reliability")
        if facet_reliability is not None:
            if (
                not isinstance(facet_reliability, dict)
                or len(facet_reliability) > 8
                or any(
                    not isinstance(name, str)
                    or name not in _ANSWER_FACET_HINTS
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= float(value) <= 1.0
                    for name, value in facet_reliability.items()
                )
            ):
                return {
                    "status": "error",
                    "message": (
                        "latent_reason facet_reliability must map known facet "
                        "names to floats in [0, 1]"
                    ),
                }
        objective = prompt if isinstance(prompt, str) else ""
        if not objective and isinstance(episode_messages, list):
            for message in reversed(episode_messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        objective = content
                        break
        candidate_verifier = EpisodeTaskVerifier(
            objective,
            facet_reliability=facet_reliability,
            response_contract=str(response_contract or ""),
        )
        try:
            from core.brain.llm.latent_cortex.critic_identity import (
                build_critic_identity,
                build_shared_blind_spot_evidence,
                validate_critic_identity,
                validate_shared_blind_spot_evidence,
            )

            critic_identity = build_critic_identity(
                candidate_verifier,
                worker_identity=worker_identity,
            )
            critic_identity = validate_critic_identity(
                critic_identity,
                worker_identity=worker_identity,
            )
            generator_sha = critic_identity["generator_identity"]["function_sha256"]
            critic_sha = critic_identity["critic_function_sha256"]
            evidence = config.critic_blind_spot_evidence
            if evidence is None:
                evidence = build_shared_blind_spot_evidence(
                    bucket=f"{str(job.get('domain', 'general'))[:120]}|runtime",
                    generator_function_sha256=generator_sha,
                    critic_function_sha256=critic_sha,
                    checked_outcomes=[],
                )
            shared_blind_spots = validate_shared_blind_spot_evidence(
                evidence,
                generator_function_sha256=generator_sha,
                critic_function_sha256=critic_sha,
            )
            if shared_blind_spots["critic_reliability_admitted"] is not True:
                raise ValueError("shared_blind_spot_upper_bound_exceeded")
            task_verifier = candidate_verifier
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            verifier_unavailable_reason = f"critic_identity_or_reliability_unproven:{exc}"
            from core.runtime.errors import record_degradation

            record_degradation(
                "latent_cortex.critic",
                exc,
                severity="error",
                action=(
                    "ran the latent episode without critic authority because "
                    "function separation or shared-blind-spot evidence failed"
                ),
            )
            logger.error("Latent critic authority rejected: %s", exc)
    if action_intervention is not None:
        try:
            from core.brain.llm.latent_cortex.action_intervention import (
                consume_action_intervention_once,
            )
            from core.brain.llm.latent_cortex.runtime_identity import (
                latent_request_payload_sha256,
            )

            intervention_request_sha256 = latent_request_payload_sha256(
                prompt=prompt,
                messages=messages,
                domain=str(job.get("domain", "general")),
                config=job.get("config"),
                budget=job.get("budget"),
                runtime_controls=job.get("runtime_controls"),
                cognitive_context=cognitive_context,
                operation_authority=operation_authority,
                action_policy_evidence=action_policy_evidence,
                external_execution_offer=external_execution_offer,
                response_contract=response_contract,
                verifier_guidance=(True if job.get("verifier_guidance") else None),
                facet_reliability=job.get("facet_reliability"),
            )
            if (
                intervention_request_sha256
                != action_intervention["authority_payload"]["request_payload_sha256"]
            ):
                raise ValueError("action intervention request payload differs")
            action_intervention_consumption = consume_action_intervention_once(
                action_intervention
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "status": "error",
                "message": (f"latent_reason action intervention replay admission rejected: {exc}"),
            }
    # Recurrent recall reads the shared datastore. Without a principal the
    # store searches every entry, so this episode inherits the principal the
    # job names and gets nothing when the job names nobody.
    from core.brain.nonparametric_binding import binding_for_job

    recall_binding = binding_for_job(job, source_id="latent_recurrent_recall")

    def reason_with_continuation(**continuation_kwargs: Any) -> Any:
        return engine.reason(
            prompt=prompt if isinstance(prompt, str) else None,
            memory_principal=(recall_binding.principal if recall_binding else ""),
            messages=episode_messages,
            budget=budget,
            domain=str(job.get("domain", "general")),
            verifier=task_verifier,
            cognitive_context=cognitive_context,
            action_policy_evidence=action_policy_evidence,
            action_intervention=action_intervention,
            action_intervention_consumption=action_intervention_consumption,
            external_execution_offer=external_execution_offer,
            cancel_check=cancel_check,
            progress=progress,
            **continuation_kwargs,
        )

    public_action_state_receipt: dict[str, Any] | None = None
    action_state_restore_receipt: dict[str, Any] | None = None
    try:
        if action_state_runtime is None:
            result = reason_with_continuation()
        else:
            from core.brain.llm.latent_cortex.action_state_capture import (
                ActionStateCaptureError,
                build_action_state_capture_receipt,
                validate_action_state_capture_receipt,
            )
            from core.brain.llm.latent_cortex.action_state_runtime import (
                build_action_state_restore_receipt,
                continuation_from_private_state,
                open_action_state_store,
            )

            action_state_store, action_state_custodian = open_action_state_store()
            if action_state_runtime.mode == "capture":
                captured: dict[str, Any] = {}

                def publish_continuation(continuation: Any) -> None:
                    publication = action_state_store.publish(
                        action_state_runtime.admission,
                        continuation.private_state,
                        created_at_unix=int(time.time()),
                    )
                    captured["continuation"] = continuation
                    captured["publication"] = publication

                result = reason_with_continuation(
                    action_continuation_capture=publish_continuation,
                    action_continuation_runner_state=action_state_runtime.runner_state,
                    action_continuation_capture_only=True,
                )
                continuation = captured.get("continuation")
                publication = captured.get("publication")
                if continuation is None or publication is None or not result.ok:
                    raise ValueError("action-state continuation capture did not complete")
                public_action_state_receipt = build_action_state_capture_receipt(
                    admission=action_state_runtime.admission,
                    publication=publication,
                    worker_private_key=(worker_capture_signing_identity.private_key),
                    captured_at_unix=int(time.time()),
                    latent_reason_request=action_state_runtime.latent_reason_request,
                    model_identity=action_state_runtime.model_identity,
                    execution_identity=action_state_runtime.execution_identity,
                    runtime_identity=action_state_runtime_identity,
                    episode_step=continuation.episode_step,
                    schedule_step=continuation.schedule_step,
                    branch_id=continuation.branch_id,
                    layer_index=continuation.layer_index,
                    kv_position=continuation.kv_position,
                )
            else:
                publication = action_state_store.publication_for_request(
                    action_state_runtime.admission
                )
                public_action_state_receipt = validate_action_state_capture_receipt(
                    action_state_runtime.capture_receipt,
                    request=action_state_runtime.admission.request,
                    publication=publication,
                    trusted_root_public_key_pem=(
                        action_state_runtime.trusted_root_public_key_pem
                    ),
                    expected_supervisor_public_key=(
                        action_state_runtime.capture_supervisor_public_key
                    ),
                    latent_reason_request=action_state_runtime.latent_reason_request,
                    model_identity=action_state_runtime.model_identity,
                    execution_identity=action_state_runtime.execution_identity,
                    runtime_identity=action_state_runtime_identity,
                    expected_campaign_design_sha256=(
                        action_state_runtime.admission.payload[
                            "campaign_design_sha256"
                        ]
                    ),
                )
                restored_result: dict[str, Any] = {}

                def install_and_run(private_state: dict[str, Any]) -> str:
                    continuation = continuation_from_private_state(
                        private_state,
                        public_action_state_receipt,
                    )
                    verified: dict[str, str] = {}
                    episode = reason_with_continuation(
                        action_continuation_restore=continuation,
                        action_continuation_runner_state=(
                            action_state_runtime.runner_state
                        ),
                        action_continuation_restore_verified=(
                            lambda state_sha256: verified.setdefault(
                                "state_sha256", state_sha256
                            )
                        ),
                    )
                    if not episode.ok:
                        raise RuntimeError(
                            f"restored action episode failed:{episode.reason}"
                        )
                    expected = public_action_state_receipt["state_sha256"]
                    if verified.get("state_sha256") != expected:
                        raise RuntimeError(
                            "restored action state was not verified before action"
                        )
                    restored_result["result"] = episode
                    return expected

                restore = action_state_store.restore_and_apply(
                    publication.handle,
                    action_state_runtime.admission,
                    arm=str(action_state_runtime.arm),
                    restored_at_unix=int(time.time()),
                    apply_state=install_and_run,
                    application_worker_identity=(
                        action_state_runtime.resident_worker_origin_binding[
                            "worker_identity"
                        ]
                    ),
                )
                result = restored_result.get("result")
                if result is None:
                    raise RuntimeError("restored action episode result missing")
                lifecycle_receipts: dict[str, Any] = {"complete": False}
                try:
                    seal_receipt = action_state_store.seal(
                        publication.handle,
                        action_state_runtime.admission,
                        sealed_at_unix=int(time.time()),
                    )
                except ActionStateCaptureError as exc:
                    if exc.code != "private_snapshot_pair_incomplete":
                        raise
                else:
                    erasure_receipt = action_state_store.erase(
                        publication.handle,
                        action_state_runtime.admission,
                        erased_at_unix=int(time.time()),
                    )
                    lifecycle_receipts = {
                        "complete": True,
                        "seal_receipt": seal_receipt,
                        "erasure_receipt": erasure_receipt,
                    }
                action_state_restore_receipt = build_action_state_restore_receipt(
                    capture_receipt=public_action_state_receipt,
                    custody_restore_receipt=restore.receipt,
                    action_intervention=action_intervention,
                    runtime_identity=action_state_runtime_identity,
                    worker_private_key=(worker_capture_signing_identity.private_key),
                    custody_lifecycle_receipts=lifecycle_receipts,
                    resident_worker_origin_binding=(
                        action_state_runtime.resident_worker_origin_binding
                    ),
                )
    except LatentEngineBusyError as exc:
        # Single-flight refusal, not a broken episode. Nothing touched the
        # model, so the worker stays usable and the caller can come back.
        return {
            "status": "error",
            "message": f"latent_reason refused: {exc}",
            "retryable": True,
            "requires_worker_recycle": False,
            "requires_cache_clear": False,
        }
    finally:
        if action_state_store is not None:
            action_state_store.close()
        if action_state_custodian is not None:
            action_state_custodian.close()
    if task_verifier is not None:
        excluded = set(
            result.receipt.verifier_preflight.get(
                "control_evaluation_indices",
                [],
            )
        )
        excluded.update(
            result.receipt.decoy_verification.get(
                "control_evaluation_indices",
                [],
            )
        )
        result.receipt.verifier_guidance = task_verifier.to_receipt(
            exclude_evaluation_indices=excluded,
        )
    elif verifier_requested:
        # Legible in the receipt: downstream must be able to tell "no verifier
        # was wanted" from "a verifier was wanted and could not be built".
        if tokenizer is None:
            result.receipt.verifier_guidance = {
                "requested": True,
                "available": False,
                "reason": "tokenizer_unavailable",
            }
        else:
            result.receipt.verifier_guidance = {
                "requested": True,
                "available": False,
                "reason": verifier_unavailable_reason or "critic_unavailable",
            }
    receipt = result.receipt
    receipt.critic_identity = dict(critic_identity)
    receipt.shared_blind_spots = dict(shared_blind_spots)
    if verifier_requested and task_verifier is None:
        receipt.flag("critic_authority_unproven")
    receipt.runtime_operation_authority = dict(operation_authority or {})
    receipt.worker_boot_id = str(worker_identity.get("worker_boot_id") or "")
    receipt.worker_pid = int(worker_identity.get("worker_pid") or 0)
    receipt.worker_model_path = str(worker_identity.get("worker_model_path") or "")
    receipt.worker_model_parameter_count = int(
        worker_identity.get("worker_model_parameter_count") or 0
    )
    receipt.worker_model_stored_parameter_element_count = int(
        worker_identity.get("worker_model_stored_parameter_element_count") or 0
    )
    receipt.worker_model_parameter_count_basis = str(
        worker_identity.get("worker_model_parameter_count_basis") or ""
    )
    receipt.worker_source_sha256 = str(worker_identity.get("worker_source_sha256") or "")
    receipt.worker_identity = dict(worker_identity)
    receipt.worker_affective_steering_active = bool(
        worker_identity.get("worker_affective_steering_active", False)
    )
    receipt.worker_affective_steering_alpha = float(
        worker_identity.get("worker_affective_steering_alpha") or 0.0
    )
    receipt.input_context_compaction = dict(context_compaction)
    control_state = dict(surface_control_state or {})
    applied_alpha = control_state.get("surface_alpha_applied")
    receipt.episode_affective_steering_applied = bool(
        receipt.worker_affective_steering_active
        and isinstance(applied_alpha, (int, float))
        and not isinstance(applied_alpha, bool)
    )
    receipt.episode_affective_steering_alpha = (
        float(applied_alpha) if receipt.episode_affective_steering_applied else 0.0
    )
    from core.brain.llm.latent_cortex.runtime_identity import (
        latent_request_payload_sha256,
    )

    receipt.request_payload_sha256 = latent_request_payload_sha256(
        prompt=prompt,
        messages=messages,
        domain=str(job.get("domain", "general")),
        config=job.get("config"),
        budget=job.get("budget"),
        runtime_controls=job.get("runtime_controls"),
        cognitive_context=cognitive_context,
        operation_authority=operation_authority,
        action_policy_evidence=action_policy_evidence,
        action_intervention=action_intervention,
        external_execution_offer=external_execution_offer,
        response_contract=response_contract,
        verifier_guidance=True if job.get("verifier_guidance") else None,
        facet_reliability=job.get("facet_reliability"),
    )
    # The engine measures model state but cannot identify the worker process.
    # Bind those measurements to this exact boot and serving stack before any
    # caller is allowed to keep the resident worker alive.
    from core.brain.llm.latent_cortex.runtime_integrity import (
        bind_worker_runtime_integrity,
        runtime_integrity_safe,
    )

    try:
        receipt.runtime_integrity = bind_worker_runtime_integrity(
            receipt.runtime_integrity,
            worker_identity=worker_identity,
        )
    except (ImportError, TypeError, ValueError) as exc:
        receipt.flag(f"runtime_integrity_binding_failed:{type(exc).__name__}")
        result.ok = False
        result.text = ""
        result.tokens = []
        result.reason = f"runtime integrity could not be proven: {exc}"

    integrity_safe = runtime_integrity_safe(
        receipt.runtime_integrity,
        require_worker=True,
        expected_episode_id=receipt.episode_id,
        expected_input_tokens_sha256=receipt.input_tokens_sha256,
        expected_worker_identity=worker_identity,
        expected_fast_weights_applied=receipt.fast_weights_applied,
        expected_fast_weights_attach_attempted=(
            receipt.fast_weights_attach_attempted
        ),
        expected_checkpoint_fingerprint=receipt.checkpoint_fingerprint,
        expected_checkpoint_method=receipt.checkpoint_fingerprint_method,
        expected_checkpoint_file_count=receipt.checkpoint_file_count,
    )
    if not integrity_safe:
        receipt.flag("runtime_integrity_unproven")
        result.ok = False
        result.text = ""
        result.tokens = []
        result.reason = result.reason or "runtime integrity is unproven"

    if (
        integrity_safe
        and job.get("foreground_request") is True
        and result.tokens
        and "native_thinking_prefix_open" in receipt.honest_flags
    ):
        from core.brain.llm.chat_format import split_native_thinking_generation
        from core.brain.llm.thinking_reserve import (
            record_budget_that_ran_out_thinking,
            record_reasoning_cost,
        )

        try:
            channels = split_native_thinking_generation(
                tokenizer.decode(result.tokens), native_thinking=True,
            )
            if channels.boundary_closed:
                record_reasoning_cost(
                    reasoning_chars=len(channels.reasoning),
                    surface_chars=len(channels.surface),
                    generated_tokens=len(result.tokens),
                    model=model_path,
                )
            else:
                # This is a censored observation: all observed tokens were
                # private, regardless of which resource ended generation.
                record_budget_that_ran_out_thinking(
                    budget_tokens=len(result.tokens), model=model_path,
                )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            receipt.flag("native_thinking_cost_observation_failed")
            logger.warning("Could not retain latent reasoning cost: %s", exc)

    # The client performs the final causal-envelope reconstruction after it
    # captures runtime/app provenance, but the worker boundary itself must
    # already be complete and independently reconstructable.
    from core.brain.llm.latent_cortex.causal_receipt import build_causal_receipt

    receipt.causal_receipt = build_causal_receipt(receipt.to_dict())
    body = result.to_dict()
    body["status"] = "ok" if result.ok else "error"
    if public_action_state_receipt is not None:
        body["action_state_capture_receipt"] = dict(
            public_action_state_receipt
        )
        body["action_state_runtime_mode"] = str(action_state_runtime.mode)
    if action_state_restore_receipt is not None:
        body["action_state_restore_receipt"] = dict(
            action_state_restore_receipt
        )
    if not result.ok:
        body["message"] = result.reason
    # Compatibility booleans remain telemetry only. The measured, worker-bound
    # proof is the sole authority for cache retention and process reuse.
    body["requires_cache_clear"] = bool(
        (
            result.receipt.fast_weights_applied
            or result.receipt.fast_weights_attach_attempted
        )
        and not integrity_safe
    )
    # Memory exhaustion is not a clean failure even when the erase proof
    # holds: the process just could not get what it asked for, and the next
    # episode inherits whatever fragmentation caused it.
    memory_exhausted = "fallback_refused_memory_exhaustion" in set(
        result.receipt.honest_flags
    )
    body["requires_worker_recycle"] = not integrity_safe or memory_exhausted
    if not integrity_safe:
        logger.warning(
            "Latent episode returned no complete worker-bound runtime-integrity "
            "proof; recycling rather than trusting unverified resident state."
        )
    if action_state_runtime is not None:
        from core.brain.llm.latent_cortex.action_state_runtime import (
            assert_public_runtime_result,
        )

        assert_public_runtime_result(body)
    return body


__all__ = [
    "budget_from_job",
    "config_from_job",
    "cortex_enabled",
    "handle_latent_reason",
]
