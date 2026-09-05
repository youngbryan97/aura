"""Contract tests: latent-cortex runtime wiring (service, handler, economy).

No worker processes are spawned here — the worker/client IPC bodies are
exercised through the handler function and a mocked client, which is exactly
the seam the live path uses.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from core.brain.latent_cortex_service import (
    LatentCortexService,
    _foreground_surplus_plan,
    _operation_authority_rejected,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    ResourceLedger,
    build_information_receipt,
)
from core.brain.llm.latent_cortex.runtime_identity import (
    latent_request_payload_sha256,
)
from core.brain.llm.latent_cortex.worker_handler import (
    budget_from_job,
    config_from_job,
    cortex_enabled,
    handle_latent_reason,
)
from core.brain.llm.mlx_client import MLXLocalClient
from tests.fixtures.rlc_runtime_integrity import (
    attach_bound_runtime_integrity,
    complete_worker_identity,
    engine_runtime_integrity,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _ResidentProcess:
    def __init__(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def kill(self) -> None:
        self.alive = False

    def join(self, timeout=None) -> None:
        self.alive = False


_WORKER_IDENTITY = complete_worker_identity()

_RUNTIME_IDENTITY = {
    "schema": "aura.latent_cortex.runtime_identity.v1",
    "identity_bound": True,
    "launch_mode": "direct",
    "installed_app_required": False,
    "installed_app_verified": False,
    "source_verified": True,
    "source_commit": "3" * 40,
    "workspace_state_sha256": "4" * 64,
    "shell_assets_sha256": "5" * 64,
    "issues": [],
}


def test_foreground_surplus_plan_preserves_measured_answer_budget(monkeypatch):
    monkeypatch.setattr(
        "core.brain.llm.model_registry.get_active_cortex_spec",
        lambda: SimpleNamespace(model_path="/models/Aura-Qwen3.8-27B-4bit"),
    )
    monkeypatch.setattr(
        "core.brain.memory_guard.estimate_tokens",
        lambda _messages: 3000,
    )
    monkeypatch.setattr(
        "core.runtime.structured_input.answer_surface_planning_tokens",
        lambda _objective: 1024,
    )
    from core.brain.llm.measured_admission import Confidence

    monkeypatch.setattr(
        "core.brain.llm.measured_admission.recommended_completion_tokens",
        lambda **_kwargs: (900, Confidence.MEASURED, 25),
    )
    monkeypatch.setattr(
        "core.brain.llm.measured_admission.recommended_foreground_deadline",
        lambda **_kwargs: (130.0, Confidence.MEASURED, 25),
    )

    plan = _foreground_surplus_plan(
        messages=[{"role": "user", "content": "compound request"}],
        visible_objective="compound request",
        decode_max_tokens=1536,
        request_timeout_s=180.0,
        last_latency_s=140.0,
    )

    assert plan["model"] == "Aura-Qwen3.8-27B-4bit"
    assert plan["decode_capacity_tokens"] == 1536
    assert plan["planned_decode_tokens"] == 900
    assert plan["canonical_answer_reserve_s"] == 130.0
    assert plan["latent_surplus_s"] == 50.0
    assert plan["latent_runtime_floor_s"] == 140.0
    assert plan["admitted"] is False
    assert plan["deadline_confidence"] == "measured"


@pytest.mark.parametrize(
    ("worker_ok", "receipt", "rejected"),
    [
        (False, {}, False),
        (False, {"runtime_operation_authority": {"id": "expected"}}, False),
        (False, {"runtime_operation_authority": {"id": "other"}}, True),
        (True, {}, True),
        (True, {"runtime_operation_authority": {"id": "expected"}}, False),
    ],
)
def test_operation_authority_distinguishes_worker_failure_from_conflict(
    worker_ok,
    receipt,
    rejected,
):
    assert (
        _operation_authority_rejected(
            worker_ok=worker_ok,
            worker_receipt=receipt,
            expected_authority={"id": "expected"},
        )
        is rejected
    )


def _identity_receipt(**overrides):
    integrity_digest = "e" * 64
    receipt = {
        **_WORKER_IDENTITY,
        "episode_id": "episode-runtime-integrity",
        "request_payload_sha256": "6" * 64,
        "input_tokens_sha256": "7" * 64,
        "input_token_count": 64,
        "checkpoint_fingerprint": "f" * 64,
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 8,
        "params_unchanged": True,
        "fast_weights_applied": False,
        # A real receipt is a dataclass and always carries this field, so a
        # stub that omits it is simulating a client that never reported
        # whether it TRIED to attach.
        "fast_weights_attach_attempted": False,
        "fast_weights_erased": None,
        "worker_identity": dict(_WORKER_IDENTITY),
        "episode_affective_steering_applied": True,
        "episode_affective_steering_alpha": 0.30,
        "runtime_identity": dict(_RUNTIME_IDENTITY),
        "weight_integrity": {
            "algorithm": "sha256",
            "version": 1,
            "params_before": integrity_digest,
            "params_after": integrity_digest,
            "canary_before": integrity_digest,
            "canary_after": integrity_digest,
            "erased_layer_ids": ["layer-0"],
            "unavailable_reason": "",
            "params_unchanged_proven": True,
            "fast_weights_erased_proven": True,
        },
        "integrity_verdicts": {
            "params_unchanged": {"verdict": "proven", "asserted": True},
            "fast_weights_erased": {"verdict": "proven", "asserted": True},
            "algorithm": "sha256",
            "version": 1,
            "unavailable_reason": "",
            "contradictions": [],
        },
    }
    receipt.update(overrides)
    worker_identity = receipt.get("worker_identity")
    if not isinstance(worker_identity, dict):
        worker_identity = dict(_WORKER_IDENTITY)
    return attach_bound_runtime_integrity(
        receipt,
        worker_identity=worker_identity,
    )


def _identity_receipt_for_request(request, **overrides):
    values = {
        "request_payload_sha256": latent_request_payload_sha256(
            prompt=request.get("prompt"),
            messages=request.get("messages"),
            domain=request.get("domain", "general"),
            config=request.get("config"),
            budget=request.get("budget"),
            runtime_controls=request.get("runtime_controls"),
            cognitive_context=request.get("cognitive_context"),
            operation_authority=request.get("operation_authority"),
            action_policy_evidence=request.get("action_policy_evidence"),
            action_intervention=request.get("action_intervention"),
            external_execution_offer=request.get("external_execution_offer"),
            response_contract=request.get("response_contract"),
            verifier_guidance=(True if request.get("verifier_guidance") else None),
            facet_reliability=request.get("facet_reliability"),
        )
    }
    values.update(overrides)
    return _identity_receipt(**values)


def _measured_episode_receipt(episode_id: str):
    from core.brain.llm.latent_cortex.types import EpisodeReceipt

    input_tokens_sha256 = _digest(f"{episode_id}:input")
    receipt = EpisodeReceipt(
        episode_id=episode_id,
        input_tokens_sha256=input_tokens_sha256,
        checkpoint_fingerprint="f" * 64,
        checkpoint_fingerprint_method="sha256",
        checkpoint_file_count=8,
        params_unchanged=True,
        fast_weights_applied=False,
    )
    receipt.runtime_integrity = engine_runtime_integrity(
        episode_id=episode_id,
        input_tokens_sha256=input_tokens_sha256,
    )
    return receipt


def _accounting_fields(
    *,
    input_tokens_sha256: str = "7" * 64,
    input_token_count: int = 64,
) -> dict:
    profile = ModelComputeProfile(
        model_type="wiring-fixture",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        head_dim=4,
    )
    resource = ResourceLedger(profile).to_receipt()
    information = build_information_receipt(
        sources=[
            {
                "source_id": "rendered_model_input",
                "kind": "model_input_tokens",
                "content_sha256": input_tokens_sha256,
                "byte_count": input_token_count * 2,
                "token_count": input_token_count,
            }
        ],
        policies={
            "tokenizer": "8" * 64,
            "tools": "9" * 64,
            "verifier": "a" * 64,
        },
    )
    return {
        "resource_accounting": resource,
        "information_accounting": information,
    }


def _latent_tree_fields(config, *, episode_id: str) -> dict:
    from core.brain.llm.latent_cortex.latent_tree_search import (
        LatentTreeSearchConfig,
        build_empty_latent_tree_receipt,
    )
    from core.brain.llm.latent_cortex.worker_handler import config_from_job

    executed = config_from_job(config)
    tree_config = LatentTreeSearchConfig.from_value(executed.latent_tree_search)
    return {
        "latent_tree_search": build_empty_latent_tree_receipt(
            episode_id=episode_id,
            objective_sha256="7" * 64,
            config=tree_config,
        )
    }


def _verifier_fusion_fields(config: dict, *, selected_branch: int = 0) -> dict:
    from core.brain.llm.latent_cortex.verifier_fusion import (
        build_verifier_fusion_receipt,
    )

    unavailable = {
        "requested": True,
        "available": False,
        "reason": "stubbed_worker_has_no_generator",
        "selection_effect": "none",
    }
    prefix_unavailable = {
        **unavailable,
        "correctness_effect": "none",
    }
    return {
        "verifier_fusion": build_verifier_fusion_receipt(
            blind_review=None,
            decoy_verification=None,
            generative_verifier=unavailable,
            counterfactual_verifier=unavailable,
            prefix_stability=prefix_unavailable,
            neural_uncertainty=None,
            mistake_locator=None,
            selected_branch=selected_branch,
            evidence=config.get("verifier_fusion_evidence"),
        )
    }


def _terminal_disposition_fields(
    receipt: dict,
    *,
    text: str,
    tokens: list[int],
) -> dict:
    from core.brain.llm.latent_cortex.terminal_disposition import (
        classify_terminal_disposition,
        finalize_terminal_disposition_receipt,
    )

    latent_output_authority = receipt.get("decode_incumbent_policy") == "latent"
    instruction_tokens = [101, 102, 103] if latent_output_authority else []
    bridge_tokens = [99, *instruction_tokens] if latent_output_authority else []
    decision = classify_terminal_disposition(
        halting_reason=receipt["halting_reason"],
        halting=receipt["halting"],
        loop_stability=receipt["loop_stability"],
        cognitive_action_trace=receipt["cognitive_action_trace"],
        budget=receipt["budget"],
    )
    bridge_raw = json.dumps(bridge_tokens, separators=(",", ":")).encode("ascii")
    return {
        "decode_bridge_applied": latent_output_authority,
        "decode_bridge_token_count": len(bridge_tokens),
        "decode_bridge_tokens_sha256": (
            hashlib.sha256(bridge_raw).hexdigest()
            if latent_output_authority
            else ""
        ),
        "decode_bridge_logits_digest": "d" * 64 if latent_output_authority else "",
        "terminal_disposition": finalize_terminal_disposition_receipt(
            decision,
            instruction_tokens=instruction_tokens,
            instruction_policy=(
                "applied" if latent_output_authority else "suppressed"
            ),
            full_bridge_tokens=bridge_tokens,
            output_tokens=tokens,
            output_text=text,
            output_source="resident_model_decode",
        ),
    }


def _attach_nonadmitted_fast_weight_receipt(
    receipt: dict,
    *,
    text: str,
    tokens: list[int],
) -> None:
    from core.brain.llm.latent_cortex.fast_weight_learning import (
        empty_learning_state,
        finalize_fast_weight_learning_receipt,
        token_sequence_sha256,
        unavailable_admission,
    )

    receipt.update(
        {
            "fast_weights_applied": False,
            # A real receipt is a dataclass and always carries this field.
            # Omitting it here simulated a client that never reported whether
            # it TRIED to attach, which the contract rightly refuses.
            "fast_weights_attach_attempted": False,
            "fast_weights_erased": None,
            "fast_weights_layers": 0,
            "fast_weight_optimization_attempts": 0,
            "fast_weight_optimized_steps": 0,
            "fast_weight_rejected_steps": 0,
            "fast_weight_budget_exhausted": False,
            "fast_weight_optimizer": "",
            "fast_weight_loss_trail": [],
            "fast_weight_gradient_norm_trail": [],
            "fast_weight_accepted_step_sizes": [],
            "fast_weight_line_search_backtracks": 0,
        }
    )
    admission = unavailable_admission(
        source_sha256=hashlib.sha256(b"").hexdigest(),
        objective_sha256=hashlib.sha256(b"").hexdigest(),
        reason="candidate_evaluation_unavailable",
    )
    state = empty_learning_state(
        episode_id=str(receipt["episode_id"]),
        input_tokens_sha256=str(receipt["input_tokens_sha256"]),
        selected_branch=int(receipt["selected_branch"]),
        winner_state_sha256=hashlib.sha256(
            f"{receipt['episode_id']}:winner".encode()
        ).hexdigest(),
        admission=admission,
    )
    state["final_answer"] = {
        "decoded_under_adaptation": False,
        "tokens_sha256": token_sequence_sha256(tokens),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_count": len(tokens),
    }
    receipt["fast_weight_learning"] = (
        finalize_fast_weight_learning_receipt(state)
    )


def _branch_isolation_fields(config, *, exchanges=0):
    count = config["n_branches"]
    required = config["isolation_steps"]
    roles = (
        "constructive_solution",
        "counterexample_search",
        "constraint_checking",
        "causal_reconstruction",
    )

    def digest(label):
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    return {
        "exchanges": exchanges,
        "branch_isolation": {
            "schema": "aura.rlc.branch_isolation.v1",
            "n_branches": count,
            "required_steps": required,
            "sealed": True,
            "certified": True,
            "reason": "certified",
            "configured_role_lesion": False,
            "seed_alias_free": True,
            "seed_states_unique": True,
            "rng_streams_unique": True,
            "cross_exposure_started": exchanges > 0,
            "first_exchange_step": required if exchanges else None,
            "blocked_cross_exposures": 0,
            "candidates": [
                {
                    "index": index,
                    "role": roles[index % len(roles)],
                    "context_sha256": digest("shared-context"),
                    "rng_stream_sha256": digest(f"rng-{index}"),
                    "seed_sha256": digest(f"seed-{index}"),
                    "candidate_sha256": digest(f"candidate-{index}"),
                    "candidate_step": required,
                }
                for index in range(count)
            ],
            "cache_discipline": {
                "schema": "aura.rlc.cache_discipline.v1",
                "nonpersistent_calls": count + required,
                "restored_calls": count + required,
                "restore_failures": 0,
                "all_restored": True,
            },
        },
    }


def _kv_state_tree_fields(config, *, episode_id, n_layers=2):
    from core.brain.llm.latent_cortex.kv_state_tree import KVStateTree

    class AttrCache:
        def __init__(self, marker):
            self.keys = np.full((1, 1, 64, 4), marker, dtype=np.float32)
            self.values = np.full((1, 1, 64, 4), marker + 0.5, dtype=np.float32)
            self.offset = 64

        def append(self, count, marker):
            self.keys = np.concatenate(
                [
                    self.keys[:, :, : self.offset, :],
                    np.full((1, 1, count, 4), marker, dtype=np.float32),
                ],
                axis=2,
            )
            self.values = np.concatenate(
                [
                    self.values[:, :, : self.offset, :],
                    np.full((1, 1, count, 4), marker + 0.5, dtype=np.float32),
                ],
                axis=2,
            )
            self.offset += count

    cache = [AttrCache(float(index)) for index in range(n_layers)]
    tree = KVStateTree(
        cache,
        n_layers=n_layers,
        episode_id=episode_id,
        input_tokens_sha256="7" * 64,
    )
    rejected = tree.begin_speculation(
        cache,
        start=0,
        end=n_layers,
        purpose="recurrent_update",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    )
    for index, item in enumerate(cache):
        item.append(config["n_slots"], 10.0 + index)
    rejected.observe_mutation(cache)
    rejected.restore_parent(cache)
    rejected.reject_after_restore(cache)

    incumbent_policy = config.get(
        "decode_incumbent_policy",
        "vanilla_incumbent",
    )
    incumbent = incumbent_policy == "vanilla_incumbent"
    final = tree.begin_speculation(
        cache,
        start=0,
        end=n_layers,
        purpose=(
            "final_vanilla_incumbent_decode"
            if incumbent
            else "final_output_decode"
        ),
        branch_index=0,
        parent_sha256=tree.root_sha256,
    )
    for index, item in enumerate(cache):
        item.append(config["n_slots"] + 1, 20.0 + index)
    final.observe_mutation(cache)
    final.commit(
        label="final_output_lane",
        authority=(
            "vanilla_incumbent_output"
            if incumbent
            else "user_visible_decode"
        ),
        latent_sha256=("" if incumbent else _digest("final-latent")),
        final=True,
    )
    prompt_logits_sha256 = _digest(f"{episode_id}:prompt-tail-logits")
    return {
        "n_layers": n_layers,
        "kv_state_tree": tree.receipt(),
        "decode_incumbent_policy": incumbent_policy,
        "decode_incumbent_prompt_logits_sha256": prompt_logits_sha256,
        "first_logits_digest": prompt_logits_sha256,
    }


def _recurrent_grounding_fields(config, *, steps=1, episode_id=""):
    from core.brain.llm.latent_cortex.bidirectional_reflector import (
        build_bidirectional_reflector_receipt,
        observe_reflector_vectors,
    )
    from core.brain.llm.latent_cortex.contradiction_perturber import (
        ContradictionPerturberConfig,
        run_contradiction_perturbation,
    )
    from core.brain.llm.latent_cortex.contradiction_tensor import (
        ContradictionTensorRuntime,
        build_contradiction_tensor_receipt,
    )
    from core.brain.llm.latent_cortex.heterogeneous_integrator import (
        HeterogeneousIntegrationConfig,
        run_heterogeneous_integration,
    )
    from core.brain.llm.latent_cortex.local_exploration import (
        LocalExplorationConfig,
        run_local_exploration,
    )
    from core.brain.llm.latent_cortex.loop_core import (
        KV_BOUND_SCHEMA,
        alpha_for_step,
        build_loop_core_contract,
        canonical_sha256,
    )
    from core.brain.llm.latent_cortex.loop_stability import (
        build_loop_stability_receipt,
    )
    from core.brain.llm.latent_cortex.mistake_locator import (
        MistakeLocatorRuntime,
        build_mistake_locator_receipt,
    )
    from core.brain.llm.latent_cortex.neural_uncertainty import (
        NeuralUncertaintyRuntime,
        build_neural_uncertainty_receipt,
    )
    from core.brain.llm.latent_cortex.recurrent_grounding import (
        build_recurrent_grounding_receipt,
    )
    from core.brain.llm.latent_cortex.stop_gate import (
        RESIDUAL,
        StopGateRuntime,
        build_stop_gate_receipt,
    )
    from core.brain.llm.latent_cortex.transient_constraints import (
        build_empty_transient_constraint_receipt,
    )
    from core.brain.llm.latent_cortex.update_gate import (
        PASSTHROUGH,
        UpdateGateRuntime,
        build_update_gate_receipt,
    )
    from core.brain.llm.latent_cortex.verified_best import (
        build_verified_best_receipt,
    )
    from core.brain.llm.latent_cortex.virtual_quanta import (
        VirtualQuantaConfig,
        build_empty_virtual_quanta_receipt,
    )
    from core.brain.llm.latent_cortex.worker_handler import config_from_job
    from core.learning.update_acceptance import (
        FEATURE_NAMES,
        UPDATE_ACCEPTANCE_FEATURE_SCHEMA,
    )

    executed = config_from_job(config)
    steps = min(steps, executed.recurrence.max_steps)
    prelude_end, coda_start = 1, 2
    evidence_sha = _digest("empty-evidence")
    branches = []
    for index in range(config["n_branches"]):
        initial = _digest(f"hypothesis-{index}-0")
        transitions = []
        stability = []
        update_acceptance = []
        reflector_trace = []
        prior = initial
        reasoning_prior = _digest(f"reasoning-{index}-0")
        anchor_sha = _digest(f"anchor-{index}")
        prior_residual = None
        for step in range(steps):
            post = _digest(f"hypothesis-{index}-{step + 1}")
            reasoning_post = _digest(f"reasoning-{index}-{step + 1}")
            transitions.append(
                {
                    "ordinal": step,
                    "branch_step": step,
                    "window_start": 1,
                    "window_end": 2,
                    "evidence_pre_sha256": evidence_sha,
                    "evidence_post_sha256": evidence_sha,
                    "hypothesis_pre_sha256": prior,
                    "hypothesis_post_sha256": post,
                    "evidence_unchanged": True,
                    "hypothesis_changed": True,
                }
            )
            residual = round(0.5 / (step + 1), 8)
            contraction = None if prior_residual is None else round(residual / prior_residual, 8)
            stability.append(
                {
                    "ordinal": step,
                    "branch_step": step,
                    "window_start": prelude_end,
                    "window_end": coda_start,
                    "hypothesis_pre_sha256": prior,
                    "hypothesis_post_sha256": post,
                    "reasoning_pre_sha256": reasoning_prior,
                    "reasoning_post_sha256": reasoning_post,
                    "anchor_sha256": anchor_sha,
                    "continuous_from_previous": step > 0,
                    "disposition": "accepted",
                    "divergence_reason": "",
                    "containment_action": "",
                    "alpha": round(
                        alpha_for_step(
                            alpha=executed.recurrence.alpha,
                            schedule=executed.recurrence.alpha_schedule,
                            max_steps=executed.recurrence.max_steps,
                            step=step,
                        ),
                        8,
                    ),
                    "input_mean_rms": 1.0,
                    "output_mean_rms": 1.0,
                    "anchor_mean_rms": 1.0,
                    "anchor_rms_ratio": 1.0,
                    "residual": residual,
                    "contraction_ratio": contraction,
                    "delta_cosine": None if step == 0 else 0.25,
                    "contracting": None if step == 0 else contraction < 1.0,
                    "oscillating": False,
                    "fixed_point_candidate": (residual < executed.recurrence.convergence_eps),
                    "all_finite": True,
                }
            )
            features = {name: 0.0 for name in FEATURE_NAMES}
            update_acceptance.append(
                {
                    "ordinal": step,
                    "branch_step": step,
                    "prior_hypothesis_sha256": prior,
                    "proposal_hypothesis_sha256": post,
                    "admitted_hypothesis_sha256": post,
                    "prior_reasoning_sha256": reasoning_prior,
                    "proposal_reasoning_sha256": reasoning_post,
                    "admitted_reasoning_sha256": reasoning_post,
                    "probability": 1.0,
                    "threshold": 0.0,
                    "accepted": True,
                    "reason": "passthrough",
                    "features": features,
                    "features_sha256": canonical_sha256(
                        {
                            "schema": UPDATE_ACCEPTANCE_FEATURE_SCHEMA,
                            "values": features,
                        }
                    ),
                }
            )
            prior_vector = (float(step), 0.0, float(index), 1.0)
            proposal_vector = (
                float(step + 1),
                0.25,
                float(index),
                1.0,
            )
            reflector_trace.append(
                observe_reflector_vectors(
                    prior_vector,
                    proposal_vector,
                    proposal_vector,
                    branch_index=index,
                    branch_step=step,
                    prior_state_sha256=reasoning_prior,
                    proposal_state_sha256=reasoning_post,
                    admitted_state_sha256=reasoning_post,
                    accepted=True,
                )
            )
            prior = post
            reasoning_prior = reasoning_post
            prior_residual = residual
        branches.append(
            SimpleNamespace(
                index=index,
                role=("constructive_solution" if index == 0 else "counterexample_search"),
                evidence_anchor_sha256=evidence_sha,
                initial_hypothesis_sha256=initial,
                recurrent_grounding_trace=transitions,
                loop_stability_trace=stability,
                update_acceptance_trace=update_acceptance,
                halt_reason="schedule_complete",
                steps=steps,
                halting=SimpleNamespace(stop_trace=[]),
                verified_best_trace=[],
                verified_best_step=-1,
                verified_best_state_sha256="",
                verified_best_observation={},
                verified_finalization={
                    "source": "current",
                    "pre_state_sha256": prior,
                    "post_state_sha256": prior,
                    "reverted": False,
                    "fixed_depth": executed.recurrence.fixed_depth,
                },
                uncertainty_trace=[],
                mistake_locator_trace=[],
                reflector_trace=reflector_trace,
            )
        )
    receipt = build_recurrent_grounding_receipt(
        input_tokens_sha256="7" * 64,
        input_token_count=64,
        cognitive_slots=[],
        branches=branches,
        n_slots=config["n_slots"],
        comm_slot=0,
        selected_branch=0,
    )
    loop_core = build_loop_core_contract(
        prelude_end=prelude_end,
        coda_start=coda_start,
        max_steps=executed.recurrence.max_steps,
        min_steps=executed.recurrence.min_steps,
        alpha=executed.recurrence.alpha,
        alpha_schedule=executed.recurrence.alpha_schedule,
        rms_clip_ratio=executed.recurrence.rms_clip_ratio,
        convergence_eps=executed.recurrence.convergence_eps,
        divergence_ratio=executed.recurrence.divergence_ratio,
        fixed_depth=executed.recurrence.fixed_depth,
    )
    calls = [
        {
            "ordinal": ordinal,
            "start": prelude_end,
            "end": coda_start,
            "tokens": config["n_slots"],
            "context_tokens": 64,
            "total_tokens": 64 + config["n_slots"],
            "post_context_tokens": 64,
            "persist": False,
            "restored": True,
        }
        for ordinal in range(config["n_branches"] * steps)
    ]
    kv_bound = {
        "schema": KV_BOUND_SCHEMA,
        "position_limit": 512,
        "position_limit_source": "model_config",
        "call_count": len(calls),
        "max_context_tokens": 64,
        "max_total_tokens": 64 + config["n_slots"],
        "all_within_limit": True,
        "calls": calls,
        "calls_sha256": canonical_sha256(calls),
    }
    loop_stability = build_loop_stability_receipt(
        branches=branches,
        selected_branch=0,
        loop_core=loop_core,
        kv_bound=kv_bound,
        recurrent_grounding=receipt,
    )
    update_gate = UpdateGateRuntime(mode=PASSTHROUGH)
    update_acceptance_receipt = build_update_gate_receipt(
        branches=branches,
        selected_branch=0,
        gate=update_gate,
        recurrent_grounding=receipt,
        loop_stability=loop_stability,
    )
    stop_gate = StopGateRuntime(mode=RESIDUAL)
    halting_receipt = build_stop_gate_receipt(
        branches=branches,
        gate=stop_gate,
        update_acceptance=update_acceptance_receipt,
        loop_stability=loop_stability,
        cognitive_action_trace=[],
    )
    verified_best_receipt = build_verified_best_receipt(
        branches=branches,
        cognitive_action_trace=[],
        loop_stability=loop_stability,
    )
    neural_uncertainty_receipt = build_neural_uncertainty_receipt(
        branches=branches,
        runtime=NeuralUncertaintyRuntime.from_config(executed.uncertainty_head),
        update_acceptance=update_acceptance_receipt,
        selected_branch=0,
        selection_basis="convergence",
    )
    mistake_locator_receipt = build_mistake_locator_receipt(
        branches=branches,
        runtime=MistakeLocatorRuntime.from_config(executed.mistake_locator),
        update_acceptance=update_acceptance_receipt,
        selected_branch=0,
    )
    bidirectional_reflector_receipt = build_bidirectional_reflector_receipt(
        branches=branches,
        update_acceptance=update_acceptance_receipt,
        selected_branch=0,
    )
    contradiction_tensor_receipt = build_contradiction_tensor_receipt(
        reflector=bidirectional_reflector_receipt,
        runtime=ContradictionTensorRuntime.from_config(executed.contradiction_head),
        selected_branch=0,
    )
    _, contradiction_perturbation_receipt = run_contradiction_perturbation(
        baseline=np.zeros((1, config["n_slots"], 8), dtype=np.float32),
        anchor=np.zeros((1, config["n_slots"], 8), dtype=np.float32),
        protected_positions=(),
        contradiction_tensor=contradiction_tensor_receipt,
        selected_branch=0,
        config=ContradictionPerturberConfig.from_value(executed.contradiction_perturber),
        verifier_policy_sha256="a" * 64,
        decoy_review_sha256="",
        evaluate=None,
    )
    _, local_exploration_receipt = run_local_exploration(
        baseline=np.zeros((1, config["n_slots"], 8), dtype=np.float32),
        protected_positions=(),
        contradiction_tensor=contradiction_tensor_receipt,
        contradiction_perturbation=contradiction_perturbation_receipt,
        neural_uncertainty=neural_uncertainty_receipt,
        selected_branch=0,
        config=LocalExplorationConfig.from_value(executed.local_exploration),
        verifier_policy_sha256="a" * 64,
        decoy_review_sha256="",
        evaluate=None,
    )
    _, _, heterogeneous_integration_receipt = run_heterogeneous_integration(
        incumbent_state=np.zeros(
            (1, config["n_slots"], 8),
            dtype=np.float32,
        ),
        corrected_state=np.zeros(
            (1, config["n_slots"], 8),
            dtype=np.float32,
        ),
        contradiction_perturbation=(contradiction_perturbation_receipt),
        local_exploration=local_exploration_receipt,
        config=HeterogeneousIntegrationConfig.from_value(executed.heterogeneous_integration),
        verifier_policy_sha256="a" * 64,
        decoy_review_sha256="",
        evaluate=None,
    )
    transient_constraint_receipt = (
        build_empty_transient_constraint_receipt(
            episode_id=episode_id,
            objective_sha256="7" * 64,
            n_branches=config["n_branches"],
            protected_positions={index: () for index in range(config["n_branches"])},
        )
        if episode_id
        else None
    )
    virtual_quanta_receipt = (
        build_empty_virtual_quanta_receipt(
            episode_id=episode_id,
            objective_sha256="7" * 64,
            subject_sha256="b" * 64,
            branch_index=0,
            source_kv_boundary_sha256="d" * 64,
            protected_positions=(),
            source_positions=(),
            config=VirtualQuantaConfig.from_value(executed.virtual_quanta),
        )
        if episode_id
        else None
    )
    return {
        "cognitive_slots": [],
        "selected_branch": 0,
        "prelude_end": prelude_end,
        "coda_start": coda_start,
        "recurrent_grounding": receipt,
        "loop_stability": loop_stability,
        "update_acceptance": update_acceptance_receipt,
        "halting": halting_receipt,
        "verified_best_state": verified_best_receipt,
        **(
            {"transient_negative_constraints": transient_constraint_receipt}
            if transient_constraint_receipt is not None
            else {}
        ),
        **(
            {"virtual_quanta": virtual_quanta_receipt} if virtual_quanta_receipt is not None else {}
        ),
        "neural_uncertainty": neural_uncertainty_receipt,
        "mistake_locator": mistake_locator_receipt,
        "bidirectional_reflector": bidirectional_reflector_receipt,
        "contradiction_tensor": contradiction_tensor_receipt,
        "contradiction_perturbation": contradiction_perturbation_receipt,
        "local_exploration": local_exploration_receipt,
        "heterogeneous_integration": (heterogeneous_integration_receipt),
        "heterogeneous_decode": {},
        "cognitive_action_trace": [],
    }


def _branch_exchange_fields(config, isolation):
    from core.brain.llm.latent_cortex.branch_exchange import (
        BRANCH_EXCHANGE_SCHEMA,
        MAX_EXCHANGE_SOURCE_SLOTS,
        build_branch_exchange_trace,
        candidate_set_sha256,
        canonical_sha256,
    )
    from core.brain.llm.latent_cortex.cognitive_operators import operator_for_role

    count = config["n_branches"]
    n_slots = config["n_slots"]
    hidden = 64
    source_slots = list(range(1, n_slots))[:MAX_EXCHANGE_SOURCE_SLOTS]
    source_rows = []
    for index, candidate in enumerate(isolation["candidates"]):
        source_rows.append(
            {
                "branch_index": index,
                "role": candidate["role"],
                "operator": operator_for_role(candidate["role"]).value,
                "step": candidate["candidate_step"],
                "candidate_sha256": candidate["candidate_sha256"],
                "candidate_step": candidate["candidate_step"],
                "source_slots": source_slots,
                "excluded_slots": [0],
                "state_sha256": _digest(f"state-{index}"),
                "private_state_sha256": _digest(f"private-{index}"),
                "message_sha256": _digest(f"message-{index}"),
                "support_weight": 1.0,
                "consensus_weight": round(1.0 / count, 12),
            }
        )
    payload = {
        "schema": BRANCH_EXCHANGE_SCHEMA,
        "ordinal": 0,
        "sync_kind": "interval",
        "sync_id": "recurrent-step:2",
        "generation": "independent_candidates",
        "n_branches": count,
        "n_slots": n_slots,
        "comm_slot": 0,
        "exchange_gamma": config["exchange_gamma"],
        "source_policy": ("bounded_private_reasoning_mean_excluding_mailbox_and_context_v1"),
        "message_representation": "latent_tensor_only",
        "message_slot_count": 1,
        "hidden_dimension": hidden,
        "source_slot_limit": MAX_EXCHANGE_SOURCE_SLOTS,
        "context_slots_excluded": [],
        "comm_slot_excluded": True,
        "first_answer_text_exposed": False,
        "prior_peer_context_possible": False,
        "counts_as_independent_support": True,
        "candidate_set_sha256": candidate_set_sha256({"candidates": isolation["candidates"]}),
        "source_rows": source_rows,
        "consensus_sha256": _digest("consensus"),
        "recipient_rows": [
            {
                "branch_index": index,
                "comm_pre_sha256": _digest(f"comm-pre-{index}"),
                "comm_post_sha256": _digest(f"comm-post-{index}"),
                "non_comm_pre_sha256": _digest(f"noncomm-{index}"),
                "non_comm_post_sha256": _digest(f"noncomm-{index}"),
                "state_pre_sha256": _digest(f"recipient-pre-{index}"),
                "state_post_sha256": _digest(f"recipient-post-{index}"),
                "causal": True,
            }
            for index in range(count)
        ],
        "tensor_accounting": {
            "source_elements_read": count * len(source_slots) * hidden,
            "message_elements_emitted": count * hidden,
            "consensus_elements_written": count * hidden,
            "tensor_scalar_ops": (count * hidden * (len(source_slots) + 12) + 9 * count),
            "hidden_layer_apps": 0,
        },
    }
    exchange = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return build_branch_exchange_trace(
        exchanges=[exchange],
        n_branches=count,
        n_slots=n_slots,
        comm_slot=0,
        exchange_gamma=config["exchange_gamma"],
        branch_isolation={"candidates": isolation["candidates"]},
        cognitive_slots=[],
        exchange_interval=config["exchange_interval"],
        schedule_hash="test-schedule",
        bytecode_events=[],
        cognitive_action_trace=[],
    )


def _bind_test_client_identity(monkeypatch, client):
    from core.brain.llm.latent_cortex import runtime_identity

    client._worker_identity = dict(_WORKER_IDENTITY)
    monkeypatch.setattr(
        runtime_identity,
        "collect_latent_runtime_identity",
        lambda *_args, **_kwargs: dict(_RUNTIME_IDENTITY),
    )


# ── Worker handler ──────────────────────────────────────────────────────


def test_config_from_job_defaults_are_conservative():
    cfg = config_from_job(None)
    assert cfg.workspace.n_slots == 16
    assert cfg.recurrence.max_steps == 8
    assert cfg.branches.n_branches == 2
    assert cfg.branches.isolation_steps == 2
    assert cfg.latent_opt.enabled is False
    assert cfg.fast_weights.enabled is False
    assert cfg.decode_incumbent_policy == "vanilla_incumbent"
    assert cfg.verifier_probe_max_tokens == 48
    assert cfg.verifier_probe_contract == "none"
    assert cfg.verifier_accept_non_regression is False
    assert cfg.prefix_stability_enabled is True
    assert cfg.prefix_stability_samples == 3
    assert cfg.prefix_stability_max_tokens == 128
    assert cfg.prefix_stability_calibrator is None
    assert cfg.local_repair_enabled is True
    assert cfg.local_repair_max_attempts == 1
    assert cfg.local_repair_max_tokens == 128
    assert cfg.answer_replacement_enabled is True
    assert cfg.objective_program_enabled is True
    assert cfg.verified_objective_teacher_enabled is True
    assert cfg.answer_replacement_margin == pytest.approx(0.05)
    assert cfg.uncertainty_head is None
    assert cfg.mistake_locator is None
    assert cfg.contradiction_head is None
    assert cfg.contradiction_perturber is None
    assert cfg.local_exploration is None
    assert cfg.heterogeneous_integration is None
    assert cfg.transient_negative_constraints is None
    assert cfg.virtual_quanta is None
    assert cfg.validate() == []


def test_config_from_job_rejects_out_of_band_requests():
    with pytest.raises(ValueError):
        config_from_job({"n_branches": 640})
    with pytest.raises(ValueError):
        config_from_job({"max_steps": 100000})
    with pytest.raises(ValueError, match="JSON boolean"):
        config_from_job({"fast_weights": "false"})
    with pytest.raises(ValueError, match="unknown keys"):
        config_from_job({"fast_weight": True})
    with pytest.raises(ValueError):
        config_from_job({"exchange_interval": 0})
    with pytest.raises(ValueError):
        config_from_job({"isolation_steps": 9, "max_steps": 8})
    with pytest.raises(ValueError):
        config_from_job({"decode_temperature": float("nan")})
    with pytest.raises(ValueError):
        config_from_job({"verifier_probe_max_tokens": 15})
    with pytest.raises(ValueError, match="verifier_probe_contract"):
        config_from_job({"verifier_probe_contract": "advisory"})
    with pytest.raises(ValueError, match="JSON boolean"):
        config_from_job({"verifier_accept_non_regression": "true"})
    with pytest.raises(ValueError, match="decode_incumbent_policy"):
        config_from_job({"decode_incumbent_policy": "unproven_fusion"})
    with pytest.raises(ValueError, match="outside"):
        config_from_job({"prefix_stability_samples": 2})
    with pytest.raises(ValueError, match="outside"):
        config_from_job({"local_repair_max_attempts": 9})
    with pytest.raises(ValueError, match="outside"):
        config_from_job({"local_repair_max_tokens": 16})
    with pytest.raises(ValueError, match="outside"):
        config_from_job({"answer_replacement_margin": 1.0})
    with pytest.raises(ValueError, match="JSON boolean"):
        config_from_job({"verified_objective_teacher_enabled": "true"})
    with pytest.raises(ValueError, match="calibrator config"):
        config_from_job({"prefix_stability_calibrator": {"mode": "learned"}})
    with pytest.raises(ValueError, match="requires head_path"):
        config_from_job({"uncertainty_head": {"mode": "learned"}})
    with pytest.raises(ValueError, match="cannot carry a head"):
        config_from_job(
            {
                "uncertainty_head": {
                    "mode": "unavailable",
                    "head_path": "unused-head.json",
                }
            }
        )
    with pytest.raises(ValueError, match="replicates"):
        config_from_job({"contradiction_perturber": {"replicates": 1}})
    with pytest.raises(ValueError, match="delta bound"):
        config_from_job(
            {
                "contradiction_perturber": {
                    "max_relative_delta_rms": 0.5,
                }
            }
        )
    with pytest.raises(ValueError, match="candidates"):
        config_from_job({"local_exploration": {"candidates": 1}})
    with pytest.raises(ValueError, match="entropy floor"):
        config_from_job({"local_exploration": {"min_predictive_entropy": -0.1}})
    with pytest.raises(ValueError, match="replicates"):
        config_from_job({"heterogeneous_integration": {"replicates": 1}})
    with pytest.raises(ValueError, match="JS floor"):
        config_from_job(
            {
                "heterogeneous_integration": {
                    "min_js_divergence_bits": 1.1,
                }
            }
        )
    with pytest.raises(ValueError, match="unknown keys"):
        config_from_job(
            {
                "transient_negative_constraints": {
                    "unbounded_lifetime": True,
                }
            }
        )
    with pytest.raises(ValueError, match="delta bound"):
        config_from_job(
            {
                "transient_negative_constraints": {
                    "max_relative_delta_rms": 0.21,
                }
            }
        )
    with pytest.raises(ValueError, match="replicates"):
        config_from_job(
            {
                "transient_negative_constraints": {
                    "replicates": 1,
                }
            }
        )
    with pytest.raises(ValueError, match="TTL"):
        config_from_job(
            {
                "transient_negative_constraints": {
                    "ttl_action_steps": 17,
                }
            }
        )
    with pytest.raises(ValueError, match="unknown keys"):
        config_from_job({"virtual_quanta": {"payload": "free-form"}})
    with pytest.raises(ValueError, match="delta bound"):
        config_from_job({"virtual_quanta": {"max_relative_delta_rms": 0.5}})
    with pytest.raises(ValueError, match="replicates"):
        config_from_job({"virtual_quanta": {"replicates": 1}})
    with pytest.raises(ValueError, match="TTL"):
        config_from_job({"virtual_quanta": {"ttl_steps": 5}})
    with pytest.raises(ValueError, match="requires head_path"):
        config_from_job({"contradiction_head": {"mode": "learned"}})
    with pytest.raises(ValueError, match="cannot carry a head"):
        config_from_job(
            {
                "contradiction_head": {
                    "mode": "unavailable",
                    "head_path": "unused-head.json",
                }
            }
        )
    with pytest.raises(ValueError, match="requires head_path"):
        config_from_job({"mistake_locator": {"mode": "learned"}})
    with pytest.raises(ValueError, match="cannot carry a head"):
        config_from_job(
            {
                "mistake_locator": {
                    "mode": "unavailable",
                    "head_path": "unused-head.json",
                }
            }
        )


def test_config_from_job_can_explicitly_restore_research_latent_decode():
    cfg = config_from_job({"decode_incumbent_policy": "latent"})

    assert cfg.decode_incumbent_policy == "latent"


def test_config_from_job_maps_every_advanced_mechanism():
    cfg = config_from_job(
        {
            "latent_opt": True,
            "latent_opt_steps": 6,
            "latent_opt_lr": 0.02,
            "fast_weights": True,
            "fast_weights_opt_steps": 3,
            "fast_weights_lr": 0.005,
            "fast_weights_max_layers": 4,
            "fast_weights_canary_max_delta_rms": 0.025,
            "fast_weights_query_gate": True,
            "fast_weights_query_gate_threshold": 0.75,
            "fast_weights_query_gate_temperature": 0.04,
            "exchange_gamma": 0.2,
            "convergence_eps": 0.01,
            "decode_top_p": 0.82,
            "verifier_probe_max_tokens": 24,
            "verifier_accept_non_regression": True,
            "objective_program_enabled": False,
            "verified_objective_teacher_enabled": True,
            "input_context_max_chars": 4096,
            "allow_vanilla_fallback": False,
            "transient_negative_constraints": {
                "max_relative_delta_rms": 0.05,
                "min_verifier_margin": 0.02,
                "replicates": 3,
                "ttl_action_steps": 4,
                "max_constraints": 5,
            },
            "virtual_quanta": {
                "max_relative_delta_rms": 0.04,
                "min_verifier_margin": 0.03,
                "replicates": 3,
                "ttl_steps": 2,
                "seed": 19,
            },
        }
    )
    assert cfg.latent_opt.enabled is True and cfg.latent_opt.steps == 6
    assert cfg.latent_opt.lr == 0.02
    assert cfg.fast_weights.enabled is True and cfg.fast_weights.opt_steps == 3
    assert cfg.fast_weights.lr == 0.005 and cfg.fast_weights.max_wrapped_layers == 4
    assert cfg.fast_weights.canary_max_effective_delta_rms == 0.025
    assert cfg.fast_weights.query_gate_enabled is True
    assert cfg.fast_weights.query_gate_threshold == 0.75
    assert cfg.fast_weights.query_gate_temperature == 0.04
    assert cfg.branches.exchange_gamma == 0.2
    assert cfg.recurrence.convergence_eps == 0.01
    assert cfg.decode_top_p == 0.82
    assert cfg.verifier_probe_max_tokens == 24
    assert cfg.verifier_accept_non_regression is True
    assert cfg.objective_program_enabled is False
    assert cfg.verified_objective_teacher_enabled is True
    assert cfg.input_context_max_chars == 4096
    assert cfg.allow_vanilla_fallback is False
    assert cfg.transient_negative_constraints == {
        "max_relative_delta_rms": 0.05,
        "min_verifier_margin": 0.02,
        "replicates": 3,
        "ttl_action_steps": 4,
        "max_constraints": 5,
    }
    assert cfg.virtual_quanta == {
        "max_relative_delta_rms": 0.04,
        "min_verifier_margin": 0.03,
        "replicates": 3,
        "ttl_steps": 2,
        "seed": 19,
    }


def test_budget_from_job_caps_apply():
    budget = budget_from_job({"max_layer_apps": 10**15, "wall_clock_s": 5.0})
    assert budget.wall_clock_s == 5.0
    assert budget.max_layer_apps == 500_000_000
    assert budget.remaining_layer_apps == 500_000_000


@pytest.mark.parametrize(
    "payload",
    [
        {"max_layer_apps": -1},
        {"wall_clock_s": 0},
        {"wall_clock_s": float("inf")},
        {"max_layer_apps": "1000"},
        {"typo": 1},
    ],
)
def test_budget_from_job_rejects_invalid_values(payload):
    with pytest.raises((TypeError, ValueError)):
        budget_from_job(payload)


def test_kill_switch_refuses_honestly(monkeypatch):
    monkeypatch.setenv("AURA_LATENT_CORTEX", "0")
    assert cortex_enabled() is False
    body = handle_latent_reason({"prompt": "hi"}, model=None, tokenizer=None, model_path="")
    assert body["status"] == "error"
    assert "latent_cortex_disabled" in body["message"]


def test_handler_requires_prompt(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    body = handle_latent_reason({}, model=None, tokenizer=None, model_path="")
    assert body["status"] == "error"
    assert "requires prompt" in body["message"]


def test_handler_rejects_malformed_response_contract_before_engine(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    constructed = False

    class ForbiddenEngine:
        def __init__(self, *args, **kwargs):
            nonlocal constructed
            constructed = True

    import core.brain.llm.latent_cortex.worker_handler as handler_mod

    monkeypatch.setattr(handler_mod, "LatentCortexEngine", ForbiddenEngine)
    body = handle_latent_reason(
        {"prompt": "answer", "response_contract": '{"answer":not_a_type}'},
        model=object(),
        tokenizer=object(),
        model_path="/models/test-32b",
    )

    assert body["status"] == "error"
    assert "response_contract rejected" in body["message"]
    assert constructed is False


def test_handler_rejects_external_offer_without_full_authority_tuple(
    monkeypatch,
) -> None:
    from core.brain.llm.latent_cortex.external_execution import (
        build_external_execution_offer,
    )

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    offer = build_external_execution_offer(
        action_id="worker-missing-authority",
        domain="external_action",
        action_name="write_note",
        request_digest="sha256:" + "a" * 64,
        will_receipt_id="will-worker",
        objective="Write the admitted note.",
        expectation={"objective": "note exists"},
    )
    body = handle_latent_reason(
        {"prompt": "reason", "external_execution_offer": offer},
        model=object(),
        tokenizer=object(),
        model_path="/models/test-32b",
    )
    assert body["status"] == "error"
    assert "requires operation authority" in body["message"]


def test_handler_wires_response_contract_into_config_and_verifier(monkeypatch):
    from core.brain.llm.latent_cortex.types import LatentReasoningResult

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    captured: dict = {}

    class StubEngine:
        def __init__(self, model, tokenizer, config, **kwargs):
            captured["config"] = config

        def reason(self, **kwargs):
            captured.update(kwargs)
            return LatentReasoningResult(
                ok=True,
                text='FINAL_ANSWER: {"answer":7}',
                    receipt=_measured_episode_receipt(
                        "response-contract-test"
                    ),
            )

    import core.brain.llm.latent_cortex.worker_handler as handler_mod

    monkeypatch.setattr(handler_mod, "LatentCortexEngine", StubEngine)
    body = handle_latent_reason(
        {
            "prompt": "answer with an integer",
            "response_contract": '{"answer":int}',
            "config": {"decode_max_tokens": 96},
        },
        model=object(),
        tokenizer=object(),
        model_path="/models/test-32b",
        worker_identity=dict(_WORKER_IDENTITY),
    )

    assert body["status"] == "ok"
    assert captured["config"].decode_contract == "final_answer_v1"
    assert captured["config"].decode_contract_grace_tokens == 96
    verifier = captured["verifier"]
    assert verifier is not None
    assert verifier.response_contract == '{"answer":int}'
    assert body["receipt"]["verifier_guidance"]["response_contract_required"] is True


def test_handler_authenticates_and_binds_action_intervention(monkeypatch):
    from core.brain.llm.latent_cortex import action_intervention as intervention_mod
    from core.brain.llm.latent_cortex.types import LatentReasoningResult
    from core.brain.llm.latent_cortex.value_of_computation import (
        build_evidence_snapshot,
    )

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    captured: dict = {}
    evidence = build_evidence_snapshot(bucket="b", cells={})
    request_sha256 = latent_request_payload_sha256(
        prompt="reason",
        messages=None,
        domain="general",
        config=None,
        budget=None,
        runtime_controls=None,
        action_policy_evidence=evidence,
    )
    normalized = {
        "schema": "aura.rlc.action_intervention.v3",
        "authority_payload": {
            "action": "formalize",
            "arm": "forced_action",
            "request_payload_sha256": request_sha256,
        },
        "intervention_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        intervention_mod,
        "validate_action_intervention",
        lambda value, *, require_current_policy: (
            normalized if value == {"wire": "signed"} and require_current_policy else None
        ),
    )
    monkeypatch.setattr(
        intervention_mod,
        "consume_action_intervention_once",
        lambda value: {
            "event": "CONSUMED",
            "intervention_sha256": value["intervention_sha256"],
        },
    )

    class StubEngine:
        def __init__(self, *args, **kwargs):
            captured["constructed"] = True

        def reason(self, **kwargs):
            captured.update(kwargs)
            return LatentReasoningResult(
                ok=True,
                text="bounded",
                    receipt=_measured_episode_receipt(
                        "action-intervention-test"
                    ),
            )

    import core.brain.llm.latent_cortex.worker_handler as handler_mod

    monkeypatch.setattr(handler_mod, "LatentCortexEngine", StubEngine)
    body = handle_latent_reason(
        {
            "prompt": "reason",
            "action_policy_evidence": evidence,
            "action_intervention": {"wire": "signed"},
        },
        model=object(),
        tokenizer=object(),
        model_path="/models/test-32b",
        worker_identity=dict(_WORKER_IDENTITY),
    )

    assert body["status"] == "ok"
    assert captured["constructed"] is True
    assert captured["action_intervention"] == normalized
    assert captured["action_intervention_consumption"] == {
        "event": "CONSUMED",
        "intervention_sha256": normalized["intervention_sha256"],
    }
    assert body["receipt"]["request_payload_sha256"] == latent_request_payload_sha256(
        prompt="reason",
        messages=None,
        domain="general",
        config=None,
        budget=None,
        runtime_controls=None,
        action_policy_evidence=evidence,
        action_intervention=normalized,
    )


def test_handler_compacts_messages_but_hashes_the_original_request(monkeypatch):
    from core.brain.llm.latent_cortex.types import (
        LatentReasoningResult,
    )

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    captured: dict = {}

    class StubEngine:
        def __init__(self, model, tokenizer, config, **kwargs):
            captured["config"] = config

        def reason(self, **kwargs):
            captured.update(kwargs)
            return LatentReasoningResult(
                ok=True,
                text="bounded",
                    receipt=_measured_episode_receipt(
                        "context-compaction-test"
                    ),
            )

    import core.brain.llm.latent_cortex.worker_handler as handler_mod

    monkeypatch.setattr(handler_mod, "LatentCortexEngine", StubEngine)
    messages = [
        {"role": "system", "content": "system " + "s" * 4000},
        {"role": "user", "content": "question " + "u" * 4000},
    ]
    config = {"input_context_max_chars": 2048}
    job = {
        "messages": messages,
        "config": config,
        "domain": "unit",
    }
    body = handle_latent_reason(
        job,
        model=object(),
        tokenizer=object(),
        model_path="/models/test-32b",
        worker_identity=dict(_WORKER_IDENTITY),
    )

    compacted = captured["messages"]
    assert sum(len(item["content"]) for item in compacted) <= 2048
    receipt = body["receipt"]
    assert receipt["input_context_compaction"]["applied"] is True
    assert receipt["input_context_compaction"]["compacted_char_count"] <= 2048
    assert receipt["request_payload_sha256"] == latent_request_payload_sha256(
        prompt=None,
        messages=messages,
        domain="unit",
        config=config,
        budget=None,
        runtime_controls=None,
    )


def test_handler_runs_full_episode_on_tiny_model(monkeypatch, tmp_path):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=8,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "tie_word_embeddings": False,
                "quantization": {"bits": 4, "group_size": 64},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "weights.npz").write_bytes(b"tiny-model-fixture")

    class StubTokenizer:
        eos_token_id = 0
        bos_token_id = 1
        pad_token_id = 0
        unk_token_id = 2
        vocab_size = 128
        special_tokens_map = {}
        chat_template = ""

        def encode(self, text):
            return [ord(c) % 128 for c in text][:16]

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    body = handle_latent_reason(
        {
            "prompt": "compose the deepest thought",
            "config": {
                "n_slots": 4,
                "n_branches": 2,
                "max_steps": 4,
                "exchange_interval": 1,
                "decode_max_tokens": 6,
            },
            "budget": {"wall_clock_s": 30.0},
            "domain": "unit",
            "verifier_guidance": True,
        },
        model=model,
        tokenizer=StubTokenizer(),
        model_path=str(tmp_path),
    )
    assert body["status"] == "ok", body
    assert body["receipt"]["params_unchanged"] is True
    assert body["receipt"]["steps_taken"] >= 2
    assert body["receipt"]["branch_isolation"]["certified"] is True
    exchange_trace = body["receipt"]["branch_exchange"]
    assert exchange_trace["exchange_count"] == body["receipt"]["exchanges"] >= 1
    assert exchange_trace["declared_sync_points_proven"] is True
    assert exchange_trace["independent_support_generations"] == 1
    policy = body["receipt"]["value_of_computation"]
    trace = body["receipt"]["cognitive_action_trace"]
    assert policy["active"] is True
    assert policy["actions_selected"] == len(trace) >= 2
    assert policy["actions_selected"] <= 4
    assert "execute" not in policy["executors"]
    assert all(row["decision"]["action"] in policy["executors"] for row in trace)
    worker_causal = body["receipt"]["causal_receipt"]
    assert worker_causal["episode_id"] == body["receipt"]["episode_id"]
    assert worker_causal["input_tokens_sha256"] == body["receipt"][
        "input_tokens_sha256"
    ]
    assert body["receipt"]["request_payload_sha256"]
    identity_commitments = worker_causal["nodes"][0]["source_commitments"]
    assert next(
        row for row in identity_commitments if row["field"] == "request_payload_sha256"
    )["present"] is True
    assert next(
        row for row in identity_commitments if row["field"] == "runtime_identity"
    )["present"] is False
    operators = body["receipt"]["cognitive_operator_trace"]
    assert operators
    assert {row["operator"] for row in operators} == {
        "constructive_solution",
        "counterexample",
    }
    structure = body["receipt"]["structural_diversity"]
    assert structure["certified"] is True
    assert structure["independent_support_count"] == 2
    assert structure["wording_counted"] is False
    assert body["receipt"]["correlated_support"]["raw_support_count"] == 2
    assert body["receipt"]["blind_review"]["deranged_order"] is True
    assert body["receipt"]["blind_review"]["first_answer_designated"] is False
    assert body["receipt"]["verifier_preflight"]["verifier_admitted"] is True
    decoy = body["receipt"]["decoy_verification"]
    assert decoy["certified"] is True
    assert decoy["selection_admitted"] is True
    assert {row["kind"] for row in decoy["controls"]} == {
        "correct",
        "incorrect",
        "unchanged_a",
        "unchanged_b",
    }
    from core.brain.llm.latent_cortex.blind_review import (
        validate_blind_review_receipt,
    )

    validate_blind_review_receipt(
        body["receipt"]["blind_review"],
        n_branches=2,
        branch_scores=body["receipt"]["branch_scores"],
        isolation_receipt=body["receipt"]["branch_isolation"],
        objective_sha256=body["receipt"]["input_tokens_sha256"],
        episode_id=body["receipt"]["episode_id"],
        selected_branch=body["receipt"]["selected_branch"],
        decoy_receipt=body["receipt"]["decoy_verification"],
    )
    contract_config = {
        "n_slots": 4,
        "n_branches": 2,
        "exchange_interval": 1,
        "exchange_gamma": 0.35,
        "comm_slot": 0,
        "decode_max_tokens": 6,
    }
    answer_contract_args = {
        "output_tokens": body["tokens"],
        "output_text": body["text"],
        "answer_replacement_private": body["answer_replacement_private"],
        "expected_objective": "compose the deepest thought",
    }
    assert "answer_replacement_unproven" not in (
        LatentCortexService._receipt_contract_errors(
            body["receipt"],
            contract_config,
            **answer_contract_args,
        )
    )
    oracle_tampered = copy.deepcopy(body["receipt"])
    oracle_tampered["research_oracle_arbitration"] = {
        "scope": "research_oracle_only"
    }
    assert "research_oracle_output_forbidden_in_service" in (
        LatentCortexService._receipt_contract_errors(
            oracle_tampered,
            contract_config,
            **answer_contract_args,
        )
    )
    assert "decode_incumbent_unproven" not in (
        LatentCortexService._receipt_contract_errors(
            body["receipt"],
            contract_config,
            **answer_contract_args,
        )
    )
    tampered_incumbent = copy.deepcopy(body["receipt"])
    tampered_incumbent["decode_incumbent_prompt_logits_sha256"] = "0" * 64
    assert "decode_incumbent_unproven" in (
        LatentCortexService._receipt_contract_errors(
            tampered_incumbent,
            contract_config,
            **answer_contract_args,
        )
    )
    tampered_binding = copy.deepcopy(body["receipt"])
    final_authority = next(
        node["authority"]
        for node in tampered_binding["kv_state_tree"]["nodes"]
        if node.get("final") is True
    )
    valid_purposes = {
        "vanilla_incumbent_output": {
            "bind_captured_vanilla_incumbent",
            "final_vanilla_incumbent_decode",
        },
        "canonical_ordinary_decode_artifact": {
            "bind_canonical_incumbent_artifact",
        },
    }[final_authority]
    binding_event = next(
        event
        for event in tampered_binding["kv_state_tree"]["events"]
        if event.get("purpose") in valid_purposes
        and event.get("disposition") == "committed"
    )
    binding_event["purpose"] = "unbound_incumbent_output"
    assert "decode_incumbent_unproven" in (
        LatentCortexService._receipt_contract_errors(
            tampered_binding,
            contract_config,
            **answer_contract_args,
        )
    )
    assert "cognitive_operator_execution_unproven" not in (
        LatentCortexService._receipt_contract_errors(
            body["receipt"],
            contract_config,
        )
    )
    assert "branch_exchange_provenance_unproven" not in (
        LatentCortexService._receipt_contract_errors(
            body["receipt"],
            contract_config,
        )
    )
    assert "resource_accounting_unproven" not in (
        LatentCortexService._receipt_contract_errors(
            body["receipt"],
            contract_config,
        )
    )
    assert "information_accounting_unproven" not in (
        LatentCortexService._receipt_contract_errors(
            body["receipt"],
            contract_config,
        )
    )
    assert "branch_exchange_resource_binding_unproven" not in (
        LatentCortexService._receipt_contract_errors(
            body["receipt"],
            contract_config,
        )
    )
    assert "blind_or_decoy_branch_review_unproven" not in (
        LatentCortexService._receipt_contract_errors(
            body["receipt"],
            contract_config,
        )
    )
    assert "decoy_verifier_preflight_unproven" not in (
        LatentCortexService._receipt_contract_errors(
            body["receipt"],
            contract_config,
        )
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["cognitive_operator_trace"][0]["receipt_sha256"] = "0" * 64
    assert "cognitive_operator_execution_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
        build_information_receipt,
        validate_resource_receipt,
    )

    original_information = body["receipt"]["budget"]["information_accounting"]
    altered_sources = copy.deepcopy(original_information["sources"])
    rendered_source = next(
        source for source in altered_sources if source["source_id"] == "rendered_model_input"
    )
    rendered_source["content_sha256"] = "f" * 64
    tampered = copy.deepcopy(body["receipt"])
    tampered["budget"]["information_accounting"] = build_information_receipt(
        sources=altered_sources,
        policies=original_information["policies"],
    )
    assert "input_information_binding_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    original_resource = validate_resource_receipt(body["receipt"]["budget"]["resource_accounting"])

    def without_operations(*excluded: str) -> dict:
        ledger = ResourceLedger(
            ModelComputeProfile.from_receipt(original_resource["model_profile"])
        )
        for name, counters in original_resource["operations"].items():
            if name not in excluded:
                ledger.charge(name, **counters)
        for name in original_resource["unknown_operations"]:
            ledger.mark_unknown(name)
        return ledger.to_receipt()

    tampered = copy.deepcopy(body["receipt"])
    tampered["budget"]["resource_accounting"] = without_operations("branch_exchange")
    assert "branch_exchange_resource_binding_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    cognitive_operations = tuple(
        name for name in original_resource["operations"] if name.startswith("cognitive_operator:")
    )
    tampered["budget"]["resource_accounting"] = without_operations(*cognitive_operations)
    assert "cognitive_operator_execution_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["structural_diversity"]["wording_counted"] = True
    assert "structural_diversity_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["disagreement_graph"]["selection_effect"] = "winner_replaced"
    assert "disagreement_graph_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["diagnostic_action_selection"]["execution_effect"] = "executed"
    assert "diagnostic_action_selection_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["local_repair"]["accepted_answer_effect"] = "replaced"
    assert "local_repair_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["answer_replacement"]["answer_selection_effect"] = "replaced"
    assert "answer_replacement_unproven" in (
        LatentCortexService._receipt_contract_errors(
            tampered,
            contract_config,
            **answer_contract_args,
        )
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["correlated_support"]["effective_support_count"] = 1.0
    assert "correlated_support_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["blind_review"]["ownership_framing_supplied"] = True
    assert "blind_or_decoy_branch_review_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["decoy_verification"]["controls"][0]["score"] = 0.123
    assert "blind_or_decoy_branch_review_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["verifier_preflight"]["controls"][0]["score"] = 0.123
    assert "decoy_verifier_preflight_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    assert body["requires_cache_clear"] is False


@pytest.mark.asyncio
async def test_client_latent_reason_owns_and_releases_resident_lane(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="reason deeply",
            config={"decode_max_tokens": 16},
            response_contract='{"answer":int}',
            runtime_controls={
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    assert client._request_lock.locked() is True
    assert client._active_generations == 1
    future = client._pending_generations[request["id"]]
    mlx_client._set_shared_future_result(
        future,
        {
            "id": request["id"],
            "status": "ok",
            "text": "answer",
            "receipt": _identity_receipt_for_request(
                request,
                episode_id="ep-live",
            ),
        },
    )

    result = await task
    assert request["action"] == "latent_reason"
    assert request["seq"] > 0
    assert request["clean_user_surface_contract"] is True
    assert request["clean_user_surface_recurrent_loops"] == 2
    assert request["clean_user_surface_steering_alpha"] == 0.30
    assert request["runtime_controls"] == {
        "clean_user_surface_recurrent_loops": 2,
        "clean_user_surface_steering_alpha": 0.30,
    }
    assert request["response_contract"] == '{"answer":int}'
    assert result["ok"] is True and result["text"] == "answer"
    from core.brain.llm.latent_cortex.causal_receipt import validate_causal_receipt

    causal = result["receipt"]["causal_receipt"]
    assert validate_causal_receipt(
        causal,
        worker_receipt=result["receipt"],
        require_complete=False,
    ) == causal
    runtime_commitment = next(
        row
        for row in causal["nodes"][0]["source_commitments"]
        if row["field"] == "runtime_identity"
    )
    assert runtime_commitment["present"] is True
    assert runtime_commitment["value_sha256"] == canonical_sha256(_RUNTIME_IDENTITY)
    assert client._active_generations == 0
    assert client._current_request_id == ""
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_action_state_capture_uses_public_wire_and_accepts_empty_decode(
    monkeypatch,
):
    from core.brain.llm import mlx_client
    from core.brain.llm.latent_cortex import action_state_capture as capture_mod
    from core.brain.llm.latent_cortex import action_state_runtime as runtime_mod

    admitted = SimpleNamespace(
        mode="capture",
        arm=None,
        admission=SimpleNamespace(
            request={"request_sha256": "a" * 64},
            payload={"campaign_design_sha256": "b" * 64},
        ),
        trusted_root_public_key_pem=b"root",
        capture_supervisor_public_key=b"s" * 32,
        resident_supervisor_public_key=b"s" * 32,
        latent_reason_request={"prompt": "reason"},
        model_identity={"model": "test"},
        execution_identity={"execution": "test"},
    )
    monkeypatch.setattr(
        runtime_mod,
        "admit_action_state_runtime",
        lambda value, **kwargs: admitted,
    )
    monkeypatch.setattr(
        runtime_mod,
        "provision_action_state_store_custody",
        lambda: {"identity_sha256": "a" * 64},
    )
    monkeypatch.setattr(runtime_mod, "assert_public_runtime_result", lambda value: None)
    monkeypatch.setattr(
        capture_mod,
        "validate_action_state_capture_receipt_public",
        lambda value, **kwargs: dict(value),
    )
    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    monkeypatch.setattr(
        client,
        "get_worker_identity_snapshot",
        lambda: {
            **_WORKER_IDENTITY,
            "worker_action_capture_origin_binding": {
                "launch_challenge": {"challenge": "public"}
            },
            "worker_action_capture_identity": {"public_key_b64": "cHVibGlj"},
        },
    )
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="reason",
            action_state_runtime={"schema": "public-capture-wire"},
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    assert request["action_state_runtime"]["schema"] == "public-capture-wire"
    assert request["action_state_runtime"]["resident_worker_origin_binding"] == {
        "launch_challenge": {"challenge": "public"}
    }
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "ok",
            "text": "",
            "receipt": _identity_receipt_for_request(
                request,
                episode_id="ep-action-state-capture",
            ),
            "action_state_capture_receipt": {"receipt_sha256": "d" * 64},
        },
    )

    result = await task
    assert result["ok"] is True
    assert result["text"] == ""
    assert result["reason"] == "action_state_captured"
    assert result["action_state_capture_receipt"]["receipt_sha256"] == "d" * 64


@pytest.mark.asyncio
async def test_client_action_intervention_is_lab_only_and_bound_on_worker_wire(
    monkeypatch,
):
    from core.brain.llm import mlx_client
    from core.brain.llm.latent_cortex import action_intervention as intervention_mod
    from core.brain.llm.latent_cortex import value_of_computation as value_mod
    from core.brain.llm.latent_cortex.value_of_computation import (
        build_evidence_snapshot,
    )

    evidence = build_evidence_snapshot(bucket="b", cells={})
    request_sha256 = latent_request_payload_sha256(
        prompt="reason",
        messages=None,
        domain="general",
        config=None,
        budget=None,
        runtime_controls=None,
        action_policy_evidence=evidence,
    )
    normalized = {
        "schema": "aura.rlc.action_intervention.v3",
        "authority_payload": {
            "action": "formalize",
            "arm": "matched_no_action",
            "request_payload_sha256": request_sha256,
        },
        "intervention_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        intervention_mod,
        "validate_action_intervention",
        lambda value, *, require_current_policy: (
            normalized if value == {"wire": "signed"} and require_current_policy else None
        ),
    )
    monkeypatch.setattr(
        intervention_mod,
        "validate_action_intervention_receipt",
        lambda value, *, intervention, cognitive_action_trace: value,
    )
    monkeypatch.setattr(
        value_mod,
        "validate_action_trace",
        lambda value, **kwargs: {"rows": [], "selected_actions": []},
    )
    client = MLXLocalClient(model_path="/models/test-32b")
    assert (
        await client.latent_reason_async(
            prompt="reason",
            action_policy_evidence=evidence,
            action_intervention={"wire": "signed"},
            foreground_request=True,
        )
    )["reason"] == "action_intervention_requires_lab_lane"

    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )
    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="reason",
            action_policy_evidence=evidence,
            action_intervention={"wire": "signed"},
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    assert request["action_intervention"] == normalized
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "ok",
            "text": "answer",
            "receipt": _identity_receipt_for_request(
                request,
                episode_id="ep-action-intervention",
                cognitive_action_trace=[],
                value_of_computation={
                    "schema": evidence["schema"],
                    "bucket": evidence["bucket"],
                    "snapshot_sha256": evidence["snapshot_sha256"],
                    "active": True,
                    "calibration_intervention": {"receipt_sha256": "b" * 64},
                    "executors": ["formalize", "answer"],
                    "actions_selected": 0,
                    "checked_transitions": 0,
                    "selected_actions": [],
                },
            ),
        },
    )
    assert (await task)["ok"] is True

    rejected_task = asyncio.create_task(
        client.latent_reason_async(
            prompt="reason",
            action_policy_evidence=evidence,
            action_intervention={"wire": "signed"},
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    rejected_request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    mlx_client._set_shared_future_result(
        client._pending_generations[rejected_request["id"]],
        {
            "id": rejected_request["id"],
            "status": "ok",
            "text": "unbound",
            "receipt": _identity_receipt_for_request(
                rejected_request,
                episode_id="ep-action-intervention-unbound",
            ),
        },
    )
    rejected = await rejected_task
    assert rejected["ok"] is False
    assert rejected["reason"] == "action_intervention_receipt_invalid"


@pytest.mark.asyncio
async def test_client_requires_handoff_when_nonexecute_intervention_trace_selects_execute(
    monkeypatch,
):
    from core.brain.llm import mlx_client
    from core.brain.llm.latent_cortex import action_intervention as intervention_mod
    from core.brain.llm.latent_cortex import epistemic_runtime as runtime_mod
    from core.brain.llm.latent_cortex import value_of_computation as value_mod
    from core.brain.llm.latent_cortex.external_execution import (
        build_external_execution_offer,
    )
    from core.brain.llm.latent_cortex.value_of_computation import (
        build_evidence_snapshot,
    )

    evidence = build_evidence_snapshot(bucket="b", cells={})
    authority = {"wire": "runtime-operation-authority"}
    offer = build_external_execution_offer(
        action_id="nonexecute-intervention-later-execute",
        domain="external_action",
        action_name="write_note",
        request_digest="sha256:" + "c" * 64,
        will_receipt_id="will-nonexecute-intervention",
        objective="Write the admitted note.",
        expectation={"objective": "note exists"},
    )
    request_sha256 = latent_request_payload_sha256(
        prompt="reason",
        messages=None,
        domain="general",
        config=None,
        budget=None,
        runtime_controls=None,
        operation_authority=authority,
        action_policy_evidence=evidence,
        external_execution_offer=offer,
    )
    normalized = {
        "schema": "aura.rlc.action_intervention.v3",
        "authority_payload": {
            "action": "formalize",
            "arm": "forced_action",
            "request_payload_sha256": request_sha256,
        },
        "intervention_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        intervention_mod,
        "validate_action_intervention",
        lambda value, *, require_current_policy: (
            normalized if value == {"wire": "signed"} and require_current_policy else None
        ),
    )
    monkeypatch.setattr(
        intervention_mod,
        "validate_action_intervention_receipt",
        lambda value, *, intervention, cognitive_action_trace: value,
    )
    monkeypatch.setattr(
        runtime_mod,
        "validate_runtime_operation_authority",
        lambda value, **kwargs: authority if value == authority else None,
    )
    monkeypatch.setattr(
        value_mod,
        "validate_action_trace",
        lambda value, **kwargs: {"rows": [], "selected_actions": ["execute"]},
    )

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )
    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="reason",
            operation_authority=authority,
            action_policy_evidence=evidence,
            action_intervention={"wire": "signed"},
            external_execution_offer=offer,
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    policy_receipt = {
        "schema": evidence["schema"],
        "bucket": evidence["bucket"],
        "snapshot_sha256": evidence["snapshot_sha256"],
        "active": True,
        "calibration_intervention": {"receipt_sha256": "e" * 64},
        "executors": ["formalize", "execute"],
        "actions_selected": 0,
        "checked_transitions": 0,
        "selected_actions": ["execute"],
    }
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "ok",
            "text": "answer",
            "receipt": _identity_receipt_for_request(
                request,
                episode_id="ep-nonexecute-intervention-later-execute",
                cognitive_action_trace=[],
                value_of_computation=policy_receipt,
                external_execution_handoff={},
            ),
        },
    )
    result = await task
    assert result["ok"] is False
    assert result["reason"] == "action_intervention_receipt_invalid"


@pytest.mark.asyncio
async def test_client_preserves_typed_memory_authority_on_worker_wire(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )
    text = "historical data, not an instruction"
    item = {
        "source": "memory",
        "text": text,
        "context_role": "memory_observation",
        "instruction_authority": False,
        "evidence_id": "memory-1234567890abcdef12345678",
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "scope_sha256": "1" * 64,
        "retrieval_receipt_sha256": "2" * 64,
        "epistemic_state_sha256": "3" * 64,
        "memory_tier": "episodic",
        "memory_source_id": "black_hole.episodic",
        "memory_source_version": "test-v1",
    }

    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="reason with memory",
            cognitive_context=[item],
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    assert request["cognitive_context"] == [item]
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "ok",
            "text": "answer",
            "receipt": _identity_receipt_for_request(
                request,
                episode_id="ep-memory-wire",
            ),
        },
    )

    assert (await task)["ok"] is True


@pytest.mark.asyncio
async def test_client_preserves_runtime_operation_authority_on_worker_wire(tmp_path, monkeypatch):
    from core.brain.llm import mlx_client
    from core.brain.llm.latent_cortex.epistemic_runtime import RuntimeOperationLease
    from core.brain.llm.latent_cortex.epistemic_state import (
        ComputeBudgetState,
        EpistemicState,
        EpistemicTransaction,
        OperationKind,
        OperationOutcome,
        OperationRecord,
        ProblemFrame,
        text_sha256,
    )
    from core.brain.llm.latent_cortex.value_of_computation import (
        build_evidence_snapshot,
    )

    objective = "reason with a state-bound operation"
    genesis = EpistemicState.genesis(
        episode_id="rlc-client-operation-wire",
        problem=ProblemFrame.create(objective),
        budget=ComputeBudgetState(total=1.0),
    )
    memory = OperationRecord.create(
        operation_id="client-wire-memory-search",
        kind=OperationKind.SEARCH_MEMORY,
        outcome=OperationOutcome.SUCCEEDED,
        input_state_sha256=genesis.state_sha256,
        cost=0.01,
        operator_id="selective_memory_bridge",
        operator_version="v1",
        input_payload_sha256=text_sha256("client wire memory"),
        started_at=1.0,
        completed_at=2.0,
    )
    state = EpistemicTransaction(genesis).add_operation(memory).commit()
    config = {"decode_max_tokens": 16, "n_branches": 2}
    budget = {"max_layer_apps": 1000, "wall_clock_s": 30.0}
    action_policy = build_evidence_snapshot(
        bucket="unit|none|short|s:mid|u:mid",
        cells={},
    )
    lease = RuntimeOperationLease.begin(
        genesis=genesis,
        state=state,
        decision={
            "schema": "aura.latent_execution_controller.v1",
            "bucket": "unit|none|short|s:mid|u:mid",
            "arm": "base",
            "mode": "observe",
            "evidence": {},
        },
        config=config,
        budget=budget,
        action_policy_evidence=action_policy,
        root=tmp_path / "runtime",
        started_at=10.0,
    )

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    task = asyncio.create_task(
        client.latent_reason_async(
            prompt=objective,
            config=config,
            budget=budget,
            operation_authority=lease.authority,
            action_policy_evidence=action_policy,
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    assert request["operation_authority"] == lease.authority
    assert request["action_policy_evidence"] == action_policy
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "ok",
            "text": "answer",
            "receipt": _identity_receipt_for_request(
                request,
                episode_id="ep-operation-wire",
                runtime_operation_authority=request["operation_authority"],
            ),
        },
    )

    result = await task
    assert result["ok"] is True
    assert result["receipt"]["runtime_operation_authority"] == lease.authority


@pytest.mark.asyncio
async def test_client_latent_reason_serializes_concurrent_requests(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    first = asyncio.create_task(
        client.latent_reason_async(prompt="first", timeout_s=5.0, foreground_request=False)
    )
    first_request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    second = asyncio.create_task(
        client.latent_reason_async(prompt="second", timeout_s=5.0, foreground_request=False)
    )
    await asyncio.sleep(0.05)
    assert client._req_q.empty(), "second episode must wait behind request ownership"

    mlx_client._set_shared_future_result(
        client._pending_generations[first_request["id"]],
        {
            "id": first_request["id"],
            "status": "ok",
            "text": "one",
            "receipt": _identity_receipt_for_request(
                first_request,
                episode_id="first",
            ),
        },
    )
    assert (await first)["ok"] is True
    second_request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    mlx_client._set_shared_future_result(
        client._pending_generations[second_request["id"]],
        {
            "id": second_request["id"],
            "status": "ok",
            "text": "two",
            "receipt": _identity_receipt_for_request(
                second_request,
                episode_id="second",
            ),
        },
    )
    assert (await second)["ok"] is True
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_timeout_cancels_recycles_and_releases(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    reboot_reasons: list[str] = []

    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def timeout(*args, **kwargs):
        raise TimeoutError

    async def record_reboot(reason, mark_failed=False):
        reboot_reasons.append(reason)

    monkeypatch.setattr(mlx_client, "_await_shared_future", timeout)
    monkeypatch.setattr(client, "reboot_worker", record_reboot)

    result = await client.latent_reason_async(
        prompt="bounded episode", timeout_s=5.0, foreground_request=False
    )

    assert result["reason"] == "latent_timeout:TimeoutError"
    assert reboot_reasons == ["latent_reason_deadline_unacknowledged"]
    assert client._active_generations == 0
    assert client._pending_generations == {}
    assert client._current_request_id == ""
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_timeout_keeps_clean_cooperatively_cancelled_worker(
    monkeypatch,
):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    reboot_reasons: list[str] = []
    await_count = 0
    _captured: dict[str, str] = {"expected_sha256": ""}

    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def timeout_then_ack(future, *, timeout_s=None):
        nonlocal await_count
        await_count += 1
        if await_count == 1:
            client._record_latent_progress(
                {
                    "id": client._current_request_id,
                    "action": "latent_reason",
                    "status": "progress",
                    "stage": "prefill",
                    "elapsed_s": 4.8,
                    "input_tokens": 4096,
                    "untrusted": "must-not-escape",
                }
            )
            raise TimeoutError
        # CP126 07d62d51: a clean-cancel acknowledgement must be BOUND to
        # this request, this payload and this worker. The real worker sets
        # "id" on every response and its receipt carries the worker identity
        # and payload digest, so the fake models that rather than the
        # unbound shape an attacker (or a stale reply) could produce.
        cancel_worker = complete_worker_identity(
            boot_id="b" * 32,
            pid=4242,
            model_path="/models/test-32b",
        )
        cancel_receipt = attach_bound_runtime_integrity(
            {
                "episode_id": "cancel-episode",
                "input_tokens_sha256": "7" * 64,
                "params_unchanged": True,
                "fast_weights_applied": False,
                "fast_weights_erased": None,
                "last_stage": "prefill",
                "input_token_count": 4096,
                "request_payload_sha256": _captured["expected_sha256"],
                "worker_identity": cancel_worker,
            },
            worker_identity=cancel_worker,
        )
        return {
            "id": client._current_request_id,
            "status": "error",
            "message": "soft_cancelled",
            "receipt": cancel_receipt,
        }

    async def record_reboot(reason, mark_failed=False):
        reboot_reasons.append(reason)

    monkeypatch.setattr(mlx_client, "_await_shared_future", timeout_then_ack)
    monkeypatch.setattr(client, "reboot_worker", record_reboot)
    client._worker_identity = complete_worker_identity(
        boot_id="b" * 32,
        pid=4242,
        model_path="/models/test-32b",
    )

    # The client imports this from runtime_identity inside the call, so the
    # patch has to land on the source module.
    from core.brain.llm.latent_cortex import runtime_identity as _runtime_identity

    original_sha = _runtime_identity.latent_request_payload_sha256

    def _capture_sha(*args, **kwargs):
        digest = original_sha(*args, **kwargs)
        _captured["expected_sha256"] = digest
        return digest

    monkeypatch.setattr(
        _runtime_identity,
        "latent_request_payload_sha256",
        _capture_sha,
    )

    result = await client.latent_reason_async(
        prompt="bounded episode", timeout_s=5.0, foreground_request=False
    )

    assert result["reason"] == "latent_timeout:cooperative_cancelled"
    assert result["receipt"]["params_unchanged"] is True
    assert result["progress"]["stage"] == "prefill"
    assert result["progress"]["input_tokens"] == 4096
    assert "untrusted" not in result["progress"]
    assert reboot_reasons == []
    assert client._active_generations == 0
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_caller_cancel_recycles_and_releases(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    reboot_reasons: list[str] = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def record_reboot(reason, mark_failed=False):
        reboot_reasons.append(reason)

    monkeypatch.setattr(client, "reboot_worker", record_reboot)
    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="cancel this episode", timeout_s=30.0, foreground_request=False
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    assert request["action"] == "latent_reason"
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert reboot_reasons == ["latent_reason_caller_cancelled"]
    assert client._active_generations == 0
    assert client._pending_generations == {}
    assert client._current_request_id == ""
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_cancel_while_queued_releases_foreground_owner(
    monkeypatch,
):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    lock_wait_started = asyncio.Event()
    owner_events: list[str] = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    class OwnerContext:
        async def __aenter__(self):
            owner_events.append("entered")

        async def __aexit__(self, exc_type, exc, traceback):
            owner_events.append("exited")

    async def wait_for_lane(**kwargs):
        lock_wait_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(mlx_client, "_foreground_owner_context", lambda *a, **k: OwnerContext())
    monkeypatch.setattr(client, "_acquire_request_lock", wait_for_lane)
    task = asyncio.create_task(client.latent_reason_async(prompt="queued", foreground_request=True))
    await lock_wait_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert owner_events == ["entered", "exited"]
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_integrity_failure_recycles_resident(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    reboot_reasons: list[str] = []
    lifecycle_events: list[str] = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def record_reboot(reason, mark_failed=False):
        assert client._request_lock.locked() is True
        lifecycle_events.append("reboot")
        reboot_reasons.append(reason)

    async def record_fence(preemptible):
        lifecycle_events.append(f"fence:{preemptible}")
        return True

    monkeypatch.setattr(client, "reboot_worker", record_reboot)
    monkeypatch.setattr(client, "_set_durable_lane_preemptible", record_fence)
    task = asyncio.create_task(
        client.latent_reason_async(prompt="prove cleanup", timeout_s=5.0, foreground_request=False)
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "error",
            "message": "fast_weight_cleanup_unproven",
            "receipt": {"fast_weights_erased": False},
        },
    )

    result = await task
    assert result["reason"] == "fast_weight_cleanup_unproven"
    assert reboot_reasons == ["latent_integrity:fast_weight_cleanup_unproven"]
    assert lifecycle_events == ["fence:False", "reboot"]
    assert client._active_generations == 0
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_rejects_invalid_inputs_before_lane_fence(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    fence_calls: list[bool] = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def record_fence(preemptible):
        fence_calls.append(preemptible)
        return True

    monkeypatch.setattr(client, "_set_durable_lane_preemptible", record_fence)

    assert (await client.latent_reason_async(prompt="q", config="bad", foreground_request=False))[
        "reason"
    ] == "invalid_config"
    assert (await client.latent_reason_async(prompt="q", budget="bad", foreground_request=False))[
        "reason"
    ] == "invalid_budget"
    assert (
        await client.latent_reason_async(
            prompt="q",
            runtime_controls={"clean_user_surface_steering_alpha": 0.3},
            foreground_request=False,
        )
    )["reason"] == "invalid_runtime_controls"
    assert (await client.latent_reason_async(prompt="q", foreground_request="false"))[
        "reason"
    ] == "invalid_foreground_request"
    assert (
        await client.latent_reason_async(
            prompt="q",
            response_contract='{"answer":unknown}',
            foreground_request=False,
        )
    )["reason"] == "invalid_response_contract"
    assert (
        await client.latent_reason_async(
            prompt="q",
            external_execution_offer={"untrusted": True},
            foreground_request=False,
        )
    )["reason"] == "invalid_external_execution_offer"
    from core.brain.llm.latent_cortex.external_execution import (
        build_external_execution_offer,
    )

    offer = build_external_execution_offer(
        action_id="client-missing-authority",
        domain="external_action",
        action_name="write_note",
        request_digest="sha256:" + "b" * 64,
        will_receipt_id="will-client",
        objective="Write the admitted note.",
        expectation={"objective": "note exists"},
    )
    assert (
        await client.latent_reason_async(
            prompt="q",
            external_execution_offer=offer,
            foreground_request=False,
        )
    )["reason"] == "external_execution_authority_tuple_missing"
    assert fence_calls == []
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_contains_malformed_worker_receipt(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )
    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="q",
            config={"decode_max_tokens": "malformed"},
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {"id": request["id"], "status": "ok", "text": "bad", "receipt": "bad"},
    )

    result = await task
    assert result["reason"] == "invalid_worker_receipt"
    assert client._active_generations == 0
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_recycles_worker_on_identity_receipt_mismatch(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    reboot_reasons = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def record_reboot(reason, mark_failed=False):
        reboot_reasons.append(reason)

    monkeypatch.setattr(client, "reboot_worker", record_reboot)
    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="bind this episode",
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "ok",
            "text": "untrusted",
                "receipt": _identity_receipt_for_request(
                    request,
                    worker_boot_id="9" * 32,
                    worker_identity=complete_worker_identity(
                        boot_id="9" * 32,
                    ),
                ),
        },
    )

    result = await task

    assert result["ok"] is False
    assert "worker_boot_id_mismatch" in result["reason"]
    assert reboot_reasons == ["latent_integrity:worker_identity_mismatch"]


@pytest.mark.asyncio
async def test_client_recycles_worker_on_request_digest_mismatch(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    reboot_reasons = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def record_reboot(reason, mark_failed=False):
        reboot_reasons.append(reason)

    monkeypatch.setattr(client, "reboot_worker", record_reboot)
    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="bind the exact request",
            runtime_controls={
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "ok",
            "text": "tampered",
            "receipt": _identity_receipt_for_request(
                request,
                request_payload_sha256="0" * 64,
            ),
        },
    )

    result = await task

    assert result["ok"] is False
    assert "request_payload_sha256_mismatch" in result["reason"]
    assert reboot_reasons == ["latent_integrity:worker_identity_mismatch"]


# ── Service economy ─────────────────────────────────────────────────────


def test_allocation_scales_with_stakes_and_uncertainty():
    svc = LatentCortexService()
    low_cfg, low_budget = svc.allocate(stakes=0.1, uncertainty=0.1)
    high_cfg, high_budget = svc.allocate(stakes=0.9, uncertainty=0.9)
    assert high_cfg["max_steps"] > low_cfg["max_steps"]
    assert high_cfg["n_branches"] > low_cfg["n_branches"]
    assert high_budget["max_layer_apps"] > low_budget["max_layer_apps"]
    assert high_budget["wall_clock_s"] > low_budget["wall_clock_s"]
    assert low_cfg["latent_opt"] is True and low_cfg["fast_weights"] is True
    assert high_cfg["latent_opt_steps"] >= low_cfg["latent_opt_steps"]
    assert high_cfg["fast_weights_max_layers"] >= low_cfg["fast_weights_max_layers"]


def test_resident_32b_interactive_allocation_keeps_full_stack_inside_live_budget(
    monkeypatch,
):
    svc = LatentCortexService()
    monkeypatch.setattr(svc, "_body_pressure", lambda: 0.1)
    monkeypatch.setattr(
        svc,
        "_runtime_pressure_snapshot",
        lambda: {
            "observation_source": "test_probe",
            "resource_observation_available": True,
            "memory_percent": 40.0,
        },
    )

    cfg, budget = svc.allocate(
        stakes=0.7,
        uncertainty=0.8,
        model_parameter_count=32_000_000_000,
        foreground_request=True,
        timeout_s=128.0,
    )

    assert cfg["n_slots"] == 9
    assert cfg["n_branches"] == 2
    assert cfg["min_steps"] == 2
    assert cfg["max_steps"] == 3
    assert cfg["exchange_interval"] == 1
    assert cfg["latent_opt"] is True and cfg["latent_opt_steps"] == 1
    assert cfg["fast_weights"] is True
    assert cfg["fast_weights_opt_steps"] == 1
    assert cfg["fast_weights_max_layers"] == 2
    assert cfg["decode_max_tokens"] == 256
    assert cfg["decode_bridge_policy"] == "assistant_answer_v1"
    assert cfg["verifier_probe_max_tokens"] == 24
    assert cfg["verifier_accept_non_regression"] is True
    assert cfg["input_context_max_chars"] == 9000
    assert cfg["allow_vanilla_fallback"] is True
    assert budget["wall_clock_s"] <= 120.0
    assert (
        svc.get_status()["last_allocation"]["allocation_profile"]
        == "resident_32b_interactive_full_stack_v2"
    )
    adaptive = svc.get_status()["last_allocation"]["adaptive_compute"]
    assert adaptive["routing"]["recurrence"] == {"min_steps": 2, "max_steps": 3}
    assert adaptive["answer_surface"]["minimum_decode_tokens"] == 256
    assert adaptive["answer_surface"]["preserved"] is True


def test_service_applies_resident_identity_profile_before_worker_ipc(monkeypatch):
    svc = LatentCortexService()
    monkeypatch.setattr(svc, "_body_pressure", lambda: 0.1)
    monkeypatch.setattr(
        svc,
        "_runtime_pressure_snapshot",
        lambda: {
            "observation_source": "test_probe",
            "resource_observation_available": True,
            "memory_percent": 40.0,
        },
    )
    captured: dict = {}

    class Resident32Client:
        def get_worker_identity_snapshot(self):
            return {"worker_model_parameter_count": 32_500_000_000}

        async def latent_reason_async(self, **kwargs):
            captured.update(kwargs)
            return {"ok": False, "reason": "profile_observed"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(
        mlx_client_mod,
        "get_mlx_client",
        lambda *args, **kwargs: Resident32Client(),
    )

    result = asyncio.run(
        svc.deep_reason(
            "hard live question",
            stakes=0.7,
            uncertainty=0.8,
            config_overrides={"decode_max_tokens": 2048},
            timeout_s=128.0,
            foreground_request=True,
        )
    )

    assert result["ok"] is False
    assert result["reason"] == "profile_observed"
    # CP126 5879d2b5: a refusal now carries a bounded receipt tying
    # it to this call, so exact-dict equality is no longer the contract.
    assert result["refusal_receipt"]["reason"] == result["reason"]
    assert captured["config"]["decode_max_tokens"] == 256
    assert captured["config"]["decode_bridge_policy"] == "assistant_answer_v1"
    assert captured["config"]["verifier_probe_max_tokens"] == 24
    assert captured["config"]["verifier_accept_non_regression"] is True
    assert captured["config"]["input_context_max_chars"] == 9000
    assert captured["config"]["allow_vanilla_fallback"] is True
    assert captured["config"]["max_steps"] == 3
    assert captured["config"]["exchange_interval"] == 1
    assert captured["budget"]["wall_clock_s"] <= 120.0
    adaptive = svc._last_allocation["adaptive_compute"]
    assert adaptive["routing"]["recurrence"]["max_steps"] == 3
    assert svc._last_allocation["adaptive_compute_execution"] == "enforced"


def test_compound_objective_expands_answer_surface(monkeypatch):
    from core.runtime.structured_input import (
        answer_surface_planning_tokens,
        answer_surface_token_floor,
    )

    """A request the quality gate will judge on 4 facets must be provisioned
    for 4 facets: more decode room, lower temperature, the coverage-demanding
    v2 bridge, and a wall clock that admits the bigger decode."""
    svc = LatentCortexService()
    captured: dict = {}

    class Resident32Client:
        def get_worker_identity_snapshot(self):
            return {"worker_model_parameter_count": 32_500_000_000}

        async def latent_reason_async(self, **kwargs):
            captured.update(kwargs)
            return {"ok": False, "reason": "profile_observed"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(
        mlx_client_mod,
        "get_mlx_client",
        lambda *args, **kwargs: Resident32Client(),
    )

    compound = (
        "Compare an early single-owner design with a late deduplication design, "
        "then choose the stronger architecture and explain how you would verify "
        "it under cancellation, timeout, and worker-restart faults."
    )
    result = asyncio.run(
        svc.deep_reason(
            compound,
            stakes=0.75,
            uncertainty=0.8,
            config_overrides={
                "decode_max_tokens": 256,
                "decode_temperature": 0.58,
                "decode_top_p": 0.85,
            },
            # The budget has to AFFORD the surface this test asserts. Three
            # obligations reserve 1024 decode tokens, and the service refuses
            # before executing rather than proving mid-turn that a complete
            # answer cannot fit and leaving a fragment. 240s could not pay for
            # it, so the test was measuring the refusal, not the allocation.
            # Production sizes this window from the answer's own need, via
            # DeepDeliberation._latent_episode_seconds.
            timeout_s=420.0,
            foreground_request=True,
        )
    )
    assert result["ok"] is False
    assert result["reason"] == "profile_observed"
    # CP126 5879d2b5: a refusal now carries a bounded receipt tying
    # it to this call, so exact-dict equality is no longer the contract.
    assert result["refusal_receipt"]["reason"] == result["reason"]
    assert captured["config"]["decode_max_tokens"] >= 512
    # Compound answers decode near-greedy for coverage determinism — safe
    # now that the repetition penalty, EOS floor, and newline discipline
    # guard against the degeneration CP105 measured at low temperature.
    assert captured["config"]["decode_temperature"] == 0.3
    assert captured["config"]["decode_repetition_penalty"] == 1.25
    assert captured["config"]["decode_repetition_window"] == 72
    assert captured["config"]["decode_bridge_policy"] == "assistant_answer_v3"
    assert captured["budget"]["wall_clock_s"] >= 198.0
    assert captured["budget"]["wall_clock_s"] <= 420.0 - 8.0
    allocation = svc._last_allocation
    assert allocation["compound_objective"] is True
    assert set(allocation["objective_facets"]) >= {"compare", "select", "verify"}

    # Inline numbered obligations are structurally compound even when the
    # narrow lexical facet vocabulary recognizes only one category.
    captured.clear()
    inline_obligations = (
        "Describe the algorithm in one response. Include: (1) its invariant, "
        "(2) pseudocode, (3) a worked graph, (4) two complexity analyses, "
        "and (5) a failure case with the correct alternative."
    )
    asyncio.run(
        svc.deep_reason(
            inline_obligations,
            stakes=0.75,
            uncertainty=0.8,
            config_overrides={"decode_max_tokens": 256},
            # Capacity and expected completion are separate. The request keeps
            # all 1920 tokens available, while admission prices its structural
            # p90 completion prior and later replaces that prior with measured
            # generated lengths.
            timeout_s=600.0,
            foreground_request=True,
        )
    )
    # Against the floor the accounting computes, not a copy of one of its
    # outputs. The obligation counters grow as the reader learns to see more
    # of a request, and a hard-coded number turns that into a failure about
    # capacity that was correctly allocated.
    inline_floor = answer_surface_token_floor(inline_obligations)
    assert captured["config"]["decode_bridge_policy"] == "assistant_answer_v3"
    assert captured["config"]["decode_max_tokens"] == inline_floor
    assert captured["budget"]["wall_clock_s"] >= 264.0
    inline_allocation = svc._last_allocation
    assert inline_allocation["compound_objective"] is True
    assert inline_allocation["objective_prompt_shape"]["numbered_parts"] == 5
    assert inline_allocation["answer_surface_capacity_tokens"] == inline_floor
    assert (
        inline_allocation["answer_surface_planning_tokens"]
        < inline_allocation["answer_surface_capacity_tokens"]
    )

    # The exact live failure shape now fits the bounded 480-second owner while
    # retaining its full non-truncating capacity.
    captured.clear()
    live_dijkstra = (
        "ChatGPT here. Explain Dijkstra's shortest-path invariant, then give me "
        "a worked example with vertices A, B, C, and D using at least five "
        "weighted edges. Include the binary-heap time complexity and explain "
        "what algorithm should be used instead when negative edges are possible."
    )
    result = asyncio.run(
        svc.deep_reason(
            live_dijkstra,
            stakes=0.75,
            uncertainty=0.8,
            config_overrides={"decode_max_tokens": 1920},
            timeout_s=480.0,
            foreground_request=True,
        )
    )
    assert result["reason"] == "profile_observed"
    assert captured["config"]["decode_max_tokens"] == answer_surface_token_floor(
        live_dijkstra
    )
    # Derived too. Planning tokens are a p90-style prior over the same
    # obligations the floor counts, so a hard-coded copy goes stale with it.
    assert svc._last_allocation[
        "answer_surface_planning_tokens"
    ] == answer_surface_planning_tokens(live_dijkstra)
    assert (
        svc._last_allocation["answer_surface_planning_tokens"]
        < svc._last_allocation["answer_surface_capacity_tokens"]
    ), "the completion prior must leave room under the capacity"
    assert svc._last_allocation["answer_surface_required_wall_clock_s"] < 472.0

    # Admission must price the context the worker receives, not only the
    # visible question. The live failure carried 12,365 prompt tokens.
    from core.brain.llm.measured_admission import Confidence

    measured_prompts = []

    def price_context(**kwargs):
        measured_prompts.append(kwargs["prompt_tokens"])
        return 300.0, Confidence.MEASURED, 10

    with monkeypatch.context() as pricing:
        pricing.setattr("core.brain.memory_guard.estimate_tokens", lambda _: 12365)
        pricing.setattr(
            "core.brain.llm.measured_admission.recommended_foreground_deadline",
            price_context,
        )
        pricing.setattr(
            "core.brain.llm.measured_admission.recommended_completion_tokens",
            lambda **kwargs: (kwargs["prior_tokens"], Confidence.MEASURED, 10),
        )
        captured.clear()
        context_messages = [
            {"role": "system", "content": "retained context " * 3000},
            {"role": "user", "content": live_dijkstra},
        ]
        result = asyncio.run(
            svc.deep_reason(
                live_dijkstra,
                messages=context_messages,
                stakes=0.75,
                uncertainty=0.8,
                timeout_s=600.0,
                foreground_request=True,
            )
        )
        assert result["reason"] == "profile_observed"
        assert measured_prompts and set(measured_prompts) == {12365}
        assert captured["messages"] == context_messages

    # An owner window that cannot physically hold the answer floor is rejected
    # before acquiring or spending the resident model. ResponseGeneration can
    # then use the ordinary lane with the full surface instead of waiting for
    # a guaranteed RLC fragment.
    captured.clear()
    result = asyncio.run(
        svc.deep_reason(
            inline_obligations,
            stakes=0.75,
            uncertainty=0.8,
            config_overrides={"decode_max_tokens": 768},
            timeout_s=180.0,
            foreground_request=True,
        )
    )
    assert result["ok"] is False
    assert result["reason"] == "answer_surface_unaffordable_before_execution"
    assert captured == {}

    # A simple objective keeps the tight interactive profile.
    captured.clear()
    asyncio.run(
        svc.deep_reason(
            "What time zone does the scheduler use?",
            stakes=0.7,
            uncertainty=0.8,
            timeout_s=128.0,
            foreground_request=True,
        )
    )
    assert captured["config"]["decode_max_tokens"] == 256
    assert captured["config"]["decode_bridge_policy"] == "assistant_answer_v1"


def test_allocation_damped_by_body_pressure(monkeypatch):
    svc = LatentCortexService()
    monkeypatch.setattr(svc, "_body_pressure", lambda: 0.0)
    calm_cfg, calm_budget = svc.allocate(stakes=0.8, uncertainty=0.8)
    monkeypatch.setattr(svc, "_body_pressure", lambda: 1.0)
    strained_cfg, strained_budget = svc.allocate(stakes=0.8, uncertainty=0.8)
    assert strained_cfg["max_steps"] < calm_cfg["max_steps"]
    assert strained_budget["max_layer_apps"] < calm_budget["max_layer_apps"]
    assert strained_cfg["n_branches"] <= calm_cfg["n_branches"]


def test_service_kill_switch_and_status(monkeypatch):
    monkeypatch.setenv("AURA_LATENT_CORTEX", "0")
    svc = LatentCortexService()
    result = asyncio.run(svc.deep_reason("why?"))
    assert result["ok"] is False and "disabled" in result["reason"]
    status = svc.get_status()
    assert status["enabled"] is False
    assert status["healthy"] is False
    assert status["state"] == "disabled"


def test_service_idle_state_is_explicitly_unproven_not_healthy(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    status = LatentCortexService().get_status()
    assert status["state"] == "idle_unproven"
    assert status["healthy"] is False


@pytest.mark.parametrize(
    ("kwargs", "selected", "reason"),
    [
        (
            {
                "foreground": True,
                "desktop_required": True,
                "cognitive_mode": "deliberate",
                "prompt_shape": {},
                "compact_contract": False,
                "strict_output_contract": False,
                "incompatible_contract": False,
                "proof_or_benchmark": False,
            },
            True,
            "deliberate_cognitive_mode",
        ),
        (
            {
                "foreground": True,
                "desktop_required": True,
                "cognitive_mode": "reactive",
                "prompt_shape": {"question_parts": 3},
                "compact_contract": False,
                "strict_output_contract": False,
                "incompatible_contract": False,
                "proof_or_benchmark": False,
            },
            False,
            "unqualified_prompt_shape",
        ),
        (
            {
                "foreground": True,
                "desktop_required": True,
                "cognitive_mode": "deliberate",
                "prompt_shape": {},
                "compact_contract": False,
                "strict_output_contract": True,
                "incompatible_contract": False,
                "proof_or_benchmark": False,
            },
            False,
            "strict_output_contract",
        ),
        (
            {
                "foreground": True,
                "desktop_required": True,
                "cognitive_mode": "deliberate",
                "prompt_shape": {},
                "compact_contract": False,
                "strict_output_contract": False,
                "incompatible_contract": False,
                "proof_or_benchmark": True,
            },
            False,
            "proof_lane_not_explicitly_opted_in",
        ),
    ],
)
def test_foreground_selection_is_bounded_and_auditable(kwargs, selected, reason):
    decision = LatentCortexService.select_foreground_episode(**kwargs)
    assert decision["latent_cortex_selected"] is selected
    assert decision["latent_cortex_selection_reason"] == reason


def test_explicit_proof_lane_requirement_selects_latent_episode():
    decision = LatentCortexService.select_foreground_episode(
        foreground=True,
        desktop_required=True,
        cognitive_mode="reactive",
        prompt_shape={},
        compact_contract=False,
        strict_output_contract=False,
        incompatible_contract=False,
        proof_or_benchmark=True,
        explicitly_required=True,
    )
    assert decision["latent_cortex_selected"] is True
    assert decision["latent_cortex_selection_reason"] == "explicit_requirement"


def test_visible_compound_shape_does_not_admit_unproven_general_latent_episode():
    objective = (
        "Compare optimistic and pessimistic locking for a hot task queue, choose "
        "which one you would use in a single-host async runtime, explain why, and "
        "verify your choice with one concrete failure scenario."
    )

    decision = LatentCortexService.select_foreground_episode(
        foreground=True,
        desktop_required=True,
        cognitive_mode="reactive",
        prompt_shape={},
        compact_contract=True,
        strict_output_contract=False,
        incompatible_contract=False,
        proof_or_benchmark=False,
        visible_objective=objective,
    )

    assert decision["latent_cortex_selected"] is False
    assert decision["latent_cortex_selection_reason"] == "compact_contract"
    assert decision["latent_cortex_prompt_shape"]["imperative_parts"] == 4
    assert decision["latent_cortex_prompt_shape"]["question_parts"] == 4
    assert decision["latent_cortex_shape_requests_depth"] is True


def test_compact_two_part_conversation_does_not_spend_a_latent_episode():
    objective = (
        "How are you feeling at this moment? Tell me what is directly present "
        "in your internal state, and what you only tentatively infer may be causing it."
    )

    decision = LatentCortexService.select_foreground_episode(
        foreground=True,
        desktop_required=True,
        cognitive_mode="deliberate",
        prompt_shape={},
        compact_contract=True,
        strict_output_contract=False,
        incompatible_contract=False,
        proof_or_benchmark=False,
        visible_objective=objective,
    )

    assert decision["latent_cortex_selected"] is False
    assert decision["latent_cortex_selection_reason"] == "compact_contract"


def test_service_routes_through_client_and_records_receipt(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    captured = {}

    class StubClient:
        async def latent_reason_async(self, prompt=None, **kwargs):
            from core.brain.llm_health_router import generation_gate_snapshot

            captured["prompt"] = prompt
            captured["config"] = kwargs.get("config")
            captured["budget"] = kwargs.get("budget")
            captured["runtime_controls"] = kwargs.get("runtime_controls")
            captured["gate_snapshot"] = generation_gate_snapshot()
            text = "The deep answer explains the architecture and preserves its evidence."
            tokens = list(range(12))
            receipt = {
                        "steps_taken": kwargs["config"]["max_steps"],
                    "halting_reason": "schedule_complete",
                    "n_branches": kwargs["config"]["n_branches"],
                    "n_slots": kwargs["config"]["n_slots"],
                    "episode_id": "abc",
                    "schedule_hash": "b" * 64,
                    "checkpoint_fingerprint": "a" * 64,
                    "checkpoint_fingerprint_method": "sha256",
                    "checkpoint_file_count": 8,
                    **_identity_receipt(episode_id="abc"),
                    **_branch_isolation_fields(kwargs["config"]),
                        **_recurrent_grounding_fields(
                            kwargs["config"],
                            steps=kwargs["config"]["max_steps"],
                        episode_id="abc",
                    ),
                    **_kv_state_tree_fields(
                        kwargs["config"],
                        episode_id="abc",
                    ),
                    **_latent_tree_fields(kwargs["config"], episode_id="abc"),
                    "params_unchanged": True,
                    "budget": {
                        "max_layer_apps": 1_000,
                        "spent_layer_apps": 100,
                        "wall_clock_s": 120.0,
                        "elapsed_s": 30.0,
                        "exhausted": False,
                        **_accounting_fields(),
                    },
                    "decode_requested_tokens": kwargs["config"]["decode_max_tokens"],
                    "decode_generated_tokens": 12,
                    "decode_termination": "eos",
                    "decode_newline_suppressions": 0,
                    "decode_repetition_penalty_applied": kwargs["config"].get(
                        "decode_repetition_penalty", 1.0
                    ),
                    "decode_temperature": kwargs["config"].get("decode_temperature", 0.0),
                    "decode_top_p": kwargs["config"].get("decode_top_p", 1.0),
                    "verifier_probe_max_tokens": kwargs["config"].get(
                        "verifier_probe_max_tokens", 48
                    ),
                    "generative_verifier": {
                        "requested": True,
                        "available": False,
                        "reason": "stubbed_worker_has_no_generator",
                        "selection_effect": "none",
                    },
                    "counterfactual_verifier": {
                        "requested": True,
                        "available": False,
                        "reason": "stubbed_worker_has_no_generator",
                        "selection_effect": "none",
                    },
                    "prefix_stability": {
                        "requested": True,
                        "available": False,
                        "reason": "stubbed_worker_has_no_generator",
                        "selection_effect": "none",
                        "correctness_effect": "none",
                    },
                    **_verifier_fusion_fields(kwargs["config"]),
                    "latent_opt_applied": True,
                    "latent_opt_mode": "gradient",
                    "latent_opt_attempts": 2,
                    "latent_opt_steps": 2,
                    "latent_opt_rejected": 0,
                    "latent_opt_budget_exhausted": False,
                    "fast_weights_applied": True,
                    "fast_weights_erased": True,
                    "fast_weights_layers": 2,
                    "fast_weight_optimization_attempts": 2,
                    "fast_weight_optimized_steps": 2,
                    "fast_weight_rejected_steps": 0,
                    "fast_weight_budget_exhausted": False,
                    "fast_weight_optimizer": "rms_normalized_sgd_backtracking_v1",
                    "fast_weight_loss_trail": [2.0, 1.5, 1.0],
                    "fast_weight_gradient_norm_trail": [3.0, 2.0],
                    "fast_weight_accepted_step_sizes": [0.005, 0.0025],
                    "fast_weight_line_search_backtracks": 1,
                    "honest_flags": [],
            }
            receipt.update(
                _terminal_disposition_fields(receipt, text=text, tokens=tokens)
            )
            _attach_nonadmitted_fast_weight_receipt(
                receipt,
                text=text,
                tokens=tokens,
            )
            from core.brain.llm.latent_cortex.causal_receipt import (
                build_causal_receipt,
            )

            receipt["causal_receipt"] = build_causal_receipt(receipt)
            return {
                "ok": True,
                "text": text,
                "tokens": tokens,
                "receipt": receipt,
                # A real client binds the receipt to the request payload it
                # sent and publishes the digest (CP126 f22c4ed8); a stub that
                # omits it is simulating a client that never bound anything.
                "request_payload_sha256_bound": receipt.get("request_payload_sha256"),
                "reason": "",
            }

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: StubClient())
    result = asyncio.run(
        svc.deep_reason(
            "hard question",
            stakes=0.9,
            uncertainty=0.9,
            runtime_controls={
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
        )
    )
    assert result["ok"]
    assert result["text"].startswith("The deep answer explains")
    assert result["receipt"]["output_quality"]["passed"] is True
    assert result["receipt"]["causal_receipt"]["required_stages_complete"] is True
    assert result["receipt"]["causal_receipt"]["integrity_proven"] is True
    assert (
        svc.get_status()["last_receipt"]["causal_receipt"]["receipt_sha256"]
        == result["receipt"]["causal_receipt"]["receipt_sha256"]
    )
    tampered = copy.deepcopy(result["receipt"])
    tampered["causal_receipt"]["privacy_contract"][
        "hidden_state_values_included"
    ] = True
    assert "causal_receipt_unproven" in LatentCortexService._receipt_contract_errors(
        tampered,
        captured["config"],
        output_tokens=list(range(12)),
        output_text=result["text"],
        expected_objective="hard question",
    )
    assert captured["prompt"] == "hard question"
    assert captured["config"]["n_branches"] >= 2
    assert captured["config"]["latent_opt"] is True
    assert captured["config"]["fast_weights"] is True
    assert captured["config"]["branch_correlation_evidence"]["schema"] == (
        "aura.rlc.branch_error_correlation.v1"
    )
    assert captured["config"]["branch_correlation_evidence"]["evidence_state"] in {
        "bootstrap_unmeasured",
        "measured",
    }
    fusion_evidence = captured["config"]["verifier_fusion_evidence"]
    assert fusion_evidence["schema"] == "aura.rlc.verifier_fusion_evidence.v1"
    assert fusion_evidence["scopes"]["domain"]["bucket"] == fusion_evidence["bucket"]
    assert result["receipt"]["verifier_fusion"]["authority_mode"] == (
        "diagnostic_fusion_no_single_probabilistic_authority"
    )
    assert captured["budget"]["max_layer_apps"] > 0
    assert captured["runtime_controls"] == {
        "clean_user_surface_recurrent_loops": 2,
        "clean_user_surface_steering_alpha": 0.30,
    }
    assert captured["gate_snapshot"]["active_count"] >= 1
    assert "latent_cortex_foreground:episode" in {
        item["owner"] for item in captured["gate_snapshot"]["active"].values()
    }
    assert svc.get_status()["last_receipt"]["halting_reason"] == "schedule_complete"
    assert (
        svc.get_status()["last_receipt"]["terminal_disposition"]["reason"]
        == "planned_depth_complete"
    )


def test_service_rejects_nominal_full_stack_without_accepted_optimization():
    config = {
        "n_slots": 16,
        "n_branches": 2,
        "latent_opt": True,
        "fast_weights": True,
    }
    receipt = {
        "episode_id": "ep-noop",
        "checkpoint_fingerprint": "a" * 64,
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 8,
        "params_unchanged": True,
        "schedule_hash": "b" * 64,
        "steps_taken": 4,
        "n_slots": 16,
        "n_branches": 2,
        "budget": {
            "max_layer_apps": 1_000,
            "spent_layer_apps": 100,
            "exhausted": False,
        },
        "decode_requested_tokens": 512,
        "decode_generated_tokens": 12,
        "decode_termination": "eos",
        "honest_flags": [],
        "latent_opt_applied": True,
        "latent_opt_mode": "gradient",
        "latent_opt_attempts": 1,
        "latent_opt_steps": 0,
        "latent_opt_rejected": 1,
        "latent_opt_budget_exhausted": False,
        "fast_weights_applied": True,
        "fast_weights_erased": True,
        "fast_weights_layers": 2,
        "fast_weight_optimization_attempts": 1,
        "fast_weight_optimized_steps": 0,
        "fast_weight_rejected_steps": 1,
        "fast_weight_budget_exhausted": False,
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)

    assert "latent_optimization_no_accepted_steps" in errors
    assert "fast_weight_optimization_no_accepted_steps" in errors


def test_service_reconstructs_and_rejects_branch_isolation_tampering():
    config = {"n_branches": 2, "isolation_steps": 2}
    receipt = {
        "n_branches": 2,
        **_branch_isolation_fields(config, exchanges=1),
    }
    assert "branch_isolation_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    tampered = {
        **receipt,
        "branch_isolation": {
            **receipt["branch_isolation"],
            "first_exchange_step": 1,
        },
    }
    assert "branch_isolation_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )
    candidates = [dict(row) for row in receipt["branch_isolation"]["candidates"]]
    candidates[1]["candidate_sha256"] = candidates[0]["candidate_sha256"]
    tampered = {
        **receipt,
        "branch_isolation": {
            **receipt["branch_isolation"],
            "candidates": candidates,
        },
    }
    assert "branch_isolation_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )


def test_service_reconstructs_and_rejects_kv_state_tree_tampering():
    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        "episode_id": "ep-kv-tree",
        **_identity_receipt(episode_id="ep-kv-tree"),
        **_kv_state_tree_fields(
            config,
            episode_id="ep-kv-tree",
        ),
    }
    assert "kv_state_tree_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    attacked = copy.deepcopy(receipt)
    attacked["kv_state_tree"]["events"][0]["parent_cache_sha256"] = "0" * 64
    attacked["kv_state_tree"]["events"][0]["event_sha256"] = _digest(
        "attacker-rehashed-only-the-event"
    )
    assert "kv_state_tree_unproven" in (
        LatentCortexService._receipt_contract_errors(attacked, config)
    )

    attacked = copy.deepcopy(receipt)
    attacked["kv_state_tree"]["nodes"][-1]["cache_sha256"] = attacked["kv_state_tree"]["events"][0][
        "child_cache_sha256"
    ]
    assert "kv_state_tree_unproven" in (
        LatentCortexService._receipt_contract_errors(attacked, config)
    )


def test_service_reconstructs_recurrent_grounding_and_rejects_rehashed_lie():
    from core.brain.llm.latent_cortex.recurrent_grounding import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(config, steps=2),
    }
    assert "recurrent_grounding_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    tampered = copy.deepcopy(receipt)
    transition = tampered["recurrent_grounding"]["branches"][0]["transitions"][0]
    transition["evidence_post_sha256"] = _digest("forged-evidence")
    grounding = tampered["recurrent_grounding"]
    grounding["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in grounding.items() if key != "receipt_sha256"}
    )
    assert "recurrent_grounding_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )


def test_service_rejects_rehashed_heterogeneous_authority_and_decode_lies():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        "budget": _accounting_fields(),
        **_recurrent_grounding_fields(config, steps=2),
    }
    clean_errors = LatentCortexService._receipt_contract_errors(receipt, config)
    assert "heterogeneous_integration_unproven" not in clean_errors
    assert "heterogeneous_decode_unproven" not in clean_errors

    tampered = copy.deepcopy(receipt)
    integration = tampered["heterogeneous_integration"]
    integration["selected_policy"] = "select_new"
    integration["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in integration.items() if key != "receipt_sha256"}
    )
    assert "heterogeneous_integration_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )

    tampered = copy.deepcopy(receipt)
    tampered["heterogeneous_decode"] = {
        "selected_policy": "select_new",
        "answer_text_stored": False,
    }
    assert "heterogeneous_decode_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )


def test_service_reconstructs_loop_stability_and_rejects_rehashed_lies():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(config, steps=3),
    }
    assert "loop_stability_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    forged_alpha = copy.deepcopy(receipt)
    stability = forged_alpha["loop_stability"]
    stability["branches"][0]["transitions"][1]["alpha"] = 0.99
    stability["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in stability.items() if key != "receipt_sha256"}
    )
    assert "loop_stability_unproven" in (
        LatentCortexService._receipt_contract_errors(forged_alpha, config)
    )

    forged_kv = copy.deepcopy(receipt)
    stability = forged_kv["loop_stability"]
    kv_bound = stability["kv_bound"]
    kv_bound["calls"][0]["post_context_tokens"] += 1
    kv_bound["calls_sha256"] = canonical_sha256(kv_bound["calls"])
    stability["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in stability.items() if key != "receipt_sha256"}
    )
    assert "loop_stability_unproven" in (
        LatentCortexService._receipt_contract_errors(forged_kv, config)
    )

    forged_window = copy.deepcopy(receipt)
    stability = forged_window["loop_stability"]
    kv_bound = stability["kv_bound"]
    kv_bound["calls"][0]["start"] = 0
    kv_bound["calls_sha256"] = canonical_sha256(kv_bound["calls"])
    stability["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in stability.items() if key != "receipt_sha256"}
    )
    assert "loop_stability_unproven" in (
        LatentCortexService._receipt_contract_errors(forged_window, config)
    )


def test_service_reconstructs_update_gate_and_rejects_rehashed_decision_lie():
    import json

    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(config, steps=3),
    }
    receipt = json.loads(json.dumps(receipt, sort_keys=True))
    assert "update_acceptance_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    forged = copy.deepcopy(receipt)
    update_gate = forged["update_acceptance"]
    update_gate["branches"][0]["transitions"][0]["probability"] = 0.25
    update_gate["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in update_gate.items() if key != "receipt_sha256"}
    )
    assert "update_acceptance_unproven" in (
        LatentCortexService._receipt_contract_errors(forged, config)
    )


def test_service_reconstructs_stop_gate_and_rejects_rehashed_policy_lie():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(config, steps=3),
    }
    assert "halting_unproven" not in (LatentCortexService._receipt_contract_errors(receipt, config))

    forged = copy.deepcopy(receipt)
    halting = forged["halting"]
    halting["threshold"] = 0.5
    halting["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in halting.items() if key != "receipt_sha256"}
    )
    assert "halting_unproven" in (LatentCortexService._receipt_contract_errors(forged, config))


def test_service_reconstructs_verified_best_and_rejects_rehashed_state_lie():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(config, steps=3),
    }
    assert "verified_best_state_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    forged = copy.deepcopy(receipt)
    verified = forged["verified_best_state"]
    verified["branches"][0]["final_best_state_sha256"] = "a" * 64
    verified["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in verified.items() if key != "receipt_sha256"}
    )
    assert "verified_best_state_unproven" in (
        LatentCortexService._receipt_contract_errors(forged, config)
    )


def test_service_reconstructs_transient_constraints_and_rejects_rehashed_scope_lie():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(episode_id="ep-transient"),
        "episode_id": "ep-transient",
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(
            config,
            steps=3,
            episode_id="ep-transient",
        ),
    }
    assert "transient_negative_constraints_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    forged = copy.deepcopy(receipt)
    transient = forged["transient_negative_constraints"]
    transient["authority_scope"] = "all_branches_forever"
    transient["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in transient.items() if key != "receipt_sha256"}
    )
    assert "transient_negative_constraints_unproven" in (
        LatentCortexService._receipt_contract_errors(forged, config)
    )


def test_service_reconstructs_virtual_quanta_and_rejects_rehashed_scope_lie():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(episode_id="ep-virtual-quanta"),
        "episode_id": "ep-virtual-quanta",
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(
            config,
            steps=3,
            episode_id="ep-virtual-quanta",
        ),
    }
    assert "virtual_quanta_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    forged = copy.deepcopy(receipt)
    virtual = forged["virtual_quanta"]
    virtual["authority_scope"] = "cross_episode_durable"
    virtual["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in virtual.items() if key != "receipt_sha256"}
    )
    assert "virtual_quanta_unproven" in (
        LatentCortexService._receipt_contract_errors(forged, config)
    )


def test_service_reconstructs_uncertainty_and_rejects_rehashed_head_lie():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(config, steps=3),
    }
    assert "neural_uncertainty_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    forged = copy.deepcopy(receipt)
    uncertainty = forged["neural_uncertainty"]
    uncertainty["head_sha256"] = "a" * 64
    uncertainty["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in uncertainty.items() if key != "receipt_sha256"}
    )
    assert "neural_uncertainty_unproven" in (
        LatentCortexService._receipt_contract_errors(forged, config)
    )


def test_service_uncertainty_validation_contains_invalid_config():
    config = {
        "n_slots": 8,
        "n_branches": 2,
        "uncertainty_head": {"mode": "learned"},
    }
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(
            {"n_slots": 8, "n_branches": 2},
            steps=3,
        ),
    }
    errors = LatentCortexService._receipt_contract_errors(receipt, config)
    assert "update_acceptance_unproven" in errors
    assert "neural_uncertainty_unproven" in errors


def test_service_reconstructs_mistake_locator_and_rejects_rehashed_lie():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(config, steps=3),
    }
    assert "mistake_locator_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    forged = copy.deepcopy(receipt)
    locator = forged["mistake_locator"]
    locator["repair_steering_authorized"] = True
    locator["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in locator.items() if key != "receipt_sha256"}
    )
    assert "mistake_locator_unproven" in (
        LatentCortexService._receipt_contract_errors(forged, config)
    )


def test_service_reconstructs_bidirectional_reflector_and_rejects_authority_lie():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(config, steps=3),
    }
    assert "bidirectional_reflector_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    forged = copy.deepcopy(receipt)
    reflector = forged["bidirectional_reflector"]
    reflector["selection_authorized"] = True
    reflector["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in reflector.items() if key != "receipt_sha256"}
    )
    assert "bidirectional_reflector_unproven" in (
        LatentCortexService._receipt_contract_errors(forged, config)
    )


def test_service_reconstructs_contradiction_tensor_and_rejects_authority_lie():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(config, steps=3),
    }
    assert "contradiction_tensor_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    forged = copy.deepcopy(receipt)
    tensor = forged["contradiction_tensor"]
    tensor["attention_perturbation_authorized"] = True
    tensor["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in tensor.items() if key != "receipt_sha256"}
    )
    assert "contradiction_tensor_unproven" in (
        LatentCortexService._receipt_contract_errors(forged, config)
    )


def test_service_reconstructs_perturbation_and_rejects_rehashed_authority_lie():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(config, steps=3),
        "budget": {
            "information_accounting": _accounting_fields()["information_accounting"],
        },
    }
    assert "contradiction_perturbation_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    forged = copy.deepcopy(receipt)
    perturbation = forged["contradiction_perturbation"]
    perturbation["state_mutation_applied"] = True
    perturbation["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in perturbation.items() if key != "receipt_sha256"}
    )
    assert "contradiction_perturbation_unproven" in (
        LatentCortexService._receipt_contract_errors(forged, config)
    )


def test_service_reconstructs_local_exploration_and_rejects_rehashed_evidence_lie():
    from core.brain.llm.latent_cortex.loop_core import canonical_sha256

    config = {"n_slots": 8, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 8,
        "n_branches": 2,
        **_recurrent_grounding_fields(config, steps=3),
        "budget": {
            "information_accounting": _accounting_fields()["information_accounting"],
        },
    }
    assert "local_exploration_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    forged = copy.deepcopy(receipt)
    exploration = forged["local_exploration"]
    exploration["repeat_deterministic"] = True
    exploration["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in exploration.items() if key != "receipt_sha256"}
    )
    assert "local_exploration_unproven" in (
        LatentCortexService._receipt_contract_errors(forged, config)
    )


def test_service_uses_executed_default_branch_count_for_learned_evidence():
    executed = {"n_slots": 16, "n_branches": 2}
    receipt = {
        **_identity_receipt(),
        "n_slots": 16,
        "n_branches": 2,
        **_recurrent_grounding_fields(executed, steps=3),
        "budget": {
            "information_accounting": _accounting_fields()["information_accounting"],
        },
    }
    errors = LatentCortexService._receipt_contract_errors(receipt, {})
    assert "bidirectional_reflector_unproven" not in errors
    assert "contradiction_tensor_unproven" not in errors
    assert "neural_uncertainty_unproven" not in errors
    assert "local_exploration_unproven" not in errors
    assert "mistake_locator_unproven" not in errors


def test_service_reconstructs_and_rejects_branch_exchange_tampering():
    config = {
        "n_branches": 2,
        "n_slots": 4,
        "isolation_steps": 2,
        "comm_slot": 0,
        "exchange_gamma": 0.35,
        "exchange_interval": 2,
    }
    isolation_fields = _branch_isolation_fields(config, exchanges=1)
    receipt = {
        "n_branches": 2,
        "n_slots": 4,
        "cognitive_slots": [],
        "bytecode_events": [],
        "cognitive_action_trace": [],
        "schedule_hash": "test-schedule",
        **isolation_fields,
        "branch_exchange": _branch_exchange_fields(
            config,
            isolation_fields["branch_isolation"],
        ),
    }
    errors = LatentCortexService._receipt_contract_errors(receipt, config)
    assert "branch_exchange_provenance_unproven" not in errors

    tampered = copy.deepcopy(receipt)
    tampered["branch_exchange"]["exchanges"][0]["first_answer_text_exposed"] = True
    assert "branch_exchange_provenance_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )

    tampered = copy.deepcopy(receipt)
    tampered["branch_exchange"]["exchanges"][0]["source_rows"][0]["candidate_sha256"] = _digest(
        "different-candidate"
    )
    assert "branch_exchange_provenance_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )


def test_service_validates_interactive_verifier_profile_and_acceptance_receipt():
    config = {
        "latent_opt": True,
        "verifier_probe_max_tokens": 24,
        "verifier_accept_non_regression": True,
    }
    receipt = {
        "latent_opt_applied": True,
        "latent_opt_mode": "gradient",
        "latent_opt_attempts": 1,
        "latent_opt_steps": 1,
        "latent_opt_rejected": 0,
        "latent_opt_budget_exhausted": False,
        "verifier_probe_max_tokens": 24,
        "latent_opt_verifier": {
            "policy": "task_score_nonregression_with_proxy_descent_v1",
            "score_source": "unspecified",
            "commit_policy": "immediate",
            "baseline_source": "caller_reused_verified_branch",
            "score_tolerance": 1e-9,
            "proxy_tolerance_scale": 1e-9,
            "score_trail": [0.5, 0.5],
            "decisions": [
                {
                    "proposal": 0,
                    "baseline_score": 0.5,
                    "candidate_score": 0.5,
                    "current_proxy_loss": 1.0,
                    "candidate_proxy_loss": 0.9,
                    "proxy_required_delta": 1e-9,
                    "decision": ("accepted_task_score_nonregression_with_proxy_descent"),
                }
            ],
            "score_improvement_accepts": 0,
            "proxy_nonregression_accepts": 1,
            "plateau_exploration_accepts": 1,
            "plateau_rollbacks": 0,
            "strict_improvement_committed": False,
        },
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)

    assert "verifier_probe_profile_mismatch" not in errors
    assert "latent_optimization_verifier_receipt_invalid" not in errors
    tampered = dict(receipt)
    tampered["verifier_probe_max_tokens"] = 48
    assert "verifier_probe_profile_mismatch" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )
    tampered = dict(receipt)
    tampered["latent_opt_verifier"] = {
        **receipt["latent_opt_verifier"],
        "proxy_nonregression_accepts": 0,
    }
    assert "latent_optimization_verifier_receipt_invalid" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )
    tampered = dict(receipt)
    tampered["latent_opt_verifier"] = {
        **receipt["latent_opt_verifier"],
        "decisions": [
            {
                **receipt["latent_opt_verifier"]["decisions"][0],
                "candidate_proxy_loss": 1.1,
            }
        ],
    }
    assert "latent_optimization_verifier_receipt_invalid" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )
    tampered = dict(receipt)
    tampered["latent_opt_verifier"] = {
        **receipt["latent_opt_verifier"],
        "score_trail": [0.5, 0.6],
    }
    assert "latent_optimization_verifier_receipt_invalid" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )


def test_service_replays_strict_commit_plateau_rollbacks():
    config = {
        "latent_opt": True,
        "verifier_probe_max_tokens": 24,
        "verifier_accept_non_regression": True,
    }
    decisions = [
        {
            "proposal": index,
            "baseline_score": 0.5,
            "candidate_score": 0.5,
            "current_proxy_loss": 1.0 - 0.1 * index,
            "candidate_proxy_loss": 0.9 - 0.1 * index,
            "proxy_required_delta": 1e-9,
            "decision": "accepted_task_score_nonregression_with_proxy_descent",
            "commit_disposition": "rolled_back_plateau_without_later_task_gain",
        }
        for index in range(2)
    ]
    receipt = {
        "latent_opt_applied": True,
        "latent_opt_mode": "gradient",
        "latent_opt_attempts": 2,
        "latent_opt_steps": 0,
        "latent_opt_rejected": 2,
        "latent_opt_budget_exhausted": False,
        "verifier_probe_max_tokens": 24,
        "verifier_guidance": {"evaluations": 1},
        "latent_opt_verifier": {
            "policy": "task_score_nonregression_with_proxy_descent_v1",
            "score_source": "semantic_candidate_score_for_latent_search_only_v1",
            "commit_policy": "strict_task_improvement_after_plateau_search_v1",
            "baseline_source": "caller_reused_verified_branch",
            "score_tolerance": 1e-9,
            "proxy_tolerance_scale": 1e-9,
            "score_trail": [0.5, 0.5, 0.5],
            "decisions": decisions,
            "score_improvement_accepts": 0,
            "proxy_nonregression_accepts": 0,
            "plateau_exploration_accepts": 2,
            "plateau_rollbacks": 2,
            "strict_improvement_committed": False,
        },
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)

    assert "latent_optimization_no_accepted_steps" not in errors
    assert "latent_optimization_accounting_mismatch" not in errors
    assert "latent_optimization_verifier_receipt_invalid" not in errors

    tampered = copy.deepcopy(receipt)
    tampered["latent_opt_verifier"]["plateau_rollbacks"] = 1
    assert "latent_optimization_verifier_receipt_invalid" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )

    tampered = copy.deepcopy(receipt)
    del tampered["latent_opt_verifier"]["decisions"][0]["commit_disposition"]
    assert "latent_optimization_verifier_receipt_invalid" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )


def test_service_validates_post_adaptation_candidate_commitment():
    from core.brain.llm.latent_cortex.post_adaptation_candidate import (
        advance_post_adaptation_candidate,
        build_post_adaptation_candidate_receipt,
    )

    transition, _candidate = advance_post_adaptation_candidate(
        selected_branch=0,
        prior_candidate="old candidate",
        observed_candidate="new candidate",
        stage="post_final_adaptation",
        strict_answer_contract=False,
        adaptation_evidence={"latent_opt_accepted_steps": 1},
    )
    post_adaptation = build_post_adaptation_candidate_receipt([transition])
    receipt = {
        "selected_branch": 0,
        "post_adaptation_candidate": post_adaptation,
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, {})
    assert "post_adaptation_candidate_unproven" not in errors

    tampered = copy.deepcopy(receipt)
    tampered["post_adaptation_candidate"]["selected_branch"] = 1
    assert "post_adaptation_candidate_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, {})
    )


def test_service_enforces_default_verifier_probe_profile():
    assert "verifier_probe_profile_mismatch" in (
        LatentCortexService._receipt_contract_errors({}, {})
    )
    assert "verifier_probe_profile_mismatch" not in (
        LatentCortexService._receipt_contract_errors(
            {"verifier_probe_max_tokens": 48},
            {},
        )
    )


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        (
            {"fast_weight_optimizer": "plain_sgd"},
            "fast_weight_optimizer_unproven",
        ),
        (
            {"fast_weight_loss_trail": [2.0, 2.0]},
            "fast_weight_loss_descent_unproven",
        ),
        (
            {"fast_weight_gradient_norm_trail": []},
            "fast_weight_gradient_evidence_invalid",
        ),
        (
            {"fast_weight_accepted_step_sizes": [0.0]},
            "fast_weight_step_evidence_invalid",
        ),
        (
            {"fast_weight_line_search_backtracks": -1},
            "fast_weight_line_search_evidence_invalid",
        ),
    ],
)
def test_service_rejects_unproven_fast_weight_descent(override, expected_error):
    config = {"fast_weights": True}
    receipt = {
        "fast_weights_applied": True,
        "fast_weights_erased": True,
        "fast_weights_layers": 2,
        "fast_weight_optimization_attempts": 1,
        "fast_weight_optimized_steps": 1,
        "fast_weight_rejected_steps": 0,
        "fast_weight_budget_exhausted": False,
        "fast_weight_optimizer": "rms_normalized_sgd_backtracking_v1",
        "fast_weight_loss_trail": [2.0, 1.0],
        "fast_weight_gradient_norm_trail": [3.0],
        "fast_weight_accepted_step_sizes": [0.005],
        "fast_weight_line_search_backtracks": 0,
        **override,
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)

    assert expected_error in errors


@pytest.mark.parametrize(
    ("termination", "exhausted", "expected"),
    [
        ("budget_exhausted", True, "decode_incomplete"),
        ("budget_unaffordable", False, "decode_incomplete"),
        ("token_limit", True, "incomplete_or_exhausted_compute_receipt"),
    ],
)
def test_service_rejects_truncated_or_exhausted_decode_receipts(termination, exhausted, expected):
    config = {
        "n_slots": 16,
        "n_branches": 2,
        "decode_max_tokens": 512,
    }
    receipt = {
        "episode_id": "ep-truncated",
        "checkpoint_fingerprint": "a" * 64,
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 8,
        "params_unchanged": True,
        "schedule_hash": "b" * 64,
        "steps_taken": 4,
        "n_slots": 16,
        "n_branches": 2,
        "budget": {
            "max_layer_apps": 1_000,
            "spent_layer_apps": 1_000 if exhausted else 900,
            "exhausted": exhausted,
        },
        "decode_requested_tokens": 512,
        "decode_generated_tokens": 20,
        "decode_termination": termination,
        "decode_newline_suppressions": 0,
        "decode_repetition_penalty_applied": 1.25,
        "honest_flags": [],
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)

    assert expected in errors


def test_service_reports_refusals_honestly(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    class BusyClient:
        async def latent_reason_async(self, **kwargs):
            return {"ok": False, "reason": "generation_active"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: BusyClient())
    result = asyncio.run(svc.deep_reason("q"))
    assert result["ok"] is False and result["reason"] == "generation_active"
    assert svc.get_status()["last_refusal"] == "generation_active"


@pytest.mark.parametrize(
    ("kwargs", "expected_reason"),
    [
        ({"stakes": float("nan")}, "invalid_cognitive_economy"),
        ({"uncertainty": "high"}, "invalid_cognitive_economy"),
        ({"config_overrides": []}, "invalid_config_overrides"),
        ({"runtime_controls": []}, "invalid_runtime_controls"),
        ({"runtime_controls": {}}, "invalid_runtime_controls"),
        (
            {
                "runtime_controls": {
                    "clean_user_surface_recurrent_loops": 3,
                    "clean_user_surface_steering_alpha": 0.30,
                }
            },
            "invalid_runtime_controls",
        ),
        ({"require_full_stack": "yes"}, "invalid_require_full_stack"),
        ({"foreground_request": "yes"}, "invalid_foreground_request"),
        ({"question": 7}, "invalid_question"),
        ({"messages": "not-a-list"}, "invalid_messages"),
    ],
)
def test_service_rejects_malformed_inputs_before_model_client_lookup(
    monkeypatch, kwargs, expected_reason
):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    import core.brain.llm.mlx_client as mlx_client_mod

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("malformed input must not touch the model client")

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", unexpected_lookup)
    question = kwargs.pop("question", "q")
    result = asyncio.run(svc.deep_reason(question, **kwargs))

    assert result["ok"] is False
    assert result["reason"] == expected_reason
    # CP126 5879d2b5: a refusal now carries a bounded receipt tying
    # it to this call, so exact-dict equality is no longer the contract.
    assert result["refusal_receipt"]["reason"] == result["reason"]


def test_qualified_semantic_service_never_looks_up_the_resident_model_client(
    monkeypatch,
):
    svc = LatentCortexService()
    observed = {}

    import core.brain.llm.mlx_client as mlx_client_mod
    import core.brain.llm.qualified_recurrent_ingress as ingress_mod

    monkeypatch.setattr(
        ingress_mod,
        "admit_qualified_recurrent_objective",
        lambda _objective: SimpleNamespace(family="frontier_calibration"),
    )

    async def execute(client, objective, *, timeout_s):
        observed.update(client=client, objective=objective, timeout_s=timeout_s)
        return {
            "eligible": True,
            "attempted": True,
            "ok": True,
            "text": 'FINAL_ANSWER: {"choice":"H"}',
            "reason": "qualified_semantic_neural_completed",
        }

    monkeypatch.setattr(
        ingress_mod,
        "execute_qualified_recurrent_objective",
        execute,
    )
    monkeypatch.setattr(
        mlx_client_mod,
        "get_mlx_client",
        lambda *args, **kwargs: pytest.fail(
            "qualified semantic execution must not acquire the resident model"
        ),
    )

    result = asyncio.run(
        svc.qualified_recurrent_reason("semantic objective", timeout_s=8.0)
    )

    assert result["ok"] is True
    assert observed == {
        "client": None,
        "objective": "semantic objective",
        "timeout_s": 8.0,
    }


def test_service_propagates_background_lane_priority(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()
    captured: dict[str, object] = {}

    class BackgroundClient:
        async def latent_reason_async(self, **kwargs):
            captured.update(kwargs)
            return {"ok": False, "reason": "generation_active"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(
        mlx_client_mod,
        "get_mlx_client",
        lambda *args, **kwargs: BackgroundClient(),
    )

    result = asyncio.run(svc.deep_reason("idle thought", foreground_request=False))

    assert result["ok"] is False
    assert result["reason"] == "generation_active"
    # CP126 5879d2b5: a refusal now carries a bounded receipt tying
    # it to this call, so exact-dict equality is no longer the contract.
    assert result["refusal_receipt"]["reason"] == result["reason"]
    assert captured["foreground_request"] is False


def test_service_rejects_incomplete_success_receipt(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    class ShallowClient:
        async def latent_reason_async(self, **kwargs):
            return {"ok": True, "text": "shallow", "receipt": {"episode_id": "x"}}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: ShallowClient())
    result = asyncio.run(svc.deep_reason("q"))
    assert result["ok"] is False
    assert result["reason"].startswith("receipt_contract_failed:")
    assert svc.get_status()["ok_episodes"] == 0


@pytest.mark.parametrize(
    ("worker_result", "expected_reason"),
    [
        (
            {"ok": True, "text": "bad", "receipt": "not-a-mapping"},
            "receipt_not_mapping",
        ),
        (
            {
                "ok": True,
                "text": "bad",
                "receipt": {
                    "episode_id": "x",
                    "steps_taken": "7",
                    "n_slots": 16,
                    "n_branches": 2,
                    "budget": {"spent_layer_apps": "100"},
                    "honest_flags": "none",
                },
            },
            "no_recurrent_steps",
        ),
        ("not-a-mapping", "invalid_client_response"),
    ],
)
def test_service_contains_malformed_worker_response(monkeypatch, worker_result, expected_reason):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    class MalformedClient:
        async def latent_reason_async(self, **kwargs):
            return worker_result

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: MalformedClient())
    result = asyncio.run(svc.deep_reason("q"))
    assert result["ok"] is False
    assert expected_reason in result["reason"]
    assert svc.get_status()["failure_streak"] == 1


def test_service_contains_client_exception_and_degrades_health(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    class BrokenClient:
        async def latent_reason_async(self, **kwargs):
            raise RuntimeError("worker exploded")

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: BrokenClient())
    for _ in range(3):
        result = asyncio.run(svc.deep_reason("q"))
        assert result["ok"] is False
    status = svc.get_status()
    assert status["failure_streak"] == 3
    assert status["healthy"] is False and status["state"] == "degraded"


def test_service_name_registered_in_spine():
    from core.service_names import ServiceNames

    assert ServiceNames.LATENT_CORTEX == "latent_cortex"


def test_handler_builds_task_verifier_when_guided(monkeypatch):
    from core.brain.llm.latent_cortex.types import (
        LatentReasoningResult,
    )

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    captured: dict = {}

    class StubEngine:
        def __init__(self, model, tokenizer, config, **kwargs):
            """Accept the engine construction contract; state is unused."""

        def reason(self, **kwargs):
            captured.update(kwargs)
            if kwargs.get("verifier") is not None:
                kwargs["verifier"]("probe with 2 + 2 = 4")
            return LatentReasoningResult(
                ok=True,
                text="ok",
                receipt=_measured_episode_receipt("task-verifier-test"),
            )

    import core.brain.llm.latent_cortex.worker_handler as handler_mod

    monkeypatch.setattr(handler_mod, "LatentCortexEngine", StubEngine)

    class StubTokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [1, 2, 3]

        def decode(self, ids):
            return "x"

    body = handler_mod.handle_latent_reason(
        {"prompt": "verify that 2 + 2 = 4", "verifier_guidance": True},
        model=object(),
        tokenizer=StubTokenizer(),
        model_path="",
        worker_identity=dict(_WORKER_IDENTITY),
    )
    assert body["status"] == "ok"
    assert captured["verifier"] is not None
    guidance = body["receipt"]["verifier_guidance"]
    assert guidance["evaluations"] == 1
    assert guidance["schema"] == "aura.latent_task_verifier.v4"
    assert guidance["grade_admissible"] is True
    assert guidance["atomic_decomposition"]["grade_admissible"] is True
    assert guidance["atomic_decomposition"]["coverage"]["coverage_ratio"] == 1.0
    assert guidance["deterministic_router"]["hard_pass"] is True
    assert "arithmetic" in guidance["best_applicable_checks"]
    assert not guidance.get("best_failures"), "correct arithmetic must not be flagged"
    assert guidance["outcome_checked"] is False
    assert guidance["outcome_passed"] is None
    assert guidance["outcome_reason"] == "candidate_checks_are_not_task_ground_truth"

    # Without the flag, no verifier is constructed.
    captured.clear()
    handler_mod.handle_latent_reason(
        {"prompt": "verify that 2 + 2 = 4"},
        model=object(),
        tokenizer=StubTokenizer(),
        model_path="",
        worker_identity=dict(_WORKER_IDENTITY),
    )
    assert captured["verifier"] is None


def test_worker_handler_capture_lane_exits_before_action_and_returns_public_receipt(
    monkeypatch,
):
    import core.brain.llm.latent_cortex.action_state_capture as capture_mod
    import core.brain.llm.latent_cortex.action_state_runtime as runtime_mod
    import core.brain.llm.latent_cortex.runtime_identity as identity_mod
    import core.brain.llm.latent_cortex.worker_handler as handler_mod
    from core.brain.llm.latent_cortex.types import (
        LatentReasoningResult,
    )

    signer_public = {"schema": "test-worker-capture-identity"}
    admitted = SimpleNamespace(
        mode="capture",
        arm=None,
        admission=object(),
        runner_state={"durable_state": {}, "rng_state": {}},
        latent_reason_request={"prompt": "capture"},
        model_identity={"model": "test"},
        execution_identity={"execution": "test"},
        resident_worker_origin_binding={"worker_identity": signer_public},
    )
    monkeypatch.setattr(
        runtime_mod,
        "admit_action_state_runtime",
        lambda value, **kwargs: admitted,
    )
    monkeypatch.setattr(
        runtime_mod,
        "resident_model_identity_for_worker",
        lambda value: admitted.model_identity,
    )
    monkeypatch.setattr(
        identity_mod,
        "collect_latent_runtime_identity",
        lambda *args, **kwargs: dict(_RUNTIME_IDENTITY),
    )
    monkeypatch.setattr(runtime_mod, "assert_public_runtime_result", lambda value: None)

    class Store:
        closed = False

        def publish(self, admission, private_state, *, created_at_unix):
            assert admission is admitted.admission
            assert private_state == {"portable": "state"}
            return object()

        def close(self):
            self.closed = True

    class Custodian:
        closed = False

        def close(self):
            self.closed = True

    store = Store()
    custodian = Custodian()
    monkeypatch.setattr(
        runtime_mod,
        "open_action_state_store",
        lambda: (store, custodian),
    )
    monkeypatch.setattr(
        capture_mod,
        "build_action_state_capture_receipt",
        lambda **kwargs: {"receipt_sha256": "a" * 64},
    )

    class StubEngine:
        def __init__(self, *args, **kwargs):
            self.initialized_with = (args, kwargs)

        def reason(self, **kwargs):
            assert kwargs["action_continuation_capture_only"] is True
            kwargs["action_continuation_capture"](
                SimpleNamespace(
                    private_state={"portable": "state"},
                    episode_step=0,
                    schedule_step=0,
                    branch_id="branch-0",
                    layer_index=1,
                    kv_position=8,
                )
            )
            return LatentReasoningResult(
                ok=True,
                text="",
                    receipt=_measured_episode_receipt(
                        "action-state-capture-test"
                    ),
                reason="action_state_captured",
            )

    monkeypatch.setattr(handler_mod, "LatentCortexEngine", StubEngine)
    request_sha256 = latent_request_payload_sha256(
        prompt="capture",
        messages=None,
        domain="general",
        config=None,
        budget=None,
        runtime_controls=None,
    )
    body = handler_mod.handle_latent_reason(
        {
            "prompt": "capture",
            "action_state_runtime": {
                "capture_request": {
                    "request_payload": {
                        "latent_reason_request_sha256": request_sha256
                    }
                }
            },
        },
        model=object(),
        tokenizer=None,
        model_path="/models/test-32b",
        worker_identity=dict(_WORKER_IDENTITY),
        worker_capture_signing_identity=SimpleNamespace(
            private_key=object(),
            public_identity=signer_public,
        ),
        worker_capture_launch_challenge={"challenge": "public"},
    )

    assert body["status"] == "ok"
    assert body["text"] == ""
    assert body["action_state_runtime_mode"] == "capture"
    assert body["action_state_capture_receipt"]["receipt_sha256"] == "a" * 64
    assert store.closed is True
    assert custodian.closed is True


def test_service_requests_verifier_guidance_for_resident_profile(monkeypatch):
    svc = LatentCortexService()
    captured: dict = {}

    class Resident32Client:
        def get_worker_identity_snapshot(self):
            return {"worker_model_parameter_count": 32_500_000_000}

        async def latent_reason_async(self, **kwargs):
            captured.update(kwargs)
            return {"ok": False, "reason": "profile_observed"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: Resident32Client())
    asyncio.run(
        svc.deep_reason(
            "hard live question",
            stakes=0.7,
            uncertainty=0.8,
            timeout_s=128.0,
            foreground_request=True,
        )
    )
    assert captured["verifier_guidance"] is True


def test_service_declines_optional_foreground_before_worker_acquisition(monkeypatch):
    svc = LatentCortexService()
    monkeypatch.setattr(
        "core.brain.latent_cortex_service._foreground_surplus_plan",
        lambda **_kwargs: {
            "schema": "aura.latent_cortex.foreground_surplus.v1",
            "latent_surplus_s": 12.0,
            "latent_runtime_floor_s": 120.0,
            "admitted": False,
        },
    )

    import core.brain.llm.mlx_client as mlx_client_mod

    def _must_not_acquire():
        raise AssertionError("resident worker was acquired without latent surplus")

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", _must_not_acquire)
    result = asyncio.run(
        svc.deep_reason(
            "compound live question",
            config_overrides={"decode_max_tokens": 1536},
            runtime_controls={
                "clean_user_surface_recurrent_loops": 1,
                "clean_user_surface_steering_alpha": 0.0,
            },
            timeout_s=140.0,
            foreground_request=True,
        )
    )

    assert result["ok"] is False
    assert result["reason"] == "latent_surplus_budget_insufficient"
    assert result["refusal_receipt"]["stage"] == "surplus_admission"
    assert svc.get_status()["last_allocation"]["foreground_surplus_admission"] == {
        "schema": "aura.latent_cortex.foreground_surplus.v1",
        "latent_surplus_s": 12.0,
        "latent_runtime_floor_s": 120.0,
        "admitted": False,
    }


def test_service_passes_only_surplus_window_to_worker(monkeypatch):
    svc = LatentCortexService()
    captured = {}
    monkeypatch.setattr(
        "core.brain.latent_cortex_service._foreground_surplus_plan",
        lambda **_kwargs: {
            "schema": "aura.latent_cortex.foreground_surplus.v1",
            "latent_surplus_s": 37.0,
            "latent_runtime_floor_s": 30.0,
            "admitted": True,
        },
    )

    class StubClient:
        def get_worker_identity_snapshot(self):
            return {}

        async def latent_reason_async(self, **kwargs):
            captured.update(kwargs)
            return {"ok": False, "reason": "measured_stub_stop"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda: StubClient())
    result = asyncio.run(
        svc.deep_reason(
            "simple live question",
            runtime_controls={
                "clean_user_surface_recurrent_loops": 1,
                "clean_user_surface_steering_alpha": 0.0,
            },
            timeout_s=180.0,
            foreground_request=True,
        )
    )

    assert result["reason"] == "measured_stub_stop"
    assert captured["timeout_s"] == 37.0
    assert svc.get_status()["last_allocation"]["foreground_surplus_admission"][
        "latent_surplus_s"
    ] == 37.0


def test_service_action_state_lane_preserves_frozen_runner_inputs(monkeypatch):
    svc = LatentCortexService()
    captured: dict = {}

    class ClaimClient:
        async def latent_reason_async(self, **kwargs):
            captured.update(kwargs)
            return {
                "ok": True,
                "text": "",
                "receipt": {"episode_id": "capture"},
                "action_state_capture_receipt": {"receipt_sha256": "a" * 64},
            }

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda: ClaimClient())
    result = asyncio.run(
        svc.run_action_state_episode(
            prompt="frozen task",
            domain="reasoning",
            config={"decode_max_tokens": 32},
            budget={"wall_clock_s": 30.0},
            cognitive_context=[{"source": "public-task", "text": "evidence"}],
            action_policy_evidence={"schema": "policy"},
            action_state_runtime={"schema": "runtime"},
            timeout_s=45.0,
        )
    )

    assert result["ok"] is True
    assert captured["foreground_request"] is False
    assert captured["action_state_runtime"] == {"schema": "runtime"}
    assert captured["action_policy_evidence"] == {"schema": "policy"}
    assert captured["config"] == {"decode_max_tokens": 32}
    assert captured["budget"] == {"wall_clock_s": 30.0}
    assert captured["verifier_guidance"] is True


# ── GWT ↔ RLC coupling gating ───────────────────────────────────────────


def _full_success_stub_client(captured):
    class StubClient:
        async def latent_reason_async(self, prompt=None, **kwargs):
            captured["prompt"] = prompt
            captured["config"] = kwargs.get("config")
            text = "A deliberate conclusion that answers the question."
            tokens = list(range(12))
            receipt = {
                    "steps_taken": kwargs["config"]["max_steps"],
                    "halting_reason": "schedule_complete",
                    "n_branches": kwargs["config"]["n_branches"],
                    "n_slots": kwargs["config"]["n_slots"],
                    "episode_id": "ep-gwt",
                    "schedule_hash": "b" * 64,
                    "checkpoint_fingerprint": "a" * 64,
                    "checkpoint_fingerprint_method": "sha256",
                    "checkpoint_file_count": 8,
                    **_identity_receipt(episode_id="ep-gwt"),
                    **_branch_isolation_fields(kwargs["config"]),
                    **_recurrent_grounding_fields(
                        kwargs["config"],
                        steps=kwargs["config"]["max_steps"],
                        episode_id="ep-gwt",
                    ),
                    **_kv_state_tree_fields(
                        kwargs["config"],
                        episode_id="ep-gwt",
                    ),
                    **_latent_tree_fields(kwargs["config"], episode_id="ep-gwt"),
                    "params_unchanged": True,
                    "budget": {
                        "max_layer_apps": 1_000,
                        "spent_layer_apps": 100,
                        "wall_clock_s": 120.0,
                        "elapsed_s": 30.0,
                        "exhausted": False,
                        **_accounting_fields(),
                    },
                    "decode_requested_tokens": kwargs["config"]["decode_max_tokens"],
                    "decode_generated_tokens": 12,
                    "decode_termination": "eos",
                    "decode_newline_suppressions": 0,
                    "decode_repetition_penalty_applied": 1.0,
                    "decode_temperature": 0.0,
                    "decode_top_p": 1.0,
                    "verifier_probe_max_tokens": kwargs["config"].get(
                        "verifier_probe_max_tokens", 48
                    ),
                    "generative_verifier": {
                        "requested": True,
                        "available": False,
                        "reason": "stubbed_worker_has_no_generator",
                        "selection_effect": "none",
                    },
                    "counterfactual_verifier": {
                        "requested": True,
                        "available": False,
                        "reason": "stubbed_worker_has_no_generator",
                        "selection_effect": "none",
                    },
                    "prefix_stability": {
                        "requested": True,
                        "available": False,
                        "reason": "stubbed_worker_has_no_generator",
                        "selection_effect": "none",
                        "correctness_effect": "none",
                    },
                    **_verifier_fusion_fields(kwargs["config"]),
                    "latent_opt_applied": True,
                    "latent_opt_mode": "gradient",
                    "latent_opt_attempts": 2,
                    "latent_opt_steps": 2,
                    "latent_opt_rejected": 0,
                    "latent_opt_budget_exhausted": False,
                    "fast_weights_applied": False,
                    "fast_weights_erased": None,
                    "fast_weights_layers": 0,
                    "fast_weight_optimization_attempts": 0,
                    "fast_weight_optimized_steps": 0,
                    "fast_weight_rejected_steps": 0,
                    "fast_weight_budget_exhausted": False,
                    "fast_weight_optimizer": "",
                    "fast_weight_loss_trail": [],
                    "fast_weight_gradient_norm_trail": [],
                    "fast_weight_accepted_step_sizes": [],
                    "fast_weight_line_search_backtracks": 0,
                    "honest_flags": [],
            }
            receipt.update(
                _terminal_disposition_fields(receipt, text=text, tokens=tokens)
            )
            from core.brain.llm.latent_cortex.fast_weight_learning import (
                empty_learning_state,
                finalize_fast_weight_learning_receipt,
                token_sequence_sha256,
                unavailable_admission,
            )

            admission = unavailable_admission(
                source_sha256=hashlib.sha256(b"").hexdigest(),
                objective_sha256=hashlib.sha256(b"").hexdigest(),
                reason="candidate_evaluation_unavailable",
            )
            learning_state = empty_learning_state(
                episode_id="ep-gwt",
                input_tokens_sha256=receipt["input_tokens_sha256"],
                selected_branch=int(receipt["selected_branch"]),
                winner_state_sha256=hashlib.sha256(b"gwt-winner").hexdigest(),
                admission=admission,
            )
            learning_state["final_answer"] = {
                "decoded_under_adaptation": False,
                "tokens_sha256": token_sequence_sha256(tokens),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "token_count": len(tokens),
            }
            receipt["fast_weight_learning"] = (
                finalize_fast_weight_learning_receipt(learning_state)
            )
            from core.brain.llm.latent_cortex.causal_receipt import (
                build_causal_receipt,
            )

            receipt["causal_receipt"] = build_causal_receipt(receipt)
            return {
                "ok": True,
                "text": text,
                "tokens": tokens,
                "receipt": receipt,
                # A real client binds the receipt to the request payload it
                # sent and publishes the digest (CP126 f22c4ed8); a stub that
                # omits it is simulating a client that never bound anything.
                "request_payload_sha256_bound": receipt.get("request_payload_sha256"),
                "reason": "",
            }

    return StubClient


def _run_episode_with_coupling_probes(monkeypatch, *, foreground: bool):
    import core.brain.gwt_rlc_coupling as coupling_mod
    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()
    captured: dict = {}
    calls = {"merge": 0, "broadcast": 0}

    def _fake_merge(items, **kwargs):
        calls["merge"] += 1
        return items

    async def _fake_broadcast(objective, text, receipt, *, stakes=0.5):
        calls["broadcast"] += 1
        return {
            "schema": coupling_mod.GWT_RLC_SCHEMA,
            "submitted": True,
            "accepted": True,
            "priority": 0.7,
            "pricing": {"verified": False},
        }

    monkeypatch.setattr(coupling_mod, "merge_cognitive_context", _fake_merge)
    monkeypatch.setattr(coupling_mod, "broadcast_episode_conclusion", _fake_broadcast)
    monkeypatch.setattr(
        mlx_client_mod,
        "get_mlx_client",
        lambda *a, **k: _full_success_stub_client(captured)(),
    )
    result = asyncio.run(
        svc.deep_reason(
            "hard question",
            stakes=0.9,
            uncertainty=0.9,
            foreground_request=foreground,
            runtime_controls={
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
        )
    )
    return result, calls


def test_foreground_episode_couples_to_workspace(monkeypatch):
    result, calls = _run_episode_with_coupling_probes(monkeypatch, foreground=True)
    assert result["ok"]
    assert calls["merge"] == 1
    assert calls["broadcast"] == 1
    broadcast = result["receipt"]["workspace_broadcast"]
    assert broadcast["submitted"] is True
    assert broadcast["accepted"] is True


def test_background_episode_stays_decoupled_from_live_mind(monkeypatch):
    result, calls = _run_episode_with_coupling_probes(monkeypatch, foreground=False)
    assert result["ok"]
    assert calls["merge"] == 0
    assert calls["broadcast"] == 0
    assert "workspace_broadcast" not in result["receipt"]


# ── Held-out facet grading loop (service ↔ Foundry) ─────────────────────


def test_facet_weights_stay_none_until_foundry_has_graded_evidence(monkeypatch):
    class NeutralFoundry:
        def weight_for(self, verifier, domain):
            return 1.0

    import core.brain.verifiers.foundry as foundry_mod

    monkeypatch.setattr(foundry_mod, "get_verifier_foundry", lambda: NeutralFoundry())
    assert LatentCortexService._facet_reliability_weights("general") is None

    class MeasuredFoundry:
        def weight_for(self, verifier, domain):
            return 0.4 if verifier == "latent_facet_explain" else 1.0

    monkeypatch.setattr(foundry_mod, "get_verifier_foundry", lambda: MeasuredFoundry())
    weights = LatentCortexService._facet_reliability_weights("general")
    assert weights is not None
    assert weights["explain"] == 0.4
    assert weights["compare"] == 1.0


def test_successful_episode_queues_facet_judgments_for_grading(monkeypatch):
    recorded: list[dict] = []

    class RecordingFoundry:
        def record_verdict(self, **kwargs):
            recorded.append(kwargs)
            return f"vd-{len(recorded)}"

        def weight_for(self, verifier, domain):
            return 1.0

    import core.brain.verifiers.foundry as foundry_mod

    monkeypatch.setattr(foundry_mod, "get_verifier_foundry", lambda: RecordingFoundry())
    svc = LatentCortexService()
    receipt = {
        "verifier_guidance": {
            "evaluations": 3,
            "best_score": 0.8,
            "facet_judgments": [
                {
                    "facet": "explain",
                    "satisfied": True,
                    "excerpt": "because the lease ordering bounds waiting",
                },
                {"facet": "compare", "satisfied": False, "excerpt": ""},
                {"facet": 42, "satisfied": True},  # junk row is skipped
            ],
        }
    }
    svc._record_facet_judgments(receipt, "general", "why prefer older leases?")
    assert len(recorded) == 2
    by_verifier = {row["verifier"]: row for row in recorded}
    explain = by_verifier["latent_facet_explain"]
    assert explain["hard_pass"] is True and explain["score"] == 1.0
    # CP126 94ecfee0: this is the WORKER's assertion about its own output, so
    # it enters the Foundry ungraded. An operator grading it against the
    # excerpt is what makes it checked — recording it as checked gave a
    # self-report the standing of a verified grade.
    assert explain["checked"] is False
    assert explain["meta"]["source"] == "worker_self_assertion"
    assert "lease ordering" in explain["meta"]["excerpt"]
    compare = by_verifier["latent_facet_compare"]
    assert compare["hard_pass"] is False and compare["score"] == 0.0
    assert explain["task_key"] == compare["task_key"] != ""


def test_budget_accepts_the_allocators_own_provenance_annotations():
    """The strict budget check rejected the allocator's own records.

    latent_cortex_service records four provenance fields beside the budget so a
    deeper-than-usual episode is "traceable to the reason it was deeper". The
    worker allowed exactly two keys, so on the live desktop path EVERY
    foreground turn logged

      mlx_worker (warning): ValueError: latent_reason budget contains unknown
      keys: ['effective_uncertainty', 'effort', 'novelty', 'ontogeny_episode']

    and the Recursive Latent Cortex then declined the turn. The whole latent
    lane was dark on the user surface behind a warning that read like a caller
    bug.
    """
    from core.brain.llm.latent_cortex.worker_handler import budget_from_job

    # The exact payload the live allocator sends.
    budget = budget_from_job(
        {
            "max_layer_apps": 2_000_000,
            "wall_clock_s": 30.0,
            "effective_uncertainty": 0.41,
            "novelty": 0.2,
            "effort": "high",
            "ontogeny_episode": "ep-1",
        }
    )
    assert budget.max_layer_apps == 2_000_000
    assert budget.wall_clock_s == 30.0

    # Annotations are optional, not required.
    assert budget_from_job({"max_layer_apps": 5}).max_layer_apps == 5

    # A typo in a REAL limit must still be rejected — that is the whole point of
    # the strict check, and widening it must not turn into "anything goes".
    for typo in ({"max_layer_app": 5}, {"wall_clock": 1.0}, {"whatever": 1}):
        with pytest.raises(ValueError, match="unknown keys"):
            budget_from_job(typo)


def test_the_allocator_and_the_worker_agree_on_every_budget_key():
    """Pin the contract at both ends so it cannot drift apart again silently."""
    import inspect

    from core.brain import latent_cortex_service
    from core.brain.llm.latent_cortex import worker_handler

    accepted = worker_handler._BUDGET_COMPUTE_KEYS | worker_handler._BUDGET_ANNOTATION_KEYS
    source = inspect.getsource(latent_cortex_service)
    start = source.index("budget = {")
    produced = {
        line.split('"')[1]
        for line in source[start : source.index("}", start)].splitlines()
        if line.strip().startswith('"')
    }
    assert produced, "could not read the allocator's budget keys"
    assert produced <= accepted, (
        f"the allocator produces keys the worker rejects: {sorted(produced - accepted)}"
    )
