"""Unified public causal DAG over one latent-cortex episode.

The envelope commits existing independently validated receipts without copying
their contents. It is intentionally unsuitable for private reasoning: nodes
contain field names, content hashes, shapes/counts, and ordering only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.loop_core import canonical_sha256

SCHEMA = "aura.rlc.causal_receipt.v1"
POLICY_VERSION = "spark-054.v1"


@dataclass(frozen=True, slots=True)
class _Stage:
    name: str
    required: bool
    sources: tuple[str, ...]
    anchors: tuple[str, ...]


_STAGES = (
    _Stage(
        "identity_and_ingress",
        True,
        (
            "episode_id",
            "request_payload_sha256",
            "input_tokens_sha256",
            "input_token_count",
            "input_context_compaction",
            "runtime_identity",
        ),
        (
            "episode_id",
            "request_payload_sha256",
            "input_tokens_sha256",
            "runtime_identity",
        ),
    ),
    _Stage(
        "state_lineage",
        True,
        (
            "cognitive_slots",
            "recurrent_grounding",
            "loop_stability",
            "kv_state_tree",
        ),
        ("recurrent_grounding", "loop_stability", "kv_state_tree"),
    ),
    _Stage(
        "cognitive_operators",
        True,
        (
            "value_of_computation",
            "cognitive_action_trace",
            "cognitive_operator_trace",
            "context_focus_trace",
            "structural_diversity",
            "disagreement_graph",
            "diagnostic_action_selection",
        ),
        ("cognitive_action_trace",),
    ),
    _Stage(
        "branch_isolation_and_exchange",
        True,
        (
            "branch_isolation",
            "branch_exchange",
            "branch_scores",
            "selected_branch",
            "exchanges",
        ),
        ("branch_isolation", "selected_branch"),
    ),
    _Stage(
        "tool_memory_and_external_evidence",
        False,
        (
            "nonparametric_memory",
            "runtime_operation_authority",
            "context_focus_trace",
            "external_execution_handoff",
        ),
        (),
    ),
    _Stage(
        "verification",
        True,
        (
            "branch_contract",
            "verifier_preflight",
            "blind_review",
            "decoy_verification",
            "generative_verifier",
            "counterfactual_verifier",
            "prefix_stability",
            "verifier_fusion",
            "neural_uncertainty",
            "mistake_locator",
            "bidirectional_reflector",
            "contradiction_tensor",
        ),
        ("verifier_fusion", "neural_uncertainty", "mistake_locator"),
    ),
    _Stage(
        "accepted_and_rejected_updates",
        True,
        (
            "update_acceptance",
            "verified_best_state",
            "transient_negative_constraints",
            "virtual_quanta",
            "latent_tree_search",
            "contradiction_perturbation",
            "local_exploration",
            "heterogeneous_integration",
            "local_repair",
            "answer_replacement",
        ),
        ("update_acceptance", "verified_best_state"),
    ),
    _Stage(
        "compute_accounting",
        True,
        ("budget", "stage_timings_s", "probe_cache", "latent_telemetry"),
        ("budget",),
    ),
    _Stage(
        "temporary_and_durable_adaptation",
        True,
        (
            "recurrence_adapter",
            "latent_opt_applied",
            "latent_opt_attempts",
            "latent_opt_steps",
            "latent_opt_rejected",
            "latent_opt_verifier",
            "fast_weights_applied",
            "fast_weight_optimization_attempts",
            "fast_weight_optimized_steps",
            "fast_weight_rejected_steps",
            "fast_weight_canaries",
            "fast_weight_verifier",
            "fast_weight_learning",
            "fast_weight_cleanup",
            "post_adaptation_candidate",
        ),
        ("latent_opt_applied", "fast_weights_applied"),
    ),
    _Stage(
        "stopping",
        True,
        ("halting_reason", "halting", "terminal_disposition"),
        ("halting_reason", "halting", "terminal_disposition"),
    ),
    _Stage(
        "final_synthesis",
        True,
        (
            "decode_requested_tokens",
            "decode_generated_tokens",
            "decode_termination",
            "decode_bridge_applied",
            "decode_bridge_policy",
            "decode_bridge_token_count",
            "decode_bridge_tokens_sha256",
            "decode_bridge_logits_digest",
            "decode_incumbent_policy",
            "decode_incumbent_prompt_logits_sha256",
            "incumbent_artifact",
            "heterogeneous_decode",
            "post_adaptation_candidate",
            "answer_replacement",
            "terminal_disposition",
        ),
        (
            "decode_generated_tokens",
            "decode_termination",
            "terminal_disposition",
        ),
    ),
    _Stage(
        "runtime_and_model_integrity",
        True,
        (
            "checkpoint_fingerprint",
            "worker_boot_id",
            "worker_pid",
            "worker_model_path",
            "worker_source_sha256",
            "params_unchanged",
            "fast_weights_erased",
            "weight_integrity",
            "integrity_verdicts",
            "runtime_integrity",
            "worker_identity",
            "runtime_identity",
        ),
        (
            "checkpoint_fingerprint",
            "worker_boot_id",
            "worker_pid",
            "worker_model_path",
            "worker_source_sha256",
            "runtime_integrity",
            "worker_identity",
            "runtime_identity",
        ),
    ),
)


def _present(value: Any) -> bool:
    return value not in (None, "", (), [], {})


def _shape(value: Any) -> str:
    if isinstance(value, Mapping):
        return f"mapping:{len(value)}"
    if isinstance(value, (list, tuple)):
        return f"sequence:{len(value)}"
    if value is None:
        return "missing"
    return type(value).__name__


def _commit_source(name: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = receipt.get(name)
    return {
        "field": name,
        "present": _present(value),
        "shape": _shape(value),
        "value_sha256": canonical_sha256(value),
    }


def _anchor_present(name: str, receipt: Mapping[str, Any]) -> bool:
    value = receipt.get(name)
    if name.endswith("_trace"):
        return name in receipt and isinstance(value, list)
    if name in {"selected_branch", "exchanges", "input_token_count"}:
        return type(value) is int and value >= 0
    if name in {
        "latent_opt_applied",
        "fast_weights_applied",
        "params_unchanged",
        "fast_weights_erased",
    }:
        return type(value) is bool
    if name == "decode_generated_tokens":
        return type(value) is int and value > 0
    return _present(value)


def _runtime_integrity_proven(receipt: Mapping[str, Any]) -> bool:
    try:
        from core.brain.llm.latent_cortex.runtime_integrity import (
            validate_runtime_integrity_receipt,
        )

        worker_identity = receipt.get("worker_identity")
        if not isinstance(worker_identity, Mapping):
            return False
        proof = validate_runtime_integrity_receipt(
            receipt.get("runtime_integrity"),
            require_worker=True,
            expected_episode_id=str(receipt.get("episode_id") or ""),
            expected_input_tokens_sha256=str(
                receipt.get("input_tokens_sha256") or ""
            ),
            expected_worker_identity=worker_identity,
            expected_fast_weights_applied=(
                receipt.get("fast_weights_applied") is True
            ),
            expected_fast_weights_attach_attempted=(
                receipt.get("fast_weights_attach_attempted") is True
            ),
            expected_checkpoint_fingerprint=str(
                receipt.get("checkpoint_fingerprint") or ""
            ),
            expected_checkpoint_method=str(
                receipt.get("checkpoint_fingerprint_method") or ""
            ),
            expected_checkpoint_file_count=receipt.get(
                "checkpoint_file_count"
            ),
        )
    # not a failure: a receipt that will not validate has not validated, and False is
    # the refusing direction.
    except (ImportError, TypeError, ValueError):
        return False
    return proof["verdict"]["safe_to_continue"] is True


def build_causal_receipt(worker_receipt: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(worker_receipt, Mapping):
        raise TypeError("worker receipt must be a mapping")
    episode_id = worker_receipt.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("causal receipt requires an episode identity")
    nodes: list[dict[str, Any]] = []
    previous = ""
    missing: list[str] = []
    for ordinal, stage in enumerate(_STAGES):
        commitments = [_commit_source(name, worker_receipt) for name in stage.sources]
        anchors_present = bool(stage.anchors) and all(
            _anchor_present(name, worker_receipt) for name in stage.anchors
        )
        if stage.name == "runtime_and_model_integrity":
            anchors_present = anchors_present and _runtime_integrity_proven(
                worker_receipt
            )
        any_present = any(row["present"] for row in commitments)
        status = (
            "complete"
            if anchors_present
            else "incomplete"
            if stage.required
            else "observed"
            if any_present
            else "not_applicable"
        )
        if stage.required and status != "complete":
            missing.append(stage.name)
        body = {
            "ordinal": ordinal,
            "stage": stage.name,
            "required": stage.required,
            "status": status,
            "prior_node_sha256": previous,
            "source_commitments": commitments,
        }
        node = {**body, "node_sha256": canonical_sha256(body)}
        nodes.append(node)
        previous = node["node_sha256"]

    terminal = worker_receipt.get("terminal_disposition")
    language = terminal.get("language") if isinstance(terminal, Mapping) else None
    final_output_sha256 = (
        language.get("output_text_sha256") if isinstance(language, Mapping) else ""
    )
    integrity_proven = _runtime_integrity_proven(worker_receipt)
    payload = {
        "schema": SCHEMA,
        "policy_version": POLICY_VERSION,
        "episode_id": episode_id,
        "input_tokens_sha256": worker_receipt.get("input_tokens_sha256", ""),
        "stage_order": [stage.name for stage in _STAGES],
        "nodes": nodes,
        "required_stages_complete": not missing,
        "missing_required_stages": missing,
        "integrity_proven": integrity_proven,
        "final_output_text_sha256": final_output_sha256,
        "privacy_contract": {
            "representation": "public_commitments_counts_dispositions_only",
            "private_chain_of_thought_included": False,
            "hidden_state_values_included": False,
            "raw_tool_secret_values_included": False,
        },
        "root_node_sha256": previous,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_causal_receipt(
    value: Any,
    *,
    worker_receipt: Mapping[str, Any],
    require_complete: bool,
) -> dict[str, Any]:
    if type(require_complete) is not bool:
        raise TypeError("causal receipt completeness requirement must be boolean")
    expected = build_causal_receipt(worker_receipt)
    if not isinstance(value, dict) or value != expected:
        raise ValueError("causal receipt differs from independently reconstructed DAG")
    if require_complete and (
        value["required_stages_complete"] is not True
        or value["missing_required_stages"]
        or value["integrity_proven"] is not True
    ):
        raise ValueError("causal receipt is incomplete or integrity is unproven")
    return value


__all__ = [
    "POLICY_VERSION",
    "SCHEMA",
    "build_causal_receipt",
    "validate_causal_receipt",
]
