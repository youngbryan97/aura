"""core/brain/latent_cortex_service.py

Orchestrator-side facade for the Recursive Latent Cortex
(docs/RECURSIVE_LATENT_CORTEX.md). The engine itself runs inside the MLX
worker on the RESIDENT model; this service is the cognitive economy around
it — it decides how much latent computation a problem deserves and routes
the episode through the worker IPC.

The allocation policy is the spec's: thought (T, branches, budget) scales
with stakes and uncertainty, and is DAMPED by the body's real+anticipatory
pressure — a system heading toward crisis spends less on deep thought, which
is exactly what the allostasis seam is for.

Fail-honest: any refusal (kill switch, busy lane, no resident model, worker
error) returns ``{"ok": False, "reason": ...}`` with bounded evidence so the
caller can decide whether no model work ran or the single model owner was
already exhausted. Nothing here fakes an answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.output_quality import evaluate_latent_output
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.LatentCortexService")

#: What the worker writes only once it has actually selected actions.
_WRITTEN_WHEN_ACTIONS_ARE_SELECTED = frozenset(
    {"executors", "actions_selected", "checked_transitions", "selected_actions"}
)


class _ActionSelectionNeverRanError(Exception):
    """The episode stopped before it chose any actions.

    Not a ValueError, because every ValueError in that check means the worker
    said something the host disagrees with, and this means the worker never
    said anything.
    """


_GENERAL_LATENT_UNMEASURED_FLOOR_SECONDS = 120.0


def _an_interval_a_return_can_travel(config: Mapping[str, Any]) -> int:
    """The exchange interval, pulled in until a return has somewhere to go.

    An interval at or above the allocated depth leaves the only eligible
    exchange on the last recurrent step, where the consensus reaches the answer
    through the persisted cache and no further thinking at all. Pulling it in
    costs nothing: the exchange spends no layer applications -- it is a mean
    over the private slots, a softmax over K agreements, and one blend -- so an
    earlier exchange is the same work at a point where its result can be read.
    """
    from core.brain.llm.latent_cortex.branch_exchange import (
        exchange_steps_a_return_can_travel,
    )
    from core.brain.llm.latent_cortex.types import BranchConfig

    interval = int(config.get("exchange_interval", BranchConfig().exchange_interval))
    branches = int(config.get("n_branches") or 1)
    if branches <= 1:
        return interval
    depth = int(config["max_steps"])
    isolation = int(config.get("isolation_steps") or 1)
    pulled = max(1, min(interval, depth - 1))
    if exchange_steps_a_return_can_travel(
        n_branches=branches,
        max_steps=depth,
        isolation_steps=isolation,
        exchange_interval=pulled,
    ):
        return pulled
    # Depth, not the interval, is the binding constraint here: the episode
    # stops at or before the step isolation seals. Leave the interval alone
    # rather than pretend a shorter one buys anything.
    return interval


def _input_prompt_tokens(messages: list | None, objective: str) -> int:
    """Price the supplied context, retaining the cold-start minimum."""
    from core.brain.memory_guard import estimate_tokens

    inputs = messages or [{"role": "user", "content": objective}]
    return max(2048, estimate_tokens(inputs))


def _foreground_surplus_plan(
    *,
    messages: list | None,
    visible_objective: str,
    decode_max_tokens: int,
    request_timeout_s: float,
    last_latency_s: float,
) -> dict[str, Any]:
    """Reserve a complete canonical answer before optional recurrent work."""

    requested_s = max(0.0, float(request_timeout_s))
    decode_capacity = max(64, min(4096, int(decode_max_tokens)))
    model = "unknown"
    prompt_tokens = max(2048, 1800 + len(visible_objective) // 4)
    planned_decode_tokens = min(decode_capacity, 384)
    deadline_confidence = "no_samples"
    deadline_samples = 0
    length_confidence = "no_samples"
    length_samples = 0
    try:
        from core.brain.llm.measured_admission import (
            recommended_completion_tokens,
            recommended_foreground_deadline,
        )
        from core.brain.llm.model_registry import get_active_cortex_spec
        from core.runtime.structured_input import answer_surface_planning_tokens

        spec = get_active_cortex_spec()
        if spec is not None:
            model = os.path.basename(str(spec.model_path))
        prompt_tokens = _input_prompt_tokens(messages, visible_objective)
        planned_decode_tokens = min(
            decode_capacity,
            answer_surface_planning_tokens(visible_objective),
        )
        (
            planned_decode_tokens,
            length_confidence_value,
            length_samples,
        ) = recommended_completion_tokens(
            model=model,
            prompt_tokens=prompt_tokens,
            maximum_tokens=decode_capacity,
            prior_tokens=planned_decode_tokens,
        )
        (
            answer_reserve_s,
            deadline_confidence_value,
            deadline_samples,
        ) = recommended_foreground_deadline(
            model=model,
            prompt_tokens=prompt_tokens,
            decode_tokens=planned_decode_tokens,
            minimum_seconds=0.0,
            maximum_seconds=requested_s,
        )
        deadline_confidence = deadline_confidence_value.value
        length_confidence = length_confidence_value.value
    except (ArithmeticError, AttributeError, ImportError, OSError, TypeError, ValueError):
        # A failed estimator cannot make optional work cheaper. This prior is
        # deliberately conservative and remains labeled as unmeasured.
        answer_reserve_s = min(
            requested_s,
            4.0 + (0.40 * float(planned_decode_tokens)),
        )

    runtime_floor_s = max(
        _GENERAL_LATENT_UNMEASURED_FLOOR_SECONDS,
        max(0.0, float(last_latency_s)),
    )
    surplus_s = max(0.0, requested_s - float(answer_reserve_s))
    return {
        "schema": "aura.latent_cortex.foreground_surplus.v1",
        "model": model,
        "request_timeout_s": round(requested_s, 3),
        "prompt_tokens": int(prompt_tokens),
        "decode_capacity_tokens": decode_capacity,
        "planned_decode_tokens": int(planned_decode_tokens),
        "canonical_answer_reserve_s": round(float(answer_reserve_s), 3),
        "latent_surplus_s": round(surplus_s, 3),
        "latent_runtime_floor_s": round(runtime_floor_s, 3),
        "deadline_confidence": deadline_confidence,
        "deadline_samples": int(deadline_samples),
        "length_confidence": length_confidence,
        "length_samples": int(length_samples),
        "admitted": surplus_s >= runtime_floor_s,
    }


def _operation_authority_rejected(
    *,
    worker_ok: bool,
    worker_receipt: dict[str, Any],
    expected_authority: dict[str, Any] | None,
) -> bool:
    """Reject absent success authority and any explicit conflicting authority."""

    authority_present = "runtime_operation_authority" in worker_receipt
    authority_matches = (
        authority_present
        and worker_receipt.get("runtime_operation_authority") == expected_authority
    )
    return bool(
        (worker_ok and not authority_matches)
        or (authority_present and not authority_matches)
    )

#: Depth multipliers for the ontogeny effort choice, mirrored here so the
#: allocation path costs no import. Kept in step with
#: ``core.ontogeny.control_points.EFFORT_MULTIPLIER``, which is the contract.
_EFFORT_MULTIPLIER = {"lean": 0.75, "standard": 1.0, "deep": 1.3}

# Explicit flag vocabularies. Anything outside both is a configuration
# error, not a silent activation (CP126 d9a04e05).
_TRUTHY_FLAG_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
_FALSEY_FLAG_VALUES = frozenset({"0", "false", "no", "off", "disabled", ""})

_CONTROLLER_SURFACE_OVERRIDE_KEYS = {
    "decode_max_tokens",
    "decode_temperature",
    "decode_top_p",
}


#: A margin so the episode lands, is proven, and releases the owner strictly
#: inside the window rather than at its edge.
_LATENT_WATCHDOG_MARGIN_S = 5.0


def _runtime_bounded_wall_clock_s(
    requested_s: float,
    *,
    foreground_request: bool = False,
) -> float:
    """Clamp a latent wall-clock allowance to what the worker tolerates.

    The lane may only plan for time the runtime will actually give it. Asking
    the client for its own ceiling makes this a derivation rather than a
    second constant to keep in sync — and if the client cannot be reached, the
    requested value stands, because guessing a smaller number here would be
    the same mistake in the other direction.
    """
    requested = max(1.0, float(requested_s))
    try:
        from core.brain.llm.mlx_client import get_mlx_client

        client = get_mlx_client()
        ceiling = float(
            client._first_token_hard_ceiling(foreground_request=foreground_request)
        )
    except Exception as exc:  # noqa: BLE001 - a budget may never take a turn down
        logger.debug("Latent wall-clock bound unavailable, using the request: %s", exc)
        return requested
    if ceiling <= 0.0:
        return requested
    return max(1.0, min(requested, ceiling - _LATENT_WATCHDOG_MARGIN_S))


def _controller_accepts_overrides(overrides: dict[str, Any] | None) -> bool:
    """Keep live answer-surface tuning from disabling cognitive control.

    Structural overrides still opt out because experiment and lesion callers
    must receive the exact recurrence/branch/schedule arm they requested.
    """

    return overrides is None or set(overrides) <= _CONTROLLER_SURFACE_OVERRIDE_KEYS


def _cortex_enabled() -> bool:
    # CP126 d9a04e05. This enabled the cortex for every value except the
    # exact string "0", so "false", "no", "disabled", "off" and any typo all
    # ACTIVATED resident latent execution — the opposite of what the operator
    # wrote. A flag that ignores what it was told is worse than no flag.
    raw = str(os.environ.get("AURA_LATENT_CORTEX", "1")).strip().lower()
    if raw in _TRUTHY_FLAG_VALUES:
        return True
    if raw in _FALSEY_FLAG_VALUES:
        return False
    # Neither: the operator meant something we do not understand. Default on
    # (this subsystem is on by default) but say so, rather than silently
    # reinterpreting the instruction.
    record_degradation(
        "latent_cortex_service",
        ValueError(f"unrecognised AURA_LATENT_CORTEX value {raw!r}"),
        severity="warning",
        action="kept the latent cortex at its default state after an unreadable flag",
    )
    return True


def _integrity_verdict(
    receipt: Any,
    claim: str,
    *,
    expected_worker_identity: dict[str, Any] | None = None,
) -> str:
    """Return the independently reconstructed measured integrity verdict."""
    if not isinstance(receipt, dict):
        return "unproven"
    try:
        from core.brain.llm.latent_cortex.runtime_integrity import (
            runtime_integrity_claim_verdict,
        )

        worker_identity = receipt.get("worker_identity")
        if not isinstance(worker_identity, dict):
            return "unproven"
        return runtime_integrity_claim_verdict(
            receipt.get("runtime_integrity"),
            claim,
            expected_episode_id=str(receipt.get("episode_id") or ""),
            expected_input_tokens_sha256=str(
                receipt.get("input_tokens_sha256") or ""
            ),
            expected_worker_identity=(
                expected_worker_identity or worker_identity
            ),
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
    except (ImportError, TypeError, ValueError):
        return "unproven"


def _erased_layers_declared(receipt: Any) -> bool:
    """Whether the teardown enumerated the layers it claims to have cleared."""
    if not isinstance(receipt, dict):
        return False
    proof = receipt.get("weight_integrity")
    if not isinstance(proof, dict):
        return False
    layers = proof.get("erased_layer_ids")
    return bool(isinstance(layers, (list, tuple)) and layers)


def _controller_outcome(
    verifier_evidence: Any,
) -> tuple[float, bool, bool, str]:
    """Extract only independently graded task outcomes for bandit learning."""

    if not isinstance(verifier_evidence, dict):
        return 0.0, False, False, "verifier_evidence_missing"
    raw_best = verifier_evidence.get("best_score")
    best_score = 0.0
    if (
        isinstance(raw_best, (int, float))
        and not isinstance(raw_best, bool)
        and math.isfinite(float(raw_best))
    ):
        best_score = max(0.0, min(1.0, float(raw_best)))
    checked = verifier_evidence.get("outcome_checked") is True
    passed = checked and verifier_evidence.get("outcome_passed") is True
    reason = (
        "independent_grade"
        if checked
        else str(verifier_evidence.get("outcome_reason") or "task_ground_truth_unavailable")
    )
    return best_score, checked, passed, reason


def _resident_state_reusable(receipt: dict[str, Any]) -> bool:
    """Prove that a completed worker episode left the resident model reusable."""

    if receipt.get("params_unchanged") is not True:
        return False
    worker_identity = receipt.get("worker_identity")
    runtime_integrity = receipt.get("runtime_integrity")
    if not isinstance(worker_identity, dict) or not isinstance(runtime_integrity, dict):
        return False
    try:
        from core.brain.llm.latent_cortex.runtime_integrity import (
            runtime_integrity_safe,
        )

        return runtime_integrity_safe(
            runtime_integrity,
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
            expected_checkpoint_file_count=int(
                receipt.get("checkpoint_file_count") or 0
            ),
        )
    except (ImportError, TypeError, ValueError, OverflowError):
        return False


# What to assume when the body cannot be read. 0.0 means "maximum headroom"
# on this scale, so unknown must never map to it. High enough to damp heavy
# allocation, low enough that a body which never reports does not freeze the
# service outright.
#: Wall-clock reserved for handoff, decode flush and receipt writing, so the
#: episode does not consume the caller's entire deadline (CP126 d607a287).
_DEADLINE_RESERVE_S = 8.0
#: On short deadlines a constant reserve would consume most of the budget, so
#: the reserve is proportional instead.
_DEADLINE_RESERVE_FRACTION = 0.2
#: Below this there is no episode worth planning; the caller gets the floor
#: and the overrun is theirs to see rather than hidden in a larger budget.
_MIN_USABLE_WALL_CLOCK_S = 0.5

#: Receipt size budget (CP126 09f2fbcf). A receipt is evidence about one
#: episode; past these bounds it is a payload, and the facade refuses to walk
#: it rather than spending the process's memory on a worker's output.
_MAX_RECEIPT_KEYS = 512
_MAX_RECEIPT_ITEMS = 200_000
_MAX_RECEIPT_DEPTH = 24

#: Request schema bounds (CP126 1a992727). An oversized or malformed request
#: reaches IPC and the calibration stores that learn from it, so it is refused
#: at the door rather than truncated somewhere downstream.
_MAX_QUESTION_CHARS = 200_000
_MAX_MESSAGES = 512
_MAX_MESSAGES_CHARS = 400_000
_MAX_DOMAIN_CHARS = 64
_ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})
_ALLOWED_MESSAGE_KEYS = frozenset({"role", "content", "name", "tool_call_id"})

_UNKNOWN_BODY_PRESSURE = 0.6


def _unit_signal(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return min(1.0, max(0.0, number))


def _check_the_exchange_count_contract(
    *,
    config: Any,
    errors: Any,
    exchanges: Any,
    receipt: Any,
    resource_accounting: Any,
) -> None:
    """Check the exchange-count half of the receipt contract.

    Moved out of ``LatentCortexService._receipt_contract_errors`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 5 name(s) from the turn and hands back
    0.
    """
    if type(exchanges) is int and exchanges >= 0 and (
        exchanges > 0 or receipt.get("branch_exchange") not in ({}, None)
    ):
        try:
            from core.brain.llm.latent_cortex.branch_exchange import (
                validate_branch_exchange_trace,
            )

            exchange_trace = validate_branch_exchange_trace(
                receipt.get("branch_exchange"),
                exchange_count=exchanges,
                n_branches=int(config.get("n_branches")),
                n_slots=int(config.get("n_slots")),
                comm_slot=int(config.get("comm_slot", 0)),
                exchange_gamma=float(config.get("exchange_gamma", 0.35)),
                branch_isolation=receipt.get("branch_isolation"),
                cognitive_slots=receipt.get("cognitive_slots"),
                exchange_interval=int(config.get("exchange_interval", 4)),
                schedule_hash=str(receipt.get("schedule_hash") or ""),
                bytecode_events=receipt.get("bytecode_events"),
                cognitive_action_trace=receipt.get("cognitive_action_trace"),
            )
            expected_reads = 0
            expected_writes = 0
            expected_scalar_ops = 0
            for exchange_row in exchange_trace["exchanges"]:
                accounting = exchange_row["tensor_accounting"]
                expected_reads += accounting["source_elements_read"]
                expected_writes += (
                    accounting["message_elements_emitted"]
                    + accounting["consensus_elements_written"]
                )
                expected_scalar_ops += accounting["tensor_scalar_ops"]
            operation = (
                resource_accounting.get("operations", {}).get("branch_exchange")
                if resource_accounting is not None
                else None
            )
            if exchanges == 0 and operation is None:
                return
            if (
                not isinstance(operation, dict)
                or operation.get("tensor_element_reads") != expected_reads
                or operation.get("tensor_element_writes") != expected_writes
                or operation.get("tensor_scalar_ops") != expected_scalar_ops
                or any(
                    operation.get(name) != 0
                    for name in operation
                    if name
                    not in {
                        "tensor_element_reads",
                        "tensor_element_writes",
                        "tensor_scalar_ops",
                    }
                )
            ):
                errors.append("branch_exchange_resource_binding_unproven")
        except (ImportError, TypeError, ValueError):
            errors.append("branch_exchange_provenance_unproven")
    elif receipt.get("branch_exchange") not in ({}, None):
        errors.append("unexpected_branch_exchange_trace")


def _check_the_latent_optimiser_contract(
    *,
    config: Any,
    errors: Any,
    nonnegative_int: Any,
    positive_int: Any,
    receipt: Any,
    verifier_arbitration_valid: Any,
) -> None:
    """Check the latent-optimiser half of the receipt contract.

    Moved out of ``LatentCortexService._receipt_contract_errors`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 6 name(s) from the turn and hands back
    0.
    """
    if config.get("latent_opt") is True:
        if receipt.get("latent_opt_applied") is not True:
            errors.append("latent_optimization_not_applied")
        if receipt.get("latent_opt_mode") != "gradient":
            errors.append("latent_optimization_wrong_mode")
        if not positive_int(receipt, "latent_opt_attempts"):
            errors.append("latent_optimization_not_attempted")
        # Under verifier guidance, zero ACCEPTED steps is a legitimate
        # verified outcome (every proposal was checked and declined) —
        # the verifier evidence must exist to earn that exemption.
        verifier_evidence = receipt.get("verifier_guidance")
        # CP126 94593618: int() on a worker-supplied field raises on a
        # string or a list, and this sits OUTSIDE any protective
        # conversion — so a malformed receipt escaped the contract as an
        # exception instead of the promised ok=false with a reason. A
        # validator that can be crashed by the thing it validates is not
        # a validator.
        verifier_ran = (
            isinstance(verifier_evidence, dict)
            and type(verifier_evidence.get("evaluations")) is int
            and verifier_evidence["evaluations"] > 0
        )
        if not positive_int(receipt, "latent_opt_steps") and not verifier_ran:
            errors.append("latent_optimization_no_accepted_steps")
        if not nonnegative_int(receipt, "latent_opt_rejected"):
            errors.append("latent_optimization_rejection_count_invalid")
        elif (
            positive_int(receipt, "latent_opt_attempts")
            and nonnegative_int(receipt, "latent_opt_steps")
            and (
                receipt["latent_opt_attempts"]
                != receipt["latent_opt_steps"] + receipt["latent_opt_rejected"]
            )
        ):
            errors.append("latent_optimization_accounting_mismatch")
        if receipt.get("latent_opt_budget_exhausted") is not False:
            errors.append("latent_optimization_budget_exhausted")
        if config.get("verifier_accept_non_regression") is True:
            arbitration = receipt.get("latent_opt_verifier")
            if not verifier_arbitration_valid(
                arbitration,
                attempts=int(receipt.get("latent_opt_attempts") or 0),
                accepted_steps=int(receipt.get("latent_opt_steps") or 0),
            ):
                errors.append("latent_optimization_verifier_receipt_invalid")


def _recurrence_halt_reason(receipt: Mapping[str, Any]) -> str:
    """Why the episode stopped short, when it recorded a reason.

    The stop gate writes ``halt_reason`` per branch. An episode that halted for
    a named reason ran its recurrence and chose to stop; an episode with no
    reason at all did not run it. Only the second is a fault, and telling them
    apart is what stops an adaptive halt reading as a broken request.
    """
    gate = receipt.get("stop_gate")
    branches = gate.get("branches") if isinstance(gate, Mapping) else None
    if not isinstance(branches, (list, tuple)):
        return ""
    reasons = [
        str(branch.get("halt_reason") or "")
        for branch in branches
        if isinstance(branch, Mapping) and branch.get("halt_reason")
    ]
    return reasons[0] if reasons else ""



from .latent_receipt_contract import _ChecksTheReceiptContract


from .latent_receipt_evidence import _ChecksTheReceiptEvidence


class LatentCortexService(_ChecksTheReceiptEvidence, _ChecksTheReceiptContract):
    """Budget allocation + IPC routing for latent-reasoning episodes."""

    def __init__(self, orchestrator: Any = None) -> None:
        self.orchestrator = orchestrator
        self._episodes = 0
        self._ok_episodes = 0
        self._last_receipt: dict[str, Any] = {}
        self._last_failure_receipt: dict[str, Any] = {}
        self._last_progress: dict[str, Any] = {}
        self._last_refusal = ""
        self._failure_streak = 0
        self._last_attempt_at = 0.0
        self._last_success_at = 0.0
        self._last_latency_s = 0.0
        self._last_allocation: dict[str, Any] = {}
        # Which controller decision has already had an outcome recorded.
        self._controller_outcome_recorded_for: str | None = None
        self._last_replay_sft_publication: dict[str, Any] = {}
        logger.info("🧠 LatentCortexService initialized (Recursive Latent Cortex)")

    @staticmethod
    async def _capture_verified_replay(
        *,
        receipt: dict[str, Any],
        private_evidence: Any,
        objective: str,
        output_text: Any,
        output_tokens: Any,
        output_quality: Any,
    ) -> dict[str, Any]:
        """Persist a proven local correction without blocking the event loop."""

        replacement = receipt.get("answer_replacement")
        if not (
            isinstance(replacement, dict)
            and replacement.get("decision") == "replace"
        ):
            return {
                "schema": "aura.rlc.verified_replay_host.v1",
                "status": "not_applicable",
                "reason": "no_applied_verified_local_repair",
                "learning_effect": "none",
            }
        if (
            not isinstance(private_evidence, dict)
            or not isinstance(output_text, str)
            or not isinstance(output_tokens, list)
            or not isinstance(output_quality, dict)
        ):
            return {
                "schema": "aura.rlc.verified_replay_host.v1",
                "status": "not_persisted",
                "reason": "verified_repair_private_evidence_unavailable",
                "learning_effect": "none",
            }
        try:
            from core.brain.llm.latent_cortex.verified_replay_buffer import (
                persist_runtime_verified_replay,
                validate_verified_replay_receipt,
            )

            stored = await asyncio.to_thread(
                persist_runtime_verified_replay,
                receipt=receipt,
                private_evidence=private_evidence,
                objective=objective,
                output_text=output_text,
                output_tokens=output_tokens,
                output_quality=output_quality,
            )
            stored = validate_verified_replay_receipt(stored)
        except (
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            record_degradation(
                "latent_cortex.verified_replay",
                exc,
                action=(
                    "returned the verified answer but did not grant learning "
                    "authority or persist an unverifiable repair trace"
                ),
                severity="warning",
            )
            return {
                "schema": "aura.rlc.verified_replay_host.v1",
                "status": "not_persisted",
                "reason": f"{type(exc).__name__}",
                "learning_effect": "none",
            }
        return {
            "schema": "aura.rlc.verified_replay_host.v1",
            "status": stored["status"],
            "reason": "",
            "learning_effect": "encrypted_verified_experience_retained",
            "experience_sha256": stored["experience_sha256"],
            "entry_sha256": stored["entry_sha256"],
            "sequence": stored["sequence"],
            "store_revision": stored["store_revision"],
            "store_sha256": stored["store_sha256"],
            "entry_count": stored["entry_count"],
            "retired_count": stored["retired_count"],
            "receipt_sha256": stored["receipt_sha256"],
        }

    async def publish_verified_replay_sft(
        self,
        *,
        privacy_clearances: dict[str, dict[str, Any]],
        reference_index: dict[str, Any],
        publication_root: Path | None = None,
        replay_path: Path | None = None,
        partition_ratios: dict[str, int] | None = None,
        minimum_rows_per_split: int = 1,
    ) -> dict[str, Any]:
        """Publish a Horcrux-backed replay snapshot without training authority."""

        from core.learning.verified_replay_sft_publication import (
            publish_runtime_verified_replay_sft,
        )

        report = await asyncio.to_thread(
            publish_runtime_verified_replay_sft,
            privacy_clearances=privacy_clearances,
            reference_index=reference_index,
            publication_root=publication_root,
            replay_path=replay_path,
            partition_ratios=partition_ratios,
            minimum_rows_per_split=minimum_rows_per_split,
        )
        self._last_replay_sft_publication = dict(report)
        return dict(report)

    # ── Cognitive economy ───────────────────────────────────────────────

    def _body_pressure(self) -> float:
        """Total real+anticipatory body pressure in [0, 1].

        CP126 9bc8e55e. Unknown used to mean 0.0 — which in this scale is
        "fully healthy, maximum headroom". So an absent, incompatible,
        stale or failing body was rewarded with UNDAMPED compute, silently,
        with no degradation receipt. The signal disappeared in exactly the
        state where it was most needed.

        Unknown now means conservatively pressured: enough to damp heavy
        allocation, not so much that a body which never reports freezes the
        service. The failure is recorded, so a persistent unknown is visible
        rather than indistinguishable from a calm body.
        """
        try:
            from core.being.aura_now import BodyState

            state = getattr(self.orchestrator, "state", None)
            # total_pressure is a @property, not a method. Calling it raised
            # TypeError: 'float' object is not callable on EVERY invocation,
            # so this function has never once returned a real reading — it
            # always fell into the handler below, which used to answer 0.0
            # and hand the latent cortex undamped compute while reporting
            # nothing. Found in a live log only after that handler was made
            # to record itself.
            return float(BodyState.from_aura_state(state).total_pressure)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            if not getattr(self, "_body_pressure_unknown_reported", False):
                self._body_pressure_unknown_reported = True
                record_degradation(
                    "latent_cortex_service",
                    exc,
                    severity="warning",
                    action=(
                        "body pressure unobservable; damping allocation at "
                        f"{_UNKNOWN_BODY_PRESSURE} instead of assuming full headroom"
                    ),
                )
            return _UNKNOWN_BODY_PRESSURE

    def _runtime_pressure_snapshot(self) -> dict[str, Any]:
        """Read the canonical admission signal without creating new policy."""

        try:
            from core.runtime.control_plane import get_runtime_control_plane

            snapshot = get_runtime_control_plane().admission.pressure_snapshot()
            payload = snapshot.to_dict()
            if not isinstance(payload, dict):
                raise TypeError("runtime pressure snapshot is not a mapping")
            return payload
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            if not getattr(self, "_runtime_pressure_unknown_reported", False):
                self._runtime_pressure_unknown_reported = True
                record_degradation(
                    "latent_cortex_service",
                    exc,
                    severity="warning",
                    action=(
                        "runtime pressure unavailable; adaptive compute uses "
                        "conservative unknown-resource headroom"
                    ),
                )
            return {
                "observation_source": "unavailable",
                "resource_observation_available": False,
                "red_zones": ["pressure_provider_unavailable"],
            }

    #: How much above-average novelty may add to effective uncertainty.
    #:
    #: Bounded on purpose. Novelty is a *measurement* — how far the current
    #: state sits from the centre of the distribution Aura has actually lived
    #: — not a model's prediction, so it needs no earned authority to be acted
    #: on. But it is one signal among several, and a situation being unfamiliar
    #: is a reason to think a little harder, never a reason to abandon the
    #: stakes and uncertainty the caller actually measured.
    _NOVELTY_EFFORT_WEIGHT = 0.2

    def _novelty_adjusted_uncertainty(self, uncertainty: float) -> tuple[float, float | None]:
        """Let an unfamiliar situation buy a little more thought.

        The allocator has always scaled depth with stakes and uncertainty, both
        of which describe the *task*. Neither notices that Aura has never been
        anywhere like this before, which is exactly when the surface of a
        problem is least informative about its difficulty.

        Only above-average novelty counts. An ordinary moment is left exactly
        as the caller measured it — a signal that moves every allocation is a
        signal that has stopped saying anything.
        """
        try:
            from core.ontogeny.service import get_ontogeny

            novelty = float(get_ontogeny().novelty())
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
            record_degradation(
                "latent_cortex_service", exc, severity="debug",
                action="novelty unavailable; allocation uses the caller's uncertainty alone",
            )
            return uncertainty, None
        excess = max(0.0, novelty - 0.5) * 2.0
        adjusted = min(1.0, uncertainty + self._NOVELTY_EFFORT_WEIGHT * excess)
        return adjusted, novelty

    def _effort_choice(
        self, *, stakes: float, uncertainty: float, novelty: float | None,
        foreground: bool, model_parameter_count: int,
    ) -> tuple[str, str | None]:
        """Ask the organ how hard to think. Returns (effort, episode_id).

        Unlike novelty — which is a measurement and needs no permission — this
        is a learned *choice*, so it goes through the full ladder and returns
        the incumbent's "standard" until a head has earned otherwise. It is
        also the control point that is hardest to grade honestly: whether a
        depth was right is only answerable once a verifier has graded the
        answer, which happens elsewhere and often not at all. The resolver
        refuses every proxy for that, so this may sit unpromoted indefinitely.
        That is the design working, not the design failing.
        """
        try:
            import math as _math

            from core.ontogeny.control_points import COGNITION_EFFORT
            from core.ontogeny.service import get_ontogeny

            angle = 2.0 * _math.pi * (time.localtime().tm_hour / 24.0)
            verdict = get_ontogeny().consider(
                COGNITION_EFFORT,
                {
                    "stakes": float(stakes),
                    "uncertainty": float(uncertainty),
                    "novelty": float(novelty if novelty is not None else 0.5),
                    "body_pressure": float(self._body_pressure()),
                    "foreground": 1.0 if foreground else 0.0,
                    "resident_scale": 1.0 if model_parameter_count >= 20_000_000_000 else 0.0,
                    "hour_of_day_sin": _math.sin(angle),
                    "hour_of_day_cos": _math.cos(angle),
                },
                incumbent_choice="standard",
                seed=f"effort:{stakes:.3f}:{uncertainty:.3f}:{time.time():.3f}",
                # High-stakes thinking is never explored with. The exploration
                # slice buys evidence with latency, not with the quality of an
                # answer somebody is waiting on.
                stakes=max(float(stakes), 0.75 if foreground else 0.0),
            )
            return verdict.choice, verdict.episode_id
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
            record_degradation(
                "latent_cortex_service", exc, severity="debug",
                action="effort left at the allocator's own depth",
            )
            return "standard", None

    def allocate(
        self,
        *,
        stakes: float,
        uncertainty: float,
        objective: str = "",
        model_parameter_count: int = 0,
        foreground_request: bool = False,
        timeout_s: float | None = None,
        requested_decode_tokens: int | None = None,
        private_reasoning_tokens: int = 0,
    ) -> tuple[dict, dict]:
        """(config, budget) for one episode: a POLICY allocation.

        More stakes/uncertainty ⇒ deeper recurrence, wider branches, bigger
        budget. Body pressure damps everything — deep thought is a luxury a
        strained body rations first.

        CP126 60007362: this said "the Will's thought allocation". It is not.
        No Will service is consulted, no scoped authority is taken, no policy
        decision is made and no allocation receipt is signed by governance —
        the method clamps caller-supplied stakes and uncertainty, reads body
        pressure, and applies fixed arithmetic. Naming a heuristic after the
        governance organ makes it look authorized when nothing authorized it,
        which is exactly the shape of claim this campaign removes.

        What it IS is a deterministic compute policy, and the receipt returned
        alongside the budget records the inputs and coefficients so the
        arithmetic can be checked rather than trusted (CP126 8a7e39cc).
        """
        stakes = _unit_signal(stakes, name="stakes")
        uncertainty = _unit_signal(uncertainty, name="uncertainty")
        uncertainty, novelty = self._novelty_adjusted_uncertainty(uncertainty)
        effort, effort_episode = self._effort_choice(
            stakes=stakes, uncertainty=uncertainty, novelty=novelty,
            foreground=foreground_request, model_parameter_count=model_parameter_count,
        )
        try:
            pressure = _unit_signal(self._body_pressure(), name="body_pressure")
        except ValueError:
            # Same reasoning: an invalid reading is not evidence of headroom.
            pressure = _UNKNOWN_BODY_PRESSURE
        if (
            isinstance(model_parameter_count, bool)
            or not isinstance(model_parameter_count, int)
            or model_parameter_count < 0
        ):
            raise ValueError("model_parameter_count must be a non-negative integer")
        if type(foreground_request) is not bool:
            raise ValueError("foreground_request must be a boolean")
        if not isinstance(objective, str):
            raise ValueError("objective must be text")
        if requested_decode_tokens is not None and (
            type(requested_decode_tokens) is not int
            or requested_decode_tokens <= 0
        ):
            raise ValueError("requested_decode_tokens must be a positive integer")
        if type(private_reasoning_tokens) is not int or private_reasoning_tokens < 0:
            raise ValueError("private_reasoning_tokens must be a non-negative integer")
        owner_timeout_s: float | None = None
        if timeout_s is not None:
            try:
                owner_timeout_s = float(timeout_s)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("timeout_s must be finite and positive") from exc
            if not math.isfinite(owner_timeout_s) or owner_timeout_s <= 0.0:
                raise ValueError("timeout_s must be finite and positive")
        headroom = 1.0 - 0.7 * pressure

        # The organ's effort choice scales depth inside the same hard bounds
        # the allocator has always enforced. It may tune how hard she thinks;
        # it may not remove the floor or raise the ceiling.
        effort_scale = _EFFORT_MULTIPLIER.get(effort, 1.0)
        max_steps = max(2, min(16, round((4 + 10 * uncertainty) * headroom * effort_scale)))
        n_branches = 1 if stakes < 0.3 else (3 if stakes > 0.75 and headroom > 0.6 else 2)
        intensity = max(stakes, uncertainty)
        latent_opt_steps = max(1, min(4, round((1 + 3 * intensity) * headroom)))
        fast_weight_steps = max(1, min(3, round((1 + 2 * intensity) * headroom)))
        fast_weight_layers = max(2, min(8, round((2 + 6 * intensity) * headroom)))
        config = {
            "n_slots": 16,
            # This is an execution-memory bound, not a reasoning reduction.
            # Exact cache state connects chunks while transient layer graphs
            # are released between them.
            "prefill_chunk_tokens": 128,
            "max_steps": max_steps,
            "min_steps": 2,
            "n_branches": n_branches,
            "isolation_steps": 2,
            "alpha_schedule": "cosine",
            # The production service exercises the complete machine. Ablation
            # arms belong to the falsification harness, never the live default.
            "latent_opt": True,
            "latent_opt_steps": latent_opt_steps,
            "latent_opt_lr": 0.03,
            "fast_weights": True,
            "fast_weights_rank": 2,
            "fast_weights_opt_steps": fast_weight_steps,
            "fast_weights_lr": 0.005,
            "fast_weights_max_layers": fast_weight_layers,
            "fast_weights_canary_max_delta_rms": 0.05,
            "decode_max_tokens": 512,
            "verifier_probe_max_tokens": 48,
            "verifier_accept_non_regression": False,
            "generative_verifier_enabled": True,
            "generative_verifier_max_atoms": 1,
            "generative_verifier_max_tokens": 160,
            "counterfactual_verifier_enabled": True,
            "counterfactual_verifier_max_atoms": 1,
            "counterfactual_verifier_max_interventions": 2,
            "counterfactual_verifier_max_tokens": 128,
            "prefix_stability_enabled": True,
            "prefix_stability_samples": 3,
            "prefix_stability_max_tokens": 128,
            "prefix_stability_temperature": 0.35,
            "prefix_stability_top_p": 0.9,
            "prefix_stability_seed": 104_729,
            "prefix_stability_calibrator": None,
        }
        budget = {
            "max_layer_apps": int((2_000_000 + 8_000_000 * stakes) * headroom),
            # BOUNDED BY WHAT THE RUNTIME WILL ACTUALLY ALLOW.
            #
            # This computed up to 120s on its own authority while the MLX
            # worker's progress watchdog cancels any job showing no TOKEN
            # activity past its first-token ceiling — 40s on the foreground
            # lane. branch_select does layer applications and emits no
            # tokens, so a perfectly healthy latent episode looks exactly
            # like a livelocked one from outside.
            #
            # Measured live 2026-07-28: "Did you know the Earth's core is
            # cold?" — answered twice in twelve seconds earlier the same
            # evening — spent 88.6s and 177,120 layer applications in
            # branch_select, blew the 40s budget, was soft-cancelled, and
            # the person got "I couldn't get to an answer I'd stand behind."
            # Two budgets, neither aware of the other, and the lane could
            # never use the allowance it granted itself.
            "wall_clock_s": _runtime_bounded_wall_clock_s(
                30.0 + 90.0 * stakes * headroom,
                foreground_request=foreground_request,
            ),
            # Recorded so an allocation can be explained after the fact: a
            # deeper-than-usual episode should be traceable to the reason it
            # was deeper, not just observed to have been.
            "effective_uncertainty": round(uncertainty, 4),
            "novelty": round(novelty, 4) if novelty is not None else None,
            "effort": effort,
            "ontogeny_episode": effort_episode,
        }
        allocation_profile = "general_full_stack_v1"
        if foreground_request and model_parameter_count >= 20_000_000_000:
            # Interactive resident-scale profile: every production mechanism
            # remains causal, but the cortex lane receives a bounded amount of
            # virtual width and optimizer work instead of a small-model lab
            # schedule that cannot meet the desktop deadline.
            allocation_profile = "resident_32b_interactive_full_stack_v2"
            config.update(
                {
                    # Nine positions preserve one mailbox, six organ evidence
                    # rows, an optional one-shot continuation fragment, and a
                    # persistent private hypothesis. The prior four-slot
                    # profile silently dropped most admitted evidence.
                    "n_slots": 9,
                    "max_steps": 2,
                    "min_steps": 2,
                    "n_branches": 2 if stakes >= 0.3 else 1,
                    "exchange_interval": 1,
                    "latent_opt_steps": 1,
                    "fast_weights_opt_steps": 1,
                    "fast_weights_max_layers": 2,
                    # Mechanically-clean episode synapses become durable
                    # learning CANDIDATES (consumer + compounding gates
                    # decide; nothing consolidates from inside an episode).
                    "fast_weights_export_candidates": True,
                    "decode_max_tokens": 256,
                    "decode_bridge_policy": "assistant_answer_v1",
                    # Product probes are previews used for branch/adaptation
                    # arbitration, not user-facing drafts. CP120 measured five
                    # 48-token probes consuming ~68s; 24 tokens plus verified
                    # branch-baseline reuse preserves the answer budget while
                    # keeping every arbitration mechanism causal and receipted.
                    "verifier_probe_max_tokens": 24,
                    "verifier_accept_non_regression": True,
                    "generative_verifier_enabled": True,
                    "generative_verifier_max_atoms": 1,
                    # The strict JSON contract includes a 64-character claim
                    # commitment; shorter budgets truncate before the witness.
                    "generative_verifier_max_tokens": 128,
                    "counterfactual_verifier_enabled": True,
                    "counterfactual_verifier_max_atoms": 1,
                    "counterfactual_verifier_max_interventions": 2,
                    "counterfactual_verifier_max_tokens": 96,
                    "prefix_stability_enabled": True,
                    "prefix_stability_samples": 3,
                    "prefix_stability_max_tokens": 128,
                    "prefix_stability_temperature": 0.35,
                    "prefix_stability_top_p": 0.9,
                    "prefix_stability_seed": 104_729,
                    "prefix_stability_calibrator": None,
                    "input_context_max_chars": 9000,
                    # The live cortex is an optional improvement over the
                    # resident model's ordinary answer. Admission already
                    # reserves a clean prefill+decode for this path, so a
                    # numerically invalid recurrent phase must spend that
                    # reserve instead of turning healthy chat into a total
                    # failure. Scientific and lesion callers still provide
                    # ``allow_vanilla_fallback=False`` explicitly; only this
                    # interactive product profile is recovery-capable.
                    "allow_vanilla_fallback": True,
                }
            )
        # CP126 d607a287: the caller's deadline was applied ONLY inside the
        # foreground >=20B branch, so every other profile ignored it entirely
        # and planned against a budget the caller could not wait for. And the
        # clamp that did exist used max(15.0, timeout - 8.0), which HANDS BACK
        # 15 seconds when the caller has 10 — a floor that outranks the
        # deadline is not a deadline.
        #
        # Every profile is clamped, and the reserve scales with what is
        # actually available so a short deadline still leaves margin instead
        # of being overrun by a constant.
        if owner_timeout_s is not None:
            reserve = min(_DEADLINE_RESERVE_S, owner_timeout_s * _DEADLINE_RESERVE_FRACTION)
            usable = max(_MIN_USABLE_WALL_CLOCK_S, owner_timeout_s - reserve)
            # Foreground work already has a caller-owned deadline. Stakes
            # govern optional computation, not a second cancellation clock.
            budget["wall_clock_s"] = (
                usable
                if foreground_request
                else min(float(budget["wall_clock_s"]), usable)
            )
        if requested_decode_tokens is not None:
            # Answer-surface overrides are part of the allocation request, not
            # a post-hoc mutation. The adaptive plan must commit the same floor
            # that its learned-controller and worker stages will enforce.
            config["decode_max_tokens"] = requested_decode_tokens
        answer_capacity = int(config["decode_max_tokens"])
        if private_reasoning_tokens:
            config["decode_max_tokens"] = max(
                answer_capacity, min(8192, answer_capacity + private_reasoning_tokens),
            )
        private_allowance = int(config["decode_max_tokens"]) - answer_capacity
        from core.brain.llm.latent_cortex.adaptive_compute import (
            apply_adaptive_compute_plan,
            build_adaptive_compute_plan,
        )

        adaptive_plan = build_adaptive_compute_plan(
            objective=objective,
            stakes=stakes,
            uncertainty=uncertainty,
            body_pressure=pressure,
            deadline_s=(
                owner_timeout_s
                if owner_timeout_s is not None
                else float(budget["wall_clock_s"])
            ),
            resource_snapshot=self._runtime_pressure_snapshot(),
            foreground_request=foreground_request,
            model_parameter_count=model_parameter_count,
            requested_decode_tokens=int(config["decode_max_tokens"]),
            isolation_steps=int(config["isolation_steps"]),
        )
        config, budget = apply_adaptive_compute_plan(config, budget, adaptive_plan)
        config["exchange_interval"] = _an_interval_a_return_can_travel(config)
        # CP126 8a7e39cc: the coefficients and thresholds below are a fixed
        # heuristic with no model-specific calibration, no uncertainty
        # interval, no control-policy comparison and no safety-outcome
        # evidence. That is a defensible starting policy and an indefensible
        # thing to leave unstated, so the receipt records the inputs, the
        # profile, the deadline actually granted, and — explicitly — that the
        # policy is uncalibrated. A reader can then check the arithmetic
        # instead of trusting the number, and the absence of calibration is a
        # visible gap rather than an implied endorsement.
        self._last_allocation = {
            "schema": "aura.latent_cortex.allocation_receipt.v1",
            "stakes": stakes,
            "uncertainty": uncertainty,
            "novelty": novelty,
            "effort": str(effort),
            "body_pressure": pressure,
            "headroom": headroom,
            "allocation_profile": allocation_profile,
            "model_parameter_count": model_parameter_count,
            "owner_timeout_s": owner_timeout_s,
            "granted_wall_clock_s": float(budget.get("wall_clock_s", 0.0)),
            "deadline_respected": (
                owner_timeout_s is None
                or float(budget.get("wall_clock_s", 0.0)) <= owner_timeout_s
            ),
            # Honesty fields: this is a policy, not a governed decision, and
            # its constants are not calibrated against outcomes.
            "authority": "policy_heuristic",
            "will_decision": None,
            "calibrated": False,
            "calibration_basis": (
                "hand-tuned coefficients; no model-specific calibration, "
                "uncertainty interval, or control-policy comparison"
            ),
            "adaptive_compute": adaptive_plan,
            "native_reasoning_allowance": {
                "answer_capacity_tokens": answer_capacity,
                "requested_private_tokens": private_reasoning_tokens,
                "admitted_private_tokens": private_allowance,
            },
            "config": dict(config),
            "budget": dict(budget),
        }
        return config, budget







    @staticmethod
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
        """Return a deterministic, auditable decision for live latent routing."""
        # Kept as the public service API for existing callers, but owned by a
        # lightweight module so an ordinary route decision does not import the
        # numerical RLC engine before it has actually been selected.
        from core.brain.foreground_latent_runtime import select_foreground_episode

        return select_foreground_episode(
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

    @staticmethod
    def _attestation_disclosure(receipt: Any) -> dict[str, Any]:
        """What this episode's evidence actually establishes.

        The honest floor for four findings that share one root: the facade has
        no trust domain separate from the worker it is checking.

        * ``e1c09324`` — identity fields (booleans, counts, hashes, git ids,
          schedule hashes) are checked for FORMAT and internal consistency,
          never recomputed from trusted local files or a signed attestation.
        * ``9b110bc5`` — ``params_unchanged`` cannot prove which effective
          model answered: active adapter/LoRA ids, hashes, ordering, scale,
          load generation and tokenizer identity are not bound.
        * ``fc19e25e`` — the "independent" verifier replay recomputes
          decisions from scores and trails supplied in the SAME receipt. It
          reruns no task verifier and inspects no candidate output.
        * ``265b0fae`` — a successful episode ran no same-model vanilla
          control, no frontier baseline, no held-out correctness test and no
          calibration, so it establishes accepted EXECUTION, not superior
          reasoning.

        Closing these properly needs a signed worker attestation and an
        out-of-band verifier — a subsystem, not a patch. Until that exists the
        claims are stated at their true strength instead of being implied at a
        higher one.
        """
        receipt_map = receipt if isinstance(receipt, dict) else {}
        return {
            "schema": "aura.latent_cortex.attestation_disclosure.v1",
            "trust_domain": "worker_self_reported",
            "independently_verified": False,
            "proves": [
                "the receipt is well-formed and internally consistent",
                "the client bound the receipt to the request payload it sent",
                "the declared compute stayed within the facade's allocation",
            ],
            "does_not_prove": [
                "identity fields were recomputed from trusted local state",
                "which effective model (adapter/LoRA/tokenizer) produced the answer",
                "the verifier replay used evidence from a separate trust domain",
                "the answer is better than a vanilla or frontier baseline",
            ],
            "model_identity_complete": bool(
                receipt_map.get("active_adapters") is not None
                and receipt_map.get("tokenizer_sha256")
            ),
            "verifier_replay_independent": False,
            "reasoning_gain_established": False,
            "required_for_independence": [
                "signed worker attestation verified against an authority registry",
                "out-of-band verifier rerun on the candidate output",
                "same-model vanilla control on a held-out set",
            ],
        }

    @staticmethod
    def _request_schema_error(
        question: str | None, messages: list | None, domain: str
    ) -> str:
        """The first schema violation in this request, or "" (CP126 1a992727)."""
        if isinstance(question, str) and len(question) > _MAX_QUESTION_CHARS:
            return f"question_too_large:{len(question)}"
        if not isinstance(domain, str) or not domain.strip():
            return "invalid_domain"
        if len(domain) > _MAX_DOMAIN_CHARS or any(ord(ch) < 32 for ch in domain):
            return "invalid_domain"
        if messages is None:
            return ""
        if len(messages) > _MAX_MESSAGES:
            return f"too_many_messages:{len(messages)}"
        total_chars = 0
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                return f"message_{index}_not_mapping"
            unknown = set(message) - _ALLOWED_MESSAGE_KEYS
            if unknown:
                # Unknown fields are refused rather than ignored: a field this
                # layer does not understand is one it cannot bound, and it
                # still reaches the worker.
                return f"message_{index}_unknown_fields:{','.join(sorted(unknown))[:80]}"
            role = message.get("role")
            if not isinstance(role, str) or role not in _ALLOWED_MESSAGE_ROLES:
                return f"message_{index}_invalid_role"
            content = message.get("content")
            if not isinstance(content, str):
                return f"message_{index}_invalid_content"
            total_chars += len(content)
            if total_chars > _MAX_MESSAGES_CHARS:
                return f"messages_too_large:{total_chars}"
        return ""


    def _teach_controller_this_arm_failed(self, reason: str) -> None:
        """Tell the bandit about the arms that did NOT work.

        CP126 d4a5bb97. Controller learning lived only on the success path,
        so a failed, timed-out, invalid-receipt or quality-rejected arm
        produced no outcome at all. The bandit therefore saw only the
        episodes that worked — a costly arm that fails often could never be
        learned as costly, because its failures were invisible while its
        occasional successes were not. Selection bias, in a component whose
        entire job is choosing between arms.

        The outcome is recorded as ``checked=False``: this is an execution
        failure, not an independent grade of the answer. That distinction is
        the same one the success path already makes — a refusal is real
        evidence that the arm did not deliver, and it is not evidence about
        correctness.
        """
        decision = None
        try:
            allocation = getattr(self, "_last_allocation", None)
            if isinstance(allocation, dict):
                decision = allocation.get("execution_controller")
        except (AttributeError, TypeError):
            decision = None
        if not isinstance(decision, dict) or not decision.get("decision_id"):
            return
        # One decision, one outcome. A refusal that fires twice for the same
        # episode must not weight the arm twice. Keyed by decision id rather
        # than a flag on the dict: that dict is spread into the receipt, and a
        # bookkeeping key would become part of a published payload.
        decision_id = str(decision.get("decision_id") or "")
        if decision_id and decision_id == getattr(
            self, "_controller_outcome_recorded_for", None
        ):
            return
        self._controller_outcome_recorded_for = decision_id
        try:
            from core.brain.llm.latent_cortex.execution_controller import (
                get_execution_controller,
            )

            get_execution_controller().record_outcome(
                bucket=str(decision.get("bucket") or ""),
                arm=str(decision.get("arm") or "base"),
                verified_score=0.0,
                success=False,
                checked=False,
                wall_clock_s=0.0,
                decision_id=decision_id,
            )
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
        ) as exc:
            record_degradation(
                "latent_cortex_service",
                exc,
                severity="warning",
                action=(
                    "the execution controller did not learn that this arm failed; "
                    f"its statistics remain biased toward successes ({reason})"
                ),
            )

    def recurrence_depth_status(
        self, requested_loops: int | None = None
    ) -> dict[str, Any]:
        """What the last turn asked recurrence for, and what it did.

        Derived from the stored receipt rather than recorded during
        verification, because the verifier is a static method and mutating
        service state from it would tie a pure check to one instance.

        A surface that asks for two loops and gets one every turn is worth
        seeing even when the halt was correct, which is why an adaptive halt is
        reported here instead of disappearing once it stopped being an error.
        """
        receipt = self._last_receipt or {}
        steps = receipt.get("steps_taken")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            return {"measured": False}
        halt_reason = _recurrence_halt_reason(receipt)
        status: dict[str, Any] = {
            "measured": True,
            "steps_taken": steps,
            "halt_reason": halt_reason,
        }
        if isinstance(requested_loops, int) and not isinstance(requested_loops, bool):
            status["requested_loops"] = requested_loops
            if steps < requested_loops:
                status["state"] = "halted_early" if halt_reason else "depth_not_applied"
                status["expected"] = bool(halt_reason)
            else:
                status["state"] = "served_requested_depth"
                status["expected"] = True
        return status

    def _record_failure(
        self, reason: str, *, stage: str = "", evidence: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Refuse with a receipt tying the refusal to THIS call.

        CP126 5879d2b5: the module's contract promises a refusal carries
        bounded evidence, and this returned only ok and reason while mutating
        shared counters. Every disabled, invalid, busy, missing-model and
        client failure was therefore indistinguishable from every other
        instance of itself — no episode id, no stage, no timestamp, nothing
        connecting the refusal to the call that caused it. A failure lane that
        cannot be attributed is a failure lane that cannot be debugged, which
        is why the same refusals kept being re-diagnosed from scratch.
        """
        self._failure_streak += 1
        self._last_refusal = str(reason or "unknown")
        self._teach_controller_this_arm_failed(self._last_refusal)
        receipt: dict[str, Any] = {
            "schema": "aura.latent_cortex.refusal_receipt.v1",
            "refusal_id": f"refusal-{uuid.uuid4().hex[:12]}",
            "reason": self._last_refusal,
            # The reason's first segment is its class; the rest is detail.
            "reason_class": self._last_refusal.split(":", 1)[0],
            "stage": str(stage or "pre_dispatch"),
            "at": time.time(),
            "failure_streak": self._failure_streak,
            "ok_episodes": self._ok_episodes,
            "last_success_at": self._last_success_at,
        }
        if isinstance(evidence, dict):
            # Bounded: a refusal receipt must not become a payload.
            for index, (key, value) in enumerate(evidence.items()):
                if index >= 24:
                    receipt["evidence_truncated"] = True
                    break
                receipt[str(key)[:64]] = (
                    value if isinstance(value, (int, float, bool)) or value is None
                    else str(value)[:400]
                )
        return {
            "ok": False,
            "reason": self._last_refusal,
            "receipt": receipt,
            "refusal_receipt": receipt,
        }

    def foreground_admission(self) -> dict[str, Any]:
        """Return whether the unchanged general foreground path can succeed.

        A complete worker receipt that lacks the terminal decode bridge is a
        deterministic source-contract failure, not a stochastic model miss.
        Re-running the same source spends roughly a minute and returns the
        same refusal. Keep exact qualified neural ingress available (it runs
        before this gate), but circuit the general episode until this service
        restarts with changed code or records a successful episode.
        """

        reason = str(self._last_refusal or "")
        prefix = "receipt_contract_failed:"
        if self._failure_streak < 1 or not reason.startswith(prefix):
            return {"admitted": True, "reason": "no_deterministic_contract_failure"}
        failures = {
            item.strip()
            for item in reason[len(prefix) :].split(",")
            if item.strip()
        }
        terminal_bridge_failures = {
            "terminal_disposition_unproven",
            "answer_replacement_unproven",
            "decode_bridge_unapplied",
            "decode_bridge_tokens_missing",
            "decode_bridge_token_identity_unproven",
            "decode_bridge_logits_unproven",
        }
        matched = sorted(failures & terminal_bridge_failures)
        if not matched:
            return {"admitted": True, "reason": "contract_failure_may_be_transient"}
        return {
            "admitted": False,
            "reason": "unchanged_terminal_bridge_contract_failure",
            "failure_streak": self._failure_streak,
            "contract_failures": matched,
        }

    @staticmethod
    def _visible_objective(question: str | None, messages: list | None) -> str:
        if isinstance(question, str) and question.strip():
            return question.strip()
        for message in reversed(messages or []):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = [
                    str(item.get("text") or "").strip()
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                rendered = "\n".join(part for part in parts if part).strip()
                if rendered:
                    return rendered
        return ""

    @staticmethod
    def _facet_reliability_weights(domain: str) -> dict[str, float] | None:
        """Foundry-calibrated facet weights; None until any facet is measured.

        weight_for stays 1.0 below 10 graded verdicts, so this returns None
        (wire unchanged) until an operator has actually graded facet
        judgments — behavior never shifts on ungraded speculation.
        """
        try:
            from core.brain.llm.latent_cortex.task_verifiers import (
                _ANSWER_FACET_HINTS,
            )
            from core.brain.verifiers.foundry import get_verifier_foundry

            foundry = get_verifier_foundry()
            weights = {
                name: float(foundry.weight_for(f"latent_facet_{name}", str(domain)))
                for name in _ANSWER_FACET_HINTS
            }
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return None
        if all(value == 1.0 for value in weights.values()):
            return None
        return weights

    def _record_facet_judgments(self, receipt: dict[str, Any], domain: str, objective: str) -> None:
        """Feed the episode's facet judgments to the Foundry grade queue.

        Each judgment (facet, satisfied, excerpt) becomes an ungraded
        verdict an operator can grade against the excerpt — the held-out
        loop that keeps 'because'-without-explaining from ever paying.
        """
        guidance = receipt.get("verifier_guidance")
        if not isinstance(guidance, dict):
            return
        judgments = guidance.get("facet_judgments")
        if not isinstance(judgments, list) or not judgments:
            return
        try:
            import hashlib

            from core.brain.verifiers.foundry import get_verifier_foundry

            foundry = get_verifier_foundry()
            task_key = hashlib.sha256(str(objective or "").encode("utf-8")).hexdigest()[:16]
            for row in judgments[:8]:
                facet = row.get("facet") if isinstance(row, dict) else None
                if not isinstance(facet, str) or not facet:
                    continue
                # CP126 94ecfee0: bool() on a worker-supplied field turns any
                # non-empty string into True — including the string "false".
                # A facet the worker reported as unsatisfied could therefore be
                # recorded as a hard pass, and these verdicts train the
                # Foundry's reliability statistics.
                raw_satisfied = row.get("satisfied")
                if type(raw_satisfied) is not bool:
                    # Not a verdict. Recording it either way would invent one.
                    continue
                satisfied = raw_satisfied
                foundry.record_verdict(
                    verifier=f"latent_facet_{facet}",
                    domain=str(domain),
                    hard_pass=satisfied,
                    score=1.0 if satisfied else 0.0,
                    # CP126 94ecfee0: this is the WORKER's assertion about its
                    # own output, not an independent check. Recording it as
                    # checked=True let a self-report enter the Foundry with the
                    # standing of a verified grade; an operator grading against
                    # the excerpt is what makes it checked.
                    checked=False,
                    task_key=task_key,
                    meta={
                        "excerpt": str(row.get("excerpt") or "")[:200],
                        "source": "worker_self_assertion",
                        "independently_checked": False,
                    },
                )
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
        ) as exc:
            logger.debug("Facet judgment recording skipped: %s", exc)

    @staticmethod
    async def _broadcast_conclusion(
        result: dict[str, Any],
        *,
        objective: str,
        stakes: float,
    ) -> None:
        """Publish exactly the conclusion returned by a foreground call."""

        receipt = result.get("receipt")
        if result.get("ok") is not True or not isinstance(receipt, dict):
            return
        try:
            from core.brain.gwt_rlc_coupling import broadcast_episode_conclusion

            receipt["workspace_broadcast"] = await broadcast_episode_conclusion(
                objective,
                str(result.get("text") or ""),
                receipt,
                stakes=stakes,
            )
            result["receipt"] = receipt
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.debug("Workspace broadcast of conclusion skipped: %s", exc)

    async def deep_reason_with_acquisition(
        self,
        question: str | None = None,
        *,
        messages: list | None = None,
        orchestrator: Any = None,
        tenant_id: str = "local",
        user_id: str = "owner",
        session_id: str = "local",
        **reason_kwargs: Any,
    ) -> dict[str, Any]:
        """Run one episode plus at most one governed retrieval continuation."""

        started = time.monotonic()
        reason_kwargs = dict(reason_kwargs)
        reason_kwargs.pop("publish_workspace_conclusion", None)
        if reason_kwargs.get("external_execution_offer") is not None:
            return self._record_failure(
                "external_execution_requires_single_episode"
            )
        foreground = reason_kwargs.get("foreground_request", True)
        first = await self.deep_reason(
            question,
            messages=messages,
            publish_workspace_conclusion=False,
            **reason_kwargs,
        )
        if first.get("ok") is not True:
            return first

        objective = self._visible_objective(question, messages)
        first_receipt = first.get("receipt")
        original_context = reason_kwargs.get("cognitive_context")
        actual_context = (
            first_receipt.get("cognitive_slots")
            if isinstance(first_receipt, dict)
            and isinstance(first_receipt.get("cognitive_slots"), list)
            and first_receipt["cognitive_slots"]
            else original_context
        )
        try:
            from core.brain.llm.latent_cortex.cognitive_acquisition import (
                acquisition_has_new_context,
                build_acquisition_receipt,
                build_acquisition_request,
                build_continuation_receipt,
                validate_acquisition_receipt,
                validate_continuation_receipt,
            )

            request = build_acquisition_request(
                objective=objective,
                first_text=str(first.get("text") or ""),
                first_receipt=first_receipt,
                cognitive_context=actual_context,
            )
        except (ImportError, TypeError, ValueError) as exc:
            record_degradation(
                "latent_cortex.cognitive_acquisition",
                exc,
                action="retained the first proven answer after acquisition request validation failed",
                severity="warning",
            )
            if foreground is True:
                await self._broadcast_conclusion(
                    first,
                    objective=objective,
                    stakes=float(reason_kwargs.get("stakes", 0.5)),
                )
            return first
        if request is None:
            if foreground is True:
                await self._broadcast_conclusion(
                    first,
                    objective=objective,
                    stakes=float(reason_kwargs.get("stakes", 0.5)),
                )
            return first

        adaptive_acquisition: dict[str, Any] | None = None
        if isinstance(first_receipt, dict):
            execution = first_receipt.get("adaptive_compute")
            plan = execution.get("plan") if isinstance(execution, dict) else None
            if isinstance(plan, dict):
                try:
                    from core.brain.llm.latent_cortex.adaptive_compute import (
                        build_adaptive_acquisition_receipt,
                        validate_adaptive_acquisition_receipt,
                    )

                    tools = plan.get("routing", {}).get("tools", {})
                    authorized = tools.get("max_acquisitions") == 1
                    adaptive_acquisition = build_adaptive_acquisition_receipt(
                        plan=plan,
                        request_sha256=str(request["request_sha256"]),
                        attempted=authorized,
                    )
                    validate_adaptive_acquisition_receipt(adaptive_acquisition)
                    first_receipt["adaptive_acquisition"] = adaptive_acquisition
                    if not authorized:
                        self._last_receipt = first_receipt
                        if foreground is True:
                            await self._broadcast_conclusion(
                                first,
                                objective=objective,
                                stakes=float(reason_kwargs.get("stakes", 0.5)),
                            )
                        return first
                except (ImportError, AttributeError, TypeError, ValueError):
                    return self._record_failure(
                        "adaptive_acquisition_authority_invalid"
                    )

        acquisition_started = time.monotonic()
        continuation_inputs: dict[str, Any] = {}
        try:
            from core.brain.llm.latent_cortex.cognitive_acquisition import (
                COGNITIVE_COMPUTE_ACTIONS,
            )
            from core.brain.llm.latent_cortex.epistemic_state import OperationKind

            action = OperationKind(request["action"])
            if action in COGNITIVE_COMPUTE_ACTIONS:
                from core.brain.capability_evidence_context import (
                    CapabilityEvidenceBundle,
                    merge_capability_evidence,
                )
                from core.brain.cortex_compute_acquisition import (
                    acquire_cognitive_compute,
                )

                compute = await acquire_cognitive_compute(
                    objective=objective,
                    first_text=str(first.get("text") or ""),
                    action=action,
                    timeout_s=max(
                        0.1,
                        min(
                            12.0,
                            float(reason_kwargs.get("timeout_s", 300.0))
                            - (time.monotonic() - started)
                            - 15.0,
                        ),
                    ),
                )
                acquired_context, merge_receipt = merge_capability_evidence(
                    original_context if isinstance(original_context, list) else None,
                    CapabilityEvidenceBundle(compute.context, compute.receipt),
                )
                ingress_receipt = {
                    "schema": "aura.rlc.compute_ingress.v1",
                    "compute": compute.receipt,
                    "context_merge": merge_receipt,
                    "absent_sources": (
                        [] if compute.context else ["symbolic_compute"]
                    ),
                }
            else:
                from core.brain.cognitive_ingress import (
                    assemble_cognitive_ingress_async,
                    cognitive_context_items,
                )

                ingress = await assemble_cognitive_ingress_async(
                    self.orchestrator if orchestrator is None else orchestrator,
                    objective,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=session_id,
                    retrieval_query=str(request["retrieval_query"]),
                    acquisition_source=(
                        "memory"
                        if request["action"] == "search_memory"
                        else "reference"
                    ),
                )
                acquired_context = cognitive_context_items(ingress) or None
                ingress_receipt = ingress.to_receipt()
                continuation_inputs = {
                    "stakes": max(
                        float(reason_kwargs.get("stakes", 0.5)),
                        float(ingress.stakes),
                    ),
                    "uncertainty": float(ingress.uncertainty),
                    "epistemic_genesis": ingress.epistemic_genesis,
                    "epistemic_state": ingress.epistemic_state,
                    "selective_memory_result": ingress.memory_result,
                }
            if action is OperationKind.RETRIEVE_EVIDENCE:
                from core.brain.cortex_web_acquisition import (
                    acquire_live_web_evidence,
                    should_acquire_live_web,
                )

                local_context_is_new = acquisition_has_new_context(
                    request,
                    acquired_context,
                )
                use_web, web_reason = should_acquire_live_web(
                    objective,
                    str(request["retrieval_query"]),
                    local_context_is_new=local_context_is_new,
                )
                ingress_receipt["live_web_selection"] = {
                    "selected": use_web,
                    "reason": web_reason,
                    "local_context_is_new": local_context_is_new,
                }
                if use_web:
                    remaining_acquisition_s = max(
                        1.0,
                        min(
                            20.0,
                            float(reason_kwargs.get("timeout_s", 300.0))
                            - (time.monotonic() - started)
                            - 15.0,
                        ),
                    )
                    web = await acquire_live_web_evidence(
                        self.orchestrator if orchestrator is None else orchestrator,
                        objective=objective,
                        retrieval_query=str(request["retrieval_query"]),
                        cognitive_context=acquired_context,
                        selection_reason=web_reason,
                        timeout_s=remaining_acquisition_s,
                    )
                    acquired_context = web.context
                    ingress_receipt["live_web_acquisition"] = web.receipt
            acquisition = build_acquisition_receipt(
                request,
                acquired_context=acquired_context,
                ingress_receipt=ingress_receipt,
                elapsed_s=time.monotonic() - acquisition_started,
            )
            validate_acquisition_receipt(acquisition, request=request)
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            record_degradation(
                "latent_cortex.cognitive_acquisition",
                exc,
                action="retained the first proven answer after the bounded acquisition failed",
                severity="warning",
            )
            acquisition = build_acquisition_receipt(
                request,
                acquired_context=None,
                ingress_receipt={"status": "failed", "error_type": type(exc).__name__},
                elapsed_s=min(30.0, time.monotonic() - acquisition_started),
                error_code=f"acquisition_{type(exc).__name__.lower()}",
            )
            continuation = build_continuation_receipt(
                request,
                acquisition,
                first_result=first,
                second_result=None,
                returned_round=1,
                continuation_reason="acquisition_failed",
            )
            validate_continuation_receipt(continuation)
            first_receipt["cognitive_acquisition"] = continuation
            self._last_receipt = first_receipt
            if foreground is True:
                await self._broadcast_conclusion(
                    first,
                    objective=objective,
                    stakes=float(reason_kwargs.get("stakes", 0.5)),
                )
            return first

        if acquisition["status"] != "completed_new_context":
            continuation = build_continuation_receipt(
                request,
                acquisition,
                first_result=first,
                second_result=None,
                returned_round=1,
                continuation_reason="no_new_context",
            )
            validate_continuation_receipt(continuation)
            first_receipt["cognitive_acquisition"] = continuation
            self._last_receipt = first_receipt
            if foreground is True:
                await self._broadcast_conclusion(
                    first,
                    objective=objective,
                    stakes=float(reason_kwargs.get("stakes", 0.5)),
                )
            return first

        timeout_s = float(reason_kwargs.get("timeout_s", 300.0))
        remaining_s = timeout_s - (time.monotonic() - started)
        if remaining_s < 15.0:
            continuation = build_continuation_receipt(
                request,
                acquisition,
                first_result=first,
                second_result=None,
                returned_round=1,
                continuation_reason="budget_insufficient",
            )
            validate_continuation_receipt(continuation)
            first_receipt["cognitive_acquisition"] = continuation
            self._last_receipt = first_receipt
            if foreground is True:
                await self._broadcast_conclusion(
                    first,
                    objective=objective,
                    stakes=float(reason_kwargs.get("stakes", 0.5)),
                )
            return first

        second_kwargs = dict(reason_kwargs)
        second_kwargs.update(
            {
                "timeout_s": remaining_s,
                "cognitive_context": acquired_context,
                **continuation_inputs,
            }
        )
        second = await self.deep_reason(
            question,
            messages=messages,
            publish_workspace_conclusion=bool(foreground),
            **second_kwargs,
        )
        returned_round = 2 if second.get("ok") is True else 1
        returned = second if returned_round == 2 else first
        continuation = build_continuation_receipt(
            request,
            acquisition,
            first_result=first,
            second_result=second,
            returned_round=returned_round,
            continuation_reason=(
                "second_episode_succeeded"
                if returned_round == 2
                else "second_episode_failed"
            ),
        )
        validate_continuation_receipt(continuation)
        returned_receipt = returned.get("receipt")
        if isinstance(returned_receipt, dict):
            returned_receipt["cognitive_acquisition"] = continuation
            if adaptive_acquisition is not None:
                returned_receipt["adaptive_acquisition"] = adaptive_acquisition
            self._last_receipt = returned_receipt
        if returned_round == 1 and foreground is True:
            await self._broadcast_conclusion(
                first,
                objective=objective,
                stakes=float(reason_kwargs.get("stakes", 0.5)),
            )
        return returned

    async def run_action_state_episode(
        self,
        *,
        prompt: str | None = None,
        messages: list | None = None,
        domain: str,
        config: dict[str, Any],
        budget: dict[str, Any],
        cognitive_context: list | None,
        action_policy_evidence: dict[str, Any],
        action_state_runtime: dict[str, Any],
        action_intervention: dict[str, Any] | None = None,
        external_execution_offer: dict[str, Any] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        """Run the claim-grade first-action lane without product-policy drift.

        The campaign runner supplies the complete signed public experiment
        contract.  The service deliberately does not synthesize body, Will,
        memory, or controller inputs on this lane because doing so after the
        runner froze its request would change the paired prestate.
        """

        if not isinstance(action_state_runtime, dict):
            return self._record_failure("invalid_action_state_runtime")
        if action_intervention is not None and not isinstance(
            action_intervention, dict
        ):
            return self._record_failure("invalid_action_intervention")
        try:
            from core.brain.llm.mlx_client import get_mlx_client

            client = get_mlx_client()
            result = await client.latent_reason_async(
                prompt=prompt,
                messages=messages,
                config=config,
                budget=budget,
                domain=domain,
                timeout_s=timeout_s,
                foreground_request=False,
                cognitive_context=cognitive_context,
                action_policy_evidence=action_policy_evidence,
                action_intervention=action_intervention,
                action_state_runtime=action_state_runtime,
                external_execution_offer=external_execution_offer,
                verifier_guidance=True,
            )
        except asyncio.CancelledError:
            raise
        except (
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            TimeoutError,
        ) as exc:
            record_degradation(
                "latent_cortex.action_state_lane",
                exc,
                action="contained claim-grade action-state lane failure",
                severity="error",
            )
            return self._record_failure(
                f"action_state_runtime_failed:{type(exc).__name__}"
            )
        if result.get("ok") is not True:
            self._last_failure_receipt = dict(result.get("receipt") or {})
            return self._record_failure(
                str(result.get("reason") or "action_state_runtime_failed")
            )
        self._last_receipt = dict(result.get("receipt") or {})
        self._last_progress = dict(result.get("progress") or {})
        return result

    # ── The episode ─────────────────────────────────────────────────────
    async def qualified_recurrent_reason(
        self,
        objective: str,
        *,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Use retained recurrent tissue only inside its certified grammar."""

        if not isinstance(objective, str) or not objective.strip():
            return {
                "eligible": False,
                "attempted": False,
                "ok": False,
                "reason": "qualified_recurrent_objective_invalid",
            }
        try:
            bounded_timeout = float(timeout_s)
            if not math.isfinite(bounded_timeout) or bounded_timeout <= 0.0:
                raise ValueError("qualified recurrent timeout is invalid")
            from core.brain.llm.qualified_recurrent_ingress import (
                admit_qualified_recurrent_objective,
                execute_qualified_recurrent_objective,
            )

            normalized = objective.strip()
            admission = admit_qualified_recurrent_objective(normalized)
            if admission is None:
                return {
                    "eligible": False,
                    "attempted": False,
                    "ok": False,
                    "reason": "qualified_recurrent_objective_unsupported",
                }

            # CP568 semantic tissue owns its model identity in the signed
            # activation and executes without model generation.  Looking up
            # the resident client here coupled that path to cold model startup
            # and model-lane contention even though it never uses the client.
            # Legacy typed families still require the worker and retain the
            # existing client-bound authority checks.
            client = None
            if not str(admission.family).startswith("frontier_"):
                from core.brain.llm.mlx_client import get_mlx_client

                client = get_mlx_client()

            return await execute_qualified_recurrent_objective(
                client,
                normalized,
                timeout_s=min(300.0, max(5.0, bounded_timeout)),
            )
        except asyncio.CancelledError:
            raise
        except (
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            TimeoutError,
        ) as exc:
            record_degradation(
                "latent_cortex.qualified_recurrent_ingress",
                exc,
                action=(
                    "refused activated typed recurrent serving whose exact ingress "
                    "contract failed"
                ),
                severity="error",
            )
            return {
                "eligible": True,
                "attempted": True,
                "ok": False,
                "reason": f"qualified_recurrent_ingress_failed:{type(exc).__name__}",
            }

    async def deep_reason(
        self,
        question: str | None = None,
        *,
        messages: list | None = None,
        stakes: float = 0.5,
        uncertainty: float = 0.5,
        domain: str = "general",
        config_overrides: dict[str, Any] | None = None,
        runtime_controls: dict[str, Any] | None = None,
        timeout_s: float = 300.0,
        require_full_stack: bool = True,
        foreground_request: bool = True,
        cognitive_context: list | None = None,
        epistemic_genesis: Any | None = None,
        epistemic_state: Any | None = None,
        selective_memory_result: Any | None = None,
        publish_workspace_conclusion: bool = True,
        external_execution_offer: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one latent-reasoning episode on the resident model."""
        if not _cortex_enabled():
            return self._record_failure("disabled:AURA_LATENT_CORTEX=0")
        if question is not None and not isinstance(question, str):
            return self._record_failure("invalid_question")
        if messages is not None and not isinstance(messages, list):
            return self._record_failure("invalid_messages")
        if not (isinstance(question, str) and question.strip()) and not messages:
            return self._record_failure("empty_question")
        # CP126 1a992727: type checks are not a schema. Nothing bounded the
        # text or the list, validated message roles and content shapes,
        # rejected unknown message fields, or constrained the domain — so an
        # oversized or malformed request went straight into IPC and into the
        # downstream calibration stores that learn from it. A store poisoned
        # by one malformed request keeps mis-scoring long after the request
        # is gone.
        schema_error = self._request_schema_error(question, messages, domain)
        if schema_error:
            return self._record_failure(schema_error, stage="request_schema")
        if config_overrides is not None and not isinstance(config_overrides, dict):
            return self._record_failure("invalid_config_overrides")
        if runtime_controls is not None and not isinstance(runtime_controls, dict):
            return self._record_failure("invalid_runtime_controls")
        if runtime_controls is not None:
            expected_control_keys = {
                "clean_user_surface_recurrent_loops",
                "clean_user_surface_steering_alpha",
            }
            recurrent_loops = runtime_controls.get("clean_user_surface_recurrent_loops")
            steering_alpha = runtime_controls.get("clean_user_surface_steering_alpha")
            if (
                set(runtime_controls) != expected_control_keys
                or type(recurrent_loops) is not int
                or not 1 <= recurrent_loops <= 2
                or isinstance(steering_alpha, bool)
                or not isinstance(steering_alpha, (int, float))
                or not math.isfinite(float(steering_alpha))
                or not 0.0 <= float(steering_alpha) <= 1.0
            ):
                return self._record_failure("invalid_runtime_controls")
        if type(require_full_stack) is not bool:
            return self._record_failure("invalid_require_full_stack")
        if type(foreground_request) is not bool:
            return self._record_failure("invalid_foreground_request")
        if type(publish_workspace_conclusion) is not bool:
            return self._record_failure("invalid_publish_workspace_conclusion")
        try:
            from core.brain.llm.latent_cortex.cognitive_context import (
                normalize_cognitive_context,
            )

            cognitive_context = normalize_cognitive_context(cognitive_context) or None
        except (TypeError, ValueError):
            return self._record_failure("invalid_cognitive_context")
        try:
            from core.brain.llm.latent_cortex.external_execution import (
                validate_external_execution_offer,
            )

            external_execution_offer = (
                validate_external_execution_offer(external_execution_offer)
                if external_execution_offer is not None
                else None
            )
        except (ImportError, TypeError, ValueError):
            return self._record_failure("invalid_external_execution_offer")
        memory_context_present = any(
            isinstance(entry, dict) and entry.get("context_role") == "memory_observation"
            for entry in (cognitive_context or [])
        )
        if (
            memory_context_present
            or epistemic_genesis is not None
            or epistemic_state is not None
            or selective_memory_result is not None
        ):
            try:
                from core.brain.llm.latent_cortex.epistemic_memory import (
                    SelectiveMemoryResult,
                    validate_memory_context_items,
                )
                from core.brain.llm.latent_cortex.epistemic_state import EpistemicState

                if (
                    not isinstance(epistemic_genesis, EpistemicState)
                    or epistemic_genesis.version != 0
                    or not isinstance(epistemic_state, EpistemicState)
                    or not isinstance(selective_memory_result, SelectiveMemoryResult)
                    or epistemic_state.episode_id != epistemic_genesis.episode_id
                    or epistemic_state.problem != epistemic_genesis.problem
                    or epistemic_state.parent_sha256 != epistemic_genesis.state_sha256
                ):
                    return self._record_failure("invalid_epistemic_memory_authority")
                visible_objective = self._visible_objective(question, messages)
                if (
                    epistemic_state.problem.objective_sha256
                    != hashlib.sha256(visible_objective.encode("utf-8")).hexdigest()
                ):
                    return self._record_failure("epistemic_objective_mismatch")
                validate_memory_context_items(
                    epistemic_state,
                    selective_memory_result,
                    cognitive_context or [],
                )
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                logger.warning("Epistemic memory authority rejected: %s", exc)
                return self._record_failure("invalid_epistemic_memory_authority")
        try:
            timeout_s = float(timeout_s)
        except (TypeError, ValueError, OverflowError):
            return self._record_failure("invalid_timeout")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            return self._record_failure("invalid_timeout")
        try:
            stakes = _unit_signal(stakes, name="stakes")
            uncertainty = _unit_signal(uncertainty, name="uncertainty")
        except ValueError:
            return self._record_failure("invalid_cognitive_economy")
        foreground_surplus_plan: dict[str, Any] | None = None
        if foreground_request and require_full_stack and runtime_controls is not None:
            requested_decode_tokens = (
                config_overrides.get("decode_max_tokens")
                if isinstance(config_overrides, dict)
                else None
            )
            decode_capacity = (
                requested_decode_tokens
                if type(requested_decode_tokens) is int and requested_decode_tokens > 0
                else 768
            )
            foreground_surplus_plan = _foreground_surplus_plan(
                messages=messages,
                visible_objective=self._visible_objective(question, messages),
                decode_max_tokens=decode_capacity,
                request_timeout_s=timeout_s,
                last_latency_s=self._last_latency_s,
            )
            if foreground_surplus_plan["admitted"] is not True:
                self._last_allocation = {
                    "foreground_surplus_admission": dict(foreground_surplus_plan)
                }
                return self._record_failure(
                    "latent_surplus_budget_insufficient",
                    stage="surplus_admission",
                    evidence=foreground_surplus_plan,
                )
            timeout_s = float(foreground_surplus_plan["latent_surplus_s"])
        try:
            from core.brain.llm.mlx_client import get_mlx_client
            from core.runtime.errors import DependencyUnavailable, ModelUnavailable

            client = get_mlx_client()
        except (
            ImportError,
            DependencyUnavailable,
            ModelUnavailable,
            OSError,
            TimeoutError,
        ) as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action="refused latent episode: resident model client unavailable",
            )
            return self._record_failure(f"client_unavailable:{type(exc).__name__}")
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action=(
                    "refused latent episode whose resident model client "
                    "failed an integrity check"
                ),
                severity="degraded",
            )
            return self._record_failure(
                f"client_integrity_failure:{type(exc).__name__}"
            )
        if client is None:
            return self._record_failure("no_resident_model")
        worker_identity: dict[str, Any] = {}
        identity_getter = getattr(client, "get_worker_identity_snapshot", None)
        if callable(identity_getter):
            try:
                candidate_identity = identity_getter()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                candidate_identity = {}
            if isinstance(candidate_identity, dict):
                worker_identity = dict(candidate_identity)
        try:
            model_parameter_count = int(worker_identity.get("worker_model_parameter_count") or 0)
        except (TypeError, ValueError, OverflowError):
            model_parameter_count = 0
        try:
            visible_objective = self._visible_objective(question, messages)
            private_reasoning_tokens = 0
            model_path = str(worker_identity.get("worker_model_path") or "")
            if foreground_request and model_path and _controller_accepts_overrides(config_overrides):
                from core.brain.llm.chat_format import thinking_enabled_for_generation
                from core.brain.llm.thinking_reserve import reserve_tokens

                if thinking_enabled_for_generation(
                    model_path, answer_is_derived_here=True,
                ) is True:
                    private_reasoning_tokens = max(0, int(reserve_tokens(model_path)))
            requested_decode_tokens = None
            if (
                config_overrides is not None
                and _controller_accepts_overrides(config_overrides)
                and "decode_max_tokens" in config_overrides
            ):
                requested_decode_tokens = config_overrides["decode_max_tokens"]
                if (
                    type(requested_decode_tokens) is not int
                    or requested_decode_tokens <= 0
                ):
                    return self._record_failure("invalid_decode_token_override")
            config, budget = self.allocate(
                stakes=stakes,
                uncertainty=uncertainty,
                objective=visible_objective,
                model_parameter_count=max(0, model_parameter_count),
                foreground_request=foreground_request,
                timeout_s=timeout_s,
                requested_decode_tokens=requested_decode_tokens,
                private_reasoning_tokens=private_reasoning_tokens,
            )
        except (TypeError, ValueError, OverflowError):
            return self._record_failure("invalid_cognitive_economy")
        if foreground_surplus_plan is not None:
            self._last_allocation["foreground_surplus_admission"] = dict(
                foreground_surplus_plan
            )
        allocation_profile = str(self._last_allocation.get("allocation_profile") or "")
        adaptive_plan: dict[str, Any] | None = dict(
            self._last_allocation.get("adaptive_compute") or {}
        )
        if config_overrides is not None:
            allocated_decode_capacity = config["decode_max_tokens"]
            config.update(dict(config_overrides))
            if private_reasoning_tokens:
                config["decode_max_tokens"] = max(
                    config["decode_max_tokens"], allocated_decode_capacity,
                )
        if not _controller_accepts_overrides(config_overrides):
            adaptive_plan = None
            self._last_allocation["adaptive_compute_execution"] = (
                "explicit_structural_override"
            )
        # Learned execution controller: evidence-gated arm selection over
        # the base allocation (deeper recurrence / wider branches /
        # probe-guided bytecode / lean ΔW). Exploits only after Wilson
        # separation on graded outcomes in this context bucket; explores
        # sparsely; never touches explicit operator overrides.
        controller_decision: dict[str, Any] | None = None
        action_policy_evidence: dict[str, Any] | None = None
        if foreground_request and _controller_accepts_overrides(config_overrides):
            try:
                from core.brain.llm.latent_cortex.execution_controller import (
                    controller_enabled,
                    get_execution_controller,
                )

                if not controller_enabled():
                    raise RuntimeError("execution controller disabled")
                controller = get_execution_controller()
                controller_decision = controller.choose(
                    objective=self._visible_objective(question, messages),
                    domain=domain,
                    stakes=stakes,
                    uncertainty=uncertainty,
                )
                snapshot_builder = getattr(
                    controller,
                    "action_evidence_snapshot",
                    None,
                )
                if callable(snapshot_builder):
                    action_policy_evidence = snapshot_builder(
                        bucket=str(controller_decision["bucket"])
                    )
                else:
                    from core.brain.llm.latent_cortex.value_of_computation import (
                        build_evidence_snapshot,
                    )

                    action_policy_evidence = build_evidence_snapshot(
                        bucket=str(controller_decision["bucket"]),
                        cells={},
                    )
                if controller_decision["arm"] != "base":
                    # CP126 ea828a97: apply_arm could silently leave the config
                    # unchanged (e.g. probe-guided bytecode with no recurrent
                    # region) while the decision still named the treatment, so
                    # the outcome was credited to bytecode that never ran. Take
                    # the application RECEIPT and fall the decision back to
                    # base when the arm did not actually apply.
                    application = controller.apply_arm_receipt(
                        controller_decision["arm"],
                        config,
                        recurrent_region=(
                            (16, 48)
                            if allocation_profile == "resident_32b_interactive_full_stack_v2"
                            else None
                        ),
                    )
                    config = application["config"]
                    controller_decision["applied"] = bool(application["applied"])
                    controller_decision["application_reason"] = str(application.get("reason") or "")
                    if not application["applied"]:
                        controller_decision["arm"] = str(application.get("effective_arm") or "base")
                self._last_allocation["execution_controller"] = controller_decision
            except (
                ImportError,
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
                OSError,
            ) as exc:
                logger.debug("Execution controller unavailable: %s", exc)
                controller_decision = None
                action_policy_evidence = None
        if adaptive_plan is not None:
            try:
                from core.brain.llm.latent_cortex.adaptive_compute import (
                    enforce_adaptive_compute_limits,
                )

                config = enforce_adaptive_compute_limits(config, adaptive_plan)
                self._last_allocation["adaptive_compute_execution"] = "enforced"
            except (ImportError, TypeError, ValueError, OverflowError):
                return self._record_failure("adaptive_compute_enforcement_failed")
        if external_execution_offer is not None and (
            controller_decision is None or action_policy_evidence is None
        ):
            return self._record_failure(
                "external_execution_controller_unavailable"
            )
        try:
            from core.brain.llm.latent_cortex.branches import BRANCH_ROLES
            from core.brain.llm.latent_cortex.correlated_support import (
                get_branch_correlation_ledger,
            )
            from core.brain.llm.latent_cortex.execution_controller import context_bucket

            branch_count = int(config.get("n_branches") or 1)
            correlation_roles = list(BRANCH_ROLES[:branch_count])
            correlation_bucket = (
                str(controller_decision.get("bucket") or "")
                if controller_decision is not None
                else context_bucket(
                    self._visible_objective(question, messages),
                    domain,
                    stakes,
                    uncertainty,
                )
            )
            correlation_ledger = get_branch_correlation_ledger()
            config["branch_correlation_evidence"] = correlation_ledger.evidence(
                bucket=correlation_bucket,
                roles=correlation_roles,
            )
            self._last_allocation["correlated_support"] = {
                **correlation_ledger.status(),
                "bucket": correlation_bucket,
                "roles": correlation_roles,
                "evidence_state": config["branch_correlation_evidence"]["evidence_state"],
            }
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.debug("Branch correlation evidence unavailable: %s", exc)
            config["branch_correlation_evidence"] = None
        try:
            from core.brain.llm.latent_cortex.execution_controller import context_bucket
            from core.brain.llm.latent_cortex.verifier_fusion import (
                get_verifier_fusion_ledger,
            )

            verifier_bucket = (
                str(controller_decision.get("bucket") or "")
                if controller_decision is not None
                else context_bucket(
                    self._visible_objective(question, messages),
                    domain,
                    stakes,
                    uncertainty,
                )
            )
            verifier_ledger = get_verifier_fusion_ledger()
            config["verifier_fusion_evidence"] = verifier_ledger.evidence(
                bucket=verifier_bucket
            )
            self._last_allocation["verifier_fusion"] = {
                **verifier_ledger.status(),
                "bucket": verifier_bucket,
                "evidence_state": config["verifier_fusion_evidence"][
                    "evidence_state"
                ],
            }
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.debug("Verifier fusion evidence unavailable: %s", exc)
            config["verifier_fusion_evidence"] = None
        if allocation_profile == "resident_32b_interactive_full_stack_v2":
            try:
                from core.brain.llm.latent_cortex.critic_identity import (
                    build_critic_source_identity,
                    build_generator_function_identity,
                    get_critic_blind_spot_ledger,
                )
                from core.brain.llm.latent_cortex.execution_controller import (
                    context_bucket,
                )

                generator_identity = build_generator_function_identity(worker_identity)
                critic_source = build_critic_source_identity()
                critic_ledger = get_critic_blind_spot_ledger()
                critic_bucket = (
                    str(controller_decision.get("bucket") or "")
                    if controller_decision is not None
                    else context_bucket(
                        self._visible_objective(question, messages),
                        domain,
                        stakes,
                        uncertainty,
                    )
                )
                config["critic_blind_spot_evidence"] = critic_ledger.evidence(
                    bucket=critic_bucket,
                    generator_function_sha256=generator_identity["function_sha256"],
                    critic_function_sha256=critic_source["source_closure_sha256"],
                )
                self._last_allocation["critic_blind_spots"] = {
                    **critic_ledger.status(),
                    "bucket": critic_bucket,
                    "evidence_state": config["critic_blind_spot_evidence"]["evidence_state"],
                    "critic_reliability_admitted": config["critic_blind_spot_evidence"][
                        "critic_reliability_admitted"
                    ],
                }
            except (
                ImportError,
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.error("Critic identity/evidence unavailable: %s", exc)
                return self._record_failure(
                    f"critic_identity_evidence_unavailable:{type(exc).__name__}"
                )
        if allocation_profile == "resident_32b_interactive_full_stack_v2":
            try:
                requested_decode_tokens = int(config.get("decode_max_tokens") or 256)
            except (TypeError, ValueError, OverflowError):
                return self._record_failure("invalid_decode_token_override")
            # Compound-aware answer surface: the SAME facet definition the
            # product-quality gate judges by decides how much room and how
            # much sampling discipline the answer gets. CP103's live turn
            # proved a 4-facet request cannot earn the gate inside 256 tokens
            # at persona-lane temperature — the episode was mechanically
            # complete and the ANSWER SURFACE was the only failing stage.
            from core.brain.llm.latent_cortex.output_quality import request_facets
            from core.runtime.structured_input import (
                analyze_prompt_shape,
                answer_surface_planning_tokens,
                answer_surface_token_floor,
            )

            visible_objective = self._visible_objective(question, messages)
            objective_facets = request_facets(visible_objective)
            objective_shape = analyze_prompt_shape(visible_objective).to_dict()
            compound_objective = bool(
                len(objective_facets) >= 2
                or objective_shape["requires_single_reply_coverage"]
                or objective_shape["numbered_parts"] >= 2
                or objective_shape["imperative_parts"] >= 3
                or objective_shape["question_parts"] >= 3
            )
            self._last_allocation["objective_facets"] = list(objective_facets)
            self._last_allocation["objective_prompt_shape"] = objective_shape
            self._last_allocation["compound_objective"] = compound_objective
            admitted_private_tokens = self._last_allocation[
                "native_reasoning_allowance"
            ]["admitted_private_tokens"]
            if compound_objective:
                structural_answer_floor = answer_surface_token_floor(
                    visible_objective
                )
                config["decode_max_tokens"] = max(
                    min(8192, structural_answer_floor + admitted_private_tokens),
                    requested_decode_tokens,
                    320,
                )
                config["decode_bridge_policy"] = "assistant_answer_v3"
                # EOS floor: a compound answer abandoned 16 tokens in is
                # sampling variance, not a decision (CP116 live evidence).
                config["decode_min_tokens"] = 96
                # Coverage determinism: compound answers must satisfy every
                # requested facet inside a bounded budget; persona-lane
                # temperature (0.58) makes that a coin flip (CP113-117 each
                # failed on a different sampling tail). CP105's lesson was
                # that a temperature clamp WITHOUT a degeneration guard
                # loops; the repetition penalty, EOS floor, and newline
                # discipline now make low-temperature decoding safe.
                try:
                    requested_temperature = float(config.get("decode_temperature") or 0.0)
                except (TypeError, ValueError, OverflowError):
                    return self._record_failure("invalid_decode_temperature_override")
                config["decode_temperature"] = min(0.3, max(0.0, requested_temperature))
            else:
                config["decode_max_tokens"] = max(
                    64,
                    requested_decode_tokens,
                )
                config["decode_bridge_policy"] = "assistant_answer_v1"
                config["decode_min_tokens"] = 48
            # Degeneration guard for every resident answer: CP105's live turn
            # proved a repetition loop survives temperature tuning (one line
            # ~80 times at t=0.35, trigram diversity 0.012). The persona
            # lane's temperature is kept; the CTRL-style penalty is what
            # actually prevents loops.
            config["decode_repetition_penalty"] = 1.25
            config["decode_repetition_window"] = 72
            if compound_objective:
                # Reserve the answer floor before optional recurrent work.
                #
                # This read `65.0 + (0.26 * tokens)` — a conservative profile
                # written down once. Against the 112s ceiling its own caller
                # allows (deep_deliberation caps at min(120, timeout*2)), the
                # smallest compound surface of 768 tokens asks for 264.7s, so
                # EVERY compound objective was refused before execution and
                # every measurement of foreground deep reasoning was a
                # measurement of something that never ran. The harder the
                # question, the more certainly it was refused.
                #
                # The machine already measures this. `measured_admission`
                # keeps p90 prefill, decode and overhead per task shape from
                # completed generations, and chat.py already sizes the turn
                # deadline from it; only this decision was still using the
                # written-down number. An unmeasured shape falls back to the
                # module's own static prior, so the conservative behaviour
                # survives exactly as long as there is nothing better.
                capacity_decode_tokens = int(config["decode_max_tokens"])
                target_decode_tokens = answer_surface_planning_tokens(
                    visible_objective
                ) + admitted_private_tokens
                try:
                    from core.brain.llm.measured_admission import (
                        recommended_completion_tokens,
                        recommended_foreground_deadline,
                    )
                    from core.brain.llm.model_registry import runtime_model_measurement_key

                    prompt_tokens = _input_prompt_tokens(messages, question)
                    target_decode_tokens, length_confidence, length_samples = (
                        recommended_completion_tokens(
                            model=runtime_model_measurement_key(),
                            prompt_tokens=prompt_tokens,
                            maximum_tokens=capacity_decode_tokens,
                            prior_tokens=target_decode_tokens,
                        )
                    )
                    required_wall_clock_s = 65.0 + (0.26 * target_decode_tokens)
                    measured_s, _confidence, samples = recommended_foreground_deadline(
                        model=runtime_model_measurement_key(),
                        prompt_tokens=prompt_tokens,
                        decode_tokens=max(1, target_decode_tokens),
                        minimum_seconds=0.0,
                        maximum_seconds=float("inf"),
                    )
                    if samples > 0 and measured_s > 0.0:
                        required_wall_clock_s = float(measured_s)
                    else:
                        from core.brain.llm.generation_allowance import resident_generation_seconds

                        live_seconds = resident_generation_seconds(
                            messages or [{"role": "user", "content": question}],
                            target_decode_tokens,
                            private_tokens_included=True,
                        )
                        if live_seconds > 0.0:
                            required_wall_clock_s = live_seconds
                    self._last_allocation[
                        "answer_surface_wall_clock_samples"
                    ] = int(samples)
                    self._last_allocation.update(
                        {
                            "answer_surface_prompt_tokens": prompt_tokens,
                            "answer_surface_capacity_tokens": capacity_decode_tokens,
                            "answer_surface_planning_tokens": target_decode_tokens,
                            "answer_surface_length_confidence": length_confidence.value,
                            "answer_surface_length_samples": int(length_samples),
                        }
                    )
                except (ArithmeticError, ImportError, TypeError, ValueError) as exc:
                    required_wall_clock_s = 65.0 + (0.26 * target_decode_tokens)
                    record_degradation(
                        "latent_cortex.answer_surface_admission",
                        exc,
                        severity="debug",
                        action="priced the answer surface from the static profile",
                        enforce_failure_policy=False,
                    )
                available_wall_clock_s = max(15.0, float(timeout_s) - 8.0)
                budget["wall_clock_s"] = min(
                    max(required_wall_clock_s, float(budget.get("wall_clock_s") or 0.0)),
                    available_wall_clock_s,
                )
                self._last_allocation["answer_surface_required_wall_clock_s"] = round(
                    required_wall_clock_s, 3
                )
                self._last_allocation["answer_surface_available_wall_clock_s"] = round(
                    available_wall_clock_s, 3
                )
                if available_wall_clock_s + 1e-9 < required_wall_clock_s:
                    # Do not spend most of the turn proving that a complete
                    # answer cannot fit and then leave a fragment. No model
                    # owner has been acquired yet, so ResponseGeneration can
                    # use the same resident checkpoint's ordinary lane with
                    # the full answer surface immediately.
                    self._last_allocation["config"] = dict(config)
                    self._last_allocation["budget"] = dict(budget)
                    return self._record_failure(
                        "answer_surface_unaffordable_before_execution",
                        stage="answer_surface_admission",
                        evidence={
                            key: value
                            for key, value in self._last_allocation.items()
                            if key.startswith("answer_surface_")
                        },
                    )
        if require_full_stack:
            config["latent_opt"] = True
            config["latent_opt_control"] = False
            config["fast_weights"] = True
        self._last_allocation["config"] = dict(config)
        self._last_allocation["budget"] = dict(budget)
        if epistemic_state is not None:
            self._last_allocation["epistemic_state"] = {
                "schema": epistemic_state.schema,
                "episode_id": epistemic_state.episode_id,
                "version": epistemic_state.version,
                "state_sha256": epistemic_state.state_sha256,
                "memory_result_sha256": selective_memory_result.result_sha256,
            }

        try:
            from core.brain.llm_health_router import (
                acquire_external_generation_gate_lease,
                release_external_generation_gate_lease,
            )
            from core.runtime.errors import DependencyUnavailable

            generation_lease_id = await acquire_external_generation_gate_lease(
                owner=(
                    "latent_cortex_foreground:episode"
                    if foreground_request
                    else "latent_cortex_lab:episode"
                ),
                timeout_s=timeout_s + 10.0,
                wait_s=min(5.0, timeout_s),
            )
        except (
            ImportError,
            DependencyUnavailable,
            OSError,
            TimeoutError,
        ) as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action="refused latent episode whose process-wide generation lease was unavailable",
                severity="warning",
            )
            return self._record_failure(f"generation_lease_unavailable:{type(exc).__name__}")
        except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action=(
                    "refused latent episode whose generation lease path "
                    "failed an integrity check"
                ),
                severity="degraded",
            )
            return self._record_failure(
                f"generation_lease_integrity_failure:{type(exc).__name__}"
            )
        if generation_lease_id is None:
            return self._record_failure("generation_gate_busy")

        # GWT→RLC coupling: the mind's live broadcast and its strongest
        # competing coalitions seed identifiable thought slots alongside the
        # organ items — deliberation runs over what consciousness is actually
        # about, not just the prompt. Organ items keep priority; coalitions
        # only fill remaining slots. Lab/background episodes stay decoupled:
        # live workspace state would confound controlled experiments.
        if foreground_request:
            try:
                from core.brain.gwt_rlc_coupling import merge_cognitive_context

                cognitive_context = merge_cognitive_context(cognitive_context)
            except (
                ImportError,
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.debug("Workspace coalition merge skipped: %s", exc)

        operation_lease = None
        operation_authority: dict[str, Any] | None = None
        operation_cost_receipt: dict[str, Any] = {}
        action_transitions: list[dict[str, Any]] = []

        async def complete_runtime_operation(
            outcome: Any,
            *,
            worker_receipt: Any,
            failure_code: str = "",
        ) -> dict[str, Any] | None:
            nonlocal operation_cost_receipt
            if operation_lease is None:
                return None
            from core.brain.llm.latent_cortex.epistemic_runtime import (
                measured_operation_cost,
            )
            from core.brain.llm.latent_cortex.epistemic_state import (
                OperationOutcome,
            )

            terminal_outcome = outcome
            terminal_failure = failure_code
            journal_action_transitions: tuple[dict[str, Any], ...] = tuple(action_transitions)
            action_costs: tuple[float, ...] = ()
            try:
                cost, operation_cost_receipt = measured_operation_cost(
                    worker_receipt,
                    requested_budget=budget,
                    state=operation_lease.state,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                terminal_outcome = OperationOutcome.FAILED
                terminal_failure = "compute_receipt_invalid"
                cost = 0.0
                operation_cost_receipt = {
                    "basis": "invalid_worker_compute_receipt",
                    "error_type": type(exc).__name__,
                }
                journal_action_transitions = ()
            if journal_action_transitions:
                remaining_state_budget = max(
                    0.0,
                    operation_lease.state.budget.total - operation_lease.state.budget.used,
                )
                mutable_action_costs = [
                    float(row["metrics"]["cost"]) * remaining_state_budget
                    for row in journal_action_transitions
                ]
                action_cost_total = math.fsum(mutable_action_costs)
                rounding_tolerance = max(1e-10, remaining_state_budget * 1e-7)
                if action_cost_total > cost + rounding_tolerance:
                    terminal_outcome = OperationOutcome.FAILED
                    terminal_failure = "compute_receipt_invalid"
                    operation_cost_receipt["action_cost_error"] = (
                        "action_transition_cost_exceeds_worker_total"
                    )
                    journal_action_transitions = ()
                    mutable_action_costs = []
                elif action_cost_total > cost and mutable_action_costs:
                    mutable_action_costs[-1] = max(
                        0.0,
                        mutable_action_costs[-1] - (action_cost_total - cost),
                    )
                action_costs = tuple(mutable_action_costs)
                operation_cost_receipt["action_state_cost"] = round(math.fsum(action_costs), 12)
                operation_cost_receipt["action_operation_count"] = len(journal_action_transitions)
            if terminal_outcome is not OperationOutcome.SUCCEEDED and not terminal_failure:
                terminal_failure = "worker_operation_failed"
            detail = (
                f"worker outcome={terminal_outcome.value}; "
                f"cost basis={operation_cost_receipt.get('basis', 'unmeasured')}"
            )
            await asyncio.to_thread(
                operation_lease.complete,
                outcome=terminal_outcome,
                cost=cost,
                action_transitions=journal_action_transitions,
                action_costs=action_costs,
                failure_code=terminal_failure,
                detail=detail,
            )
            receipt = operation_lease.to_receipt()
            receipt["compute"] = dict(operation_cost_receipt)
            return receipt

        self._episodes += 1
        self._last_attempt_at = time.time()
        self._last_progress = {}
        self._last_failure_receipt = {}
        started = time.monotonic()
        try:
            if controller_decision is not None:
                try:
                    from core.brain.llm.latent_cortex.epistemic_memory import (
                        validate_memory_context_items,
                    )
                    from core.brain.llm.latent_cortex.epistemic_runtime import (
                        RuntimeOperationLease,
                    )
                    from core.brain.llm.latent_cortex.epistemic_state import (
                        ComputeBudgetState,
                        EpistemicState,
                        ProblemFrame,
                    )
                    from core.config import DATA_DIR

                    if epistemic_state is None:
                        objective = self._visible_objective(question, messages)
                        epistemic_genesis = EpistemicState.genesis(
                            episode_id=f"rlc-runtime-{uuid.uuid4().hex[:24]}",
                            problem=ProblemFrame.create(objective),
                            budget=ComputeBudgetState(total=1.0),
                        )
                        epistemic_state = epistemic_genesis

                    operation_lease = await asyncio.to_thread(
                        RuntimeOperationLease.begin,
                        genesis=epistemic_genesis,
                        state=epistemic_state,
                        decision=controller_decision,
                        config=config,
                        budget=budget,
                        action_policy_evidence=action_policy_evidence,
                        external_execution_offer=external_execution_offer,
                        root=Path(DATA_DIR) / "latent_cortex" / "epistemic_runtime",
                    )
                    operation_authority = dict(operation_lease.authority)
                    epistemic_state = operation_lease.state
                    rebound_context = []
                    for entry in cognitive_context or []:
                        if (
                            isinstance(entry, dict)
                            and entry.get("context_role") == "memory_observation"
                        ):
                            rebound_context.append(
                                {
                                    **entry,
                                    "epistemic_state_sha256": epistemic_state.state_sha256,
                                }
                            )
                        else:
                            rebound_context.append(entry)
                    cognitive_context = rebound_context or None
                    if selective_memory_result is not None:
                        validate_memory_context_items(
                            epistemic_state,
                            selective_memory_result,
                            cognitive_context or [],
                        )
                    self._last_allocation["runtime_operation"] = {
                        "schema": operation_authority["schema"],
                        "operation_id": operation_authority["operation_id"],
                        "operation_kind": operation_authority["operation_kind"],
                        "attempt_sha256": operation_authority["attempt_sha256"],
                        "admitted_state_sha256": operation_authority["admitted_state_sha256"],
                    }
                except (
                    ImportError,
                    AttributeError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    record_degradation(
                        "latent_cortex.operation_admission",
                        exc,
                        action=(
                            "refused recurrent compute whose operation could not "
                            "be admitted and journaled"
                        ),
                        severity="error",
                    )
                    return self._record_failure(
                        f"runtime_operation_admission_failed:{type(exc).__name__}"
                    )
            try:
                result = await client.latent_reason_async(
                    prompt=question,
                    messages=messages,
                    config=config,
                    budget=budget,
                    domain=domain,
                    runtime_controls=runtime_controls,
                    timeout_s=timeout_s,
                    foreground_request=foreground_request,
                    # Typed cognitive-slot ingress: organ content (memory,
                    # goals, world model, interoception, self-model) seeds
                    # identifiable workspace slots inside the episode.
                    cognitive_context=cognitive_context,
                    operation_authority=operation_authority,
                    action_policy_evidence=action_policy_evidence,
                    external_execution_offer=external_execution_offer,
                    # Foreground resident episodes select branches and accept
                    # latent-opt proposals by deterministic task-typed checks
                    # (arithmetic recomputation, code syntax, facet coverage,
                    # grounding) — verified correctness, not convergence.
                    verifier_guidance=(
                        allocation_profile == "resident_32b_interactive_full_stack_v2"
                    ),
                    # Held-out calibration: facets whose cue-detectors humans
                    # keep overruling (Foundry grades) are muted inside the
                    # episode's verifier. None until grading evidence exists.
                    facet_reliability=self._facet_reliability_weights(domain),
                )
            except asyncio.CancelledError:
                if operation_lease is not None:
                    from core.brain.llm.latent_cortex.epistemic_state import (
                        OperationOutcome,
                    )

                    await complete_runtime_operation(
                        OperationOutcome.CANCELLED,
                        worker_receipt={},
                        failure_code="caller_cancelled",
                    )
                raise
            except (
                OSError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
                TimeoutError,
            ) as exc:
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="contained resident latent episode failure and preserved caller fallback",
                    severity="warning",
                )
                self._last_latency_s = time.monotonic() - started
                if operation_lease is not None:
                    from core.brain.llm.latent_cortex.epistemic_state import (
                        OperationOutcome,
                    )

                    await complete_runtime_operation(
                        OperationOutcome.FAILED,
                        worker_receipt={},
                        failure_code="client_exception",
                    )
                # The reason travels up to the action lane, where it becomes a
                # refusal to act. "client_error:AttributeError" told the lane
                # nothing it could act on and told the log nothing it could be
                # found by, so a browser action was refused for a type bug that
                # took a subsystem read to locate. The raise site costs nothing
                # and is the whole difference.
                from core.runtime.errors import _raise_site

                return self._record_failure(
                    f"client_error:{type(exc).__name__}@{_raise_site(exc)}"
                )
        finally:
            release_external_generation_gate_lease(generation_lease_id)
        elapsed = time.monotonic() - started
        self._last_latency_s = elapsed
        if not isinstance(result, dict):
            if operation_lease is not None:
                from core.brain.llm.latent_cortex.epistemic_state import (
                    OperationOutcome,
                )

                await complete_runtime_operation(
                    OperationOutcome.FAILED,
                    worker_receipt={},
                    failure_code="invalid_client_response",
                )
            return self._record_failure("invalid_client_response")
        raw_receipt = result.get("receipt")
        result_receipt = dict(raw_receipt) if isinstance(raw_receipt, dict) else {}
        # This code runs only after the client RPC returned and the process-wide
        # generation lease's finally block released. The worker's integrity
        # receipt independently proves whether temporary model state was erased.
        # Callers may open the ordinary lane only when both facts are explicit;
        # an episode id or a terminal stage alone is not ownership evidence.
        result_receipt["resident_owner_released"] = True
        result_receipt["resident_state_reusable"] = _resident_state_reusable(
            result_receipt
        )
        result["receipt"] = result_receipt
        action_policy_matches = action_policy_evidence is None
        why_the_policy_did_not_match = ""
        never_selected_actions = False
        if action_policy_evidence is not None:
            try:
                from core.brain.llm.latent_cortex.epistemic_state import (
                    OperationKind,
                )
                from core.brain.llm.latent_cortex.value_of_computation import (
                    validate_action_trace,
                )

                policy_receipt = result_receipt.get("value_of_computation")
                raw_trace = result_receipt.get("cognitive_action_trace")
                policy_fields = {
                    "schema",
                    "bucket",
                    "snapshot_sha256",
                    "active",
                    "executors",
                    "actions_selected",
                    "checked_transitions",
                    "selected_actions",
                }
                # Nine conditions, and the one that fires is the one worth
                # saying. Together they were "the receipt is incomplete",
                # which is where the trail went cold on a browser action that
                # never started.
                if not isinstance(policy_receipt, dict):
                    raise ValueError(
                        "worker action policy receipt is incomplete: no receipt at all "
                        f"(got {type(policy_receipt).__name__})"
                    )
                missing = policy_fields - set(policy_receipt)
                extra = set(policy_receipt) - policy_fields
                # Never reaching action selection is not disagreeing about it.
                #
                # These four are written at the very end of a full episode. An
                # episode that stopped earlier — on its budget, or on any of
                # the ways one ends — has a receipt without them, and that was
                # read as the worker's policy CONTRADICTING the host's. It is
                # not a contradiction: nothing was claimed, so nothing can
                # disagree. Reported as a mismatch it is ineligible for bypass,
                # and a browser action that had cleared every authority gate
                # was refused before it started.
                #
                # LIVE 2026-08-31, and this is what took the demo down.
                if missing == _WRITTEN_WHEN_ACTIONS_ARE_SELECTED and not extra:
                    raise _ActionSelectionNeverRanError(
                        "the episode ended before it selected any actions"
                    )
                if missing or extra:
                    raise ValueError(
                        "worker action policy receipt is incomplete: fields differ"
                        + (f", missing {sorted(missing)}" if missing else "")
                        + (f", unexpected {sorted(extra)}" if extra else "")
                    )
                for part in ("schema", "snapshot_sha256", "bucket"):
                    if policy_receipt.get(part) != action_policy_evidence[part]:
                        raise ValueError(
                            f"worker action policy receipt is incomplete: {part} differs "
                            f"({policy_receipt.get(part)!r} against "
                            f"{action_policy_evidence[part]!r})"
                        )
                if policy_receipt.get("active") is not True:
                    raise ValueError(
                        "worker action policy receipt is incomplete: it is not active"
                    )
                if not isinstance(raw_trace, list) or not raw_trace:
                    raise ValueError(
                        "worker action policy receipt is incomplete: no action trace "
                        f"(got {type(raw_trace).__name__} of "
                        f"{len(raw_trace) if isinstance(raw_trace, list) else 0})"
                    )
                if policy_receipt.get("actions_selected") != len(raw_trace):
                    raise ValueError(
                        "worker action policy receipt is incomplete: it counts "
                        f"{policy_receipt.get('actions_selected')!r} action(s) and the "
                        f"trace has {len(raw_trace)}"
                    )
                raw_executors = policy_receipt.get("executors")
                if (
                    not isinstance(raw_executors, list)
                    or not raw_executors
                    or len(raw_executors) > len(OperationKind)
                ):
                    raise ValueError("worker action executor inventory is invalid")
                try:
                    executors = tuple(OperationKind(item) for item in raw_executors)
                except (TypeError, ValueError) as exc:
                    raise ValueError("worker action executor inventory is invalid") from exc
                execute_advertised = OperationKind.EXECUTE in executors
                if (
                    len(set(executors)) != len(executors)
                    or execute_advertised != (external_execution_offer is not None)
                ):
                    raise ValueError("worker action executor inventory is invalid")
                validated_trace = validate_action_trace(
                    raw_trace,
                    evidence_snapshot=action_policy_evidence,
                    executors=executors,
                )
                for validated_row in validated_trace["rows"]:
                    decision = validated_row["decision"]
                    transition = validated_row["transition"]
                    if (
                        transition["snapshot_sha256"] != action_policy_evidence["snapshot_sha256"]
                        or transition["bucket"] != action_policy_evidence["bucket"]
                        or decision["snapshot_sha256"] != action_policy_evidence["snapshot_sha256"]
                        or decision["bucket"] != action_policy_evidence["bucket"]
                    ):
                        raise ValueError("worker action transition authority differs")
                    action_transitions.append(transition)
                selected_actions = [row["action"] for row in action_transitions]
                checked_transitions = sum(int(row["checked"]) for row in action_transitions)
                if (
                    validated_trace["selected_actions"] != selected_actions
                    or policy_receipt.get("selected_actions") != selected_actions
                    or policy_receipt.get("checked_transitions") != checked_transitions
                ):
                    raise ValueError("worker action policy summary differs from trace")
                raw_handoff = result_receipt.get("external_execution_handoff")
                if external_execution_offer is not None:
                    from core.brain.llm.latent_cortex.external_execution import (
                        validate_external_execution_handoff,
                    )

                    validate_external_execution_handoff(
                        raw_handoff,
                        offer=external_execution_offer,
                        cognitive_action_trace=raw_trace,
                    )
                elif raw_handoff not in ({}, None):
                    raise ValueError("worker emitted unoffered external execution handoff")
                action_policy_matches = True
            except _ActionSelectionNeverRanError as exc:
                # A different thing from a mismatch, and it must not be
                # reported as one: an episode that never got that far has made
                # no claim about actions at all.
                action_transitions.clear()
                action_policy_matches = False
                never_selected_actions = True
                logger.info("no action policy to check: %s", exc)
            except (ImportError, TypeError, ValueError) as exc:
                # Nine checks above raise with nine different messages, and
                # every one of them arrived here and became a bare False. The
                # refusal downstream then said only
                # "runtime_action_policy_receipt_mismatch", which is true of
                # all nine and tells nobody which — and that refusal stops a
                # browser action before it begins. The reason is already in
                # hand; it costs nothing to keep it.
                action_transitions.clear()
                action_policy_matches = False
                why_the_policy_did_not_match = f"{type(exc).__name__}: {exc}"
                logger.info(
                    "worker action policy did not match: %s", why_the_policy_did_not_match
                )
        contract_errors: list[str] = []
        quality_receipt: dict[str, Any] | None = None
        host_incumbent: tuple[str, list[int]] | None = None
        private_answer_replacement = result.pop(
            "answer_replacement_private",
            None,
        )
        visible_objective = self._visible_objective(question, messages)
        if result.get("ok") is True:
            # CP126 f22c4ed8: the facade cannot recompute this digest
            # correctly. mlx_client hashes WIRE-NORMALIZED config, budget and
            # runtime controls plus operation authority, action policy
            # evidence, intervention, response contract, verifier guidance and
            # facet reliability — duplicating that normalization here would
            # drift and start rejecting valid receipts on the live path.
            #
            # The client already performs the binding against the payload it
            # actually sent (mlx_client._latent_reason, request_payload_sha256
            # mismatch) and refuses the result. So the facade's job is to
            # confirm the binding HAPPENED rather than to redo it with
            # different inputs — an unbound receipt is reported instead of
            # being waved through by a shape check.
            expected_request_sha256 = str(
                result.get("request_payload_sha256_bound") or ""
            )

            # CP126 94593618: the contract validator is the fail-honest path,
            # so it must not be able to raise past deep_reason's promise of
            # ok=false with a reason. A malformed receipt is a contract
            # failure, not an exception for the caller to handle.
            contract_errors = self._safe_receipt_contract_errors(
                raw_receipt,
                config,
                runtime_controls,
                worker_identity,
                result.get("tokens"),
                domain,
                output_text=result.get("text"),
                answer_replacement_private=private_answer_replacement,
                expected_objective=visible_objective,
                expected_request_payload_sha256=expected_request_sha256,
                allocated_budget=budget,
            )
            if contract_errors == ["answer_replacement_unproven"]:
                raw_flags = result_receipt.get("honest_flags")
                flags = (
                    {
                        str(flag or "").strip()
                        for flag in raw_flags
                        if str(flag or "").strip()
                    }
                    if isinstance(raw_flags, list)
                    else set()
                )
                if "vanilla_incumbent_captured_before_adaptation" in flags:
                    try:
                        from core.brain.llm.latent_cortex.answer_replacement import (
                            validate_pre_adaptation_incumbent,
                        )

                        (
                            incumbent_text,
                            incumbent_tokens,
                            incumbent_disposition,
                        ) = validate_pre_adaptation_incumbent(
                            result_receipt.get("answer_replacement"),
                            private_evidence=private_answer_replacement,
                            expected_objective=visible_objective,
                        )
                        incumbent_quality = evaluate_latent_output(
                            incumbent_text,
                            generated_tokens=len(incumbent_tokens),
                            termination=result_receipt.get("decode_termination"),
                            objective=visible_objective,
                        )
                        if incumbent_quality.get("passed") is not True:
                            raise ValueError(
                                "pre-adaptation incumbent failed output quality: "
                                + ",".join(incumbent_quality.get("reasons") or [])
                            )
                    except (ImportError, KeyError, TypeError, ValueError) as exc:
                        logger.warning(
                            "Pre-adaptation incumbent could not be reconstructed: %s",
                            str(exc)[:400],
                        )
                    else:
                        host_incumbent = (incumbent_text, incumbent_tokens)
                        result_receipt["host_incumbent_disposition"] = (
                            incumbent_disposition
                        )
                        result_receipt["host_incumbent_output_quality"] = (
                            incumbent_quality
                        )
                        result["text"] = incumbent_text
                        result["tokens"] = incumbent_tokens
                        result["receipt"] = result_receipt
            if not contract_errors and adaptive_plan is not None:
                try:
                    from core.brain.llm.latent_cortex.adaptive_compute import (
                        build_adaptive_execution_receipt,
                        validate_adaptive_execution_receipt,
                    )

                    adaptive_execution = build_adaptive_execution_receipt(
                            plan=adaptive_plan,
                            config=config,
                            budget=budget,
                            worker_receipt=result_receipt,
                    )
                    validate_adaptive_execution_receipt(adaptive_execution)
                    result_receipt["adaptive_compute"] = adaptive_execution
                except (ImportError, TypeError, ValueError, OverflowError):
                    contract_errors.append("adaptive_compute_execution_unproven")
            if not contract_errors:
                quality_receipt = evaluate_latent_output(
                    result.get("text"),
                    generated_tokens=result_receipt.get("decode_generated_tokens"),
                    termination=result_receipt.get("decode_termination"),
                    objective=visible_objective,
                )
                result_receipt["output_quality"] = quality_receipt
                result["receipt"] = result_receipt
        if operation_lease is not None:
            from core.brain.llm.latent_cortex.epistemic_state import (
                OperationOutcome,
            )

            worker_authority = result_receipt.get("runtime_operation_authority")
            authority_matches = worker_authority == operation_authority
            authority_rejected = _operation_authority_rejected(
                worker_ok=result.get("ok") is True,
                worker_receipt=result_receipt,
                expected_authority=operation_authority,
            )
            worker_succeeded = (
                result.get("ok") is True
                and authority_matches
                and action_policy_matches
                and not contract_errors
                and quality_receipt is not None
                and quality_receipt.get("passed") is True
            )
            try:
                operation_receipt = await complete_runtime_operation(
                    (OperationOutcome.SUCCEEDED if worker_succeeded else OperationOutcome.FAILED),
                    worker_receipt=result_receipt,
                    failure_code=(
                        ""
                        if worker_succeeded
                        else (
                            "operation_authority_mismatch"
                            if authority_rejected
                            else (
                                "action_policy_receipt_mismatch"
                                if result.get("ok") is True and not action_policy_matches
                                else (
                                    "worker_receipt_contract_failed"
                                    if contract_errors
                                    else (
                                        "output_quality_failed"
                                        if quality_receipt is not None
                                        and quality_receipt.get("passed") is not True
                                        else "worker_operation_failed"
                                    )
                                )
                            )
                        )
                    ),
                )
            except (
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                record_degradation(
                    "latent_cortex.operation_completion",
                    exc,
                    action=(
                        "refused recurrent output whose terminal operation state "
                        "could not be journaled"
                    ),
                    severity="error",
                )
                return self._record_failure(
                    f"runtime_operation_completion_failed:{type(exc).__name__}"
                )
            epistemic_state = operation_lease.state
            result_receipt["epistemic_operation"] = operation_receipt
            result["receipt"] = result_receipt
            if authority_rejected:
                reason = "runtime_operation_authority_mismatch"
                failed = dict(result)
                failed.update(self._record_failure(reason))
                failed["receipt"] = result_receipt
                self._last_failure_receipt = result_receipt
                return failed
            if result.get("ok") is True and not action_policy_matches:
                # Named apart, because they want opposite answers. A mismatch
                # is an integrity failure and may never bypass. An episode
                # that never selected an action has nothing to be wrong about,
                # and the act it was rehearsing should go ahead without it.
                reason = (
                    "no_action_policy_to_check"
                    if never_selected_actions
                    else "runtime_action_policy_receipt_mismatch"
                )
                if why_the_policy_did_not_match:
                    reason = f"{reason}:{why_the_policy_did_not_match}"
                failed = dict(result)
                failed.update(self._record_failure(reason))
                failed["receipt"] = result_receipt
                self._last_failure_receipt = result_receipt
                return failed
        if action_policy_matches and action_policy_evidence is not None:
            result_receipt["host_action_policy_evidence"] = dict(
                action_policy_evidence
            )
            result["receipt"] = result_receipt
        if epistemic_state is not None:
            epistemic_state_receipt = {
                "schema": epistemic_state.schema,
                "episode_id": epistemic_state.episode_id,
                "version": epistemic_state.version,
                "state_sha256": epistemic_state.state_sha256,
            }
            if selective_memory_result is not None:
                epistemic_state_receipt.update(
                    {
                        "memory_result_sha256": selective_memory_result.result_sha256,
                        "memory_evidence_ids": [
                            candidate.evidence_id
                            for candidate in selective_memory_result.candidates
                        ],
                    }
                )
            result_receipt["epistemic_state"] = epistemic_state_receipt
        raw_progress = result.get("progress")
        self._last_progress = dict(raw_progress) if isinstance(raw_progress, dict) else {}
        if result.get("ok"):
            if contract_errors:
                reason = "receipt_contract_failed:" + ",".join(contract_errors)
                # Report this condition when it APPEARS or CHANGES, not once
                # per turn.
                #
                # LIVE, 2026-08-10: the recurrent cortex declined every single
                # foreground turn with an identical contract failure — the
                # decode bridge is not wired on this path, so the refusal is
                # correct and it is also unchanging. Recording it per turn
                # opened a fresh incident every turn (INC-…-0003, -0005, -0006,
                # -0007 in one hour, each auto-resolving after 300s only to be
                # replaced), and each one pushed resilience toward the
                # depletion state that suppresses execution.
                #
                # This is exactly the case CONTRIBUTING/CLAUDE describe: log a
                # persistent, total condition at info and record a degradation
                # when it is new or has changed. _record_failure below still
                # increments the streak and retains the receipt, so nothing
                # about the refusal becomes invisible — only the duplicate
                # incident does.
                if reason == self._last_refusal:
                    logger.info(
                        "Recurrent latent cortex still declining on the same "
                        "unchanged contract failure (streak=%d): %s",
                        self._failure_streak + 1,
                        reason,
                    )
                else:
                    record_degradation(
                        "latent_cortex",
                        RuntimeError(reason),
                        action="refused to count incomplete latent episode as successful",
                        severity="degraded",
                    )
                failed = dict(result)
                failed.update(self._record_failure(reason))
                if host_incumbent is not None:
                    failed["text"], failed["tokens"] = host_incumbent
                failed["receipt"] = result_receipt
                self._last_failure_receipt = result_receipt
                return failed
            if quality_receipt is None or quality_receipt.get("passed") is not True:
                reasons = (
                    quality_receipt.get("reasons")
                    if quality_receipt is not None
                    else ["missing_quality_receipt"]
                )
                reason = "output_quality_failed:" + ",".join(
                    str(item) for item in reasons or ["unknown"]
                )
                record_degradation(
                    "latent_cortex.output_quality",
                    RuntimeError(reason),
                    action=(
                        "refused a mechanically complete latent episode whose visible answer did not satisfy the product contract"
                    ),
                    severity="degraded",
                )
                failed = dict(result)
                failed.update(self._record_failure(reason))
                failed["receipt"] = result_receipt
                self._last_failure_receipt = result_receipt
                return failed
            result_receipt["verified_replay"] = await self._capture_verified_replay(
                receipt=result_receipt,
                private_evidence=private_answer_replacement,
                objective=visible_objective,
                output_text=result.get("text"),
                output_tokens=result.get("tokens"),
                output_quality=quality_receipt,
            )
            result["receipt"] = result_receipt
            # CP126 e1c09324 / 9b110bc5 / fc19e25e / 265b0fae. Everything this
            # facade verifies about the episode comes from the worker's own
            # receipt. The format checks are real and the internal-consistency
            # checks are real, but a dishonest or corrupted worker can submit a
            # self-consistent receipt and nothing here would know: the facade
            # and the worker are not separate trust domains.
            #
            # Rather than let "ok" imply more than it earns, every episode says
            # exactly what it proved and what it did not. A consumer promoting
            # an adapter, citing a reasoning gain, or trusting an answer's
            # provenance must read this block, not the boolean.
            result["attestation"] = self._attestation_disclosure(result_receipt)
            self._ok_episodes += 1
            self._failure_streak = 0
            self._last_refusal = ""
            self._last_success_at = time.time()
            # The grading path for cognition.effort. Until this existed the
            # control point was registered, recording, and permanently
            # unpromotable: nothing ever called note_grade(), so every effort
            # decision resolved UNOBSERVED. It reports the SAME independently
            # graded outcome the bandit is allowed to learn from — a verifier's
            # judgement of the answer — and nothing else. If no verifier graded
            # this episode (outcome_checked is False), nothing is reported and
            # the decision stays honestly UNOBSERVED rather than being taught
            # from latency, convergence, or the answer's own confidence.
            #
            # Deliberately outside the controller branch below: the effort
            # choice is made on every episode, so it is graded on every episode
            # a verifier actually graded, not only on the ones that also took
            # the execution-controller path.
            effort_episode_id = str(budget.get("ontogeny_episode") or "")
            if effort_episode_id:
                try:
                    effort_score, effort_checked, _passed, _reason = _controller_outcome(
                        result_receipt.get("verifier_guidance")
                    )
                    if effort_checked:
                        from core.ontogeny.control_points import get_effort_resolver

                        get_effort_resolver().note_grade(
                            effort_episode_id, verified_score=effort_score
                        )
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation(
                        "latent_cortex_service", exc, severity="debug",
                        action="effort decision left ungraded for this episode",
                    )
            # Controller learning accepts only an independently graded task
            # outcome. Candidate-local arithmetic, syntax, facet, and
            # grounding scores still steer this episode, but cannot become a
            # Wilson trial or teach the bandit that the whole answer was right.
            if controller_decision is not None:
                try:
                    from core.brain.llm.latent_cortex.execution_controller import (
                        get_execution_controller,
                    )

                    verifier_evidence = result_receipt.get("verifier_guidance")
                    (
                        best_score,
                        outcome_checked,
                        outcome_passed,
                        outcome_reason,
                    ) = _controller_outcome(verifier_evidence)
                    # CP126 3b3d44e8: the outcome must be bound to the DECISION
                    # that produced it — a caller-asserted bucket/arm could
                    # credit any arm, including recording a base execution as
                    # a treatment.
                    self._controller_outcome_recorded_for = str(
                        controller_decision.get("decision_id") or ""
                    )
                    outcome_recorded = get_execution_controller().record_outcome(
                        bucket=str(controller_decision.get("bucket") or ""),
                        arm=str(controller_decision.get("arm") or "base"),
                        verified_score=best_score,
                        success=outcome_passed,
                        checked=outcome_checked,
                        wall_clock_s=time.monotonic() - started,
                        decision_id=str(controller_decision.get("decision_id") or ""),
                    )
                    checked_action_transitions = [
                        row for row in action_transitions if row["checked"] is True
                    ]
                    action_outcomes_recorded = (
                        get_execution_controller().record_action_transitions(
                            checked_action_transitions
                        )
                        if checked_action_transitions
                        else False
                    )
                    result_receipt["execution_controller"] = {
                        **controller_decision,
                        "outcome_recorded": outcome_recorded,
                        "outcome_checked": outcome_checked,
                        "outcome_passed": (outcome_passed if outcome_checked else None),
                        "outcome_reason": outcome_reason,
                        "action_transitions_checked": len(checked_action_transitions),
                        "action_outcomes_recorded": action_outcomes_recorded,
                    }
                    result["receipt"] = result_receipt
                except (
                    ImportError,
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    OSError,
                ) as exc:
                    logger.debug("Controller outcome not recorded: %s", exc)
            # Identity consistency: the canonical self verifies the
            # conclusion (persona displacement, forbidden intentions, core
            # values). The verdict PRICES the broadcast — an inconsistent
            # thought must outcompete honestly, never silently erased.
            try:
                from core.self.identity_consistency import (
                    check_identity_consistency,
                )

                result_receipt["identity_consistency"] = check_identity_consistency(
                    str(result.get("text") or "")
                )
                result["receipt"] = result_receipt
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Identity consistency check skipped: %s", exc)
            # Held-out grading queue: every facet judgment on the winning
            # candidate becomes a gradeable Foundry verdict (excerpt
            # attached), so facet-cue reliability is measured against human
            # ground truth instead of trusted forever.
            self._record_facet_judgments(
                result_receipt,
                domain,
                self._visible_objective(question, messages),
            )
            # RLC→GWT coupling: the conclusion returns to the workspace as a
            # competing coalition (priced by how it was earned) BEFORE any
            # action path consumes it — deliberation revises the broadcast,
            # the broadcast reaches the Will through the normal competition.
            # Lab/background episodes never write into the live mind.
            if foreground_request and publish_workspace_conclusion:
                await self._broadcast_conclusion(
                    result,
                    objective=self._visible_objective(question, messages),
                    stakes=stakes,
                )
                result_receipt = result.get("receipt", result_receipt)
            self._last_receipt = result_receipt
            self._last_failure_receipt = {}
            logger.info(
                "🧠 Latent episode ok: %d steps, %d branches, halt=%s, %.1fs",
                int(self._last_receipt.get("steps_taken") or 0),
                int(self._last_receipt.get("n_branches") or 0),
                self._last_receipt.get("halting_reason"),
                elapsed,
            )
        else:
            self._last_failure_receipt = result_receipt
            # CP126 5879d2b5: the refusal receipt was BUILT here and then
            # thrown away — the raw client dict was returned, so a client
            # failure reached the caller with no stage, no timing and nothing
            # tying it to this call. Attach it.
            refusal = self._record_failure(
                str(result.get("reason") or "unknown"),
                stage=str(
                    result_receipt.get("last_stage")
                    or self._last_progress.get("stage")
                    or "client"
                ),
                evidence={
                    "input_token_count": result_receipt.get("input_token_count"),
                    "elapsed_s": round(elapsed, 3),
                },
            )
            result.setdefault("refusal_receipt", refusal["refusal_receipt"])
            logger.info(
                "🧠 Latent episode refused/failed: %s (%.1fs) stage=%s "
                "input_tokens=%s timings=%s progress=%s",
                self._last_refusal,
                elapsed,
                result_receipt.get("last_stage") or self._last_progress.get("stage") or "unknown",
                result_receipt.get("input_token_count")
                or self._last_progress.get("input_tokens")
                or "unknown",
                result_receipt.get("stage_timings_s") or {},
                self._last_progress,
            )
        return result

    # ── Health ──────────────────────────────────────────────────────────
    def get_status(self) -> dict[str, Any]:
        enabled = _cortex_enabled()
        state = (
            "disabled"
            if not enabled
            else "degraded"
            if self._failure_streak >= 3
            else "operational"
            if self._last_success_at > 0.0
            else "idle_unproven"
        )
        try:
            from core.brain.llm.semantic_neural_serving import (
                semantic_neural_default_serving_status,
            )

            qualified_recurrent_serving = semantic_neural_default_serving_status()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            qualified_recurrent_serving = {
                "active": False,
                "reason": f"qualified_recurrent_status_failed:{type(exc).__name__}",
            }
        return {
            "enabled": enabled,
            "state": state,
            "episodes": self._episodes,
            "ok_episodes": self._ok_episodes,
            "failure_streak": self._failure_streak,
            "last_refusal": self._last_refusal,
            "last_attempt_at": self._last_attempt_at,
            "last_success_at": self._last_success_at,
            "last_latency_s": round(self._last_latency_s, 3),
            "last_allocation": dict(self._last_allocation),
            "last_progress": dict(self._last_progress),
            "last_replay_sft_publication": {
                key: self._last_replay_sft_publication.get(key)
                for key in (
                    "status",
                    "generation_id",
                    "candidate_package_sha256",
                    "evaluator_package_sha256",
                    "custody_root_sha256",
                    "publication_commit_sha256",
                    "protector_key_provenance",
                    "protector_key_identity_sha256",
                    "trainer_ready",
                    "training_authority",
                )
                if key in self._last_replay_sft_publication
            },
            "last_failure_receipt": {
                key: self._last_failure_receipt.get(key)
                for key in (
                    "episode_id",
                    "input_token_count",
                    "params_unchanged",
                    "fast_weights_attach_attempted",
                    "fast_weights_applied",
                    "fast_weights_erased",
                    "fast_weight_optimizer",
                    "fast_weight_loss_trail",
                    "fast_weight_gradient_norm_trail",
                    "fast_weight_accepted_step_sizes",
                    "fast_weight_line_search_backtracks",
                    "fast_weight_learning",
                    "fast_weight_cleanup",
                    "worker_identity",
                    "runtime_integrity",
                    "verifier_probe_max_tokens",
                    "verifier_probe_contract",
                    "latent_opt_verifier",
                    "last_stage",
                    "stage_timings_s",
                    "honest_flags",
                    "epistemic_state",
                    "adaptive_compute",
                    "adaptive_acquisition",
                    "resident_owner_released",
                    "resident_state_reusable",
                )
                if key in self._last_failure_receipt
            },
            "last_receipt": {
                k: self._last_receipt.get(k)
                for k in (
                    "episode_id",
                    "steps_taken",
                    "halting_reason",
                    "terminal_disposition",
                    "causal_receipt",
                    "n_slots",
                    "n_branches",
                    "exchanges",
                    "schedule_hash",
                    "checkpoint_fingerprint",
                    "checkpoint_fingerprint_method",
                    "checkpoint_file_count",
                    "worker_boot_id",
                    "worker_pid",
                    "worker_model_path",
                    "worker_model_parameter_count",
                    "worker_model_stored_parameter_element_count",
                    "worker_model_parameter_count_basis",
                    "worker_source_sha256",
                    "worker_affective_steering_active",
                    "worker_affective_steering_alpha",
                    "episode_affective_steering_applied",
                    "episode_affective_steering_alpha",
                    "request_payload_sha256",
                    "input_tokens_sha256",
                    "input_token_count",
                    "input_context_compaction",
                    "worker_identity",
                    "runtime_integrity",
                    "runtime_identity",
                    "params_unchanged",
                    "latent_opt_applied",
                    "latent_opt_attempts",
                    "latent_opt_steps",
                    "latent_opt_budget_exhausted",
                    "verifier_probe_max_tokens",
                    "verifier_probe_contract",
                    "latent_opt_verifier",
                    "fast_weights_attach_attempted",
                    "fast_weights_applied",
                    "fast_weight_optimization_attempts",
                    "fast_weight_optimized_steps",
                    "fast_weight_budget_exhausted",
                    "fast_weight_learning",
                    "fast_weight_cleanup",
                    "fast_weights_erased",
                    "decode_requested_tokens",
                    "decode_generated_tokens",
                    "decode_termination",
                    "last_stage",
                    "stage_timings_s",
                    "honest_flags",
                    "epistemic_state",
                )
                if k in self._last_receipt
            },
            "qualified_recurrent_serving": qualified_recurrent_serving,
            "healthy": state == "operational",
        }


_INSTANCE: LatentCortexService | None = None


def get_latent_cortex_service(orchestrator: Any = None) -> LatentCortexService:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LatentCortexService(orchestrator=orchestrator)
    return _INSTANCE


def _report_qualified_recurrent_serving_status(status: dict[str, Any]) -> None:
    """Make the promoted bounded serving lane explicit at runtime boot."""

    reason = str(status.get("reason") or "unknown")
    if status.get("active") is True:
        receipt = status.get("receipt")
        receipt = receipt if isinstance(receipt, dict) else {}
        families = receipt.get("allowed_families")
        family_count = len(families) if isinstance(families, list) else 0
        logger.info(
            "Qualified semantic-neural serving ACTIVE: package=%s mode=%s "
            "promotion=%s families=%d activation=%s",
            receipt.get("package_id") or "unknown",
            receipt.get("mode") or "unknown",
            receipt.get("promotion_mode") or "unknown",
            family_count,
            receipt.get("activation_sha256") or "unknown",
        )
        return
    if reason == "semantic_neural_serving_disabled":
        logger.info("Qualified semantic-neural serving disabled by explicit kill switch")
        return
    if (
        status.get("lifecycle") == "deferred"
        and reason == "semantic_neural_model_basis_migration_deferred"
    ):
        logger.info(
            "Qualified semantic-neural serving DEFERRED for active cortex basis: "
            "descriptor=%s authority=%s historical_activation_preserved=%s",
            status.get("model_descriptor_sha256") or "unknown",
            status.get("authority_sha256") or "unknown",
            status.get("historical_activation_preserved") is True,
        )
        return
    record_degradation(
        "latent_cortex.qualified_recurrent_serving",
        RuntimeError(reason),
        severity="warning",
        action=(
            "kept the general latent cortex available while exposing the inactive "
            "certified qualified package"
        ),
        enforce_failure_policy=False,
    )


def register_latent_cortex(orchestrator: Any = None) -> LatentCortexService:
    from core.runtime.service_registry import get_runtime_service, register_runtime_service
    from core.service_names import ServiceNames

    inst = get_runtime_service(
        ServiceNames.LATENT_CORTEX, default=None
    ) or get_latent_cortex_service(orchestrator)
    register_runtime_service(
        ServiceNames.LATENT_CORTEX,
        inst,
        required=False,
        owner="core/brain/latent_cortex_service.py",
        registered_by="register_latent_cortex",
    )
    try:
        from core.brain.llm.semantic_neural_serving import (
            semantic_neural_default_serving_status,
        )

        qualified_status = semantic_neural_default_serving_status()
        _report_qualified_recurrent_serving_status(qualified_status)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "latent_cortex.qualified_recurrent_serving",
            exc,
            severity="warning",
            action="exposed qualified-package status initialization failure",
            enforce_failure_policy=False,
        )
    # The selective-memory bridge resolves organs through the same runtime
    # registry as every other cognitive signal. Register the existing
    # playbook and reasoning-reflection stores as named organs; this does not
    # create new memory databases or duplicate their ownership.
    for service_name, getter in (
        (
            "procedural_memory",
            lambda: __import__(
                "core.brain.procedural_memory",
                fromlist=["get_procedural_memory"],
            ).get_procedural_memory(),
        ),
        (
            "reasoning_memory",
            lambda: __import__(
                "core.brain.reasoning_memory",
                fromlist=["get_reasoning_memory"],
            ).get_reasoning_memory(),
        ),
    ):
        if get_runtime_service(service_name, default=None) is not None:
            continue
        try:
            register_runtime_service(
                service_name,
                getter(),
                required=False,
                owner="core/brain/latent_cortex_service.py",
                registered_by="register_latent_cortex",
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Selective-memory organ registration skipped (%s): %s", service_name, exc)
    return inst


__all__ = [
    "LatentCortexService",
    "get_latent_cortex_service",
    "register_latent_cortex",
]
