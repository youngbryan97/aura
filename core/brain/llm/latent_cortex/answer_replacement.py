"""Confidence-bound authority for promoting a locally repaired answer.

The authority object is deliberately narrow: full-span validity of atomic
claims handled by semantic deterministic verifiers. Syntax-only parsers can
refute malformed code/JSON, but cannot certify semantic correctness. Unknown
or partially covered prose keeps the interval at [0, 1].

Private candidate text crosses the worker/service IPC boundary only long enough
for the service to rebuild decomposition and verifier evidence. Public receipts
contain commitments, intervals, and decisions, never hidden candidate prose.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
    validate_atomic_decomposition_envelope,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    build_deterministic_router_receipt,
    validate_deterministic_router_envelope,
)
from core.brain.llm.latent_cortex.local_repair import (
    validate_local_repair_receipt,
)
from core.brain.llm.latent_cortex.objective_program_verifier import (
    verify_objective_program,
)
from core.brain.llm.latent_cortex.neural_objective_producer import (
    solve_objective_program_neural as solve_objective_program,
    validate_objective_program_solution_neural as validate_objective_program_solution,
)

ANSWER_REPLACEMENT_SCHEMA = "aura.rlc.answer_replacement.v5"
ANSWER_REPLACEMENT_PRIVATE_SCHEMA = "aura.rlc.answer_replacement_private.v3"
HOST_INCUMBENT_DISPOSITION_SCHEMA = "aura.rlc.host_incumbent_disposition.v1"
DEFAULT_REPLACEMENT_MARGIN = 0.05
MAX_REPLACEMENT_OUTPUT_TOKENS = 1024
# Baseline evidence binds output that was already admitted by the engine's
# decode limit (8192) plus its contract-completion grace (4096).  It is not a
# replacement candidate and must not inherit the narrower promotion ceiling.
MAX_BASELINE_EVIDENCE_TOKENS = 12_288
_REFUTATION_VERIFIERS = {
    "exact_integer_arithmetic",
    "exact_modular_arithmetic",
    "exact_objective_program",
    "python_ast",
    "json_parser",
}
_SEMANTIC_EXACT_VERIFIERS = {
    "exact_integer_arithmetic",
    "exact_modular_arithmetic",
    "exact_objective_program",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FULL_INTEGER_ARITHMETIC_RE = re.compile(
    r"\s*-?\d{1,12}\s*[+\-*/x×]\s*-?\d{1,12}\s*=\s*-?\d{1,12}\s*[.!?]?\s*\Z"
)
_ANSWER_REPLACEMENT_FIELDS = {
    "schema",
    "disagreement_graph_sha256",
    "diagnostic_selection_sha256",
    "local_repair_sha256",
    "private_evidence_required",
    "private_evidence_sha256",
    "selected_branch",
    "policy",
    "baseline_decomposition",
    "baseline_routes",
    "baseline_quality",
    "selected_branch_quality",
    "candidates",
    "intended_decision",
    "decision",
    "reason",
    "selected_request_id",
    "baseline_decode",
    "accepted_output",
    "answer_selection_effect",
    "latent_state_effect",
    "authority",
    "receipt_sha256",
}
_HOST_INCUMBENT_DISPOSITION_FIELDS = {
    "schema",
    "authority",
    "source_answer_replacement_sha256",
    "private_evidence_sha256",
    "objective_sha256",
    "text_sha256",
    "token_count",
    "tokens_sha256",
    "baseline_decomposition_sha256",
    "baseline_routes_sha256",
    "baseline_quality_sha256",
    "receipt_sha256",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token_sha(tokens: Sequence[int]) -> str:
    return _sha(list(tokens))


def _validate_public_receipt_commitment(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ANSWER_REPLACEMENT_FIELDS:
        raise ValueError("answer replacement receipt fields differ")
    payload = {
        key: value[key]
        for key in _ANSWER_REPLACEMENT_FIELDS - {"receipt_sha256"}
    }
    if (
        value["schema"] != ANSWER_REPLACEMENT_SCHEMA
        or value["receipt_sha256"] != _sha(payload)
    ):
        raise ValueError("answer replacement receipt commitment mismatch")
    return dict(value)


def _margin(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
    ):
        raise ValueError("answer replacement margin must be finite in [0, 1)")
    return round(float(value), 10)


def _output_limit(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_REPLACEMENT_OUTPUT_TOKENS:
        raise ValueError("answer replacement output limit is invalid")
    return value


def _quality_interval(
    decomposition: Mapping[str, Any],
    routes: Mapping[str, Any],
    *,
    candidate: str,
    objective: str,
) -> dict[str, Any]:
    atomic = validate_atomic_decomposition_envelope(decomposition)
    routed = validate_deterministic_router_envelope(
        routes,
        atomic_receipt=atomic,
    )
    if _text_sha(candidate) != atomic["source_sha256"]:
        raise ValueError("answer replacement candidate source differs")
    refuted = [
        row
        for row in routed["routes"]
        if row["verifier"] in _REFUTATION_VERIFIERS and row["outcome"] == "refuted"
    ]
    objective_solution_available = solve_objective_program(objective) is not None
    objective_program_verdict = (
        verify_objective_program(candidate, objective=objective)
        if objective_solution_available
        else None
    )
    semantic_verified = 0
    partial_or_nonsemantic = 0
    objective_program_verified = False
    for atom, route in zip(atomic["atoms"], routed["routes"], strict=True):
        fragment = candidate[int(atom["start"]) : int(atom["end"])]
        if (
            route["verifier"] == "exact_objective_program"
            and route["outcome"] == "verified"
        ):
            objective_program_verified = True
        full_span_semantic = bool(
            route["verifier"] in _SEMANTIC_EXACT_VERIFIERS
            and route["outcome"] == "verified"
            and (
                route["verifier"] == "exact_objective_program"
                or _FULL_INTEGER_ARITHMETIC_RE.fullmatch(fragment)
            )
        )
        semantic_verified += int(full_span_semantic)
        partial_or_nonsemantic += int(not full_span_semantic)
    every_atom_semantically_verified = bool(atomic["atoms"]) and (
        semantic_verified == len(atomic["atoms"])
    )
    if objective_solution_available and objective_program_verdict is None:
        # A recognized public objective program defines an exact terminal
        # answer object.  Text that never supplies that object is not an
        # unmeasured answer with a possible score of one: it deterministically
        # fails the completion contract.  Keeping it at [0, 1] made a
        # token-limited incumbent impossible to displace even when another
        # candidate was independently verified at [1, 1].
        lower = upper = 0.0
        basis = "objective_program_contract_incomplete"
    elif (
        objective_program_verdict is not None
        and objective_program_verdict["outcome"] == "refuted"
    ):
        lower = upper = 0.0
        basis = "deterministic_exact_refutation"
    elif (
        objective_program_verdict is not None
        and objective_program_verdict["outcome"] == "verified"
    ):
        lower = upper = 1.0
        basis = "objective_program_exact_complete"
    elif refuted or atomic["grade_admissible"] is not True:
        lower = upper = 0.0
        basis = "deterministic_exact_refutation" if refuted else "structural_grade_refutation"
    elif objective_program_verified:
        lower = upper = 1.0
        basis = "objective_program_exact_complete"
    elif every_atom_semantically_verified:
        lower = upper = 1.0
        basis = "full_span_semantic_exact_complete"
    else:
        lower, upper = 0.0, 1.0
        basis = "incomplete_semantic_exact_coverage"
    payload = {
        "object": "conjunctive_full_span_exact_claim_validity",
        "lower_bound": lower,
        "upper_bound": upper,
        "basis": basis,
        "atom_count": len(atomic["atoms"]),
        "semantic_exact_verified_count": semantic_verified,
        "exact_refuted_count": len(refuted),
        "partial_or_nonsemantic_count": partial_or_nonsemantic,
        "decomposition_sha256": atomic["receipt_sha256"],
        "routes_sha256": routed["receipt_sha256"],
    }
    return {**payload, "interval_sha256": _sha(payload)}


def _normalize_private_evidence(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "objective",
        "branch_candidates",
        "generated_repairs",
        "objective_program_solution",
        "objective_program_solution_receipt",
        "baseline_text",
        "baseline_tokens",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("answer replacement private evidence fields differ")
    branches = value["branch_candidates"]
    repairs = value["generated_repairs"]
    objective_solution = value["objective_program_solution"]
    objective_solution_receipt = value["objective_program_solution_receipt"]
    baseline_tokens = value["baseline_tokens"]
    if (
        value["schema"] != ANSWER_REPLACEMENT_PRIVATE_SCHEMA
        or not isinstance(value["objective"], str)
        or not isinstance(value["baseline_text"], str)
        or not isinstance(baseline_tokens, list)
        or any(type(token) is not int or token < 0 for token in baseline_tokens)
        or not isinstance(branches, Mapping)
        or len(branches) > 64
        or any(
            not isinstance(key, str)
            or not key.isdigit()
            or not isinstance(text, str)
            or len(text) > 131_072
            for key, text in branches.items()
        )
        or not isinstance(repairs, Mapping)
        or len(repairs) > 8
        or any(
            _SHA256_RE.fullmatch(str(key)) is None
            or not isinstance(text, str)
            or len(text) > 131_072
            for key, text in repairs.items()
        )
        or not isinstance(objective_solution, str)
        or len(objective_solution) > 131_072
        or not isinstance(objective_solution_receipt, Mapping)
    ):
        raise ValueError("answer replacement private evidence is invalid")
    if len(baseline_tokens) > MAX_BASELINE_EVIDENCE_TOKENS:
        raise ValueError(
            "answer replacement private evidence baseline token limit exceeded"
        )
    if objective_solution:
        validate_objective_program_solution(
            objective_solution_receipt,
            objective=value["objective"],
            candidate=objective_solution,
        )
    elif objective_solution_receipt:
        raise ValueError("objective program solution receipt has no candidate")
    return {
        "schema": ANSWER_REPLACEMENT_PRIVATE_SCHEMA,
        "objective": value["objective"],
        "branch_candidates": {str(key): str(text) for key, text in sorted(branches.items())},
        "generated_repairs": {str(key): str(text) for key, text in sorted(repairs.items())},
        "objective_program_solution": objective_solution,
        "objective_program_solution_receipt": dict(objective_solution_receipt),
        "baseline_text": value["baseline_text"],
        "baseline_tokens": list(baseline_tokens),
    }


def _candidate_inventory(
    *,
    disagreement_graph: Mapping[str, Any],
    diagnostic_selection: Mapping[str, Any],
    local_repair: Mapping[str, Any],
    private_evidence: Mapping[str, Any],
    selected_branch: int,
    baseline_quality: Mapping[str, Any],
    margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decompositions = disagreement_graph.get("candidate_decompositions")
    candidate_routes = diagnostic_selection.get("candidate_routes")
    if not isinstance(decompositions, Mapping) or not isinstance(
        candidate_routes,
        Mapping,
    ):
        raise ValueError("answer replacement candidate inventory is missing")
    branch_texts = private_evidence["branch_candidates"]
    if set(branch_texts) != set(decompositions):
        raise ValueError("answer replacement private branch coverage differs")
    branch_quality: dict[int, dict[str, Any]] = {}
    for index in sorted(decompositions, key=int):
        text = branch_texts[index]
        rebuilt_decomposition = build_atomic_decomposition(
            text,
            objective=private_evidence["objective"],
        )
        rebuilt_routes = build_deterministic_router_receipt(
            text,
            objective=private_evidence["objective"],
            atomic_receipt=rebuilt_decomposition,
        )
        if (
            rebuilt_decomposition != decompositions[index]
            or rebuilt_routes != candidate_routes[index]
        ):
            raise ValueError("answer replacement candidate evidence reconstruction differs")
        branch_quality[int(index)] = _quality_interval(
            rebuilt_decomposition,
            rebuilt_routes,
            candidate=text,
            objective=private_evidence["objective"],
        )
    if selected_branch in branch_quality:
        selected_quality = branch_quality[selected_branch]
    else:
        selected_quality = _unavailable_quality()

    repairs = private_evidence["generated_repairs"]
    rows: list[dict[str, Any]] = []
    for request, transaction in zip(
        local_repair["requests"],
        local_repair["transactions"],
        strict=True,
    ):
        branch = int(request["branch"])
        replacement_available = transaction["status"] == "repaired_candidate_admitted"
        replacement_text = repairs.get(request["request_id"])
        if replacement_available and not isinstance(replacement_text, str):
            raise ValueError("admitted replacement private source is absent")
        if not replacement_available and replacement_text is not None:
            raise ValueError("rejected replacement retained private authority")
        replacement_quality = (
            _quality_interval(
                transaction["replacement_decomposition"],
                transaction["replacement_routes"],
                candidate=replacement_text,
                objective=private_evidence["objective"],
            )
            if replacement_available
            else None
        )
        if replacement_available:
            rebuilt_replacement_decomposition = build_atomic_decomposition(
                replacement_text,
                objective=private_evidence["objective"],
            )
            rebuilt_replacement_routes = build_deterministic_router_receipt(
                replacement_text,
                objective=private_evidence["objective"],
                atomic_receipt=rebuilt_replacement_decomposition,
            )
            if (
                rebuilt_replacement_decomposition
                != transaction["replacement_decomposition"]
                or rebuilt_replacement_routes != transaction["replacement_routes"]
            ):
                raise ValueError(
                    "answer replacement repair evidence reconstruction differs"
                )
        original_routes = validate_deterministic_router_envelope(
            candidate_routes[str(branch)],
            atomic_receipt=decompositions[str(branch)],
        )
        failed_ordinal = int(request["failed_atom_ordinal"])
        original_failed_route = original_routes["routes"][failed_ordinal]
        replacement_failed_route = (
            transaction["replacement_routes"]["routes"][failed_ordinal]
            if replacement_available
            else None
        )
        same_verifier_class = bool(
            replacement_failed_route
            and original_failed_route["verifier"] == request["required_verifier"]
            and replacement_failed_route["verifier"] == request["required_verifier"]
            and original_failed_route["outcome"] == "refuted"
            and replacement_failed_route["outcome"] == "verified"
        )
        dominates = bool(
            branch == selected_branch
            and replacement_quality is not None
            and same_verifier_class
            and float(replacement_quality["lower_bound"])
            > float(baseline_quality["upper_bound"]) + margin
        )
        payload = {
            "request_id": request["request_id"],
            "branch": branch,
            "transaction_sha256": transaction["transaction_sha256"],
            "transaction_status": transaction["status"],
            "required_verifier": request["required_verifier"],
            "same_verifier_class": same_verifier_class,
            "source_branch_quality": branch_quality[branch],
            "replacement_quality": replacement_quality,
            "dominance_margin": margin,
            "compared_against": "actual_final_decode",
            "dominates": dominates,
        }
        rows.append({**payload, "candidate_decision_sha256": _sha(payload)})

    # Branch candidates are promotable too, not just repairs of the incumbent.
    #
    # Until now `rows` came only from local_repair requests, so the recurrent
    # path's own answers -- the entire product of the workspace, the branches
    # and the recurrence -- had no route to becoming the output under
    # vanilla_incumbent. The only way a latent answer could ever be served was
    # decode_incumbent_policy="latent", which hands it the output
    # unconditionally and removes the floor. That is why the floor and the gain
    # were mutually exclusive: repairs could win safely, branches could only
    # win recklessly.
    #
    # A branch wins here on exactly the same evidence rule as a repair: its
    # lower confidence bound must clear the incumbent's upper bound plus the
    # margin. The floor is preserved because the incumbent is ordinary decode
    # and nothing displaces it without dominating it on measured evidence.
    for index in sorted(branch_quality):
        quality = branch_quality[index]
        # An unmeasured candidate has bounds [0, 1] and can never dominate; it
        # is recorded rather than silently skipped so the receipt shows every
        # candidate that was considered.
        dominates = bool(
            float(quality["lower_bound"])
            > float(baseline_quality["upper_bound"]) + margin
        )
        payload = {
            "request_id": f"branch-{index}",
            "branch": index,
            "transaction_sha256": "",
            "transaction_status": "branch_candidate",
            "required_verifier": "",
            "same_verifier_class": False,
            "source_branch_quality": quality,
            "replacement_quality": quality,
            "dominance_margin": margin,
            "compared_against": "actual_final_decode",
            "dominates": dominates,
        }
        rows.append({**payload, "candidate_decision_sha256": _sha(payload)})
    objective_solution = private_evidence["objective_program_solution"]
    if objective_solution:
        solution_receipt = validate_objective_program_solution(
            private_evidence["objective_program_solution_receipt"],
            objective=private_evidence["objective"],
            candidate=objective_solution,
        )
        solution_decomposition = build_atomic_decomposition(
            objective_solution,
            objective=private_evidence["objective"],
        )
        solution_routes = build_deterministic_router_receipt(
            objective_solution,
            objective=private_evidence["objective"],
            atomic_receipt=solution_decomposition,
        )
        solution_quality = _quality_interval(
            solution_decomposition,
            solution_routes,
            candidate=objective_solution,
            objective=private_evidence["objective"],
        )
        dominates = bool(
            float(solution_quality["lower_bound"])
            > float(baseline_quality["upper_bound"]) + margin
        )
        payload = {
            "request_id": "objective-program",
            "branch": -1,
            "transaction_sha256": solution_receipt["receipt_sha256"],
            "transaction_status": "objective_program_solution",
            "required_verifier": "exact_objective_program",
            "same_verifier_class": True,
            "source_branch_quality": solution_quality,
            "replacement_quality": solution_quality,
            "dominance_margin": margin,
            "compared_against": "actual_final_decode",
            "dominates": dominates,
        }
        rows.append({**payload, "candidate_decision_sha256": _sha(payload)})
    return rows, selected_quality


def _unavailable_quality() -> dict[str, Any]:
    payload = {
        "object": "conjunctive_full_span_exact_claim_validity",
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "basis": "candidate_probe_unavailable",
        "atom_count": 0,
        "semantic_exact_verified_count": 0,
        "exact_refuted_count": 0,
        "partial_or_nonsemantic_count": 0,
        "decomposition_sha256": "",
        "routes_sha256": "",
    }
    return {**payload, "interval_sha256": _sha(payload)}


def _intended_decision(
    rows: list[dict[str, Any]],
    *,
    enabled: bool,
    selected_branch_quality: Mapping[str, Any],
    baseline_quality: Mapping[str, Any],
) -> tuple[str, str, str]:
    if not enabled:
        return "retain", "answer_replacement_disabled", ""
    dominant = [row for row in rows if row["dominates"]]
    if dominant:
        winner = sorted(
            dominant,
            key=lambda row: (
                -float(row["replacement_quality"]["lower_bound"]),
                row["request_id"],
            ),
        )[0]
        return (
            "replace",
            "replacement_lower_bound_exceeds_final_decode_upper_bound_plus_margin",
            str(winner["request_id"]),
        )
    if not rows:
        if baseline_quality["basis"] == "deterministic_exact_refutation":
            return "abstain", "known_refutation_has_no_dominant_repair", ""
        return "retain", "no_local_repair_candidates", ""
    if baseline_quality["basis"] == "full_span_semantic_exact_complete":
        return "retain", "final_decode_already_exactly_verified", ""
    if (
        baseline_quality["basis"] == "deterministic_exact_refutation"
        or selected_branch_quality["basis"] == "deterministic_exact_refutation"
    ):
        return "abstain", "known_refutation_has_no_dominant_repair", ""
    return "retain", "no_proven_dominance_or_known_refutation", ""


def build_answer_replacement_receipt(
    *,
    disagreement_graph: Any,
    diagnostic_selection: Any,
    local_repair: Any,
    selected_branch: int,
    branch_candidates: Mapping[int, str],
    generated_repairs: Mapping[str, Mapping[str, Any]],
    objective: str,
    baseline_text: str,
    baseline_tokens: Sequence[int],
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
    enabled: bool = True,
    objective_program_enabled: bool = True,
    margin: float = DEFAULT_REPLACEMENT_MARGIN,
    max_output_tokens: int,
) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    """Select and bind output, returning private evidence for service replay."""

    if type(enabled) is not bool:
        raise ValueError("answer replacement enabled flag must be boolean")
    if type(objective_program_enabled) is not bool:
        raise ValueError("objective program enabled flag must be boolean")
    if type(selected_branch) is not int or selected_branch < 0:
        raise ValueError("answer replacement selected branch is invalid")
    if not isinstance(disagreement_graph, Mapping) or not isinstance(
        diagnostic_selection,
        Mapping,
    ):
        raise ValueError("answer replacement upstream evidence is missing")
    if not isinstance(local_repair, Mapping):
        raise ValueError("answer replacement local repair is missing")
    if not isinstance(objective, str) or not isinstance(baseline_text, str):
        raise ValueError("answer replacement text inputs are invalid")
    baseline = list(baseline_tokens)
    if any(type(token) is not int or token < 0 for token in baseline):
        raise ValueError("answer replacement baseline tokens are invalid")
    normalized_margin = _margin(margin)
    output_limit = _output_limit(max_output_tokens)
    validate_local_repair_receipt(
        local_repair,
        disagreement_graph=disagreement_graph,
        diagnostic_selection=diagnostic_selection,
    )
    baseline_decomposition = build_atomic_decomposition(
        baseline_text,
        objective=objective,
    )
    baseline_routes = build_deterministic_router_receipt(
        baseline_text,
        objective=objective,
        atomic_receipt=baseline_decomposition,
    )
    baseline_quality = _quality_interval(
        baseline_decomposition,
        baseline_routes,
        candidate=baseline_text,
        objective=objective,
    )
    admitted_request_ids = {
        str(transaction["request_id"])
        for transaction in local_repair["transactions"]
        if transaction["status"] == "repaired_candidate_admitted"
    }
    objective_program_solution = (
        solve_objective_program(objective) if objective_program_enabled else None
    )
    objective_solution_text = (
        objective_program_solution[0] if objective_program_solution is not None else ""
    )
    objective_solution_receipt = (
        objective_program_solution[1] if objective_program_solution is not None else {}
    )
    private_evidence_required = (
        bool(branch_candidates)
        or bool(local_repair["requests"])
        or bool(objective_solution_text)
        or baseline_quality["basis"] == "deterministic_exact_refutation"
    )
    private_evidence = (
        _normalize_private_evidence(
            {
                "schema": ANSWER_REPLACEMENT_PRIVATE_SCHEMA,
                "objective": objective,
                "branch_candidates": {
                    str(index): text for index, text in branch_candidates.items()
                },
                "generated_repairs": {
                    request_id: str(result["candidate"])
                    for request_id, result in generated_repairs.items()
                    if request_id in admitted_request_ids
                    if isinstance(result, Mapping) and isinstance(result.get("candidate"), str)
                },
                "objective_program_solution": objective_solution_text,
                "objective_program_solution_receipt": objective_solution_receipt,
                "baseline_text": baseline_text,
                "baseline_tokens": baseline,
            }
        )
        if private_evidence_required
        else {}
    )
    if private_evidence_required:
        rows, selected_quality = _candidate_inventory(
            disagreement_graph=disagreement_graph,
            diagnostic_selection=diagnostic_selection,
            local_repair=local_repair,
            private_evidence=private_evidence,
            selected_branch=selected_branch,
            baseline_quality=baseline_quality,
            margin=normalized_margin,
        )
    else:
        rows, selected_quality = [], _unavailable_quality()
    intended, reason, selected_request_id = _intended_decision(
        rows,
        enabled=enabled,
        selected_branch_quality=selected_quality,
        baseline_quality=baseline_quality,
    )
    decision = intended
    binding_status = "not_required"
    accepted_text = baseline_text
    accepted_tokens = baseline
    if intended == "replace":
        # A promoted candidate is either a repair of the incumbent or a branch
        # answer; both bind their output the same way, by exact text/token
        # round-trip, and both fail closed to abstain if that binding cannot be
        # proven. Branch ids carry the "branch-" prefix their row was built
        # with, so the source is unambiguous.
        if selected_request_id == "objective-program":
            candidate = private_evidence["objective_program_solution"]
        elif selected_request_id.startswith("branch-"):
            candidate = private_evidence["branch_candidates"].get(
                selected_request_id.removeprefix("branch-")
            )
        else:
            candidate = private_evidence["generated_repairs"].get(selected_request_id)
        try:
            if not isinstance(candidate, str):
                raise ValueError("replacement private source is absent")
            encoded = list(encode(candidate))
            if (
                not encoded
                or any(type(token) is not int or token < 0 for token in encoded)
                or len(encoded) > output_limit
                or decode(encoded) != candidate
            ):
                raise ValueError("replacement output binding failed")
        except (AttributeError, KeyError, TypeError, ValueError):
            decision = "abstain"
            reason = (
                "objective_program_output_binding_failed"
                if selected_request_id == "objective-program"
                else
                "dominant_branch_output_binding_failed"
                if selected_request_id.startswith("branch-")
                else "dominant_repair_output_binding_failed"
            )
            binding_status = "failed_closed"
            accepted_text = ""
            accepted_tokens = []
        else:
            binding_status = "exact_text_token_roundtrip"
            accepted_text = candidate
            accepted_tokens = encoded
    elif intended == "abstain":
        accepted_text = ""
        accepted_tokens = []
    baseline_binding = {
        "text_sha256": _text_sha(baseline_text),
        "token_count": len(baseline),
        "tokens_sha256": _token_sha(baseline),
    }
    output_binding = {
        "source": (
            (
                "objective_program_solution"
                if selected_request_id == "objective-program"
                else "branch_candidate"
                if selected_request_id.startswith("branch-")
                else "repaired_candidate"
            )
            if decision == "replace"
            else "baseline_decode"
            if decision == "retain"
            else "none"
        ),
        "text_sha256": _text_sha(accepted_text) if decision != "abstain" else "",
        "token_count": len(accepted_tokens),
        "tokens_sha256": (_token_sha(accepted_tokens) if decision != "abstain" else ""),
        "binding_status": binding_status,
    }
    policy = {
        "enabled": enabled,
        "objective_program_enabled": objective_program_enabled,
        "margin": normalized_margin,
        "max_output_tokens": output_limit,
        "interval_object": "conjunctive_full_span_exact_claim_validity",
        "replacement_rule": "new_lower_gt_final_decode_upper_plus_margin",
        "syntax_only_verifier_policy": "refutation_only",
        "unknown_claim_policy": "interval_zero_to_one_no_authority",
        "objective_completion_gate": "parent_service_output_quality",
    }
    payload = {
        "schema": ANSWER_REPLACEMENT_SCHEMA,
        "disagreement_graph_sha256": disagreement_graph["receipt_sha256"],
        "diagnostic_selection_sha256": diagnostic_selection["receipt_sha256"],
        "local_repair_sha256": local_repair["receipt_sha256"],
        "private_evidence_required": private_evidence_required,
        "private_evidence_sha256": _sha(private_evidence),
        "selected_branch": selected_branch,
        "policy": policy,
        "baseline_decomposition": baseline_decomposition,
        "baseline_routes": baseline_routes,
        "baseline_quality": baseline_quality,
        "selected_branch_quality": selected_quality,
        "candidates": rows,
        "intended_decision": intended,
        "decision": decision,
        "reason": reason,
        "selected_request_id": selected_request_id,
        "baseline_decode": baseline_binding,
        "accepted_output": output_binding,
        "answer_selection_effect": (
            "replaced"
            if decision == "replace"
            else "retained"
            if decision == "retain"
            else "abstained"
        ),
        "latent_state_effect": "none",
        "authority": "confidence_bound_answer_replacement",
    }
    receipt = {**payload, "receipt_sha256": _sha(payload)}
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=disagreement_graph,
        diagnostic_selection=diagnostic_selection,
        local_repair=local_repair,
        private_evidence=private_evidence,
        expected_objective=objective,
        expected_selected_branch=selected_branch,
        expected_enabled=enabled,
        expected_objective_program_enabled=objective_program_enabled,
        expected_margin=normalized_margin,
        expected_max_output_tokens=output_limit,
        expected_output_text=accepted_text,
        expected_output_tokens=accepted_tokens,
    )
    return receipt, accepted_tokens, private_evidence


def validate_answer_replacement_receipt(
    value: Any,
    *,
    disagreement_graph: Any,
    diagnostic_selection: Any,
    local_repair: Any,
    private_evidence: Any,
    expected_objective: str,
    expected_selected_branch: int,
    expected_enabled: bool,
    expected_objective_program_enabled: bool = True,
    expected_margin: float,
    expected_max_output_tokens: int,
    expected_output_text: str | None = None,
    expected_output_tokens: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Rebuild baseline and repair evidence in the validating trust domain."""

    value = _validate_public_receipt_commitment(value)
    if type(expected_enabled) is not bool or not isinstance(
        expected_objective,
        str,
    ):
        raise ValueError("answer replacement expected policy is invalid")
    margin = _margin(expected_margin)
    output_limit = _output_limit(expected_max_output_tokens)
    validate_local_repair_receipt(
        local_repair,
        disagreement_graph=disagreement_graph,
        diagnostic_selection=diagnostic_selection,
    )
    candidate_decompositions = disagreement_graph.get("candidate_decompositions")
    if type(expected_objective_program_enabled) is not bool:
        raise ValueError("answer replacement expected objective-program policy is invalid")
    expected_objective_solution = (
        solve_objective_program(expected_objective)
        if expected_objective_program_enabled
        else None
    )
    private_required = (
        bool(candidate_decompositions)
        or bool(local_repair["requests"])
        or expected_objective_solution is not None
        or (
            isinstance(value.get("baseline_quality"), Mapping)
            and value["baseline_quality"].get("basis") == "deterministic_exact_refutation"
        )
    )
    if value["private_evidence_required"] is not private_required:
        raise ValueError("answer replacement private evidence policy differs")
    if private_required:
        private = _normalize_private_evidence(private_evidence)
        if private["objective"] != expected_objective or value["private_evidence_sha256"] != _sha(
            private
        ):
            raise ValueError("answer replacement private evidence binding differs")
        baseline_text = private["baseline_text"]
        baseline_tokens = private["baseline_tokens"]
    else:
        if private_evidence != {} or value["private_evidence_sha256"] != _sha({}):
            raise ValueError("answer replacement no-op retained private evidence")
        if expected_output_text is None or expected_output_tokens is None:
            raise ValueError("answer replacement no-op output binding is absent")
        baseline_text = expected_output_text
        baseline_tokens = list(expected_output_tokens)
        if any(type(token) is not int or token < 0 for token in baseline_tokens):
            raise ValueError("answer replacement no-op output tokens are invalid")
        private = {}
    baseline_decomposition = build_atomic_decomposition(
        baseline_text,
        objective=expected_objective,
    )
    baseline_routes = build_deterministic_router_receipt(
        baseline_text,
        objective=expected_objective,
        atomic_receipt=baseline_decomposition,
    )
    baseline_quality = _quality_interval(
        baseline_decomposition,
        baseline_routes,
        candidate=baseline_text,
        objective=expected_objective,
    )
    if private_required:
        expected_rows, selected_quality = _candidate_inventory(
            disagreement_graph=disagreement_graph,
            diagnostic_selection=diagnostic_selection,
            local_repair=local_repair,
            private_evidence=private,
            selected_branch=expected_selected_branch,
            baseline_quality=baseline_quality,
            margin=margin,
        )
    else:
        expected_rows, selected_quality = [], _unavailable_quality()
    intended, expected_reason, selected_request_id = _intended_decision(
        expected_rows,
        enabled=expected_enabled,
        selected_branch_quality=selected_quality,
        baseline_quality=baseline_quality,
    )
    policy = {
        "enabled": expected_enabled,
        "objective_program_enabled": expected_objective_program_enabled,
        "margin": margin,
        "max_output_tokens": output_limit,
        "interval_object": "conjunctive_full_span_exact_claim_validity",
        "replacement_rule": "new_lower_gt_final_decode_upper_plus_margin",
        "syntax_only_verifier_policy": "refutation_only",
        "unknown_claim_policy": "interval_zero_to_one_no_authority",
        "objective_completion_gate": "parent_service_output_quality",
    }
    baseline = value["baseline_decode"]
    binding = value["accepted_output"]
    if (
        value["schema"] != ANSWER_REPLACEMENT_SCHEMA
        or value["disagreement_graph_sha256"] != disagreement_graph.get("receipt_sha256")
        or value["diagnostic_selection_sha256"] != diagnostic_selection.get("receipt_sha256")
        or value["local_repair_sha256"] != local_repair.get("receipt_sha256")
        or value["selected_branch"] != expected_selected_branch
        or value["policy"] != policy
        or value["baseline_decomposition"] != baseline_decomposition
        or value["baseline_routes"] != baseline_routes
        or value["baseline_quality"] != baseline_quality
        or value["selected_branch_quality"] != selected_quality
        or value["candidates"] != expected_rows
        or value["intended_decision"] != intended
        or value["selected_request_id"] != selected_request_id
        or baseline
        != {
            "text_sha256": _text_sha(baseline_text),
            "token_count": len(baseline_tokens),
            "tokens_sha256": _token_sha(baseline_tokens),
        }
        or not isinstance(binding, Mapping)
        or set(binding)
        != {
            "source",
            "text_sha256",
            "token_count",
            "tokens_sha256",
            "binding_status",
        }
        or value["latent_state_effect"] != "none"
        or value["authority"] != "confidence_bound_answer_replacement"
    ):
        raise ValueError("answer replacement reconstruction differs")
    decision = value["decision"]
    if decision == "replace":
        # Every promoted source is validated by exact text/token round-trip
        # against private evidence reconstructed in this trust domain.
        is_objective_solution = selected_request_id == "objective-program"
        is_branch = selected_request_id.startswith("branch-")
        candidate = (
            private["objective_program_solution"]
            if is_objective_solution
            else private["branch_candidates"].get(
                selected_request_id.removeprefix("branch-")
            )
            if is_branch
            else private["generated_repairs"].get(selected_request_id)
        )
        expected_effect = "replaced"
        if (
            intended != "replace"
            or value["reason"] != expected_reason
            or not isinstance(candidate, str)
            or binding["source"]
            != (
                "objective_program_solution"
                if is_objective_solution
                else "branch_candidate"
                if is_branch
                else "repaired_candidate"
            )
            or binding["binding_status"] != "exact_text_token_roundtrip"
            or binding["text_sha256"] != _text_sha(candidate)
            or not 0 < binding["token_count"] <= output_limit
            or _SHA256_RE.fullmatch(str(binding["tokens_sha256"])) is None
        ):
            raise ValueError("answer replacement authority is invalid")
    elif decision == "retain":
        expected_effect = "retained"
        if (
            intended != "retain"
            or value["reason"] != expected_reason
            or binding["source"] != "baseline_decode"
            or binding["binding_status"] != "not_required"
            or binding["text_sha256"] != _text_sha(baseline_text)
            or _SHA256_RE.fullmatch(str(binding["tokens_sha256"])) is None
        ):
            raise ValueError("answer retention binding is invalid")
    elif decision == "abstain":
        expected_effect = "abstained"
        binding_failure = (
            intended == "replace"
            and value["reason"]
            in {
                "dominant_repair_output_binding_failed",
                "dominant_branch_output_binding_failed",
                "objective_program_output_binding_failed",
            }
            and binding["binding_status"] == "failed_closed"
        )
        if (
            not ((intended == "abstain" and value["reason"] == expected_reason) or binding_failure)
            or binding["source"] != "none"
            or binding["text_sha256"] != ""
            or binding["token_count"] != 0
            or binding["tokens_sha256"] != ""
            or binding["binding_status"] not in {"not_required", "failed_closed"}
        ):
            raise ValueError("answer abstention binding is invalid")
    else:
        raise ValueError("answer replacement decision is invalid")
    if (
        value["answer_selection_effect"] != expected_effect
        or (
            expected_output_text is not None
            and binding["text_sha256"]
            != (_text_sha(expected_output_text) if decision != "abstain" else "")
        )
        or (
            expected_output_tokens is not None
            and (
                binding["token_count"] != len(expected_output_tokens)
                or binding["tokens_sha256"]
                != (_token_sha(expected_output_tokens) if decision != "abstain" else "")
            )
        )
    ):
        raise ValueError("answer replacement output binding differs")
    return dict(value)


def validate_pre_adaptation_incumbent(
    value: Any,
    *,
    private_evidence: Any,
    expected_objective: str,
) -> tuple[str, list[int], dict[str, Any]]:
    """Reconstruct only the ordinary incumbent bound before adaptation.

    This authority is deliberately independent of replacement promotion. It
    exists for the case where the worker produced and committed an ordinary
    answer, then a later optional replacement check failed. The failed
    recurrent candidate stays failed; this function can authorize only the
    exact pre-adaptation text and tokens already committed by the receipt.
    """

    public = _validate_public_receipt_commitment(value)
    if not isinstance(expected_objective, str):
        raise ValueError("host incumbent objective is invalid")
    if public.get("private_evidence_required") is not True:
        raise ValueError("host incumbent private evidence is unavailable")
    private = _normalize_private_evidence(private_evidence)
    if (
        private["objective"] != expected_objective
        or public.get("private_evidence_sha256") != _sha(private)
    ):
        raise ValueError("host incumbent private evidence binding differs")

    baseline_text = private["baseline_text"]
    baseline_tokens = private["baseline_tokens"]
    if not baseline_text.strip() or not baseline_tokens:
        raise ValueError("host incumbent is empty")
    decomposition = build_atomic_decomposition(
        baseline_text,
        objective=expected_objective,
    )
    routes = build_deterministic_router_receipt(
        baseline_text,
        objective=expected_objective,
        atomic_receipt=decomposition,
    )
    quality = _quality_interval(
        decomposition,
        routes,
        candidate=baseline_text,
        objective=expected_objective,
    )
    expected_binding = {
        "text_sha256": _text_sha(baseline_text),
        "token_count": len(baseline_tokens),
        "tokens_sha256": _token_sha(baseline_tokens),
    }
    if (
        public.get("baseline_decomposition") != decomposition
        or public.get("baseline_routes") != routes
        or public.get("baseline_quality") != quality
        or public.get("baseline_decode") != expected_binding
    ):
        raise ValueError("host incumbent reconstruction differs")

    payload = {
        "schema": HOST_INCUMBENT_DISPOSITION_SCHEMA,
        "authority": "host_reconstructed_pre_adaptation_incumbent",
        "source_answer_replacement_sha256": public["receipt_sha256"],
        "private_evidence_sha256": public["private_evidence_sha256"],
        "objective_sha256": _text_sha(expected_objective),
        "text_sha256": expected_binding["text_sha256"],
        "token_count": expected_binding["token_count"],
        "tokens_sha256": expected_binding["tokens_sha256"],
        "baseline_decomposition_sha256": decomposition["receipt_sha256"],
        "baseline_routes_sha256": routes["receipt_sha256"],
        "baseline_quality_sha256": quality["interval_sha256"],
    }
    return baseline_text, list(baseline_tokens), {
        **payload,
        "receipt_sha256": _sha(payload),
    }


def validate_host_incumbent_disposition(
    value: Any,
    *,
    answer_replacement_receipt: Any,
    expected_text: str,
    expected_tokens: Sequence[int],
) -> dict[str, Any]:
    """Validate the host's exact baseline disposition at the serving boundary."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _HOST_INCUMBENT_DISPOSITION_FIELDS
    ):
        raise ValueError("host incumbent disposition fields differ")
    payload = {
        key: value[key]
        for key in _HOST_INCUMBENT_DISPOSITION_FIELDS - {"receipt_sha256"}
    }
    public = _validate_public_receipt_commitment(answer_replacement_receipt)
    tokens = list(expected_tokens)
    if (
        value["schema"] != HOST_INCUMBENT_DISPOSITION_SCHEMA
        or value["authority"]
        != "host_reconstructed_pre_adaptation_incumbent"
        or value["receipt_sha256"] != _sha(payload)
        or value["source_answer_replacement_sha256"]
        != public["receipt_sha256"]
        or value["private_evidence_sha256"]
        != public["private_evidence_sha256"]
        or any(
            _SHA256_RE.fullmatch(str(value[field])) is None
            for field in (
                "objective_sha256",
                "baseline_decomposition_sha256",
                "baseline_routes_sha256",
                "baseline_quality_sha256",
            )
        )
        or value["text_sha256"] != _text_sha(expected_text)
        or value["token_count"] != len(tokens)
        or value["tokens_sha256"] != _token_sha(tokens)
    ):
        raise ValueError("host incumbent disposition binding differs")
    return dict(value)


__all__ = [
    "ANSWER_REPLACEMENT_PRIVATE_SCHEMA",
    "ANSWER_REPLACEMENT_SCHEMA",
    "HOST_INCUMBENT_DISPOSITION_SCHEMA",
    "DEFAULT_REPLACEMENT_MARGIN",
    "MAX_BASELINE_EVIDENCE_TOKENS",
    "MAX_REPLACEMENT_OUTPUT_TOKENS",
    "build_answer_replacement_receipt",
    "validate_answer_replacement_receipt",
    "validate_host_incumbent_disposition",
    "validate_pre_adaptation_incumbent",
]
