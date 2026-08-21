from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm.latent_cortex.answer_replacement import (
    MAX_BASELINE_EVIDENCE_TOKENS,
    MAX_REPLACEMENT_OUTPUT_TOKENS,
    build_answer_replacement_receipt,
    validate_answer_replacement_receipt,
    validate_host_incumbent_disposition,
    validate_pre_adaptation_incumbent,
)
from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
)
from core.brain.llm.latent_cortex.diagnostic_action_selector import (
    build_candidate_routes,
    build_diagnostic_action_selector_receipt,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.local_repair import (
    build_local_repair_receipt,
    prepare_local_repair_requests,
)
from core.brain.llm.latent_cortex.value_of_computation import (
    build_evidence_snapshot,
)
from tests.sealed_artifact_support import require_mathematics_memory_tissue


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scenario(
    *,
    left: str,
    right: str,
    repaired: str,
    objective: str = "Return the exactly correct arithmetic answer.",
) -> tuple[str, dict[int, str], dict, dict, dict, dict[str, dict]]:
    candidates = {0: left, 1: right}
    decompositions = {
        str(index): build_atomic_decomposition(text, objective=objective)
        for index, text in candidates.items()
    }
    graph_payload = {
        "n_branches": 2,
        "candidate_decompositions": decompositions,
        "branches": [
            {
                "index": index,
                "operator_transition_count": 1,
                "operator_program_sha256": _digest(f"program-{index}"),
                "candidate_decomposition_sha256": decompositions[str(index)]["receipt_sha256"],
            }
            for index in range(2)
        ],
        "pairwise": [
            {
                "left": 0,
                "right": 1,
                "localized": True,
                "causal_divergence": {
                    "available": True,
                    "kind": "causal_transition",
                    "action_step": 1,
                },
                "candidate_divergence": {
                    "available": True,
                    "kind": "atomic_claim",
                    "atom_ordinal": 0,
                    "left": {
                        "atom_id": "a000",
                        "text_sha256": decompositions["0"]["atoms"][0]["text_sha256"],
                    },
                    "right": {
                        "atom_id": "a000",
                        "text_sha256": decompositions["1"]["atoms"][0]["text_sha256"],
                    },
                },
            }
        ],
    }
    graph = {**graph_payload, "receipt_sha256": _digest("answer-graph")}
    routes = build_candidate_routes(
        candidates,
        objective=objective,
        candidate_decompositions=decompositions,
    )
    snapshot = build_evidence_snapshot(bucket="answer-replacement", cells={})
    selector = build_diagnostic_action_selector_receipt(
        disagreement_graph=graph,
        candidate_routes=routes,
        action_policy_evidence=snapshot,
        value_policy={
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "executors": [
                OperationKind.CHECK_ASSUMPTION.value,
                OperationKind.REGENERATE_FROM_PREFIX.value,
            ],
        },
        action_trace=[
            {
                "state_signal": {
                    "has_memory": False,
                    "has_evidence": False,
                    "has_verifier": True,
                    "has_savepoint": True,
                }
            }
        ],
    )
    requests = prepare_local_repair_requests(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
    )
    generated = (
        {
            requests[0]["request_id"]: {
                "candidate": repaired,
                "generation_context": {
                    "prompt_sha256": requests[0]["prompt_sha256"],
                    "generated_token_count": 16,
                    "termination": "eos",
                    "initial_cache_offsets": [0, 0],
                    "final_cache_offsets": [16, 16],
                    "all_initial_offsets_zero": True,
                    "solver_context_imported": False,
                    "parameter_relation": "shared_resident_checkpoint",
                },
            }
        }
        if requests
        else {}
    )
    local_repair = build_local_repair_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
        generated_repairs=generated,
    )
    return objective, candidates, graph, selector, local_repair, generated


def _encode(value: str) -> list[int]:
    return list(value.encode("utf-8"))


def _decode(tokens) -> str:
    return bytes(tokens).decode("utf-8")


def _build(
    *,
    left: str = "2 + 2 = 5.",
    right: str = "2 + 2 = 4.",
    repaired: str = "2 + 2 = 4.",
    selected_branch: int = 0,
    enabled: bool = True,
    baseline_text: str | None = None,
):
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left=left,
        right=right,
        repaired=repaired,
    )
    baseline_text = left if baseline_text is None else baseline_text
    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=selected_branch,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=baseline_text,
        baseline_tokens=_encode(baseline_text),
        encode=_encode,
        decode=_decode,
        enabled=enabled,
        margin=0.05,
        max_output_tokens=64,
    )
    return receipt, tokens, graph, selector, local_repair, private, objective


def test_pre_adaptation_incumbent_survives_independent_replacement_policy_failure():
    (
        receipt,
        tokens,
        graph,
        selector,
        local_repair,
        private,
        objective,
    ) = _build()

    with pytest.raises(ValueError, match="reconstruction differs"):
        validate_answer_replacement_receipt(
            receipt,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            private_evidence=private,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=63,
            expected_output_text=_decode(tokens),
            expected_output_tokens=tokens,
        )

    text, baseline_tokens, disposition = validate_pre_adaptation_incumbent(
        receipt,
        private_evidence=private,
        expected_objective=objective,
    )

    assert text == private["baseline_text"]
    assert baseline_tokens == private["baseline_tokens"]
    validate_host_incumbent_disposition(
        disposition,
        answer_replacement_receipt=receipt,
        expected_text=text,
        expected_tokens=baseline_tokens,
    )


def test_pre_adaptation_incumbent_rejects_private_or_serving_tampering():
    receipt, _tokens, _graph, _selector, _repair, private, objective = _build()
    tampered_private = copy.deepcopy(private)
    tampered_private["baseline_text"] += " altered"

    with pytest.raises(ValueError, match="private evidence binding differs"):
        validate_pre_adaptation_incumbent(
            receipt,
            private_evidence=tampered_private,
            expected_objective=objective,
        )

    text, baseline_tokens, disposition = validate_pre_adaptation_incumbent(
        receipt,
        private_evidence=private,
        expected_objective=objective,
    )
    with pytest.raises(ValueError, match="disposition binding differs"):
        validate_host_incumbent_disposition(
            disposition,
            answer_replacement_receipt=receipt,
            expected_text=text + " altered",
            expected_tokens=baseline_tokens,
        )
    with pytest.raises(ValueError, match="disposition binding differs"):
        validate_host_incumbent_disposition(
            disposition,
            answer_replacement_receipt=receipt,
            expected_text=text,
            expected_tokens=[*baseline_tokens, 1],
        )


def test_complete_exact_repair_replaces_only_after_nonoverlap_margin():
    (
        receipt,
        tokens,
        graph,
        selector,
        local_repair,
        private,
        objective,
    ) = _build()

    assert receipt["intended_decision"] == "replace"
    assert receipt["decision"] == "replace"
    assert receipt["answer_selection_effect"] == "replaced"
    candidate = receipt["candidates"][0]
    assert candidate["source_branch_quality"]["lower_bound"] == 0.0
    assert candidate["source_branch_quality"]["upper_bound"] == 0.0
    assert candidate["replacement_quality"]["lower_bound"] == 1.0
    assert candidate["replacement_quality"]["upper_bound"] == 1.0
    assert candidate["same_verifier_class"] is True
    assert candidate["dominates"] is True
    assert _decode(tokens) == "2 + 2 = 4."
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence=private,
        expected_objective=objective,
        expected_selected_branch=0,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=64,
        expected_output_text="2 + 2 = 4.",
        expected_output_tokens=tokens,
    )


def test_unknown_claim_keeps_repair_interval_open_and_forces_abstention():
    receipt, tokens, *_ = _build(
        left="2 + 2 = 5. This is the answer.",
        right="2 + 2 = 4. This is the answer.",
        repaired="2 + 2 = 4. This is the answer.",
    )

    assert receipt["candidates"][0]["replacement_quality"]["basis"] == (
        "incomplete_semantic_exact_coverage"
    )
    assert receipt["candidates"][0]["replacement_quality"]["upper_bound"] == 1.0
    assert receipt["decision"] == "abstain"
    assert receipt["answer_selection_effect"] == "abstained"
    assert tokens == []


def test_refutation_on_nonselected_branch_does_not_replace_selected_answer():
    receipt, tokens, *_ = _build(
        selected_branch=1,
        baseline_text="2 + 2 = 4.",
    )

    assert receipt["decision"] == "retain"
    assert receipt["reason"] == "final_decode_already_exactly_verified"
    assert _decode(tokens) == "2 + 2 = 4."


def test_explicit_disable_retains_baseline_without_borrowing_authority():
    receipt, tokens, *_ = _build(enabled=False)

    assert receipt["intended_decision"] == "retain"
    assert receipt["decision"] == "retain"
    assert receipt["reason"] == "answer_replacement_disabled"
    assert _decode(tokens) == "2 + 2 = 5."


def test_no_repair_candidate_inventory_is_replayed_before_retaining_baseline():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 4.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )
    assert local_repair["requests"] == []
    baseline = "2 + 2 = 4."
    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=baseline,
        baseline_tokens=_encode(baseline),
        encode=_encode,
        decode=_decode,
        enabled=True,
        margin=0.05,
        max_output_tokens=64,
    )

    assert private["branch_candidates"] == {"0": baseline, "1": baseline}
    assert receipt["private_evidence_required"] is True
    assert len(receipt["candidates"]) == 2
    assert receipt["decision"] == "retain"
    assert receipt["reason"] == "final_decode_already_exactly_verified"
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence=private,
        expected_objective=objective,
        expected_selected_branch=0,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=64,
        expected_output_text=baseline,
        expected_output_tokens=tokens,
    )
    tampered = copy.deepcopy(private)
    tampered["branch_candidates"]["0"] = "2 + 2 = 9."
    with pytest.raises(ValueError, match="binding|reconstruction"):
        validate_answer_replacement_receipt(
            receipt,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            private_evidence=tampered,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text=baseline,
            expected_output_tokens=tokens,
        )


def test_public_objective_program_promotes_correct_branch_over_wrong_incumbent():
    objective = (
        "Start at the given value and apply each operation modulo 19: start=17. "
        "Operations: -11, *12. You may reason before answering. Finish with exactly "
        "one final line using the envelope FINAL_ANSWER: <JSON object>."
    )
    correct = 'FINAL_ANSWER: {"residue":15}'
    wrong = 'FINAL_ANSWER: {"residue":14}'
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left=correct,
        right=correct,
        repaired=correct,
        objective=objective,
    )
    assert local_repair["requests"] == []

    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=wrong,
        baseline_tokens=_encode(wrong),
        encode=_encode,
        decode=_decode,
        enabled=True,
        margin=0.05,
        max_output_tokens=64,
    )

    assert receipt["baseline_quality"]["basis"] == "deterministic_exact_refutation"
    assert receipt["candidates"][0]["replacement_quality"]["basis"] == (
        "objective_program_exact_complete"
    )
    assert receipt["candidates"][0]["dominates"] is True
    assert receipt["decision"] == "replace"
    assert receipt["accepted_output"]["source"] == "branch_candidate"
    assert _decode(tokens) == correct
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence=private,
        expected_objective=objective,
        expected_selected_branch=0,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=64,
        expected_output_text=correct,
        expected_output_tokens=tokens,
    )


def test_public_objective_solver_recovers_when_incumbent_and_branches_are_wrong():
    objective = (
        "Start at the given value and apply each operation modulo 19: start=12. "
        "Operations: *18, *12. You may reason before answering. Finish with exactly "
        "one final line using the envelope FINAL_ANSWER: <JSON object>."
    )
    wrong = 'FINAL_ANSWER: {"residue":7}'
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left=wrong,
        right=wrong,
        repaired=wrong,
        objective=objective,
    )
    assert len(local_repair["requests"]) == 1
    assert local_repair["transactions"][0]["status"] == "repaired_candidate_rejected"

    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=wrong,
        baseline_tokens=_encode(wrong),
        encode=_encode,
        decode=_decode,
        enabled=True,
        margin=0.05,
        max_output_tokens=256,
    )

    assert receipt["decision"] == "replace"
    assert receipt["selected_request_id"] == "objective-program"
    assert receipt["accepted_output"]["source"] == "objective_program_solution"
    assert _decode(tokens).endswith('FINAL_ANSWER: {"residue":8}')
    assert "Step 1:" in _decode(tokens)
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence=private,
        expected_objective=objective,
        expected_selected_branch=0,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=256,
        expected_output_text=_decode(tokens),
        expected_output_tokens=tokens,
    )


def test_certified_recurrent_program_replaces_wrong_complete_engine_candidates():
    from core.learning.recurrence_curriculum import modular_chain

    task = modular_chain(8, 90_193)
    wrong = 'FINAL_ANSWER: {"residue":999}'
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left=wrong,
        right=wrong,
        repaired=wrong,
        objective=task.prompt,
    )

    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=wrong,
        baseline_tokens=_encode(wrong),
        encode=_encode,
        decode=_decode,
        enabled=True,
        margin=0.05,
        max_output_tokens=512,
    )

    producer = private["objective_program_solution_receipt"]
    assert producer["execution"]["engine"] == "systematic_neural_alu.v1"
    assert producer["execution"]["teacher_available"] is False
    assert producer["execution"]["student_rollin"]["transition_count"] == 8
    assert receipt["decision"] == "replace"
    assert receipt["selected_request_id"] == "objective-program"
    assert receipt["accepted_output"]["source"] == "objective_program_solution"
    assert _decode(tokens).endswith(task.answer)
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence=private,
        expected_objective=objective,
        expected_selected_branch=0,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=512,
        expected_output_text=_decode(tokens),
        expected_output_tokens=tokens,
    )


def test_sealed_recurrent_memory_replaces_a_wrong_mathematics_decode():
    require_mathematics_memory_tissue()
    from core.brain.llm.latent_cortex.frontier_tasks import generate_task

    task = generate_task("mathematics", seed=1_037, difficulty=3)
    wrong = 'FINAL_ANSWER: {"count":0,"witness":[]}'
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left=wrong,
        right=wrong,
        repaired=wrong,
        objective=task.public.prompt,
    )

    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=wrong,
        baseline_tokens=_encode(wrong),
        encode=_encode,
        decode=_decode,
        enabled=True,
        margin=0.05,
        max_output_tokens=512,
    )

    producer = private["objective_program_solution_receipt"]
    execution = producer["execution"]
    assert execution["engine"] == "mathematics_memory_tissue.v1"
    assert execution["student_rollin"]["teacher_available"] is False
    assert execution["student_rollin"]["verifier_available"] is False
    assert receipt["baseline_quality"]["basis"] == "deterministic_exact_refutation"
    assert receipt["decision"] == "replace"
    assert receipt["accepted_output"]["source"] == "objective_program_solution"
    assert task.score(_decode(tokens)).correct is True
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence=private,
        expected_objective=objective,
        expected_selected_branch=0,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=512,
        expected_output_text=_decode(tokens),
        expected_output_tokens=tokens,
    )


def test_public_objective_solver_replaces_wrong_fenced_json_incumbent():
    objective = (
        "Evaluate this 2-operation expression with 1=true, 0=false, and xor meaning "
        "exactly one operand is true: ((1 and 1) or 0). Return a value of 1 or 0. "
        "You may reason before answering. Finish with exactly one final line using "
        "the envelope FINAL_ANSWER: <JSON object>."
    )
    wrong = '```json\n{\n  "value": 0\n}\n```'
    correct = (
        "Evaluate ((1 and 1) or 0) using not, and, xor, then or precedence.\n"
        "The bounded parser executed 2 operations and the expression is true, encoded as 1.\n"
        'FINAL_ANSWER: {"value":1}'
    )
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left=wrong,
        right=wrong,
        repaired=wrong,
        objective=objective,
    )

    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=wrong,
        baseline_tokens=_encode(wrong),
        encode=_encode,
        decode=_decode,
        enabled=True,
        margin=0.05,
        max_output_tokens=256,
    )

    assert receipt["baseline_quality"]["basis"] == "deterministic_exact_refutation"
    assert receipt["decision"] == "replace"
    assert receipt["selected_request_id"] == "objective-program"
    assert receipt["accepted_output"]["source"] == "objective_program_solution"
    assert _decode(tokens) == correct
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence=private,
        expected_objective=objective,
        expected_selected_branch=0,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=256,
        expected_output_text=correct,
        expected_output_tokens=tokens,
    )


def test_no_repair_budget_never_returns_a_deterministically_refuted_decode():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 4.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )
    assert local_repair["requests"] == []
    baseline = "2 + 2 = 5."

    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=baseline,
        baseline_tokens=_encode(baseline),
        encode=_encode,
        decode=_decode,
        enabled=True,
        margin=0.05,
        max_output_tokens=64,
    )

    assert private["baseline_text"] == baseline
    assert receipt["private_evidence_required"] is True
    assert receipt["baseline_quality"]["basis"] == "deterministic_exact_refutation"
    # The safety property in this test's name -- a deterministically refuted
    # decode is never returned -- still holds, and is now satisfied by a
    # STRONGER outcome than abstention. Branch candidates became promotable
    # under the same lower-bound-dominance rule repairs use, so an
    # arithmetically verified branch ("2 + 2 = 4.", every atom verified, lower
    # bound 1.0) displaces a refuted baseline (bounds [0, 0]) instead of the
    # system giving up. Abstaining was previously the only option because the
    # recurrent path had no route to the output at all.
    assert receipt["decision"] == "replace"
    assert receipt["reason"] == (
        "replacement_lower_bound_exceeds_final_decode_upper_bound_plus_margin"
    )
    assert receipt["accepted_output"]["source"] == "branch_candidate"
    # The refuted baseline is not what gets served -- the point of the test.
    assert tokens
    assert _decode(tokens) != baseline


def test_output_text_tamper_is_rejected_by_service_reconstruction():
    receipt, tokens, graph, selector, local_repair, private, objective = _build()

    with pytest.raises(ValueError, match="output binding"):
        validate_answer_replacement_receipt(
            receipt,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            private_evidence=private,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text="2 + 2 = 9.",
            expected_output_tokens=tokens,
        )


def test_policy_margin_tamper_cannot_create_replacement_authority():
    receipt, tokens, graph, selector, local_repair, private, objective = _build()
    tampered = copy.deepcopy(receipt)
    tampered["policy"]["margin"] = 0.0

    with pytest.raises(ValueError, match="commitment"):
        validate_answer_replacement_receipt(
            tampered,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            private_evidence=private,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text="2 + 2 = 4.",
            expected_output_tokens=tokens,
        )


def test_stale_local_repair_commitment_is_rejected():
    receipt, tokens, graph, selector, local_repair, private, objective = _build()
    stale = copy.deepcopy(local_repair)
    stale["receipt_sha256"] = "0" * 64

    with pytest.raises(ValueError):
        validate_answer_replacement_receipt(
            receipt,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=stale,
            private_evidence=private,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text="2 + 2 = 4.",
            expected_output_tokens=tokens,
        )


def test_failed_tokenizer_roundtrip_abstains_instead_of_silent_retain():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )
    receipt, tokens, _private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text="2 + 2 = 5.",
        baseline_tokens=[1],
        encode=lambda _value: [1],
        decode=lambda _tokens: "different text",
        max_output_tokens=64,
    )

    assert receipt["intended_decision"] == "replace"
    assert receipt["decision"] == "abstain"
    assert receipt["reason"] == "dominant_repair_output_binding_failed"
    assert tokens == []


def test_tokenizer_expansion_beyond_output_ceiling_fails_closed():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )
    oversized = list(range(65))
    receipt, tokens, _private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text="2 + 2 = 5.",
        baseline_tokens=[1],
        encode=lambda _value: oversized,
        decode=lambda _tokens: "2 + 2 = 4.",
        max_output_tokens=64,
    )

    assert receipt["decision"] == "abstain"
    assert receipt["accepted_output"]["binding_status"] == "failed_closed"
    assert tokens == []


def test_true_arithmetic_with_false_prose_cannot_receive_certain_interval():
    receipt, tokens, *_ = _build(
        left="2 + 2 = 5 and Earth is flat.",
        right="2 + 2 = 4 and Earth is flat.",
        repaired="2 + 2 = 4 and Earth is flat.",
    )

    quality = receipt["candidates"][0]["replacement_quality"]
    assert quality["basis"] == "incomplete_semantic_exact_coverage"
    assert quality["lower_bound"] == 0.0
    assert quality["upper_bound"] == 1.0
    assert receipt["decision"] == "abstain"
    assert tokens == []


def test_python_parse_success_is_syntax_evidence_not_semantic_certainty():
    receipt, tokens, *_ = _build(
        left="```python\nif True print('bad')\n```",
        right="```python\nprint('valid')\n```",
        repaired="```python\nprint('valid')\n```",
    )

    quality = receipt["candidates"][0]["replacement_quality"]
    assert quality["basis"] == "incomplete_semantic_exact_coverage"
    assert quality["semantic_exact_verified_count"] == 0
    assert receipt["decision"] == "abstain"
    assert tokens == []


def test_actual_final_decode_is_the_comparator_not_short_branch_probe():
    receipt, tokens, *_ = _build(
        baseline_text="2 + 2 = 4.",
    )

    assert receipt["selected_branch_quality"]["basis"] == ("deterministic_exact_refutation")
    assert receipt["baseline_quality"]["basis"] == ("full_span_semantic_exact_complete")
    assert receipt["decision"] == "retain"
    assert receipt["reason"] == "final_decode_already_exactly_verified"
    assert _decode(tokens) == "2 + 2 = 4."


def test_exact_objective_candidate_displaces_contract_incomplete_incumbent():
    objective = (
        "Start at the given value and apply each operation modulo 13: "
        "start=1. Operations: *6, +12. Return the final residue."
    )
    correct = 'FINAL_ANSWER: {"residue":5}'
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left='FINAL_ANSWER: {"residue":12}',
        right=correct,
        repaired=correct,
        objective=objective,
    )

    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=1,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text="Let's calculate each operation before giving the final",
        baseline_tokens=_encode("Let's calculate each operation before giving the final"),
        encode=_encode,
        decode=_decode,
        margin=0.05,
        max_output_tokens=128,
    )

    assert receipt["schema"] == "aura.rlc.answer_replacement.v5"
    assert receipt["baseline_quality"]["basis"] == (
        "objective_program_contract_incomplete"
    )
    assert receipt["baseline_quality"]["upper_bound"] == 0.0
    assert receipt["decision"] == "replace"
    assert receipt["accepted_output"]["source"] in {
        "branch_candidate",
        "objective_program_solution",
    }
    assert _decode(tokens).endswith('FINAL_ANSWER: {"residue":5}')
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence=private,
        expected_objective=objective,
        expected_selected_branch=1,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=128,
        expected_output_text=_decode(tokens),
        expected_output_tokens=tokens,
    )


def test_refuted_selected_branch_abstains_when_request_budget_omits_it():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="3 + 3 = 7.",
        repaired="2 + 2 = 4.",
    )
    assert local_repair["request_count"] == 1
    assert local_repair["requests"][0]["branch"] == 0
    receipt, tokens, _private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=1,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text="ordinary unverified decode",
        baseline_tokens=[1],
        encode=_encode,
        decode=_decode,
        max_output_tokens=64,
    )

    assert receipt["selected_branch_quality"]["basis"] == ("deterministic_exact_refutation")
    assert receipt["decision"] == "abstain"
    assert tokens == []


def test_service_reexecution_rejects_tampered_private_baseline():
    receipt, tokens, graph, selector, local_repair, private, objective = _build()
    tampered_private = copy.deepcopy(private)
    tampered_private["baseline_text"] = "2 + 2 = 4."

    with pytest.raises(ValueError, match="private evidence"):
        validate_answer_replacement_receipt(
            receipt,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            private_evidence=tampered_private,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text="2 + 2 = 4.",
            expected_output_tokens=tokens,
        )


def test_service_reexecution_rejects_tampered_private_baseline_tokens():
    receipt, tokens, graph, selector, local_repair, private, objective = _build()
    tampered_private = copy.deepcopy(private)
    tampered_private["baseline_tokens"] = [9, 9, 9]

    with pytest.raises(ValueError, match="private evidence"):
        validate_answer_replacement_receipt(
            receipt,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            private_evidence=tampered_private,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text="2 + 2 = 4.",
            expected_output_tokens=tokens,
        )


def test_private_baseline_evidence_accepts_engine_decode_beyond_replacement_limit():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )
    baseline_tokens = [1] * (MAX_REPLACEMENT_OUTPUT_TOKENS + 1)

    receipt, accepted_tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text="ordinary unverified decode",
        baseline_tokens=baseline_tokens,
        encode=_encode,
        decode=_decode,
        max_output_tokens=64,
    )

    assert len(private["baseline_tokens"]) == MAX_REPLACEMENT_OUTPUT_TOKENS + 1
    assert receipt["baseline_decode"]["token_count"] == len(baseline_tokens)
    assert receipt["decision"] == "abstain"
    assert accepted_tokens == []
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence=private,
        expected_objective=objective,
        expected_selected_branch=0,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=64,
        expected_output_text="",
        expected_output_tokens=accepted_tokens,
    )


def test_private_baseline_evidence_rejects_tokens_beyond_engine_decode_envelope():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )

    with pytest.raises(ValueError, match="baseline token limit exceeded"):
        build_answer_replacement_receipt(
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            selected_branch=0,
            branch_candidates=candidates,
            generated_repairs=generated,
            objective=objective,
            baseline_text="ordinary unverified decode",
            baseline_tokens=[1] * (MAX_BASELINE_EVIDENCE_TOKENS + 1),
            encode=_encode,
            decode=_decode,
            max_output_tokens=64,
        )


def test_rejected_generated_repair_has_no_private_authority_or_fallback():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 5.",
    )
    assert local_repair["transactions"][0]["status"] == ("repaired_candidate_rejected")
    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text="ordinary unverified decode",
        baseline_tokens=[1],
        encode=_encode,
        decode=_decode,
        max_output_tokens=64,
    )

    assert private["generated_repairs"] == {}
    assert receipt["decision"] == "abstain"
    assert tokens == []
