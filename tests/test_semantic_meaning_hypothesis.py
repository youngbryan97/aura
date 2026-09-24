"""A source occurrence is not a numerical value or a program vote."""

from dataclasses import replace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_candidate_bank import SemanticCandidate, SemanticCandidateBank
from core.learning.semantic_meaning_hypothesis import (
    GroundedProgramProposal,
    meaning_hypotheses_from_bank,
)
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import SemanticTransductionOutcome


def _bank() -> SemanticCandidateBank:
    source = "a" * 64
    inputs = (TokenSpan(0, 1), TokenSpan(2, 3))
    candidate = SemanticCandidate(
        Program(2, (Instruction("sub", (0, 1)),)), 0.7,
        (TokenSpan(4, 5),), 0, 0,
        ((TokenSpan(1, 2), TokenSpan(3, 4)),),
        ((inputs[0], inputs[1]),), "optimizer_selected",
    )
    selected = SemanticTransductionOutcome(None, "test", {}, {})
    body = {"schema": "aura.semantic_candidate_bank.v3", "source_text_sha256": source,
            "candidates": [candidate.to_dict()],
            "input_spans": [span.to_dict() for span in inputs],
            "selected_program_sha256": None, "selected_refusal": "test",
            "selected_search_interrupted": False, "search_complete": False}
    return SemanticCandidateBank(selected, (candidate,), inputs,
                                 {**body, "receipt_sha256": _sha(body)})


def test_meaning_hypothesis_preserves_source_occurrences_and_compiles():
    bank = _bank()
    hypothesis, = bank.meaning_hypotheses()
    assert hypothesis.input_occurrences[0].identity != hypothesis.input_occurrences[1].identity
    assert hypothesis.operations[0].bindings[0].mention.identity != (
        hypothesis.operations[0].bindings[1].mention.identity)
    assert hypothesis.to_program() == bank.candidates[0].program
    grounded = GroundedProgramProposal(hypothesis, hypothesis.to_program())
    assert len(grounded.receipt_sha256) == 64


def test_role_intervention_changes_program_and_receipt():
    hypothesis, = meaning_hypotheses_from_bank(_bank())
    operation, = hypothesis.operations
    left, right = operation.bindings
    changed = replace(hypothesis, operations=(replace(operation, bindings=(
        replace(left, register=right.register), replace(right, register=left.register))),))
    assert changed.identity != hypothesis.identity
    assert changed.to_program().instructions[0].args == (1, 0)
    assert GroundedProgramProposal(changed, changed.to_program()).receipt_sha256 != (
        GroundedProgramProposal(hypothesis, hypothesis.to_program()).receipt_sha256)
    with pytest.raises(ValueError, match="differs"):
        GroundedProgramProposal(changed, hypothesis.to_program())


def test_candidate_bank_integrity_precedes_meaning_projection():
    bank = _bank()
    tampered = replace(bank, input_spans=(TokenSpan(0, 1), TokenSpan(3, 4)))
    with pytest.raises(ValueError, match="payload differs"):
        meaning_hypotheses_from_bank(tampered)


def test_missing_mention_evidence_does_not_become_a_meaning_hypothesis():
    bank = _bank()
    candidate = replace(bank.candidates[0], argument_spans=None, definition_spans=None,
                        definition_provenance="unavailable")
    body = {key: value for key, value in bank.receipt.items() if key != "receipt_sha256"}
    body["candidates"] = [candidate.to_dict()]
    changed = replace(bank, candidates=(candidate,), receipt={**body, "receipt_sha256": _sha(body)})
    assert meaning_hypotheses_from_bank(changed) == ()


def test_invalid_operation_or_score_cannot_enter_meaning_state():
    hypothesis, = _bank().meaning_hypotheses()
    with pytest.raises(ValueError, match="source or evidence"):
        replace(hypothesis, source_score=float("nan"))
    operation, = hypothesis.operations
    with pytest.raises(ValueError, match="invalid role structure"):
        replace(operation, name="unknown_primitive")
