"""Observation collisions require semantic evidence, not different label hashes."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_observation_feasibility import audit_observation_feasibility
from core.learning.semantic_program_ir import TokenSpan


def example(op="add", *, source="a", basis="basis", split="train", hidden=1.):
    instructions = (Instruction(op, (0, 1)),)
    ir = SimpleNamespace(source_token_ids=(1, 2, 3), source_text_sha256=source,
        model_basis_receipt_sha256=basis, input_spans=(TokenSpan(0, 1), TokenSpan(2, 3)),
        instructions=instructions, to_program=lambda: Program(2, (Instruction(op, (0, 1)),)))
    vector = np.asarray([hidden, 1.], dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return SimpleNamespace(ir=ir, public_inputs=(2, 3), split=split,
                           hidden_states=np.tile(vector, (3, 1)))


def test_source_names_cannot_hide_contradictory_observations():
    result = audit_observation_feasibility([example(), example("sub", source="b")])
    assert result["contradictions"] == 1
    assert result["distinct_observations"] == 1
    assert not result["learnability_proven"]


def test_same_semantics_is_not_a_contradiction():
    result = audit_observation_feasibility([example(), example(source="b")])
    assert result["comparisons"][0]["comparison"]["status"] == "equivalent"
    assert result["contradictions"] == result["unresolved"] == 0


def test_different_hidden_or_basis_is_distinguishable():
    result = audit_observation_feasibility([example(), example("sub", hidden=2.),
                                          example("sub", basis="other")])
    assert result["distinct_observations"] == 3
    assert result["comparisons"] == []


def test_unknown_comparison_is_not_a_pass(monkeypatch):
    monkeypatch.setattr("core.learning.semantic_observation_feasibility.compare_program_meanings",
                        lambda *args: {"status": "unknown"})
    result = audit_observation_feasibility([example(), example("sub")])
    assert result["unresolved"] == 1 and result["contradictions"] == 0


def test_sealed_test_rows_are_excluded():
    with pytest.raises(ValueError, match="sealed"):
        audit_observation_feasibility([example(split="test")])
