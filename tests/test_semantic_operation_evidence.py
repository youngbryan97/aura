"""Operation prototypes use source spans, not validation answer keys."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_operation_evidence import OperationEvidence
from core.learning.semantic_program_ir import TokenSpan


def _example(source, operation, vector, *, split="train"):
    features = np.asarray((vector, (0., 1.)), dtype=np.float32)
    return SimpleNamespace(split=split, hidden_states=features,
                           ir=SimpleNamespace(source_text_sha256=source,
                                              instructions=(SimpleNamespace(
                                                  op=operation,
                                                  operation_span=TokenSpan(0, 1)),)))


def test_source_operation_evidence_selects_unseen_wording_by_token_features():
    examples = (_example("source-add", "add", (1., 0.)),
                _example("source-sub", "sub", (-1., 0.)))
    model = OperationEvidence.fit(examples, source_ids={"source-add", "source-sub"})
    programs = (Program(2, (Instruction("sub", (0, 1)),)),
                Program(2, (Instruction("add", (0, 1)),)))
    spans = ((TokenSpan(0, 1),), (TokenSpan(0, 1),))
    unseen = np.asarray(((1., 0.), (0., 1.)), dtype=np.float32)
    assert model.choose(unseen, programs, spans) == 1
    assert model.support == {"add": 1, "sub": 1}
    assert model.scores(unseen, programs, spans)[1] > model.scores(unseen, programs, spans)[0]


def test_operation_evidence_rejects_validation_training_and_missing_sources():
    validation = _example("held", "add", (1., 0.), split="validation")
    with pytest.raises(ValueError, match="source-only"):
        OperationEvidence.fit((validation,), source_ids={"held"})
    with pytest.raises(ValueError, match="incomplete"):
        OperationEvidence.fit((_example("seen", "add", (1., 0.)),),
                              source_ids={"seen", "missing"})


def test_operation_evidence_rejects_unsupported_op_and_span_geometry():
    model = OperationEvidence.fit((_example("source", "add", (1., 0.)),),
                                  source_ids={"source"})
    add = Program(2, (Instruction("add", (0, 1)),))
    sub = Program(2, (Instruction("sub", (0, 1)),))
    features = np.asarray(((1., 0.), (0., 1.)), dtype=np.float32)
    with pytest.raises(ValueError, match="no source-trained"):
        model.choose(features, (sub,), ((TokenSpan(0, 1),),))
    with pytest.raises(ValueError, match="typed floor"):
        model.choose(features, (add,), ((),))


def test_positioned_operation_evidence_distinguishes_later_step():
    examples = (
        SimpleNamespace(split="train", hidden_states=np.asarray(((1., 0.), (0., 1.))),
                        ir=SimpleNamespace(source_text_sha256="a", instructions=(
                            SimpleNamespace(op="add", operation_span=TokenSpan(0, 1)),
                            SimpleNamespace(op="sub", operation_span=TokenSpan(1, 2))))),
        SimpleNamespace(split="train", hidden_states=np.asarray(((0., 1.), (1., 0.))),
                        ir=SimpleNamespace(source_text_sha256="b", instructions=(
                            SimpleNamespace(op="sub", operation_span=TokenSpan(0, 1)),
                            SimpleNamespace(op="add", operation_span=TokenSpan(1, 2))))),
    )
    model = OperationEvidence.fit(examples, source_ids={"a", "b"})
    correct = Program(2, (Instruction("add", (0, 1)), Instruction("sub", (2, 1))))
    rival = Program(2, (Instruction("add", (0, 1)), Instruction("add", (2, 1))))
    spans = ((TokenSpan(0, 1), TokenSpan(1, 2)),) * 2
    assert model.choose(examples[0].hidden_states, (correct, rival), spans,
                        positioned=True) == 0
