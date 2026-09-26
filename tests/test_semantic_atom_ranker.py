"""Local operation and reference supervision must reach candidate selection."""

import pytest
import torch
from torch.nn import functional as functional

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_atom_ranker import AtomAlignedProgramRanker
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_request_context import RequestContextConfig


def _case():
    torch.manual_seed(91)
    model = AtomAlignedProgramRanker(RequestContextConfig(
        12, width=8, heads=2, layers=1, position_mode="relative",
        feature_scaling="unit_variance"))
    source = functional.normalize(torch.randn(5, 12), dim=-1)
    inputs = (TokenSpan(0, 1), TokenSpan(4, 5))
    programs = (Program(2, (Instruction("sub", (0, 1)),)),
                Program(2, (Instruction("sub", (1, 0)),)),
                Program(2, (Instruction("add", (0, 1)),)))
    anchors = ((TokenSpan(2, 3),),) * 3
    return model, source, inputs, programs, anchors


def test_local_training_and_selection_share_factors_and_gradients():
    model, source, inputs, programs, anchors = _case()
    loss, counts = model.source_atom_loss(source, inputs, ("integer",) * 2,
                                          programs[0], operation_spans=anchors[0])
    scores = model(source, inputs, ("integer",) * 2, programs, operation_spans=anchors)
    # Selection centers each two-way reference against its uniform baseline.
    torch.testing.assert_close(scores[0], -3 * loss + 2 * torch.log(torch.tensor(2.)))
    assert counts["operation_total"] == 1 and counts["reference_total"] == 2
    loss.backward()
    for parameter in (model.operation_classifier.weight, model.query[0].weight,
                      model.key.weight, model.context.project.weight):
        assert parameter.grad.abs().sum() > 0
    assert model.context.restore.weight.grad is None


def test_candidates_are_permutation_equivariant_and_target_free():
    model, source, inputs, programs, anchors = _case()
    normal = model(source, inputs, ("integer",) * 2, programs, operation_spans=anchors)
    reverse = model(source, inputs, ("integer",) * 2, programs[::-1],
                    operation_spans=anchors[::-1])
    torch.testing.assert_close(normal, reverse.flip(0))
    assert not torch.isclose(normal[0], normal[1])
    assert not model.argument_evidence and model.retain_evidence_variants


def test_source_register_permutation_preserves_corresponding_graph_score():
    model, source, inputs, programs, anchors = _case()
    normal = model(source, inputs, ("integer",) * 2, (programs[0],),
                   operation_spans=anchors[:1])
    remapped = model(source, inputs[::-1], ("integer",) * 2, (programs[1],),
                     operation_spans=anchors[:1])
    torch.testing.assert_close(normal, remapped)


def test_computed_registers_keep_their_own_operation_anchors():
    model, source, inputs, _programs, _anchors = _case()
    program = Program(2, (Instruction("add", (0, 1)), Instruction("sub", (2, 0))))
    anchors = (TokenSpan(1, 2), TokenSpan(3, 4))
    loss, counts = model.source_atom_loss(source, inputs, ("integer",) * 2,
                                          program, operation_spans=anchors)
    assert torch.isfinite(loss)
    assert counts["operation_total"] == 2 and counts["reference_total"] == 4
    scores = model(source, inputs, ("integer",) * 2, (program,),
                   operation_spans=(anchors,))
    assert scores.shape == (1,) and torch.isfinite(scores).all()


def test_atom_supervision_changes_the_actual_selection():
    model, source, inputs, programs, anchors = _case()
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    for _ in range(40):
        loss, _counts = model.source_atom_loss(source, inputs, ("integer",) * 2,
                                               programs[1], operation_spans=anchors[1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    scores = model(source, inputs, ("integer",) * 2, programs, operation_spans=anchors)
    assert int(scores.argmax()) == 1


def test_invalid_anchors_and_role_types_are_rejected():
    model, source, inputs, programs, anchors = _case()
    with pytest.raises(ValueError, match="anchors"):
        model(source, inputs, ("integer",) * 2, programs)
    with pytest.raises(ValueError, match="role type"):
        model(source, inputs, ("integer_sequence", "integer"), programs,
              operation_spans=anchors)
    with pytest.raises(ValueError, match="grammar"):
        model(source, inputs, ("integer",) * 2,
              (Program(2, (Instruction("sub", (0, 2)),)),),
              operation_spans=anchors[:1])
