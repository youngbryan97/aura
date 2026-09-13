"""Earlier graph assignments must constrain candidates before beam truncation."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import (
    RegisterUseContract,
    _OperationNode,
    _prefix_feasible_arguments,
)


def _search(options, *, spans=None, dependencies=((),)):
    if spans is None:
        spans = ((TokenSpan(0, 1), TokenSpan(2, 3)),)
    nodes = tuple(
        _OperationNode(TokenSpan(index, index + 1), "add", 1.0, 1.0, 1.0)
        for index in (4, 12)
    )
    return _prefix_feasible_arguments(
        options,
        arguments=((0, 1),),
        spans=spans,
        dependencies=dependencies,
        operation_nodes=nodes,
        n_inputs=3,
        contract=RegisterUseContract(1, 1, 1, 1, True),
        beam=1,
    )


def test_used_registers_cannot_consume_the_only_continuation_slot():
    options = (
        ((100.0, 0, TokenSpan(6, 7)), (1.0, 2, TokenSpan(7, 8))),
        ((100.0, 1, TokenSpan(8, 9)), (1.0, 3, TokenSpan(9, 10))),
    )
    assert _search(options) == [(2.0, (2, 3), (TokenSpan(7, 8), TokenSpan(9, 10)))]


def test_prior_mentions_cannot_consume_the_only_continuation_slot():
    options = (
        ((100.0, 2, TokenSpan(0, 1)), (1.0, 2, TokenSpan(7, 8))),
        ((1.0, 3, TokenSpan(9, 10)),),
    )
    assert _search(options)[0][1] == (2, 3)


def test_cycle_and_missing_register_constraints_still_apply():
    options = (((1.0, 2, TokenSpan(7, 8)),), ((1.0, 3, TokenSpan(9, 10)),))
    assert _search(options, dependencies=((1,),)) == []
    assert _search((options[0], ((1.0, 2, TokenSpan(9, 10)),))) == []


def test_candidate_identity_changes_but_coefficients_and_legacy_replay_do_not():
    path = Path(__file__).parents[1] / "artifacts/rlc/semantic_program_27b_frozen_path_v1/transducer.json"
    frozen = compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))
    candidate = frozen.with_prefix_feasible_arguments()
    assert candidate.receipt_sha256 != frozen.receipt_sha256
    assert candidate.training_receipt["coefficient_sha256"] == frozen.training_receipt["coefficient_sha256"]
    assert "argument_search_strategy" not in frozen.training_receipt
    replay = compositional_semantic_program_transducer_from_dict(candidate.to_dict())
    assert replay.receipt_sha256 == candidate.receipt_sha256
    assert replay.training_receipt["argument_search_strategy"] == "prefix_feasible_v1"


def test_proposal_training_uses_the_runtime_operation_banks(monkeypatch):
    from core.learning import semantic_program_transducer_fitting as fitting
    from core.learning.semantic_program_floor import semantic_primitive_type_signature
    from core.learning.semantic_program_transducer import LinearPointerHead
    from tests.test_semantic_program_shared_transducer import _examples

    item = _examples()[0]
    width = item.hidden_states.shape[1]
    pointer = LinearPointerHead(np.zeros(width), 0.0, np.zeros(width), 0.0)
    calls = []

    def banks(scores, **kwargs):
        calls.append(kwargs)
        return tuple(
            ((instruction.argument_spans[1], 0.0),)
            for instruction in item.ir.instructions
        )

    monkeypatch.setattr(fitting, "_argument_proposals_by_operation", banks)
    type_bounds = {
        kind: 3
        for instruction in item.ir.instructions
        for kind in semantic_primitive_type_signature(instruction.op)[0]
    }
    features, labels, weights, positives, negatives = fitting._argument_proposal_rows(
        (item,),
        argument_pointer=pointer,
        position=0,
        max_span_tokens=3,
        max_argument_span_tokens_by_type=type_bounds,
        hidden_channels=item.hidden_channels,
        hidden_channel_widths=item.hidden_channel_widths,
    )
    assert len(calls) == 1
    assert calls[0]["clause_local"] is True
    assert tuple(node.span for node in calls[0]["operation_nodes"]) == tuple(
        instruction.operation_span for instruction in item.ir.instructions
    )
    assert positives == negatives == len(item.ir.instructions)
    assert features.shape[0] == labels.size == weights.size == positives + negatives


def test_proposal_refit_preserves_other_heads_and_excludes_test_labels():
    from dataclasses import replace

    from core.learning.semantic_program_compositional_transducer import (
        fit_compositional_semantic_program_transducer,
        refit_compositional_argument_proposals,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    model = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    candidate = refit_compositional_argument_proposals(model, examples)
    assert candidate.operation_pointer is model.operation_pointer
    assert candidate.argument_pointer is model.argument_pointer
    assert candidate.definition_relation_head is model.definition_relation_head
    assert candidate.register_use_contract == model.register_use_contract
    assert candidate.training_receipt["argument_proposal_refit"]["test_examples_used"] == 0
    without_test = tuple(item for item in examples if item.split != "test")
    assert refit_compositional_argument_proposals(model, without_test).receipt_sha256 == candidate.receipt_sha256
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256
    with pytest.raises(ValueError, match="train and validation"):
        refit_compositional_argument_proposals(model, tuple(item for item in examples if item.split == "test"))
    with pytest.raises(ValueError, match="overlap"):
        refit_compositional_argument_proposals(
            model, (replace(examples[0], split="train"), replace(examples[0], split="validation"))
        )


def test_refit_publishes_new_calibration_with_changed_scale(monkeypatch):
    from core.learning import semantic_program_compositional_transducer as module
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    model = module.fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    scale = model.argument_proposal_scale + 1.0
    calibration = [{"proposal_scale": scale, "selected": True}]
    monkeypatch.setattr(module, "_select_argument_proposal_scale", lambda *args, **kwargs: (scale, calibration))
    candidate = module.refit_compositional_argument_proposals(model, examples)
    assert candidate.argument_proposal_scale == scale
    assert candidate.training_receipt["argument_proposal_fit"]["scale_selection"] == calibration
    assert candidate.training_receipt["argument_proposal_refit"]["calibration"] == calibration
    assert module.compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256
