"""Earlier graph assignments must constrain candidates before beam truncation."""

import json
from pathlib import Path

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
