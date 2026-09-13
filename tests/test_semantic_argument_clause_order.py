"""Execution scheduling cannot change a source clause's proposal budget."""

import itertools

import numpy as np
import pytest

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import LinearPointerSequenceScores
from core.learning.semantic_program_transducer_fitting import (
    _argument_proposals_by_operation,
    _OperationNode,
)


@pytest.mark.parametrize("clause_local", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_proposal_banks_follow_source_order_not_execution_order(seed, clause_local):
    rng = np.random.default_rng(seed)
    scores = LinearPointerSequenceScores(rng.normal(size=300), rng.normal(size=300))
    nodes = tuple(
        _OperationNode(TokenSpan(start, start + 2), operation, 1., 1., 1.)
        for start, operation in ((90, "add"), (160, "sub"), (240, "mul"))
    )

    def banks(ordered):
        return _argument_proposals_by_operation(
            scores, input_spans=(TokenSpan(4, 6),), operation_nodes=ordered,
            max_span_tokens=3, clause_local=clause_local,
        )

    expected = dict(zip(nodes, banks(nodes), strict=True))
    for order in itertools.permutations(nodes):
        assert dict(zip(order, banks(order), strict=True)) == expected


def test_later_clause_retains_its_budget_when_it_executes_first():
    scores = LinearPointerSequenceScores(-np.arange(300, dtype=np.float32), np.zeros(300))
    early = _OperationNode(TokenSpan(140, 141), "add", 1., 1., 1.)
    late = _OperationNode(TokenSpan(200, 201), "sub", 1., 1., 1.)
    banks = _argument_proposals_by_operation(
        scores, input_spans=(), operation_nodes=(late, early),
        max_span_tokens=1, clause_local=True,
    )
    assert TokenSpan(220, 221) in {span for span, _ in banks[0]}
    assert TokenSpan(220, 221) not in {span for span, _ in banks[1]}


def test_empty_operation_chart_has_no_proposal_banks():
    assert _argument_proposals_by_operation(
        LinearPointerSequenceScores(np.zeros(3), np.zeros(3)), input_spans=(),
        operation_nodes=(), max_span_tokens=1, clause_local=True,
    ) == ()
