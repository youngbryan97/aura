"""Only mathematically impossible operation charts are removed before the beam."""

import itertools
from collections import Counter

import pytest

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import (
    RegisterUseContract, _OperationNode, _operation_chart_candidates,
    _operation_chart_use_feasible,
)


def node(index, op="add", score=10.):
    return _OperationNode(TokenSpan(index * 2, index * 2 + 1), op, score, score, 1.)


def test_impossible_single_nodes_cannot_exhaust_the_chart_beam():
    nodes = tuple(node(index, score=10. - index) for index in range(5))
    contract = RegisterUseContract(1, 1, 1, 1, True)
    legacy = _operation_chart_candidates(nodes, max_steps=2, length_penalty=20., limit=2)
    assert all(len(chart) == 1 for chart in legacy)
    kept = _operation_chart_candidates(nodes, max_steps=2, length_penalty=20., limit=2,
        feasible=lambda chart: _operation_chart_use_feasible(chart, n_inputs=3, contract=contract))
    assert len(kept) == 2 and all(len(chart) == 2 for chart in kept)


@pytest.mark.parametrize("inputs", [1, 2, 3])
@pytest.mark.parametrize("operations", [("add",), ("neg",), ("add", "add"), ("neg", "add")])
@pytest.mark.parametrize("contract", [RegisterUseContract(1, 1, 1, 1, True), RegisterUseContract(0, 3, 0, 3, False)])
def test_every_enumerated_valid_connected_program_passes_the_bound(inputs, operations, contract):
    arities = [1 if op == "neg" else 2 for op in operations]
    charts = tuple(node(index, op) for index, op in enumerate(operations))
    choices = [tuple(itertools.product(range(inputs + index), repeat=arity)) for index, arity in enumerate(arities)]
    for arguments in itertools.product(*choices):
        if contract.distinct_arguments and any(len(set(args)) != len(args) for args in arguments):
            continue
        counts = Counter(register for args in arguments for register in args)
        if not contract.accepts_complete(counts, n_inputs=inputs, operation_count=len(operations), sink=len(operations)-1):
            continue
        if any(counts[inputs + index] < 1 for index in range(len(operations)-1)):
            continue
        assert _operation_chart_use_feasible(charts, n_inputs=inputs, contract=contract)


def test_unknown_operation_is_not_a_feasibility_claim():
    assert not _operation_chart_use_feasible((node(0, "not-an-operation"),), n_inputs=2,
        contract=RegisterUseContract(0, 3, 0, 3, False))


def test_partial_arity_states_preserve_lower_scored_feasible_charts():
    nodes = (node(0, 'neg', 20.), node(1, 'neg', 19.), node(2, 'add', 2.), node(3, 'add', 1.))
    contract = RegisterUseContract(1, 1, 1, 1, True)
    feasible = lambda chart: _operation_chart_use_feasible(chart, n_inputs=3, contract=contract)
    assert not _operation_chart_candidates(nodes, max_steps=2, length_penalty=0., limit=1, feasible=feasible)
    charts = _operation_chart_candidates(nodes, max_steps=2, length_penalty=0., limit=1,
                                        feasible=feasible, preserve_arity_states=True)
    assert len(charts) == 1
    assert tuple(n.operation for n in charts[0]) == ('add', 'add')


def test_arity_state_search_matches_exhaustive_feasible_top_k():
    nodes = tuple(node(index, 'neg' if index % 2 else 'add', float(8-index)) for index in range(6))
    contract = RegisterUseContract(1, 1, 1, 1, True)
    feasible = lambda chart: _operation_chart_use_feasible(chart, n_inputs=3, contract=contract)
    exhaustive = [chart for count in range(1, 4) for chart in itertools.combinations(nodes, count) if feasible(chart)]
    exhaustive.sort(key=lambda chart: (-sum(n.score for n in chart), len(chart), tuple((n.span.start, n.span.end) for n in chart)))
    actual = _operation_chart_candidates(nodes, max_steps=3, length_penalty=0., limit=3,
                                        feasible=feasible, preserve_arity_states=True)
    assert actual == tuple(exhaustive[:3])


@pytest.mark.parametrize('seed', range(5))
def test_arity_state_search_matches_overlapping_exhaustive_charts(seed):
    import random
    randomizer = random.Random(seed)
    nodes = tuple(_OperationNode(TokenSpan(i, i + 2), 'neg' if i % 3 else 'add',
                                 randomizer.random(), 0., 1.) for i in range(8))
    contract = RegisterUseContract(1, 1, 1, 1, True)
    feasible = lambda chart: _operation_chart_use_feasible(chart, n_inputs=2, contract=contract)
    exhaustive = [chart for count in range(1, 4) for chart in itertools.combinations(nodes, count)
                  if all(a.span.end <= b.span.start for a, b in zip(chart, chart[1:])) and feasible(chart)]
    exhaustive.sort(key=lambda chart: (-sum(n.score for n in chart) + .2 * len(chart), len(chart),
                                      tuple((n.span.start, n.span.end) for n in chart)))
    assert _operation_chart_candidates(nodes, max_steps=3, length_penalty=.2, limit=3,
                                       feasible=feasible, preserve_arity_states=True) == tuple(exhaustive[:3])
