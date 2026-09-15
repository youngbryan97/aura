"""Compare global typed-graph optimization with exhaustive small-chart search."""

import itertools
from collections import Counter

import numpy as np
import pytest

from core.learning.semantic_argument_optimization import optimize_argument_chart
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import (
    RegisterUseContract,
    _operation_order,
    _OperationNode,
    _overlap,
)


def brute(options, contract, n_inputs=3, definition_options=None, definition_scores=None):
    nodes = tuple(
        _OperationNode(TokenSpan(20 + 2 * i, 21 + 2 * i), "add", 0, 0, 1)
        for i in range(len(options))
    )
    best = None
    slots = [slot for node in options for slot in node]
    labels = [slot for node in definition_options for slot in node] if definition_options is not None else None
    for indices in itertools.product(*(range(len(slot)) for slot in slots)):
        choices = [slot[index] for slot, index in zip(slots, indices, strict=True)]
        if labels is not None:
            definitions = {}
            for choice, candidates, index in zip(choices, labels, indices, strict=True):
                definitions.setdefault(choice[1], set()).add(candidates[index])
            if any(len(values) != 1 for values in definitions.values()):
                continue
        spans = [x[2] for x in choices]
        if any(_overlap(a, b) for a, b in itertools.combinations(spans, 2)):
            continue
        cursor = 0
        arguments = []
        for node in options:
            arguments.append(tuple(c[1] for c in choices[cursor:cursor + len(node)]))
            cursor += len(node)
        if contract.distinct_arguments and any(len(set(a)) != len(a) for a in arguments):
            continue
        dependencies = tuple(
            tuple(r - n_inputs for r in a if r >= n_inputs) for a in arguments
        )
        if _operation_order(dependencies, nodes, require_connected=True) is None:
            continue
        referenced = {d for ds in dependencies for d in ds}
        sink = next(i for i in range(len(options)) if i not in referenced)
        if not contract.accepts_complete(Counter(r for a in arguments for r in a),
                                         n_inputs=n_inputs, operation_count=len(options), sink=sink):
            continue
        score = sum(x[0] for x in choices)
        if definition_scores is not None:
            score += sum(definition_scores[register, next(iter(spans))]
                         for register, spans in definitions.items())
        best = score if best is None else max(best, score)
    return best


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("screen", [False, True])
def test_definition_attachment_objective_matches_exhaustive_graph_search(seed, screen):
    rng = np.random.default_rng(seed)
    names = (TokenSpan(10, 11), TokenSpan(12, 13))
    options = tuple(tuple(tuple(
        (float(rng.normal()), register, TokenSpan(2 * node + slot, 2 * node + slot + 1))
        for register in range(5) if register != 3 + node for _name in names
    ) for slot in range(2)) for node in range(2))
    labels = tuple(tuple(tuple(name for register in range(5) if register != 3 + node
                                for name in names) for _slot in range(2)) for node in range(2))
    scores = {(register, name): float(rng.normal()) for register in range(5) for name in names}
    contract = RegisterUseContract(1, 1, 1, 1, True)
    expected = brute(options, contract, definition_options=labels, definition_scores=scores)
    observed = optimize_argument_chart(options, n_inputs=3, contract=contract,
                                       definition_options=labels, definition_scores=scores,
                                       prune_dominated=screen)
    assert observed is not None
    assert observed[0] == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("seed", range(6))
def test_optimum_matches_exhaustive_search(seed):
    rng = np.random.default_rng(seed)
    options = tuple(tuple(tuple(
        (float(rng.normal()), register, TokenSpan(2 * node + slot, 2 * node + slot + 1))
        for register in range(5) if register != 3 + node
    ) for slot in range(2)) for node in range(2))
    contract = RegisterUseContract(1, 1, 1, 1, True)
    expected = brute(options, contract)
    result = optimize_argument_chart(options, n_inputs=3, contract=contract)
    assert result is not None
    assert result[0] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("arities", [(1,), (2,), (1, 2), (2, 1, 2)])
@pytest.mark.parametrize("distinct", [False, True])
@pytest.mark.parametrize("seed", [31, 73])
@pytest.mark.parametrize("screen", [False, True])
def test_varying_arity_use_bounds_and_overlap_match_exhaustive_search(arities, distinct, seed, screen):
    rng = np.random.default_rng(seed)
    options = tuple(tuple(tuple(
        (float(rng.normal()), register, TokenSpan(start, start + 2))
        for register in range(2 + len(arities)) if register != 2 + node
        for start in [int(rng.integers(0, 8))]
    ) for _slot in range(arity)) for node, arity in enumerate(arities))
    contract = RegisterUseContract(0, 2, 0, 3, distinct)
    expected = brute(options, contract, n_inputs=2)
    result = optimize_argument_chart(options, n_inputs=2, contract=contract, prune_dominated=screen)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert result[0] == pytest.approx(expected, abs=1e-9)
        permuted = tuple(tuple(tuple(reversed(slot)) for slot in node) for node in options)
        assert optimize_argument_chart(permuted, n_inputs=2, contract=contract)[0] == result[0]


def test_overlap_makes_otherwise_valid_graph_infeasible():
    options = ((((1., 0, TokenSpan(0, 2)),), ((1., 1, TokenSpan(1, 3)),)),)
    assert optimize_argument_chart(options, n_inputs=2,
        contract=RegisterUseContract(1, 1, 0, 1, True)) is None


def test_cycle_cannot_satisfy_connectivity_by_itself():
    options = ((((1., 2, TokenSpan(0, 1)),),), (((1., 1, TokenSpan(1, 2)),),))
    assert optimize_argument_chart(options, n_inputs=1,
        contract=RegisterUseContract(0, 1, 0, 1, True)) is None


def test_disconnected_operations_are_rejected_even_if_minimum_use_is_zero():
    options = ((((1., 0, TokenSpan(0, 1)),),), (((1., 0, TokenSpan(1, 2)),),))
    assert optimize_argument_chart(options, n_inputs=1,
        contract=RegisterUseContract(0, 2, 0, 2, True)) is None


def test_self_reference_is_not_an_argument_option():
    with pytest.raises(ValueError, match="invalid typed argument"):
        optimize_argument_chart(((((1., 1, TokenSpan(0, 1)),),),), n_inputs=1,
            contract=RegisterUseContract(0, 2, 0, 2, True))


def test_solver_limit_is_distinct_from_infeasibility(monkeypatch):
    from types import SimpleNamespace

    import scipy.optimize

    from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError

    monkeypatch.setattr(scipy.optimize, "milp", lambda *args, **kwargs:
                        SimpleNamespace(status=1, x=None))
    with pytest.raises(ArgumentOptimizationIncompleteError, match="status:1"):
        optimize_argument_chart(((((1., 0, TokenSpan(0, 1)),),),), n_inputs=1,
            contract=RegisterUseContract(1, 1, 0, 1, True))


@pytest.mark.parametrize("point", [[1., 1., 0.], [0., 1., 0.], [.5, 1., 0.],
                                   [float("nan"), 1., 0.], [1., 1.], [2., 1., 0.]])
def test_solver_output_is_checked_independently_of_its_status(monkeypatch, point):
    from types import SimpleNamespace

    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "milp", lambda *args, **kwargs:
                        SimpleNamespace(status=0, x=np.asarray(point)))
    options = ((((-1., 0, TokenSpan(0, 1)),),),)
    if point == [1., 1., 0.]:
        assert optimize_argument_chart(options, n_inputs=1,
            contract=RegisterUseContract(1, 1, 0, 1, True))[1] == ((0,),)
    else:
        with pytest.raises(ValueError, match="invalid solution"):
            optimize_argument_chart(options, n_inputs=1,
                contract=RegisterUseContract(1, 1, 0, 1, True))


def test_candidate_roundtrip_preserves_coefficients_and_changes_search_identity():
    import json
    from pathlib import Path

    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )

    path = Path(__file__).parents[1] / "artifacts/rlc/semantic_program_27b_frozen_path_v1/transducer.json"
    parent = compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))
    candidate = parent.with_global_constraint_arguments()
    assert candidate._coefficient_body() == parent._coefficient_body()
    assert candidate.receipt_sha256 != parent.receipt_sha256
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256


def test_joint_definitions_cannot_change_between_uses_of_one_register():
    a, b = TokenSpan(10, 11), TokenSpan(20, 21)
    options = (((
        (5.0, 0, TokenSpan(0, 1)), (3.0, 0, TokenSpan(0, 1)),
    ), (
        (3.0, 0, TokenSpan(2, 3)), (5.0, 0, TokenSpan(2, 3)),
    )),)
    contract = RegisterUseContract(2, 2, 0, 1, False)
    unconstrained = optimize_argument_chart(options, n_inputs=1, contract=contract)
    consistent = optimize_argument_chart(
        options, n_inputs=1, contract=contract, definition_options=(((a, b), (a, b)),),
    )
    assert unconstrained[0] == 10.0
    assert consistent[0] == 8.0


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("distinct", [False, True])
def test_joint_definition_optimum_matches_exhaustive_search(seed, distinct):
    rng = np.random.default_rng(seed)
    options = tuple(tuple(tuple(
        (float(rng.normal()), register, TokenSpan(2 * node + slot, 2 * node + slot + 1))
        for register in range(4) if register != 2 + node
        for _definition in range(2)
    ) for slot in range(2)) for node in range(2))
    definitions = tuple(tuple(tuple(
        TokenSpan(20 + 2 * register + definition, 21 + 2 * register + definition)
        for register in range(4) if register != 2 + node
        for definition in range(2)
    ) for _slot in range(2)) for node in range(2))
    contract = RegisterUseContract(0, 3, 1, 2, distinct)
    expected = brute(options, contract, n_inputs=2, definition_options=definitions)
    result = optimize_argument_chart(
        options, n_inputs=2, contract=contract, definition_options=definitions,
    )
    assert result is not None
    assert result[0] == pytest.approx(expected, abs=1e-9)
    reversed_options = tuple(tuple(tuple(reversed(slot)) for slot in node) for node in options)
    reversed_definitions = tuple(tuple(tuple(reversed(slot)) for slot in node) for node in definitions)
    assert optimize_argument_chart(
        reversed_options, n_inputs=2, contract=contract, definition_options=reversed_definitions,
    )[0] == pytest.approx(result[0], abs=1e-9)


def test_conflicting_definitions_can_make_an_otherwise_feasible_graph_impossible():
    options = ((((1.0, 0, TokenSpan(0, 1)),), ((1.0, 0, TokenSpan(2, 3)),)),)
    assert optimize_argument_chart(
        options, n_inputs=1, contract=RegisterUseContract(2, 2, 0, 1, False),
        definition_options=(((TokenSpan(10, 11),), (TokenSpan(12, 13),)),),
    ) is None


def test_definition_options_must_cover_every_argument_choice():
    with pytest.raises(ValueError, match="definition options differ"):
        optimize_argument_chart(
            ((((1.0, 0, TokenSpan(0, 1)),),),), n_inputs=1,
            contract=RegisterUseContract(1, 1, 0, 1, True), definition_options=(),
        )


def test_joint_definition_policy_reaches_the_existing_decoder(monkeypatch):
    from core.learning import semantic_argument_optimization as solver
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    parent = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    model = parent.with_order_invariant_argument_graph().with_joint_definition_graph()
    assert model._coefficient_body() == parent._coefficient_body()
    assert compositional_semantic_program_transducer_from_dict(model.to_dict()).receipt_sha256 == model.receipt_sha256
    original = solver.optimize_argument_chart
    calls = []

    def capture(options, **kwargs):
        labels = kwargs["definition_options"]
        assert labels is not None
        assert len(labels) == len(options)
        calls.append(labels)
        return original(options, **kwargs)

    monkeypatch.setattr(solver, "optimize_argument_chart", capture)
    item = next(item for item in examples if item.split == "test")
    result = model.decode(
        source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=item.ir.model_basis_receipt_sha256,
    )
    assert calls
    assert result.ir is not None
