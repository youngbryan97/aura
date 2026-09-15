"""Equivalent graphs are positives; finite agreement is not equivalence proof."""

from dataclasses import replace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_argument_chart import ScoredArgumentChart
from core.learning.semantic_graph_counterexamples import (
    argument_graph_program,
    compare_program_meanings,
    find_graph_counterexample,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import RegisterUseContract, _OperationNode


def chart(n_inputs=3):
    options = (tuple(tuple((float(10 - register - position), register,
                           TokenSpan(2 * position, 2 * position + 1))
                          for register in range(n_inputs)) for position in range(2)),)
    factors = tuple(tuple(tuple((option[0], 0., 0., 0.) for option in slot) for slot in row) for row in options)
    return ScoredArgumentChart(options, n_inputs, RegisterUseContract(0, 2, 0, 2, True), option_factors=factors)


def nodes(op="add"):
    return (_OperationNode(TokenSpan(10, 11), op, 0., 0., 1.),)


@pytest.mark.parametrize("prune", [False, True])
def test_multiple_graph_exclusions_remove_all_mentions_but_keep_other_graphs(prune):
    value = replace(chart(2), prune_dominated=prune)
    assert value.solve(excluded_graphs=[((0, 1),)])[1] == ((1, 0),)
    assert value.solve(excluded_graphs=[((0, 1),), ((1, 0),)]) is None
    assert value.solve(excluded_arguments=((0, 1),), excluded_graphs=[((1, 0),)]) is None
    with pytest.raises(ValueError, match="excluded"):
        value.solve(excluded_graphs=[((True, 0),)])


def test_commuted_high_score_is_not_a_training_negative():
    value = chart(2)
    result = find_graph_counterexample(value, nodes(), ((0, 1),), probes=[(2, 7)])
    assert result.negative is None
    assert result.receipt["status"] == "no_incorrect_graph"
    assert result.receipt["search_complete"]
    assert result.receipt["examined"][0]["comparison"]["status"] == "equivalent"


def test_real_wrong_binding_has_a_floor_execution_witness():
    result = find_graph_counterexample(chart(), nodes(), ((0, 1),), probes=[(2, 7, 11)])
    assert result.negative is not None
    assert result.receipt["highest_incorrect_proven"]
    comparison = result.receipt["examined"][-1]["comparison"]
    assert comparison["status"] == "different"
    witness = comparison["witness"]
    assert witness["outputs"][0] == 9 and witness["outputs"][1] != 9
    assert all(r["execution_engine"] == "universal_metered_floor" for r in witness["execution_receipts"])


def test_no_distinguishing_probe_is_unknown_not_an_incorrect_label():
    target = Program(2, (Instruction("sub", (0, 1)),))
    other = Program(2, (Instruction("sub", (1, 0)),))
    assert compare_program_meanings(target, other, [(3, 3)])["status"] == "unknown"
    result = find_graph_counterexample(chart(2), nodes("sub"), ((0, 1),), probes=[(3, 3)])
    assert result.negative is None and result.receipt["status"] == "equivalence_unresolved"
    assert result.receipt["search_complete"]
    assert not result.receipt["highest_incorrect_proven"]


def test_probe_failure_never_becomes_a_difference_witness():
    target = Program(2, (Instruction("idiv", (0, 1)),))
    other = Program(2, (Instruction("sub", (1, 0)),))
    report = compare_program_meanings(target, other, [(3, 0)])
    assert report["status"] == "unknown" and report["failed_probes"] == 1


def test_associative_and_distributive_meanings_are_proved_without_probes():
    left = Program(3, (Instruction("add", (0, 1)), Instruction("mul", (3, 2))))
    right = Program(3, (Instruction("mul", (0, 2)), Instruction("mul", (1, 2)), Instruction("add", (3, 4))))
    result = compare_program_meanings(left, right, [])
    assert result["status"] == "equivalent"
    assert result["method"] == "floor_integer_polynomial_v1"
    for x in range(-4, 5):
        for y in range(-4, 5):
            assert left.run((x, y, 7)) == right.run((x, y, 7))


def test_polynomial_certificate_rejects_partial_operations_and_bounded_expansion():
    from core.learning.semantic_program_floor import semantic_program_polynomial_key

    partial = Program(2, (Instruction("idiv", (0, 1)), Instruction("sub", (2, 2))))
    zero = Program(2, (Instruction("sub", (0, 0)),))
    assert semantic_program_polynomial_key(partial) is None
    assert compare_program_meanings(partial, zero, [(3, 0)])["status"] == "unknown"
    wide = Program(2, (Instruction("add", (0, 1)), Instruction("mul", (2, 2))))
    assert semantic_program_polynomial_key(wide, max_terms=3) is None
    assert semantic_program_polynomial_key(wide, max_terms=4) is not None
    with pytest.raises(ValueError):
        semantic_program_polynomial_key(wide, max_terms=0)


def test_polynomial_proof_keeps_input_identity_and_multiplicity():
    left = Program(2, (Instruction("add", (0, 0)),))
    right = Program(2, (Instruction("add", (0, 1)),))
    assert compare_program_meanings(left, right, [])['status'] == 'unknown'
    assert compare_program_meanings(left, right, [(2, 3)])['status'] == 'different'


def test_graph_budget_is_not_claimed_as_semantic_exhaustion():
    result = find_graph_counterexample(chart(), nodes(), ((0, 1),), probes=[], max_graphs=1)
    assert result.negative is None and result.receipt["status"] == "search_incomplete"
    assert not result.receipt["search_complete"]


def test_topological_normalization_preserves_forward_reference_meaning():
    operations = (nodes("sub")[0], replace(nodes("add")[0], span=TokenSpan(12, 13)))
    program = argument_graph_program(operations, ((3, 0), (0, 1)), n_inputs=2)
    assert program == Program(2, (Instruction("add", (0, 1)), Instruction("sub", (2, 0))))
    assert program.run((3, 7)) == 7
    with pytest.raises(ValueError, match="topological"):
        argument_graph_program(operations, ((3, 0), (2, 1)), n_inputs=2)


def test_higher_equivalent_realization_is_kept_as_positive():
    value = chart(2)
    options = ((((0., 0, TokenSpan(0, 1)), (10., 1, TokenSpan(0, 1))),
                ((0., 1, TokenSpan(2, 3)), (10., 0, TokenSpan(2, 3)))),)
    factors = tuple(tuple(tuple((option[0], 0., 0., 0.) for option in slot) for slot in row) for row in options)
    value = replace(value, options=options, option_factors=factors)
    result = find_graph_counterexample(value, nodes(), ((0, 1),), probes=[(2, 7)])
    assert result.positive[0][0] == 20.
    assert result.negative is None


def test_chart_construction_does_not_launch_an_unused_optimization(monkeypatch):
    from core.learning.semantic_program_compositional_transducer import fit_compositional_semantic_program_transducer
    from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    model = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding()).with_global_constraint_arguments()
    item = examples[0]
    monkeypatch.setattr(ScoredArgumentChart, "solve", lambda *a, **k: pytest.fail("unrequested optimization"))
    captured = []
    kwargs = dict(model=model, hidden=item.hidden_states, inputs=item.public_inputs,
        input_spans=item.ir.input_spans,
        operation_nodes=tuple(_OperationNode(i.operation_span, i.op, 0., 0., 1.) for i in item.ir.instructions),
        argument_pointer_scores=model.argument_pointer.score_sequence(item.hidden_states))
    assert _assign_typed_arguments(**kwargs, chart_observer=captured.append,
                                  retain_score_factors=True, build_only=True) is None
    assert len(captured) == 1 and captured[0].option_factors is not None
    with pytest.raises(ValueError, match="observer"):
        _assign_typed_arguments(**kwargs, build_only=True)


def test_incomplete_solver_is_retained_without_false_negative_label(monkeypatch):
    from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError

    def incomplete(*args, **kwargs):
        raise ArgumentOptimizationIncompleteError("limit")
    monkeypatch.setattr(ScoredArgumentChart, "solve_with_factors", incomplete)
    result = find_graph_counterexample(chart(), nodes(), ((0, 1),), solve_time_limit_s=1.)
    assert result.negative is None and result.receipt["status"] == "positive_search_incomplete"
    assert not result.receipt["search_complete"]
