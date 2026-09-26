"""Equivalent graphs are positives; finite agreement is not equivalence proof."""

from dataclasses import replace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_argument_chart import ScoredArgumentChart
from core.learning.semantic_graph_counterexamples import (
    argument_graph_program,
    compare_program_meanings,
    compare_transition_counterfactuals,
    compare_transition_horizons,
    find_graph_counterexample,
    find_program_counterexample,
    uniform_reduction_equivalence,
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


def test_transition_probe_finds_delayed_difference_without_repeated_seed_credit():
    increment = Program(2, (Instruction("add", (0, 1)),))
    double = Program(2, (Instruction("mul", (0, 1)),))
    report = compare_transition_horizons(
        increment, double, ((2, 2), (2, 2)), feedback_input=0, horizon=4)
    assert report["status"] == "different"
    assert report["distinct_initial_contexts"] == 1
    assert [step["step"] for step in report["witness"]["trace"]] == [1, 2]
    assert report["witness"]["trace"][-1]["outputs"] == [6, 8]
    assert compare_program_meanings(
        increment, double, ((2, 2),), transition_feedback_input=0,
        transition_horizon=4)["witness_sha256"] == report["witness_sha256"]


def test_transition_probe_keeps_sampled_agreement_unproven():
    forward = Program(2, (Instruction("add", (0, 1)),))
    reverse = Program(2, (Instruction("add", (1, 0)),))
    report = compare_transition_horizons(
        forward, reverse, ((2, 1), (2, 1)), feedback_input=0, horizon=4)
    assert report["status"] == "unknown"
    assert report["steps_checked"] == 4
    assert report["distinct_initial_contexts"] == 1
    assert report["failed_traces"] == 0
    varied = compare_transition_counterfactuals(
        forward, reverse, (2, 1), feedback_input=0, count=4, seed=9, horizon=3)
    assert varied == compare_transition_counterfactuals(
        forward, reverse, (2, 1), feedback_input=0, count=4, seed=9, horizon=3)
    assert varied["status"] == "unknown"
    assert varied["distinct_initial_contexts"] > 1
    with pytest.raises(ValueError, match="bounded matching state geometry"):
        compare_transition_horizons(forward, reverse, ((2, 1),), feedback_input=0, horizon=17)


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


def test_other_operation_chart_is_compared_without_forcing_source_operations():
    target = Program(2, (Instruction("mul", (0, 1)),))
    result = find_program_counterexample(chart(2), nodes("sub"), target, probes=[(2, 7)])
    assert result.positive is None
    assert result.negative is not None and result.receipt["highest_incorrect_proven"]
    witness = result.receipt["examined"][0]["comparison"]["witness"]
    assert witness["outputs"][0] == 14 and witness["outputs"][1] != 14


def test_cross_chart_equivalent_program_can_supply_a_positive_without_a_source_binding():
    result = find_program_counterexample(chart(2), nodes(),
        Program(2, (Instruction("add", (1, 0)),)), probes=[])
    assert result.positive is not None and result.negative is None
    assert result.receipt["search_complete"]


def test_cross_chart_finite_agreement_is_not_retained_as_positive():
    result = find_program_counterexample(chart(2), nodes("sub"),
        Program(2, (Instruction("mul", (0, 1)),)), probes=[(0, 0)])
    assert result.positive is None and result.negative is None
    assert result.receipt["status"] == "equivalence_unresolved"


def test_no_distinguishing_probe_is_unknown_not_an_incorrect_label():
    target = Program(2, (Instruction("sub", (0, 1)),))
    other = Program(2, (Instruction("sub", (1, 0)),))
    assert compare_program_meanings(target, other, [(3, 3)])["status"] == "unknown"
    result = find_graph_counterexample(chart(2), nodes("sub"), ((0, 1),), probes=[(3, 3)])
    assert result.negative is None and result.receipt["status"] == "equivalence_unresolved"
    assert result.receipt["search_complete"]
    assert not result.receipt["highest_incorrect_proven"]


def test_definedness_difference_has_a_reference_checked_floor_witness():
    target = Program(2, (Instruction("idiv", (0, 1)),))
    other = Program(2, (Instruction("sub", (1, 0)),))
    report = compare_program_meanings(target, other, [(3, 0)])
    assert report["status"] == "different" and report["distinction"] == "domain"
    undefined, defined = report["witness"]["outcomes"]
    assert undefined["status"] == "undefined" and not undefined["reference_defined"]
    assert undefined["compiled_receipt"] and undefined["failure"].startswith("Stuck:")
    assert defined["status"] == "value" and defined["reference_defined"]
    assert defined["execution_receipt"]["execution_engine"] == "universal_metered_floor"
    reverse = compare_program_meanings(other, target, [(3, 0)])
    assert reverse["status"] == "different" and reverse["distinction"] == "domain"


def test_joint_domain_rejection_does_not_prove_equivalence():
    first = Program(2, (Instruction("idiv", (0, 1)),))
    second = Program(2, (Instruction("mod", (0, 1)),))
    report = compare_program_meanings(first, second, [(3, 0)])
    assert report["status"] == "unknown"
    assert report["jointly_undefined_probes"] == 1 and report["failed_probes"] == 0


@pytest.mark.parametrize("failure", [RuntimeError("worker gone"), ValueError("bad receipt")])
def test_infrastructure_failure_is_not_a_domain_witness(monkeypatch, failure):
    import core.learning.semantic_graph_counterexamples as module

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(module, "execute_semantic_floor_program", fail)
    report = compare_program_meanings(Program(2, (Instruction("idiv", (0, 1)),)),
                                     Program(2, (Instruction("sub", (1, 0)),)), [(3, 0)])
    assert report["status"] == "unknown" and report["failed_probes"] == 1


def test_exhaustion_cannot_supply_a_domain_witness():
    report = compare_program_meanings(Program(2, (Instruction("idiv", (0, 1)),)),
                                     Program(2, (Instruction("sub", (1, 0)),)), [(3, 0)], fuel=1)
    assert report["status"] == "unknown" and report["failed_probes"] == 1


def test_lowering_disagreement_cannot_supply_a_difference_witness(monkeypatch):
    import core.learning.semantic_graph_counterexamples as module
    from types import SimpleNamespace

    monkeypatch.setattr(module, "execute_semantic_floor_program",
                        lambda *a, **k: SimpleNamespace(result=123456, receipt={"bad": True}))
    report = compare_program_meanings(Program(2, (Instruction("add", (0, 1)),)),
                                     Program(2, (Instruction("sub", (1, 0)),)), [(3, 0)])
    assert report["status"] == "unknown" and report["failed_probes"] == 1


def test_floor_stuck_on_a_defined_reference_remains_an_error(monkeypatch):
    import core.learning.semantic_graph_counterexamples as module
    from core.cognition.the_floor_she_stands_on import Stuck

    def broken_floor(*args, **kwargs):
        raise Stuck("incorrect lowering")

    monkeypatch.setattr(module, "execute_semantic_floor_program", broken_floor)
    report = compare_program_meanings(Program(2, (Instruction("idiv", (0, 1)),)),
                                     Program(2, (Instruction("sub", (1, 0)),)), [(3, 0)])
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
    assert compare_program_meanings(partial, zero, [(3, 0)])["distinction"] == "domain"
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


@pytest.mark.parametrize('op', ['add', 'mul'])
def test_uniform_linear_reduction_proves_whole_class_without_negative_search(op, monkeypatch):
    options = tuple(tuple(tuple((float(register), register, TokenSpan(2 * (2 * node + position),
                                2 * (2 * node + position) + 1))
                                for register in range(5) if register != 3 + node)
                          for position in range(2)) for node in range(2))
    factors = tuple(tuple(tuple((choice[0], 0., 0., 0.) for choice in slot) for slot in row) for row in options)
    value = ScoredArgumentChart(options, 3, RegisterUseContract(1, 1, 1, 1, True), option_factors=factors)
    operations = (nodes(op)[0], replace(nodes(op)[0], span=TokenSpan(12, 13)))
    original = ScoredArgumentChart.solve_with_factors
    def solve(self, **kwargs):
        assert not kwargs.get('excluded_graphs'), 'equivalent class must not launch negative search'
        return original(self, **kwargs)
    monkeypatch.setattr(ScoredArgumentChart, 'solve_with_factors', solve)
    result = find_graph_counterexample(value, operations, ((0, 1), (3, 2)), max_graphs=1)
    assert result.negative is None and result.receipt['search_complete']
    assert result.receipt['equivalence_class_proof']['method'] == 'linear_use_integer_monoid_v1'
    assert not result.receipt['examined']
    assert uniform_reduction_equivalence(replace(value, contract=RegisterUseContract(0, 2, 1, 1, True)), operations) is None
    assert uniform_reduction_equivalence(value, (nodes('sub')[0], operations[1])) is None
    assert uniform_reduction_equivalence(value, (nodes('sub')[0], nodes('sub')[0])) is None
