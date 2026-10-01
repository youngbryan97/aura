from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from core.learning.semantic_grounded_binding_engine import project_grounded_evidence
from core.learning.semantic_relational_pointer import (
    RelationalBindingPointer,
    semantic_role_features,
)
from tools.semantic_grounded_batched_chart import (
    BatchedGroundedBindingChartSolver,
    conditioned_scores,
)


@pytest.mark.parametrize("rounds", [0, 1, 2, 4])
@pytest.mark.parametrize("projection", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_vmap_preserves_every_individually_conditioned_pointer_score(rounds, projection, dtype):
    from tests.test_semantic_grounded_binding_engine import source
    mx.random.seed(27)
    evidence = source("public").evidence
    def depth_bank(bank):
        return {key: mx.random.normal((3, 4)).astype(dtype) for key in bank}
    evidence = replace(evidence, operations=depth_bank(evidence.operations),
        mentions=depth_bank(evidence.mentions), candidates=depth_bank(evidence.candidates))
    pointer = RelationalBindingPointer(4, depths=3, relation_width=8, rounds=rounds)
    pointer.lora_depth_query = mx.random.normal(pointer.lora_depth_query.shape)
    pointer.lora_depth_filler = mx.random.normal(pointer.lora_depth_filler.shape)
    pointer.lora_b = mx.random.normal(pointer.lora_b.shape)
    roles, candidates = len(evidence.roles), len(evidence.context.referents)
    adjacency = mx.ones((roles + candidates,) * 2)
    evidence = replace(evidence, adjacency=adjacency)
    basis = None
    if projection:
        basis = SimpleNamespace(basis=np.array([[1.], [0.], [0.], [0.]], dtype=np.float32))
    alternatives = [(0, column, mx.random.normal((3, 4)).astype(dtype),
        mx.random.normal((3, 4)).astype(dtype) if index % 2 else None)
        for index, column in enumerate([0, 1] * 17)]
    actual = conditioned_scores(pointer, evidence, alternatives, projection=basis, batch_size=16)
    expected = []
    for row, column, mention, definition in alternatives:
        role = evidence.roles[row]
        candidate = evidence.context.referents[column]
        conditioned = project_grounded_evidence(replace(evidence,
            mentions={**evidence.mentions, role.identity: mention},
            candidates={**evidence.candidates, **({candidate.key: definition} if definition is not None else {})}), basis)
        arrays, allowed = conditioned.arrays()
        expected.append(pointer(*arrays, adjacency=adjacency, allowed=allowed,
            role_features=semantic_role_features(evidence.roles))[row, column].item())
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_batched_full_chart_preserves_arguments_margin_and_edge_evidence():
    from tests.test_semantic_grounded_chart_bridge import fixture
    individual, chart, arguments = fixture()
    expected = individual(chart, **arguments)
    batched = BatchedGroundedBindingChartSolver(individual.engine, individual.source_id, individual.depth_states)
    actual = batched(chart, **arguments)
    assert actual[1:] == expected[1:]
    assert actual[0] == pytest.approx(expected[0], abs=1e-5)
    assert batched.last_resolution["margin"] == pytest.approx(individual.last_resolution["margin"], abs=1e-5)
    old, new = individual.last_resolution["edge_evidence"], batched.last_resolution["edge_evidence"]
    assert len(new) == len(old) and batched.last_resolution["all_options_retained"]
    for left, right in zip(old, new, strict=True):
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], float):
                assert right[key] == pytest.approx(left[key], abs=1e-5)
            else:
                assert right[key] == left[key]


def test_batched_pointer_rejects_inadmissible_and_nonfinite_conditions():
    from tests.test_semantic_grounded_binding_engine import source
    evidence = source("public").evidence
    pointer = RelationalBindingPointer(4)
    with pytest.raises(ValueError, match="inadmissible"):
        conditioned_scores(pointer, evidence, [(0, 99, mx.zeros((1, 4)), None)])
    with pytest.raises(ValueError, match="finite"):
        conditioned_scores(pointer, evidence, [(0, 0, mx.full((1, 4), mx.nan), None)])
    with pytest.raises(ValueError, match="bounded batch"):
        conditioned_scores(pointer, evidence, [(0, 0, mx.zeros((1, 4)), None)], batch_size=0)


@pytest.mark.parametrize("seed", range(6))
def test_certified_reduction_preserves_best_and_second_register_graph_with_definitions(seed):
    from core.learning.semantic_argument_chart import ScoredArgumentChart
    from core.learning.semantic_program_ir import TokenSpan
    from core.learning.semantic_program_transducer_fitting import RegisterUseContract
    from tests.test_semantic_argument_optimization import brute
    from tools.semantic_grounded_score_execution import reduce_dominated_mentions

    rng = np.random.default_rng(seed)
    names = (TokenSpan(20, 21), TokenSpan(22, 23))
    options, definitions = [], []
    for node in range(2):
        positions, labels = [], []
        for slot in range(2):
            pool, attached = [], []
            for register in range(5):
                if register == 3 + node:
                    continue
                for name in names:
                    score = float(rng.normal())
                    start = 3 * (2 * node + slot)
                    pool.extend(((score, register, TokenSpan(start, start + 1)),
                        (score - 1., register, TokenSpan(start, start + 2))))
                    attached.extend((name, name))
            positions.append(tuple(pool))
            labels.append(tuple(attached))
        options.append(tuple(positions))
        definitions.append(tuple(labels))
    scores = {(register, name): float(rng.normal()) for register in range(5) for name in names}
    contract = RegisterUseContract(1, 1, 1, 1, True)
    chart = ScoredArgumentChart(tuple(options), 3, contract, tuple(definitions), scores)
    reduced, witnesses, adjustment = reduce_dominated_mentions(chart)
    assert len(witnesses) == sum(len(slot) for node in reduced.options for slot in node)
    assert adjustment > 0
    original, actual = chart.solve(), reduced.solve()
    assert actual[0] == pytest.approx(original[0], abs=1e-6)
    assert actual[1] == original[1]
    assert actual[0] == pytest.approx(brute(reduced.options, contract,
        definition_options=reduced.definition_options, definition_scores=scores), abs=1e-6)
    old_second, new_second = chart.solve(excluded_arguments=original[1]), reduced.solve(excluded_arguments=original[1])
    assert (old_second is None) == (new_second is None)
    if old_second is not None:
        assert new_second[0] == pytest.approx(old_second[0], abs=1e-6)
        assert new_second[1] == old_second[1]
    for node, slot, removed, retained in witnesses:
        left, right = chart.options[node][slot][removed], chart.options[node][slot][retained]
        assert right[0] >= left[0] and left[1] == right[1]
        assert left[2].start <= right[2].start and right[2].end <= left[2].end
        assert chart.definition_options[node][slot][removed] == chart.definition_options[node][slot][retained]


def test_reduction_preserves_ambiguity_tolerance_from_every_original_option():
    from core.learning.semantic_argument_chart import ScoredArgumentChart
    from core.learning.semantic_program_ir import TokenSpan
    from tools.semantic_grounded_score_execution import reduce_dominated_mentions
    from tests.test_semantic_grounded_chart_bridge import fixture
    from core.learning.semantic_context_binding import BindingContext, BindingRole, ContextReferent

    _bridge, chart, _args = fixture()
    options = tuple(tuple(((float(register == slot) * .001, register, TokenSpan(3 + slot, 4 + slot)),
        *tuple((-100000., register, TokenSpan(3 + slot, 4 + slot)) for _ in range(4)))
        for register in range(2)) for slot in range(2))
    options = (tuple(tuple(choice for group in groups for choice in group) for groups in options),)
    chart = ScoredArgumentChart(options, 2, chart.contract)
    records = tuple(ContextReferent("public", str(index), "integer", "input") for index in range(2))
    context = BindingContext(records, {"integer": None})
    roles = (tuple(BindingRole(str(slot), f"sub:operand:{slot}", "integer") for slot in range(2)),)
    keys = {index: record.key for index, record in enumerate(records)}
    expected = chart.solve_grounded(context, roles, keys)
    reduced, _witnesses, adjustment = reduce_dominated_mentions(chart)
    actual = reduced.solve_grounded(context, roles, keys, minimum_margin=adjustment)
    assert expected.status == actual.status == "ambiguous"
    assert actual.margin == pytest.approx(expected.margin, abs=1e-8)


@pytest.mark.parametrize("offset", [-70., 0., 70., 1000.])
@pytest.mark.parametrize("weight", [0., .25, 1., 2.])
def test_conditional_relation_update_is_offset_invariant_and_preserves_baseline_mass(offset, weight):
    from scipy.special import logsumexp
    from tools.semantic_grounded_score_execution import conditional_role_update

    baseline, learned = (5., -2., 3., 1.), (1., 4., 2., -3.)
    expected, _ = conditional_role_update(baseline, learned, weight=weight)
    actual, receipt = conditional_role_update(baseline, [value + offset for value in learned], weight=weight)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert logsumexp(actual) == pytest.approx(logsumexp(baseline), abs=1e-12)
    assert np.argmax(actual) == np.argmax(np.asarray(baseline) + weight * np.asarray(learned))
    neutral, _ = conditional_role_update(baseline, [offset] * len(baseline), weight=weight)
    assert neutral == baseline
    assert receipt["uniform_evidence_is_neutral"] and not receipt["target_available"]


def test_uniform_pointer_bias_cannot_reward_an_unnecessary_operation():
    from tools.semantic_grounded_score_execution import conditional_role_update

    correct_baseline, extra_baseline = ((5., 0.),) * 6, ((5., 0.),) * 8
    operation_scores = (10., -7.)
    raw = [sum(max(value + 70. for value in slot) for slot in graph) + operation
        for graph, operation in zip((correct_baseline, extra_baseline), operation_scores, strict=True)]
    assert raw[1] > raw[0]
    normalized = [sum(max(conditional_role_update(slot, (70., 70.), weight=1.)[0])
        for slot in graph) + operation
        for graph, operation in zip((correct_baseline, extra_baseline), operation_scores, strict=True)]
    assert normalized[0] > normalized[1]


def test_conditional_relation_policy_changes_only_the_declared_chart_score_update():
    from tests.test_semantic_grounded_chart_bridge import fixture

    individual, chart, arguments = fixture()
    bridge = BatchedGroundedBindingChartSolver(individual.engine, individual.source_id,
        individual.depth_states, score_policy="conditional_likelihood")
    result = bridge(chart, **arguments)
    assert result is not None
    assert bridge.last_resolution["relation_score_policy"] == "conditional_likelihood"
    assert len(bridge.last_resolution["role_updates"]) == 2
    assert len(bridge.last_resolution["edge_evidence"]) == 4
    assert all(policy["choice_count"] == 2 and not policy["target_available"]
        for policy in bridge.last_resolution["role_updates"])


@pytest.mark.parametrize("baseline,learned,weight", [([], [], 1.), ([0.], [], 1.),
    ([0.], [mx.nan], 1.), ([0.], [0.], -1.)])
def test_conditional_role_update_rejects_incomplete_nonfinite_or_negative_weight(baseline, learned, weight):
    from tools.semantic_grounded_score_execution import conditional_role_update
    with pytest.raises(ValueError, match="complete finite"):
        conditional_role_update(baseline, learned, weight=weight)
