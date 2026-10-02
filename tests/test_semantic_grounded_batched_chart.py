import math
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


@pytest.mark.parametrize("rounds", [0, 2, 4])
@pytest.mark.parametrize("role_queries", [False, True])
@pytest.mark.parametrize("graph", ["absent", "dense", "sparse", "isolated"])
def test_selected_edge_readout_matches_original_full_pointer_for_mixed_roles(rounds, role_queries, graph):
    from core.learning.semantic_context_binding import BindingContext, BindingRole, ContextReferent
    from core.learning.semantic_grounded_binding_engine import GroundedBindingEvidence

    mx.random.seed(145)
    records = tuple(ContextReferent("public", str(index), "Number", "observed") for index in range(4))
    context = BindingContext(records, {"Number": None})
    roles = tuple(BindingRole(str(index), name, "Number")
        for index, name in enumerate(("value", "minuend", "subtrahend")))
    operations = {role.identity: mx.random.normal((3, 4)) for role in roles}
    mentions = {role.identity: mx.random.normal((3, 4)) for role in roles}
    candidates = {record.key: mx.random.normal((3, 4)) for record in records}
    adjacency = None
    if graph != "absent":
        adjacency = mx.ones((7, 7))
        if graph == "sparse":
            adjacency = mx.eye(7) + mx.roll(mx.eye(7), 1, axis=1)
        elif graph == "isolated":
            adjacency[2] = 0.
            adjacency[5] = 0.
    evidence = GroundedBindingEvidence("public", context, roles, operations, mentions, candidates, adjacency)
    pointer = RelationalBindingPointer(4, depths=3, relation_width=8, rounds=rounds, role_queries=role_queries)
    pointer.lora_depth_query = mx.random.normal(pointer.lora_depth_query.shape)
    pointer.lora_depth_filler = mx.random.normal(pointer.lora_depth_filler.shape)
    pointer.lora_b = mx.random.normal(pointer.lora_b.shape)
    alternatives = [(index % 3, (index // 3) % 4, mx.random.normal((3, 4)),
        None if index % 2 else mx.random.normal((3, 4))) for index in range(35)]
    actual = conditioned_scores(pointer, evidence, alternatives, batch_size=16)
    expected = []
    for row, column, mention, definition in alternatives:
        conditioned = replace(evidence, mentions={**mentions, roles[row].identity: mention},
            candidates={**candidates, **({records[column].key: definition} if definition is not None else {})})
        arrays, allowed = conditioned.arrays()
        expected.append(pointer(*arrays, adjacency=adjacency, allowed=allowed,
            role_features=semantic_role_features(roles))[row, column].item())
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("adjacency", [mx.full((3, 3), mx.nan), -mx.ones((3, 3)), mx.ones((2, 2))])
def test_selected_edge_readout_retains_original_graph_validation(adjacency):
    from tests.test_semantic_grounded_binding_engine import source

    evidence = replace(source("public").evidence, adjacency=adjacency)
    with pytest.raises(ValueError, match="graph evidence"):
        conditioned_scores(RelationalBindingPointer(4), evidence, [(0, 0, mx.zeros((1, 4)), None)])


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
    from core.learning.semantic_context_binding import BindingContext, BindingRole, ContextReferent
    from core.learning.semantic_program_ir import TokenSpan
    from tests.test_semantic_grounded_chart_bridge import fixture
    from tools.semantic_grounded_score_execution import reduce_dominated_mentions

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


@pytest.mark.parametrize("seed", range(6))
def test_conditional_chart_bound_covers_arbitrary_learned_logits_and_definitions(seed):
    from core.learning.semantic_argument_chart import ScoredArgumentChart
    from core.learning.semantic_program_ir import TokenSpan
    from core.learning.semantic_program_transducer_fitting import RegisterUseContract
    from tools.semantic_grounded_score_execution import conditional_chart_upper_bound, conditional_role_update

    rng = np.random.default_rng(seed)
    options = tuple(tuple(tuple((float(rng.normal()), register, TokenSpan(slot, slot + 1))
        for register in range(5)) for slot in range(2)) for _ in range(2))
    scores = {(register, TokenSpan(name, name + 1)): float(rng.normal())
        for register in range(5) for name in (5, 7)}
    chart = ScoredArgumentChart(options, 3, RegisterUseContract(1, 1, 1, 1, True),
        definition_scores=scores, choice_log_normalizer=2.3)
    upper = conditional_chart_upper_bound(chart)
    for weight in (0., .1, 1., 100.):
        values = [max(conditional_role_update([choice[0] for choice in slot],
            rng.normal(0., 100., len(slot)), weight=weight)[0]) for node in options for slot in node]
        definition_upper = sum(max(0., *(score for (identity, _span), score in scores.items() if identity == register))
            for register in range(5))
        assert sum(values) + definition_upper - chart.choice_log_normalizer <= upper + 1e-10


def test_partition_bound_skips_only_a_proved_losing_complete_chart(monkeypatch):
    from tools import semantic_grounded_batched_chart as execution
    individual, chart, arguments = fixture_for_bound()
    bridge = BatchedGroundedBindingChartSolver(individual.engine, individual.source_id,
        individual.depth_states, score_policy="conditional_likelihood", length_penalty=0.)
    first = bridge(chart, **arguments)
    assert first is not None and bridge.last_resolution["status"] == "bound"
    winner_score = bridge.best_joint_score
    losing = {**arguments, "operation_nodes": tuple(SimpleNamespace(**{**vars(node), "score": -1000.})
        for node in arguments["operation_nodes"])}
    def must_not_score(*args, **kwargs):
        raise AssertionError("proved losing chart reached learned scoring")
    monkeypatch.setattr(execution, "conditioned_scores", must_not_score)
    assert bridge(chart, **losing) is None
    proof = bridge.last_resolution
    assert proof["status"] == "certified_pruned"
    assert proof["chart_upper_bound"] + proof["bound_roundoff_guard"] < winner_score
    assert not proof["all_options_retained"] and bridge.best_joint_score == winner_score


def fixture_for_bound():
    from tests.test_semantic_grounded_chart_bridge import fixture
    individual, chart, arguments = fixture()
    arguments = {**arguments, "operation_nodes": tuple(SimpleNamespace(**vars(node), score=0.)
        for node in arguments["operation_nodes"])}
    return individual, chart, arguments


def test_partition_bound_keeps_equal_or_better_charts_and_raw_evidence():
    individual, chart, arguments = fixture_for_bound()
    for policy in ("raw", "conditional_likelihood"):
        bridge = BatchedGroundedBindingChartSolver(individual.engine, individual.source_id,
            individual.depth_states, score_policy=policy, length_penalty=0.)
        assert bridge(chart, **arguments) is not None
        assert bridge(chart, **arguments) is not None
        assert bridge.last_resolution["status"] == "bound"
        better = {**arguments, "operation_nodes": tuple(SimpleNamespace(**{**vars(node), "score": 1000.})
            for node in arguments["operation_nodes"])}
        assert bridge(chart, **better) is not None
        assert bridge.last_resolution["status"] == "bound"


@pytest.mark.parametrize("rejection", ["order", "register_use"])
def test_partition_bound_never_uses_an_assignment_the_parent_rejects(monkeypatch, rejection):
    from tools import semantic_grounded_batched_chart as execution

    individual, chart, arguments = fixture_for_bound()
    bridge = BatchedGroundedBindingChartSolver(individual.engine, individual.source_id,
        individual.depth_states, score_policy="conditional_likelihood", length_penalty=0.)
    if rejection == "order":
        monkeypatch.setattr(execution, "_operation_order", lambda *_args, **_kwargs: None)
    else:
        monkeypatch.setattr(type(chart.contract), "accepts_complete", lambda *_args, **_kwargs: False)
    for operation_score in (0., -1000.):
        nodes = tuple(SimpleNamespace(**{**vars(node), "score": operation_score})
            for node in arguments["operation_nodes"])
        assert bridge(chart, **{**arguments, "operation_nodes": nodes}) is not None
        assert not bridge.last_resolution["parent_assignment_checks_passed"]
        assert bridge.best_joint_score == -math.inf


def test_complete_graph_selector_agrees_with_unpruned_execution():
    from core.learning.semantic_argument_chart import select_operation_argument_graph

    individual, chart, arguments = fixture_for_bound()
    charts = tuple(tuple(SimpleNamespace(**{**vars(node), "score": score})
        for node in arguments["operation_nodes"]) for score in (0., -1000., 50., 50., -500.))
    outcomes = []
    for penalty in (None, 0.):
        bridge = BatchedGroundedBindingChartSolver(individual.engine, individual.source_id,
            individual.depth_states, score_policy="conditional_likelihood", length_penalty=penalty)
        def assign(nodes):
            result = bridge(chart, **{**arguments, "operation_nodes": nodes})
            return None if result is None else SimpleNamespace(score=result[0], assignment=result,
                operation_nodes=nodes)
        selected = select_operation_argument_graph(charts, assign, length_penalty=0., joint=True)
        outcomes.append((selected.score, selected.assignment, selected.operation_nodes))
        assert sum(row["status"] == "certified_pruned" for row in bridge.resolutions) == (0 if penalty is None else 2)
    assert outcomes[0] == outcomes[1]


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
