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
