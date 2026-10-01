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
def test_vmap_preserves_every_individually_conditioned_pointer_score(rounds, projection):
    from tests.test_semantic_grounded_binding_engine import source
    mx.random.seed(27)
    evidence = source("public").evidence
    pointer = RelationalBindingPointer(4, relation_width=8, rounds=rounds)
    pointer.lora_b = mx.random.normal(pointer.lora_b.shape)
    roles, candidates = len(evidence.roles), len(evidence.context.referents)
    adjacency = mx.ones((roles + candidates,) * 2)
    evidence = replace(evidence, adjacency=adjacency)
    basis = None
    if projection:
        basis = SimpleNamespace(basis=np.array([[1.], [0.], [0.], [0.]], dtype=np.float32))
    alternatives = [(0, column, mx.random.normal((1, 4)), mx.random.normal((1, 4)) if index % 2 else None)
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
