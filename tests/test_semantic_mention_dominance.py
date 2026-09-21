"""Pruning preserves the best feasible overlap-constrained assignment."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from core.learning.semantic_argument_optimization import (
    optimize_argument_chart,
    _dual_bound_screen,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import (
    RegisterUseContract,
    _retained_argument_mentions,
)
from tests.test_semantic_argument_optimization import brute


def _retain(candidates):
    return _retained_argument_mentions(candidates, overlap_complete=True)


def test_locally_fifth_mention_is_needed_by_the_whole_graph():
    crowded = [(10. - i, TokenSpan(0, i + 2)) for i in range(4)]
    useful = (1., TokenSpan(6, 7))
    choices = [*crowded, useful]
    other = ((2., 1, TokenSpan(0, 1)),)
    contract = RegisterUseContract(1, 1, 0, 1, True)

    def solve(selected):
        options = ((tuple((score, 0, span) for score, span in selected), other),)
        return optimize_argument_chart(options, n_inputs=2, contract=contract)

    assert solve(_retained_argument_mentions(choices)) is None
    assert solve(_retain(choices))[0] == solve(choices)[0] == 3.
    assert useful in _retain(choices)


def test_only_a_better_subset_can_dominate_a_mention():
    candidates = [(5., TokenSpan(0, 4)), (5., TokenSpan(1, 2)),
                  (6., TokenSpan(0, 5)), (4., TokenSpan(6, 7)),
                  (5., TokenSpan(1, 2))]
    expected = [(6., TokenSpan(0, 5)), (5., TokenSpan(1, 2)), (4., TokenSpan(6, 7))]
    assert _retain(candidates) == expected
    assert all(_retain(order) == expected for order in itertools.permutations(candidates))


@pytest.mark.parametrize("seed", range(8))
def test_dominance_matches_unpruned_exhaustive_search_with_definition_identity(seed):
    rng = np.random.default_rng(seed)
    definitions = (TokenSpan(10, 11), TokenSpan(12, 13))
    rows, labels, retained_rows, retained_labels = [], [], [], []
    for _slot in range(2):
        raw, names, kept, kept_names = [], [], [], []
        for register, definition in itertools.product(range(2), definitions):
            candidates = [(float(rng.integers(-3, 4)), TokenSpan(start, start + width))
                          for start, width in ((0, 3), (1, 1), (4, 3), (5, 1), (7, 1))]
            raw.extend((score, register, span) for score, span in candidates)
            names.extend([definition] * len(candidates))
            selected = _retain(candidates)
            kept.extend((score, register, span) for score, span in selected)
            kept_names.extend([definition] * len(selected))
        rows.append(tuple(raw))
        labels.append(tuple(names))
        retained_rows.append(tuple(kept))
        retained_labels.append(tuple(kept_names))
    contract = RegisterUseContract(0, 2, 0, 1, False)
    attachment = {(register, definition): float(rng.normal())
                  for register, definition in itertools.product(range(2), definitions)}
    expected = brute((tuple(rows),), contract, n_inputs=2,
                     definition_options=(tuple(labels),), definition_scores=attachment)
    actual = optimize_argument_chart((tuple(retained_rows),), n_inputs=2, contract=contract,
        definition_options=(tuple(retained_labels),), definition_scores=attachment)
    assert actual is not None
    assert actual[0] == pytest.approx(expected, abs=1e-8)


def test_policy_roundtrip_changes_identity_without_retraining():
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )

    path = Path(__file__).parents[1] / "artifacts/rlc/semantic_program_27b_frozen_path_v1/transducer.json"
    parent = compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))
    candidate = parent.with_overlap_complete_mentions()
    assert parent._coefficient_body() == candidate._coefficient_body()
    assert parent.receipt_sha256 != candidate.receipt_sha256
    assert candidate.training_receipt["argument_proposal_retention"] == "overlap_dominance_v3"
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_cannot_suppress_candidates(score):
    with pytest.raises(ValueError, match="nonfinite"):
        _retain([(score, TokenSpan(0, 1)), (1., TokenSpan(2, 3))])


def test_registered_replacement_invariant():
    from core.learning.semantic_program_transducer_fitting import (
        _mention_pruning_preserves_feasible_replacement,
    )

    assert _mention_pruning_preserves_feasible_replacement()


@pytest.mark.parametrize('seed', range(20))
def test_indexed_dominance_matches_exhaustive_rule_and_order(seed):
    rng = np.random.default_rng(seed)
    candidates = [(float(rng.integers(-10, 11)), TokenSpan(int(start), int(start + width)))
                  for start, width in zip(rng.integers(0, 100, 250), rng.integers(1, 25, 250))]
    expected = []
    for score, span in sorted(candidates, key=lambda item:
            (-item[0], item[1].end-item[1].start, item[1].start, item[1].end)):
        if not any(previous >= score and span.start <= other.start and other.end <= span.end
                   for previous, other in expected):
            expected.append((score, span))
    assert _retain(candidates) == expected
    assert _retain(reversed(candidates)) == expected


def test_dominance_index_handles_empty_and_disjoint_inventories():
    assert _retain([]) == []
    candidates = [(1., TokenSpan(i*2, i*2+1)) for i in range(1000)]
    assert _retain(candidates) == candidates


def test_dual_bound_removes_only_binary_choices_that_cannot_match_incumbent():
    from scipy.sparse import csc_matrix

    objective = np.array([1., 3., 1.])
    matrix = csc_matrix([[1., 1., 0.]])
    upper = np.ones(3)
    result = _dual_bound_screen(objective, matrix, [1.], [1.], upper, 1., binary_count=2)
    assert result.tolist() == [1., 0., 1.]
    assert upper.tolist() == [1., 1., 1.]


def test_dual_screening_keeps_lower_score_when_needed_by_another_slot():
    options = ((((5., 0, TokenSpan(0, 1)), (4., 0, TokenSpan(2, 3))),
                ((1., 1, TokenSpan(0, 1)),)),)
    actual = optimize_argument_chart(options, n_inputs=2,
        contract=RegisterUseContract(1, 1, 0, 1, True), prune_dominated=True)
    assert actual[0] == 5.


@pytest.mark.parametrize("seed", range(12))
def test_screening_preserves_global_definition_and_attachment_optimum(seed):
    rng = np.random.default_rng(seed)
    definitions = (TokenSpan(20, 21), TokenSpan(22, 23))
    options, labels = [], []
    for _slot in range(3):
        candidates, names = [], []
        for register, definition in itertools.product(range(2), definitions):
            for _ in range(2):
                start = int(rng.integers(0, 8))
                candidates.append((float(rng.integers(-3, 5)), register,
                                   TokenSpan(start, start + int(rng.integers(1, 4)))))
                names.append(definition)
        options.append(tuple(candidates))
        labels.append(tuple(names))
    options, labels = (tuple(options),), (tuple(labels),)
    scores = {(register, name): float(rng.normal())
              for register, name in itertools.product(range(2), definitions)}
    contract = RegisterUseContract(0, 3, 0, 1, False)
    expected = brute(options, contract, n_inputs=2, definition_options=labels,
                     definition_scores=scores)
    actual = optimize_argument_chart(options, n_inputs=2, contract=contract,
        definition_options=labels, definition_scores=scores, prune_dominated=True)
    assert actual is not None
    assert actual[0] == pytest.approx(expected, abs=1e-8)


def test_chart_diagnostic_uses_the_same_pruning_policy():
    from core.learning.semantic_argument_chart import ScoredArgumentChart

    options = ((((5., 0, TokenSpan(0, 1)), (4., 0, TokenSpan(2, 3))),
                ((1., 1, TokenSpan(0, 1)),)),)
    chart = ScoredArgumentChart(options, n_inputs=2,
        contract=RegisterUseContract(1, 1, 0, 1, True), prune_dominated=True)
    result = chart.diagnose_target(((0, 1),))
    assert result["cause"] == "target_selected"
    assert result["target_score"] == 5.
