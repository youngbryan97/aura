"""Compare the lazy grammar search with exhaustive reference enumeration."""

import itertools
import json
import math
import random
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_operation_search import OperationChartSearch, OperationSearchIncompleteError
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import _OperationNode, _operation_nodes


def node(start, end, score, operation="add"):
    return _OperationNode(TokenSpan(start, end), operation, score, score, 1.)


def key(chart):
    return tuple((n.span.start, n.span.end, n.operation) for n in chart)


def score(chart, penalty):
    return sum((Fraction(float(n.score)) for n in chart), Fraction(0)) - Fraction(penalty) * len(chart)


def brute(nodes, max_steps):
    unique = {}
    for n in nodes:
        k = key([n])
        if k not in unique or n.score > unique[k].score:
            unique[k] = n
    ordered = sorted(unique.values(), key=lambda n: key([n]))
    return [chart for size in range(1, max_steps + 1) for chart in itertools.combinations(ordered, size)
            if all(a.span.end <= b.span.start for a, b in zip(chart, chart[1:]))]


@pytest.mark.parametrize("seed", range(12))
def test_lazy_search_matches_every_brute_force_chart_and_bounds_the_remainder(seed):
    rng = random.Random(seed)
    nodes = [node(start, start + rng.randrange(1, 4), rng.uniform(-9, 9), rng.choice(["add", "subtract"]))
             for start in range(7)]
    nodes += [node(0, 2, 2.), node(0, 2, -3.)]
    penalty = rng.uniform(-3, 3)
    expected = {key(c): score(c, penalty) for c in brute(nodes, 4)}
    search = OperationChartSearch(nodes, max_steps=4, length_penalty=penalty)
    previous = None
    while expected:
        assert Fraction(search.remaining_operation_score_upper_bound) >= max(expected.values())
        chart = next(search)
        actual = score(chart, penalty)
        assert actual == expected.pop(key(chart))
        assert previous is None or previous >= actual
        previous = actual
    assert list(search) == []
    assert search.complete and search.remaining_operation_score_upper_bound == -math.inf


def test_feasibility_is_checked_on_whole_charts_not_required_of_prefixes():
    nodes = [node(i, i + 1, -i) for i in range(6)]
    feasible = lambda c: len(c) == 3 and c[-1].span.start == 5
    search = OperationChartSearch(nodes, max_steps=3, length_penalty=0, feasible=feasible)
    assert {key(c) for c in search} == {key(c) for c in brute(nodes, 3) if feasible(c)}
    assert search.complete


def test_interruption_keeps_the_frontier_and_never_claims_complete():
    nodes = [node(i, i + 1, -i) for i in range(20)]
    search = OperationChartSearch(nodes, max_steps=1, length_penalty=0, max_expansions=8)
    seen = []
    with pytest.raises(OperationSearchIncompleteError, match="operation_search_incomplete"):
        # The search raises long before this bound; the bound is so a search
        # that never raises is a failed test, not a hung one.
        for _ in range(10_000):
            seen.append(next(search))
    assert not search.complete and search.remaining_operation_score_upper_bound > -math.inf
    search.max_expansions = None
    seen.extend(search)
    assert {key(c) for c in seen} == {key(c) for c in brute(nodes, 1)}
    assert len(seen) == 20 and search.complete


def test_exact_bound_survives_large_cancellation_and_penalty_rounding():
    nodes = [node(0, 1, 1e16), node(1, 2, -1e16), node(2, 3, .1)]
    search = OperationChartSearch(nodes, max_steps=3, length_penalty=.1)
    remaining = {key(c): score(c, .1) for c in brute(nodes, 3)}
    for _ in range(len(remaining)):
        assert Fraction(search.remaining_operation_score_upper_bound) >= max(remaining.values())
        remaining.pop(key(next(search)))
    assert list(search) == []


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"max_steps": True},
                                    {"length_penalty": float("nan")}, {"max_expansions": 0}])
def test_invalid_search_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError):
        OperationChartSearch([], **dict({"max_steps": 1, "length_penalty": 0}, **kwargs))


def test_complete_inventory_retains_more_than_256_spans_and_all_labels():
    spans = [TokenSpan(i, i + 1) for i in range(300)]
    class Pointer:
        def decode_candidates(self, hidden, *, limit, max_span_tokens):
            return [(s, -float(i)) for i, s in enumerate(spans[:limit])]
    classifier = SimpleNamespace(labels=("add", "subtract"), modes=(),
        predict_probabilities=lambda _: np.array([.9, .1]))
    arguments = dict(pointer=Pointer(), classifier=classifier, hidden=np.zeros((300, 1)), input_spans=(),
                     max_span_tokens=1, hidden_channels=(), hidden_channel_widths=(), label_limit=2)
    assert len(_operation_nodes(**arguments)) == 512
    complete = _operation_nodes(**arguments, complete_inventory=True)
    assert len(complete) == 600
    assert {(n.span.start, n.operation) for n in complete} == {
        (i, op) for i in range(300) for op in classifier.labels}


def test_complete_search_roundtrip_preserves_learned_coefficients():
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict

    path = Path(__file__).parents[1] / "artifacts/rlc/semantic_program_27b_frozen_path_v1/transducer.json"
    parent = compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))
    candidate = parent.with_complete_operation_search(max_expansions=1000)
    body = candidate.to_dict()
    original = parent.to_dict()
    assert {k: v for k, v in body.items() if k != "training_receipt"} == {
        k: v for k, v in original.items() if k != "training_receipt"}
    assert candidate.receipt_sha256 != parent.receipt_sha256
    assert candidate.training_receipt["operation_assignment_policy"] == "joint_factor_score_v2"
    assert candidate.training_receipt["operation_label_limit"] == len(parent.operation_head.labels)
    assert compositional_semantic_program_transducer_from_dict(body).receipt_sha256 == candidate.receipt_sha256
    with pytest.raises(ValueError):
        parent.with_complete_operation_search(max_expansions=-1)


def test_joint_selection_reaches_an_answer_below_the_old_top_sixteen():
    from core.learning.semantic_argument_chart import select_operation_argument_graph

    nodes = [node(i, i + 1, -float(i)) for i in range(25)]
    search = OperationChartSearch(nodes, max_steps=1, length_penalty=0)
    result = select_operation_argument_graph(search,
        lambda chart: SimpleNamespace(score=100. if chart[0].span.start == 24 else 0., chart=chart),
        length_penalty=0, joint=True)
    assert result.chart[0].span.start == 24
    assert search.complete and search.yielded == 25


def test_transducer_decode_preserves_incomplete_search_instead_of_emitting_an_unproved_winner():
    from core.learning.semantic_program_compositional_transducer import fit_compositional_semantic_program_transducer
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    model = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    model = model.with_complete_operation_search(max_expansions=1)
    item = examples[0]
    outcome = model.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=model.model_basis_sha256)
    assert outcome.ir is None
    assert "operation_search_incomplete" in outcome.refusal
