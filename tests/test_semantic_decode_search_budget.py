"""Offline search allowances reach the solver without changing serving defaults."""

from types import SimpleNamespace

import pytest

from core.learning import semantic_program_compositional_transducer as transducer
from core.learning.semantic_argument_chart import ScoredArgumentChart
from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
from tests.test_semantic_relation_graph_learning import model_examples


@pytest.fixture(scope='module')
def fixture():
    model, examples = model_examples()
    item = examples[0]
    kwargs = dict(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=model.model_basis_sha256)
    return model, kwargs


@pytest.mark.parametrize('joint', [False, True])
def test_decode_passes_remaining_allowance_and_does_not_accept_unproven_result(fixture, monkeypatch, joint):
    model, kwargs = fixture
    if joint:
        model = model.with_joint_operation_argument_scores()
    seen = []
    def assign(**kwargs):
        seen.append(kwargs['time_limit_s'])
        raise ArgumentOptimizationIncompleteError('solver_time_limit')
    monkeypatch.setattr(transducer, '_assign_typed_arguments', assign)
    outcome = model.decode(**kwargs, search_time_limit_s=3.)
    assert seen and 0 < seen[0] <= 3.
    assert outcome.ir is None and outcome.refusal == 'solver_time_limit'
    assert outcome.search_interrupted is True


def test_real_chart_solver_receives_allowance_and_preserves_completed_answer(fixture, monkeypatch):
    model, kwargs = fixture
    expected = model.decode(**kwargs)
    original = ScoredArgumentChart.solve
    seen = []
    def solve(self, **kwargs):
        seen.append(kwargs.get('time_limit_s'))
        return original(self, **kwargs)
    monkeypatch.setattr(ScoredArgumentChart, 'solve', solve)
    outcome = model.decode(**kwargs, search_time_limit_s=10.)
    assert seen and all(value is not None and 0 < value <= 10. for value in seen)
    assert outcome.ir == expected.ir
    assert outcome.pointer_scores == expected.pointer_scores
    assert outcome.search_interrupted is False


def test_default_serving_search_remains_unbounded_by_diagnostic_clock(fixture, monkeypatch):
    model, kwargs = fixture
    seen = []
    def assign(**kwargs):
        seen.append(kwargs['time_limit_s'])
        return None
    monkeypatch.setattr(transducer, '_assign_typed_arguments', assign)
    model.decode(**kwargs)
    assert seen and all(value is None for value in seen)


def test_expired_decode_budget_is_not_graph_infeasibility(fixture, monkeypatch):
    model, kwargs = fixture
    ticks = iter((0., 2.))
    monkeypatch.setattr(transducer, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    outcome = model.decode(**kwargs, search_time_limit_s=1.)
    assert outcome.ir is None and outcome.refusal == 'decode_search_budget_exhausted'
    assert outcome.search_interrupted is True


@pytest.mark.parametrize('value', [0, -1, True, float('inf'), float('nan'), '3'])
def test_invalid_decode_allowance(fixture, value):
    model, kwargs = fixture
    with pytest.raises(ValueError, match='positive and finite'):
        model.decode(**kwargs, search_time_limit_s=value)
