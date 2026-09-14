from types import SimpleNamespace

import pytest

from core.learning.semantic_argument_chart import select_operation_argument_graph


def test_complete_graph_evidence_can_defeat_first_feasible_chart():
    charts = [(SimpleNamespace(score=8.),), (SimpleNamespace(score=7.),)]
    candidates = [SimpleNamespace(score=-10.), SimpleNamespace(score=4.)]
    calls = []
    def assign(chart):
        index = next(i for i, c in enumerate(charts) if c is chart)
        calls.append(index)
        return candidates[index]
    assert select_operation_argument_graph(charts, assign, length_penalty=1.) is candidates[0]
    assert calls == [0]
    calls.clear()
    assert select_operation_argument_graph(charts, assign, length_penalty=1., joint=True) is candidates[1]
    assert calls == [0, 1]


def test_joint_selection_does_not_invent_a_feasible_graph():
    assert select_operation_argument_graph([()], lambda _: None, length_penalty=0., joint=True) is None
    with pytest.raises(ValueError, match='nonfinite'):
        select_operation_argument_graph([()], lambda _: SimpleNamespace(score=float('nan')), length_penalty=0., joint=True)


def test_bounded_assignment_receives_the_exact_incumbent_residual():
    charts = [(SimpleNamespace(score=3.),), (SimpleNamespace(score=2.),)]
    candidate = SimpleNamespace(score=5.)
    thresholds = []
    def assign(chart, minimum):
        thresholds.append(minimum)
        return candidate if chart is charts[0] else None
    assert select_operation_argument_graph(charts, lambda _: pytest.fail('unbounded call'),
        length_penalty=1., joint=True, bounded_assign=assign) is candidate
    assert thresholds == [-float('inf'), 6.]


def test_score_bound_dominates_optimal_joint_assignment():
    from core.learning.semantic_argument_chart import ScoredArgumentChart
    from core.learning.semantic_program_ir import TokenSpan
    from core.learning.semantic_program_transducer_fitting import RegisterUseContract

    a, b = TokenSpan(0, 1), TokenSpan(2, 3)
    chart = ScoredArgumentChart(
        ((((5., 0, a), (3., 1, b)), ((4., 0, a), (2., 1, b))),),
        2, RegisterUseContract(1, 1, 1, 1, True),
    )
    assert chart.score_upper_bound() == 9.
    assert chart.solve()[0] == 7.
    scored = ScoredArgumentChart(chart.options, 2, chart.contract,
        definition_scores={(0, a): 4., (0, b): 3., (1, a): -5.})
    assert scored.score_upper_bound() == 13.
