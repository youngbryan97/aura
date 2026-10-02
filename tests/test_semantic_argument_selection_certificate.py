"""An independently checked feasible graph is not a claimed optimum."""

from dataclasses import replace

import numpy as np
import pytest

from core.learning.semantic_argument_chart import ScoredArgumentChart
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import RegisterUseContract


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("definitions", [False, True])
def test_selection_certificate_matches_exact_optimizer_and_all_selected_factors(seed, definitions):
    rng = np.random.default_rng(seed)
    options = tuple(tuple(tuple((float(rng.normal()), register, TokenSpan(2 * step + slot, 2 * step + slot + 1))
        for register in range(5) if register != 3 + step) for slot in range(2)) for step in range(2))
    names = tuple(tuple(tuple(TokenSpan(20 + register, 21 + register) for _score, register, _span in pool)
        for pool in node) for node in options) if definitions else None
    scores = {(register, TokenSpan(20 + register, 21 + register)): float(rng.normal())
        for register in range(5)} if definitions else None
    chart = ScoredArgumentChart(options, 3, RegisterUseContract(1, 1, 1, 1, True),
        definition_options=names, definition_scores=scores).with_conditional_choices()
    selected = []
    optimum = chart.solve(selection_observer=selected.append)
    certificate = chart.certify_selection(selected[0])
    assert certificate[0] == pytest.approx(optimum[0], abs=1e-8)
    assert certificate[1:] == optimum[1:]


def chart_for(registers, *, spans=None, distinct=False, minimum=0, definitions=None):
    if spans is None:
        spans = tuple(tuple(TokenSpan(2 * step + slot, 2 * step + slot + 1)
            for slot in range(len(node))) for step, node in enumerate(registers))
    return ScoredArgumentChart(tuple(tuple(((1., register, span),)
        for register, span in zip(node, row, strict=True)) for node, row in zip(registers, spans, strict=True)),
        2, RegisterUseContract(minimum, 4, 0, 4, distinct), definition_options=definitions)


@pytest.mark.parametrize("chart,reason", [
    (chart_for(((2,),)), "invalid typed option"),
    (chart_for(((3,), (2,))), "dependency cycle"),
    (chart_for(((0,), (1,))), "register-use bounds"),
    (chart_for(((0, 0),), distinct=True), "distinct operand"),
    (chart_for(((0,),), minimum=1), "register-use bounds"),
    (chart_for(((0, 1),), spans=((TokenSpan(0, 2), TokenSpan(1, 3)),)), "overlapping"),
    (chart_for(((0, 0),), definitions=(((TokenSpan(10, 11),), (TokenSpan(12, 13),)),)), "definition"),
])
def test_selection_certificate_rejects_every_global_constraint_violation(chart, reason):
    with pytest.raises(ValueError, match=reason):
        chart.certify_selection(tuple(tuple(0 for _slot in node) for node in chart.options))


@pytest.mark.parametrize("indices", [(), ((0,),), ((False, 0),), ((-1, 0),), ((1, 0),)])
def test_selection_certificate_rejects_missing_or_noninteger_option_identity(indices):
    with pytest.raises(ValueError):
        chart_for(((0, 1),)).certify_selection(indices)


def test_selection_certificate_does_not_search_or_certify_optimality(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "milp", lambda *_args, **_kwargs: pytest.fail("unexpected search"))
    chart = chart_for(((0, 1),))
    assert chart.certify_selection(((0, 0),))[0] == 2.
    with pytest.raises(ValueError, match="nonfinite"):
        replace(chart, definition_options=(((TokenSpan(10, 11),), (TokenSpan(12, 13),)),),
            definition_scores={(0, TokenSpan(10, 11)): float("nan"),
                               (1, TokenSpan(12, 13)): 1.}).certify_selection(((0, 0),))
