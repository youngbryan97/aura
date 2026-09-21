"""Optional graph search phases cannot escape the caller's declared allowance."""

from types import SimpleNamespace

import pytest
import scipy.optimize

from core.learning import semantic_argument_optimization as optimizer
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import RegisterUseContract


def run(**kwargs):
    return optimizer.optimize_argument_chart(((((1., 0, TokenSpan(0, 1)),),),),
        n_inputs=1, contract=RegisterUseContract(1, 1, 0, 1, True),
        prune_dominated=True, **kwargs)


def test_shortlist_screen_and_final_solver_share_one_allowance(monkeypatch):
    clock = [0.]
    integer, relaxation = [], []
    original = scipy.optimize.milp
    monkeypatch.setattr(optimizer.time, 'monotonic', lambda: clock[0])

    def milp(*args, **kwargs):
        integer.append(kwargs['options']['time_limit'])
        clock[0] += 1.
        return original(*args, **kwargs)

    def linprog(*args, **kwargs):
        relaxation.append(kwargs['options']['time_limit'])
        clock[0] += 2.
        return SimpleNamespace(success=False)

    monkeypatch.setattr(scipy.optimize, 'milp', milp)
    monkeypatch.setattr(scipy.optimize, 'linprog', linprog)
    assert run(time_limit_s=10.)[1] == ((0,),)
    assert integer == [10., 7.]
    assert relaxation == [9.]


def test_expired_optional_screen_never_certifies_a_shortlist_optimum(monkeypatch):
    clock = [0.]
    calls = []
    original = scipy.optimize.milp
    monkeypatch.setattr(optimizer.time, 'monotonic', lambda: clock[0])

    def milp(*args, **kwargs):
        calls.append('integer')
        return original(*args, **kwargs)

    def linprog(*args, **kwargs):
        calls.append('relaxation')
        clock[0] += kwargs['options']['time_limit']
        return SimpleNamespace(success=False)

    monkeypatch.setattr(scipy.optimize, 'milp', milp)
    monkeypatch.setattr(scipy.optimize, 'linprog', linprog)
    with pytest.raises(optimizer.ArgumentOptimizationIncompleteError, match='budget_exhausted'):
        run(time_limit_s=2.)
    assert calls == ['integer', 'relaxation']


def test_no_declared_allowance_does_not_invent_a_timeout(monkeypatch):
    calls = []
    original = scipy.optimize.milp

    def milp(*args, **kwargs):
        calls.append(kwargs['options'])
        return original(*args, **kwargs)

    def linprog(*args, **kwargs):
        calls.append(kwargs['options'])
        return SimpleNamespace(success=False)

    monkeypatch.setattr(scipy.optimize, 'milp', milp)
    monkeypatch.setattr(scipy.optimize, 'linprog', linprog)
    assert run()[1] == ((0,),)
    assert len(calls) == 3 and all('time_limit' not in options for options in calls)
