"""Optional correspondence repays measured work before delaying another picture."""
from types import SimpleNamespace

import pytest

from core.perception import source_geometry_in_current_pixels as geometry
from core.perception import play_observation_without_holding_the_loop as observation
from core.runtime.optional_work_shares_time import OptionalWorkShare
from tests.test_fresh_source_geometry_reaches_ordinary_observation import scene, _state

pytestmark = pytest.mark.unit


def test_optional_refitting_must_repay_its_cost_between_pictures(monkeypatch):
    worker = geometry.SourceGeometryInCurrentPixels()
    worker.records['retained search hint'] = object()
    clock, calls = [0.], []
    monkeypatch.setattr(geometry.time, 'perf_counter', lambda: clock[0])

    def measure(*args, **kwargs):
        calls.append(kwargs)
        clock[0] += .09
        return ('current verified fit',), ()

    monkeypatch.setattr(worker, 'measure', measure)
    outputs = [worker.measure_if_budget(None, float(n), {}, (), (), (), essential_seconds=.09)
               for n in range(9)]
    assert len(calls) == 3
    assert all(call == {'one_position': True} for call in calls)
    assert [n for n, (frames, _) in enumerate(outputs) if frames] == [0, 3, 6]
    assert all(status == ('fresh_reference_work_deferred',) for frames, status in outputs if not frames)


def test_unmeasured_mandatory_cost_cannot_buy_optional_work(monkeypatch):
    worker = geometry.SourceGeometryInCurrentPixels()
    worker.records['hint'] = object()
    monkeypatch.setattr(worker, 'measure', lambda *args, **kwargs: pytest.fail('no measured budget'))
    for cost in (-1., float('inf'), float('nan'), True):
        frames, status = worker.measure_if_budget(None, 1., {}, (), (), (), essential_seconds=cost)
        assert not frames and status == ('fresh_reference_work_deferred',)


def test_a_failed_optional_fit_still_spends_its_observed_cost(monkeypatch):
    worker = geometry.SourceGeometryInCurrentPixels()
    worker.records['hint'] = object()
    clock = [0.]
    monkeypatch.setattr(geometry.time, 'perf_counter', lambda: clock[0])

    def failed(*args, **kwargs):
        clock[0] += .09
        raise RuntimeError('fixture failure')

    monkeypatch.setattr(worker, 'measure', failed)
    with pytest.raises(RuntimeError, match='fixture failure'):
        worker.measure_if_budget(None, 1., {}, (), (), (), essential_seconds=.09)
    monkeypatch.setattr(worker, 'measure', lambda *args, **kwargs: pytest.fail('debt still outstanding'))
    assert worker.measure_if_budget(None, 2., {}, (), (), (), essential_seconds=.09)[0] == ()


def test_the_ordinary_worker_supplies_its_measured_mandatory_duration(monkeypatch):
    clock, costs = [2.], []
    monkeypatch.setattr(observation.time, 'perf_counter', lambda: clock[0])

    class Eyes:
        def see(self, picture, at, **kwargs):
            clock[0] += .06
            return []

    def detached(eyes):
        clock[0] += .03
        return {}

    def optional(*args, essential_seconds):
        costs.append(essential_seconds)
        return (), ('fresh_reference_work_deferred',)

    monkeypatch.setattr(observation, '_detached_state', detached)
    events, state = observation._seen(Eyes(), None, 1.,
        fresh_references=SimpleNamespace(measure_if_budget=optional))
    assert events == [] and costs == pytest.approx([.09])
    assert state['fresh_source_reference_frames'] == ()
    assert state['fresh_source_reference_status'] == ('fresh_reference_work_deferred',)


def test_the_work_budget_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert geometry._optional_reference_share_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'optional_geometry_repays_observation_time')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


@pytest.mark.parametrize('costs', [[.003, .007, .2, .01], [.5], [.03, .15, .03]])
def test_variable_work_spends_no_more_than_earned_credit_and_one_final_operation(costs):
    quota = OptionalWorkShare()
    essential, spent = 0., 0.
    for n in range(120):
        duration = (.01, .06, .2)[n % 3]
        essential += duration
        if quota.reserve(duration):
            # All earlier overspend must be repaid before another operation.
            assert spent <= essential/3 + 1e-12
            cost = costs[n % len(costs)]
            quota.spent(cost)
            spent += cost
            assert spent <= essential/3 + cost + 1e-12
        assert quota.credit <= .1


def test_a_single_stall_cannot_bank_unbounded_work_or_forgive_an_unknown_cost():
    quota = OptionalWorkShare()
    assert quota.reserve(60.) and quota.credit == .1
    with pytest.raises(ValueError, match='unmeasured'):
        quota.spent(float('nan'))
    assert not quota.reserve(60.) and not quota.active
    with pytest.raises(ValueError, match='reservation'):
        quota.spent(0.)


def test_a_budgeted_fit_still_requires_the_current_captured_material(scene):
    proposal, table, _, current = scene
    worker = geometry.SourceGeometryInCurrentPixels()
    frames, _ = worker.measure_if_budget(current, 2., _state(current, 2.), (proposal,), (table,),
                                         essential_seconds=.3)
    assert frames and frames[0].pixels.at == 2.
    absent = current * 0
    frames, _ = worker.measure_if_budget(absent, 3., _state(absent, 3.), (), (table,), essential_seconds=.3)
    assert not frames


@pytest.mark.parametrize('settings', [dict(fraction=0.), dict(fraction=True), dict(credit_cap_s=2.),
                                     dict(admission_s=-1.), dict(fraction=float('nan'))])
def test_invalid_budget_policy_is_refused(settings):
    with pytest.raises(ValueError):
        OptionalWorkShare(**settings)
