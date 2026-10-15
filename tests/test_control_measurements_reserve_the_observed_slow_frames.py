"""A fixed physical trial must admit its measurement under observed capture jitter."""
import pytest

from core.agency.time_to_observe_a_control import observed_response_seconds
from core.agency.which_one_answers_to_her import PUSH_PICTURES, _Speeds

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('slow', [.18, .25, .3, .35])
def test_a_recent_slow_capture_cannot_leave_the_reserved_push_unmeasurable(slow):
    options = dict(least_s=.45, settle_s=.2, samples=PUSH_PICTURES, span_s=.15, longest_s=1.8)
    duration = observed_response_seconds([.1]*11+[slow], **options)
    assert duration is not None and duration <= options['longest_s']
    measured = _Speeds()
    at = 0.
    while at+slow < duration-1e-9:
        at += slow
        if at >= options['settle_s']:
            measured.add('engine', 0., -60.*at, press=0., when=at)
    assert len(measured.pushes()['engine']) == 1
    assert measured.pushes()['engine'][0] == pytest.approx((0., -60.))


def test_phase_reservation_has_a_registered_measured_claim():
    from core.agency.time_to_observe_a_control import _phase_reservation_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _phase_reservation_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'new_phases_keep_their_observation_reservation')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


def test_an_observed_gap_cannot_make_the_finite_physical_budget_unbounded():
    options = dict(least_s=.45, settle_s=.2, samples=4, span_s=.15, longest_s=.8)
    assert observed_response_seconds([.1]*11+[.3], **options) is None


@pytest.mark.asyncio
async def test_a_new_counter_phase_does_not_discard_the_current_observation_reservation(monkeypatch):
    from types import SimpleNamespace

    from core.agency import trying_controls_with_a_view_of_their_response as trying
    from core.agency.giving_a_trial_time import ControlTrial
    from core.agency.playing_as_it_happens import _Run
    from core.agency.which_one_answers_to_her import WhichIsHers

    run = _Run(keys=['engine'], began=0., tried_at=0., held='', held_since=1.)
    run.intervals.extend([.1]*9+[.245, .125, .15])
    previous = ControlTrial('', 1., 1.734, 0)
    run.body_trial, run.held_trying = previous, True
    run.counter_trial = SimpleNamespace(key='engine', done=False, phase_seconds=.734, until=5.,
                                        choose=lambda *_a, **_kw: 'engine')
    leases = []
    monkeypatch.setattr(trying, 'lease_control_trial', lambda *_a, **_kw: leases.append(run.body_trial))

    async def delivered(_hands, current, _hers, key, at, *, trying):
        current.held, current.held_since, current.held_trying = key, at, trying

    await trying.try_controls(None, run, WhichIsHers(), 2., SimpleNamespace(things={}),
                              hold=delivered, seconds=.45)
    assert len(leases) == 1 and run.body_trial.until - run.body_trial.began == pytest.approx(1.18)
    assert previous.until == 1.734 and run.counter_trial.until == 5.
    measured = _Speeds()
    # The first usable observation may arrive late within the observed gap.
    # The old .734-second request supplies three of these; the new one supplies four.
    for dt in (.435, .558, .692, .845):
        if dt < run.body_trial.until - run.body_trial.began:
            measured.add('engine', 0., -60.*dt, press=2., when=2.+dt)
    assert measured.pushes()['engine'][0] == pytest.approx((0., -60.))
