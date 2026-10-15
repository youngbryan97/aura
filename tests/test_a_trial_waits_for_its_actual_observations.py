"""The observation estimate can be late without spending a pulse on three samples."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.agency.enough_observations_for_a_trial import (
    _trial_observation_goal_invariant,
    trial_has_its_observations,
)
from core.agency.giving_a_trial_time import ControlTrial
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


def _owned():
    hers = WhichIsHers()
    hers.number, hers.kind, hers.carried = 1, 1, True
    body = SimpleNamespace(number=1, born=0., kind=1, seen=10., x=120., y=120.,
                           vx=0., vy=0., w=14., h=14., motion_evidence='direct')
    return hers, SimpleNamespace(things={1: body}, shape=(240, 240))


@pytest.mark.asyncio
async def test_the_actual_executor_waits_for_a_fourth_late_sample_then_physically_releases(monkeypatch):
    from core.agency import trying_controls_with_a_view_of_their_response as trying
    from core.agency.countering_motion_as_it_appears import CounteringMotion
    from core.agency.inputs_while_measuring_control import hold_control
    from core.agency.playing_as_it_happens import _Run

    hers, moves = _owned()
    now, actions, leases = [10.], [], []
    run = _Run(keys=['q'], began=0., tried_at=0.)
    run.intervals.extend([.15]*12)
    run.counter_trial = CounteringMotion(done=True)
    monkeypatch.setattr(trying, 'lease_control_trial', lambda *_a, **_k: leases.append(run.body_trial))

    class Hands:
        async def down(self, key):
            actions.append(('down', key))

        async def up(self, key):
            actions.append(('up', key))

    async def actual_hold(hands, run, hers, key, at, *, trying):
        await hold_control(hands, run, hers, key, at, trying=trying, now=lambda: now[0])

    await trying.try_controls(Hands(), run, hers, now[0], moves, hold=actual_hold, seconds=.45)
    pulse = run.body_trial
    assert pulse.expected_until == pytest.approx(10.8) and pulse.until == pytest.approx(11.8)
    for at in (10.45, 10.58, 10.70):
        hers._hers.add('q', 100.*at, 0., press=pulse.began, when=at)
    now[0] = moves.things[1].seen = 10.9
    await trying.try_controls(Hands(), run, hers, now[0], moves, hold=actual_hold, seconds=.45)
    assert run.body_trial is pulse and actions == [('down', 'q')]
    assert pulse.until == 11.8
    now[0] = moves.things[1].seen = 11.05
    hers._hers.add('q', 100.*now[0], 0., press=pulse.began, when=now[0])
    await trying.try_controls(Hands(), run, hers, now[0], moves, hold=actual_hold, seconds=.45)
    rest = run.body_trial
    assert actions == [('down', 'q'), ('up', 'q')]
    assert not rest.key and not run.held_trying and not rest.independent
    assert rest.began == now[0] and rest.expected_until == pytest.approx(11.85)
    assert rest.until == pytest.approx(12.85) and len(leases) == 1
    assert hers.push_of('q') is None  # one physical pulse remains one trial
    assert not hers._by_thing and not hers.identification.receipts
    for at in (11.3, 11.4, 11.5, 11.6):
        hers._hers.add('', 0., 0., press=rest.began, when=at)
    now[0] = moves.things[1].seen = 11.6
    await trying.try_controls(Hands(), run, hers, now[0], moves, hold=actual_hold, seconds=.45)
    assert actions == [('down', 'q'), ('up', 'q'), ('down', 'q')]
    assert run._unknown_control_trials[rest.owner].issued == {'q': 2}
    assert rest.until == pytest.approx(12.85)
    await actual_hold(Hands(), run, hers, '', now[0], trying=False)


@pytest.mark.parametrize('invalid', ['owner', 'independent', 'old', 'nonfinite', 'duplicate', 'span', 'interrupted'])
def test_only_current_settled_finite_distinct_observations_can_end_a_trial(invalid):
    hers, moves = _owned()
    trial = ControlTrial('q', 10., 11.8, 0, (0, 1, 0.), False, 10.8)
    values = [(t, 100.*t, 0.) for t in (10.3, 10.4, 10.5, 10.6)]
    if invalid == 'owner':
        trial = replace(trial, owner=(1, 1, 0.))
    elif invalid == 'independent':
        trial = replace(trial, independent=True)
    elif invalid == 'old':
        values[0] = (9.9, 990., 0.)
    elif invalid == 'nonfinite':
        values[-1] = (10.6, float('nan'), 0.)
    elif invalid == 'duplicate':
        values[-1] = values[-2]
    elif invalid == 'span':
        values = [(t, 1., 0.) for t in (10.3, 10.31, 10.32, 10.33)]
    else:
        hers._hers.interrupted.add(('q', 10.))
    hers._hers.timed[('q', 10.)] = values
    assert not trial_has_its_observations(trial, hers, moves, 10.7)


def test_observation_acquisition_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _trial_observation_goal_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'trial_acquisition_requires_owned_samples')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
