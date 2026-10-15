"""The next experiment uses observed motion and room, without assuming key meanings."""
from types import SimpleNamespace

import pytest

from core.agency.trials_with_room_to_observe import _trial_priority_invariant, trials_with_room
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


def _scene(axis=0, direction=-1):
    position, velocity = [120., 120.], [0., 0.]
    position[axis] = 60. if direction < 0 else 180.
    velocity[axis] = direction * 40.
    body = SimpleNamespace(number=1, kind=1, born=0., seen=10., x=position[0], y=position[1],
                           vx=velocity[0], vy=velocity[1], w=14., h=14., motion_evidence='direct')
    moves = SimpleNamespace(things={1: body}, shape=(240, 240))
    hers = WhichIsHers()
    hers.number, hers.kind, hers.carried = 1, 1, True
    for key, began, push in [('q', 1., direction * 110.), ('r', 3., -direction * 110.), ('', 5., 0.)]:
        for dt in (.2, .3, .4, .5):
            value = [0., 0.]
            value[axis] = dt * push
            hers._hers.add(key, *value, press=began, when=began + dt)
    return hers, moves


@pytest.mark.parametrize('axis,direction', [(0, -1), (0, 1), (1, -1), (1, 1)])
def test_one_pulse_ranks_the_next_experiment_without_certifying_it(axis, direction):
    hers, moves = _scene(axis, direction)
    assert trials_with_room(['q', 'r'], hers, moves, 10., .65) == ['r', 'q']
    assert hers.number == 1 and not hers.identification.receipts
    assert hers.push_of('q') is hers.push_of('r') is None


@pytest.mark.parametrize('missing', ['rest', 'pulse', 'fresh', 'carried', 'time', 'distinct', 'interrupted'])
def test_an_unmeasured_projection_keeps_the_callers_trial_order(missing):
    hers, moves = _scene()
    record = hers._hers
    at = 10.
    if missing == 'rest':
        record.timed.pop(('', 5.))
    elif missing == 'pulse':
        record.timed.pop(('r', 3.))
    elif missing == 'fresh':
        moves.things[1].seen = 9.9
    elif missing == 'carried':
        hers.carried = False
    elif missing == 'time':
        at = float('nan')
    elif missing == 'distinct':
        record.timed[('r', 3.)][-1] = record.timed[('r', 3.)][-2]
    else:
        record.interrupted.add(('r', 3.))
    assert trials_with_room(['q', 'r'], hers, moves, at, .65) == ['q', 'r']


@pytest.mark.asyncio
@pytest.mark.parametrize('axis,direction', [(0, -1), (0, 1), (1, -1), (1, 1)])
async def test_the_trial_executor_uses_room_to_break_a_finite_budget_tie(monkeypatch, axis, direction):
    from core.agency import trying_controls_with_a_view_of_their_response as trying
    from core.agency.calibration_for_a_current_body import calibration_owner
    from core.agency.countering_motion_as_it_appears import CounteringMotion
    from core.agency.playing_as_it_happens import _Run
    from core.agency.unknown_controls_get_their_turn import UnknownControlTrials

    hers, moves = _scene(axis, direction)
    run = _Run(keys=['q', 'r'], began=0., held='', held_since=9.3)
    run.tried_at = 9.
    run.intervals.extend([.1, .1, .1])
    run.counter_trial = CounteringMotion(done=True)
    owner = calibration_owner(hers, moves.things[1])
    run._unknown_control_trials = {owner: UnknownControlTrials(owner, {'q': 1, 'r': 1})}
    leases, actions = [], []
    monkeypatch.setattr(trying, 'lease_control_trial', lambda *args, **kwargs: leases.append(run.body_trial))

    async def delivered(_hands, run, _hers, key, at, *, trying):
        actions.append(key)
        run.held, run.held_since, run.held_trying = key, at, trying

    await trying.try_controls(None, run, hers, 10., moves, hold=delivered, seconds=.45)
    assert actions == ['r'] and len(leases) == 1
    assert leases[0].until > 10. and leases[0].until <= 11.8
    assert run._unknown_control_trials[owner].issued == {'q': 1, 'r': 2}
    assert hers.push_of('r') is None and not hers.identification.receipts
    assert not run.held_trying and not leases[0].independent
    assert leases[0].owner == owner


def test_trial_priority_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _trial_priority_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'trial_priority_cannot_certify_a_response')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
