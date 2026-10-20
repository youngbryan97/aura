"""Discovery and prepared execution respect the same physical input ownership."""
from types import SimpleNamespace

import pytest

from core.agency.a_control_trial_and_its_observation_reserve import (
    _trial_reserve_invariant, acceleration_is_admitted, decide_observation_reserve,
)
from core.agency.giving_a_trial_time import ControlTrial
from core.agency.inputs_while_measuring_control import current_owned_control_trial
from tests.test_a_control_trial_preserves_its_view import _scene

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('active', [False, True])
def test_known_protection_has_one_explicit_observation_deadline(active):
    _body, hers, moves, _inputs = _scene()
    trial = ControlTrial('unknown', .8, 1.5, 2)
    run = SimpleNamespace(keys=['q'], body_trial=trial)
    decision = decide_observation_reserve(run, hers, moves, 1., .65, active=active)
    assert decision.kind == ('abort_for_protection' if active else 'protect_observation')
    assert decision.key == 'q' and decision.deadline == 1.65
    assert decision.input_owner == (('unknown', .8, 1.5, 2) if active else None)


def test_an_unsafe_experiment_without_a_measured_rescue_requests_observation():
    _body, hers, moves, inputs = _scene()
    inputs.pop('q')
    decision = decide_observation_reserve(SimpleNamespace(keys=['q']), hers, moves, 1., .65, active=False)
    assert decision.kind == 'need_observation'
    assert decision.reason == 'no_verified_control_keeps_observation'
    assert decision.key is None and decision.clearance < 0


def test_unknown_dynamics_remain_a_bounded_experiment_without_a_survival_claim():
    _body, hers, moves, _inputs = _scene()
    hers.carried = False
    decision = decide_observation_reserve(SimpleNamespace(keys=['q']), hers, moves, 1., .65, active=False)
    assert decision.kind == 'bounded_experiment' and decision.clearance is None
    assert decision.reason == 'acceleration_reserve_unavailable'


def test_fresh_admitted_acceleration_can_protect_through_a_temporary_classification_gap():
    body, hers, moves, _inputs = _scene()
    body.born = 0.
    hers.carried = False
    prediction = SimpleNamespace(selected=True, owner=(1, 0., 0.))
    hers.motion_family_reading = (.9, 1, SimpleNamespace(prediction=prediction))
    models = SimpleNamespace(selected=True, owner=prediction.owner)
    hers._hers = SimpleNamespace(intervals=SimpleNamespace(families=models))
    assert acceleration_is_admitted(hers, body, 1.)
    assert decide_observation_reserve(SimpleNamespace(keys=['q']), hers, moves, 1., .65,
                                      active=False).key == 'q'
    models.selected = False
    assert not acceleration_is_admitted(hers, body, 1.)
    models.selected, body.born = True, .2
    assert not acceleration_is_admitted(hers, body, 1.)


@pytest.mark.asyncio
@pytest.mark.parametrize('field,key,expired', [('body_trial', 'opaque-a', False),
    ('body_trial', 'opaque-a', True), ('body_trial', '', False), ('facing_trial', 'opaque-a', False)])
async def test_prepared_motion_cannot_steal_an_owned_press_or_its_neutral_observation(field, key, expired):
    from core.agency.acting_on_a_prepared_motion import act_on_prepared_motion
    from tests.test_prepared_motion_keeps_its_input_set import fixture

    run, moves, hers, raw, hands, at = fixture()
    run.held, run.held_since, run.held_trying = key, at-.1, False
    if key:
        raw.held.add(key)
    trial = ControlTrial(key, run.held_since, at-.05 if expired else at+.5,
                         hers.identification.epoch, independent=False)
    setattr(run, field, trial)
    assert current_owned_control_trial(run, hers, at) is trial
    assert not await act_on_prepared_motion(hands, run, moves, hers, at)
    assert raw.held == ({key} if key else set()) and not raw.events
    assert run.prepared_motion_binding.follower.cursor == 0


def test_an_unrelated_receipt_cannot_reserve_input_ownership():
    hers = SimpleNamespace(identification=SimpleNamespace(epoch=2))
    trial = ControlTrial('q', 1., 2., 2)
    run = SimpleNamespace(held='q', held_since=1.1, held_trying=True, body_trial=trial)
    assert current_owned_control_trial(run, hers, 1.5) is None
    run.held_since = 1.
    assert current_owned_control_trial(run, hers, 1.5) is trial
    hers.identification.epoch = 3
    assert current_owned_control_trial(run, hers, 1.5) is None


def test_trial_reserve_has_an_executable_invariant():
    assert _trial_reserve_invariant() == ()
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'discovery_records_its_observation_deadline_and_owner')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


@pytest.mark.asyncio
async def test_the_actual_discovery_executor_records_and_delivers_protection_without_a_new_trial(monkeypatch):
    from core.agency.playing_as_it_happens import _Run
    from core.agency.trying_controls_with_a_view_of_their_response import try_controls

    _body, hers, moves, _inputs = _scene()
    hers.kind, hers.keys_known = 1, lambda _keys: False
    moves.things = {_body.number: _body}
    run = _Run(keys=['q'], began=0.)
    monkeypatch.setattr('core.agency.trying_controls_with_a_view_of_their_response.time',
                        SimpleNamespace(monotonic=lambda: 1.1))
    issued = []
    async def hold(_hands, current, _hers, key, _at, *, trying):
        issued.append((key, trying))
        current.held, current.held_since = key, 1.1
    await try_controls(None, run, hers, 1., moves, hold=hold, seconds=.45)
    assert issued == [('q', False)]
    assert run.body_trial is None and run.trial_decision.kind == 'protect_observation'
    assert run.tried_at == 1.1 and run.trying == 0
