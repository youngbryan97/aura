"""Favored actions cannot interrupt or indefinitely defer isolated control discovery."""
from types import SimpleNamespace

import pytest

from core.agency import playing_as_it_happens as play
from core.agency.giving_a_trial_time import ControlTrial
from core.agency.inputs_while_measuring_control import body_probe_is_active
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


def _setup(monkeypatch, *, trying=4, key='a'):
    now, actions = [10.7], []
    monkeypatch.setattr(play.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(play, 'a_key_that_pays', lambda _keys: 'b')
    run = play._Run(keys=['a', 'b'], began=0., trying=trying, held=key, held_since=10.,
                    held_trying=True, tried_at=10., body_trial=ControlTrial(key, 10., 11.2, 0))
    hers = WhichIsHers()
    hers.holding(key, 10., trying=True)
    choosing = SimpleNamespace(mine=None, ways={}, lifts=[])
    moves = SimpleNamespace(seeing=True, things={})

    class Hands:
        async def down(self, key):
            actions.append((now[0], 'down', key))

        async def up(self, key):
            actions.append((now[0], 'up', key))

        async def tap(self, key):
            actions.append((now[0], 'tap', key))

    return now, run, hers, choosing, moves, actions, Hands()


@pytest.mark.asyncio
@pytest.mark.parametrize('trying,key', [(4, 'a'), (7, 'b'), (8, '')])
async def test_the_actual_actor_preserves_the_active_trial_and_its_last_rest(monkeypatch, trying, key):
    now, run, hers, choosing, moves, actions, hands = _setup(monkeypatch, trying=trying, key=key)
    for at in (10.7, 10.8, 11.1):
        now[0] = at
        await play._act(hands, run, moves, hers, None, choosing, at)
    assert not actions and run.body_trial.until == 11.2
    assert run.trying == trying and run.tried_at == 10.


@pytest.mark.asyncio
async def test_the_second_independent_cycle_precedes_a_favored_tap(monkeypatch):
    now, run, hers, choosing, moves, actions, hands = _setup(monkeypatch, key='')
    now[0] = 11.3
    await play._act(hands, run, moves, hers, None, choosing, now[0])
    assert run.trying == 5
    assert actions == [(11.3, 'down', 'a')]
    assert run.body_trial.key == 'a' and run.body_trial.until > now[0]


@pytest.mark.asyncio
async def test_the_favored_action_is_available_after_the_finite_discovery_pass(monkeypatch):
    now, run, hers, choosing, moves, actions, hands = _setup(monkeypatch, trying=8, key='')
    now[0] = 11.3
    await play._act(hands, run, moves, hers, None, choosing, now[0])
    assert actions == [(11.3, 'tap', 'b')]
    assert run.tried_at == 10. and run.last_payoff_trial_at == 11.3


@pytest.mark.asyncio
async def test_repeated_favored_actions_do_not_postpone_control_rechecking(monkeypatch):
    now, run, hers, choosing, moves, actions, hands = _setup(monkeypatch, trying=8, key='')
    now[0] = 11.3
    while now[0] < 10. + play.RECHECK_CONTROLS_S:
        await play._act(hands, run, moves, hers, None, choosing, now[0])
        assert run.tried_at == 10.
        now[0] += .6
    assert all(kind == 'tap' and key == 'b' for _at, kind, key in actions)
    now[0] = 10. + play.RECHECK_CONTROLS_S + .1
    before = len(actions)
    await play._act(hands, run, moves, hers, None, choosing, now[0])
    assert run.control_probe_retries == 1 and run.trying == 1 and run.held_trying
    assert actions[before:] == [(now[0], 'down', 'a')]
    assert run.tried_at == now[0] and hers.identification.epoch == 1


@pytest.mark.asyncio
async def test_a_favored_tap_keeps_its_own_spacing(monkeypatch):
    now, run, hers, choosing, moves, actions, hands = _setup(monkeypatch, trying=8, key='')
    for at in (11.3, 11.4, 11.5, 11.6):
        now[0] = at
        await play._act(hands, run, moves, hers, None, choosing, at)
    assert actions == [(11.3, 'tap', 'b')]
    assert run.tried_at == 10.


def test_a_changed_receipt_or_experiment_cannot_extend_the_last_trial(monkeypatch):
    _now, run, hers, _choosing, _moves, _actions, _hands = _setup(monkeypatch, trying=8)
    assert body_probe_is_active(run, hers, pointer_steps=6, at=10.7)
    run.held_since = 10.1
    assert not body_probe_is_active(run, hers, pointer_steps=6, at=10.7)
    run.held_since = 10.
    hers.identification.reset('different measured body')
    assert not body_probe_is_active(run, hers, pointer_steps=6, at=10.7)


@pytest.mark.asyncio
async def test_an_explicit_current_screen_request_can_preempt_discovery(monkeypatch):
    now, run, hers, choosing, moves, actions, hands = _setup(monkeypatch)

    class ScreenRequest:
        def asks(self, at):
            return at == now[0]

        async def press(self, hands, say):
            await hands.tap('urgent')

    run.shown = ScreenRequest()
    await play._act(hands, run, moves, hers, None, choosing, now[0])
    assert actions == [(10.7, 'up', 'a'), (10.7, 'tap', 'urgent')]


def test_an_identified_body_with_known_controls_does_not_start_another_discovery_pass():
    hers = SimpleNamespace(number=7, kind=2, follows_pointer=False, keys_known=lambda _keys: True)
    run = play._Run(keys=['a', 'b'], began=0., trying=0)
    assert not play._still_trying_keys(run, hers, SimpleNamespace(mine=object()), False, at=1.)


@pytest.mark.asyncio
async def test_no_moving_way_does_not_reenter_a_finished_probe_until_rechecking_is_due(monkeypatch):
    now, run, hers, choosing, moves, actions, hands = _setup(monkeypatch, trying=3,key='')
    body = SimpleNamespace(number=7,born=1.)
    hers.number,hers.kind = 7,2
    hers.keys_known = lambda _keys: True
    hers.way_of = lambda _key: (0.,0.)
    moves.things,run.things,choosing.mine = {7:body},{7:body},body
    run.body_trial = None
    async def unexpected(*_args, **_kwargs):
        pytest.fail('finished discovery was reentered without a new experiment')
    monkeypatch.setattr(play,'_try_the_keys',unexpected)
    monkeypatch.setattr(play,'_a_facing_trial',lambda *_a,**_k: _false())
    async def nothing(*_args,**_kwargs):
        return None
    monkeypatch.setattr(play,'_fire_while_finding_herself',nothing)
    monkeypatch.setattr(play,'_try_the_pointer',nothing)
    run.pointed = 100
    now[0] = 11.3
    await play._act(hands,run,moves,hers,None,choosing,now[0])
    assert actions == [(11.3,'tap','b')]
    assert run.trying == 3 and run.control_probe_retries == 0 and hers.number == 7
    now[0] = 10.+play.RECHECK_CONTROLS_S+.1
    issued=[]
    async def discovery(*_args,**_kwargs):
        issued.append(hers.identification.epoch)
    monkeypatch.setattr(play,'_try_the_keys',discovery)
    await play._act(hands,run,moves,hers,None,choosing,now[0])
    assert issued == [1] and hers.number is None and run.control_probe_retries == 1


async def _false():
    return False


def test_the_discovery_phase_has_a_registered_measured_claim():
    from core.agency.inputs_while_measuring_control import _body_probe_phase_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _body_probe_phase_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'control_discovery_keeps_its_finite_observation_phase')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
