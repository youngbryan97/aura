"""Unavailable control responses cannot starve the other current inputs."""
from __future__ import annotations

import asyncio
import math
import time
from collections import Counter
from types import SimpleNamespace

import pytest

from core.agency.a_trial_has_a_physical_end import cancel_trial_deadline
from core.agency.calibration_for_a_current_body import (
    MAX_CALIBRATION_CONTROLS,
    MAX_CALIBRATION_OWNERS,
)
from core.agency.countering_motion_as_it_appears import CounteringMotion
from core.agency.inputs_while_measuring_control import hold_control
from core.agency.playing_as_it_happens import _Run
from core.agency.trying_controls_with_a_view_of_their_response import try_controls
from core.agency.unknown_controls_get_their_turn import (
    PRESSES_PER_CONTROL,
    _unknown_control_fairness_invariant,
    next_unknown_control,
    unknown_control_trials_finished,
)
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


def _owner():
    return (SimpleNamespace(keys=['opaque-a', 'opaque-b', 'opaque-c']),
            SimpleNamespace(number=7, identification=SimpleNamespace(epoch=2)),
            SimpleNamespace(number=7, born=1.))


def test_unknown_control_fairness_has_a_registered_invariant():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _unknown_control_fairness_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'unknown_controls_share_a_finite_current_body_budget')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())


def test_removed_readded_and_reordered_controls_keep_their_issued_counts():
    run, hers, mine = _owner()
    assert next_unknown_control(run, hers, mine, run.keys) == 'opaque-a'
    assert next_unknown_control(run, hers, mine, ['opaque-a']) == 'opaque-a'
    run.keys = ['opaque-c', 'opaque-b']
    assert next_unknown_control(run, hers, mine, ['opaque-a']) is None
    assert next_unknown_control(run, hers, mine, run.keys) == 'opaque-c'
    run.keys = ['opaque-b', 'opaque-a', 'opaque-c']
    assert next_unknown_control(run, hers, mine, run.keys) == 'opaque-b'
    assert next_unknown_control(run, hers, mine, ['opaque-a']) is None


@pytest.mark.parametrize('changed', ['epoch', 'number', 'birth'])
def test_only_a_new_current_body_gets_a_separate_budget(changed):
    run, hers, mine = _owner()
    for _ in range(PRESSES_PER_CONTROL):
        assert next_unknown_control(run, hers, mine, ['opaque-a']) == 'opaque-a'
    if changed == 'epoch':
        hers.identification.epoch += 1
    elif changed == 'number':
        hers.number = mine.number = 8
    else:
        mine.born += 1.
    assert next_unknown_control(run, hers, mine, ['opaque-a']) == 'opaque-a'
    if changed == 'epoch':
        hers.identification.epoch -= 1
    elif changed == 'number':
        hers.number = mine.number = 7
    else:
        mine.born -= 1.
    assert next_unknown_control(run, hers, mine, ['opaque-a']) is None


def test_missing_or_mismatched_birth_and_authority_do_not_issue_a_trial():
    run, hers, mine = _owner()
    assert next_unknown_control(run, hers, SimpleNamespace(number=7), run.keys) is None
    mine.born = math.nan
    assert next_unknown_control(run, hers, mine, run.keys) is None
    mine.born, mine.number = 1., 8
    assert next_unknown_control(run, hers, mine, run.keys) is None
    mine.number = 7
    assert next_unknown_control(run, hers, mine, ['unauthorized']) is None
    assert run._unknown_control_trials == {}
    assert run._unknown_control_probe_status == 'no authorized unknown control'


def test_owner_and_distinct_control_limits_cannot_be_renewed_by_churn():
    run, hers, mine = _owner()
    for epoch in range(MAX_CALIBRATION_OWNERS):
        hers.identification.epoch = epoch
        assert next_unknown_control(run, hers, mine, ['opaque-a']) == 'opaque-a'
    hers.identification.epoch += 1
    assert next_unknown_control(run, hers, mine, ['opaque-a']) is None
    hers.identification.epoch = 0
    assert next_unknown_control(run, hers, mine, ['opaque-a']) == 'opaque-a'
    assert next_unknown_control(run, hers, mine, ['opaque-a']) is None
    run, hers, mine = _owner()
    for number in range(MAX_CALIBRATION_CONTROLS):
        run.keys = [f'control-{number}']
        assert next_unknown_control(run, hers, mine, run.keys) == run.keys[0]
    run.keys = ['one-more']
    assert next_unknown_control(run, hers, mine, run.keys) is None


@pytest.mark.asyncio
async def test_stalled_first_unknown_response_does_not_starve_physical_trials_or_renew_them():
    keys, delivered = ['opaque-a', 'opaque-b', 'opaque-c'], []
    run = _Run(keys=list(keys), began=time.monotonic(), tried_at=-math.inf)
    run.counter_trial = CounteringMotion(done=True)
    hers = WhichIsHers()
    hers.number, hers.kind = 7, 1
    body = SimpleNamespace(number=7, born=1.)
    moves = SimpleNamespace(things={7: body})

    class Hands:
        async def down(self, key):
            delivered.append(('down', key))

        async def up(self, key):
            delivered.append(('up', key))

    hands = Hands()
    try:
        for turn in range(PRESSES_PER_CONTROL * len(keys)):
            run.keys = list(reversed(keys)) if turn % 2 else list(keys)
            await try_controls(hands, run, hers, time.monotonic(), moves, hold=hold_control, seconds=.01)
            pulse, deadline = run.body_trial, run._trial_input_deadline
            assert pulse.key in keys and run.held == pulse.key
            issued = len(delivered)
            await try_controls(hands, run, hers, pulse.began, moves, hold=hold_control, seconds=.01)
            assert run.body_trial is pulse and len(delivered) == issued
            await asyncio.wait_for(deadline, 1.)
            rest = run.body_trial
            assert rest.key == '' and rest.until > rest.began
            await try_controls(hands, run, hers, rest.began, moves, hold=hold_control, seconds=.01)
            assert run.body_trial is rest and not run.held
            await asyncio.sleep(max(0., rest.until - time.monotonic()) + .001)
        downs = [key for kind, key in delivered if kind == 'down']
        assert set(downs[:len(keys)]) == set(keys)
        assert Counter(downs) == Counter({key: PRESSES_PER_CONTROL for key in keys})
        assert delivered == [event for key in downs for event in [('down', key), ('up', key)]]
        assert len([r for r in run._input_clock_receipts if r[2]]) == len(downs)
        for _ in range(3):
            run.keys.reverse()
            await try_controls(hands, run, hers, time.monotonic(), moves, hold=hold_control, seconds=.01)
        assert len([event for event in delivered if event[0] == 'down']) == len(downs)
        assert hers.number == 7 and not hers.keys_known(keys)
        assert not hers._hers.by_press and not hers._by_thing
    finally:
        cancel_trial_deadline(run)
        if run.held:
            await hold_control(hands, run, hers, '', time.monotonic())


@pytest.mark.asyncio
async def test_counter_experiment_has_precedence_over_unknown_control_order():
    run = _Run(keys=['opaque-a', 'opaque-b'], began=time.monotonic(), tried_at=-math.inf)
    run.counter_trial = SimpleNamespace(key='opaque-b', done=False, phase_seconds=.01,
                                        choose=lambda *_a, **_k: 'opaque-b')
    hers = WhichIsHers()
    hers.number, hers.kind = 7, 1
    moves = SimpleNamespace(things={7: SimpleNamespace(number=7, born=1.)})
    async def hold(_hands, current, _hers, key, _at, **_options):
        current.held, current.held_since, current.held_trying = key, time.monotonic(), True
    try:
        await try_controls(object(), run, hers, time.monotonic(), moves, hold=hold, seconds=.01)
        assert run.body_trial.key == 'opaque-b'
        assert not hasattr(run, '_unknown_control_trials')
    finally:
        cancel_trial_deadline(run)


@pytest.mark.asyncio
async def test_an_ordinary_held_key_is_released_and_rested_before_a_fresh_trial():
    run = _Run(keys=['opaque-a'], began=time.monotonic(), tried_at=-math.inf)
    run.counter_trial = CounteringMotion(done=True)
    hers = WhichIsHers()
    hers.number, hers.kind = 7, 1
    moves = SimpleNamespace(things={7: SimpleNamespace(number=7, born=1.)})
    actions = []
    hands = SimpleNamespace()
    async def down(key):
        actions.append(('down', key))
    async def up(key):
        actions.append(('up', key))
    hands.down, hands.up = down, up
    try:
        await hold_control(hands, run, hers, 'opaque-a', time.monotonic(), trying=False)
        await try_controls(hands, run, hers, time.monotonic(), moves, hold=hold_control, seconds=.01)
        rest = run.body_trial
        assert rest.key == '' and not hasattr(run, '_unknown_control_trials')
        await try_controls(hands, run, hers, rest.began, moves, hold=hold_control, seconds=.01)
        assert actions == [('down', 'opaque-a'), ('up', 'opaque-a')]
        await asyncio.sleep(max(0., rest.until - time.monotonic()) + .001)
        await try_controls(hands, run, hers, time.monotonic(), moves, hold=hold_control, seconds=.01)
        assert actions[-1] == ('down', 'opaque-a') and run.body_trial.key == 'opaque-a'
        assert run._unknown_control_trials[(0, 7, 1.)].issued == {'opaque-a': 1}
    finally:
        cancel_trial_deadline(run)
        if run.held:
            await hold_control(hands, run, hers, '', time.monotonic())


@pytest.mark.asyncio
async def test_a_failed_dispatch_consumes_its_allowance_before_the_next_control():
    run = _Run(keys=['opaque-a', 'opaque-b'], began=time.monotonic(), tried_at=-math.inf)
    run.counter_trial = CounteringMotion(done=True)
    hers = WhichIsHers()
    hers.number, hers.kind = 7, 1
    moves = SimpleNamespace(things={7: SimpleNamespace(number=7, born=1.)})
    chosen = []
    async def failed(_hands, _run, _hers, key, _at, **_options):
        chosen.append(key)
        raise RuntimeError('delivery unavailable')
    for _ in range(2):
        with pytest.raises(RuntimeError, match='delivery unavailable'):
            await try_controls(object(), run, hers, time.monotonic(), moves, hold=failed, seconds=.01)
    assert chosen == ['opaque-a', 'opaque-b']
    assert run._unknown_control_trials[(0, 7, 1.)].issued == {'opaque-a': 1, 'opaque-b': 1}
    assert not hers._hers.by_press


def test_exhaustion_is_current_to_the_body_and_remaining_unknown_controls():
    run, hers, mine = _owner()
    run.things = {mine.number: mine}
    hers.kind, hers.carried, hers.way_of = 1, False, lambda _key: None
    for _ in range(PRESSES_PER_CONTROL):
        assert next_unknown_control(run, hers, mine, ['opaque-a']) == 'opaque-a'
    assert not unknown_control_trials_finished(run, hers)
    hers.way_of = lambda key: (1., 0.) if key != 'opaque-a' else None
    assert unknown_control_trials_finished(run, hers)
    run.keys = ['opaque-c', 'opaque-a', 'opaque-b']
    assert unknown_control_trials_finished(run, hers)
    hers.identification.epoch += 1
    assert not unknown_control_trials_finished(run, hers)
    assert next_unknown_control(run, hers, mine, ['opaque-a']) == 'opaque-a'
    assert not unknown_control_trials_finished(run, hers)


@pytest.mark.parametrize('birth', [None, float('nan')])
def test_an_unavailable_owner_yields_instead_of_leaving_discovery_busy(birth):
    from core.agency.inputs_while_measuring_control import body_probe_is_active

    run, hers, mine = _owner()
    run.things, run.trying = {7: mine}, 0
    hers.kind, hers.carried, hers.keys_known = 1, False, lambda _keys: False
    mine.born = birth
    assert next_unknown_control(run, hers, mine, run.keys) is None
    assert not body_probe_is_active(run, hers, pointer_steps=0, at=2.)
    assert run._unknown_control_probe_status == 'current body or control receipt unavailable'


@pytest.mark.asyncio
async def test_exhausted_unknown_control_exits_body_probe_and_reaches_ordinary_planning(monkeypatch):
    from core.agency import playing_as_it_happens as play
    from core.agency.giving_a_trial_time import ControlTrial
    from core.agency.inputs_while_measuring_control import body_probe_is_active

    run = _Run(keys=['opaque-a', 'opaque-b', 'opaque-c'], began=0., trying=2, tried_at=0.)
    hers = WhichIsHers()
    hers.number, hers.kind = 7, 1
    mine = SimpleNamespace(number=7, born=1.)
    moves = SimpleNamespace(things={7: mine}, seeing=True)
    run.things = moves.things
    hers.way_of = lambda key: None if key == 'opaque-a' else (1., 0.)
    for _ in range(PRESSES_PER_CONTROL):
        assert next_unknown_control(run, hers, mine, ['opaque-a']) == 'opaque-a'
    assert run.trying < 4 * len(run.keys) and not hers.keys_known(run.keys)
    run.held, run.held_since, run.held_trying = '', 2., True
    run.body_trial = ControlTrial('', 2., 3., hers.identification.epoch)
    assert body_probe_is_active(run, hers, pointer_steps=0, at=2.5)
    assert not body_probe_is_active(run, hers, pointer_steps=0, at=3.)
    progressed = []
    async def prepared(*_args, **_kwargs):
        return False
    async def planning(*_args, **_kwargs):
        progressed.append('ordinary planning')
        return True
    import core.agency.steering_from_a_model as steering
    monkeypatch.setattr(play, 'act_on_prepared_motion', prepared)
    monkeypatch.setattr(steering, 'steer_with_a_model', planning)
    choosing = SimpleNamespace(mine=mine)
    await play._act(object(), run, moves, hers, None, choosing, 3.)
    assert progressed == ['ordinary planning']


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', ['epoch', 'number', 'birth'])
async def test_the_actual_actor_gives_a_new_owner_its_trials_after_old_discovery_is_spent(monkeypatch, changed):
    from core.agency import playing_as_it_happens as play
    from core.agency import steering_from_a_model as steering
    from core.agency.inputs_while_measuring_control import body_probe_is_active

    keys, delivered, planned = ['opaque-a', 'opaque-b', 'opaque-c'], [], []
    run = _Run(keys=keys, began=time.monotonic(), trying=4*len(keys), tried_at=-math.inf)
    run.counter_trial = CounteringMotion(done=True)
    hers = WhichIsHers()
    hers.number, hers.kind = 7, 1
    mine = SimpleNamespace(number=7, born=1.)
    for _ in range(PRESSES_PER_CONTROL*len(keys)):
        assert next_unknown_control(run, hers, mine, keys) is not None
    old_owner = next(iter(run._unknown_control_trials))
    if changed == 'epoch':
        hers.identification.epoch += 1
    elif changed == 'number':
        hers.number = mine.number = 8
    else:
        mine.born = 2.
    moves = SimpleNamespace(things={mine.number:mine}, seeing=True)
    run.things = moves.things
    choosing = SimpleNamespace(mine=mine)
    assert body_probe_is_active(run, hers, pointer_steps=0, at=time.monotonic())
    async def prepared(*_args, **_kwargs):
        return False
    async def planning(*_args, **_kwargs):
        planned.append(True)
        return True
    async def short_trial(hands, run, hers, at, moves):
        await try_controls(hands, run, hers, at, moves, hold=hold_control, seconds=.01)
    async def down(key):
        delivered.append(('down',key))
    async def up(key):
        delivered.append(('up',key))
    monkeypatch.setattr(play, 'act_on_prepared_motion', prepared)
    monkeypatch.setattr(play, '_try_the_keys', short_trial)
    monkeypatch.setattr(steering, 'steer_with_a_model', planning)
    hands = SimpleNamespace(down=down,up=up)
    try:
        for _ in range(PRESSES_PER_CONTROL*len(keys)):
            await play._act(hands, run, moves, hers, None, choosing, time.monotonic())
            assert run.held in keys and not planned
            await asyncio.wait_for(run._trial_input_deadline, 1.)
            rest = run.body_trial
            assert not run.held and rest.key == ''
            await play._act(hands, run, moves, hers, None, choosing, rest.began)
            assert not planned and run.body_trial is rest
            await asyncio.sleep(max(0.,rest.until-time.monotonic())+.001)
        assert Counter(key for kind,key in delivered if kind=='down') == Counter({key:2 for key in keys})
        assert set(run._unknown_control_trials[old_owner].issued.values()) == {PRESSES_PER_CONTROL}
        assert not hers.keys_known(keys) and not hers._hers.by_press
        assert not body_probe_is_active(run, hers, pointer_steps=0, at=time.monotonic())
        await play._act(hands, run, moves, hers, None, choosing, time.monotonic())
        assert planned == [True]
    finally:
        cancel_trial_deadline(run)
        if run.held:
            await hold_control(hands, run, hers, '', time.monotonic())
