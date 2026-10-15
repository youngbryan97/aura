"""Finite experiments have a physical end when observation or UI work stalls."""
import asyncio
import math
import time
from types import SimpleNamespace

import pytest

from core.agency.a_trial_has_a_physical_end import cancel_trial_deadline, lease_control_trial
from core.agency.countering_motion_as_it_appears import CounteringMotion
from core.agency.giving_a_trial_time import ControlTrial, try_a_facing_control
from core.agency.inputs_while_measuring_control import hold_control
from core.agency.playing_as_it_happens import _Run
from core.agency.trying_controls_with_a_view_of_their_response import try_controls
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def setup():
    run = _Run(keys=['opaque-a', 'opaque-b'], began=time.monotonic(), tried_at=-math.inf)
    hers, actions = WhichIsHers(), []

    class Hands:
        async def down(self, key):
            actions.append(('down', key, time.monotonic()))
        async def up(self, key):
            actions.append(('up', key, time.monotonic()))

    return run, hers, actions, Hands()


async def test_actual_discovery_releases_then_preserves_its_rest_without_another_picture():
    run, hers, actions, hands = setup()
    moves = SimpleNamespace(things={}, shape=(200, 200))
    await try_controls(hands, run, hers, time.monotonic(), moves, hold=hold_control, seconds=.015)
    pulse = run.body_trial
    deadline = run._trial_input_deadline
    await asyncio.wait_for(deadline, timeout=1.)
    assert [(kind, key) for kind, key, _at in actions] == [('down', 'opaque-a'), ('up', 'opaque-a')]
    assert actions[-1][2] >= pulse.until and not run.held
    assert run.body_trial.key == '' and run.body_trial.began == run.held_since
    assert run.body_trial.until - run.body_trial.began == pytest.approx(pulse.until - pulse.began)
    assert run.held_trying and hers._held[-1][1:] == ('', True)
    # A picture during that rest cannot skip it, even after the body is found.
    await try_controls(hands, run, hers, run.body_trial.began, moves, hold=hold_control, seconds=.015)
    assert len(actions) == 2


async def test_a_feedback_chosen_trial_keeps_its_deadline_and_its_nonindependent_rest():
    from core.agency.inputs_while_measuring_control import body_probe_is_active

    run, hers, actions, hands = setup()
    await hold_control(hands, run, hers, 'opaque-a', time.monotonic(), trying=False)
    run.body_trial = ControlTrial(run.held, run.held_since, run.held_since + .015,
                                  hers.identification.epoch, independent=False,
                                  expected_until=run.held_since + .005)
    pulse = run.body_trial
    assert lease_control_trial(hands, run, hers, pulse, field='body_trial')
    assert body_probe_is_active(run, hers, pointer_steps=0, at=pulse.began)
    await asyncio.wait_for(run._trial_input_deadline, timeout=1.)
    assert [(kind, key) for kind, key, _at in actions] == [('down', 'opaque-a'), ('up', 'opaque-a')]
    assert actions[-1][2] >= pulse.until and not run.held and not run.held_trying
    assert run.body_trial.key == '' and not run.body_trial.independent
    assert run.body_trial.expected_until - run.body_trial.began == pytest.approx(.005)
    assert hers._held[-1][1:] == ('', False)
    assert body_probe_is_active(run, hers, pointer_steps=0, at=run.body_trial.began)
    moves = SimpleNamespace(things={}, shape=(200, 200))
    await try_controls(hands, run, hers, run.body_trial.began, moves, hold=hold_control, seconds=.015)
    assert len(actions) == 2


async def test_actual_facing_trial_releases_without_a_new_picture():
    run, hers, actions, hands = setup()
    hers.number = 1
    run.faces = SimpleNamespace(trial=lambda _keys: 'opaque-b')
    choosing = SimpleNamespace(mine=SimpleNamespace(number=1))
    assert await try_a_facing_control(hands, run, hers, choosing, time.monotonic(),
                                     hold=hold_control, seconds=.015, limit=10)
    deadline = run._trial_input_deadline
    await asyncio.wait_for(deadline, timeout=1.)
    assert [(kind, key) for kind, key, _at in actions] == [('down', 'opaque-b'), ('up', 'opaque-b')]
    assert not run.held and run.facing_trial is None


async def test_a_new_press_retires_the_old_lease():
    run, hers, actions, hands = setup()
    await hold_control(hands, run, hers, 'opaque-a', time.monotonic(), trying=True)
    run.body_trial = ControlTrial(run.held, run.held_since, run.held_since + .02, hers.identification.epoch)
    assert lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
    old_deadline = run._trial_input_deadline
    await hold_control(hands, run, hers, 'opaque-b', time.monotonic(), trying=False)
    await asyncio.gather(old_deadline, return_exceptions=True)
    assert run.held == 'opaque-b' and actions[-1][:2] == ('down', 'opaque-b')
    assert run._trial_input_deadline is None


async def test_a_policy_cannot_extend_the_same_delivered_press():
    run, hers, actions, hands = setup()
    await hold_control(hands, run, hers, 'opaque-a', time.monotonic(), trying=True)
    run.body_trial = ControlTrial(run.held, run.held_since, run.held_since + .015, hers.identification.epoch)
    assert lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
    deadline = run._trial_input_deadline
    run.body_trial = ControlTrial(run.held, run.held_since, run.held_since + 30., hers.identification.epoch)
    assert not lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
    await asyncio.wait_for(deadline, timeout=1.)
    assert not run.held and len(actions) == 2


async def test_a_changed_identity_still_releases_but_cannot_extend_old_measurements():
    run, hers, _actions, hands = setup()
    await hold_control(hands, run, hers, 'opaque-a', time.monotonic(), trying=True)
    run.body_trial = ControlTrial(run.held, run.held_since, run.held_since + .015, hers.identification.epoch)
    assert lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
    deadline = run._trial_input_deadline
    hers.identification.reset('another observation context')
    await asyncio.wait_for(deadline, timeout=1.)
    assert not run.held and not run.held_trying and run.body_trial is None


async def test_a_preplanned_counter_pulse_finishes_into_its_rest_once():
    run, hers, _actions, hands = setup()
    counter = run.counter_trial = CounteringMotion(key='opaque-a', epoch=0, phase_seconds=.015,
                                                 until=time.monotonic() + 1.)
    await hold_control(hands, run, hers, 'opaque-a', time.monotonic(), trying=True)
    began = run.held_since
    run.body_trial = ControlTrial(run.held, began, began + .015, 0)
    assert lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
    await asyncio.wait_for(run._trial_input_deadline, timeout=1.)
    assert counter.phase == 1 and not counter.done
    assert counter._continue(run.held_since + .005, '', run.held_since, 0, 4) == ''
    assert counter._continue(run.held_since + .02, '', run.held_since, 0, 4) == 'opaque-a'
    assert not counter.released_at_deadline('opaque-a', began, run.held_since, 0, steps=4)
    assert counter.phase == 2


async def test_context_can_change_while_the_physical_release_is_in_flight():
    run, hers, _actions, hands = setup()
    original = hands.up

    async def up(key):
        await original(key)
        hers.identification.reset('the page changed during input delivery')

    hands.up = up
    await hold_control(hands, run, hers, 'opaque-a', time.monotonic(), trying=True)
    run.body_trial = ControlTrial(run.held, run.held_since, run.held_since + .015, 0)
    assert lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
    await asyncio.wait_for(run._trial_input_deadline, timeout=1.)
    assert not run.held and not run.held_trying and run.body_trial is None


async def test_the_owner_can_cancel_its_trial_lease_on_exit():
    run, hers, actions, hands = setup()
    await hold_control(hands, run, hers, 'opaque-a', time.monotonic(), trying=True)
    run.body_trial = ControlTrial(run.held, run.held_since, run.held_since + 5., 0)
    assert lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
    deadline = run._trial_input_deadline
    cancel_trial_deadline(run)
    await asyncio.gather(deadline, return_exceptions=True)
    assert deadline.cancelled() and len(actions) == 1


async def test_unbounded_or_unowned_trials_cannot_acquire_a_lease():
    run, hers, actions, hands = setup()
    await hold_control(hands, run, hers, 'opaque-a', time.monotonic(), trying=False)
    run.body_trial = ControlTrial(run.held, run.held_since, math.inf, 0)
    assert not lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
    assert len(actions) == 1 and getattr(run, '_trial_input_deadline', None) is None


async def test_the_physical_deadline_has_a_registered_measured_claim():
    from core.agency.a_trial_has_a_physical_end import _physical_trial_deadline_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _physical_trial_deadline_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'a_control_trial_has_a_physical_deadline')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


async def test_leaving_play_cancels_deadlines_and_records_the_real_primary_release():
    from core.agency.inputs_while_measuring_control import release_after_play

    run, hers, actions, hands = setup()
    await hold_control(hands, run, hers, 'opaque-a', time.monotonic(), trying=True)
    run.body_trial = ControlTrial(run.held, run.held_since, run.held_since + 5., 0)
    assert lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
    deadline = run._trial_input_deadline
    await release_after_play(hands, run, hers)
    await asyncio.gather(deadline, return_exceptions=True)
    assert deadline.cancelled() and not run.held and hers._held[-1][1:] == ('', False)
    assert [(kind, key) for kind, key, _at in actions] == [('down', 'opaque-a'), ('up', 'opaque-a')]


async def test_a_rest_asked_for_with_nothing_held_begins_now():
    """Offline 2026-10-13 a game was ended in its first half minute: a trial's rest came up with nothing held, the hold
    had nothing to do and left no time, and the trial's end was reckoned from none."""
    run, hers, actions, hands = setup()
    run.trying = 1                                                        # the turn after a key's is a rest
    moves = SimpleNamespace(things={}, shape=(200, 200))
    at = time.monotonic()
    await try_controls(hands, run, hers, at, moves, hold=hold_control, seconds=.015)
    assert run.body_trial.key == "" and run.body_trial.began == at and run.body_trial.until > at
    assert actions == []
    cancel_trial_deadline(run)
