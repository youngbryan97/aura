"""Control measurements have enough samples at slow and changing picture rates."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency import playing_as_it_happens as play
from core.agency.time_to_observe_a_control import observed_response_seconds
from core.agency.what_meeting_things_does import WhatMeetingDoes
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.what_moves_in_the_picture import Kind, Thing

pytestmark = pytest.mark.unit
OPTIONS = dict(least_s=.45, settle_s=.2, samples=4, span_s=.15, longest_s=1.8)


@pytest.mark.parametrize("cadence", [1 / 60, 1 / 15, .1, 1 / 7.5, .2, .35])
def test_the_interval_reserves_samples_after_settlement(cadence):
    duration = observed_response_seconds([cadence] * 6, **OPTIONS)
    assert duration is not None and .45 <= duration <= 1.8
    assert duration >= .2 + 4 * cadence


@pytest.mark.parametrize("intervals", [[], [.1, .1], [0, -.1, float("nan"), float("inf")], [.5] * 6])
def test_unobserved_or_broken_cadence_is_unknown(intervals):
    assert observed_response_seconds(intervals, **OPTIONS) is None


def test_a_request_beyond_the_time_limit_is_unknown():
    assert observed_response_seconds([.3] * 6, **{**OPTIONS, "longest_s": .6}) is None
    assert observed_response_seconds([.1] * 6, **{**OPTIONS, "span_s": .8}) == pytest.approx(1.1)


def test_a_single_fast_picture_does_not_shorten_a_slow_observation_interval():
    assert observed_response_seconds([.1] * 5 + [.01], **OPTIONS) == pytest.approx(.6)


def test_only_the_bounded_recent_history_is_examined():
    class History:
        def __reversed__(self):
            yield from [.1] * 12
            raise AssertionError("old history was traversed")

    assert observed_response_seconds(History(), **OPTIONS) == pytest.approx(.6)


def _body(number, axis, sign):
    look = np.zeros(64)
    look[number] = 1.
    body = Thing(number, 1500., 1500., 14., 14., look, (140, 170, 220),
                 np.zeros((1, 1, 3), np.uint8), born=0., seen=0., kind=number, moved=True)
    return body, Kind(number, look, 196, body.colour)


@pytest.mark.asyncio
@pytest.mark.parametrize("hz", [7.5, 10., 15., 60.])
@pytest.mark.parametrize("axis,sign", [(0, -1), (0, 1), (1, -1), (1, 1)])
@pytest.mark.parametrize("favored", [False, True])
async def test_the_actual_actor_learns_inertial_controls_at_each_observed_rate(monkeypatch, hz, axis, sign, favored):
    now = [0.]
    monkeypatch.setattr(play.time, "monotonic", lambda: now[0])
    if favored:
        monkeypatch.setattr(play, "a_key_that_pays", lambda _keys: "other")
    actions = []

    class Hands:
        async def down(self, key):
            actions.append((now[0], key))

        async def up(self, key):
            actions.append((now[0], ""))

    body, body_kind = _body(1, axis, sign)
    other, other_kind = _body(2, axis, sign)
    other.x += 700
    moves = SimpleNamespace(shape=(5000, 5000), things={1: body, 2: other}, seeing=True,
                            kinds=[body_kind, other_kind])
    hers, meeting = WhichIsHers(), WhatMeetingDoes()
    run = play._Run(keys=["drive", "other", "idle"], began=0.)
    for step in range(1, int(16 * hz) + 1):
        now[0] = step / hz
        run.intervals.append(1 / hz)
        for thing, push in ((body, -110. if run.held == "drive" else 90.), (other, 90.)):
            velocity = [thing.vx, thing.vy]
            velocity[axis] += sign * push / hz
            thing.vx, thing.vy = velocity
            thing.x += thing.vx / hz
            thing.y += thing.vy / hz
            thing.seen = now[0]
            thing.path.append((now[0], thing.x, thing.y))
        hers.saw(moves, [], now[0])
        choosing = play._Choosing(moves, hers, meeting, run.keys)
        await play._act(Hands(), run, moves, hers, meeting, choosing, now[0])
        if hers.keys_known(run.keys):
            break
    assert hers.number == body.number and hers.carried and hers.keys_known(run.keys)
    for key in ["", *run.keys]:
        expected = [0., 0.]
        expected[axis] = sign * (-110. if key == "drive" else 90.)
        assert hers.push_of(key) == pytest.approx(expected, abs=1e-7)
    assert now[0] < 16 and all(any(k == key for _at, k in actions) for key in run.keys)


@pytest.mark.asyncio
async def test_changing_cadence_does_not_move_a_delivered_trial_deadline(monkeypatch):
    now = [10.]
    monkeypatch.setattr(play.time, "monotonic", lambda: now[0])
    run, hers = play._Run(keys=["a", "b"], began=0.), WhichIsHers()
    run.intervals.extend([.1] * 6)
    actions = []

    class Hands:
        async def down(self, key):
            actions.append(("down", key))
            now[0] += .08

        async def up(self, key):
            actions.append(("up", key))

    await play._try_the_keys(Hands(), run, hers, now[0])
    first = run.body_trial
    assert first.until == pytest.approx(10.68)
    run.intervals.clear()
    run.intervals.extend([.3] * 6)
    # A picture captured before delivery cannot cancel the input still waiting to be observed.
    await play._try_the_keys(Hands(), run, hers, 10.05)
    for at in (10.3, 10.5, 10.65):
        now[0] = at
        await play._try_the_keys(Hands(), run, hers, at)
    assert run.body_trial is first and actions == [("down", "a")]
    now[0] = 10.7
    await play._try_the_keys(Hands(), run, hers, now[0])
    assert actions == [("down", "a"), ("up", "a")]
    assert run.body_trial.key == "" and run.longest_control_trial_s <= 1.8


@pytest.mark.asyncio
async def test_a_same_named_input_with_a_new_delivery_does_not_own_the_old_trial(monkeypatch):
    from core.agency.giving_a_trial_time import ControlTrial

    monkeypatch.setattr(play.time, "monotonic", lambda: 10.)
    run, hers = play._Run(keys=["a"], began=0.), WhichIsHers()
    run.held, run.held_since, run.held_trying = "a", 9., True
    old = ControlTrial("a", 8., 12., hers.identification.epoch)
    run.body_trial = old

    class Hands:
        async def down(self, key):
            pass

        async def up(self, key):
            pass

    await play._try_the_keys(Hands(), run, hers, 10.)
    assert run.body_trial is not old


@pytest.mark.asyncio
async def test_retaining_an_input_without_a_new_receipt_cannot_renew_its_deadline(monkeypatch):
    from core.agency.giving_a_trial_time import ControlTrial

    monkeypatch.setattr(play.time, "monotonic", lambda: 10.)
    run, hers = play._Run(keys=["a"], began=0.), WhichIsHers()
    run.held, run.held_since, run.held_trying = "a", 9., True
    run.body_trial = ControlTrial("a", 9., 9.45, hers.identification.epoch)
    run.counter_trial = SimpleNamespace(key="a", done=False, phase_seconds=.45, choose=lambda *a, **k: "a")

    class Hands:
        async def down(self, key):
            raise AssertionError("unchanged input was dispatched twice")

        async def up(self, key):
            raise AssertionError("unchanged input was released")

    await play._try_the_keys(Hands(), run, hers, 10.)
    assert run.body_trial.began == 9. and run.body_trial.until == 9.45


def test_a_counter_experiment_keeps_its_chosen_phase_time():
    from core.agency.countering_motion_as_it_appears import CounteringMotion

    trial = CounteringMotion()
    thing = SimpleNamespace(number=1, seen=1., born=0., vx=0., vy=20., w=10., h=10.,
                            path=[(.7, 0., 0.), (1., 0., 6.)])
    options = dict(held="", held_since=.5, epoch=0, settle_s=.2, step_s=.6, steps=4,
                   wait_s=4., against=lambda _k, _t: "a")
    assert trial.choose(["a"], [thing], 1., **options) == "a"
    assert trial.phase_seconds == .6
    assert trial.choose(["a"], [thing], 1.61, **{**options, "held": "a", "held_since": 1., "step_s": 1.8}) == ""
    assert trial.phase_seconds == .6


@pytest.mark.asyncio
@pytest.mark.parametrize('cadence', [.21, .25, .3, .35])
async def test_facing_calibration_does_not_shorten_an_observable_request_to_the_default(cadence):
    import math
    import time

    from core.agency.a_trial_has_a_physical_end import cancel_trial_deadline
    from core.agency.giving_a_trial_time import try_a_facing_control
    from core.agency.inputs_while_measuring_control import hold_control
    from core.agency.which_way_she_faces import WhichWaySheFaces

    run = play._Run(keys=['drive', 'bend-a', 'bend-b'], began=time.monotonic(), tried_at=-math.inf)
    run.faces = WhichWaySheFaces()
    run.faces.heading_hints.add('drive')
    run.intervals.extend([cadence] * 6)
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 1
    choosing = SimpleNamespace(mine=SimpleNamespace(number=1, born=0.))
    delivered = []

    class Hands:
        async def down(self, key):
            delivered.append(('down', key))

        async def up(self, key):
            delivered.append(('up', key))

    hands = Hands()
    try:
        assert await try_a_facing_control(hands, run, hers, choosing, time.monotonic(),
            hold=hold_control, seconds=.45, limit=10)
        trial = run.facing_trial
        duration = trial.until - trial.began
        assert duration >= .2 + 4 * cadence - 1e-9 and duration <= 1.8
        assert trial.began == run.held_since and run.held == 'drive'
        assert delivered == [('down', 'drive')]
        assert run._input_clock_receipts[-1][2] == 'drive'
    finally:
        cancel_trial_deadline(run)
        if run.held:
            await hold_control(hands, run, hers, '', time.monotonic())


def test_the_observation_time_has_a_registered_measured_claim():
    from core.agency.time_to_observe_a_control import _observation_time_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _observation_time_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "control_trials_reserve_their_observation_samples")
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
