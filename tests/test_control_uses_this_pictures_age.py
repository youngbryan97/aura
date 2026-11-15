"""Uneven processing time must reach the real control decision."""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.agency import playing_as_it_happens as play
from core.agency.choosing_where_to_be import _Choosing
from core.agency.time_from_picture_to_control import (
    ControlTiming,
    _picture_age_invariant,
    allow_for_picture_age,
)
from core.agency.what_meeting_things_does import WhatMeetingDoes
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


def _decision(axis, direction, speed):
    decision = _Choosing.__new__(_Choosing)
    decision.mine = SimpleNamespace(number=7, x=400., y=400., w=12., h=12.)
    way = [0., 0.]
    way[axis] = direction * speed
    decision.ways = {"go": tuple(way), "back": tuple(-v for v in way)}
    goal = [None, None]
    goal[axis] = 400. + direction * speed * .05
    decision.target = lambda: (tuple(goal), "meet", None)
    decision.speed = lambda axis: speed
    decision.danger = lambda way: 0.
    decision.next_picture_s, decision.response_s = .02, .01
    decision.physics, decision.across, decision.updown, decision.lifts = None, False, False, []
    return decision


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("direction", [-1., 1.])
@pytest.mark.parametrize("speed", [80., 200.])
def test_current_age_prevents_overshoot_after_a_slow_frame(axis, direction, speed):
    timing = ControlTiming()
    timing.delivered(1., 1.001, 1.011)
    run = SimpleNamespace(control_timing=timing)
    decision = _decision(axis, direction, speed)
    stale_key = decision.key("go")[0]
    assert stale_key == "go"
    allow_for_picture_age(run, decision, 2., 2.04)
    fresh_key = decision.key("go")[0]
    assert fresh_key == ""
    # Independent constant-speed plant: old input lasts through processing
    # and delivery, then the chosen input lasts until the next observation.
    target = 400. + direction * speed * .05
    at_effect = 400. + direction * speed * (.04 + .01)
    stale_finish = at_effect + decision.ways[stale_key][axis] * .02
    fresh_finish = at_effect + decision.ways.get(fresh_key, (0., 0.))[axis] * .02
    assert abs(fresh_finish - target) < 1e-8
    assert abs(stale_finish - target) >= speed * .019


def test_a_slow_past_frame_does_not_make_a_fresh_frame_old():
    timing = ControlTiming()
    timing.delivered(1., 1.2, 1.21)
    decision = _decision(0, 1., 200.)
    allow_for_picture_age(SimpleNamespace(control_timing=timing), decision, 2., 2.001)
    assert decision.response_s == pytest.approx(.011)
    assert decision.key("go")[0] == "go"


@pytest.mark.parametrize("elapsed", [0., .01, .1, .4])
def test_delivery_time_does_not_include_processing_before_dispatch(elapsed):
    timing = ControlTiming()
    receipt = timing.delivered(1., 1. + elapsed, 1. + elapsed + .02)
    assert receipt is not None
    assert timing.ahead(2., 2.03) == pytest.approx(.05)


def test_motion_response_counts_only_the_part_after_delivery():
    run = play._Run(keys=["go"], began=0.)
    receipt = run.control_timing.delivered(1., 1.01, 1.02)
    run.previous_control = (7, 40., 55., 1., (0., 200.), (0., 0.), receipt)
    play._measure_response(run, SimpleNamespace(number=7, x=40., y=62.), 1.05)
    assert run.motion_responses[-1] == pytest.approx(.035)
    assert run.control_timing.effects[-1] == pytest.approx(.015)
    assert run.control_timing.ahead(2., 2.1) == pytest.approx(.125)


def test_a_response_during_delivery_does_not_add_another_delay():
    timing = ControlTiming()
    receipt = timing.delivered(1., 1.01, 1.03)
    timing.took_effect(receipt, .02)
    assert timing.effects[-1] == 0.
    assert timing.ahead(2., 2.01) == pytest.approx(.03)


@pytest.mark.parametrize("times", [(1., 2., 1.5), (2., 1., 3.), (math.nan, 1., 2.), (1., 2., math.inf)])
def test_invalid_receipts_do_not_supply_delivery_evidence(times):
    timing = ControlTiming()
    assert timing.delivered(*times) is None
    assert not timing.deliveries


def test_invalid_picture_ages_remain_unavailable():
    timing = ControlTiming()
    assert timing.ahead(2., 1.) is None
    assert timing.ahead(math.nan, 1.) is None


@pytest.mark.asyncio
async def test_a_second_input_cannot_supply_a_single_transition_delay():
    from core.agency.inputs_while_measuring_control import HandsWithControlReceipts
    from core.agency.what_a_press_does import WhatAPressDoes

    run = play._Run(keys=["go"], began=0., presses=WhatAPressDoes())
    receipt = run.control_timing.delivered(1., 1.01, 1.02)
    run.previous_control = (7, 40., 55., 1., (0., 200.), (0., 0.), receipt)
    hands = HandsWithControlReceipts(SimpleNamespace(tap=AsyncMock()), run, WhichIsHers())
    await hands.tap("another")
    play._measure_response(run, SimpleNamespace(number=7, x=40., y=62.), 1.05)
    assert not run.control_timing.effects


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["", "a"])
async def test_timed_presses_measure_delivery_without_counting_the_hold(monkeypatch, key):
    from unittest.mock import Mock

    import sys

    clock = SimpleNamespace(at=10.)
    # The press is timed by the clock of the module that holds it, which since 11f59aa32b is not this one: patched
    # here alone, the delivery was read off the real clock and came to a microsecond.
    pressing = sys.modules[play._press_in_time.__module__]
    for module in {play, pressing}:
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.at))
    timing = Mock()
    timing.press_now.return_value = True
    run = play._Run(keys=[key] if key else [], began=9., timing=timing)

    class Hands:
        async def click(self, x, y):
            clock.at += .02

        async def down(self, key):
            clock.at += .02

        async def up(self, key):
            clock.at += .01

    async def hold(seconds):
        clock.at += seconds

    monkeypatch.setattr(play.asyncio, "sleep", hold)
    assert await play._press_in_time(Hands(), run, SimpleNamespace(things={}), 9.9, None)
    assert run.control_timing.deliveries[-1] == pytest.approx(.02)
    assert timing.press_now.call_args.args[-1] == pytest.approx(.1 + play.RESPONSE_S)


@pytest.mark.asyncio
@pytest.mark.parametrize("carried", [False, True])
async def test_actual_act_refreshes_age_after_work_and_records_delivery(monkeypatch, carried):
    clock = SimpleNamespace(at=10.022)
    monkeypatch.setattr(play, "time", SimpleNamespace(monotonic=lambda: clock.at))
    hers = WhichIsHers()
    hers.number, hers.kind, hers.carried = 7, 0, carried
    for _ in range(12):
        hers._hers.add("go", 200., 0.)
        hers._hers.add("back", -200., 0.)
    hers.holding("go", 9., trying=False)
    run = play._Run(keys=["go", "back"], began=9., held="go", trying=8)
    run.control_timing.delivered(9., 9.001, 9.011)
    decision = _decision(0, 1., 200.)
    seen_ages = []
    original = decision.key

    def choose(held, **kwargs):
        seen_ages.append(decision.response_s)
        return original(held, **kwargs)

    def extra_work(*args):
        clock.at += .018

    class Hands:
        async def up(self, key):
            assert key == "go"
            clock.at += .01

        async def down(self, key):
            pytest.fail("the old motion should have reached the target before another key is needed")

    decision.key = choose
    monkeypatch.setattr(play, "_a_facing_trial", AsyncMock(return_value=False))
    monkeypatch.setattr(play, "_where_to_meet_things", extra_work)
    monkeypatch.setattr(play, "chose_in_play", lambda *args: None)
    monkeypatch.setattr(play, "_press_to_get_clear", AsyncMock())
    monkeypatch.setattr(play, "_trigger", AsyncMock())
    await play._act(Hands(), run, SimpleNamespace(things={}), hers, WhatMeetingDoes(), decision, 10.)
    assert seen_ages[-1] == pytest.approx(.05)
    assert run.held == ""
    assert run.control_delivery.began == pytest.approx(10.04)
    assert run.control_delivery.finished == pytest.approx(10.05)
    assert run.control_timing.deliveries[-1] == pytest.approx(.01)
    assert run.responses[-1] == pytest.approx(.05)
    # A predicted velocity under thrust is not a constant-velocity plant.
    assert (run.previous_control is None) is carried
    if not carried:
        assert run.previous_control[-1] is run.control_delivery


def test_the_registered_claim_runs_its_invariant():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _picture_age_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    claim = next(t for t in suite.tests() if t.name == "control_prediction_uses_the_current_picture_age")
    assert claim.predict(None) is True
    assert any(c.test == claim.name for c in suite.claims())
