"""Pointer steering and a separate trigger reach their physical inputs together."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency import playing_as_it_happens
from core.agency.playing_as_it_happens import (
    _act,
    _Choosing,
    _click_the_picture,
    _Run,
    _what_the_rules_said_of,
)
from core.agency.what_meeting_things_does import CLICK, MEET, SHOOT, WhatMeetingDoes
from core.agency.what_the_rules_said import WhatTheRulesSaid
from core.agency.which_one_answers_to_her import Makes, WhichIsHers
from core.perception.what_moves_in_the_picture import Kind, Thing

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class _Hands:
    def __init__(self):
        self.inputs = []

    async def point(self, x, y):
        self.inputs.append(("point", x, y))

    async def click(self, x, y):
        self.inputs.append(("click", x, y))

    async def tap(self, key):
        self.inputs.append(("tap", key))

    async def up(self, key):
        self.inputs.append(("up", key))


def _world(*, trigger=None, learned=False):
    at = time.monotonic()
    mine = SimpleNamespace(number=1, kind=0, x=50.0, y=100.0, moved=True)
    moves = SimpleNamespace(shape=(200, 200), things={1: mine})
    hers = WhichIsHers()
    hers.follows_pointer = True
    hers.last_seen = (mine.x, mine.y)
    choosing = SimpleNamespace(
        mine=mine, moves=moves, pointing=True, ways={}, shot=Makes(trigger or "mouse", 3, 0.0, -100.0, 2) if learned else None,
        target=lambda: ((75.0, 100.0), "shoot", object()),
        fire=lambda _aim, _why: trigger,
    )
    run = _Run(keys=[], began=at, pointer_first=True)
    return moves, hers, choosing, run, at


@pytest.mark.parametrize("trigger, physical", [("mouse", "click"), ("space", "tap")])
async def test_pointer_steering_delivers_a_learned_trigger_on_its_own_channel(trigger, physical):
    moves, hers, choosing, run, at = _world(trigger=trigger, learned=True)
    hands = _Hands()
    await _act(hands, run, moves, hers, WhatMeetingDoes(), choosing, at)
    assert hands.inputs[0][0] == physical
    assert hands.inputs[1] == ("point", 0.375, 0.5)
    if physical == "click":
        assert hands.inputs[0] == ("click", 0.25, 0.5)
    else:
        assert hands.inputs[0] == ("tap", "space")
    assert hers._taps[0].key == trigger
    assert run.taps == 1


@pytest.mark.parametrize("words, physical", [
    ("Use the mouse to move the carriage. Click the mouse button to throw a marker.", "click"),
    ("Use the mouse to steer. Press SPACE to fire a signal.", "tap"),
])
async def test_a_visible_trigger_instruction_can_discover_effects_while_pointer_steering(words, physical):
    moves, hers, choosing, run, at = _world()
    hands = _Hands()
    rules = WhatTheRulesSaid.read(words)
    await _act(hands, run, moves, hers, WhatMeetingDoes(), choosing, at, rules=rules)
    assert [item[0] for item in hands.inputs] == [physical, "point"]
    await _act(hands, run, moves, hers, WhatMeetingDoes(), choosing, at + 0.01, rules=rules)
    assert len(hands.inputs) == 2


async def test_learning_a_shot_respects_its_alignment_before_triggering():
    moves, hers, choosing, run, at = _world(learned=True)
    hands = _Hands()
    rules = WhatTheRulesSaid.read("Move with the mouse. Click the mouse to throw a marker.")
    await _act(hands, run, moves, hers, WhatMeetingDoes(), choosing, at, rules=rules)
    assert hands.inputs == [("point", 0.375, 0.5)]
    assert not hers._taps


async def test_pointer_steering_releases_an_old_held_key_before_delivering_a_trigger():
    moves, hers, choosing, run, at = _world(trigger="mouse", learned=True)
    run.held = "left"
    hands = _Hands()
    await _act(hands, run, moves, hers, WhatMeetingDoes(), choosing, at)
    assert [item[0] for item in hands.inputs] == ["up", "click", "point"]
    assert run.held == ""


@pytest.mark.parametrize("named, expected", [(False, []), (True, ["click"])])
async def test_a_named_picture_trigger_remains_available_after_motion_starts(named, expected):
    moves, _hers, _choosing, run, at = _world()
    run.pointer_trigger = named
    hands = _Hands()
    await _click_the_picture(hands, run, moves, at)
    assert [item[0] for item in hands.inputs] == expected


def _thing(number, kind, x, y, *, at=0.0, vy=0.0):
    return Thing(number, x, y, 12.0, 12.0, np.zeros(64), (255, 0, 0),
                 np.zeros((1, 1, 3), dtype=np.uint8), at, at,
                 kind=kind, moved=True, vy=vy)


def _measured_world():
    mine, target = _thing(1, 0, 40.0, 150.0), _thing(2, 1, 40.0, 20.0)
    moves = SimpleNamespace(shape=(200, 200), things={1: mine, 2: target},
                            kinds=[Kind(0, np.zeros(64), 144.0, (0, 0, 255)),
                                   Kind(1, np.zeros(64), 144.0, (255, 0, 0))])
    hers = WhichIsHers()
    # Establish identity through observed pointer trials, with no assigned owner.
    for step in range(64):
        x = 20.0 + 4.0 * (step % 16 if (step // 16) % 2 == 0 else 15 - step % 16)
        hers.pointed(x, 150.0, 1.0 + step * 0.2)
        mine.x, mine.seen = x, 1.1 + step * 0.2
        hers.saw(moves, [], mine.seen)
    assert hers.follows_pointer and hers.number == 1
    mine.x, mine.seen = 40.0, 15.0
    hers.saw(moves, [], 15.0)
    meeting = WhatMeetingDoes()
    return moves, hers, meeting


class _PhysicalHands(_Hands):
    def __init__(self, moves, clock):
        super().__init__()
        self.moves, self.clock = moves, clock
        self.births = []

    async def point(self, x, y):
        await super().point(x, y)
        self.moves.things[1].x, self.moves.things[1].y = x * 200, y * 200
        self.clock.now += 0.01

    def _emit(self):
        mine = self.moves.things[1]
        self.clock.now += 0.01
        number = 10 + len(self.births)
        shot = _thing(number, 3, mine.x, mine.y, at=self.clock.now, vy=-100.0)
        self.moves.things[number] = shot
        self.births.append({"what": "appeared", "at": self.clock.now, "thing": number,
                            "kind": 3, "x": mine.x, "y": mine.y})
        # A captured birth precedes the input call's completion.
        self.clock.now += 0.06

    async def click(self, x, y):
        await super().click(x, y)
        self.moves.things[1].x, self.moves.things[1].y = x * 200, y * 200
        self._emit()

    async def tap(self, key):
        await super().tap(key)
        self._emit()


@pytest.mark.parametrize("trigger, words", [
    ("mouse", "Aim with your mouse; click to throw at red targets."),
    ("space", "Aim with your mouse; press SPACE to fire at red targets."),
])
async def test_observed_pointer_control_learns_emissions_across_large_moves_and_delivery_delays(monkeypatch, trigger, words):
    moves, hers, meeting = _measured_world()
    clock = SimpleNamespace(now=16.0)
    monkeypatch.setattr(playing_as_it_happens, "time", SimpleNamespace(monotonic=lambda: clock.now))
    hands = _PhysicalHands(moves, clock)
    run = _Run(keys=[], began=clock.now, pointer_first=True)
    rules = WhatTheRulesSaid.read(words)
    _what_the_rules_said_of(rules, moves, meeting, run, None, clock.now)
    if trigger == "mouse":
        assert meeting.told[1] == CLICK
    _what_the_rules_said_of(rules, moves, meeting, run, None, clock.now, pointer_steers=hers.follows_pointer)
    assert meeting.told[1] == SHOOT
    coin = _thing(4, 2, 160.0, 190.0, vy=-100.0)
    moves.things[4], meeting.told[2] = coin, MEET
    origins = []
    for x in (160.0, 30.0):
        coin.x = x
        clock.now += 2.0
        moves.things[1].seen = clock.now
        hers.saw(moves, [], clock.now)
        origins.append(hers.last_seen)
        await _act(hands, run, moves, hers, meeting, _Choosing(moves, hers, meeting, []), clock.now, rules=rules)
        event = hands.births[-1]
        assert (event["x"], event["y"]) == origins[-1]
        assert event["at"] < hers._taps[-1].completed
        assert abs(moves.things[1].x - event["x"]) > 30.0
        clock.now += 0.02
        moves.things[1].seen = clock.now
        hers.saw(moves, [event], clock.now)
    assert hers.makes[trigger] == Makes(trigger, 3, 0.0, -100.0, 2)
    assert hers.last_seen == (30.0, 150.0)


async def test_real_aiming_waits_for_alignment_and_preserves_an_off_center_pose_between_stretches(monkeypatch):
    moves, hers, meeting = _measured_world()
    hers.makes["mouse"] = Makes("mouse", 3, 0.0, -100.0, 2)
    meeting.told[1] = SHOOT
    clock = SimpleNamespace(now=16.0)
    monkeypatch.setattr(playing_as_it_happens, "time", SimpleNamespace(monotonic=lambda: clock.now))
    hands = _PhysicalHands(moves, clock)
    rules = WhatTheRulesSaid.read("Click to fire at red targets.")
    moves.things[2].x = 160.0
    run = _Run(keys=[], began=clock.now, pointer_first=True)
    choosing = _Choosing(moves, hers, meeting, [])
    _goal, why, aim = choosing.target()
    assert choosing.fire(aim, why) is None
    await _act(hands, run, moves, hers, meeting, choosing, clock.now, rules=rules)
    assert hands.inputs == [("point", 0.8, 0.75)]
    for _stretch in range(2):
        clock.now += 2.0
        moves.things[1].seen = clock.now
        hers.saw(moves, [], clock.now)
        run = _Run(keys=[], began=clock.now, pointer_first=True)
        choosing = _Choosing(moves, hers, meeting, [])
        _goal, why, aim = choosing.target()
        assert choosing.fire(aim, why) == "mouse"
        await _act(hands, run, moves, hers, meeting, choosing, clock.now, rules=rules)
        assert hands.inputs[-1] == ("click", 0.8, 0.75)
        assert hers._taps[-1].origin == (160.0, 150.0)


@pytest.mark.parametrize("words", ["Click to start.", "Click to play again.", "Click for help."])
async def test_lifecycle_clicks_do_not_become_periodic_triggers(words):
    moves, hers, meeting = _measured_world()
    hands = _Hands()
    run = _Run(keys=[], began=16.0, pointer_first=True)
    await _act(hands, run, moves, hers, meeting, _Choosing(moves, hers, meeting, []),
               16.0, rules=WhatTheRulesSaid.read(words))
    assert not any(item[0] in ("click", "tap") for item in hands.inputs)


async def test_receipt_windows_reject_older_and_ambiguous_births_and_count_repeated_trials():
    moves, hers, _meeting = _measured_world()
    mine = moves.things[1]
    shot = _thing(10, 3, mine.x, mine.y, at=20.3, vy=-100.0)
    moves.things[10] = shot
    event = {"what": "appeared", "at": 20.3, "thing": 10, "kind": 3, "x": mine.x, "y": mine.y}
    hers.tapped("mouse", 20.0, began=19.9)
    hers.tapped("mouse", 20.25, began=20.2)
    hers.saw(moves, [event, event], 20.31)
    assert not hers._made and not hers.makes
    # Replaying after one receipt expires cannot turn ambiguity into attribution.
    hers.saw(moves, [event], 20.5)
    assert not hers._made
    old = {**event, "thing": 11, "at": 19.8}
    hers.saw(moves, [old], 20.5)
    assert not hers._made
    hers.numbered_afresh()
    hers.number = mine.number
    hers.tapped("mouse", 21.0)
    burst = [{**event, "thing": n, "at": 21.01} for n in (12, 13)]
    for item in burst:
        moves.things[item["thing"]] = _thing(item["thing"], 3, mine.x, mine.y, at=21.01, vy=-100.0)
    hers.saw(moves, burst, 21.02)
    assert not hers.makes, "one burst must not count as repeated input evidence"
    hers.tapped("mouse", 22.0)
    second = {**event, "thing": 14, "at": 22.01}
    moves.things[14] = _thing(14, 3, mine.x, mine.y, at=22.01, vy=-100.0)
    hers.saw(moves, [second], 22.02)
    assert hers.makes["mouse"].times == 2


async def test_retained_emissions_lose_qualification_when_fresh_inputs_have_no_effect():
    moves, hers, _meeting = _measured_world()
    hers.makes["mouse"] = Makes("mouse", 3, 0.0, -100.0, 20)
    hers.numbered_afresh()
    hers.number = 1
    assert "mouse" in hers.makes
    for trial in range(4):
        at = 20.0 + trial
        hers.tapped("mouse", at)
        hers.saw(moves, [], at + 0.5)
    assert "mouse" not in hers.makes


async def test_emission_receipt_invariant_measures_distinct_trials():
    from core.agency.which_one_answers_to_her import _emissions_require_distinct_input_receipts

    assert _emissions_require_distinct_input_receipts() == ()


async def test_many_bursts_from_few_successful_inputs_cannot_qualify_a_trigger():
    moves, hers, _meeting = _measured_world()
    event = {"what": "appeared", "kind": 3, "x": hers.last_seen[0], "y": hers.last_seen[1]}
    for trial in range(20):
        at = 20.0 + trial
        hers.tapped("mouse", at)
        burst = [{**event, "at": at + 0.01, "thing": 10 + trial * 10 + i}
                 for i in range(10)] if trial < 2 else []
        for item in burst:
            moves.things[item["thing"]] = _thing(item["thing"], 3, item["x"], item["y"], at=at + 0.01, vy=-100.0)
        hers.saw(moves, burst, at + 0.02)
        hers.saw(moves, [], at + 0.5)
    assert "mouse" not in hers.makes


async def test_learned_response_timing_preserves_sustained_fast_triggers_and_relearns_changed_delay():
    moves, hers, _meeting = _measured_world()
    event = {"what": "appeared", "kind": 3, "x": hers.last_seen[0], "y": hers.last_seen[1]}

    def emit(at, delay, number):
        hers.tapped("mouse", at + 0.02, began=at)
        birth = {**event, "thing": number, "at": at + delay}
        moves.things[number] = _thing(number, 3, event["x"], event["y"], at=at + delay, vy=-100.0)
        hers._what_keys_make(moves, [birth], max(at + delay + 0.01, at + 0.02))

    emit(20.0, 0.01, 10)
    emit(21.0, 0.01, 11)
    assert hers.emission_window("mouse", 3) == pytest.approx(0.06)
    for trial in range(20):
        emit(22.0 + trial * 0.25, 0.01, 12 + trial)
        assert "mouse" in hers.makes
    # A longer new response must cease to qualify under the old latency bound.
    for trial in range(100):
        emit(30.0 + trial * 0.25, 0.2, 100 + trial)
        if "mouse" not in hers.makes:
            break
    assert "mouse" not in hers.makes
    # Separated experiments can establish the changed response.
    hers.recheck_controls()
    emit(60.0, 0.2, 300)
    emit(62.0, 0.2, 301)
    assert hers.emission_window("mouse", 3) == pytest.approx(0.25)


async def test_retained_trigger_spaces_trials_until_current_response_timing_is_measured(monkeypatch):
    moves, hers, meeting = _measured_world()
    hers.makes["mouse"] = Makes("mouse", 3, 0.0, -100.0, 2)
    hers.numbered_afresh()
    hers.number = 1
    meeting.told[1] = SHOOT
    clock = SimpleNamespace(now=20.0)
    monkeypatch.setattr(playing_as_it_happens, "time", SimpleNamespace(monotonic=lambda: clock.now))
    hands = _PhysicalHands(moves, clock)
    run = _Run(keys=[], began=clock.now, pointer_first=True)
    rules = WhatTheRulesSaid.read("Click to fire at red targets.")
    for trial in range(3):
        await _act(hands, run, moves, hers, meeting, _Choosing(moves, hers, meeting, []), clock.now, rules=rules)
        hers._what_keys_make(moves, [hands.births[-1]], clock.now)
        completed = clock.now
        if trial == 0:
            clock.now += 0.25
            await _act(hands, run, moves, hers, meeting, _Choosing(moves, hers, meeting, []), clock.now, rules=rules)
            assert len(hands.births) == 1
            clock.now = completed + 0.5
        else:
            clock.now = completed + 0.25
    assert len(hands.births) == 3
    assert hers.emission_window("mouse", 3) is not None


async def test_measured_timing_still_accepts_a_birth_during_a_long_input_call():
    moves, hers, _meeting = _measured_world()
    event = {"what": "appeared", "kind": 3, "x": hers.last_seen[0], "y": hers.last_seen[1]}
    for at, completed, born, number in ((20.0, 20.02, 20.01, 10),
                                        (21.0, 21.02, 21.01, 11),
                                        (22.0, 22.8, 22.5, 12)):
        hers.tapped("mouse", completed, began=at)
        birth = {**event, "thing": number, "at": born}
        moves.things[number] = _thing(number, 3, event["x"], event["y"], at=born, vy=-100.0)
        hers._what_keys_make(moves, [birth], completed + 0.01)
    assert hers.makes["mouse"].times == 3
    assert hers.emission_window("mouse", 3) == pytest.approx(0.55)
    hers.tapped("mouse", 24.02, began=24.0)
    later = {**event, "thing": 13, "at": 24.5}
    moves.things[13] = _thing(13, 3, event["x"], event["y"], at=24.5, vy=-100.0)
    hers._what_keys_make(moves, [later], 24.51)
    assert hers.makes["mouse"].times == 4


async def test_long_delivery_keeps_one_frame_for_an_effect_after_completion():
    moves, hers, _meeting = _measured_world()
    hers.makes["mouse"] = Makes("mouse", 3, 0.0, -100.0, 2)
    hers._emission_delays[("mouse", 3)].update({18.0: 0.01, 19.0: 0.01})
    hers.tapped("mouse", 22.8, began=22.0)
    event = {"what": "appeared", "kind": 3, "x": hers.last_seen[0], "y": hers.last_seen[1],
             "thing": 12, "at": 22.81}
    moves.things[12] = _thing(12, 3, event["x"], event["y"], at=event["at"], vy=-100.0)
    hers._what_keys_make(moves, [event], 22.82)
    assert hers._made[("mouse", 3)] == [(22.8, 12)]


async def test_rule_bindings_follow_each_observed_control_transition():
    moves, _hers, meeting = _measured_world()
    run = _Run(keys=[], began=16.0)
    rules = WhatTheRulesSaid.read("Click to fire at red targets.")
    for pointer_steers in (False, True, False, True):
        _what_the_rules_said_of(rules, moves, meeting, run, None, 16.0, pointer_steers=pointer_steers)
        assert meeting.told[1] == (SHOOT if pointer_steers else CLICK)


async def test_trigger_spacing_respects_measured_effect_deadline(monkeypatch):
    moves, hers, meeting = _measured_world()
    hers.makes["mouse"] = Makes("mouse", 3, 0.0, -100.0, 2)
    hers._emission_delays[("mouse", 3)].update({18.0: 0.35, 19.0: 0.35})
    meeting.told[1] = SHOOT
    run = _Run(keys=[], began=20.0, pointer_first=True, last_click=20.02, last_trigger_began=20.0)
    clock = SimpleNamespace(now=20.27)
    monkeypatch.setattr(playing_as_it_happens, "time", SimpleNamespace(monotonic=lambda: clock.now))
    hands = _Hands()
    rules = WhatTheRulesSaid.read("Click to fire at red targets.")
    await _act(hands, run, moves, hers, meeting, _Choosing(moves, hers, meeting, []), clock.now, rules=rules)
    assert hands.inputs == [("point", 0.2, 0.75)]
    hands.inputs.clear()
    clock.now = 20.46
    await _act(hands, run, moves, hers, meeting, _Choosing(moves, hers, meeting, []), clock.now, rules=rules)
    assert hands.inputs == [("click", 0.2, 0.75)]
