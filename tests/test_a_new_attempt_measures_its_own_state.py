"""A retry retains learned physics and starts its own counters and controls."""
from __future__ import annotations

from core.agency.how_the_contest_stands import ContestStands
from core.agency.what_meeting_things_does import Readouts, WhatMeetingDoes
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.how_things_move_here import HowThingsMoveHere
from core.skills.screen_pursuit_as_it_happens import begin_run


def _region(text, x, y=0.1):
    return {"text": text, "center_x": x, "center_y": y, "height": 0.03}


def test_absent_end_screen_labels_cannot_override_live_scores():
    counters = Readouts()
    contest = ContestStands()
    contest.heard("First to 7 points wins")
    counters.read([_region("You 0", 0.4), _region("Opponent 7", 0.6)], 1.0, 0.05)
    contest.counted(counters.current, counters.where, 0.05, 1.0)
    assert contest.settled() == "lost"
    counters.read([_region("1", 0.4), _region("0", 0.6)], 2.0, 0.05)
    contest.counted(counters.current, counters.where, 0.05, 2.0)
    assert (contest.mine, contest.theirs) == (1, 0)
    assert contest.settled() == ""
    # History still supports confirmation when a counter is briefly unread.
    assert counters.values["opponent"] == 7


def test_a_retry_retains_world_knowledge_without_old_outcomes_or_control_identity():
    contest = ContestStands()
    contest.heard("First to 7 points wins")
    contest.counted({"you": 0, "opponent": 7}, {}, None, 1.0)
    hers = WhichIsHers()
    hers.number, hers.kind, hers.last_shape = 9, 3, (4.0, 24.0)
    hers._hers.add("q", 0.0, 100.0)
    meeting = WhatMeetingDoes()
    meeting.read([_region("Opponent 7", 0.6)], 1.0, 0.05)
    meeting._open.append({"kind": 3, "at": 1.0, "what": "touched"})
    meeting.evidence[3].touched = 4
    physics = HowThingsMoveHere()
    physics.kinds[3].meetings.append((0.5, 0.3, 1.05))
    keep = {"contest": contest, "hers": hers, "meeting": meeting, "physics": physics,
            "named_keys": ["q", "a"]}
    begin_run(keep)
    assert keep["contest"].wins.to_score == 7
    assert keep["contest"].settled() == ""
    assert keep["hers"].number is None and keep["hers"].way_of("q") is None
    assert keep["physics"] is physics and physics.kinds[3].meetings
    assert keep["named_keys"] == ["q", "a"]
    assert meeting.evidence[3].touched == 4
    assert not meeting.readouts.values and not meeting._open
    assert not meeting.read([_region("Opponent 0", 0.6)], 2.0, 0.05)


def test_long_world_names_do_not_lose_the_part_that_distinguishes_versions():
    from core.runtime.what_she_learned import named

    address = "played at file:///users/someone/" + "a-long-directory/" * 7 + "index.html"
    first, second = named(address + " revision 123456"), named(address + " revision 654321")
    assert len(first) <= 80 and len(second) <= 80
    assert first != second
    assert named(address + " revision 123456") == first
    assert named("a short world") == "a-short-world"


def test_named_directions_do_not_add_arrows_the_instructions_did_not_offer():
    from core.agency.playing_as_it_happens import controls_named_in

    assert controls_named_in("Up and down arrows move the slider.")[0] == ["up", "down"]
    assert controls_named_in("Left and right arrows steer, space jumps.")[0] == ["space", "left", "right"]
    assert controls_named_in("Use the arrow keys to move.")[0] == ["up", "down", "left", "right"]
    assert controls_named_in("Arrow keys move; W and S climb.")[0][:4] == ["up", "down", "left", "right"]
    assert controls_named_in("Arrow keys to walk. Pick up the coins.")[0] == ["up", "down", "left", "right"]


def test_a_key_is_not_recorded_as_held_before_its_command_arrived(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from core.agency import playing_as_it_happens as playing

    now = [10.0]

    class Hands:
        async def up(self, key):
            now[0] += 0.08

        async def down(self, key):
            now[0] += 0.08

    monkeypatch.setattr(playing, "time", SimpleNamespace(monotonic=lambda: now[0]))
    run = playing._Run(keys=["a", "b"], began=9, held="a")
    hers = WhichIsHers()
    hers.holding("a", 9, trying=True)
    asyncio.run(playing._hold(Hands(), run, hers, "b", 9.8, trying=True))
    assert hers._held_at(10.05)[1] == "a"
    assert hers._held[-1][0] == run.held_since == now[0]
    assert run.responses[-1] == now[0] - 9.8


def test_play_retries_a_control_whose_many_pictures_were_all_blocked():
    import asyncio
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _act, _Run

    class Hands:
        def __init__(self):
            self.pressed = []

        async def down(self, key):
            self.pressed.append(key)

        async def up(self, key):
            pass

    hands = Hands()
    hers = WhichIsHers()
    hers.number, hers.kind = 7, 2
    for _ in range(100):
        hers._hers.add("a", 0, 0, pinned=True)
    for _ in range(4):
        hers._hers.add("q", 0, 80)
    run = _Run(keys=["q", "a"], began=0, trying=8)
    choosing = SimpleNamespace(mine=object(), ways={"q": (0, 80)}, danger=lambda way: 0)
    asyncio.run(_act(hands, run, SimpleNamespace(), hers, WhatMeetingDoes(), choosing, 4))
    assert hands.pressed == ["a"]


def test_restart_keys_do_not_become_active_controls_but_their_other_roles_remain():
    from core.agency.playing_as_it_happens import controls_named_in

    assert controls_named_in("Up and down arrows move. Press SPACE to play again.", during_play=True)[0] == ["up", "down"]
    assert controls_named_in("Press ENTER to restart.", during_play=True, keys_without_words=())[0] == []
    assert controls_named_in("Space to jump. Press SPACE to begin.", during_play=True)[0] == ["space"]
    assert controls_named_in("Press SPACE to begin.")[0] == ["space"]


def test_current_counter_claim_runs_its_registered_measurement():
    from core.agency.what_meeting_things_does import _current_counter_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _current_counter_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "current_counters_exclude_absent_history")
    assert check.predict(None) is True
    assert any(c.test == check.name for c in suite.claims())


def test_returning_historical_values_breaks_the_registered_counter_claim(monkeypatch):
    from core.agency.what_meeting_things_does import _absent_counters_are_not_current

    monkeypatch.setattr(Readouts, "current", property(lambda self: dict(self.values)))
    assert not _absent_counters_are_not_current()


def test_control_uses_the_time_until_the_next_picture_instead_of_stopping_short():
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _Choosing

    decision = _Choosing.__new__(_Choosing)
    decision.mine = SimpleNamespace(x=40.0, y=50.0)
    decision.ways = {"north": (0.0, -200.0), "south": (0.0, 200.0)}
    decision.target = lambda: ((None, 60.0), "meet", None)
    decision.speed = lambda axis: 200.0
    decision.danger = lambda way: 0.0
    decision.next_picture_s = 0.02
    decision.response_s = 0.0
    assert decision.key("")[0] == "south"
    decision.mine.y = 60.0
    assert decision.key("south")[0] == ""
    # A slower frame source must allow for more movement before it can act again.
    decision.mine.y = 50.0
    decision.next_picture_s = 0.12
    assert decision.key("")[0] == ""


def test_a_delayed_command_releases_before_the_old_motion_overshoots():
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _Choosing

    decision = _Choosing.__new__(_Choosing)
    decision.mine = SimpleNamespace(x=40.0, y=55.0)
    decision.ways = {"north": (0.0, -200.0), "south": (0.0, 200.0)}
    decision.target = lambda: ((None, 60.0), "meet", None)
    decision.speed = lambda axis: 200.0
    decision.danger = lambda way: 0.0
    decision.next_picture_s, decision.response_s = 0.02, 0.025
    assert decision.key("south")[0] == ""
    decision.response_s = 0.0
    assert decision.key("south")[0] == "south"


def test_visible_movement_measures_the_environment_response_delay():
    from types import SimpleNamespace

    import pytest

    from core.agency.playing_as_it_happens import _measure_response, _Run

    run = _Run(keys=["south"], began=0)
    run.previous_control = (7, 40, 55, 1.0, (0, 200), (0, 0))
    _measure_response(run, SimpleNamespace(number=7, x=40, y=60), 1.05)
    assert run.motion_responses[-1] == pytest.approx(0.025)
    assert run.previous_control is None
    run.previous_control = (7, 40, 55, 2.0, (0, 200), (0, -200))
    _measure_response(run, SimpleNamespace(number=8, x=40, y=60), 2.05)
    assert len(run.motion_responses) == 1


def test_an_interception_is_predicted_at_first_contact_on_either_axis():
    from types import SimpleNamespace

    import pytest

    from core.agency.playing_as_it_happens import _Choosing, _contact_line

    mine = SimpleNamespace(x=20.0, y=40.0, w=10.0, h=30.0)
    coming = SimpleNamespace(x=80.0, y=30.0, w=6.0, h=6.0, vx=-100.0, vy=20.0,
                             kind=1, moved=True)
    decision = _Choosing.__new__(_Choosing)
    decision.mine, decision.physics = mine, None
    decision.moves = SimpleNamespace(shape=(150, 200), things={})
    decision.hers = SimpleNamespace(lowest=(0, 0), highest=(0, 150))
    decision.others = lambda: [coming]
    decision.stance = lambda thing: "meet"
    decision.speed = lambda axis: 100.0
    assert _contact_line(mine, coming, 0) == 28.0
    assert decision._target_on_a_line(0)[0][1] == pytest.approx(40.4)
    coming.x = 0.0
    assert _contact_line(mine, coming, 0) == 12.0
    coming.y = 100.0
    assert _contact_line(mine, coming, 1) == 58.0


def test_a_partial_trajectory_does_not_invent_walls_at_its_endpoints():
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _range_of

    moves = SimpleNamespace(shape=(150, 200), things={1: SimpleNamespace(
        kind=1, path=[(n / 20, 80, 70+n) for n in range(30)])})
    assert _range_of(moves, 1, 1) == (0, 150)
    physics = SimpleNamespace(edge=lambda kind, name: ("bounce", 12 if name == "top" else 140, 1))
    assert _range_of(moves, 1, 1, physics) == (12, 140)


def test_an_unreachable_immediate_arrival_is_not_discarded_for_a_distant_one():
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _Choosing

    soon = SimpleNamespace(x=50, y=140, w=4, h=4, vx=-100, vy=0, kind=1, moved=True)
    later = SimpleNamespace(x=180, y=40, w=4, h=4, vx=-10, vy=0, kind=1, moved=True)
    decision = _Choosing.__new__(_Choosing)
    decision.mine = SimpleNamespace(x=20, y=40, w=8, h=24)
    decision.physics = None
    decision.moves = SimpleNamespace(shape=(150, 200), things={})
    decision.others = lambda: [later, soon]
    decision.stance = lambda thing: "meet"
    decision.speed = lambda axis: 10
    assert decision._target_on_a_line(0)[2] is soon


def test_a_summary_keeps_the_observed_control_description_after_it_disappears():
    import time
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _Run, _what_it_came_to

    began = time.monotonic()
    run = _Run(keys=["q"], began=began, hers_description="blue bar")
    hers = SimpleNamespace(kind=0, makes={}, keys_that_move_her=lambda keys: {"q": (0, 80)})
    result = _what_it_came_to(run, SimpleNamespace(kinds=[]), hers, WhatMeetingDoes(), "new screen", began)
    assert result["hers"] == "blue bar"
