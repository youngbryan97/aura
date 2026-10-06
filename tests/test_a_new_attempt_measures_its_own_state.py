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
    check, = suite.tests()
    assert check.predict(None) is True
    assert suite.claims()[0].test == check.name


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
    assert decision.key("")[0] == "south"
    decision.mine.y = 60.0
    assert decision.key("south")[0] == ""
    # A slower frame source must allow for more movement before it can act again.
    decision.mine.y = 50.0
    decision.next_picture_s = 0.12
    assert decision.key("")[0] == ""


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
