"""Her live state reaches a turn by what stands out of it, not by the alphabet (core/brain/what_stands_out_in_her.py)."""
from __future__ import annotations

import json
import random

from core.brain import cognitive_engine, what_stands_out_in_her
from core.brain.what_stands_out_in_her import HerUsual, leaves, what_stands_out


def _a_turn(rng: random.Random, n: int) -> dict:
    return {
        "affect": {"valence": 0.2 + rng.gauss(0, 0.02), "label": "calm"},
        "governance": {"legacy_fallback_allowed": False},
        "ticks": 1000 + 10 * n,
        "workspace": {"focus": f"thought {n}", "source": "drive_social" if n % 5 else "memory"},
    }


def _usual(turns: int = 40) -> HerUsual:
    rng, usual = random.Random(3), HerUsual()
    for n in range(turns):
        usual.learn(_a_turn(rng, n))
    return usual


def test_what_is_the_same_on_every_turn_is_not_given():
    usual = _usual()
    said = json.loads(what_stands_out(_a_turn(random.Random(3), 40), usual, room=700))
    assert "governance" not in said and "label" not in said.get("affect", {})


def test_a_reading_far_from_its_usual_is_given_first():
    usual, now = _usual(), _a_turn(random.Random(3), 40)
    now["affect"]["valence"] = -0.7
    assert json.loads(what_stands_out(now, usual, room=40)) == {"affect": {"valence": -0.7}}


def test_a_count_that_goes_on_counting_does_not_stand_out_and_one_that_leaps_does():
    usual, now = _usual(), _a_turn(random.Random(3), 40)
    assert usual.tells(("ticks",), 1000 + 10 * 40) < what_stands_out_in_her.TELLS
    assert usual.tells(("ticks",), 9000) > what_stands_out_in_her.TELLS
    assert "ticks" not in json.loads(what_stands_out(now, usual, room=700))


def test_what_is_new_each_turn_is_told_and_a_common_value_is_not():
    usual = _usual()
    assert usual.tells(("workspace", "focus"), "thought 40") > usual.tells(("workspace", "source"), "drive_social")
    said = json.loads(what_stands_out(_a_turn(random.Random(3), 41), usual, room=700))
    assert said["workspace"]["focus"] == "thought 41" and "source" not in said["workspace"]


def test_a_turn_about_her_is_given_all_that_fits_within_its_room():
    usual, now = _usual(), _a_turn(random.Random(3), 41)
    whole = json.loads(what_stands_out(now, usual, room=2600, whole=True))
    assert {at for at, _ in leaves(whole)} == {at for at, _ in leaves(now)}
    for room in (20, 60, 100):
        said = what_stands_out(now, usual, room=room, whole=True)
        assert len(said) <= room and (not said or json.loads(said))


def test_what_is_usual_is_kept_from_one_run_to_the_next(tmp_path):
    kept = tmp_path / "usual.json"
    first = HerUsual(kept)
    for n in range(20):
        first.learn({"valence": 0.2 + 0.01 * (n % 3)})
    first.keep()
    assert HerUsual(kept).tells(("valence",), 0.9) == first.tells(("valence",), 0.9) > what_stands_out_in_her.TELLS


def test_her_live_state_holds_the_machine_and_her_timing_whatever_the_turn(monkeypatch):
    usual = HerUsual()
    monkeypatch.setattr(what_stands_out_in_her, "get_her_usual", lambda: usual)
    compact = {"governance": {"a": True}, "host": {"cpu": 5.123456}, "lane": {"state": "ready"},
               "mind_snapshot": {f"z{n}": "x" * 60 for n in range(60)}, "turn_timing": {"decode_tokens_per_second": 19.5},
               "voice": {"mood": "steady"}}
    for said in ("What's the tallest mountain?", "How are you feeling right now?"):
        told = cognitive_engine._her_live_mind(compact, {"user_message": said}, limit=2600)
        held = json.loads(told)
        assert held["host"] == {"cpu": 5.12} and held["turn_timing"]["decode_tokens_per_second"] == 19.5
        assert len(told) <= 2600


def test_what_was_read_of_the_turn_says_whether_it_is_about_her_and_not_its_words(monkeypatch):
    from core.self import how_she_does_it

    usual = HerUsual()
    monkeypatch.setattr(what_stands_out_in_her, "get_her_usual", lambda: usual)
    compact = {"mind_snapshot": {f"z{n}": "x" * 60 for n in range(60)}}
    said = {"user_message": "How deep would you go if I want frogs?"}
    try:
        how_she_does_it.hold_ways([])
        assert cognitive_engine._her_live_mind(compact, said, limit=2600) == "{}"
        how_she_does_it.hold_ways([how_she_does_it.HER_STATE])
        assert len(cognitive_engine._her_live_mind(compact, said, limit=2600)) > 2000
    finally:
        how_she_does_it.hold_ways(None)


def test_a_turn_about_something_else_is_given_no_more_room_than_a_question_on_her_condition(monkeypatch):
    usual = HerUsual()
    monkeypatch.setattr(what_stands_out_in_her, "get_her_usual", lambda: usual)
    compact = {"mind_snapshot": {f"z{n}": f"{n} {'x' * 60}" for n in range(60)}}
    for n in range(6):
        for at in compact["mind_snapshot"]:
            compact["mind_snapshot"][at] = f"{n} {at} {'x' * 60}"
        told = cognitive_engine._her_live_mind(compact, {"user_message": "What's the tallest mountain?"}, limit=2600)
        assert len(told) <= cognitive_engine._ROOM_FOR_HER_STATE + 2
