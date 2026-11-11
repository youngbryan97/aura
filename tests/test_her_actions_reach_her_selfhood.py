"""What her actions did reaches her selfhood, her sense of agency and her recurrent self-model.

MEASURED 10 November 2026 in her logs, 20 October to 10 November (L06): 1,295 completed intentions, each "1 belief
updates, 0 self-model updates"; her minimal selfhood's learned field all zeros in 122 starts, because nothing called
its reinforcement; and her recurrent self-model resumed at tick 200 on every start, last saved on 3 May.
"""
from __future__ import annotations

import json
import time

import numpy as np

import core.consciousness.agency_comparator as agency_comparator
import core.consciousness.crsm as crsm
import core.consciousness.minimal_selfhood as minimal_selfhood
from core.consciousness.minimal_selfhood import ACTION_CATEGORIES, DEFICIT_KEYS, MinimalSelfhood

TIRED = {"energy_reserves": 0.2, "resource_pressure": 0.3}
RESTED = {"energy_reserves": 0.7, "resource_pressure": 0.3}
TOOL_USE = ACTION_CATEGORIES.index("tool_use")
ENERGY = DEFICIT_KEYS.index("energy")


def _tick(selfhood: MinimalSelfhood, body: dict[str, float]) -> None:
    time.sleep(0.002)  # a tick reads her after what came before it
    selfhood.update(body, {"coherence": 0.6, "curiosity": 0.5}, {})


def test_an_action_learns_from_the_tick_that_read_her_while_she_did_it():
    selfhood = MinimalSelfhood()
    _tick(selfhood, TIRED)
    token = selfhood.action_started("tool_use")
    _tick(selfhood, RESTED)
    taught = selfhood.action_finished(token)
    assert taught == {"energy": 0.5}
    assert selfhood._W[TOOL_USE, ENERGY] > 0.0


def test_with_no_tick_since_she_set_out_it_learns_from_the_next():
    selfhood = MinimalSelfhood()
    _tick(selfhood, TIRED)
    token = selfhood.action_started("tool_use")
    assert selfhood.action_finished(token) is None
    assert selfhood._W[TOOL_USE, ENERGY] == 0.0
    _tick(selfhood, RESTED)
    assert selfhood._W[TOOL_USE, ENERGY] > 0.0
    assert selfhood.get_status()["reinforcements"] == 1


def test_an_intention_says_what_it_taught_her_selfhood(tmp_path, monkeypatch):
    from core.agency.intention_loop import IntentionLoop

    selfhood = MinimalSelfhood()
    monkeypatch.setattr(minimal_selfhood, "_INSTANCE", selfhood)
    loop = IntentionLoop(db_path=str(tmp_path / "intentions.db"))
    _tick(selfhood, TIRED)
    intention = loop.intend("read the notes", drive="curiosity", expected_outcome="read")
    loop.record_action(intention, tool_name="read_file", args={}, result="read", success=True, duration_ms=5.0)
    _tick(selfhood, RESTED)
    loop.observe(intention, observation="tool_succeeded", actual_outcome="read")
    loop.revise(intention, success=True)
    record = loop.get_intention(intention)
    assert record.self_model_updates == ["minimal_selfhood tool_use reduced energy 0.500"]


def test_an_abandoned_intention_teaches_nothing(tmp_path, monkeypatch):
    from core.agency.intention_loop import IntentionLoop

    selfhood = MinimalSelfhood()
    monkeypatch.setattr(minimal_selfhood, "_INSTANCE", selfhood)
    loop = IntentionLoop(db_path=str(tmp_path / "intentions.db"))
    _tick(selfhood, TIRED)
    intention = loop.intend("read the notes", drive="curiosity")
    loop.abandon(intention, "not needed")
    _tick(selfhood, RESTED)
    assert not selfhood._pending and not loop._selfhood_tokens
    assert selfhood._W[TOOL_USE, ENERGY] == 0.0


def test_agency_is_read_only_where_outcomes_were_compared(monkeypatch):
    from core.consciousness.selfhood_tick import read_cognition

    comparator = agency_comparator.AgencyComparator()
    monkeypatch.setattr(agency_comparator, "_instance", comparator)
    assert comparator.agency_reading() is None
    assert "agency_score" not in read_cognition(object())
    comparator.emit_efference(layer="intention_loop", predicted_state={"goal_completed": 1.0, "surprise": 0.0},
                              action_goal="read the notes", action_source="curiosity")
    comparator.compare_and_attribute(None, {"goal_completed": 1.0, "surprise": 0.0}, action_goal="read the notes")
    reading = comparator.agency_reading()
    assert reading is not None
    assert read_cognition(object())["agency_score"] == reading


def test_her_recurrent_self_model_keeps_what_it_learned_when_she_stops(tmp_path, monkeypatch):
    path = tmp_path / "crsm_state.json"
    asked: list = []
    monkeypatch.setattr(crsm, "PERSIST_PATH", path)
    monkeypatch.setattr(crsm, "_save_at_shutdown", asked.append)
    model = crsm.ContinuousRecurrentSelfModel()
    assert asked == [model], "her shutdown is asked to keep it"
    assert not model.save_if_advanced(), "nothing learned, nothing written"
    for step in range(7):
        model.update(valence=0.1 * step, arousal=0.2, energy=0.6)
    assert model.save_if_advanced()
    assert json.loads(path.read_text())["tick_count"] == 7
    assert not model.save_if_advanced()
    again = crsm.ContinuousRecurrentSelfModel()
    assert again._tick_count == 7 and np.allclose(again.hidden_state, model.hidden_state)
