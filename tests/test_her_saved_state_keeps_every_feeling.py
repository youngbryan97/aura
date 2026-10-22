"""Her saved state keeps every feeling she has, and every marker of one.

On 22 October her saved state held her eight highest emotions and her first
twelve affect markers. She boots from it, so a restart set the other thirty-six
feelings back to their defaults, and the appraisal's marker, the thirteenth,
never reached the record.
"""
from __future__ import annotations

import json

from core.state.aura_state import AuraState
from core.state.state_repository import StateRepository


def test_the_hot_snapshot_keeps_every_emotion_and_where_it_rests():
    state = AuraState()
    for index, name in enumerate(sorted(state.affect.emotions)):
        state.affect.emotions[name] = index / 100.0
    saved = json.loads(StateRepository(is_vault_owner=True)._serialize_transport_snapshot(state))["affect"]
    assert set(saved["emotions"]) == set(state.affect.emotions)
    assert set(saved["mood_baselines"]) == set(state.affect.emotions) & set(state.affect.mood_baselines)


def test_the_hot_snapshot_keeps_a_thirteenth_marker():
    state = AuraState()
    for index in range(16):
        state.affect.markers[f"reading_{index}"] = {"value": index}
    state.affect.markers["appraisal"] = {"felt": {"anger": 0.4}}
    saved = json.loads(StateRepository(is_vault_owner=True)._serialize_transport_snapshot(state))["affect"]
    assert saved["markers"]["appraisal"] == {"felt": {"anger": 0.4}}
    assert len(saved["markers"]) == 17


def test_a_restart_from_it_brings_back_a_low_feeling():
    repo = StateRepository(is_vault_owner=True)
    state = AuraState()
    state.affect.emotions["remorse"] = 0.37
    restored = repo._deserialize(repo._serialize_transport_snapshot(state))
    assert restored.affect.emotions["remorse"] == 0.37
