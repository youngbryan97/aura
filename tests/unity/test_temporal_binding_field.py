from __future__ import annotations

import time

from core.unity.temporal_binding import TemporalBindingField
from core.unity.unity_state import BoundContent


def _content(content_id: str, source: str, summary: str, ts: float) -> BoundContent:
    return BoundContent(
        content_id=content_id,
        modality="goal",
        source=source,
        summary=summary,
        salience=0.8,
        confidence=0.9,
        timestamp=ts,
        ownership="self",
        action_relevance=0.9,
        affective_charge=0.0,
    )


def test_repeated_contents_raise_temporal_continuity():
    field = TemporalBindingField(window_s=4.0)
    base_ts = time.time()
    initial = [_content("goal_1", "planner", "finish the unity layer", base_ts - 0.2)]
    first = field.bind_now("tick_1", initial, now_ts=base_ts)
    second = field.bind_now(
        "tick_2",
        [_content("goal_1", "planner", "finish the unity layer", base_ts + 0.2)],
        previous_temporal=first,
        previous_content_ids=["goal_1"],
        now_ts=base_ts + 0.2,
    )

    assert second.continuity_from_previous >= first.continuity_from_previous
    assert second.drift_from_previous <= 0.3


def test_abrupt_jump_lowers_temporal_continuity():
    field = TemporalBindingField(window_s=4.0)
    base_ts = time.time()
    first = field.bind_now("tick_1", [_content("goal_1", "planner", "answer the user", base_ts)], now_ts=base_ts)
    second = field.bind_now(
        "tick_2",
        [_content("goal_2", "world_state", "thermal emergency", base_ts + 0.2)],
        previous_temporal=first,
        previous_content_ids=["goal_1"],
        now_ts=base_ts + 0.2,
    )

    assert second.continuity_from_previous < 0.3
    assert second.drift_from_previous > 0.6


def test_stale_subsystem_produces_phase_lag_signal():
    field = TemporalBindingField(window_s=4.0)
    base_ts = time.time()
    contents = [
        _content("goal_1", "planner", "keep responding carefully", base_ts),
        _content("memory_1", "memory_retrieval", "old cached interpretation", base_ts - 2.5),
    ]
    window = field.bind_now("tick_3", contents, now_ts=base_ts)

    assert window.phase_lag["memory_retrieval"] > 2.0
    assert window.phase_lag["planner"] < 0.2
