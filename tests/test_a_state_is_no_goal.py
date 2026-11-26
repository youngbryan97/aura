"""A reading of how she is bids for her attention and is never something she has set herself to do."""
from __future__ import annotations

import pytest

from core.goals.standing_objective import is_valid_standing_objective, standing_objective_rejection_for_bound_work
from core.state.readings_of_her import (
    affect_reading,
    body_reading,
    drive_reading,
    is_a_reading_of_her,
    surge_reading,
    surprise_reading,
)

READINGS = [affect_reading("dread", 0.91), drive_reading("social", 0.95), body_reading(88.0, 41.3), surprise_reading("valence drift", 0.4),
            surge_reading(0.52), "Phenomenal Surge: High qualia intensity (||q||=5.16)"]
THINGS_TO_DO = ["Research the history of quantum error correction.", "Find out why the drive alert is raised so often",
                "Write about the affective state of dread", "Maintain and repair Aura"]


@pytest.mark.parametrize("words", READINGS)
def test_a_reading_of_her_is_read_back_and_is_no_goal(words):
    assert is_a_reading_of_her(words)
    assert not is_valid_standing_objective(words)
    assert standing_objective_rejection_for_bound_work(words) == "a_reading_of_her_state"


@pytest.mark.parametrize("words", THINGS_TO_DO)
def test_a_thing_to_do_is_not_a_reading(words):
    assert not is_a_reading_of_her(words)


def test_what_bids_for_her_attention_is_written_from_the_same_patterns():
    """A reading written anywhere else is one nothing would read back."""
    import inspect

    from core.consciousness import heartbeat

    source = inspect.getsource(heartbeat)
    for name in ("affect_reading(", "body_reading(", "drive_reading(", "surge_reading(", "surprise_reading("):
        assert f"content={name}" in source
    for written in ("Drive alert:", "Affective state:", "Phenomenal surge:", "Body integrity alert:", "Prediction surprise in"):
        assert written not in source


@pytest.mark.asyncio
async def test_what_was_kept_as_goals_before_is_let_go_and_what_is_to_be_done_is_kept(tmp_path):
    """LIVE 24 November 2026: eleven of her fifty three goals under way were readings, and one was a small model's words."""
    from core.goals.goal_engine import GoalEngine

    engine = GoalEngine(db_path=str(tmp_path / "goals.db"))
    try:
        # As the store held them, before anything stood at its door.
        kept = {}
        for text, source in [(READINGS[0], "executive_closure"), (READINGS[1], "executive_closure"), ("[AUTONOMOUS INITIATIVE] verified internal task completion", "eternal_growth"),
                             (THINGS_TO_DO[0], "executive_closure")]:
            record = await engine.add_goal(name=text[:60], objective=text, source="executive_authority", status="in_progress", metadata={"initiative_source": source})
            kept[text] = record["id"]
        engine.quarantine_transient_foreground_goals()
        stands = {text: engine.get_goal(one)["status"] for text, one in kept.items()}
    finally:
        engine.close()
    assert stands == {READINGS[0]: "abandoned", READINGS[1]: "abandoned", "[AUTONOMOUS INITIATIVE] verified internal task completion": "abandoned", THINGS_TO_DO[0]: "in_progress"}


def test_her_goal_hierarchy_neither_files_nor_hands_out_a_reading(tmp_path):
    """26 November 2026: her goal hierarchy (~/.aura/goals.json) still held readings of her state as pending goals,
    "Affective state: dread (arousal=0.91)" among them, and handed out its oldest pending goals to her own time."""
    import json

    from core.motivation.goal_hierarchy import GoalHierarchy

    path = tmp_path / "goals.json"
    path.write_text(json.dumps({
        "r1": {"id": "r1", "description": "Affective state: dread (arousal=0.91)", "status": "pending", "priority": 0.9},
        "g1": {"id": "g1", "description": "Research the history of tide clocks.", "status": "pending", "priority": 0.5},
    }))
    hierarchy = GoalHierarchy(persist_path=str(path))
    for goal in hierarchy.goals.values():
        if goal.id not in {"r1", "g1"}:
            goal.status = "completed"  # her root values, which the hierarchy sets as goals of its own
    assert hierarchy.get_next_goal().id == "g1", "the reading kept before is passed over"
    assert hierarchy.add_goal("Drive alert: social is depleted (95% urgency)") == ""
