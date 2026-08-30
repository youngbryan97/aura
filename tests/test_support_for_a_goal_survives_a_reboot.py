"""Candidates were persistent and their support was not.

The count reached disk when a candidate was born and again when it was adopted,
and every unit of support in between lived in memory. So a tension that recurred
four times across a day of uptime came back after a reboot as a tension that had
been noticed once, and a goal three-quarters of the way to being adopted started
again.

For a mind whose whole point is that experience accumulates, that is the wrong
thing to lose.
"""

from __future__ import annotations

import sqlite3

import pytest

from core.goals.emergent_goals import EmergentGoalEngine


@pytest.fixture
def engine(tmp_path):
    return EmergentGoalEngine(db_path=str(tmp_path / "emergent.sqlite3"))


def support_on_disk(engine, goal_id):
    with sqlite3.connect(engine._db_path) as conn:
        row = conn.execute(
            "SELECT support_count FROM emergent_goal_candidates WHERE goal_id = ?",
            (goal_id,),
        ).fetchone()
    return row[0] if row else None


def a_recurring_tension(engine, times=1):
    for _ in range(times):
        engine.observe("action_regret", 0.9, "the same trouble again")
    return engine.synthesize()


# ── support reaches disk as it grows ─────────────────────────────────────

def test_a_recurring_tension_is_one_tension(engine):
    """It hashed the evidence text, which grows with every observation.

    So the fourth sighting of the same trouble was a different goal from the
    first, support never accumulated on anything, and the adoption threshold
    could be reached only by a tension whose evidence read identically every
    time. The mechanism looked live and was almost unreachable.
    """
    a_recurring_tension(engine, times=2)
    first = set(engine._candidates)
    for _ in range(3):
        a_recurring_tension(engine, times=2)
    assert set(engine._candidates) == first, "the same tension became several goals"


def test_and_more_evidence_sharpens_it_rather_than_replacing_it(engine):
    a_recurring_tension(engine, times=2)
    goal_id = next(iter(engine._candidates))
    engine.observe("action_regret", 0.9, "a second, different way it went wrong")
    engine.synthesize()
    assert goal_id in engine._candidates
    assert "different way" in engine._candidates[goal_id].objective


def test_support_gathered_after_the_first_time_is_written_down(engine):
    a_recurring_tension(engine, times=2)
    made = list(engine._candidates)
    assert made, "no candidate was synthesised"
    goal_id = made[0]
    first = support_on_disk(engine, goal_id)

    a_recurring_tension(engine, times=2)
    engine.synthesize()
    assert engine._support_counts[goal_id] > 1
    assert support_on_disk(engine, goal_id) == engine._support_counts[goal_id], (
        "support grew in memory and not on disk"
    )
    assert support_on_disk(engine, goal_id) > first


def test_what_was_written_down_is_what_was_counted(engine):
    a_recurring_tension(engine, times=2)
    goal_id = next(iter(engine._candidates))
    for _ in range(3):
        a_recurring_tension(engine, times=2)
    assert support_on_disk(engine, goal_id) == engine._support_counts[goal_id]


# ── and the objective says what it is ────────────────────────────────────

def test_the_objective_is_honest_about_what_it_is():
    """The evidence is observed; the sentence around it is ours.

    The note here used to claim the objective was built from evidence "not a
    template", beside a line that is a template. That is exactly the boundary
    between recombining motives and inventing one, and it is worth stating
    rather than blurring.
    """
    import inspect

    from core.goals import emergent_goals

    source = inspect.getsource(emergent_goals)
    assert "its shape is not" in source
    assert "not drawn from a fixed designer taxonomy" not in source
