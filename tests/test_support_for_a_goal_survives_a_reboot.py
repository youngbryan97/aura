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


def test_and_more_evidence_of_the_same_thing_sharpens_it(engine):
    """More about the SAME trouble joins it. Something else is something else,
    which is what having an identity at all is for."""
    a_recurring_tension(engine, times=2)
    goal_id = next(iter(engine._candidates))
    engine.observe("action_regret", 0.9, "the same trouble again, and worse this time")
    engine.synthesize()
    assert goal_id in engine._candidates
    assert "worse this time" in engine._candidates[goal_id].objective


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


# ── and a tension is what it is ABOUT, not just its heading ──────────────

def two_troubles(engine, each=3):
    for _ in range(each):
        engine.observe("action_regret", 0.9, "the browser tab closed before the form was submitted")
    for _ in range(each):
        engine.observe("action_regret", 0.9, "a file write was refused by governance")
    return engine.synthesize()


def test_two_different_troubles_under_one_heading_are_two_goals(engine):
    """Making the identity the kind alone fixed one drift and overshot.

    A category as broad as "a regretted action" collapses every unrelated
    recurring problem into one goal, and a goal that is about everything is
    about nothing.
    """
    two_troubles(engine)
    assert len(engine._candidates) == 2


def test_and_the_same_trouble_again_is_the_same_goal(engine):
    two_troubles(engine)
    before = set(engine._candidates)
    for _ in range(2):
        engine.observe(
            "action_regret", 0.9, "the browser tab closed again before the form submitted"
        )
    engine.synthesize()
    assert set(engine._candidates) == before, "the same trouble became another goal"


def test_and_its_support_accumulates_instead(engine):
    two_troubles(engine)
    for _ in range(2):
        engine.observe(
            "action_regret", 0.9, "the browser tab closed again before the form submitted"
        )
    engine.synthesize()
    assert max(engine._support_counts.values()) > 1


def test_an_identity_holds_still_while_evidence_accumulates(engine):
    """What a group was founded on does not move; what it has gathered does."""
    two_troubles(engine)
    ids = set(engine._candidates)
    for words in (
        "the browser tab closed once more before the form submitted",
        "the tab closed and the form was not submitted again",
    ):
        engine.observe("action_regret", 0.9, words)
        engine.synthesize()
    assert set(engine._candidates) == ids


def test_what_a_piece_of_evidence_is_about_is_its_words(engine):
    about = engine._what_it_is_about("The browser TAB closed, again!")
    assert "browser" in about and "tab" in about and "closed" in about
    assert all(len(word) >= 3 for word in about)
