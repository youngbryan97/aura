"""A reading displaces a guess about the same thing, and nothing else.

LIVE, 2026-08-22. "Morning. How long have you been up, and what have you got
going on today?" came back as the uptime figure alone: 61.6 days, 2199
sessions, one thing said today. Every number was right and half the message
went unanswered, because a channel that matches returns its reading in place
of the whole reply.
"""

from __future__ import annotations

from core.conversation.composed_answer import compose_measured, coverage_of
from core.language.asking_clauses import asking_clauses, asks_more_than_one_thing

MEASURED = "Awake 61.6 days in total, across 2199 sessions."
WRITTEN = "Today there is a consolidation pass queued and the curriculum running."


def about_uptime(clause: str) -> bool:
    return "been up" in clause.lower() or "awake" in clause.lower()


def test_both_halves_of_a_joined_question_are_found():
    clauses = asking_clauses("how long have you been up, and what have you got going on today?")
    assert len(clauses) == 2
    assert clauses[0].startswith("how long")
    assert clauses[1].startswith("what have you got")


def test_a_lone_question_is_not_split():
    assert not asks_more_than_one_thing("how long have you been up?")


def test_rules_of_a_game_are_not_read_as_questions():
    """The describing sentences carry the words that trip topic patterns."""
    clauses = asking_clauses(
        "I made up a game. If it's your turn and you cannot move at all (because the "
        "other piece is directly next to yours), you lose. With perfect play, who wins?"
    )
    assert len(clauses) == 1
    assert clauses[0].startswith("With perfect play")


def test_a_reading_that_covers_everything_is_the_answer():
    assert (
        compose_measured("how long have you been up?", WRITTEN, MEASURED, about_uptime)
        == MEASURED
    )


def test_a_reading_that_covers_half_keeps_the_other_half():
    out = compose_measured(
        "how long have you been up, and what have you got going on today?",
        WRITTEN,
        MEASURED,
        about_uptime,
    )
    assert MEASURED in out
    assert WRITTEN in out
    assert out.index(MEASURED) < out.index(WRITTEN)


def test_coverage_names_what_was_left_out():
    covered, uncovered = coverage_of(
        "how long have you been up, and what have you got going on today?", about_uptime
    )
    assert len(covered) == 1 and len(uncovered) == 1
    assert "going on today" in uncovered[0]


def test_an_empty_reading_leaves_the_reply_alone():
    assert compose_measured("anything?", WRITTEN, "", about_uptime) == WRITTEN


def test_nothing_written_yet_still_serves_the_reading():
    out = compose_measured(
        "how long have you been up, and what else?", "", MEASURED, about_uptime
    )
    assert out == MEASURED
