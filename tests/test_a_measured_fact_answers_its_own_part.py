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


def test_a_channel_strikes_a_claim_its_own_record_refutes(monkeypatch):
    """Three minutes after a restart, directly beneath a measured line saying
    so, she wrote "I've been up since 0600". The reading was in the messages
    she was given: evidence informs, it does not enforce."""
    import core.self.lifetime as lifetime

    class ThreeMinutes:
        current_uptime_s = 180.0

        def current(self) -> str:
            return "3 minutes"

    monkeypatch.setattr(lifetime, "read_lifetime", lambda *a, **k: ThreeMinutes())

    kept, wrong = lifetime.strike_uptime_contradiction(
        "Morning. I've been up since 0600. My schedule includes maintenance checks."
    )
    assert "0600" not in kept
    assert "maintenance checks" in kept
    assert wrong and "3 minutes" in wrong

    # A claim that agrees with the record survives untouched.
    agrees, nothing = lifetime.strike_uptime_contradiction("I've been up for 3 minutes.")
    assert nothing is None
    assert agrees == "I've been up for 3 minutes."

    # A reply that says nothing about waking is not touched.
    unrelated, none_either = lifetime.strike_uptime_contradiction("Today is maintenance.")
    assert none_either is None
    assert unrelated == "Today is maintenance."


def test_the_composer_applies_the_refutation(monkeypatch):
    import core.self.lifetime as lifetime

    class ThreeMinutes:
        current_uptime_s = 180.0

        def current(self) -> str:
            return "3 minutes"

    monkeypatch.setattr(lifetime, "read_lifetime", lambda *a, **k: ThreeMinutes())

    out = compose_measured(
        "how long have you been up, and what have you got going on today?",
        "I've been up since 0600. Maintenance checks are queued.",
        MEASURED,
        about_uptime,
        refute=lifetime.strike_uptime_contradiction,
    )
    assert "0600" not in out
    assert "Maintenance checks are queued." in out
    assert MEASURED in out


def test_a_channel_with_no_record_refutes_nothing(monkeypatch):
    """Abstention, not invention: no measurement means no contradiction."""
    import core.self.lifetime as lifetime

    monkeypatch.setattr(lifetime, "read_lifetime", lambda *a, **k: None)
    kept, wrong = lifetime.strike_uptime_contradiction("I've been up since 0600.")
    assert wrong is None
    assert kept == "I've been up since 0600."
