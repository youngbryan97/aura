"""A phrasal-verb particle does not report that something is down.

LIVE, 2026-08-27: a question about arithmetic ending "is three examples enough
to pin it down?" was answered with the host's processor and memory percentages.
"You" gave the subject and "down" gave the trouble.
"""

from __future__ import annotations

from core.introspection.self_evidence import (
    _without_particles,
    asks_about_own_operational_state,
)


def test_pinning_something_down_is_not_a_report_of_trouble() -> None:
    asked = (
        "Three examples of something I'm doing to numbers, and I want the "
        "rule. 12 becomes 6. 30 becomes 10. 84 becomes 14. What's the rule, "
        "what does 210 become, and how confident are you - is three examples "
        "enough to pin it down?"
    )
    assert not asks_about_own_operational_state(asked)


def test_the_other_particles_read_the_same_way() -> None:
    for asked in (
        "Can you narrow it down for me?",
        "Write that down while you think it through.",
        "Should you call it off?",
        "Are you able to track it down?",
    ):
        assert not asks_about_own_operational_state(asked), asked


def test_a_thing_that_is_down_still_reads_as_trouble() -> None:
    assert asks_about_own_operational_state("Are you down?")
    assert asks_about_own_operational_state("Is the machine you run on down?")


def test_the_copula_keeps_its_pronoun_as_a_subject() -> None:
    assert "is it down" in _without_particles("tell me if is it down")
    assert "was it off" in _without_particles("was it off")


def test_a_verb_before_the_pronoun_takes_the_particle() -> None:
    assert "down" not in _without_particles("pin it down")
    assert "off" not in _without_particles("shrug it off")
