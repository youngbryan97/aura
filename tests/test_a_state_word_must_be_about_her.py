""""Doing" is ordinary English about anything.

LIVE, 2026-08-27: "Since YOUR deals.csv analysis showed West had the highest
average approved deal size, can you tell me what West is DOING differently that
the other regions should copy?" was answered with "The machine is at 22.2%
processor and 59.4% memory right now."

The "your" attached to the analysis and the "doing" attached to West. A topic
found in one part and a question found in another are not evidence about the
same thing — the failure this gate's own docstring warns about, and the reason
it says it is deliberately narrow.

So the enquiry words are split. Ones that mean nothing else — status, health,
utilisation, load — stand on their own. Ones that are ordinary English —
doing, state, working, ok, usage — have to sit next to a word for her.
"""

from __future__ import annotations

import pytest

from core.introspection.self_evidence import asks_about_own_operational_state


@pytest.mark.parametrize(
    "asked",
    [
        "Since your deals.csv analysis showed West had the highest average approved "
        "deal size, can you tell me what West is doing differently that the other "
        "regions should copy?",
        "what is your report saying about Q3 revenue?",
        "how is the West region doing this quarter?",
        "what state is the migration in?",
        "is the build working now?",
        "what is the capital of Peru",
        "my deploy is failing",
    ],
)
def test_a_state_word_about_something_else_is_not_about_her(asked: str) -> None:
    assert asks_about_own_operational_state(asked) is False


@pytest.mark.parametrize(
    "asked",
    [
        "How hard is the machine you run on working right now?",
        "how much memory are you using?",
        "are any of your subsystems degraded?",
        "what kind of load is your host under?",
        "is anything off with your runtime?",
        "what is your current utilisation?",
    ],
)
def test_a_question_naming_her_instruments_still_reaches_them(asked: str) -> None:
    assert asks_about_own_operational_state(asked) is True


def test_the_strong_words_stand_alone() -> None:
    """Nobody says "utilisation" about a sales region."""
    from core.introspection.self_evidence import _STATE_ENQUIRY_RE, _WEAK_STATE_RE

    for word in ("status", "health", "utilisation", "load"):
        assert _STATE_ENQUIRY_RE.search(f"your {word}")
    # And the weak ones need her beside them.
    assert not _WEAK_STATE_RE.search("what is West doing differently")
    assert _WEAK_STATE_RE.search("how are you doing")


@pytest.mark.parametrize(
    "asked",
    [
        "what are you doing later?",
        "what are you up to tonight?",
        "what are you working on tomorrow?",
        "what are you planning for the rest of the day?",
    ],
)
def test_asking_what_she_will_do_is_not_asking_how_she_is(asked: str) -> None:
    """A weak state word beside a future reference is asking about plans.

    Answering "what are you doing later" with a processor percentage is the
    same category error as answering "are you ok" with one.
    """
    assert asks_about_own_operational_state(asked) is False


def test_a_two_part_question_still_splits(asked: str = "") -> None:
    """One half about her machine, one half about her plans."""
    from core.conversation.composed_answer import coverage_of

    covered, uncovered = coverage_of(
        "How hard is your machine working, and what are you doing later?",
        asks_about_own_operational_state,
    )
    assert covered and uncovered
    assert "machine" in covered[0]
    assert "later" in uncovered[0]
