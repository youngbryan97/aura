""""I don't know where you grew up" is an answer. The canned line is not.

LIVE 2026-08-18: "what's the population of the town I grew up in?"

    I couldn't get to an answer I'd stand behind on that one, and I won't send
    you a thinner one and pass it off as the real thing.

The guard behind that worked perfectly. The draft was rejected for
`fabricated_shared_history` — the model had supplied a home town it was never
told — the retries ran out, and correct behaviour produced the worst possible
answer. One honest sentence was available the whole time.

Nothing here enumerates biographical facts. The question names the fact
itself, so the subject comes from the sentence and the verdict comes from the
store.
"""

from __future__ import annotations

import pytest

from core.self.person_facts import needed_person_fact, person_fact_block


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("what's the population of the town I grew up in?", "the town you grew up in"),
        ("what was the name of the school I went to?", "the school you went to"),
        ("how far is the office I work at?", "the office you work at"),
    ],
)
def test_the_question_names_the_fact_it_needs(question: str, expected: str) -> None:
    assert needed_person_fact(question) == expected


def test_a_possessive_fact_is_found() -> None:
    assert needed_person_fact("what's my sister's name?").startswith("your sister")


@pytest.mark.parametrize(
    "question",
    [
        "what is 2 + 2",
        "what did I just copy?",
        "what's on my screen?",
        # Conversation recall is the transcript's job, not biography.
        "what was my first question?",
        "what was my last message?",
    ],
)
def test_a_question_needing_no_biography_is_not_claimed(question: str) -> None:
    assert needed_person_fact(question) == ""


def test_an_unknown_fact_is_named_and_the_turn_is_not_refused() -> None:
    block = person_fact_block("what's the population of the town I grew up in?")

    assert "do NOT know the town you grew up in" in block
    assert "do not refuse the whole turn over it" in block
    assert "Do not supply a plausible one" in block


def test_a_known_fact_is_offered_instead_of_the_absence(monkeypatch) -> None:
    import core.self.person_facts as module

    monkeypatch.setattr(
        module, "_known_about_person", lambda: ["home_town: Leeds", "job: engineer"]
    )

    block = person_fact_block("what's the population of the town I grew up in?")

    assert "You do hold something about" in block
    assert "Leeds" in block


def test_nothing_is_supplied_for_an_unrelated_turn() -> None:
    assert person_fact_block("what is 2 + 2") == ""
