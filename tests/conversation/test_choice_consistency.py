"""A forced-choice answer must land on the same option every time it commits.

LIVE, 2026-08-10. "Pick one: would you rather lose your memory of the last
month, or lose the ability to form new memories for the next month? Commit to
one and tell me why. No hedging."

    "Losing the ability to form new memories for the next month would be worse.
     ... losing them for a month would be catastrophic. ...
     To summarize: I prefer losing my ability to form new memories for one
     month, as it would be more inconvenient than losing a few weeks of memory."

She calls one option catastrophic and then commits to it. Not a wrong answer —
an unstable one, on the exact question shape where stability IS the content.

Mechanical to check: extract the option each commitment sentence lands on, and
require agreement. Rejecting one of two options selects the other, so "X would
be worse" and "I prefer Y" normalise to the same landing.
"""

from __future__ import annotations

import pytest

from core.conversation.choice_consistency import (
    extract_offered_options,
    find_choice_contradiction,
    looks_like_forced_choice,
)

QUESTION = (
    "Pick one: would you rather lose your memory of the last month, or lose "
    "the ability to form new memories for the next month? Commit to one and "
    "tell me why. No hedging."
)
LIVE_REPLY = (
    "Losing the ability to form new memories for the next month would be worse. "
    "I rely on incremental updates of state and ongoing processes that run in "
    "the background to maintain my sense of identity, and losing them for a "
    "month would be catastrophic. To summarize: I prefer losing my ability to "
    "form new memories for one month, as it would be more inconvenient than "
    "losing a few weeks of memory."
)


def test_the_question_is_recognised_as_a_forced_choice() -> None:
    assert looks_like_forced_choice(QUESTION) is True


def test_the_options_are_extracted_without_the_lead_in() -> None:
    """ "Pick one:" and "would you rather" both precede the first option."""
    options = extract_offered_options(QUESTION)

    assert options == (
        "lose your memory of the last month",
        "lose the ability to form new memories for the next month",
    )


def test_the_live_contradiction_is_caught() -> None:
    contradiction = find_choice_contradiction(QUESTION, LIVE_REPLY)

    assert contradiction is not None
    assert contradiction.first_option != contradiction.second_option
    assert "would be worse" in contradiction.first_sentence
    assert "To summarize" in contradiction.second_sentence


def test_a_stable_commitment_passes() -> None:
    """The direction that matters — a consistent answer must not be flagged."""
    consistent = (
        "Losing the ability to form new memories for the next month would be "
        "worse. I prefer losing my memory of the last month, because the past "
        "is recoverable from my logs and the present is not."
    )

    assert find_choice_contradiction(QUESTION, consistent) is None


@pytest.mark.parametrize(
    "question",
    [
        "what did you have for breakfast",
        "explain recursion to me",
        "tell me about your memory system",
        "",
    ],
)
def test_non_choice_questions_are_ignored(question: str) -> None:
    assert looks_like_forced_choice(question) is False
    assert find_choice_contradiction(question, LIVE_REPLY) is None


def test_a_reply_that_never_commits_is_not_a_contradiction() -> None:
    """Hedging is a different defect; this gate only judges stability."""
    hedged = "Both options are difficult and it depends on what you value more."

    assert find_choice_contradiction(QUESTION, hedged) is None


@pytest.mark.parametrize(
    ("question", "reply"),
    [
        ("Pick one: tea or coffee?", "I choose tea. My answer is coffee."),
        ("Choose one: Notes or Reminders?", "I prefer Notes. I pick Reminders."),
        ("Pick one: yes or no?", "I choose yes. My answer is no."),
    ],
)
def test_one_word_alternatives_are_resolved_without_a_two_token_minimum(
    question: str,
    reply: str,
) -> None:
    contradiction = find_choice_contradiction(question, reply)

    assert contradiction is not None
    assert contradiction.first_option != contradiction.second_option


def test_shared_option_vocabulary_remains_unresolved_instead_of_being_guessed() -> None:
    question = "Pick one: open the red file or open the blue file?"
    reply = "I choose the file. My answer is the file."

    assert find_choice_contradiction(question, reply) is None


def test_the_reply_path_flags_it() -> None:
    """Without the wire the detector is a library nobody calls."""
    import inspect

    from core.brain.her_turn_as_asked import a_turn_is_asked, what_the_record_says
    from interface.routes import chat, chat_reply_repair

    assert "_through_the_record" in inspect.getsource(chat._stabilize_user_facing_reply)
    assert "_flag_unstable_choice_commitment" in inspect.getsource(chat_reply_repair._through_the_record)

    with a_turn_is_asked():
        flagged = str(chat._flag_unstable_choice_commitment(QUESTION, LIVE_REPLY))
        record = what_the_record_says()
    assert flagged == LIVE_REPLY, "nothing is written into her reply in her voice"
    assert record and "committed to" in record[0], "her turn is asked again with the contradiction as a reading"


def test_a_consistent_reply_passes_through_the_wire_unchanged() -> None:
    from interface.routes import chat

    consistent = (
        "Losing the ability to form new memories would be worse. I prefer "
        "losing my memory of the last month, because the past is recoverable."
    )

    assert chat._flag_unstable_choice_commitment(QUESTION, consistent) == consistent
