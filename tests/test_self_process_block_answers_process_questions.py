"""Internal telemetry is not an answer to a question about preference.

LIVE 2026-08-18: "if your attention could only go one place, where would it
go?"

    Right now I am attending to where my attention is. I still have this
    recent concern in view: what subject would you happily lose a weekend
    to?. My current bias is Happiness, leaning toward...

The self-process block exists for questions about how she works — "how does
confusion change your planning, memory use, and tool verification?" — and it
fired because the sentence contained the word "attention". The question was
about what she would CHOOSE to attend to, and it got a recital of her current
attention state and affect bias instead.

The dimension words are attention, attending, focus, noticing, present, so
"let's focus on the database schema" requested an introspection essay too.

Two separations, both already used elsewhere in this codebase: a hypothetical
asks about the will rather than the mechanism, and a cognition word used in
passing is not a question about her cognition.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _self_process_requested_dimensions


@pytest.mark.parametrize(
    "question",
    [
        "how does confusion change your planning, memory use, and tool verification?",
        "what are you attending to right now?",
        "are you present?",
        "how do you handle memory across sessions?",
    ],
)
def test_a_question_about_her_process_still_gets_the_block(question: str) -> None:
    assert _self_process_requested_dimensions(question)


@pytest.mark.parametrize(
    "question",
    [
        # Preference, not mechanism.
        "if your attention could only go one place, where would it go?",
        "what would you focus on if you had a free day?",
        "would you rather focus on breadth or depth?",
        # A cognition word used in passing.
        "let's focus on the database schema",
        "keep the present tense throughout",
    ],
)
def test_a_preference_or_a_passing_word_gets_no_telemetry(question: str) -> None:
    assert _self_process_requested_dimensions(question) == []
